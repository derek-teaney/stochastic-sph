# stochastic_stressonly.jl — is the residual move needed?  Frozen lattice at rest, constant
# e_a, m_a = 0 at t = 0; the stochastic step with the stress move only, compared with the
# stress + residual moves (λ = 1).  docs/stochastic.tex.
#
# Both keep e^{𝕊} stationary (every move satisfies detailed balance).  With the stress
# move alone a mode receives noise and damping only through the fitted gradient, so it
# relaxes at the resolved rate Γ_res(k) = 2K_res(k)/(VwT) instead of Γ(k) = 2K(k)/(VwT):
#     ⟨|m_k|²⟩/eq = 1 − exp(−2Γ t),   Γ = Γ_res (stress only) or Γ (both moves).
#
# Run:  julia -t auto --project examples/stochastic_stressonly.jl [1|2]
#   → docs/figs/stochrep_stressonly_1d.png (or _2d.png)
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "stochastic_thermalize_common.jl"))     # Kmode

const DIM = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1

function setup(dim)
    eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    dim == 1 ?
        setup_ic(0.05, 4.0, eos, (x, y) -> 4e4, (x, y) -> SVector(0.0, 0.0); Lx=1.0, sim_dim=1) :
        setup_ic(0.05, 2.5, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L=0.6, lattice=:hex)
end

# spectrum, longitudinal (1D: the x component) and, in 2D, transverse
function spec(ps, kvs, oneD)
    out = zeros(length(kvs), oneD ? 1 : 2)
    for (i, k) in enumerate(kvs)
        mk = sum(p.m * cis(-dot(k, p.x)) for p in ps)
        if oneD
            out[i, 1] = abs2(mk[1])
        else
            kh = k / norm(k);  out[i, 1] = abs2(dot(kh, mk));  out[i, 2] = abs2(kh[1] * mk[2] - kh[2] * mk[1])
        end
    end
    return out
end

function one_run(r, ps0, params, cl, eos, fz, dt, kvs, oneD, residual, snaps, nsteps)
    ps = deepcopy(ps0);  rngs = make_phys_rng(N_of(ps0), 5000r + 7);  rng = Xoshiro(r)
    S = Dict{Int,Matrix{Float64}}();  KE = zeros(nsteps)
    for n in 1:nsteps
        stochastic_step!(ps, params, cl, dt, eos; λ=1.0, residual, frozen=fz, rng, rngs, flip=isodd(n), vote=:bonds)
        n in snaps && (S[n] = spec(ps, kvs, oneD))
        KE[n] = sum(dot(q.m, q.m) for q in ps)
    end
    return (S = S, KE = KE)
end
N_of(ps) = length(ps)

params, ps0, cl, eos = setup(DIM)
oneD = DIM == 1
N = params.N;  fz = stochastic_frozen(ps0, params, cl, eos);  st = fz.stars[1]
p = ps0[1];  V = fz.V[1];  w = p.e + p.p;  T = p.T;  eq = N * V * w * T
dt = 0.25 * 0.5 * compute_dt(ps0, params, eos; mode=:viscous)
if oneD
    kvs = [SVector(2π * n / params.L[1], 0.0) for n in 1:N÷2]
    pols = [(:L, k -> SVector(1.0, 0.0))]
else
    Lx, Ly = params.L;  nmx = round(Int, Lx / params.a / 2);  nmy = round(Int, Ly / params.a / 2)
    kvs = SVector{2,Float64}[]
    for nx in 0:nmx, ny in -nmy:nmy
        (nx == 0 && ny <= 0) && continue
        push!(kvs, SVector(2π * nx / Lx, 2π * ny / Ly))
    end
    pols = [(:L, k -> k / norm(k)), (:T, k -> SVector(-k[2], k[1]) / norm(k))]
end
Γ = Dict{Symbol,Any}();  Γr = Dict{Symbol,Any}()
for (s, f) in pols
    KK = [Kmode(st, k, f(k), oneD) for k in kvs]
    Γ[s] = [2K[1] / (V * w * T) for K in KK];  Γr[s] = [2K[2] / (V * w * T) for K in KK]
end
nsteps = oneD ? 4000 : 3000
nens   = oneD ? 32 : 8
snaps  = unique(round.(Int, exp.(range(0, log(nsteps), length=30))))
@printf("%dD  N=%d  dt=%.2e  nsteps=%d (t_end=%.2f)  nens=%d\n", DIM, N, dt, nsteps, nsteps * dt, nens)

res = Dict{Bool,Any}()
for residual in (true, false)
    t0 = time()
    runs = Vector{Any}(undef, nens)
    Threads.@threads for r in 1:nens
        runs[r] = one_run(r, ps0, params, cl, eos, fz, dt, kvs, oneD, residual, snaps, nsteps)
    end
    S = Dict(n => sum(r.S[n] for r in runs) ./ (nens * eq) for n in snaps)
    KE = sum(r.KE for r in runs) ./ (nens * DIM * (N - 1) * V * w * T)
    res[residual] = (S = S, KE = KE)
    @printf("  %-22s %.0f s   final kinetic energy / equilibrium = %.3f\n",
            residual ? "stress + residual" : "stress only", time() - t0, KE[end])
end

# ── Plot ──
ka = norm.(kvs) .* params.a ./ π
if oneD
    bx = ka;  bav = v -> v
else
    edges = range(0, maximum(ka) * 1.0001, length=26);  idx = [searchsortedlast(edges, xi) for xi in ka]
    used = sort(unique(idx));  bx = [sum(ka[idx .== b]) / count(==(b), idx) for b in used]
    bav = v -> [sum(v[idx .== b]) / count(==(b), idx) for b in used]
end
fig, axs = subplots(2, 2, figsize=(15, 10))
sel = snaps[round.(Int, range(6, length(snaps), length=6))]
cm = get_cmap("viridis")
for (col, residual) in enumerate((true, false))
    ax = axs[1, col]
    for (j, n) in enumerate(sel)
        c = cm((j - 1) / (length(sel) - 1));  t = n * dt
        ax.plot(bx, bav(res[residual].S[n][:, 1]), "o", ms=3, color=c)
        g = residual ? Γ[:L] : Γr[:L]
        ax.plot(bx, bav(1 .- exp.(-2 .* g .* t)), "-", color=c, lw=1, label=@sprintf("t = %.3g", t))
    end
    ax.axhline(1, color="k", lw=0.5);  ax.set_ylim(-0.05, 1.3)
    ax.set_xlabel("k a / π");  ax.set_ylabel("⟨|m_k|²⟩ / (N V w T)" * (oneD ? "" : "  (longitudinal)"))
    ax.set_title((residual ? "stress + residual moves" : "stress move only") *
                 ": points measured, lines 1 − exp(−2Γt)" * (residual ? "" : ", Γ = Γ_res"))
    ax.legend(fontsize=8)
end
ax = axs[2, 1]
ax.semilogy(bx, bav(Γ[:L]), "k-", label="Γ(k), bond form (both moves)")
ax.semilogy(bx, bav(max.(Γr[:L], 1e-12)), "r-", label="Γ_res(k), resolved part (stress only)")
oneD || ax.semilogy(bx, bav(max.(Γr[:T], 1e-12)), "r--", label="Γ_res(k), transverse")
ax.axhline(1 / (nsteps * dt), color="gray", ls=":", label="1 / t_end")
ax.set_ylim(1e-3, 2 * maximum(Γ[:L]))
ax.set_xlabel("k a / π");  ax.set_ylabel("damping rate");  ax.set_title("damping rates of the modes");  ax.legend(fontsize=8)
ax = axs[2, 2]
t = (1:nsteps) .* dt
ax.semilogx(t, res[true].KE, "C0", label="stress + residual")
ax.semilogx(t, res[false].KE, "C3", label="stress only")
ax.axhline(1, color="k", lw=0.5)
ax.set_xlabel("t");  ax.set_ylabel("kinetic energy / equilibrium");  ax.set_title("approach to equilibrium");  ax.legend(fontsize=8)
suptitle(@sprintf("%dD frozen lattice at rest, m_a = 0 at t = 0, λ = 1: is the residual move needed?", DIM))
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_stressonly_$(DIM)d.png")
savefig(out, dpi=120);  println("Saved ", out)
