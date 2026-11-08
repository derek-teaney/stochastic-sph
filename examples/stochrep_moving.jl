# stochrep_moving.jl — stochastic equilibration of a uniformly moving 2D frozen lattice
# (docs/stochrep.tex, "A moving fluid").  No Hamiltonian step: positions, V_a and e_a are
# fixed; the stochastic viscous step (λ = 1) acts on the momenta, starting from the
# uniform momentum m_0 of the moving fluid.
#
# Prediction: e^{𝕊} at fixed e_a.  Each particle's momentum fluctuation δm = m − m_0 has
# covariance C = [−∂²S_a/∂m∂m]⁻¹ at fixed e_a, anisotropic: C_xx (along v) ≠ C_yy.
# Particles are independent, so ⟨|ê·δm_k|²⟩ = N ê·C·ê for every k ≠ 0.  C is computed
# exactly as the second moment of e^{S_a(e_a, m_0 + δm)} by numerical integration.
#
# Run:  julia -t auto --project examples/stochrep_moving.jl  → docs/figs/stochrep_moving.png
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
include(joinpath(@__DIR__, "..", "src", "rng.jl"))
include(joinpath(@__DIR__, "..", "src", "stochrep.jl"))

const VTARGET = (0.0, 0.5)
const NENS    = 8
const NSTEPS  = 4000

function setup_moving(vt)
    eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    params, ps, cl, eos = setup_ic(0.05, 2.5, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0);
                                   L=0.6, lattice=:hex)
    # momentum density M = (e+p) γ² v at the rest-frame state; iterate to hit |v| = vt
    u = vt
    for _ in 1:30
        w0 = ps[1].e + ps[1].p
        for p in ps; p.m = (p.s / p.S_smooth) * w0 / (1 - u^2) * SVector(u, 0.0); end
        thermalize!(ps, params, cl, eos)
        vnow = norm(ps[1].v_over_S * ps[1].S_smooth)
        vt == 0 && break
        u += 0.5 * (vt - vnow)
        abs(vnow - vt) < 1e-6 && break
    end
    return params, ps, cl, eos
end

# Exact covariance of δm for one particle in a large system at fixed total momentum:
# weight e^{S_a(e, m0 + δm) + λ·δm}, with the multiplier λ of the total momentum chosen so
# that ⟨δm⟩ = 0 (λ ≈ β0 = ∂(−S)/∂m at m0, corrected at the order of the non-Gaussian
# terms), by integration on a grid.
function tilted_moments(e, m0, V, eos, σ, λ)
    n = 241;  xs = range(-7σ, 7σ, length=n)
    S0 = particle_entropy(e, m0, V, eos)
    Z = 0.0;  μ = zeros(2);  M2 = zeros(2, 2)
    for x in xs, y in xs
        Sm = try particle_entropy(e, m0 + SVector(x, y), V, eos) catch; -Inf end   # unphysical: weight 0
        wt = exp(Sm - S0 + λ[1] * x + λ[2] * y)
        Z += wt;  μ .+= wt .* (x, y);  M2 .+= wt .* [x * x x * y; y * x y * y]
    end
    μ ./= Z;  return μ, M2 ./ Z .- μ * μ'
end
function exact_cov(e, m0, V, eos, σ)
    _, v0, T0 = eos_from_energy(e / V, m0 / V, eos)
    λ = collect(v0 / T0)
    μ, C = tilted_moments(e, m0, V, eos, σ, λ)
    for _ in 1:6
        λ .-= C \ μ
        μ, C = tilted_moments(e, m0, V, eos, σ, λ)
    end
    return C
end

# theory.tex, inverse susceptibility χ⁻¹_{μν} = ∂β_μ/∂T^{0ν}; at fixed energy the momentum
# covariance per particle is V [χ⁻¹_{ij}]⁻¹ (the spatial block, inverted).  Spatial block:
# (β/(wγ)) [c_s²/(1−c_s²v²) (u_i + Δ⁰_i/γ)(u_j + Δ⁰_j/γ) + Δ_ij],  Δ⁰_i = γ u_i, Δ_ij = δ_ij + u_i u_j.
function theory_cov(pa, V, eos)
    v = pa.v_over_S * pa.S_smooth;  γ = 1 / sqrt(1 - dot(v, v));  u = γ * v
    w = pa.e + pa.p;  β = 1 / (pa.T * pa.γ);  c2 = cs2(eos)
    q = 2u
    χi = (β / (w * γ)) * (c2 / (1 - c2 * dot(v, v)) * (q * q') + I + u * u')
    return V * inv(Matrix(χi))
end

# One run, from δm = 0 (a function, so the per-run arrays are not shared between threads).
function one_run(r, ps0, params, cl, eos, fz, dt, kvs, m0, C, N)
    ps = deepcopy(ps0);  rngs = make_phys_rng(N, 4000r + 11);  rng = Xoshiro(r)
    late = zeros(length(kvs), 2);  nav = 0;  hist = zeros(NSTEPS, 2);  acc = zeros(2)
    P0 = sum(q.m for q in ps)
    for n in 1:NSTEPS
        a = stochrep_step!(ps, params, cl, dt, eos; frozen=fz, rng, rngs, flip=isodd(n))
        acc .+= (a.stress, a.resid) ./ NSTEPS
        dm = [q.m - m0 for q in ps]
        hist[n, 1] = sum(x[1]^2 for x in dm) / ((N - 1) * C[1, 1])
        hist[n, 2] = sum(x[2]^2 for x in dm) / ((N - 1) * C[2, 2])
        if n > NSTEPS ÷ 2 && n % 5 == 0
            for (i, k) in enumerate(kvs)
                mk = sum(dm[j] * cis(-dot(k, ps[j].x)) for j in 1:N)
                late[i, 1] += abs2(mk[1]) / (N * C[1, 1]);  late[i, 2] += abs2(mk[2]) / (N * C[2, 2])
            end
            nav += 1
        end
    end
    return (late = late ./ nav, hist = hist, acc = acc,
            dP = norm(sum(q.m for q in ps) - P0) / max(norm(P0), sum(norm(q.m) for q in ps)))
end

res = Dict{Float64,Any}()
for vt in VTARGET
    params, ps0, cl, eos = setup_moving(vt)
    N = params.N;  fz = stochrep_frozen(ps0, params, cl, eos)
    p = ps0[1];  V = fz.V[1];  e = fz.e[1];  m0 = p.m;  v = p.v_over_S * p.S_smooth
    w = p.e + p.p;  σ = sqrt(V * w * p.T)
    C = exact_cov(e, m0, V, eos, σ);  Cth = theory_cov(p, V, eos)
    dt = 0.25 * 0.5 * compute_dt(ps0, params, eos; mode=:viscous)
    Lx, Ly = params.L;  nmx = round(Int, Lx / params.a / 2);  nmy = round(Int, Ly / params.a / 2)
    kvs = SVector{2,Float64}[]
    for nx in 0:nmx, ny in -nmy:nmy
        (nx == 0 && ny <= 0) && continue
        push!(kvs, SVector(2π * nx / Lx, 2π * ny / Ly))
    end
    @printf("|v| = %.4f  γ = %.4f  N = %d  dt = %.2e  C_xx/(VwT) = %.4f  C_yy/(VwT) = %.4f  C_xy/C_yy = %.1e   theory: %.4f %.4f\n",
            norm(v), 1 / sqrt(1 - dot(v, v)), N, dt, C[1, 1] / σ^2, C[2, 2] / σ^2, C[1, 2] / C[2, 2],
            Cth[1, 1] / σ^2, Cth[2, 2] / σ^2)
    t0 = time()
    runs = Vector{Any}(undef, NENS)
    Threads.@threads for r in 1:NENS
        runs[r] = one_run(r, ps0, params, cl, eos, fz, dt, kvs, m0, C, N)
    end
    late = sum(r.late for r in runs) ./ NENS;  hist = sum(r.hist for r in runs) ./ NENS
    acc = sum(r.acc for r in runs) ./ NENS
    @printf("   %.0f s   acceptance stress %.3f resid %.3f   late ⟨δm_x²⟩/C_xx = %.4f  ⟨δm_y²⟩/C_yy = %.4f   max |ΔP|/Σ|m| = %.1e\n",
            time() - t0, acc[1], acc[2], sum(hist[NSTEPS÷2:end, 1]) / (NSTEPS - NSTEPS ÷ 2 + 1),
            sum(hist[NSTEPS÷2:end, 2]) / (NSTEPS - NSTEPS ÷ 2 + 1), maximum(r.dP for r in runs))
    res[vt] = (v = norm(v), C = C, Cth = Cth, σ2 = σ^2, ks = norm.(kvs), kvs = kvs, late = late, hist = hist,
               acc = acc, dt = dt, a = params.a)
end
serialize(joinpath(@__DIR__, "..", "docs", "figs", "stochrep_moving.jls"), res)

# ── Plot ──
fig, axs = subplots(1, 3, figsize=(20, 5.5))
cols = Dict(0.0 => ("C0", "C2"), 0.5 => ("C3", "C1"))
for vt in VTARGET
    r = res[vt];  x = r.ks .* r.a ./ π
    edges = range(0, maximum(x) * 1.0001, length=26);  idx = [searchsortedlast(edges, xi) for xi in x]
    used = sort(unique(idx));  bx = [sum(x[idx .== b]) / count(==(b), idx) for b in used]
    bav(v) = [sum(v[idx .== b]) / count(==(b), idx) for b in used]
    cx, cy = cols[vt]
    axs[1].plot(bx, bav(r.late[:, 1]), "o-", color=cx, ms=4, label=@sprintf("|v| = %.2f, along v (x)", r.v))
    axs[1].plot(bx, bav(r.late[:, 2]), "s--", color=cy, ms=4, mfc="none", label=@sprintf("|v| = %.2f, across v (y)", r.v))
    t = (1:size(r.hist, 1)) .* r.dt
    axs[2].semilogx(t, r.hist[:, 1], color=cx, label=@sprintf("|v| = %.2f, x", r.v))
    axs[2].semilogx(t, r.hist[:, 2], "--", color=cy, label=@sprintf("|v| = %.2f, y", r.v))
    axs[3].bar([vt == 0 ? 0 : 3, vt == 0 ? 1 : 4], [r.C[1, 1], r.C[2, 2]] ./ r.σ2, color=[cx, cy])
end
axs[1].axhline(1, color="k", lw=0.5);  axs[1].set_ylim(0.8, 1.2)
axs[1].set_xlabel("|k| a / π");  axs[1].set_ylabel("⟨|ê·δm_k|²⟩ / (N ê·C·ê)")
axs[1].set_title("late-time spectrum, normalized by the exact covariance C");  axs[1].legend(fontsize=8)
axs[2].axhline(1, color="k", lw=0.5)
axs[2].set_xlabel("t");  axs[2].set_ylabel("Σ_a (δm_a·ê)² / ((N−1) ê·C·ê)")
axs[2].set_title("approach to equilibrium (8 runs)");  axs[2].legend(fontsize=8)
axs[3].set_xticks([0, 1, 3, 4]);  axs[3].set_xticklabels(["C_xx\nv=0", "C_yy\nv=0", "C_xx\nv=0.5", "C_yy\nv=0.5"])
axs[3].set_ylabel("C / (V w T)");  axs[3].set_title("exact covariance: anisotropic when moving")
suptitle("2D frozen lattice moving along x, stochastic step only (λ = 1), from δm = 0")
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_moving.png")
savefig(out, dpi=120);  println("Saved ", out)
