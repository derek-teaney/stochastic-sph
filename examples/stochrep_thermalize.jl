# stochrep_thermalize.jl — a frozen lattice with constant particle energies e_a and zero
# momenta, thermalized by the stochastic viscous step in the particle representation
# (docs/stochrep.tex).  Only the momenta move; positions, V_a and e_a are fixed.
#
# Measured: the momentum spectrum ⟨|m_k|²⟩ / (N V w T) (1 per component in equilibrium
# for λ = 1), its approach to equilibrium mode by mode, compared with
#     ⟨|m_k|²⟩/eq = T_eff(k)/T · (1 − e^{−2Γ_k t}),
# where Γ_k is the damping rate of the bond form and T_eff/T = (K_res + K_resid/λ)/K.
# λ = 1 (detailed balance), λ = 4 (cold residual), λ = Inf (deterministic residual).
#
# Run:  julia -t auto --project examples/stochrep_thermalize.jl [1|2]
#   → docs/figs/stochrep_thermalize_1d.png (and _2d.png)
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "stochrep_thermalize_common.jl"))     # Kmode

const DIM   = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
const LAMS  = (1.0, 4.0, Inf)
const VOTE  = :bonds
const DTF   = 0.25                       # Δt / default viscous half step

function setup(dim)
    eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    params, ps, cl, eos = dim == 1 ?
        setup_ic(0.05, 4.0, eos, (x, y) -> 4e4, (x, y) -> SVector(0.0, 0.0); Lx=1.0, sim_dim=1) :
        setup_ic(0.05, 2.5, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L=0.6, lattice=:hex)
    return params, ps, cl, eos
end

# ── Predictions from the frozen star of particle 1 (all particles equivalent) ──
# ── Spectra ──
function spectrum1d(ps, ks)
    [abs2(sum(p.m[1] * cis(-k * p.x[1]) for p in ps)) for k in ks]
end
function spectrum2d(ps, kv)
    L = zeros(length(kv));  T = zeros(length(kv))
    for (i, k) in enumerate(kv)
        mk = sum(p.m * cis(-dot(k, p.x)) for p in ps)
        kh = k / norm(k)
        L[i] = abs2(dot(kh, mk));  T[i] = abs2(kh[1] * mk[2] - kh[2] * mk[1])
    end
    return L, T
end

function run(dim)
    params, ps0, cl, eos = setup(dim)
    oneD = dim == 1
    N = params.N;  fz = stochrep_frozen(ps0, params, cl, eos)
    st = fz.stars[1];  p = ps0[1];  w = p.e + p.p;  V = fz.V[1];  T = p.T
    eq = N * V * w * T
    dt = DTF * 0.5 * compute_dt(ps0, params, eos; mode=:viscous)
    if oneD
        ns = collect(1:N ÷ 2);  ks = 2π .* ns ./ params.L[1]
        kvs = [SVector(k, 0.0) for k in ks];  pol = [SVector(1.0, 0.0)]
    else
        Lx, Ly = params.L;  nmx = round(Int, Lx / params.a / 2);  nmy = round(Int, Ly / params.a / 2)
        kvs = SVector{2,Float64}[]
        for nx in 0:nmx, ny in -nmy:nmy
            (nx == 0 && ny <= 0) && continue
            push!(kvs, SVector(2π * nx / Lx, 2π * ny / Ly))
        end
        ks = norm.(kvs)
    end
    # predictions per k (1D: one polarization; 2D: longitudinal and transverse)
    pred = Dict{Symbol,Any}()
    if oneD
        KK = [Kmode(st, k, SVector(1.0, 0.0), true) for k in kvs]
        pred[:L] = (Γ = [2K[1] / (V * w * T) for K in KK], R = [K[2] / K[1] for K in KK])
    else
        for (s, f) in ((:L, k -> k / norm(k)), (:T, k -> SVector(-k[2], k[1]) / norm(k)))
            KK = [Kmode(st, k, f(k), false) for k in kvs]
            pred[s] = (Γ = [2K[1] / (V * w * T) for K in KK], R = [K[2] / K[1] for K in KK])
        end
    end
    Γmin = minimum(pred[:L].Γ)
    nsteps = min(ceil(Int, 6.0 / (2Γmin * dt)), oneD ? 20000 : 6000)
    nens   = oneD ? 32 : 8
    snaps  = unique(round.(Int, exp.(range(0, log(nsteps), length=40))))
    late   = nsteps ÷ 2
    @printf("%dD: N=%d dt=%.3e nsteps=%d (Γ_min t_end = %.2f) nens=%d, %d threads\n",
            dim, N, dt, nsteps, Γmin * nsteps * dt, nens, Threads.nthreads())
    out = Dict{Any,Any}(:ks => ks, :kvs => kvs, :pred => pred, :dt => dt, :snaps => snaps,
                        :eq => eq, :a => params.a, :h => params.h)
    for λ in LAMS
        t0 = time()
        res = Vector{Any}(undef, nens)
        Threads.@threads for r in 1:nens
            ps = deepcopy(ps0);  rngs = make_phys_rng(N, 1000r + 1);  rng = Xoshiro(r)
            Ssnap = Dict{Int,Any}();  avg = nothing;  nav = 0;  acc = zeros(2);  KE = Float64[]
            for n in 1:nsteps
                a = stochrep_step!(ps, params, cl, dt, eos; λ, frozen=fz, rng, rngs, flip=isodd(n), vote=VOTE)
                acc .+= (a.stress, isnan(a.resid) ? 1.0 : a.resid)
                spec = nothing
                if n in snaps || (n > late && n % 5 == 0)
                    spec = oneD ? (spectrum1d(ps, ks),) : spectrum2d(ps, kvs)
                end
                n in snaps && (Ssnap[n] = spec)
                if n > late && n % 5 == 0
                    avg = avg === nothing ? collect(spec) : avg .+ spec;  nav += 1
                end
                push!(KE, sum(dot(q.m, q.m) for q in ps) / (dim * eq))
            end
            res[r] = (Ssnap = Ssnap, avg = avg ./ nav, acc = acc ./ nsteps, KE = KE)
        end
        out[λ] = res
        @printf("  λ=%-4s %.0f s   acceptance stress %.3f resid %.3f   late KE/eq %.3f\n", string(λ), time() - t0,
                sum(r.acc[1] for r in res) / nens, sum(r.acc[2] for r in res) / nens,
                sum(sum(r.KE[late:end]) / (nsteps - late + 1) for r in res) / nens)
    end
    serialize(joinpath(@__DIR__, "..", "docs", "figs", "stochrep_thermalize_$(dim)d.jls"), out)
    return out
end

run(DIM)
