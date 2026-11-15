# moment_fluctuations.jl — the fluctuations of the bond moment of a star (docs/inverse_moment.tex, "Fluctuations of the moment").
# For each star: lab weights ψ_ab = −W'(r)/r (the common scheme), w_ab = V_b ψ_ab / n_a with tr 𝓜_a = 2, and the traceless part
#     Q_a = 𝓜_a − 𝟙 = Σ_b w_ab (r_ab r_abᵀ − ½ r_ab² 𝟙),      tr Q_a² = Σ_{b,c} w_ab w_ac ½ r_ab² r_ac² cos 2φ_bc .
# Reported, averaged over particles and configurations: ⟨tr Q²⟩, its pair (self, b = c) part Σ_b w_ab² r_ab⁴/2, the triplet
# (b ≠ c) part as the difference, and the fraction of stars where the Λ_a correction falls back (negative weights).
# Configurations: the hexagonal lattice, Poisson positions, and hybrid Monte Carlo equilibria (src/hmc.jl) at several s_a.
#   julia -t 4 --project examples/moment_fluctuations.jl            (MF_SA = "16,54,541", MF_NUPD = 60, MF_HA = 2, MF_L = 0.6)
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))

envlist(k, d) = parse.(Float64, split(get(ENV, k, d), ","))
const SAS  = envlist("MF_SA", "16,54,541")
const NUPD = parse(Int, get(ENV, "MF_NUPD", "60"));  const BURN = NUPD ÷ 3
const HA   = parse(Float64, get(ENV, "MF_HA", "2"));  const LBOX = parse(Float64, get(ENV, "MF_L", "0.6"))
const H    = 0.05
eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)

# per-particle moment statistics of the present positions
function moment_stats(ps, params, cl)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    ψfun = common_bond_weight(params);  L = params.L;  N = length(ps)
    trQ2 = zeros(N);  self = zeros(N);  fb = falses(N)
    nl = neighbor_list!(cl, ps, params.h)
    Threads.@threads for a in 1:N
        nb = Tuple{Int,SVector{2,Float64},Float64}[]
        for b in neighbors(nl, a)
            b == a && continue
            r = minimum_image(ps[b].x - ps[a].x, L);  rm = norm(r)
            rm < 2params.h && push!(nb, (b, r, rm))
        end
        V = [ps[b].s / ps[b].S_smooth for (b, _, _) in nb];  ψ = [ψfun(rm) for (_, _, rm) in nb]
        na = sum(V[k] * ψ[k] * nb[k][3]^2 for k in eachindex(nb)) / 2
        Q = zero(SMatrix{2,2,Float64});  s = 0.0
        for (k, (_, r, rm)) in enumerate(nb)
            w = V[k] * ψ[k] / na
            Q += w * (r * r' - rm^2 / 2 * I);  s += w^2 * rm^4 / 2
        end
        trQ2[a] = tr(Q * Q);  self[a] = s
        ψ0b = Float64[];  ψout, _ = bond_weights_comoving(nb, ps, FLAT2, FLAT2, params.bond_h; ψ0 = ψ0b, ψfun)
        fb[a] = ψout === ψ0b                                              # Λ_a fell back to 0
    end
    return mean(trQ2), mean(self), mean(fb)
end

function report(name, stats)
    q = mean(s[1] for s in stats);  p = mean(s[2] for s in stats);  f = mean(s[3] for s in stats)
    dq = length(stats) > 1 ? std([s[1] for s in stats]) / sqrt(length(stats)) : 0.0
    @printf("%-22s ⟨tr Q²⟩ = %.4f ± %.4f   pair part %.4f   triplet part %+.4f   rms anisotropy %.3f   Λ fallback %.4f\n",
            name, q, dq, p, q - p, sqrt(q / 2), f)
end

params, ps, cl, _ = redirect_stdout(devnull) do
    setup_ic(H, HA, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex)
end
@printf("N = %d, h/a = %g, L = %g\n", params.N, HA, LBOX)
report("hexagonal lattice", [moment_stats(ps, params, cl)])

# Poisson positions (uniform, independent)
let rng = Xoshiro(5), st = []
    for _ in 1:10
        q = deepcopy(ps)
        for p in q;  p.x = SVector(rand(rng) * params.L[1], rand(rng) * params.L[2]);  end
        rebuild_cell_list!(cl, q)
        push!(st, moment_stats(q, params, cl))
    end
    rebuild_cell_list!(cl, ps)
    report("Poisson", st)
end

for sa in SAS
    vol = params.vol
    pr, q, c, _ = redirect_stdout(devnull) do
        setup_ic(H, HA, eos, (x, y) -> sa / vol, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex)
    end
    start = redirect_stdout(devnull) do;  equilibrium_init!(q, pr, c, eos; rng = Xoshiro(11));  end
    tg = HMCTarget(start.β0, start.β, pr.vol, eos)
    dt = 0.35 * compute_dt(q, pr, eos; mode = :ideal);  nmax = max(4, round(Int, LBOX / sqrt(1/3) / dt))
    rng = Xoshiro(12);  st = [];  acc = 0.0;  t0 = time()
    for n in 1:NUPD
        u = hmc_update!(q, pr, c, eos, tg, rng; nsteps = rand(rng, (nmax ÷ 2):nmax), dt)
        acc += u.accepted
        n > BURN && push!(st, moment_stats(q, pr, c))
    end
    report(@sprintf("HMC s_a = %g", sa), st)
    @printf("%22s (HMC acceptance %.2f, %d updates after %d burn-in, %.0f s)\n", "", acc / NUPD, NUPD - BURN, BURN, time() - t0)
end
