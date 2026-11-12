# stochrep_hamiltonian.jl — a uniform fluid at rest with the Hamiltonian step ON and the
# stochastic viscous step in the particle representation (docs/stochrep.tex), from zero
# momenta.  Three variants of the viscous step:
#     :full    stress + residual moves, λ = 1
#     :det     stress move + deterministic residual (λ = ∞)
#     :stress  stress move only
# Each time step: stochastic half step, leapfrog, stochastic half step (as generic_step!).
# Measured: the momentum spectrum ⟨|m_k|²⟩/(N V w T) versus k h at several times, the
# smoothed energy H and the particle energies Σe over time, and the lattice disorder.
#
# Environment: SR_ETAS (η/s, default 0.2), SR_S0 (entropy density, default 1e6),
#   SR_HFIX=1 (after each stochastic step, rescale all s_a by one factor to restore H),
#   SR_OUTDIR (output directory), SR_SEED0 (seed offset), SR_START (eq | cold | thermal,
#   default eq: equilibrium_init!, docs/eqstart.tex; cold: the uniform state at rest),
#   SPH_HAMILTONIAN=energy (H = Σ e_a, docs/energystep.tex; plain back map, suffix _energy).
# Defaults: kinds = full (λ = 1), vote = star, gradient-corrected map, start eq.
# Run:  julia -t auto --project examples/stochrep_hamiltonian.jl [h/a] [t_end] [nreal] [kinds, e.g. full,det] [vote] [Δt factor] [split] [nograd]
#   → docs/figs/stochrep_hamiltonian.png
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const HA    = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 2.0
const TEND  = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 4.0
const NREAL = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1          # realizations per variant
const KINDS = Tuple(Symbol.(split(length(ARGS) > 3 ? ARGS[4] : "full", ",")))       # full: λ = 1; det: λ = ∞; stress: no residual
const VOTE  = length(ARGS) > 4 ? Symbol(ARGS[5]) : :star                  # :star (single vote) or :bonds
const DTF   = length(ARGS) > 5 ? parse(Float64, ARGS[6]) : 1.0           # Δt / default
const SPLIT = length(ARGS) > 6 && ARGS[7] == "split"                    # record the energy split
const ENERGYH = get(ENV, "SPH_HAMILTONIAN", "smoothed") == "energy"   # H = Σ e_a (energystep.tex)
const GRAD  = !(length(ARGS) > 7 && ARGS[8] == "nograd") && !ENERGYH    # gradient-corrected map (never with :energy)
const ETAS  = parse(Float64, get(ENV, "SR_ETAS", "0.2"))                # η/s
const S0    = parse(Float64, get(ENV, "SR_S0", "1e6"))                  # entropy density (sets s_a)
const HFIX  = get(ENV, "SR_HFIX", "0") == "1"                           # restore H after each stochastic step
const OUTD  = get(ENV, "SR_OUTDIR", joinpath(@__DIR__, "..", "docs", "figs"));  mkpath(OUTD)
const SEED0 = parse(Int, get(ENV, "SR_SEED0", "0"))                     # seeds SEED0+1 … SEED0+NREAL
const START = Symbol(get(ENV, "SR_START", "eq"))                       # eq (hot) | cold | thermal (initial_state!)
const SUF   = (VOTE === :bonds ? "" : "_$(VOTE)") * (DTF == 1 ? "" : "_dt$(DTF)") * (GRAD ? "_grad" : "") *
              (ETAS == 0.2 ? "" : "_eta$(ETAS)") * (S0 == 1e6 ? "" : "_S$(S0)") * (HFIX ? "_hfix" : "") *
              (START === :eq ? "" : "_$(START)") * (ENERGYH ? "_energy" : "")

Hof(ps, params, cl, eos) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
                            sum(p.s * p.H_over_S for p in ps))

function spec(ps, kvs)
    L = zeros(length(kvs));  T = zeros(length(kvs))
    for (i, k) in enumerate(kvs)
        mk = sum(p.m * cis(-dot(k, p.x)) for p in ps);  kh = k / norm(k)
        L[i] = abs2(dot(kh, mk));  T[i] = abs2(kh[1] * mk[2] - kh[2] * mk[1])
    end
    return L, T
end

# Energy fix: rescale all particle entropies by one common factor c so that the smoothed
# energy H returns to Htarget (secant iteration; H grows with c through the temperature).
function restore_H!(ps, params, cl, eos, Htarget)
    s0 = [q.s for q in ps]
    Hc(c) = (for (i, q) in enumerate(ps); q.s = c * s0[i]; end; Hof(ps, params, cl, eos))
    c0, c1 = 1.0, 1.0 + 1e-4;  H0 = Hc(c0) - Htarget;  H1 = Hc(c1) - Htarget
    for _ in 1:8
        (abs(H1) < 1e-11 * abs(Htarget) || H1 == H0) && break
        c2 = c1 - H1 * (c1 - c0) / (H1 - H0);  c0, H0 = c1, H1;  c1 = c2;  H1 = Hc(c1) - Htarget
    end
    isfinite(c1) || error("restore_H!: no solution")
    return c1
end

function run(kind, seed=1)
    eos = ConformalEOS(1/3, 1.0, ETAS, 0.0)
    params, ps, cl, eos = setup_ic(0.05, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0);
                                   L=0.6, lattice=:hex)
    N = params.N;  rngs = make_phys_rng(N, 1000seed + 1);  rng = Xoshiro(1000seed + 2)
    p = ps[1];  V = p.s / p.S_smooth;  w = p.e + p.p;  T = p.T;  eq = N * V * w * T
    seed == 1 && @printf("%-7s h/a=%.1f η/s=%.2f s_a=%.1f  thermal |δv| per component=%.3f\n", kind, HA, ETAS, p.s, sqrt(T / (V * w)))
    initial_state!(ps, params, cl, eos, START; rng = Xoshiro(1000seed + 3))   # SR_START (default eq)
    dt = DTF * 0.5 * compute_dt(ps, params, eos; mode=:viscous)
    nsteps = ceil(Int, TEND / dt)
    Lx, Ly = params.L;  nmx = round(Int, Lx / params.a / 2);  nmy = round(Int, Ly / params.a / 2)
    kvs = SVector{2,Float64}[]
    for nx in 0:nmx, ny in -nmy:nmy
        (nx == 0 && ny <= 0) && continue
        push!(kvs, SVector(2π * nx / Lx, 2π * ny / Ly))
    end
    x0 = [q.x for q in ps]
    # original nearest-neighbour pairs (hex lattice: distance a) for the local disorder
    pairs = [(i, j, norm(minimum_image(ps[j].x - ps[i].x, params.L))) for i in 1:N for j in i+1:N
             if norm(minimum_image(ps[j].x - ps[i].x, params.L)) < 1.2 * params.a]
    dotm = zeros(SVector{2,Float64}, N);  cache = StochCache()      # reused arrays of the stochastic step
    λ, resid = kind === :full ? (1.0, true) : kind === :det ? (Inf, true) : (1.0, false)
    E0 = sum(particle_energy(q, eos) for q in ps);  H0 = Hof(ps, params, cl, eos)
    ts = Float64[];  Hs = Float64[];  Es = Float64[];  KE = Float64[];  dis = Float64[];  loc = Float64[];  acc = Float64[]
    snaps = Dict{Int,Any}();  snapat = unique(round.(Int, exp.(range(0, log(nsteps), length=12))))
    late = nothing;  nlate = 0
    # energy split: cumulative changes of H and Σe from the stochastic half steps and the leapfrog
    Eof(ps) = sum(particle_energy(q, eos) for q in ps)
    cH = zeros(2);  cE = zeros(2);  split_t = Float64[];  split = NTuple{4,Float64}[]
    t0 = time()
    for n in 1:nsteps
        if SPLIT
            H1 = Hof(ps, params, cl, eos);  E1 = Eof(ps)
        end
        Hpre = HFIX ? Hof(ps, params, cl, eos) : 0.0
        a1 = stochrep_step!(ps, params, cl, dt / 2, eos; λ, residual=resid, rng, rngs, flip=isodd(n), vote=VOTE, gradient=GRAD, cache)
        HFIX && restore_H!(ps, params, cl, eos, Hpre)
        if SPLIT
            H2 = Hof(ps, params, cl, eos);  E2 = Eof(ps)
        end
        leapfrog_step!(ps, params, cl, dt, eos, dotm)
        if SPLIT
            H3 = Hof(ps, params, cl, eos);  E3 = Eof(ps)
        end
        Hpre = HFIX ? Hof(ps, params, cl, eos) : 0.0
        a2 = stochrep_step!(ps, params, cl, dt / 2, eos; λ, residual=resid, rng, rngs, flip=!isodd(n), vote=VOTE, gradient=GRAD, cache)
        HFIX && restore_H!(ps, params, cl, eos, Hpre)
        if SPLIT
            H4 = Hof(ps, params, cl, eos);  E4 = Eof(ps)
            cH .+= ((H2 - H1) + (H4 - H3), H3 - H2) ./ H0;  cE .+= ((E2 - E1) + (E4 - E3), E3 - E2) ./ E0
            n % 5 == 0 && (push!(split_t, n * dt);  push!(split, (cH[1], cH[2], cE[1], cE[2])))
        end
        if n % 5 == 0 || n == 1
            push!(ts, n * dt);  push!(Hs, Hof(ps, params, cl, eos) / H0 - 1)
            push!(Es, sum(particle_energy(q, eos) for q in ps) / E0 - 1)
            push!(KE, sum(dot(q.m, q.m) for q in ps) / (2 * (N - 1) * V * w * T))
            push!(dis, sqrt(sum(sum(abs2, minimum_image(q.x - x0[i] - (sum(ps[j].x - x0[j] for j in 1:N) / N), params.L)) for (i, q) in enumerate(ps)) / N) / params.a)
            # local disorder: rms change of the original nearest-neighbour distances / a
            push!(loc, sqrt(sum((norm(minimum_image(ps[j].x - ps[i].x, params.L)) - d0)^2 for (i, j, d0) in pairs) /
                            length(pairs)) / params.a)
            push!(acc, (a1.stress + a2.stress) / 2)
        end
        n in snapat && (snaps[n] = spec(ps, kvs) ./ eq)
        if n > nsteps ÷ 2 && n % 5 == 0
            s = spec(ps, kvs) ./ eq
            late = late === nothing ? collect(s) : late .+ s;  nlate += 1
        end
    end
    @printf("%-7s seed=%d h/a=%.1f N=%d dt=%.2e steps=%d  %.0f s   KE/eq(end)=%.3f  ΔH/H=%+.2e  ΔΣe/Σe=%+.2e  displacement=%.3f a  neighbour-distance change=%.3f a\n",
            kind, seed, HA, N, dt, nsteps, time() - t0, KE[end], Hs[end], Es[end], dis[end], loc[end])
    return (ks = norm.(kvs), h = params.h, a = params.a, dt = dt, ts = ts, Hs = Hs, Es = Es, KE = KE,
            dis = dis, loc = loc, acc = acc, split_t = split_t, split = split, snaps = snaps, late = late ./ nlate, N = N)
end

jobs = [(k, s) for s in SEED0 .+ (1:NREAL) for k in KINDS]
out = Vector{Any}(undef, length(jobs))
Threads.@threads for i in eachindex(jobs)
    out[i] = run(jobs[i]...)
end
# seed 1 → the detailed figure; all seeds → the realizations figure
res = Dict(k => out[i] for (i, (k, s)) in enumerate(jobs) if s == SEED0 + 1)
serialize(joinpath(OUTD, "stochrep_hamiltonian_ha$(HA)_t$(isinteger(TEND) ? Int(TEND) : TEND)$(SUF)$(SEED0 == 0 ? "" : "_s$(SEED0)").jls"), res)
NREAL > 1 && serialize(joinpath(OUTD, "stochrep_hamiltonian_real_ha$(HA)_t$(isinteger(TEND) ? Int(TEND) : TEND)$(SUF)$(SEED0 == 0 ? "" : "_s$(SEED0)").jls"),
                       Dict(jobs[i] => out[i] for i in eachindex(jobs)))
println("done")
