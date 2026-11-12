# energystep_stoch_onestep.jl — the energy bookkeeping of the stochastic step with the two
# Hamiltonians (docs/energystep.tex, "The stochastic step with :energy", test 2).
#
# Uniform fluid at rest, L = 0.6, h/a = 2, equilibrium start (SR_START=eq), η/s = 0.2.
# NSTEP stochastic steps of length Δt/2 at fixed positions (no leapfrog), as in the coupled
# runs.  After each: ΔH/H (H = Σ s_a ℰ_a: the smoothed H for :smoothed, Σ e_a for :energy),
# Δ(Σe)/Σe, and the entropy produced.  :smoothed uses its default gradient-corrected map,
# :energy the plain back map.  Expected for :energy: ΔH ≈ Σ_a (Teff_a − T_a) Δs_a, of order h²
# per unit Δs.
#
# Run:  julia -t 4 --project examples/energystep_stoch_onestep.jl [nstep=20] [nseed=4]
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const NSTEP = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 20
const NSEED = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4
const S0 = 1e6;  const ETAS = 0.2;  const HA = 2.0

Hof(ps, params, cl, eos) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
                            sum(p.s * p.H_over_S for p in ps))
Eof(ps, eos) = sum(particle_energy(p, eos) for p in ps)

function run(ham::Symbol, seed::Int)
    ENV["SPH_HAMILTONIAN"] = string(ham)
    eos = ConformalEOS(1/3, 1.0, ETAS, 0.0)
    params, ps, cl, eos = setup_ic(0.05, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0);
                                   L = 0.6, lattice = :hex)
    N = params.N;  rngs = make_phys_rng(N, 1000seed + 1);  rng = Xoshiro(1000seed + 2)
    initial_state!(ps, params, cl, eos, :eq; rng = Xoshiro(1000seed + 3))
    dt = 0.5 * compute_dt(ps, params, eos; mode = :viscous)
    H0 = Hof(ps, params, cl, eos);  E0 = Eof(ps, eos);  S0tot = sum(p.s for p in ps)
    dH = Float64[];  dE = Float64[];  dS = Float64[];  pred = Float64[]
    cache = StochCache()
    for n in 1:NSTEP
        Hof(ps, params, cl, eos);  update_velocities!(ps, params, cl)
        dTe = [p.Teff - p.T for p in ps];  s_before = [p.s for p in ps]   # Teff − T (meaningful for :energy)
        Hb = sum(p.s * p.H_over_S for p in ps)
        stochrep_step!(ps, params, cl, dt / 2, eos; rng, rngs, flip = isodd(n), cache)
        Ha = Hof(ps, params, cl, eos)
        push!(dH, (Ha - Hb) / H0)
        push!(dE, (Eof(ps, eos) - E0) / E0)
        push!(dS, (sum(p.s for p in ps) - S0tot) / S0tot)
        push!(pred, sum(dTe[a] * (ps[a].s - s_before[a]) for a in 1:N) / H0)
    end
    return (; dH, dE, dS, pred, N, dt)
end

res = Dict{Tuple{Symbol,Int},Any}()
for ham in (:smoothed, :energy), seed in 1:NSEED
    r = run(ham, seed);  res[(ham, seed)] = r
    @printf("%-9s seed %d  N=%d dt/2=%.3g  ΔH/H per step: mean %+.2e  rms %.2e | cum ΔH/H %+.2e  Δ(Σe)/Σe %+.2e  ΔS/S %+.2e | Σ(Teff−T)Δs/H per step: mean %+.2e  corr(ΔH, pred) %.3f\n",
            ham, seed, r.N, r.dt / 2, sum(r.dH) / NSTEP, sqrt(sum(abs2, r.dH) / NSTEP), sum(r.dH), r.dE[end], r.dS[end],
            sum(r.pred) / NSTEP, sum(r.dH .* r.pred) / sqrt(sum(abs2, r.dH) * sum(abs2, r.pred) + 1e-300))
    flush(stdout)
end
mkpath(joinpath(@__DIR__, "data"))
serialize(joinpath(@__DIR__, "data", "energystep_stoch_onestep.jls"), res)
