# stochrep_equilibrium.jl — equilibrium runs of the full stochastic scheme for measuring
# correlators (docs/plan.tex).  A uniform 2D fluid at rest, started from a thermal draw of
# the momenta (thermal_init!), evolved with the Hamiltonian step and the stochastic step
# in the particle representation, recording the Fourier modes M_k, S_k (and E_k).
#
# Time stepping (Strang, merged): S(Δt/2) [L(Δt) S(Δt)]… L(Δt) S(Δt/2), i.e. one stochastic
# step of length Δt between consecutive leapfrogs (half the cost of S(Δt/2) L S(Δt/2) per step).
#
# Run:  julia -t auto --project examples/stochrep_equilibrium.jl L h/a t_end nreal [rec_every] [tag]
#   → data/equilibrium_<tag>_r<seed>.h5  (one HDF5 file per realization; see
#     write_equilibrium_h5 in src/measure.jl for the layout, read_equilibrium to load it)
# Environment: SR_OUTDIR (output directory), SR_SEED0 (seed offset), SR_ETAS, SR_S0, SR_KHMAX.
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
include(joinpath(@__DIR__, "..", "src", "rng.jl"))
include(joinpath(@__DIR__, "..", "src", "stochrep.jl"))
include(joinpath(@__DIR__, "..", "src", "measure.jl"))

const LBOX  = parse(Float64, ARGS[1])
const HA    = parse(Float64, ARGS[2])
const TEND  = parse(Float64, ARGS[3])
const NREAL = parse(Int, ARGS[4])
const RECN  = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 2          # record every RECN steps
const TAG   = length(ARGS) > 5 ? ARGS[6] : "L$(LBOX)_ha$(HA)"
const ETAS  = parse(Float64, get(ENV, "SR_ETAS", "0.2"))
const S0    = parse(Float64, get(ENV, "SR_S0", "1e6"))
const KHMAX = parse(Float64, get(ENV, "SR_KHMAX", "1.5"))
const OUT   = get(ENV, "SR_OUTDIR", joinpath(@__DIR__, "..", "data"));  mkpath(OUT)
const COMMIT = try
    c = readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
    success(`git -C $(@__DIR__) diff --quiet HEAD`) ? c : c * "+dirty"
catch
    "unknown"
end
const SEED0 = parse(Int, get(ENV, "SR_SEED0", "0"))                  # seeds SEED0+1 … SEED0+NREAL

function one_realization(seed)
    eos = ConformalEOS(1/3, 1.0, ETAS, 0.0)
    params, ps, cl, eos = setup_ic(0.05, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0);
                                   L=LBOX, lattice=:hex)
    N = params.N;  rngs = make_phys_rng(N, 1000seed + 1);  rng = Xoshiro(1000seed + 2)
    p = ps[1];  V = p.s / p.S_smooth;  w = p.e + p.p;  T = p.T;  Trel = p.T * p.γ
    thermal_init!(ps, params, cl, eos; rng = Xoshiro(1000seed + 3))
    dt = 0.5 * compute_dt(ps, params, eos; mode=:viscous)
    nsteps = ceil(Int, TEND / dt)
    rec = Recorder(box_wavevectors(params, KHMAX))
    dotm = zeros(SVector{2,Float64}, N)
    H0 = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos); sum(q.s * q.H_over_S for q in ps))
    Ht = Float64[];  KE = Float64[];  acc = zeros(2);  reset_guards!()
    t0 = time()
    cache = StochCache()                                             # reused arrays of the stochastic step
    stochrep_step!(ps, params, cl, dt / 2, eos; rng, rngs, vote=:star, cache)
    for n in 1:nsteps
        leapfrog_step!(ps, params, cl, dt, eos, dotm)
        a = stochrep_step!(ps, params, cl, n == nsteps ? dt / 2 : dt, eos; rng, rngs, vote=:star, flip=isodd(n), cache)
        acc .+= (a.stress, a.resid) ./ nsteps
        if n % RECN == 0
            record!(rec, ps, n * dt, eos)
            push!(Ht, sum(q.s * q.H_over_S for q in ps) / H0 - 1)       # smoothed state is current
            push!(KE, sum(dot(q.m, q.m) for q in ps) / (2 * (N - 1) * V * w * T))
        end
    end
    meta = (L = params.L, h = params.h, a = params.a, ha = HA, N = N, dt = dt, nsteps = nsteps,
            etas = ETAS, S0 = S0, V = V, w = w, T = T, Trel = Trel, cs2 = 1/3, seed = seed,
            acc = acc, guards = guards(), wall = time() - t0, rec_every = RECN, khmax = KHMAX,
            tag = TAG, commit = COMMIT, julia = string(VERSION), threads = Threads.nthreads())
    write_equilibrium_h5(joinpath(OUT, "equilibrium_$(TAG)_r$(seed).h5"), meta, rec, Ht, KE)
    @printf("seed %d: N=%d dt=%.2e steps=%d %.0f s  acc=(%.3f, %.3f)  KE/eq=%.3f  ΔH/H=%+.2e  guards=%s\n",
            seed, N, dt, nsteps, time() - t0, acc[1], acc[2], sum(KE[end÷2:end]) / length(KE[end÷2:end]),
            Ht[end], guards())
end

Threads.@threads for s in SEED0 .+ (1:NREAL)
    one_realization(s)
end
println("done")
