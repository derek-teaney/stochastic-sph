# stochrep_equilibrium.jl — equilibrium runs of the full stochastic scheme for measuring
# correlators (docs/plan.tex).  A uniform 2D fluid at rest, started in equilibrium
# (SR_START, default eq: initial_state! in src/measure.jl), evolved with the Hamiltonian step and the stochastic step
# in the particle representation, recording the Fourier modes M_k, S_k (and E_k).
#
# Time stepping (Strang, merged): S(Δt/2) [L(Δt) S(Δt)]… L(Δt) S(Δt/2), i.e. one stochastic
# step of length Δt between consecutive leapfrogs (half the cost of S(Δt/2) L S(Δt/2) per step).
#
# Run:  julia -t auto --project examples/stochrep_equilibrium.jl L h/a t_end nreal [rec_every] [tag]
#   → data/equilibrium_<tag>_r<seed>.h5  (one HDF5 file per realization; see
#     write_equilibrium_h5 in src/measure.jl for the layout, read_equilibrium to load it)
# Restart: a run that stops early resumes from equilibrium_<tag>_r<seed>.chk (SR_CHECKPOINT_MIN).
# Environment: SR_OUTDIR (output directory), SR_SEED0 (seed offset), SR_ETAS, SR_S0, SR_KHMAX,
#   SR_START (cold | eq | thermal; default eq; campaign `first` used thermal).
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
include(joinpath(@__DIR__, "..", "src", "rng.jl"))
include(joinpath(@__DIR__, "..", "src", "stochrep.jl"))
include(joinpath(@__DIR__, "..", "src", "measure.jl"))
include(joinpath(@__DIR__, "..", "src", "monitor.jl"))

const LBOX  = parse(Float64, ARGS[1])
const HA    = parse(Float64, ARGS[2])
const TEND  = parse(Float64, ARGS[3])
const NREAL = parse(Int, ARGS[4])
const RECN  = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 2          # record every RECN steps
const TAG   = length(ARGS) > 5 ? ARGS[6] : "L$(LBOX)_ha$(HA)"
const ETAS  = parse(Float64, get(ENV, "SR_ETAS", "0.2"))
const S0    = parse(Float64, get(ENV, "SR_S0", "1e6"))
const KHMAX = parse(Float64, get(ENV, "SR_KHMAX", "1.5"))
const START = Symbol(get(ENV, "SR_START", "eq"))
const OUT   = get(ENV, "SR_OUTDIR", joinpath(@__DIR__, "..", "data"));  mkpath(OUT)
const COMMIT = try
    c = readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
    success(`git -C $(@__DIR__) diff --quiet HEAD`) ? c : c * "+dirty"
catch
    "unknown"
end
const SEED0 = parse(Int, get(ENV, "SR_SEED0", "0"))                  # seeds SEED0·STRIDE+1 … +NREAL
const STRIDE = parse(Int, get(ENV, "SR_SEEDSTRIDE", "1"))             # = NREAL when an array task runs NREAL seeds
# Controls for the rigidity tests: SR_NOVISC=1 skips the stochastic step (ideal SPH only),
# SR_NOHAM=1 skips the leapfrog (positions frozen, stochastic step only).  SPH_KERNEL=wc4
# selects the Wendland C4 kernel (src/sph.jl).  Displacements from the initial sites are
# recorded (UL, UT) in every run.
const NOVISC = get(ENV, "SR_NOVISC", "0") == "1"
const NOHAM  = get(ENV, "SR_NOHAM", "0") == "1"
const KERNEL = get(ENV, "SPH_KERNEL", "wc2")
# SR_JITTER=j: displace each lattice site uniformly by up to ±j·a/2 in x and y (j ≳ 3: random
# positions), then rescale the s_a so that the smoothed entropy density is uniform again.
const JITTER = parse(Float64, get(ENV, "SR_JITTER", "0"))
const LAMBDA = parse(Float64, get(ENV, "SR_LAMBDA", "1"))            # residual move: 1 (detailed balance) or Inf
const DTFAC  = parse(Float64, get(ENV, "SR_DTFAC", "1"))             # Δt / default (time-step tests)

# Checkpoints: every CHKMIN minutes of wall time the full state goes to <base>.chk and
# the series so far to <base>.h5 (attribute complete = false).  If <base>.chk exists at
# start, the run resumes from it (bit-identically), so a task that hit its time limit is
# simply resubmitted.  SR_STOP_AFTER (tests only) stops after that step, after a checkpoint.
const CHKMIN    = parse(Float64, get(ENV, "SR_CHECKPOINT_MIN", "20"))
const STOPAFTER = parse(Int, get(ENV, "SR_STOP_AFTER", "0"))

function run_meta(params, st, n, wall, complete)
    (L = params.L, h = params.h, a = params.a, ha = HA, N = params.N, dt = st.dt, nsteps = st.nsteps,
     nsteps_done = n, complete = complete,
     etas = ETAS, S0 = S0, V = st.V, w = st.w, T = st.T, Trel = st.Trel, cs2 = 1/3, seed = st.seed,
     acc = st.acc .* (st.nsteps / max(n, 1)), guards = guards(), wall = wall, rec_every = RECN, khmax = KHMAX,
     monitor = haskey(st, :mon) ? monitor_summary(st.mon) : NamedTuple(),
     tag = TAG, start = string(START), jitter = JITTER, novisc = NOVISC, lambda = LAMBDA, dtfac = DTFAC, noham = NOHAM, kernel = KERNEL, hamiltonian = string(params.hamiltonian), volume = string(params.volume), commit = COMMIT, julia = string(VERSION), threads = Threads.nthreads())
end

relspread(it) = (v = collect(it); m = sum(v) / length(v); sqrt(sum(abs2, v ./ m .- 1) / length(v)))

function one_realization(seed)
    base = joinpath(OUT, "equilibrium_$(TAG)_r$(seed)");  chk = base * ".chk"
    eos = ConformalEOS(1/3, 1.0, ETAS, 0.0)
    if isfile(chk)
        st = load_checkpoint(chk)
        restore_guards!(st.guards)
        @printf("seed %d: resuming from %s at step %d of %d\n", seed, chk, st.n, st.nsteps)
    else
        params, ps, cl, eos = setup_ic(0.05, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0);
                                       L=LBOX, lattice=:hex, jitter=JITTER, seed=1000seed + 4)
        if JITTER > 0                            # uniform smoothed entropy: s_a ← s_a S0 / S_a, iterated
            for _ in 1:30
                for q in ps;  q.s *= S0 / q.S_smooth;  end
                thermalize!(ps, params, cl, eos)
            end
            @printf("seed %d: jitter %.2f a: smoothed S/S0 − 1 rms %.1e, s_a/s̄ − 1 rms %.3f\n", seed, JITTER,
                    sqrt(sum(abs2, [q.S_smooth / S0 - 1 for q in ps]) / length(ps)),
                    (sb = sum(q.s for q in ps) / length(ps); sqrt(sum(abs2, [q.s / sb - 1 for q in ps]) / length(ps))))
        end
        N = params.N;  rngs = make_phys_rng(N, 1000seed + 1);  rng = Xoshiro(1000seed + 2)
        p = ps[1];  V = p.s / p.S_smooth;  w = p.e + p.p;  T = p.T;  Trel = p.T * p.γ
        initial_state!(ps, params, cl, eos, START; rng = Xoshiro(1000seed + 3))
        dt = DTFAC * 0.5 * compute_dt(ps, params, eos; mode=:viscous)
        nsteps = ceil(Int, TEND / dt)
        H0 = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos); sum(q.s * q.H_over_S for q in ps))
        reset_guards!()
        x0 = [q.x for q in ps]
        NOVISC || stochrep_step!(ps, params, cl, dt / 2, eos; λ = LAMBDA, rng, rngs, vote=:star)
        st = (seed = seed, params = params, ps = ps, cl = cl, rngs = rngs, rng = rng,
              dotm = zeros(SVector{2,Float64}, N), rec = Recorder(box_wavevectors(params, KHMAX); x0, L = params.L),
              prec = ParticleRecorder(ps, params),
              Ht = Float64[], KE = Float64[], sspr = Float64[], Vspr = Float64[],
              acc_s = Float64[], acc_r = Float64[], grej_s = Float64[], grej_r = Float64[], vmax = Float64[], acc = zeros(2), n = 0, nsteps = nsteps, dt = dt,
              H0 = H0, V = V, w = w, T = T, Trel = Trel, wall = 0.0, guards = guards())
    end
    (; params, ps, cl, rngs, rng, dotm, rec, prec, Ht, KE, acc, nsteps, dt, H0, V, w, T) = st
    haskey(st, :sspr) || (st = merge(st, (sspr = Float64[], Vspr = Float64[])))   # older checkpoints
    haskey(st, :acc_s) || (st = merge(st, (acc_s = Float64[], acc_r = Float64[], grej_s = Float64[], grej_r = Float64[], vmax = Float64[])))
    sspr, Vspr, acc_s, acc_r, grej_s, grej_r, vmax = st.sspr, st.Vspr, st.acc_s, st.acc_r, st.grej_s, st.grej_r, st.vmax
    haskey(st, :mon) || (st = merge(st, (mon = StepMonitor(params.N),)))
    mon = st.mon;  REP = max(1, nsteps ÷ 20)          # a log line every 5% of the run
    win = zeros(4)                                   # acceptances and guard fractions since the last record
    N = params.N;  cache = StochCache()                              # reused arrays of the stochastic step
    t0 = time();  tchk = time()
    for n in st.n+1:nsteps
        NOHAM || leapfrog_step!(ps, params, cl, dt, eos, dotm)
        if !NOVISC
            a = stochrep_step!(ps, params, cl, n == nsteps ? dt / 2 : dt, eos; λ = LAMBDA, rng, rngs, vote=:star, flip=isodd(n), cache)
            acc .+= (a.stress, a.resid) ./ nsteps
            win .+= (a.stress, a.resid, a.gstress, a.gresid)
            observe_step!(mon, a)
        end
        if n % RECN == 0
            (NOVISC || NOHAM) && (smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos))  # H below needs it
            u = record!(prec, ps, n * dt)                                 # pair separation, Bragg peak
            record!(rec, ps, n * dt, eos; u)                               # Fourier modes (unwrapped u)
            push!(Ht, sum(q.s * q.H_over_S for q in ps) / H0 - 1)       # smoothed state is current
            push!(KE, sum(dot(q.m, q.m) for q in ps) / (2 * (N - 1) * V * w * T))
            push!(sspr, relspread(q.s for q in ps))                         # rms s_a/s̄ − 1
            push!(Vspr, relspread(q.s / q.S_smooth for q in ps))            # rms V_a/V̄ − 1
            # the stochastic step since the last record: acceptance of the stress (star) and the
            # residual proposals, and the fraction of each rejected by the guard (a state no
            # fluid can have; for the conformal EOS exactly |v| ≥ 1); the largest particle speed
            push!(acc_s, win[1] / RECN);  push!(acc_r, win[2] / RECN)
            push!(grej_s, win[3] / RECN);  push!(grej_r, win[4] / RECN);  win .= 0
            push!(vmax, maximum(norm(q.v_over_S * q.S_smooth) for q in ps))
            observe_particles!(mon, ps)
        end
        (n % REP == 0 || n == nsteps) && report(mon, n * dt; label = "seed $seed")
        stop = STOPAFTER > 0 && n == STOPAFTER
        if n < nsteps && (stop || time() - tchk > 60CHKMIN)
            wall = st.wall + time() - t0
            save_checkpoint(chk, merge(st, (n = n, wall = wall, guards = guards())))
            write_equilibrium_h5(base * ".h5", run_meta(params, st, n, wall, false), rec, Ht, KE; prec, extra = (; sspr, Vspr, acc_s, acc_r, grej_s, grej_r, vmax))
            tchk = time()
            stop && (@printf("seed %d: stopped after step %d (SR_STOP_AFTER)\n", seed, n);  return)
        end
    end
    wall = st.wall + time() - t0
    write_equilibrium_h5(base * ".h5", run_meta(params, st, nsteps, wall, true), rec, Ht, KE; prec, extra = (; sspr, Vspr, acc_s, acc_r, grej_s, grej_r, vmax))
    rm(chk; force = true)
    print_summary(mon; label = "seed $seed")
    @printf("seed %d: N=%d dt=%.2e steps=%d %.0f s  acc=(%.3f, %.3f)  KE/eq=%.3f  ΔH/H=%+.2e  guards=%s\n",
            seed, N, dt, nsteps, wall, acc[1], acc[2], sum(KE[end÷2:end]) / length(KE[end÷2:end]),
            Ht[end], guards())
end

Threads.@threads for s in SEED0 * STRIDE .+ (1:NREAL)
    one_realization(s)
end
println("done")
