# equilibrium.jl — equilibrium runs of the stochastic scheme, for correlators (docs/plan.tex).
#
# A uniform 2D fluid at rest or moving along x, started in equilibrium (SR_START), evolved with the ideal step (the generalized
# leapfrog) and the stochastic viscous step, recording the Fourier modes M_k, S_k (and E_k) and the pair observables.
#
# Time stepping (Strang, merged; src/run.jl strang_loop!): S(Δt/2) [L(Δt) S(Δt)]… L(Δt) S(Δt/2), one stochastic step of length Δt
# between consecutive leapfrogs.
#
# Run:  julia -t auto --project examples/equilibrium.jl L h/a t_end nreal [rec_every] [tag]
#   → <SR_OUTDIR>/equilibrium_<tag>_r<seed>.h5 (write_equilibrium_h5, src/measure.jl; read_equilibrium loads it) and the run
#     record equilibrium_<tag>_r<seed>.toml (src/runinfo.jl).
#
# A run that stops early resumes from equilibrium_<tag>_r<seed>.chk (the RunState, src/run.jl), written every SR_CHECKPOINT_MIN
# minutes: a task that hit its time limit is simply resubmitted.
#
# The inputs (RunConfig, below) are the command-line arguments and the settings of src/settings.jl (`julia src/settings.jl --list`).
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))

# ── The inputs ───────────────────────────────────────────────────────────────
# Every input of a run, read once, in groups; printed at the start and written to the run record ([config]).
#
# physics   s_rel,0 (SR_SREL0), the speed v (SR_BOOST) along x and γ, η/s, ζ/s, c_s² (SR_CS2; 0: the conformal fluid of d = 2),
#           the kernel scale h (SR_H).  The lab entropy density is S = γ s_rel,0, the momentum density M0 = w0 γ² v (target_state).
# numerics  the box L and h/a (arguments), Δt / default (SR_DTFAC), t_end, the record interval (steps), the largest kh recorded
#           (SR_KHMAX), the energy modes E_k (SR_RECORD_E; always in motion), the kernel (SPH_KERNEL).
# start     SR_START: cold | hot | hmc (default), at target_state(s_rel,0, v) (src/measure.jl, src/run.jl start_run!):
#             cold  the lattice as set up, no fluctuations; the run thermalizes.  For demonstrations.
#             hot   the lattice, (e_a, m_a) drawn at the target's β^μ and shifted to the totals of the cold start.  For
#                   demonstrations: cold and hot are on the same shell.
#             hmc   the draw, then hybrid Monte Carlo at the target's β^μ: SR_HMC_N updates (default 200) of trajectories of
#                   length up to SR_HMC_TAU (0: L/c_s), δt = SR_HMC_DTFAC times the acoustic step, a random fraction SR_HMC_TAUJIT
#                   of the length (default 0.5: a fixed length equal to a period of a box sound mode freezes that mode).  No
#                   redraw, no shift (SR_HMC_SHIFT_TOTALS=1, for debugging, shifts the sample to the cold totals).  For production.
#           SR_JITTER: displace the sites by up to ±j·a/2, then make the smoothed densities uniform again.
# output    SR_OUTDIR, SR_CHECKPOINT_MIN, the seeds SR_SEEDOFFSET + SR_SEED0·SR_SEEDSTRIDE + (1…nreal), the commit.
# tests     SR_NOVISC (no stochastic step), SR_NOHAM (no ideal step: positions frozen), SR_STOP_AFTER (stop after that step, after a
#           checkpoint).
struct RunConfig
    args     :: NamedTuple
    physics  :: NamedTuple
    numerics :: NamedTuple
    start    :: NamedTuple
    output   :: NamedTuple
    tests    :: NamedTuple
end
function RunConfig(argv)
    args = (L = parse(Float64, argv[1]), ha = parse(Float64, argv[2]), t_end = parse(Float64, argv[3]), nreal = parse(Int, argv[4]),
            rec_every = length(argv) > 4 ? parse(Int, argv[5]) : 2,
            tag = length(argv) > 5 ? argv[6] : "L$(argv[1])_ha$(argv[2])", argv = collect(argv))
    boost = setting("SR_BOOST")
    physics = (srel0 = setting("SR_SREL0"), boost = boost, gamma = 1 / sqrt(1 - boost^2), etas = setting("SR_ETAS"),
               zetas = setting("SR_ZETAS"), cs2 = setting("SR_CS2", 0.0), h = setting("SR_H"))
    numerics = (dtfac = setting("SR_DTFAC"), khmax = setting("SR_KHMAX", 1.5), record_E = setting("SR_RECORD_E") || boost != 0,
                kernel = setting("SPH_KERNEL"))
    st = setting("SR_START", :hmc)
    st in (:eq, :thermal) && error("SR_START=$st was removed on 2026-11-27 (docs/focus_plan.tex, Q9): use cold, hot or hmc " *
                                   "(hot is the former SR_POSITIONS=lattice SR_START=eq; tag pre-prune)")
    hmc = st === :hmc ? (n = setting("SR_HMC_N"), tau = setting("SR_HMC_TAU"), dtfac = setting("SR_HMC_DTFAC", 0.35),
                         taujit = setting("SR_HMC_TAUJIT", 0.5), shift_totals = setting("SR_HMC_SHIFT_TOTALS")) : nothing
    start = (start = st, jitter = setting("SR_JITTER"), hmc = hmc)
    out = setting("SR_OUTDIR", normpath(joinpath(@__DIR__, "..", "data")));  mkpath(out)
    commit = try
        c = readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
        success(`git -C $(@__DIR__) diff --quiet HEAD`) ? c : c * "+dirty"
    catch
        "unknown"
    end
    output = (outdir = out, checkpoint_min = setting("SR_CHECKPOINT_MIN"), seed0 = setting("SR_SEED0", 0),
              stride = setting("SR_SEEDSTRIDE"), seedoffset = setting("SR_SEEDOFFSET"), commit = commit)
    tests = (novisc = setting("SR_NOVISC"), noham = setting("SR_NOHAM"), stop_after = setting("SR_STOP_AFTER"))
    return RunConfig(args, physics, numerics, start, output, tests)
end
const CFG = RunConfig(ARGS)
function print_config(c::RunConfig)
    for g in (:args, :physics, :numerics, :start, :output, :tests)
        t = getfield(c, g)
        println(rpad("config $g:", 18), join(("$k = $(v isa NamedTuple ? repr(v) : v)" for (k, v) in pairs(t) if k !== :argv), ", "))
    end
end
print_config(CFG)
const P = CFG.physics;  const NU = CFG.numerics;  const A = CFG.args;  const TESTS = CFG.tests

# the settings of the stochastic step (src/stochastic.jl)
stoch_opts(params) = StochOptions(params)

# With SPH_HASTINGS: the log ratios of the proposals, recorded as hast_lq_{stress,resid}_{mean,rms}.
hastings_series(st, on) = on ? (hast_lq_stress_mean = st.lqs_mean, hast_lq_stress_rms = st.lqs_rms,
                                 hast_lq_resid_mean = st.lqr_mean, hast_lq_resid_rms = st.lqr_rms) : (;)

relspread(it) = (v = collect(it); m = sum(v) / length(v); sqrt(sum(abs2, v ./ m .- 1) / length(v)))

# ── The records ──────────────────────────────────────────────────────────────
# the metadata of the .h5 (its attributes; the keys the analysis scripts read)
function run_meta(st::RunState, n, wall, complete, eos)
    params = st.params
    (L = params.L, h = params.h, a = params.a, ha = A.ha, N = params.N, dt = st.dt, nsteps = st.nsteps,
     nsteps_done = n, complete = complete,
     etas = P.etas, srel0 = P.srel0, V = st.V, w = st.w, T = st.T, Trel = st.Trel, cs2 = cs2(eos), phys_dim = params.phys_dim,
     T_scale = eos.T_scale, seed = st.seed,
     acc = st.acc .* (st.nsteps / max(n, 1)), guards = guards(), wall = wall, rec_every = A.rec_every, khmax = NU.khmax,
     monitor = monitor_summary(st.mon),
     boost = P.boost, gamma = P.gamma, record_E = NU.record_E,
     tag = A.tag, start = string(CFG.start.start), jitter = CFG.start.jitter, novisc = TESTS.novisc, dtfac = NU.dtfac,
     noham = TESTS.noham, kernel = NU.kernel, hamiltonian = "energy", volume = "number", commit = CFG.output.commit,
     julia = string(VERSION), threads = Threads.nthreads())
end

# The run record <base>.toml (src/runinfo.jl): written at the start, at each checkpoint and at the end (when [outcome] changes).
function runinfo(base, st::RunState, eos, n, wall, complete; failure = NamedTuple())
    nk = length(st.KE);  h2 = nk ÷ 2 + 1
    res = nk == 0 ? NamedTuple() : (t_reached = n * st.dt, KE_over_eq = sum(st.KE[h2:nk]) / (nk - h2 + 1), dH_over_H = st.Ht[end])
    write_record(base;
        run = (driver = "equilibrium.jl", tag = A.tag, seed = st.seed, args = A.argv),
        config = (physics = P, numerics = NU, start = CFG.start, output = CFG.output, tests = TESTS),
        params = st.params, eos, opts = TESTS.novisc ? nothing : stoch_opts(st.params),
        driver = (L = A.L, h_over_a = A.ha, t_end = A.t_end, nreal = A.nreal, rec_every = A.rec_every, khmax = NU.khmax,
                  srel0 = P.srel0, boost = P.boost, gamma = P.gamma, record_E = NU.record_E, s_a = P.gamma * P.srel0 * st.params.vol,
                  dv = 1 / sqrt(P.srel0 * st.params.vol), dt = st.dt, dt_mode = "viscous", dt_factor = 0.5 * NU.dtfac,
                  nsteps = st.nsteps),
        start = st.start,
        outcome = merge((steps_done = n, complete = complete, wall_s = wall), failure, res,
                        (monitor = monitor_summary(st.mon), guards = guards())))
end

write_h5(base, st, eos, n, wall, complete, hast) =
    write_equilibrium_h5(base * ".h5", run_meta(st, n, wall, complete, eos), st.rec, st.Ht, st.KE; prec = st.prec,
                         extra = merge((; st.sspr, st.Vspr, st.acc_s, st.acc_r, st.grej_s, st.grej_r, st.vmax), hastings_series(st, hast)))

# A step of the generalized leapfrog failed down to Δt/16 (src/hmc.jl, LeapfrogFailure; docs/focus_plan.tex Q11): the state at the
# start of the step and the failing particles go to <base>_leapfrog_fail.jls, the record is marked failed with the reason, and the
# run stops with a nonzero exit.  (The .chk of the last checkpoint is kept.)
function leapfrog_failed(base, err, eos, st, n, wall)
    f = base * "_leapfrog_fail.jls"
    serialize(f, (step = n, t = (n - 1) * st.dt, dt = err.dt, reason = err.reason, x = err.x, m = err.m, s = err.s, bad = err.bad,
                  seed = st.seed, params = st.params, guards = guards()))
    runinfo(base, st, eos, n - 1, wall, false; failure = (failed = true, reason = "step $n: " * err.reason, diagnosis = basename(f)))
    @printf("seed %d: %s\nseed %d: the state at the start of step %d and the failing particles are in %s; stopping\n",
            st.seed, sprint(showerror, err), st.seed, n, f)
    flush(stdout);  exit(1)
end

# ── A realization ────────────────────────────────────────────────────────────
# The set-up and the start: the fluid at the target state, the start, the first stochastic half step, the state of the run.
function new_run(seed, eos)
    label = "seed $seed"
    tgt = target_state(P.srel0, P.boost, eos)
    params, ps, cl, eos, cold = setup_fluid(tgt, eos; h = P.h, ha = A.ha, L = A.L, jitter = CFG.start.jitter, seed = 1000seed + 4, label)
    N = params.N;  rngs = make_phys_rng(N, 1000seed + 1);  rng = Xoshiro(1000seed + 2)
    # the normalization of KE (the momentum fluctuations, 2(N−1) V w T) and the record's w, T, T_rel: the target state
    V = ps[1].V;  w = tgt.w0;  T = tgt.Trel0 / tgt.γ;  Trel = tgt.Trel0
    start = start_run!(ps, params, cl, eos, tgt, cold, CFG.start.start; seed, L = A.L, hmc = CFG.start.hmc, label)
    dt = NU.dtfac * 0.5 * compute_dt(ps, params, eos; mode=:viscous)
    nsteps = ceil(Int, A.t_end / dt)
    H0 = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos); sum(q.H for q in ps))
    reset_guards!();  SSCHEME && reset_sscheme_counts!()
    x0 = [q.x for q in ps]
    TESTS.novisc || stochastic_step!(ps, params, cl, dt / 2, eos, stoch_opts(params); rng, rngs)
    ks = box_wavevectors(params, NU.khmax)
    head = (lattice = "hex", positions = CFG.start.start === :hmc ? "hmc" : "lattice", bath = "nominal", jitter = CFG.start.jitter,
            srel0 = P.srel0, boost = P.boost, gamma = P.gamma, S_lab = P.gamma * P.srel0, M0 = [tgt.M0[1], 0.0],
            s_a = P.gamma * P.srel0 * params.vol, T_rel_uniform = Trel, w_uniform = w, T_rel_rest = tgt.Trel0, w_rest = tgt.w0,
            E_cold = cold.E, P_cold = collect(cold.P),
            rng_seeds = "lattice $(1000seed + 4), start $(1000seed + 3), physics $(1000seed + 1), sweeps $(1000seed + 2)")
    return RunState(seed, params, ps, cl, rngs, rng;
                    rec = Recorder(ks; energies = NU.record_E, x0, L = params.L),
                    prec = ParticleRecorder(ps, params; ks = box_wavevectors(params, NU.khmax), cl),   # + pair observables
                    nsteps, dt, H0, V, w, T, Trel, start = merge(head, start))
end

# One record: the Fourier modes and the pair observables, H, KE, the spreads of s_a and V_a, and the stochastic step since the last
# record (the acceptances and the fractions rejected by the guard — a state no fluid can have, for the conformal EOS exactly
# |v| ≥ 1 — and with SPH_HASTINGS the log ratios); the largest particle speed.
function record_state!(st::RunState, n, eos, hast)
    (; params, ps, cl, dt, H0, V, w, T, win, lqwin) = st;  N = params.N;  recn = A.rec_every
    (TESTS.novisc || TESTS.noham) && (smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos))  # H below needs it
    u = record!(st.prec, ps, n * dt; cl)                          # pair separation, Bragg peak, g(r), ψ6, N_k
    record!(st.rec, ps, n * dt, eos; u)                            # Fourier modes, direct sums (unwrapped u)
    push!(st.Ht, sum(q.H for q in ps) / H0 - 1)                    # smoothed state is current
    SSCHEME && isnan(st.Ht[end]) &&
        @printf("seed %d: a particle without a state at t = %.4f (sscheme_faillog: %d)\n", st.seed, n * dt, length(sscheme_faillog()))
    if CFG.start.jitter > 0 || CFG.start.start === :hmc            # unequal volumes: the mean flow is V_a M̄,
        Md = sum(q.m for q in ps) / sum(q.V for q in ps)           # M̄ the mean momentum density
        push!(st.KE, sum((d = q.m - q.V * Md;  dot(d, d)) for q in ps) / (2 * (N - 1) * V * w * T))
    else
        mbar = sum(q.m for q in ps) / N                            # fluctuations about the mean flow
        push!(st.KE, sum(dot(q.m - mbar, q.m - mbar) for q in ps) / (2 * (N - 1) * V * w * T))
    end
    push!(st.sspr, relspread(entropy_of(q) for q in ps))           # rms s_a/s̄ − 1
    push!(st.Vspr, relspread(q.V for q in ps))                     # rms V_a/V̄ − 1
    push!(st.acc_s, win[1] / recn);  push!(st.acc_r, win[2] / recn)
    push!(st.grej_s, win[3] / recn);  push!(st.grej_r, win[4] / recn);  win .= 0
    if hast
        push!(st.lqs_mean, lqwin[1] / recn);  push!(st.lqs_rms, sqrt(lqwin[2] / recn))
        push!(st.lqr_mean, lqwin[3] / recn);  push!(st.lqr_rms, sqrt(lqwin[4] / recn));  lqwin .= 0
    end
    push!(st.vmax, maximum(norm(q.v) for q in ps))
    observe_particles!(st.mon, ps)
end

function one_realization(seed)
    base = joinpath(CFG.output.outdir, "equilibrium_$(A.tag)_r$(seed)");  chk = base * ".chk"
    eos = P.cs2 > 0 ? ConstantCs2EOS(; cs2_val = P.cs2, η_by_s = P.etas, ζ_by_s = P.zetas) :
                      conformal_eos(default_phys_dim(); η_by_s = P.etas, ζ_by_s = P.zetas)
    if isfile(chk)
        st = resume_run(chk)
        @printf("seed %d: resuming from %s at step %d of %d\n", seed, chk, st.n, st.nsteps)
    else
        st = new_run(seed, eos)
    end
    opts = TESTS.novisc ? nothing : stoch_opts(st.params)
    hast = stoch_opts(st.params).hastings
    REP = max(1, st.nsteps ÷ 20)                     # a log line every 5% of the run
    wall0 = st.wall;  t0 = time();  tchk = time()
    st.n == 0 && runinfo(base, st, eos, 0, wall0, false)
    function after(st, n)
        (n % REP == 0 || n == st.nsteps) && report(st.mon, n * st.dt; label = "seed $seed")
        stop = TESTS.stop_after > 0 && n == TESTS.stop_after
        if n < st.nsteps && (stop || time() - tchk > 60 * CFG.output.checkpoint_min)
            wall = wall0 + time() - t0
            st.n = n;  st.wall = wall;  st.guards = guards();  st.ssref = SS_REF[]
            save_checkpoint(chk, st)
            write_h5(base, st, eos, n, wall, false, hast)
            runinfo(base, st, eos, n, wall, false)
            tchk = time()
            stop && (@printf("seed %d: stopped after step %d (SR_STOP_AFTER)\n", seed, n);  return :stop)
        end
        return :go
    end
    r = strang_loop!(st, eos, opts; noham = TESTS.noham, recn = A.rec_every, record = (st, n) -> record_state!(st, n, eos, hast),
                     after, failed = (err, n) -> leapfrog_failed(base, err, eos, st, n, wall0 + time() - t0))
    r === :stopped && return
    wall = wall0 + time() - t0;  st.n = st.nsteps
    write_h5(base, st, eos, st.nsteps, wall, true, hast)
    runinfo(base, st, eos, st.nsteps, wall, true)
    rm(chk; force = true)
    print_summary(st.mon; label = "seed $seed")
    SSCHEME && !isempty(sscheme_faillog()) && serialize(base * "_ssfail.jls", collect(sscheme_faillog()))   # the states without a Hamiltonian
    (; KE, Ht, acc) = st
    @printf("seed %d: N=%d dt=%.2e steps=%d %.0f s  acc=(%.3f, %.3f)  KE/eq=%.3f  ΔH/H=%+.2e  guards=%s\n",
            seed, st.params.N, st.dt, st.nsteps, wall, acc[1], acc[2], sum(KE[end÷2:end]) / length(KE[end÷2:end]),
            Ht[end], guards())
    SSCHEME && (c = sscheme_counts();  ss = minimum(q.s for q in st.ps);
                @printf("seed %d: S-scheme (form %s): %d solves from the label (%.2f Newton steps, %d fallbacks, %d without a state, %.2e with P_a < 0); %d states from the energy, %d refused at the wall (%.2e); smallest label now %.3f, smallest S_t now %.3f\n",
                        seed, SS_FORM, c.solves, c.iters / max(1, c.solves), c.fallbacks, c.nostate, c.negP / max(1, c.solves), c.fromenergy, c.wall, c.wall / max(1, c.fromenergy), ss,
                        minimum(q.St for q in st.ps)))
end

Threads.@threads for s in CFG.output.seedoffset + CFG.output.seed0 * CFG.output.stride .+ (1:A.nreal)
    one_realization(s)
end
println("done")
