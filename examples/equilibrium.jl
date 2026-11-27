# equilibrium.jl — equilibrium runs of the full stochastic scheme for measuring
# correlators (docs/plan.tex).  A uniform 2D fluid at rest or moving, started in equilibrium
# (SR_START, default hmc), evolved with the Hamiltonian step and the stochastic step
# in the particle representation, recording the Fourier modes M_k, S_k (and E_k).
#
# Time stepping (Strang, merged): S(Δt/2) [L(Δt) S(Δt)]… L(Δt) S(Δt/2), i.e. one stochastic
# step of length Δt between consecutive leapfrogs (half the cost of S(Δt/2) L S(Δt/2) per step).
#
# Run:  julia -t auto --project examples/equilibrium.jl L h/a t_end nreal [rec_every] [tag]
#   → data/equilibrium_<tag>_r<seed>.h5  (one HDF5 file per realization; see
#     write_equilibrium_h5 in src/measure.jl for the layout, read_equilibrium to load it)
# Restart: a run that stops early resumes from equilibrium_<tag>_r<seed>.chk (SR_CHECKPOINT_MIN).
# Environment: SR_OUTDIR (output directory), SR_SEED0 (seed offset), SR_ETAS, SR_SREL0 (s_rel,0, the rest-frame entropy density), SR_KHMAX,
#   SR_START (cold | hot | hmc; default hmc, below).
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const LBOX  = parse(Float64, ARGS[1])
const HA    = parse(Float64, ARGS[2])
# SR_H: the kernel scale h (default 0.05).  The spacing is a = h/HA, so h/a can be raised at fixed a (and fixed N in the box L)
# by raising SR_H with HA (docs/hbya.tex); with SR_SREL0 fixed the entropy per particle s_a = s_rel,0·a²·√3/2 is then fixed too.
const HKER  = setting("SR_H")
const TEND  = parse(Float64, ARGS[3])
const NREAL = parse(Int, ARGS[4])
const RECN  = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 2          # record every RECN steps
const TAG   = length(ARGS) > 5 ? ARGS[6] : "L$(LBOX)_ha$(HA)"
const ETAS  = setting("SR_ETAS")
# SR_CS2: the speed of sound of the constant-c_s² EOS (default 0: the conformal fluid in d = 2, c_s² = 1/2).  SR_CS2=1/3 with
# SR_ZETAS = (η/s)/3 reproduces the stochastic step of the runs before the dimension fix (8d50d04: c_s² = 1/3, ζ_eff = η/3) at d = 2:
# the dimension enters the tree-level step only through ζ_eff = max(ζ + 2η(1/2 − 1/d), 0.01 η).  SR_ZETAS: ζ/s (default 0).
const CS2V  = setting("SR_CS2", 0.0)
const ZETAS = setting("SR_ZETAS")
const SREL0 = setting("SR_SREL0")        # s_rel,0: the rest-frame entropy density of the run (docs/inputs.tex; was SR_S0)
const KHMAX = setting("SR_KHMAX", 1.5)
# SR_START: the start of a run (docs/focus_plan.tex, Q9; src/measure.jl initial_state!, src/hmc.jl hmc_equilibrate!), at the target
# state of target_state(s_rel,0, SR_BOOST) — the multipliers β⁰ = γ/T_rel,0, β = γv/T_rel,0 from the equation of state:
#   cold  the lattice as set up, no fluctuations; the run thermalizes.  For demonstrations.
#   hot   the lattice, (e_a, m_a) drawn at the target's β^μ, shifted so that the totals are those of the cold start (the
#         totals of the lattice as set up, recorded as E_cold, P_cold): cold and hot are on the same shell.  For demonstrations.
#   hmc   (default) the draw, then hybrid Monte Carlo at the target's β^μ (positions, and (e_a, m_a) through its heat bath).  No
#         redraw after it, no shift.  For production.  SR_HMC_SHIFT_TOTALS=1 (debugging) shifts the final sample to the cold totals.
# Until 2026-11-27 SR_START (cold | eq | thermal), SR_POSITIONS (lattice | poisson | hmc) and SR_BATH (nominal | totals) did this;
# hot is the former SR_POSITIONS=lattice SR_START=eq, cold the former SR_POSITIONS=lattice SR_START=cold (tag pre-prune).
const START = setting("SR_START", :hmc)
START in (:eq, :thermal) && error("SR_START=$START was removed on 2026-11-27 (docs/focus_plan.tex, Q9): use cold, hot or hmc " *
                                  "(hot is the former SR_POSITIONS=lattice SR_START=eq; tag pre-prune)")
START in (:cold, :hot, :hmc) || error("SR_START must be cold, hot or hmc")
const OUT   = setting("SR_OUTDIR", normpath(joinpath(@__DIR__, "..", "data")));  mkpath(OUT)
const COMMIT = try
    c = readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
    success(`git -C $(@__DIR__) diff --quiet HEAD`) ? c : c * "+dirty"
catch
    "unknown"
end
const SEED0 = setting("SR_SEED0", 0)                  # seeds SEED0·STRIDE+1 … +NREAL
const STRIDE = setting("SR_SEEDSTRIDE")             # = NREAL when an array task runs NREAL seeds
const SEEDOFF = setting("SR_SEEDOFFSET")           # added to every seed (new realizations of a set)
# Controls for the rigidity tests: SR_NOVISC=1 skips the stochastic step (ideal SPH only),
# SR_NOHAM=1 skips the leapfrog (positions frozen, stochastic step only).  SPH_KERNEL=wc4
# selects the Wendland C4 kernel (src/sph.jl).  Displacements from the initial sites are
# recorded (UL, UT) in every run.
const NOVISC = setting("SR_NOVISC")
const NOHAM  = setting("SR_NOHAM")
const KERNEL = setting("SPH_KERNEL")
# SR_JITTER=j: displace each lattice site uniformly by up to ±j·a/2 in x and y (j ≳ 3: random
# positions), then rescale the s_a so that the smoothed entropy density is uniform again.
const JITTER = setting("SR_JITTER")
# The hybrid Monte Carlo of SR_START=hmc (src/hmc.jl, docs/equilibrium_positions.tex) at the β^μ of the run: SR_HMC_N updates
# (default 200) with trajectories of length up to SR_HMC_TAU (default L/c_s, the time sound takes to cross the box) and
# δt = SR_HMC_DTFAC (default 0.35) times the acoustic step.  The defaults are those of campaign hbya (2026-11-15).
# SR_HMC_TAUJIT=j: each update's trajectory has a random length, uniform in [(1−j)τ, τ] (default 0.5; 0 is a fixed τ).  A fixed τ equal
# to a period of a box sound mode leaves that mode where it started: τ = L/c_s freezes kh = 2πh/L (S(k) stays near its lattice
# value), τ = L/2c_s the next one (campaign hmc_tau_test, 2026-11-14).  Random lengths remove these resonances.
# The SR_HMC_* settings are read only with SR_START=hmc.
const HMC = START === :hmc ? (n = setting("SR_HMC_N"), tau = setting("SR_HMC_TAU"), dtfac = setting("SR_HMC_DTFAC", 0.35),
                              taujit = setting("SR_HMC_TAUJIT", 0.5), shift_totals = setting("SR_HMC_SHIFT_TOTALS")) : nothing
# SR_BOOST=v: the fluid moves with velocity v along x (boost.tex).  The rest-frame entropy density
# is s_rel,0 as before; in the lab (density frame) S = γ s_rel,0 and M = (w0 γ² v, 0), w0 the rest-frame
# enthalpy at s_rel,0 (target_state).  The start draws at the moving β^μ of the target.  With a
# boost the energy Fourier modes E_k are recorded too (SR_RECORD_E=1 forces them at rest).
const BOOST  = setting("SR_BOOST")
const GAMMA  = 1 / sqrt(1 - BOOST^2)
const RECORD_E = setting("SR_RECORD_E") || BOOST != 0
# the settings of the stochastic step (src/stochastic.jl); the rest are its defaults
stoch_opts(params) = StochOptions(params)
const DTFAC  = setting("SR_DTFAC")             # Δt / default (time-step tests)

# With SPH_HASTINGS (inverse_moment.tex, stage 2): the log ratios of the proposals, recorded as hast_lq_{stress,resid}_{mean,rms}.
# Without it nothing is written, so the output of the other runs is unchanged.
hastings_series(st, on) = on ? (hast_lq_stress_mean = st.lqs_mean, hast_lq_stress_rms = st.lqs_rms,
                                 hast_lq_resid_mean = st.lqr_mean, hast_lq_resid_rms = st.lqr_rms) : (;)

# Checkpoints: every CHKMIN minutes of wall time the full state goes to <base>.chk and
# the series so far to <base>.h5 (attribute complete = false).  If <base>.chk exists at
# start, the run resumes from it (bit-identically), so a task that hit its time limit is
# simply resubmitted.  SR_STOP_AFTER (tests only) stops after that step, after a checkpoint.
const CHKMIN    = setting("SR_CHECKPOINT_MIN")
const STOPAFTER = setting("SR_STOP_AFTER")

function run_meta(params, st, n, wall, complete, eos)
    (L = params.L, h = params.h, a = params.a, ha = HA, N = params.N, dt = st.dt, nsteps = st.nsteps,
     nsteps_done = n, complete = complete,
     etas = ETAS, srel0 = SREL0, V = st.V, w = st.w, T = st.T, Trel = st.Trel, cs2 = cs2(eos), phys_dim = params.phys_dim, T_scale = eos.T_scale, seed = st.seed,
     acc = st.acc .* (st.nsteps / max(n, 1)), guards = guards(), wall = wall, rec_every = RECN, khmax = KHMAX,
     monitor = haskey(st, :mon) ? monitor_summary(st.mon) : NamedTuple(),
     boost = BOOST, gamma = GAMMA, record_E = RECORD_E,
     tag = TAG, start = string(START), jitter = JITTER, novisc = NOVISC, dtfac = DTFAC, noham = NOHAM, kernel = KERNEL, hamiltonian = "energy", volume = "number", commit = COMMIT, julia = string(VERSION), threads = Threads.nthreads())
end

relspread(it) = (v = collect(it); m = sum(v) / length(v); sqrt(sum(abs2, v ./ m .- 1) / length(v)))

# A step of the generalized leapfrog failed down to Δt/16 (src/hmc.jl, LeapfrogFailure; docs/focus_plan.tex Q11): the state at the
# start of the step and the failing particles go to <base>_leapfrog_fail.jls, the record is marked failed with the reason, and the
# run stops with a nonzero exit.  (The .chk of the last checkpoint is kept.)
function leapfrog_failed(base, err, params, eos, st, n, wall, seed)
    f = base * "_leapfrog_fail.jls"
    serialize(f, (step = n, t = (n - 1) * st.dt, dt = err.dt, reason = err.reason, x = err.x, m = err.m, s = err.s, bad = err.bad,
                  seed = seed, params = params, guards = guards()))
    runinfo(base, params, eos, merge(st, (wall = wall,)), n - 1, false;
            failure = (failed = true, reason = "step $n: " * err.reason, diagnosis = basename(f)))
    @printf("seed %d: %s\nseed %d: the state at the start of step %d and the failing particles are in %s; stopping\n",
            seed, sprint(showerror, err), seed, n, f)
    flush(stdout);  exit(1)
end

# The run record <base>.toml (src/runinfo.jl): every setting as used, written at the start,
# at each checkpoint and at the end (when [outcome] changes).
function runinfo(base, params, eos, st, n, complete; failure = NamedTuple())
    nk = length(st.KE);  h2 = nk ÷ 2 + 1
    res = nk == 0 ? NamedTuple() : (t_reached = n * st.dt, KE_over_eq = sum(st.KE[h2:nk]) / (nk - h2 + 1), dH_over_H = st.Ht[end])
    write_record(base;
        run = (driver = "equilibrium.jl", tag = TAG, seed = st.seed, args = ARGS),
        params, eos, opts = NOVISC ? nothing : stoch_opts(params),
        driver = (L = LBOX, h_over_a = HA, t_end = TEND, nreal = NREAL, rec_every = RECN, khmax = KHMAX,
                  srel0 = SREL0, boost = BOOST, gamma = GAMMA, record_E = RECORD_E, s_a = GAMMA * SREL0 * params.vol,
                  dv = 1 / sqrt(SREL0 * params.vol), dt = st.dt, dt_mode = "viscous", dt_factor = 0.5 * DTFAC,
                  nsteps = st.nsteps),
        start = haskey(st, :start) ? st.start : (type = string(START),),
        outcome = merge((steps_done = n, complete = complete, wall_s = st.wall), failure, res,
                        (monitor = haskey(st, :mon) ? monitor_summary(st.mon) : NamedTuple(), guards = guards())))
end

function one_realization(seed)
    base = joinpath(OUT, "equilibrium_$(TAG)_r$(seed)");  chk = base * ".chk"
    eos = CS2V > 0 ? ConstantCs2EOS(; cs2_val = CS2V, η_by_s = ETAS, ζ_by_s = ZETAS) :
                     conformal_eos(default_phys_dim(); η_by_s = ETAS, ζ_by_s = ZETAS)
    if isfile(chk)
        st = load_checkpoint(chk)
        restore_guards!(st.guards)
        # what building SimParams does and deserializing it does not: the scheme's constants (split's V̄; the smoothing length of ω,
        # positional_set!), and the reference state of the S-scheme's measure form (set from the initial condition, saved with the state)
        check_split(st.params);  check_sscheme(st.params)
        haskey(st, :ssref) && all(isfinite, st.ssref) && sscheme_reference!(st.ssref...)
        @printf("seed %d: resuming from %s at step %d of %d\n", seed, chk, st.n, st.nsteps)
    else
        # the target (src/measure.jl target_state): T_rel,0, w0 from the equation of state at s_rel,0 (not from a particle of the
        # set-up, whose T = ∂H/∂s depends on the Hamiltonian), S = γ s_rel,0, M0 = w0 γ² v, the multipliers β⁰, β
        tgt = target_state(SREL0, BOOST, eos);  Trel0 = tgt.Trel0;  w0 = tgt.w0;  Mx0 = tgt.M0[1]
        params, ps, cl, eos = setup_ic(HKER, HA, eos, (x, y) -> tgt.S, (x, y) -> tgt.M0;
                                       L=LBOX, lattice=:hex, jitter=JITTER, seed=1000seed + 4)
        SSCHEME && JITTER > 0 && error("SPH_RENORM=sscheme: SR_JITTER rescales the label as an entropy; not supported")
        if JITTER > 0                            # uniform smoothed densities: s_a ← s_a S/S_a (lab S = γ s_rel,0), iterated;
            for _ in 1:30                        # m_a = V_a M0 (the momentum density of the boosted state)
                for q in ps;  q.s *= tgt.S / q.S_smooth;  end
                thermalize!(ps, params, cl, eos)
            end
            for q in ps;  q.m = (q.s / q.S_smooth) * tgt.M0;  end
            thermalize!(ps, params, cl, eos)
            @printf("seed %d: jitter %.2f a: smoothed S/(γS0) − 1 rms %.1e, s_a/s̄ − 1 rms %.3f\n", seed, JITTER,
                    sqrt(sum(abs2, [q.S_smooth / tgt.S - 1 for q in ps]) / length(ps)),
                    (sb = sum(q.s for q in ps) / length(ps); sqrt(sum(abs2, [q.s / sb - 1 for q in ps]) / length(ps))))
        end
        cold = cold_totals(ps, params, cl, eos)          # the totals of the lattice as set up, before any draw
        N = params.N;  rngs = make_phys_rng(N, 1000seed + 1);  rng = Xoshiro(1000seed + 2)
        # the normalization of KE (the momentum fluctuations, 2(N−1) V w T) and the record's w, T, Trel: the target state
        V = ps[1].V;  w = w0;  T = Trel0 / tgt.γ;  Trel = Trel0
        if START === :hmc                        # the draw, then hybrid Monte Carlo at the target's β^μ
            equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1000seed + 6), bath = tgt.bath)
            tg = HMCTarget(tgt.β0, tgt.β, params.vol, eos)
            dtH = HMC.dtfac * compute_dt(ps, params, eos; mode = :ideal)
            nH = max(1, round(Int, (HMC.tau > 0 ? HMC.tau : LBOX * sqrt(3)) / dtH))
            th = time()
            acch = hmc_equilibrate!(ps, params, cl, eos, tg, Xoshiro(1000seed + 7); nupdates = HMC.n, nsteps = nH, dt = dtH, taujit = HMC.taujit)
            @printf("seed %d: hybrid Monte Carlo, %d updates of %d steps (τ = %.3g), acceptance %.2f, %.0f s\n",
                    seed, HMC.n, nH, nH * dtH, acch, time() - th)
            HMC.shift_totals && shift_to_totals!(ps, params, cl, eos, cold.E, cold.P)
            start = (type = "hmc", what = "equilibrium draw, then hybrid Monte Carlo at the requested beta^mu" *
                                          (HMC.shift_totals ? ", final sample shifted to the cold totals" : ""),
                     hmc_updates = HMC.n, hmc_steps = nH, hmc_taujit = HMC.taujit, hmc_dt = dtH, hmc_tau = nH * dtH, hmc_acceptance = acch,
                     hmc_shift_totals = HMC.shift_totals)
        else
            start = initial_state!(ps, params, cl, eos, START; rng = Xoshiro(1000seed + 3), bath = tgt.bath, totals = cold)
        end
        dt = DTFAC * 0.5 * compute_dt(ps, params, eos; mode=:viscous)
        nsteps = ceil(Int, TEND / dt)
        H0 = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos); sum(q.H for q in ps))
        reset_guards!();  SSCHEME && reset_sscheme_counts!()
        x0 = [q.x for q in ps]
        NOVISC || stochastic_step!(ps, params, cl, dt / 2, eos, stoch_opts(params); rng, rngs)
        st = (seed = seed, params = params, ps = ps, cl = cl, rngs = rngs, rng = rng,
              dotm = zeros(SVector{2,Float64}, N), rec = Recorder(box_wavevectors(params, KHMAX); energies = RECORD_E, x0, L = params.L),
              prec = ParticleRecorder(ps, params; ks = box_wavevectors(params, KHMAX), cl),   # + pair observables
              Ht = Float64[], KE = Float64[], sspr = Float64[], Vspr = Float64[],
              acc_s = Float64[], acc_r = Float64[], grej_s = Float64[], grej_r = Float64[], vmax = Float64[], acc = zeros(2), n = 0, nsteps = nsteps, dt = dt,
              H0 = H0, V = V, w = w, T = T, Trel = Trel, wall = 0.0, guards = guards(),
              start = merge((lattice = "hex", positions = START === :hmc ? "hmc" : "lattice", bath = "nominal", jitter = JITTER,
                             srel0 = SREL0, boost = BOOST, gamma = GAMMA,
                             S_lab = GAMMA * SREL0, M0 = [Mx0, 0.0], s_a = GAMMA * SREL0 * params.vol,
                             T_rel_uniform = Trel, w_uniform = w, T_rel_rest = Trel0, w_rest = w0,
                             E_cold = cold.E, P_cold = collect(cold.P),
                             rng_seeds = "lattice $(1000seed + 4), start $(1000seed + 3), physics $(1000seed + 1), sweeps $(1000seed + 2)"),
                            start))
    end
    (; params, ps, cl, rngs, rng, dotm, rec, prec, Ht, KE, acc, nsteps, dt, H0, V, w, T) = st
    opts = stoch_opts(params)
    haskey(st, :sspr) || (st = merge(st, (sspr = Float64[], Vspr = Float64[])))   # older checkpoints
    haskey(st, :acc_s) || (st = merge(st, (acc_s = Float64[], acc_r = Float64[], grej_s = Float64[], grej_r = Float64[], vmax = Float64[])))
    # with SPH_HASTINGS: the mean and rms of the log ratio ln q′/q of the stress and residual proposals, per record window
    haskey(st, :lqs_mean) || (st = merge(st, (lqs_mean = Float64[], lqs_rms = Float64[], lqr_mean = Float64[], lqr_rms = Float64[])))
    HAST = stoch_opts(params).hastings;  lqwin = zeros(4)
    sspr, Vspr, acc_s, acc_r, grej_s, grej_r, vmax = st.sspr, st.Vspr, st.acc_s, st.acc_r, st.grej_s, st.grej_r, st.vmax
    haskey(st, :mon) || (st = merge(st, (mon = StepMonitor(params.N),)))
    mon = st.mon;  REP = max(1, nsteps ÷ 20)          # a log line every 5% of the run
    st.n == 0 && runinfo(base, params, eos, st, 0, false)
    win = zeros(4)                                   # acceptances and guard fractions since the last record
    N = params.N;  cache = StochCache()                              # reused arrays of the stochastic step
    t0 = time();  tchk = time()
    for n in st.n+1:nsteps
        if !NOHAM
            try
                leapfrog_step!(ps, params, cl, dt, eos, dotm)
            catch err
                err isa LeapfrogFailure || rethrow()
                leapfrog_failed(base, err, params, eos, st, n, st.wall + time() - t0, seed)    # saves, records, exits nonzero
            end
        end
        if !NOVISC
            a = stochastic_step!(ps, params, cl, n == nsteps ? dt / 2 : dt, eos, opts; rng, rngs, flip=isodd(n), cache)
            acc .+= (a.stress, a.resid) ./ nsteps
            win .+= (a.stress, a.resid, a.gstress, a.gresid)
            HAST && (lqwin .+= (a.lq_stress[1], a.lq_stress[2]^2, a.lq_resid[1], a.lq_resid[2]^2))
            observe_step!(mon, a)
        end
        if n % RECN == 0
            (NOVISC || NOHAM) && (smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos))  # H below needs it
            u = record!(prec, ps, n * dt; cl)                             # pair separation, Bragg peak, g(r), ψ6, N_k
            record!(rec, ps, n * dt, eos; u)                               # Fourier modes, direct sums (unwrapped u)
            push!(Ht, sum(q.H for q in ps) / H0 - 1)       # smoothed state is current
            SSCHEME && isnan(Ht[end]) && @printf("seed %d: a particle without a state at t = %.4f (sscheme_faillog: %d)\n", seed, n * dt, length(sscheme_faillog()))
            if JITTER > 0 || START === :hmc                                  # unequal volumes: the mean flow is V_a M̄,
                Md = sum(q.m for q in ps) / sum(q.V for q in ps)    # M̄ the mean momentum density
                push!(KE, sum((d = q.m - q.V * Md;  dot(d, d)) for q in ps) / (2 * (N - 1) * V * w * T))
            else
                mbar = sum(q.m for q in ps) / N                            # fluctuations about the mean flow
                push!(KE, sum(dot(q.m - mbar, q.m - mbar) for q in ps) / (2 * (N - 1) * V * w * T))
            end
            push!(sspr, relspread(entropy_of(q) for q in ps))                       # rms s_a/s̄ − 1
            push!(Vspr, relspread(q.V for q in ps))            # rms V_a/V̄ − 1
            # the stochastic step since the last record: acceptance of the stress (star) and the
            # residual proposals, and the fraction of each rejected by the guard (a state no
            # fluid can have; for the conformal EOS exactly |v| ≥ 1); the largest particle speed
            push!(acc_s, win[1] / RECN);  push!(acc_r, win[2] / RECN)
            push!(grej_s, win[3] / RECN);  push!(grej_r, win[4] / RECN);  win .= 0
            if HAST
                push!(st.lqs_mean, lqwin[1] / RECN);  push!(st.lqs_rms, sqrt(lqwin[2] / RECN))
                push!(st.lqr_mean, lqwin[3] / RECN);  push!(st.lqr_rms, sqrt(lqwin[4] / RECN));  lqwin .= 0
            end
            push!(vmax, maximum(norm(q.v) for q in ps))
            observe_particles!(mon, ps)
        end
        (n % REP == 0 || n == nsteps) && report(mon, n * dt; label = "seed $seed")
        stop = STOPAFTER > 0 && n == STOPAFTER
        if n < nsteps && (stop || time() - tchk > 60CHKMIN)
            wall = st.wall + time() - t0
            save_checkpoint(chk, merge(st, (n = n, wall = wall, guards = guards(), ssref = SS_REF[])))
            write_equilibrium_h5(base * ".h5", run_meta(params, st, n, wall, false, eos), rec, Ht, KE; prec, extra = merge((; sspr, Vspr, acc_s, acc_r, grej_s, grej_r, vmax), hastings_series(st, HAST)))
            runinfo(base, params, eos, merge(st, (wall = wall,)), n, false)
            tchk = time()
            stop && (@printf("seed %d: stopped after step %d (SR_STOP_AFTER)\n", seed, n);  return)
        end
    end
    wall = st.wall + time() - t0
    ndone = nsteps
    write_equilibrium_h5(base * ".h5", run_meta(params, st, ndone, wall, true, eos), rec, Ht, KE; prec, extra = merge((; sspr, Vspr, acc_s, acc_r, grej_s, grej_r, vmax), hastings_series(st, HAST)))
    runinfo(base, params, eos, merge(st, (wall = wall,)), ndone, true)
    rm(chk; force = true)
    print_summary(mon; label = "seed $seed")
    SSCHEME && !isempty(sscheme_faillog()) && serialize(base * "_ssfail.jls", collect(sscheme_faillog()))   # diagnosis of the states without a Hamiltonian
    @printf("seed %d: N=%d dt=%.2e steps=%d %.0f s  acc=(%.3f, %.3f)  KE/eq=%.3f  ΔH/H=%+.2e  guards=%s\n",
            seed, N, dt, nsteps, wall, acc[1], acc[2], sum(KE[end÷2:end]) / length(KE[end÷2:end]),
            Ht[end], guards())
    SSCHEME && (c = sscheme_counts();  ss = minimum(q.s for q in ps);
                @printf("seed %d: S-scheme (form %s): %d solves from the label (%.2f Newton steps, %d fallbacks, %d without a state, %.2e with P_a < 0); %d states from the energy, %d refused at the wall (%.2e); smallest label now %.3f, smallest S_t now %.3f\n",
                        seed, SS_FORM, c.solves, c.iters / max(1, c.solves), c.fallbacks, c.nostate, c.negP / max(1, c.solves), c.fromenergy, c.wall, c.wall / max(1, c.fromenergy), ss,
                        minimum(q.St for q in ps)))
end

Threads.@threads for s in SEEDOFF + SEED0 * STRIDE .+ (1:NREAL)
    one_realization(s)
end
println("done")
