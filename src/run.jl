# run.jl — the pieces of a run of the stochastic scheme (docs/focus_plan.tex Q14), used by examples/equilibrium.jl and the other
# drivers: the fluid at the target state, the start, the state of a run (what a checkpoint holds), and the Strang loop.

# ── The fluid ────────────────────────────────────────────────────────────────
# The fluid at the target state tgt (target_state, measure.jl) on the hexagonal lattice: S = γ s_rel,0, M0 = w0 γ² v.
#
# jitter > 0: the sites displaced (setup.jl), then the s_a rescaled so that the smoothed entropy density is uniform again
# (iterated), and m_a = V_a M0.  Not with the S-scheme, whose label is not an entropy.
#
# Returns (params, ps, cl, eos, cold), cold the totals of the lattice as set up (cold_totals).
function setup_fluid(tgt, eos; h, ha, L, jitter = 0.0, seed = 42, label = "")
    params, ps, cl, eos = setup_ic(h, ha, eos, (x, y) -> tgt.S, (x, y) -> tgt.M0; L, lattice = :hex, jitter, seed)
    SSCHEME && jitter > 0 && error("SPH_RENORM=sscheme: SR_JITTER rescales the label as an entropy; not supported")
    if jitter > 0
        for _ in 1:30
            for q in ps;  q.s *= tgt.S / q.S_smooth;  end
            thermalize!(ps, params, cl, eos)
        end
        for q in ps;  q.m = (q.s / q.S_smooth) * tgt.M0;  end
        thermalize!(ps, params, cl, eos)
        @printf("%s: jitter %.2f a: smoothed S/(γS0) − 1 rms %.1e, s_a/s̄ − 1 rms %.3f\n", label, jitter,
                sqrt(sum(abs2, [q.S_smooth / tgt.S - 1 for q in ps]) / length(ps)),
                (sb = sum(q.s for q in ps) / length(ps); sqrt(sum(abs2, [q.s / sb - 1 for q in ps]) / length(ps))))
    end
    cold = cold_totals(ps, params, cl, eos)
    return params, ps, cl, eos, cold
end

# ── The start ────────────────────────────────────────────────────────────────
# SR_START (docs/focus_plan.tex Q9): cold and hot (initial_state!, measure.jl), or hmc: the draw at the target's β^μ, then hybrid
# Monte Carlo (hmc_equilibrate!, hmc.jl) with the settings hmc = (n, tau, dtfac, taujit, shift_totals); tau = 0 is √3 L (L/c_s at
# c_s² = 1/3).  The random numbers: the draw 1000 seed + 6, the chain 1000 seed + 7; the hot draw 1000 seed + 3.
# Returns the description of the start, for the run record.
function start_run!(ps, params, cl, eos, tgt, cold, start::Symbol; seed, L, hmc = nothing, label = "")
    start === :hmc || return initial_state!(ps, params, cl, eos, start; rng = Xoshiro(1000seed + 3), bath = tgt.bath, totals = cold)
    equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1000seed + 6), bath = tgt.bath)
    tg = HMCTarget(tgt.β0, tgt.β, params.vol, eos)
    dtH = hmc.dtfac * compute_dt(ps, params, eos; mode = :ideal)
    nH = max(1, round(Int, (hmc.tau > 0 ? hmc.tau : L * sqrt(3)) / dtH))
    th = time()
    acch = hmc_equilibrate!(ps, params, cl, eos, tg, Xoshiro(1000seed + 7); nupdates = hmc.n, nsteps = nH, dt = dtH, taujit = hmc.taujit)
    @printf("%s: hybrid Monte Carlo, %d updates of %d steps (τ = %.3g), acceptance %.2f, %.0f s\n",
            label, hmc.n, nH, nH * dtH, acch, time() - th)
    hmc.shift_totals && shift_to_totals!(ps, params, cl, eos, cold.E, cold.P)
    return (type = "hmc", what = "equilibrium draw, then hybrid Monte Carlo at the requested beta^mu" *
                                 (hmc.shift_totals ? ", final sample shifted to the cold totals" : ""),
            hmc_updates = hmc.n, hmc_steps = nH, hmc_taujit = hmc.taujit, hmc_dt = dtH, hmc_tau = nH * dtH, hmc_acceptance = acch,
            hmc_shift_totals = hmc.shift_totals)
end

# ── The state of a run ───────────────────────────────────────────────────────
# Everything a run carries from step to step: the particles and the random numbers, the recorders and the series, the running
# sums.  A checkpoint is this struct, serialized (save_checkpoint, measure.jl); a run resumes from it bit-identically, with the
# record windows (win, lqwin) and the monitor.
mutable struct RunState
    seed   :: Int
    params :: SimParams
    ps     :: Vector{Particle}
    cl     :: CellList2D
    rngs   :: Vector{Xoshiro}                 # the particles' random numbers
    rng    :: Xoshiro                         # the sweep orders and shifts
    dotm   :: Vector{SVector{2,Float64}}
    rec    :: Recorder                        # the Fourier modes
    prec   :: ParticleRecorder                # the pair observables
    Ht     :: Vector{Float64}                 # H/H0 − 1 at the records
    KE     :: Vector{Float64}                 # the momentum fluctuations / their equilibrium value
    sspr   :: Vector{Float64}                 # rms s_a/s̄ − 1
    Vspr   :: Vector{Float64}                 # rms V_a/V̄ − 1
    acc_s  :: Vector{Float64}                 # per record window: acceptances and guard fractions of the stress and residual moves
    acc_r  :: Vector{Float64}
    grej_s :: Vector{Float64}
    grej_r :: Vector{Float64}
    vmax   :: Vector{Float64}                 # the largest particle speed
    lqs_mean :: Vector{Float64}               # with SPH_HASTINGS: mean and rms of ln q′/q of the proposals, per record window
    lqs_rms  :: Vector{Float64}
    lqr_mean :: Vector{Float64}
    lqr_rms  :: Vector{Float64}
    acc    :: Vector{Float64}                 # the mean acceptances (stress, residual) over the run
    win    :: Vector{Float64}                 # the sums of the record window in progress
    lqwin  :: Vector{Float64}
    mon    :: StepMonitor
    n      :: Int                             # the steps done
    nsteps :: Int
    dt     :: Float64
    H0     :: Float64
    V      :: Float64                         # the normalization of KE, 2(N − 1) V w T: the target state's w, T
    w      :: Float64
    T      :: Float64
    Trel   :: Float64
    wall   :: Float64                         # wall time so far (s)
    guards :: NamedTuple
    start  :: NamedTuple                      # the description of the start (the run record's [start])
    ssref  :: Tuple{Float64,Float64}          # the reference state of the S-scheme's measure form
end
function RunState(seed, params, ps, cl, rngs, rng; rec, prec, nsteps, dt, H0, V, w, T, Trel, start)
    f() = Float64[]
    return RunState(seed, params, ps, cl, rngs, rng, zeros(SVector{2,Float64}, params.N), rec, prec, f(), f(), f(), f(),
                    f(), f(), f(), f(), f(), f(), f(), f(), f(), zeros(2), zeros(4), zeros(4), StepMonitor(params.N),
                    0, nsteps, dt, H0, V, w, T, Trel, 0.0, guards(), start, (NaN, NaN))
end

# Resume from a checkpoint: the state, and what building SimParams does and deserializing it does not (the scheme's constants:
# split's V̄, the smoothing length of ω; the reference state of the S-scheme's measure form).
function resume_run(chk)
    st = load_checkpoint(chk)
    st isa RunState || error("$chk: a checkpoint of the code before docs/focus_plan.tex Q14 (2026-11-27) does not resume; remove it " *
                             "and rerun, or resume with the tag focus-q11")
    restore_guards!(st.guards)
    check_split(st.params);  check_sscheme(st.params)
    all(isfinite, st.ssref) && sscheme_reference!(st.ssref...)
    return st
end

# ── The Strang loop ──────────────────────────────────────────────────────────
# S(Δt/2) [L(Δt) S(Δt)]… L(Δt) S(Δt/2): one stochastic step of length Δt between consecutive leapfrogs (the first half step is the
# caller's, after the start).  From step st.n + 1 to st.nsteps:
#   the ideal step (unless noham); a LeapfrogFailure goes to failed(err, n) (hmc.jl, docs/focus_plan.tex Q11);
#   the stochastic step (unless opts === nothing), its acceptances summed into st.acc, st.win, st.lqwin and the monitor;
#   record(st, n) every recn steps;  after(st, n) after each step: :stop ends the loop (returns :stopped).
function strang_loop!(st::RunState, eos, opts; noham::Bool, recn::Int, record, after, failed)
    (; params, ps, cl, rngs, rng, dotm, nsteps, dt, acc, win, lqwin, mon) = st
    hast = opts !== nothing && opts.hastings;  cache = StochCache()
    for n in st.n+1:nsteps
        if !noham
            try
                leapfrog_step!(ps, params, cl, dt, eos, dotm)
            catch err
                err isa LeapfrogFailure || rethrow()
                failed(err, n)
            end
        end
        if opts !== nothing
            a = stochastic_step!(ps, params, cl, n == nsteps ? dt / 2 : dt, eos, opts; rng, rngs, flip=isodd(n), cache)
            acc .+= (a.stress, a.resid) ./ nsteps
            win .+= (a.stress, a.resid, a.gstress, a.gresid)
            hast && (lqwin .+= (a.lq_stress[1], a.lq_stress[2]^2, a.lq_resid[1], a.lq_resid[2]^2))
            observe_step!(mon, a)
        end
        n % recn == 0 && record(st, n)
        after(st, n) === :stop && return :stopped
    end
    return :done
end
