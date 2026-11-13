# equilibrium.jl — equilibrium runs of the full stochastic scheme for measuring
# correlators (docs/plan.tex).  A uniform 2D fluid at rest, started in equilibrium
# (SR_START, default eq: initial_state! in src/measure.jl), evolved with the Hamiltonian step and the stochastic step
# in the particle representation, recording the Fourier modes M_k, S_k (and E_k).
#
# Time stepping (Strang, merged): S(Δt/2) [L(Δt) S(Δt)]… L(Δt) S(Δt/2), i.e. one stochastic
# step of length Δt between consecutive leapfrogs (half the cost of S(Δt/2) L S(Δt/2) per step).
#
# Run:  julia -t auto --project examples/equilibrium.jl L h/a t_end nreal [rec_every] [tag]
#   → data/equilibrium_<tag>_r<seed>.h5  (one HDF5 file per realization; see
#     write_equilibrium_h5 in src/measure.jl for the layout, read_equilibrium to load it)
# Restart: a run that stops early resumes from equilibrium_<tag>_r<seed>.chk (SR_CHECKPOINT_MIN).
# Environment: SR_OUTDIR (output directory), SR_SEED0 (seed offset), SR_ETAS, SR_S0, SR_KHMAX,
#   SR_START (cold | eq | thermal; default eq; campaign `first` used thermal).
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const LBOX  = parse(Float64, ARGS[1])
const HA    = parse(Float64, ARGS[2])
const TEND  = parse(Float64, ARGS[3])
const NREAL = parse(Int, ARGS[4])
const RECN  = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 2          # record every RECN steps
const TAG   = length(ARGS) > 5 ? ARGS[6] : "L$(LBOX)_ha$(HA)"
const ETAS  = setting("SR_ETAS", 0.2)
const S0    = setting("SR_S0", 1e6)
const KHMAX = setting("SR_KHMAX", 1.5)
const START = setting("SR_START", :eq)
const OUT   = setting("SR_OUTDIR", normpath(joinpath(@__DIR__, "..", "data")));  mkpath(OUT)
const COMMIT = try
    c = readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
    success(`git -C $(@__DIR__) diff --quiet HEAD`) ? c : c * "+dirty"
catch
    "unknown"
end
const SEED0 = setting("SR_SEED0", 0)                  # seeds SEED0·STRIDE+1 … +NREAL
const STRIDE = setting("SR_SEEDSTRIDE", 1)             # = NREAL when an array task runs NREAL seeds
const SEEDOFF = setting("SR_SEEDOFFSET", 0)           # added to every seed (new realizations of a set)
# Controls for the rigidity tests: SR_NOVISC=1 skips the stochastic step (ideal SPH only),
# SR_NOHAM=1 skips the leapfrog (positions frozen, stochastic step only).  SPH_KERNEL=wc4
# selects the Wendland C4 kernel (src/sph.jl).  Displacements from the initial sites are
# recorded (UL, UT) in every run.
const NOVISC = setting("SR_NOVISC", false)
const NOHAM  = setting("SR_NOHAM", false)
const KERNEL = setting("SPH_KERNEL", "wc2")
# SR_JITTER=j: displace each lattice site uniformly by up to ±j·a/2 in x and y (j ≳ 3: random
# positions), then rescale the s_a so that the smoothed entropy density is uniform again.
const JITTER = setting("SR_JITTER", 0.0)
# SR_BOOST=v: the fluid moves with velocity v along x (boost.tex).  The rest-frame entropy density
# is S0 as before; in the lab (density frame) S = γ S0 and M = (w0 γ² v, 0), w0 the rest-frame
# enthalpy at S0.  The equilibrium start then draws at the moving β^μ of these totals.  With a
# boost the energy Fourier modes E_k are recorded too (SR_RECORD_E=1 forces them at rest).
const BOOST  = setting("SR_BOOST", 0.0)
const GAMMA  = 1 / sqrt(1 - BOOST^2)
const RECORD_E = setting("SR_RECORD_E", false) || BOOST != 0
const LAMBDA = setting("SR_LAMBDA", 1.0)            # residual move: 1 (detailed balance) or Inf
const VOTE   = :star                                                 # stochastic step: one consensus vote per star
# the settings of the stochastic step (src/stochastic.jl); the rest are its defaults
stoch_opts(params) = StochOptions(params; λ = LAMBDA, vote = VOTE)
const DTFAC  = setting("SR_DTFAC", 1.0)             # Δt / default (time-step tests)
# SPH_IDEAL=grid (SimParams.ideal): the ideal step on the fixed lattice (kick, drift and return; docs/gridstep.tex,
# src/grid.jl) in place of the leapfrog.  (Was SR_GRID=1 until 2026-11-13; that name now stops the run.)  The particles never move; the stochastic step is the
# same, at fixed positions.  Its diagnostics are recorded (grid_dST, grid_hkm, grid_hke), and the static spectra of
# all N lattice modes, Σ_records δU_k δU_k† / N with U = (ε, M_x, M_y) the site densities (src/grid.jl, GridFFT),
# summed after t = 5: grid_Ck_re, grid_Ck_im (3×3×N_x×N_y), grid_Cn (the number of records), grid_kmin (2×N_x×N_y).
haskey(ENV, "SR_GRID") && error("SR_GRID was renamed: use SPH_IDEAL=grid (a SimParams option, recorded in [sim])")
const GRID   = setting("SPH_IDEAL", :particle) === :grid      # as SimParams decides (params.ideal)

# Checkpoints: every CHKMIN minutes of wall time the full state goes to <base>.chk and
# the series so far to <base>.h5 (attribute complete = false).  If <base>.chk exists at
# start, the run resumes from it (bit-identically), so a task that hit its time limit is
# simply resubmitted.  SR_STOP_AFTER (tests only) stops after that step, after a checkpoint.
const CHKMIN    = setting("SR_CHECKPOINT_MIN", 20.0)
const STOPAFTER = setting("SR_STOP_AFTER", 0)

function run_meta(params, st, n, wall, complete)
    (L = params.L, h = params.h, a = params.a, ha = HA, N = params.N, dt = st.dt, nsteps = st.nsteps,
     nsteps_done = n, complete = complete,
     etas = ETAS, S0 = S0, V = st.V, w = st.w, T = st.T, Trel = st.Trel, cs2 = 1/3, seed = st.seed,
     acc = st.acc .* (st.nsteps / max(n, 1)), guards = guards(), wall = wall, rec_every = RECN, khmax = KHMAX,
     monitor = haskey(st, :mon) ? monitor_summary(st.mon) : NamedTuple(),
     boost = BOOST, gamma = GAMMA, record_E = RECORD_E,
     grid = GRID, grid_stats = haskey(st, :gstats) ? grid_summary(st.gstats) : NamedTuple(),
     tag = TAG, start = string(START), jitter = JITTER, novisc = NOVISC, lambda = LAMBDA, dtfac = DTFAC, noham = NOHAM, kernel = KERNEL, hamiltonian = string(params.hamiltonian), volume = string(params.volume), commit = COMMIT, julia = string(VERSION), threads = Threads.nthreads())
end

# the series and static spectra of the grid runs for the .h5 (nothing for the particle step, whose files
# are unchanged)
grid_series(st, gF) = (gF === nothing || !haskey(st, :gCk)) ? NamedTuple() :
    (grid_dST = st.grid_dST, grid_hkm = st.grid_hkm, grid_hke = st.grid_hke,
     grid_Ck_re = real.(st.gCk), grid_Ck_im = imag.(st.gCk), grid_Cn = [st.gCn[]],
     grid_kmin = [gF.kmin[i, j][c] for c in 1:2, i in 1:gF.Nx, j in 1:gF.Ny], grid_V = [st.gstencil.V])

relspread(it) = (v = collect(it); m = sum(v) / length(v); sqrt(sum(abs2, v ./ m .- 1) / length(v)))

# The run record <base>.toml (src/runinfo.jl): every setting as used, written at the start,
# at each checkpoint and at the end (when [outcome] changes).
function runinfo(base, params, eos, st, n, complete)
    nk = length(st.KE);  h2 = nk ÷ 2 + 1
    res = nk == 0 ? NamedTuple() : (t_reached = n * st.dt, KE_over_eq = sum(st.KE[h2:nk]) / (nk - h2 + 1), dH_over_H = st.Ht[end])
    write_record(base;
        run = (driver = "equilibrium.jl", tag = TAG, seed = st.seed, args = ARGS),
        params, eos, opts = NOVISC ? nothing : stoch_opts(params),
        driver = (L = LBOX, h_over_a = HA, t_end = TEND, nreal = NREAL, rec_every = RECN, khmax = KHMAX,
                  S0 = S0, grid = GRID, boost = BOOST, gamma = GAMMA, record_E = RECORD_E, s_a = GAMMA * S0 * params.vol,
                  dv = 1 / sqrt(S0 * params.vol), dt = st.dt, dt_mode = "viscous", dt_factor = 0.5 * DTFAC,
                  nsteps = st.nsteps),
        start = haskey(st, :start) ? st.start : (type = string(START),),
        outcome = merge((steps_done = n, complete = complete, wall_s = st.wall), res,
                        haskey(st, :gstats) ? (grid = grid_summary(st.gstats), grid_checks = st.gchecks) : NamedTuple(),
                        (monitor = haskey(st, :mon) ? monitor_summary(st.mon) : NamedTuple(), guards = guards())))
end

function one_realization(seed)
    base = joinpath(OUT, "equilibrium_$(TAG)_r$(seed)");  chk = base * ".chk"
    eos = ConformalEOS(1/3, 1.0, ETAS, 0.0)
    if isfile(chk)
        st = load_checkpoint(chk)
        restore_guards!(st.guards)
        @printf("seed %d: resuming from %s at step %d of %d\n", seed, chk, st.n, st.nsteps)
    else
        # rest-frame state at S0: its enthalpy w0 sets the momentum density of the boosted state
        _, ps0, _, _ = setup_ic(0.05, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0);
                                L=LBOX, lattice=:hex, seed=1000seed + 4)
        w0 = ps0[1].e + ps0[1].p;  Trel0 = ps0[1].T * ps0[1].γ
        Mx0 = w0 * GAMMA^2 * BOOST
        params, ps, cl, eos = setup_ic(0.05, HA, eos, (x, y) -> GAMMA * S0, (x, y) -> SVector(Mx0, 0.0);
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
        start = initial_state!(ps, params, cl, eos, START; rng = Xoshiro(1000seed + 3))
        dt = DTFAC * 0.5 * compute_dt(ps, params, eos; mode=:viscous)
        nsteps = ceil(Int, TEND / dt)
        GRID == (params.ideal === :grid) || error("equilibrium.jl: SPH_IDEAL and SimParams.ideal disagree")
        gstencil = GRID ? GridStencil(ps, params, cl) : nothing
        gchecks = GRID ? grid_checks(gstencil, params, eos, dt) : NamedTuple()
        GRID && print_grid_checks(gchecks)
        H0 = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos); sum(q.s * q.H_over_S for q in ps))
        reset_guards!()
        x0 = [q.x for q in ps]
        NOVISC || stochastic_step!(ps, params, cl, dt / 2, eos, stoch_opts(params); rng, rngs)
        st = (seed = seed, params = params, ps = ps, cl = cl, rngs = rngs, rng = rng,
              dotm = zeros(SVector{2,Float64}, N), rec = Recorder(box_wavevectors(params, KHMAX); energies = RECORD_E, x0, L = params.L),
              prec = ParticleRecorder(ps, params),
              Ht = Float64[], KE = Float64[], sspr = Float64[], Vspr = Float64[],
              acc_s = Float64[], acc_r = Float64[], grej_s = Float64[], grej_r = Float64[], vmax = Float64[], acc = zeros(2), n = 0, nsteps = nsteps, dt = dt,
              H0 = H0, V = V, w = w, T = T, Trel = Trel, wall = 0.0, guards = guards(),
              gstencil = gstencil, gchecks = gchecks, gstats = GridStats(),
              grid_dST = Float64[], grid_hkm = Float64[], grid_hke = Float64[],
              start = merge((lattice = "hex", jitter = JITTER, S0_rest = S0, boost = BOOST, gamma = GAMMA,
                             S_lab = GAMMA * S0, M0 = [Mx0, 0.0], s_a = GAMMA * S0 * params.vol,
                             T_rel_uniform = Trel, w_uniform = w, T_rel_rest = Trel0, w_rest = w0,
                             rng_seeds = "lattice $(1000seed + 4), start $(1000seed + 3), physics $(1000seed + 1), sweeps $(1000seed + 2)"),
                            start))
    end
    (; params, ps, cl, rngs, rng, dotm, rec, prec, Ht, KE, acc, nsteps, dt, H0, V, w, T) = st
    opts = stoch_opts(params)
    haskey(st, :sspr) || (st = merge(st, (sspr = Float64[], Vspr = Float64[])))   # older checkpoints
    haskey(st, :acc_s) || (st = merge(st, (acc_s = Float64[], acc_r = Float64[], grej_s = Float64[], grej_r = Float64[], vmax = Float64[])))
    sspr, Vspr, acc_s, acc_r, grej_s, grej_r, vmax = st.sspr, st.Vspr, st.acc_s, st.acc_r, st.grej_s, st.grej_r, st.vmax
    haskey(st, :mon) || (st = merge(st, (mon = StepMonitor(params.N),)))
    mon = st.mon;  REP = max(1, nsteps ÷ 20)          # a log line every 5% of the run
    st.n == 0 && runinfo(base, params, eos, st, 0, false)
    win = zeros(4)                                   # acceptances and guard fractions since the last record
    N = params.N;  cache = StochCache()                              # reused arrays of the stochastic step
    haskey(st, :gstats) || (st = merge(st, (gstencil = nothing, gchecks = NamedTuple(), gstats = GridStats(),
                                            grid_dST = Float64[], grid_hkm = Float64[], grid_hke = Float64[])))
    gsc = GRID ? GridScratch(N) : nothing;  gfail = false
    gF = GRID ? GridFFT(ps, params) : nothing
    if GRID && !haskey(st, :gCk)
        st = merge(st, (gCk = zeros(ComplexF64, 3, 3, gF.Nx, gF.Ny), gCn = Ref(0)))
    end
    if GRID                                          # warm start of the Newton solves of the grid step
        for a in 1:N;  gsc.er[a] = ps[a].e;  end
    end
    t0 = time();  tchk = time()
    for n in st.n+1:nsteps
        if GRID
            if grid_step!(ps, st.gstencil, dt, eos, gsc, st.gstats; n) > 0
                @printf("seed %d: grid step failed (no state) at step %d: %s\n", seed, n, st.gstats.first_fail)
                gfail = true;  break
            end
        else
            NOHAM || leapfrog_step!(ps, params, cl, dt, eos, dotm)
        end
        if !NOVISC
            a = stochastic_step!(ps, params, cl, n == nsteps ? dt / 2 : dt, eos, opts; rng, rngs, flip=isodd(n), cache)
            acc .+= (a.stress, a.resid) ./ nsteps
            win .+= (a.stress, a.resid, a.gstress, a.gresid)
            observe_step!(mon, a)
        end
        if n % RECN == 0
            (NOVISC || NOHAM || GRID) && (smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos))  # H below needs it
            if GRID                                  # entropy made by the return; power at kh > 1
                push!(st.grid_dST, st.gstats.sum_dS_T)
                push!(st.grid_hkm, grid_highk_fraction(ps, params, [q.m for q in ps]))
                push!(st.grid_hke, grid_highk_fraction(ps, params, [q.s * q.H_over_S for q in ps]))
                if n * dt > 5.0                      # static spectra of all lattice modes
                    Ua = [SVector(q.s * q.H_over_S, q.m[1], q.m[2]) * (q.S_smooth / q.s) for q in ps];  Ub = sum(Ua) / N
                    Uk = [grid_fft(gF, [u[c] - Ub[c] for u in Ua]) for c in 1:3]
                    for c1 in 1:3, c2 in 1:3;  @views st.gCk[c1, c2, :, :] .+= Uk[c1] .* conj.(Uk[c2]) ./ N;  end
                    st.gCn[] += 1
                end
            end
            u = record!(prec, ps, n * dt)                                 # pair separation, Bragg peak
            GRID ? record_grid!(rec, gF, ps, n * dt, eos) :                # Fourier modes: the lattice FFT,
                   record!(rec, ps, n * dt, eos; u)                        # or direct sums (unwrapped u)
            push!(Ht, sum(q.s * q.H_over_S for q in ps) / H0 - 1)       # smoothed state is current
            mbar = sum(q.m for q in ps) / N                                # fluctuations about the mean flow
            push!(KE, sum(dot(q.m - mbar, q.m - mbar) for q in ps) / (2 * (N - 1) * V * w * T))
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
            write_equilibrium_h5(base * ".h5", run_meta(params, st, n, wall, false), rec, Ht, KE; prec, extra = merge((; sspr, Vspr, acc_s, acc_r, grej_s, grej_r, vmax), grid_series(st, gF)))
            runinfo(base, params, eos, merge(st, (wall = wall,)), n, false)
            tchk = time()
            stop && (@printf("seed %d: stopped after step %d (SR_STOP_AFTER)\n", seed, n);  return)
        end
    end
    wall = st.wall + time() - t0
    ndone = gfail ? (st.gstats.steps) : nsteps
    write_equilibrium_h5(base * ".h5", run_meta(params, st, ndone, wall, !gfail), rec, Ht, KE; prec, extra = merge((; sspr, Vspr, acc_s, acc_r, grej_s, grej_r, vmax), grid_series(st, gF)))
    runinfo(base, params, eos, merge(st, (wall = wall,)), ndone, !gfail)
    rm(chk; force = true)
    GRID && (g = grid_summary(st.gstats);  @printf("seed %d: grid ΣΔH_step/H %+.2e  ΣΔS_T/S %+.2e (Σ|ΔS_T|/S %.2e)  max|v|Δt/a %.3f  max|∇·v|Δt %.2e  roundtrip %.1e  ΔP %.1e  fails %d\n",
                                         seed, g.sum_dH_over_H, g.sum_dS_T_over_S, g.sum_absdS_T_over_S, g.disp_max, g.div_max, g.roundtrip_max, g.dP_max, g.eos_fail))
    print_summary(mon; label = "seed $seed")
    @printf("seed %d: N=%d dt=%.2e steps=%d %.0f s  acc=(%.3f, %.3f)  KE/eq=%.3f  ΔH/H=%+.2e  guards=%s\n",
            seed, N, dt, nsteps, wall, acc[1], acc[2], sum(KE[end÷2:end]) / length(KE[end÷2:end]),
            Ht[end], guards())
end

Threads.@threads for s in SEEDOFF + SEED0 * STRIDE .+ (1:NREAL)
    one_realization(s)
end
println("done")
