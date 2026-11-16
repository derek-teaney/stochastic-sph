# Stochastic SPH — instructions for Claude

Relativistic smoothed-particle hydrodynamics (2D, density frame) with a stochastic viscous
step, in Julia.  The physics and the algorithms are in `docs/` (LaTeX; read the PDFs or
the `.tex`).  Owner: D. Teaney.

## Where things are

- `docs/plan.tex` — the current plan: speed, NERSC, measurements, next steps, status.
- `docs/fullruns.tex` — what the full Hamiltonian stochastic runs found so far.
- `docs/stochastic.tex` — the stochastic step (stress move, residual move, consensus vote,
  checkerboard sweep, moving fluid).  `docs/changerep.tex`, `docs/bond_dissipation.tex`
  — the deterministic step it builds on.
- `src/` — library, loaded by `src/load.jl`: `sph.jl` (kernels, params, EOS), `celllist.jl`, `ideal.jl`
  (Hamiltonian step), `pointwise.jl` (particle representation), `viscous.jl` (deterministic
  viscous step), `stochastic.jl`, `grid.jl` (the ideal step on the fixed lattice, its lattice FFT and
  bare propagator; `SPH_IDEAL=grid`, `docs/gridstep.tex`), `measure.jl`, …  Superseded code (two-level strain form, the
  old particle sweep) in `src/legacy/`.
- **The viscous step** (since 2026-11-15, branch `inverse-moment`): the default is the inverse-moment step,
  `SPH_DISSIPATION=moment` (`docs/inverse_moment.tex`), with the Hastings correction of the stochastic step
  (`SPH_HASTINGS`, on wherever it applies: moment, λ = 1, the single vote, 2D), and Lorentz factors capped at
  `SPH_GAMMA_MAX=100`.  The faces carry the inverse bond moment (no Λ_a), the residual move is conditioned on zero
  virial, and the coefficients are evaluated at each proposal with the ratio of proposal densities in the vote: the
  stochastic step is in detailed balance across steps (without it the dynamics is 5–20% above HMC at s_a = 54–16).
  To reproduce earlier runs: `SPH_DISSIPATION=common SPH_HASTINGS=0` (2026-11-14/15; `docs/commonkernel.tex`),
  `SPH_DISSIPATION=particle` (before 2026-11-14).
  `docs/commonkernel_comove.tex`: a comoving ideal step, planned, not implemented.
- **The ideal step, the measure and the equilibrium start** (since 2026-11-15, branch `new-defaults`): the defaults
  are the generalized leapfrog (`SPH_IDEAL=generalized`, `src/hmc.jl`), the Liouville measure (`SPH_MEASURE=liouville`,
  S_a − ln T_a in the votes; docs/equilibrium_positions.tex, Appendix A), both in 2D with the energy Hamiltonian, and in
  `examples/equilibrium.jl` the hybrid Monte Carlo start of the positions (`SR_POSITIONS=hmc`, 200 updates,
  `SR_HMC_TAUJIT=0.5`, `SR_HMC_DTFAC=0.35`).  To reproduce runs before that date: `SPH_IDEAL=particle SPH_MEASURE=einstein
  SR_POSITIONS=lattice` (and the HMC settings 40, 0, 0.5 if `SR_POSITIONS=hmc` was set).  `SR_POSITIONS=lattice` remains
  for tests and large s_a; `SPH_IDEAL=grid` starts on the lattice.
- `examples/stability.jl` — energy/stability runs (uniform fluid, Hamiltonian
  step on).  `examples/equilibrium.jl` — equilibrium ensembles for correlators.
  Plotting scripts next to them.
- `test/` — the test suite, run by `test/run.sh [unit|load|regress|all]` (`docs/codeplan.tex`).
  After any change to `src/`: `test/run.sh all` — unit tests pass, every script loads, and the
  regression against `main` is bit-identical unless a change to the numerics is intended.
- `nersc/` — run-only environment, Slurm job array, campaign files, `nersc/README.md`.
- **`nersc/LOG.md` — where things stand at NERSC: setup, lessons, campaigns and their job
  IDs.  Read it first; add to it when you submit or learn something.**

## Conventions (important)

- Temperature: bare `T` = ∂H/∂s = T_rel/γ (`pa.T`); `T_rel = γT` (`pa.T * pa.γ`) appears
  only with the viscous coefficients (T_rel κ, T_rel η, T_rel ζ).
- `M = T^{0i}` is the **density-frame** momentum density — never call it "Landau".
- Notes: brief and pedagogical, displayed equations rather than inline ones, symbols
  defined, no material that was not asked for.
- **LaTeX notation** (match the existing notes, e.g. `docs/stochastic.tex`, `docs/particle_metrics.tex`):
  - the identity is `\one` with `\usepackage{dsfont}` and `\newcommand{\one}{\mathds{1}}`.  **Never
    `\mathbb 1` / `\mathbb{1}`**: amsfonts has no blackboard-bold digits, so it does not render.
  - displayed equations with `\st … \stp` (`\begin{equation}…\end{equation}`); references with `\Eq{}`,
    `\Sect{}`, `\Fig{}`; `\tr` for the trace.
  - symbols: $\bm r_{ab}=\bm x_b-\bm x_a$; $V_a$ the particle volume ($1/n_a$ in the n-scheme); $\beta_a=\bm v_a/T_a$;
    $\ell=\one-\bm v\bm v^T$ the comoving metric; $T_{\rm rel}=\gamma T$; $M$ the density-frame momentum density.
  - bonds (`stochastic.tex`, `bond_dissipation.tex`): the fitted gradient is $(\nabla\beta)_a$ (not $G_a$); the
    residual of a bond $\delta_{ab}\beta=\Delta\beta_{ab}-(\nabla\beta)_a\bm r_{ab}$; the traction (momentum $a$ gives $b$)
    $\pf_{ab}$ with `\newcommand{\pf}{\bm{\mathsf p}}`; the residual move's noise $\mathfrak p_{ab}$ and its residual
    traction $\vec p_{ab}$, with the virial $\sum_c\mathfrak p_{ac}\bm r_{ac}^T$ written out; faces $\bm A_{ab}$; the stress
    noise $\xi_a$; the traceless (comoving) strain $\mathring S_a$; $\Delta\mathbb S$ the entropy change.
  - names should say what they are: no uninformative letters such as $G$, $Y$, $y$, $t$ for quantities that
    already have a name in the notes.
- Commit or push only when the owner asks.  Commit messages end with
  `Co-Authored-By: Claude <noreply@anthropic.com>` (or the model's own line).

## Working at NERSC

You may be running on a Perlmutter login node.  Rules:

1. **Never run simulations on the login node** beyond a seconds-long smoke test.  Submit
   Slurm jobs (`nersc/array.sh`, `-q shared` or `-q regular`), or use an interactive job.
2. **Budget**: ask the owner before submitting more than ~20 node-hours in one go, and
   report the estimate (tasks × cores × hours / 128) before submitting.
3. Use the run-only environment: `julia --project=nersc`.  The depot is set by
   `JULIA_DEPOT_PATH` (see `nersc/README.md`); precompile on the login node first.
4. Output goes to `$SCRATCH/sph/<campaign>/<run>/`.  Never commit `.jls` or `.h5` data.
   Account m3722; submit from `$SCRATCH/sph` with `SR_REPO` set (`nersc/LOG.md`).
   Short tests in `-q debug`.
5. Monitor with `squeue --me`, `sacct`; read `slurm/*.out` for failures.  If a task fails,
   find the cause before resubmitting.
6. Report back through git: commit **summaries** (tables, plots, a short note in
   `docs/`), not raw data, on the working branch, when the owner asks.

## Current questions (see docs/fullruns.tex, docs/plan.tex)

- Long single runs (the owner wants runs that go for hours, ideally forever, rather than many
  short realizations): not yet tested.  The blow-up at entropy per particle ~54 (t ≈ 45) belonged
  to the old smoothed Hamiltonian and is no longer relevant.  With the n-scheme (default) the
  issue is a slow drift of H at low entropy per particle (energy2: cooling, ΔH/H −31% by
  t = 200 at s_a ≈ 54; made by the residual moves, not reduced by Δt at s_a = 541; campaign
  `cl` stable to s_a ≈ 4 with the Δt criterion).  Found 2026-11-14 (docs/equilibrium_positions.tex,
  "The generalized leapfrog"): at s_a = 54 the drift is first order in Δt and made by the
  drift–kick–drift ideal step (H is not separable); the generalized leapfrog, `SPH_IDEAL=generalized`,
  removes it (drift 1.9e-4 → ~3e-6 per unit time, full step ≈ 1.7× the cost); it is the default since 2026-11-15.
- Detailed balance across steps: solved by the Hastings correction (default since 2026-11-15; docs/inverse_moment.tex,
  Stage 2, laptop and NERSC).  At s_a ≈ 16 and below the residual log ratios have large tails (particles at γ ~ 10–100);
  NERSC campaign `hastings_low` (s_a = 12, 8, 4) explores that edge.
- Renormalization of the transport coefficients (docs/helfand.tex): the Helfand moments of the particles, the Helfand
  separation (the analogue of the pair separation D_rel(r)), normalized by the momentum fluctuations, with the statics
  (the initial pair correlation of the moments) from hybrid Monte Carlo.  First case: s_a ≈ 100 at rest.
- The equilibrium of the particle positions (docs/equilibrium_positions.tex, branch
  `equilibrium`): g(r), ψ6, S(k) from the lattice; hybrid Monte Carlo as the reference.
- Equilibrium correlators: transverse ⟨M_T(t+τ,k) M_T*(t,k)⟩ against e^{−νk²τ}, and the
  longitudinal (sound) correlator, in boxes large enough to resolve kh ≪ 1.
