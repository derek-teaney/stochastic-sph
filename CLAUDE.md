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

- Stability limit: runs with entropy per particle ~54 (thermal velocity 0.14) melt the
  lattice and blow up after t ≈ 45; at ~541 (velocity 0.043) they only heat slowly.
  Find the limit versus thermal velocity, η/s, h/a, and the energy fix `SR_HFIX=1`.
- Equilibrium correlators: transverse ⟨M_T(t+τ,k) M_T*(t,k)⟩ against e^{−νk²τ}, and the
  longitudinal (sound) correlator, in boxes large enough to resolve kh ≪ 1.
