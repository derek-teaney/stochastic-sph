# Stochastic SPH — instructions for Claude

Relativistic smoothed-particle hydrodynamics (2D, density frame) with a stochastic viscous
step, in Julia.  The physics and the algorithms are in `docs/` (LaTeX; read the PDFs or
the `.tex`).  Owner: D. Teaney.

## Where things are

- **The code has one scheme** (since 2026-11-27, branch `focus`; `docs/focus_plan.tex`, its Results section, and
  `docs/code_review_notes.md`): the n-scheme (H = Σ_a e_a, V_a = 1/n_a; `docs/volume_n.tex`), the generalized leapfrog
  (`src/hmc.jl`), the inverse-moment viscous step with the Hastings correction (`docs/inverse_moment.tex`), the Liouville measure
  (S_a − ln T_a in the votes), two dimensions (d = 2, or d = 3 for deterministic runs), the hexagonal lattice.  Everything else
  (the smoothed Hamiltonian, the entropy volume, the DKD leapfrog, the grid step, the other dissipations, λ ≠ 1, the bond votes,
  the Einstein measure, the change of representation, 1D, `src/legacy/`, the Poisson start, `SR_BATH=totals`) is in the tag
  **`pre-prune`**; the settings that chose them stop with a message (`src/settings.jl`, `RETIRED_SETTINGS`).
- **The renormalization**: `SPH_RENORM=split` (the default; `src/renorm_split.jl`), `sscheme` (`src/sscheme.jl`; forms `measure`
  (default) and `exact`, `SPH_SSCHEME_FORM`; b_ρ `SPH_SSCHEME_BRHO`), `0` (the tree level).  The positional counterterm ω
  (`SPH_POSITIONAL=1`, the default; `src/positional.jl`; Wendland C2 only).  Notes: `docs/sscheme_plan.tex`, `docs/renorm3.tex`,
  `docs/renorm2.tex`, `docs/split_dynamics.tex`.  `examples/sscheme_ref.py`, `examples/exact_count.py`: the reference values.
- **The inputs**: `src/settings.jl`, `SETTING_SPECS`, the table of every setting (`julia src/settings.jl --list`).  An unknown
  `SR_*`/`SPH_*` name stops a run (with the nearest valid name), as does a bad value; the run record lists the settings set but
  never read (`[unused]`).  s_rel,0 is `SR_SREL0`, the lab S = γ s_rel,0 (`docs/inputs.tex`).
- **The start of a run**: `SR_START = cold | hot | hmc` (default hmc) at `target_state(s_rel,0, v, eos)` (`src/measure.jl`,
  `src/run.jl`): hmc = the draw, then hybrid Monte Carlo at the requested β^μ.  hot = the draw shifted to the totals of the lattice,
  cold = the lattice.
- **A failed leapfrog step** is redone as half steps down to Δt/16 (counted: `guards().leapfrog_retries`); if it still fails the
  run stops, saving the state (`<base>_leapfrog_fail.jls`) and marking the record failed.
- `src/` — the library, loaded by `src/load.jl`: `settings.jl`, `sph.jl` (kernels, SimParams, EOS), `celllist.jl`, `ideal.jl`,
  `pointwise.jl`, `positional.jl`, `renorm_split.jl`, `sscheme*.jl`, `viscous.jl`, `setup.jl`, `rng.jl`, `stochastic.jl`,
  `measure.jl` (the start, the records, the correlators), `hmc.jl`, `monitor.jl`, `runinfo.jl` (the run record), `run.jl` (the
  pieces of a run: the fluid, the start, `RunState`, the Strang loop).
- `examples/equilibrium.jl` — the main driver (equilibrium ensembles for correlators): `RunConfig`, `RunState`, checkpoints.
  `examples/stability.jl`, `hmc_positions.jl`, `eos_measure.jl`, … — other drivers; analysis and plotting scripts next to them.
- `docs/stochastic.tex` — the stochastic step.  `docs/plan.tex`, `docs/fullruns.tex` — older plans and findings.
- `test/` — the test suite, `test/run.sh [unit|physics|load|regress|all] [-r <ref>]` (`docs/codeplan.tex`).  After any change to
  `src/`: `test/run.sh all` — unit tests pass, every script loads, and the regression against `<ref>` is bit-identical unless a change
  to the numerics is intended.  `test/regress/astcmp.jl <ref>`: the code unchanged (for changes of comments only).
- `nersc/` — run-only environment, Slurm job array, campaign files (the record of what was run: not edited), `nersc/README.md`.
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
