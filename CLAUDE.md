# Stochastic SPH — instructions for Claude

Relativistic smoothed-particle hydrodynamics (2D, density frame) with a stochastic viscous
step, in Julia.  The physics and the algorithms are in `docs/` (LaTeX; read the PDFs or
the `.tex`).  Owner: D. Teaney.

## Where things are

- `docs/plan.tex` — the current plan: speed, NERSC, measurements, next steps, status.
- `docs/fullruns.tex` — what the full Hamiltonian stochastic runs found so far.
- `docs/stochrep.tex` — the stochastic step (stress move, residual move, consensus vote,
  checkerboard sweep, moving fluid).  `docs/changerep.tex`, `docs/bond_dissipation.tex`
  — the deterministic step it builds on.
- `src/` — library, loaded by `src/load.jl`: `sph.jl` (kernels, params, EOS), `celllist.jl`, `ideal.jl`
  (Hamiltonian step), `pointwise.jl` (particle representation), `viscous.jl` (deterministic
  viscous step), `stochrep.jl`, `measure.jl`, …  Superseded code (two-level strain form, the
  old particle sweep) in `src/legacy/`.
- `examples/stochrep_hamiltonian.jl` — energy/stability runs (uniform fluid, Hamiltonian
  step on).  `examples/stochrep_equilibrium.jl` — equilibrium ensembles for correlators.
  Plotting scripts next to them.
- `test/` — checks; run the relevant ones after any change to `src/`.
- `nersc/` — run-only environment, Slurm job array, campaign files, `nersc/README.md`.
- **`nersc/LOG.md` — where things stand at NERSC: setup, lessons, campaigns and their job
  IDs.  Read it first; add to it when you submit or learn something.**

## Conventions (important)

- Temperature: bare `T` = ∂H/∂s = T_rel/γ (`pa.T`); `T_rel = γT` (`pa.T * pa.γ`) appears
  only with the viscous coefficients (T_rel κ, T_rel η, T_rel ζ).
- `M = T^{0i}` is the **density-frame** momentum density — never call it "Landau".
- Notes: brief and pedagogical, displayed equations rather than inline ones, symbols
  defined, no material that was not asked for.
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
