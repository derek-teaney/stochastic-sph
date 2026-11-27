# stochastic-sph

Smoothed-particle hydrodynamics (SPH) for **relativistic stochastic viscous
hydrodynamics in the density frame**, written in Julia.

Each SPH particle carries an extensive entropy $s_a$ and momentum $m_a$.
The ideal (Hamiltonian) part of the evolution is integrated with an
entropy-conserving symplectic leapfrog. Dissipation and thermal noise enter
through a per-particle Metropolis sweep: each particle draws a Gaussian stress
$\xi_a$ with the density-frame fluctuation–dissipation covariance
$\langle \xi^{ij}\xi^{lm}\rangle = 2T\kappa^{ijlm}$, distributes it as
momentum- and energy-conserving kicks over its neighbours, and is accepted with
probability $\min(1, e^{\Delta S_a})$. The mean drift of the sweep reproduces
the deterministic two-level SPH viscous force, so there is no free noise
amplitude.

The physics and the algorithms are described in [`docs/`](docs):

- `docs/sph2.pdf` — Hamiltonian/GENERIC formulation, kernel discretization,
  the $\kappa^{ijlm}$ tensor, the two-level dissipative force, noise generation.
- `docs/firstrun.pdf` — the stochastic step: per-particle Metropolis sweep,
  its mean drift, known biases, and diagnostics.
- `docs/bond_dissipation.pdf` — the one-level (bond) dissipative function: bond
  differences of $\beta$ with a mean-field rotation and dilation removed, at
  rest and in a moving fluid, and how the code computes its force.
- `docs/changerep.pdf` — the viscous step in a particle representation: each
  particle's own energy and thermodynamic force, the gradient-corrected conversion
  map, and its energy bookkeeping and convergence.
- `docs/stochastic.pdf` — the stochastic viscous step in the particle
  representation: a random stress per particle distributed through its faces,
  random residual tractions per bond, a consensus vote of the bonds, a
  checkerboard sweep; frozen-lattice equilibration tests at rest and moving.
- `docs/particle_metrics.pdf` — gradients and quadratic forms on general particle
  arrangements: the least-squares fit, the bond moment, and the moment-corrected
  weights used by the bond form.

These are working notes and will change.

## Layout

```
src/        library code; scripts include src/load.jl, which loads it in order
  settings.jl     the table of every setting (julia src/settings.jl --list) and setting(name)
  sph.jl          SimParams, Particle, equation of state + Newton solve, kernels
  celllist.jl     cell list and neighbour list
  ideal.jl        the ideal (Hamiltonian) step: smoothing, velocities, force, compute_dt, leapfrog
  pointwise.jl    the particle maps: particle energies and the inverse map
  positional.jl   the positional counterterm ω
  renorm_split.jl, sscheme.jl, sscheme_exact.jl   the renormalization (split, the S-scheme)
  viscous.jl      the deterministic viscous step: the inverse-moment bond dissipation, viscous_step!, generic_step!
  setup.jl        the hexagonal lattice, initial conditions
  rng.jl          per-particle random streams
  stochastic.jl   the stochastic step (stochastic_step!)
  measure.jl      the target state and the start, Fourier modes, correlators, checkpoints, HDF5 output
  hmc.jl          hybrid Monte Carlo and the generalized leapfrog (the ideal step)
  monitor.jl      running statistics printed to the log
  runinfo.jl      the run record <run>.toml (every setting as used, provenance, outcome)
  run.jl          the pieces of a run: the fluid, the start, RunState (the checkpoint), the Strang loop
test/       checks with pass/fail assertions (some also make plots)
examples/   physics validation runs (sound waves, shear waves, Gaussian pulse,
            stochastic uniform background); write PNGs next to the script
            (scripts behind the older notes: git tag notes-2026-11, docs/codeplan.tex)
docs/       notes (LaTeX source + PDF)
```

## Running

New to the code? See [`GETTING_STARTED.md`](GETTING_STARTED.md).


Requires Julia 1.12 and a Python with matplotlib for PyPlot.

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project test/testeos.jl
julia --project test/teststochastic_guard.jl
julia --project examples/testsound.jl
```

### The scheme

One scheme (since 2026-11-27; `docs/focus_plan.pdf`): the energy Hamiltonian with the number volume
$V_a=1/n_a$, the generalized leapfrog, the inverse-moment viscous step with the Hastings correction of the
stochastic step, the Liouville measure, two dimensions.  The renormalization is `SPH_RENORM` = `split`
(default), `sscheme` or `0` (the tree level).  The options removed then (the smoothed Hamiltonian, the other
dissipations, the drift–kick–drift and grid steps, the Einstein measure, 1D, …) are in the git tag
`pre-prune`; the settings that chose them stop with a message.

### Inputs

Every setting is in the table of `src/settings.jl` (`julia src/settings.jl --list`).  A name not in it, or a
bad value, stops the run.

### Initial state of the stochastic runs

`examples/equilibrium.jl` takes `SR_START`, at the requested state (`target_state`, `src/measure.jl`):

- `hmc` (default) — energies and momenta drawn at the requested $\beta^\mu$, then hybrid Monte Carlo of
  the positions and momenta (`docs/equilibrium_positions.pdf`).
- `hot` — the lattice, energies and momenta drawn at the requested $\beta^\mu$ and shifted to the totals of
  the lattice (`docs/eqstart.pdf`); the default of `examples/stability.jl`.
- `cold` — the lattice as set up; the run thermalizes from it.

### Dissipative function

The inverse-moment bond form (`docs/inverse_moment.pdf`, `src/viscous.jl`): the resolved part of the
dissipation through the least-squares gradient of $\beta$, the residual on the bonds.  `compute_dt(...;
mode=:viscous)` bounds its fastest rate so that every mode decays monotonically in each viscous half step of
`generic_step!`.  The noise is the stochastic step `stochastic_step!` (`src/stochastic.jl`,
`docs/stochastic.pdf`).

Scripts use plain `include`. To edit and re-run inside one REPL session, use
[Revise](https://github.com/timholy/Revise.jl) from your global environment and
swap `include` for `includet`.

## Status

Verified:

- Kernel smoothing, cell-list neighbour search (against brute force), EOS solve.
- Ideal dynamics: sound-wave propagation, entropy and energy conservation.
- Deterministic viscous dynamics: shear-wave and sound-wave decay rates against
  the analytic values.
- Stochastic noise draw: covariance matches $2T\kappa^{ijlm}$ (Monte Carlo check).
- Inverse-moment dissipative function: force equals $-\partial K/\partial\beta$, the continuum $K$ of
  linear fields on jittered and moving lattices, no force from rigid rotations
  (`test/testdissipative.jl`, `test/testmoment.jl`).
- Equation of state: thermodynamically consistent for a moving fluid
  (`test/testeos_consistency.jl`).
- The start (`test/teststart.jl`), checkpoints (`test/testcheckpoint.jl`), a failed leapfrog step
  (`test/testleapfrog_fail.jl`), the renormalization (`test/testsplit.jl`, `test/testsscheme*.jl`).

In progress:

- The stochastic step coupled to the Hamiltonian step (`src/stochastic.jl`,
  `docs/stochastic.pdf`, `docs/fullruns.pdf`, `docs/continuum_limit.pdf`,
  `docs/boost.pdf`); the runs at NERSC are logged in `nersc/LOG.md`.
- The plan for names, defaults, run records and tests: `docs/codeplan.pdf`.

## License

MIT — see [LICENSE](LICENSE).
