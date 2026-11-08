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
- `docs/stochrep.pdf` — the stochastic viscous step in the particle
  representation: a random stress per particle distributed through its faces,
  random residual tractions per bond, a consensus vote of the bonds, a
  checkerboard sweep; frozen-lattice equilibration tests at rest and moving.
- `docs/particle_metrics.pdf` — gradients and quadratic forms on general particle
  arrangements: the least-squares fit, the bond moment, and the moment-corrected
  weights used by the bond form.

These are working notes and will change.

## Layout

```
src/        library code (include in order: sph, celllist, setup, stochastic)
  sph.jl          SimParams, Particle, equation of state + Newton solve, kernels, κ tensor
  celllist.jl     cell list, smoothing, leapfrog, dissipative forces (two-level and bond), generic_step!
  changerep.jl    viscous step in the particle representation (included by celllist.jl)
  stochrep.jl     stochastic step in the particle representation (include after rng.jl)
  setup.jl        lattices (square / hex / 1D), initial conditions, plotting helpers
  stochastic.jl   noise draw, particle_sweep!, apply_sweep!
  rng.jl          per-particle random streams
test/       checks with pass/fail assertions (some also make plots)
examples/   physics validation runs (sound waves, shear waves, Gaussian pulse,
            stochastic uniform background); write PNGs next to the script
docs/       notes (LaTeX source + PDF)
```

## Running

New to the code? See [`GETTING_STARTED.md`](GETTING_STARTED.md).


Requires Julia 1.12 and a Python with matplotlib for PyPlot.

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project test/testeos.jl
julia --project test/teststochastic.jl
julia --project examples/testsound.jl
```

### Dissipative function

Three viscous steps are available, selected by `params.dissipation`:

- `:changerep` (default) — the bond form evaluated in the particle
  representation, with the gradient-corrected conversion map
  (`docs/changerep.pdf`). Each particle's own momentum sets its thermodynamic
  force, so short wavelengths are damped without the $R^2$ suppression of the
  smoothing, while the smoothed energy $H$ is conserved as well as with the
  smoothed forms.
- `:bond` — the one-level bond form on the smoothed $\beta$
  (`docs/bond_dissipation.pdf`). It has no null mode.
- `:two_level` — the SPH gradient of $\beta$ at each particle, squared
  (`docs/sph2.pdf`).

Any script can be run with another form from the environment:

```sh
SPH_DISSIPATION=two_level julia --project examples/testsound_viscous.jl
```

With its stronger damping of short wavelengths, `:changerep` needs smaller time
steps: `compute_dt(...; mode=:viscous)` bounds its fastest rate and shortens
$\Delta t$ so that every mode decays monotonically in each viscous half step of
`generic_step!` (0.67 of the previous viscous $\Delta t$ in 1D, 0.41 in 2D).

The noise is implemented only for the two-level form: `particle_sweep!` stops
with an error unless `params.dissipation = :two_level`.
The stochastic scripts (`test/teststochastic.jl`, `examples/teststochastic_run.jl`,
…) therefore select `:two_level` themselves.

Scripts use plain `include`. To edit and re-run inside one REPL session, use
[Revise](https://github.com/timholy/Revise.jl) from your global environment and
swap `include` for `includet`.

## Status

Verified:

- Kernel smoothing, cell-list neighbour search (against brute force), EOS solve, κ tensor.
- Ideal dynamics: sound-wave propagation, entropy and energy conservation.
- Deterministic viscous dynamics: shear-wave and sound-wave decay rates against
  the analytic values.
- Stochastic noise draw: covariance matches $2T\kappa^{ijlm}$ (Monte Carlo check).
- Bond dissipative function: force equals $-\partial K/\partial\beta$, exact for
  linear fields on jittered and moving lattices, no force from rigid rotations
  (`test/testdissipative.jl`).
- Equation of state: thermodynamically consistent for a moving fluid
  (`test/testeos_consistency.jl`).
- Particle-representation viscous step: rates, conversion map, gradient
  correction, conservation (`test/testchangerep.jl`).

In progress:

- Checks of the Metropolis sweep (`test/teststochastic.jl`, Step 3):
  conservation laws, visit-order independence, mean drift equal to the
  deterministic viscous force.
- Uniform-background equilibration against the fluctuation–dissipation
  prediction (`examples/teststochastic_run.jl`).
- Ensemble shear-wave and 2D sound-wave decay with noise.
- Stochastic step in the particle representation (`src/stochrep.jl`,
  `docs/stochrep.pdf`): equilibrates a frozen lattice to the exact $e^{S}$, at
  rest and moving, in 1D and 2D (`examples/stochrep_thermalize.jl`,
  `stochrep_smallk.jl`, `stochrep_moving.jl`). Not yet coupled to the
  Hamiltonian step.

## License

MIT — see [LICENSE](LICENSE).
