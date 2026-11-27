# Code review notes (2026-11-27)

Notes from reading the code with D. Teaney on branch `sscheme` (merged into `main` as e3333fe the same day).
Nothing here has been changed in the code.  **The decisions are collected into the staged plan
`docs/focus_plan.tex`, which is what the NERSC session follows;** this file is the discussion behind it.

## Overall

- The library (`src/`) is reasonably layered: particle maps (`pointwise.jl`), ideal step (`ideal.jl`, `hmc.jl`),
  stochastic step (`stochastic.jl`), renormalization (`renorm_split.jl`, `sscheme*.jl`), measurement (`measure.jl`).
- The drivers (`examples/`) are where the disorder is.  `equilibrium.jl` grew by accretion: careful bookkeeping
  (run record, checkpoints, bit-identical restart) but hard to read, and not reusable by other drivers.
- `hmc.jl` is a good example of the intended style: short, one job, comments match the equations.

## The driver `examples/equilibrium.jl`

- Lines 1–154 (the inputs) mix physics, numerics, start options, test switches and output, and carry dated
  history in the comments.  The physical system is set by lines 18–19, 22–23, 27–33, 80, 89.
- `one_realization` is ~190 lines: setup, HMC start, loop, measurement, checkpoint, output, with special cases
  (`GRID`, `NOHAM`, `NOVISC`, `JITTER`, `SSCHEME`) spread through it.
- `st` is an untyped NamedTuple of ~45 fields (the run state, serialized as the checkpoint), patched for older
  checkpoints at lines 241–258.
- `SR_CS2 = 0` is a placeholder meaning conformal (c_s² = 1/d); the record shows 0.0 in `[settings]`, the real
  value only in `[eos] cs2_val`.
- Other drivers duplicate its pieces: the HMC equilibration loop is written five times (`equilibrium.jl`,
  `eos_measure.jl`, `hmc_positions.jl`, `moment_fluctuations.jl`, `volume_fluctuations.jl`); h is fixed at 0.05
  in `stability.jl` and `hmc_positions.jl` (configurable via `SR_H` only in `equilibrium.jl`); only
  `equilibrium.jl` checkpoints.
- `nersc/campaign_example.txt` points to `examples/stochrep_*.jl`, which no longer exist.

Candidate restructuring (to decide; on its own branch, checked by the bit-identical regression):
1. `RunConfig`: all inputs read into one struct, grouped, printed at the start.
2. `RunState`: a declared struct in place of `st`.
3. Split `one_realization` into setup / equilibrate / evolve / finish; move the reusable parts into `src/`
   (`setup_fluid`, `equilibrate!` with the HMC loop, `evolve!`, a checkpointer).
4. Which old options to retire from the main driver (jitter, Poisson, `SR_START=thermal|cold`,
   `SR_BATH=totals`, the grid step) is the owner's call: it decides which old runs stay reproducible.

## Settings (`src/settings.jl`)

- The file holds only the mechanism (`setting`, the registry, retired names), not the settings: names and
  defaults are spread over `sph.jl`, `stochastic.jl`, `hmc.jl`, `renorm_split.jl`, `sscheme*.jl`,
  `positional.jl` and each driver.  ~43 are read by an equilibrium run, 51 distinct across all drivers.
- `SR_` comes from "stochrep" (the old name of the stochastic step and its drivers, renamed in codeplan.tex).
- Weak points: a misspelled name (`SR_ETA` for `SR_ETAS`) is silently ignored; a Bool that is not
  1/true/yes is silently false; Symbols are not validated here; no list of all settings and defaults exists
  until a run has read them.
- Candidate: a central table `SETTING_SPECS` (name, default, meaning) in `settings.jl`; stop on any `SR_*`/`SPH_*`
  in the environment that is not in it; strict Bool parsing; a `--list` to print defaults.
- Laptop practice: the prefix form `NAME=value julia …`, no `export`; check `env | grep -E '^(SR|SPH)_'`.

## Seeds

- Deterministic, never from the clock: `s = SR_SEEDOFFSET + SR_SEED0·SR_SEEDSTRIDE + i`, streams `1000s + k`.
  At NERSC `array.sh` sets `SR_SEED0 = seed − 1` (overriding any value in the campaign line).
- About half the campaigns use no `SR_SEEDOFFSET`, so their lines share seeds 1…n.  The scheme comparisons
  (`hastings_split`, `hastings_compare*`) tie the offset to the physical state (500/600/700/800), so tree, split
  and S-scheme at one state share seeds: fine for comparisons (common random numbers), not for pooling.
- To pool: give the new realizations a fresh offset block (and keep a list of used blocks in `nersc/LOG.md`).

## Change of representation: obsolete

The owner: "we aren't changing rep any more."  With the energy Hamiltonian (H = Σ_a e_a, V_a = 1/n_a) the particle
variables are the native ones; there is no map between smoothed and particle variables (docs/changerep.tex).  Code
that belongs to the change of representation, all legacy:

- `src/stochastic.jl`: `gradient_freeze`, `gradient_backmap!`, the `gradient` field of `StochOptions` (an error with the
  energy Hamiltonian), `GUARD_BACKMAP`; the comments on the step that still describe "the gradient-corrected change
  of representation".
- `src/pointwise.jl`: the gradient-corrected conversion map (`kernel_star!`, `kernel_star`, `star_fit`,
  `inverse_susceptibility`, `gradient_energy`, lines 135–196) and `pointwise_state`/`pointwise_beta!`; its header
  comment still frames the file as "the particle representation (changerep.tex)".  The two particle maps
  (`particle_forward`, `particle_inverse`) in the same file are current.
- `src/viscous.jl`: the deterministic viscous steps of the change of representation (`viscous_step_first!`,
  `viscous_step_exact!`, `viscous_step_grad!`, lines 403–496) — to check against `viscous_step_energy!`, which is
  the energy-Hamiltonian path.
- `Particle` fields used only by the smoothed Hamiltonian or the old sweep (`M_smooth` as a smoothed field,
  `Teff`, `F_over_S`, `Δs`, `Δm`).

## The measure: Liouville only

The owner: "we are only using Liouville from now on; drop Einstein and other choices of measure."  The weight is
S_a − ln T_a everywhere (flat in the entropy labels, the measure the Hamiltonian step preserves).  Where the choice
lives now (to remove, not yet done):

- `src/sph.jl`: the `measure` field of `SimParams` (line 33), its comment (lines 72–78), `default_measure` (line 123),
  the checks and message (lines 189–192), `SPH_MEASURE`.
- `src/stochastic.jl`: the `liouville::Bool` argument threaded through `vote_entropy`, `ask_star!`, `color_sweep!`,
  `ask_star_hastings!`, `color_sweep_hastings!`; the Einstein branch of `vote_entropy` (`particle_entropy` alone,
  line 226); the end-of-step branch `pa.s = liouville ? … : Sc[a]` (line 671); the check at line 632.
- `src/hmc.jl`: `liouville` in `hmc_heatbath!` (line 39); the `Σ ln T` term of the acceptance (line 136); the header
  comment on π_E.
- `src/measure.jl`: `liouville::Bool = false` keyword in `draw_particle`, `particle_moments`, `moments_table`,
  `beta_from_averages`, `draw_configuration`.  **Note: these default to Einstein.**  `equilibrium_init!` passes the
  measure explicitly, but `HMCTarget` (hmc.jl line 27) calls `particle_moments(Vref, β0, β, eos)` without it, so the
  HMC heat-bath proposal is shaped by the Einstein weight.  That affects only the proposal's efficiency, not the
  target, but it is a leftover.
- `src/renorm_split.jl` (line 194) and `src/sscheme.jl` (line 306): the `liouville ? … : s` branch and the
  "needs the Liouville measure" checks become unnecessary.
- Consequences elsewhere: with Liouville only, `resid_deterministic!` (λ = ∞) and the gradient back map are excluded
  anyway (stochastic.jl line 632), so they go too.  Campaign lines with `SPH_MEASURE=einstein` and tests that set it
  need a decision (retire the setting with a hard stop, as for `SR_S0`).

## The grid step: probably going

The owner: "the SPH grid will probably go" (tentative).  `SPH_IDEAL=grid` (docs/gridstep.tex), the ideal step on the
fixed hexagonal lattice.  Its footprint:

- `src/grid.jl` (373 lines: the step, its lattice FFT `GridFFT`, `GridStencil`, `GridStats`, the bare propagator).
- References in `src/sph.jl` (the `:grid` option and its checks), `src/measure.jl` (`record_grid!`), `src/pointwise.jl`
  (`grid_forward`/`grid_inverse` mentioned as field-level EOS uses), `src/renorm_split.jl`, `src/sscheme.jl` (the "not with
  the grid step" checks).
- `examples/equilibrium.jl`: ~44 lines (`GRID`, `grid_step!`, `gstencil`, `gstats`, `gCk`, `grid_series`, the grid
  summary, the `SR_POSITIONS` default that depends on it).
- `examples/grid_*.jl` (8 scripts), `test/testgridfft.jl`, `nersc/campaign_grid*.txt` (3 campaigns).
- If it goes: also the `SR_GRID` retired-name stop in the driver.

## The ideal step on the default path

Energy Hamiltonian, number volume, generalized leapfrog.  One right-hand-side evaluation: `smooth_entropy_energy!`
(n_a, V_a = 1/n_a) → `update_thermodynamics!`/`solve_eos!` (`particle_forward`, where the renormalization enters) →
`hamiltonian_force_energy!` (−Σ_b (p_a V_a² + p_b V_b²) ∇W_ab) or `update_velocities_energy!` (ẋ = v).  The integrator
is `generalized_leapfrog!` (hmc.jl), entered through `leapfrog_step!` → `generalized_step!`.

**The owner: "we will drop everything but the n-scheme."**  Only the energy Hamiltonian with the number volume
(H = Σ_a e_a(s_a, m_a, V_a), V_a = 1/n_a; docs/volume_n.tex) stays.  To go:

- the smoothed Hamiltonian (`SPH_HAMILTONIAN=smoothed`): the bodies of `smooth_to_particles!`, `update_velocities!`,
  `hamiltonian_force!` in `ideal.jl`, and the `hamiltonian` field of `SimParams`;
- the entropy volume (`SPH_VOLUME=entropy`): the `number = false` branches in `ideal.jl` and the `volume` field;
- what only made sense with them: the dissipation forms tied to the smoothed fields (`:two_level`, already in
  `src/legacy/`; `:bond` on the smoothed β), the change of representation (above), `Particle` fields `M_smooth` as a
  smoothed field, `Teff`, `F_over_S`, `dotx`.
- **The DKD leapfrog goes** (owner, 2026-11-27: replaced by the generalized leapfrog).  `SPH_IDEAL=particle`, the DKD
  body of `leapfrog_step!` (ideal.jl lines 233–266), the `ideal` field of `SimParams` and `default_ideal` (with the grid
  step also going, `:generalized` is the only ideal step; `leapfrog_step!` can call `generalized_step!` directly).
  No script, test or campaign file sets `SPH_IDEAL=particle` explicitly.  Consequence for 1D: `default_ideal` gives DKD
  in 1D today, so the 1D test must run with the generalized leapfrog when it is put back (nothing in
  `generalized_leapfrog!` is 2D-specific; to verify).  Also the remaining dissipation choices
  (`:particle`, `:common`; `:moment` is the default and needs the energy Hamiltonian) and the 1D path
  (`WendlandC2_1D`, `sim_dim = 1`, deterministic only; stochastic runs are 2D).
- Simplification that follows: with V_a = 1/n_a stored directly, the `S_smooth = s_a n_a` round trip (kept for
  bit-identity) can go, and `solve_eos!` can take (s_a, m_a, V_a) directly.

**1D: keep** (owner: deterministic, useful for checking).  Its own code is ~60 lines in `src/` (`WendlandC2_1D` in
`sph.jl`, the `sim_dim == 1` branch of `SimParams`, `make_1d_particles!` in `setup.jl`, `bond_weights_1d` and the `oneD`
switch in `viscous.jl`, the 1D constants in `viscous_rate_bound`); drivers and tests ~730 lines
(`testsound_viscous_1d.jl`, `testgaussian_1d.jl`, `stochastic_thermalize.jl`, `test/testdims.jl`; `test/testchangerep.jl`
goes with the change of representation).  Dependencies: the 1D drivers use `generic_step!` → `viscous_step_energy!` →
`bond_dissipation` with the `:particle` weights, so `:particle` stays as the deterministic viscous step (at least for 1D);
the 1D default integrator is DKD (`default_ideal` picks the generalized leapfrog only in 2D), but `generalized_leapfrog!`
and the `SimParams` check have no 2D restriction: test the generalized leapfrog in 1D before dropping DKD.

**Dissipation: `:moment` is the only option** (owner, 2026-11-27).  `:particle`, `:common`, `:bond`, `:two_level` go.
1D does not need `:particle`: in `_bond_dissipation!` (viscous.jl) 1D takes its own `oneD` branch whatever the label —
`bond_weights_1d`, all of the dissipation on the bonds with the closed-form T κ^{xxxx}(v), no gradient fit, no faces, no
Λ — so the label only selects the 2D path.  What stops `:moment` in 1D today is the check in `SimParams` (sph.jl line
167: `:moment` needs `sim_dim == 2`) and `default_dissipation` (`:particle` in 1D).  The 2D pieces of `:moment` (the
inverse moment 𝓜_a⁻¹, the faces, two shear components + bulk, Hastings) have no role in deterministic 1D (a 1D star is
collinear: 𝓜_a would be singular in 2×2; its 1D analogue is the scalar Σ w x²).  The 1D bond weight
ψ(ρ) ∝ (1 − ρ/2h_ψ)³ has the same shape as the 1D kernel's −W′(r)/r (Wendland C2: W′ ∝ q(1 − q/2)³), so with
`bond_h = h` the 1D weights already equal the common-kernel weights of `:moment`, except that they are evaluated at the
comoving distance √(rᵀℓ⁻¹r) instead of the lab distance (the same at rest).  So: allow `:moment` in 1D (relax the check,
drop `:particle` from the defaults), keep the 1D branch; a 1D test (testsound_viscous_1d) should be unchanged at rest.

**Plan (owner):** first switch to `:moment` as the only dissipation (2D), then, once that is in, put the 1D
deterministic test back on top of it (relax the 2D check, run the 1D sound and Gaussian tests with `:moment`).

Former wording (before the decision): older options still in `ideal.jl`: the smoothed Hamiltonian, the entropy
volume, the DKD leapfrog.
The default path enters through the old names, which dispatch to the `*_energy!` versions; `generalized_step!` ignores
the −1 (no Hamiltonian) return of `generalized_leapfrog!`.

## Other optional pathways (survey, 2026-11-27)

Scope the owner set: deterministic runs in 1D and 2D, stochastic runs in 2D; n-scheme, `:moment`, generalized leapfrog,
Liouville.  Pathways beyond those already listed above, with how often scripts, tests and campaign files use them:

**Probably dead (no users, or excluded by the decisions above)**
- Stochastic step without Hastings (`SPH_HASTINGS=0`): `propose_stress!`, `propose_resid!`, `propose_resid_moment!`,
  `ask_star!`, `color_sweep!` (~150 lines).  No campaign sets it; with `:moment` only it is reachable only by turning
  Hastings off.  With it go the `:bonds`/`:even` votes (1 example, 1 test) and `resid_deterministic!` (λ = ∞, already
  excluded by Liouville; `SR_LAMBDA`).
- `StochOptions` fields `residual` and `shift`: never set to false anywhere.
- `Star`'s frozen coefficients (T, κ, B, ℓ): with Hastings recomputed by `star_coef` at each proposal.
- `bond_h` (separate bond length): never set; with `:moment` the 2D weights use the kernel's own h.
- `make_uniform_particles` (square lattice; one test), `smooth_to_grid` (no users), `energy_pressure`.
- `Particle` fields `Δs`, `Δm` (the old sweep), the global `RNG = MersenneTwister(0)`.
- Generic `AbstractEOS` code paths (`rest_energy` fixed point, the `try` in `target_inverse`, generic `eos_newton`):
  `ConstantCs2EOS` is the only equation of state.
- `src/legacy/` (503 lines: `kappa.jl`, `particle_sweep.jl`), `examples/legacy/` (Python, reads CSVs).
- `SR_JITTER` (no campaign).

**A decision for the owner**
- Renormalization: `SPH_RENORM=0` (tree; 4 campaigns, the regression), `split` (the default; `renorm_split.jl`, 209 lines),
  `sscheme` (meant to replace split).  Once the S-scheme is the default, split goes; the tree level is worth keeping
  as the reference.  Within the S-scheme: `SPH_SSCHEME_FORM` (measure / exact, `sscheme_exact.jl`) and
  `SPH_POSITIONAL` (on by default; Wendland C2 only).  **Owner (2026-11-27): keep split for now** (with the tree level
  and the S-scheme); revisit once the S-scheme is settled.
- Starts in `equilibrium.jl`: `SR_START=thermal|cold` (`thermal_init!`; 4 old campaigns), `SR_BATH=totals`
  (`beta_from_averages`, `moments_table`; the regression against `main` uses it), `SR_POSITIONS=lattice|poisson`
  (lattice: 2 campaigns, used for large s_a and the grid; poisson: 1).
- Kernels Wendland C4/C6 (`SPH_KERNEL`; 6 campaigns of the kernel study); the positional counterterm needs C2.
- `phys_dim = 3` with `sim_dim = 2` (deterministic only; 2 examples, 4 tests).
- Frozen lattice (`frozen = stochastic_frozen(…)`; 3 examples: the equilibration tests of the stochastic step).

**Owner's decisions on this list (2026-11-27)**
- Keep the `SPH_HASTINGS` flag, hence the non-Hastings path (`propose_stress!`, `propose_resid!`,
  `propose_resid_moment!`, `ask_star!`, `color_sweep!`) and `Star`'s stored coefficients (needed without Hastings).
- Drop `SR_LAMBDA` and its path: `resid_deterministic!`, `moment_fit`, `GUARD_SUBDIV`, the λ argument and the
  λ = Inf branches (λ = 1 always).
- Drop `StochOptions.residual` and `.shift` (always true).
- Drop `bond_h` (field of `SimParams`, `bond_weight(ρ, hψ)` uses; the 1D weights then use h).
- Drop `make_uniform_particles` and `smooth_to_grid`.
- Keep the generic equation of state and `AbstractEOS`.  Keep `SR_JITTER`.
- The global `RNG` (sph.jl line 6) may go: its only uses are the jitter in `setup.jl` (seeded from `params.seed`) and
  `test/testneighbor.jl`; replace by a local `Xoshiro(params.seed)` (changes the jittered positions: not bit-identical).
- Do not change the `Particle` struct now.
- `energy_pressure` (measure.jl line 659) is NOT unused: `examples/eos_measure.jl` uses it to measure the energy
  density and the mechanical pressure (kinetic m·v part + virial P V² Σ r D part, from docs/gravity.tex) against the
  target EOS.  Keep.  (Listed above by mistake.)
- Drop the `:bonds` and `:even` votes; keep only `:star` (owner: no benefit from voting in parts, and the acceptance is
  high).  Goes: the `vote` argument and field of `StochOptions`, the σ_ab shares and the even split in `ask_star!`,
  `GUARD_EVENSPLIT`; the uses in `examples/stochastic_moving.jl` and `test/teststochastic_guard.jl`.  The owner may come
  back to `:bonds` someday: the last commit with it is the one before the removal (now e3333fe, 2026-11-27); record the
  commit in the removal's message and in `docs/stochastic.tex`, where the bond vote is described.
- Open: "we can remove" after `bond_h` — unclear what it refers to (`src/legacy/`?).

**Test aids worth keeping** (cheap, used): `SR_NOVISC`/`SR_NOHAM` (12 campaigns: rigidity tests), `SR_DTFAC`,
`SR_RECORD_E`, `SR_CS2`/`SR_ZETAS`, the 1D path.

**Analysis code in the library**: `shell_correlators`, `oscillator_corr`, `fit_oscillator`, `autocorr`, `binned_mean`,
`read_equilibrium` in `measure.jl` (used by analysis scripts; could move to an analysis file).

## Later decisions (2026-11-27): no 1D, no jitter

**The 1D path goes entirely** (owner; replaces "1D: keep" and the plan to put the 1D test back on `:moment`, above).
The 2D sound tests (`examples/testsound.jl`, `examples/testsound_viscous.jl`) remain for checking.  Goes:
`WendlandC2_1D` (kernel W, ∇W), the `sim_dim == 1` branch of `SimParams` and the `sim_dim` keyword, `sim_dim()` and
the 1D defaults in `default_dissipation`/`default_ideal`/`default_measure`, `make_1d_particles!` and its branch in
`setup_ic`, `bond_weights_1d` and the `oneD` branch of `_bond_dissipation!`, the 1D constants of `viscous_rate_bound`,
`neighbor_range` for one row (celllist.jl), the 1D drivers (`testsound_viscous_1d.jl`, `testgaussian_1d.jl`, the 1D
branch of `stochastic_thermalize.jl`), the 1D part of `test/testdims.jl` and of `check_stochastic_dims`.  With it the
question of the generalized leapfrog in 1D disappears.  `phys_dim` vs `sim_dim`: with sim_dim always 2, `phys_dim = 3`
(deterministic only) is the remaining question.

**Jitter stays after all** (owner, later still): keep the `jitter` option, but give it its own random number generator,
a local `Xoshiro(params.seed)` (or a seed passed in) inside `make_hex_particles!`, so the global `RNG` in sph.jl can
go.  Jittered positions then change (not bit-identical with old runs or tests that use jitter).  The tests below keep
working unchanged.  The paragraph that follows is superseded and kept only as the record:

~~**The jitter option goes** (owner; replaces "keep SR_JITTER").~~  Goes: `SR_JITTER` and its rescaling in `equilibrium.jl`
(the rescaling block is shared with `SR_POSITIONS=poisson`: keep it for Poisson if Poisson stays), the `jitter` keyword
of `setup_ic`/`make_hex_particles!`, and then the global `RNG` (its only use was the jitter).
**Caution:** several tests use jitter to make disordered stars: `test/testmoment.jl` (closure Σ_b A_ab r_abᵀ = V_a 𝟙 and
zero virial on jittered and Poisson stars), `test/testhastings.jl` (`star_coef` against `build_stars` on a moving
jittered lattice), `test/testdissipative.jl`, `test/testcommonkernel.jl`; also `examples/moment_det.jl`,
`hastings_baseline.jl`, `testdiffuse.jl`, `ideal_lattice_modes.jl`.  Those tests check exactness on irregular
stars, which matters because HMC positions are irregular.  They need another source of disorder: Poisson positions,
HMC positions, or a small displacement done inside the test itself.

## Next steps the owner wants (2026-11-27)

1. **A pass over the comments.**  Many comment lines are very long and hard to read.  Style: short lines, one point per
   paragraph, a blank comment line (`#` alone) between logically distinct points.  Dated history ("before 2026-11-15 …")
   can go into the notes or the commit log rather than the code once the options are removed.
2. **Restructure `examples/equilibrium.jl`**, which has become the main driver: break `one_realization` (~190 lines)
   into logical parts (setup, equilibrium start, evolution, recording, checkpoint/output), with the inputs gathered in
   one place (see "Candidate restructuring" above).

## `SR_BATH`

The equilibrium start draws each particle's (e_a, m_a) at multipliers β⁰ = γ/T_rel, β = γv/T_rel.  `SR_BATH` says where
they come from:
- `nominal` (default since 2026-11-23): the requested state, T_rel,0 from the EOS at s_rel,0 and v = `SR_BOOST`; the run
  is then at that (T_rel, v) for any Hamiltonian.  It also sets the w, T used to normalize KE and written in the record.
- `totals` (before): β^μ solved so that the expected total energy and momentum equal those of the set-up configuration
  (`beta_from_averages`, `moments_table`, a quadrature over single-particle states).  With a renormalized Hamiltonian
  the set-up state's energy is not the target's, and the solve can be far off (split, s_a = 16, v = 0.8: T_rel = 208,
  v = 0.66; campaign `hastings_split`).
- Used by: the regression against `main` (`SPH_RENORM=0 SR_BATH=totals`), one example, one test.  Once the regression
  is rebased on the tag before the removal, `totals` (and `beta_from_averages`, `moments_table` if unused elsewhere) can go.

**Decision (owner, 2026-11-27): remove `SR_BATH=totals`.**  Goes: `SR_BATH` in `equilibrium.jl` and
`examples/hmc_positions.jl` (the bath is always the requested state), the `bath === nothing` branch of
`equilibrium_init!` and `initial_state!` (make `bath` a required argument), `beta_from_averages` (measure.jl line 145;
also used by `test/testeqstart.jl` and mentioned in `examples/cl_exact_m2.jl`), the `SR_BATH=totals` of the regression
(`test/run.sh` line 91, rebased on the tag), the "set-up particle" values of w, T in the driver (lines 170, 196).
Stays: `moments_table` (the nominal draw uses it), `draw_configuration`.
**Not the same thing:** `exact_totals` (the `:eq` start shifts the drawn energies and momenta so their totals equal
given (E, M)) is separate from `SR_BATH`.  By default (E, M) are those of the current state.  After the HMC start that
state is itself an equilibrium sample at the nominal β^μ, so the shift is harmless.  With `SR_POSITIONS=lattice` (no
HMC), (E, M) are those of the set-up lattice, which with a renormalized Hamiltonian (split, S-scheme) are not the
totals of the nominal state: the start is then drawn at the nominal β^μ but shifted to the set-up totals, a mismatch
of the same kind `totals` had.  To decide: shift to the expected totals at the nominal β^μ (Σ_a of the moments), or
drop the shift.  Also: `README.md` still describes `SR_START=eq` as "the global β^μ whose averages are the totals".

## Plan for the start of a run (2026-11-27, to discuss)

`SR_BATH` did three jobs in `equilibrium.jl`, all one question ("what state is the run at?"): (1) w₀ for the flow
momentum M₀ = w₀γ²v (line 174), (2) the multipliers β⁰, β of the draw and of HMC (line 202), (3) the w, T, T_rel that
normalize KE and go in the record (lines 197–200).  `exact_totals` is a separate question (the ensemble across
realizations).  The start is now spread over `SR_START` × `SR_POSITIONS` × `SR_BATH`.

A. `target_state(srel0, v, eos)` → (T_rel,0, v, γ, e₀, w₀, S = γ s_rel,0, M₀, β⁰, β), computed once from the EOS; setup,
   draw, HMC, KE normalization and record all read it.  `SR_BATH` and `beta_from_averages` go.
B. One flag `SR_START = cold | hot | hmc` (default `hmc`):
   - `cold`: lattice, the set-up state, no fluctuations; thermalizes; at the set-up totals (T differs from the request
     at O(1/s_a): measure it; the record says so).  Demonstration.
   - `hot`: lattice, (e_a, m_a) drawn at the target β^μ; positions relax in the run.  Demonstration.
   - `hmc`: draw, then HMC at the target β^μ (positions and, by the heat bath, (e_a, m_a)).  Production.  The extra
     redraw after HMC (`initial_state!` :eq) is redundant and goes.
   `SR_START=thermal` (`thermal_init!`) and `SR_POSITIONS` go.
C. Totals (owner, 2026-11-27):
   - `hot` fixes the total energy and momentum, so that it can be compared with `cold`: the drawn configuration is
     shifted to the totals of the set-up state, the same (E, P) the cold start has.  Cold and hot then sit on the same
     microcanonical shell and must reach the same equilibrium.  (Both are then at the set-up totals, not exactly at the
     requested state: T differs at O(1/s_a) with a renormalized Hamiltonian; fine for the demonstration.)
   - `hmc`: no shift by default.  A debugging flag `SR_HMC_SHIFT_TOTALS` (Bool, default false) shifts the HMC sample to
     the same totals as `hot`, for comparisons with `cold`/`hot`.  Read only when `SR_START=hmc`.
   Implementation: compute the cold totals (E_cold, P_cold) = (Σ_a e_a, Σ_a m_a) of the set-up lattice once, before any
   draw or HMC update, and keep them (in the run state and the record).  `hot` shifts to them; `hmc` with
   `SR_HMC_SHIFT_TOTALS=true` shifts its final sample to them (its positions have moved, but the target totals are still
   the cold ones).  Plan accepted by the owner (2026-11-27).
D. Open: Poisson positions and jitter as modifiers of `cold`/`hot` only.

## Before implementation (2026-11-27)

Non-bit-identical changes: (1) the jitter RNG (jittered runs only); (2) the default `hmc` start drops the redraw after
HMC (`initial_state!` :eq with the shift to the post-HMC totals): every production run starts from a different,
statistically equivalent state — its own stage; (`hot` = today's `SR_POSITIONS=lattice SR_START=eq`: should be
bit-identical); (3) optional: dropping the `S_smooth = s_a n_a` round trip; (4) risk: refactors can change muladd fusion
(`@noinline` functions kept for bit-identity), the regression decides; (5) the `generalized_step!` failure fix (only
where a step fails).
Compatibility: old `.chk` checkpoints will not resume (do not deploy mid-campaign); record keys read by
`nersc/collect_runs.jl` and `examples/boost_analysis.jl`; regression rebased on a `pre-prune` tag first (defaults and
tree level), `test/run.sh` rewritten; hard stops for every removed setting; coordinate with the S-scheme work (same
files); tests in the `debug` queue.
Open: Poisson, C4/C6, `phys_dim = 3`, frozen lattice, `src/legacy/`, stop vs warn for unread settings.

## Decisions on the open items (owner, 2026-11-27)

1. **Poisson positions: remove** from the driver (`SR_POSITIONS=poisson`, its rescaling block shared with jitter stays
   for jitter).  `test/testmoment.jl` keeps its own Poisson stars (built inside the test) for the exactness checks.
2. **Kernels C4/C6: keep.**  Currently undeveloped (the positional counterterm is tabulated for C2 only and refuses
   them); the owner wants to bring them alive after the restructure (tabulate the positional counterterm for C4/C6).
3. **`phys_dim`: keep**, default 2.  Deterministic runs with `phys_dim = 3` (a z-independent 3D fluid in the plane)
   make sense; stochastic runs keep requiring d = 2.  The tests that exercise d = 3 and are not otherwise removed
   (`testdims.jl`, `testdissipative.jl` if its scheme survives) stay or move to the `:moment` path.
4. **Frozen lattice: keep** (the cleanest unit test of the stochastic step alone; `stochastic_thermalize.jl`,
   `stochastic_moving.jl`, `hastings_baseline.jl`).
5. **`src/legacy/`: delete** (preserved by the `pre-prune` tag; point to the tag in `docs/codeplan.tex`).
   `examples/legacy/` (Python analysis, does not load `src/`) stays.
6. **Settings: a known list and a hard stop.**  The central table in `settings.jl` lists every valid name (with
   default and meaning).  Any `SR_*` or `SPH_*` in the environment that is not in the table stops the run at load,
   with a message suggesting the nearest valid name.  Retired names stop with their own message.  Owner: "scientific
   codes should just stop; getting the right inputs is ultra important, and hard."  Also: strict Bool parsing
   (1/true/yes, 0/false/no, anything else stops) and Symbol settings checked against their allowed values.
   Settings valid in the table but not used in this mode (e.g. `SR_HMC_N` with `SR_START=cold`): **record only**
   (owner, 2026-11-27): no warning printed, no stop; the `.toml` gets an `[unused]` list (name and value) of the
   variables set in the environment but never read, computed just before the main loop and rewritten with each record.
   Rules: unknown name → stop; bad value (Bool, Symbol, number) → stop; valid but unused → `[unused]` in the record.
   Possible addition: a preflight script (`nersc/check_campaign.jl`) that checks names and values of every line of a
   campaign file on the login node before `sbatch` (needs the driver's "set up and stop" mode from `RunConfig`).

## Library files

- `hmc.jl`: good.  Tuning (`nmh = 10`, `widen = 2`, tolerances, `maxit`) are keyword defaults not in the run
  record; non-convergence of the implicit steps only warns; `generalized_leapfrog!` allocates per call; the HMC
  loop belongs here (`hmc_equilibrate!`) rather than in five drivers.
- `stochastic.jl`: the default path (Hastings, `:moment`, single vote, Liouville) is ~40% of the file.
  The rest is older variants: the non-Hastings proposals and votes (`propose_stress!`, `propose_resid!`,
  `ask_star!`, `color_sweep!`, the `:bonds`/`:even` votes), `resid_deterministic!` (λ = ∞), and the gradient
  back map (`gradient_freeze`, `gradient_backmap!`).  **The gradient back map is obsolete** (it belongs to the
  smoothed Hamiltonian; with the energy Hamiltonian it is an error).  Candidate: the default path in its own file,
  the rest in `src/legacy/`.  With Hastings, the frozen coefficients in `Star` are recomputed by `star_coef` at each
  proposal anyway.
- `monitor.jl`: the log label "|v|≥1 rejected" also counts γ > γ_max and states outside the Hamiltonian's domain.
- `sph.jl`: `default_dissipation` and `default_hamiltonian` depend on each other (no loop, but subtle).
- Load-time `if SPLIT / SSCHEME` blocks choose which methods exist: the `SPH_*` variables must be set before
  Julia starts; a precompiled package would freeze them (reason not to make `src/` a package as it stands).
