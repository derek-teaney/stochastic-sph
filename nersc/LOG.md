# NERSC log

Running record of what has been set up and run at NERSC (Perlmutter CPU), newest first.
Agents: read this before submitting anything, and add an entry when you submit, finish,
or learn something about running the code here.

## Setup (done 2026-11-08, user `dteaney`)

- Code: `~/stochastic-sph`, branch `nextstage`.
- Julia: module `julia/1.12.1` (no juliaup).  Depot:
  `JULIA_DEPOT_PATH=/global/common/software/m3722/dteaney/julia_depot`.
  Both are set in `~/.bashrc`; `nersc/array.sh` also sets them itself, so jobs do not
  depend on the shell.  Non-interactive shells (agent Bash calls) do not read `.bashrc`:
  run `module load julia/1.12.1` and export the depot first.
- Environments: `nersc/` (run-only, no plotting) and the root one (with PyPlot, built
  against the system `python` module, no Conda: `module load python` first).
- Account: **m3722** for CPU jobs.
- Layout: submit from `$SCRATCH/sph` (Slurm logs → `$SCRATCH/sph/slurm/`), data →
  `$SCRATCH/sph/<campaign>/<run>/`.  Each campaign folder also gets
  `jobscript_<jobid>.sh` (the script Slurm ran), `campaign_<jobid>.txt`, and per task
  `<run>/run_seed<n>_<jobid>_<task>.txt` (node, commit, exact command).
- Submit (see `nersc/README.md`):
  ```bash
  R=$HOME/stochastic-sph; cd $SCRATCH/sph
  sbatch --array=1-$($R/nersc/ntasks.sh $R/nersc/<file>) -A m3722 \
         --export=ALL,SR_REPO=$R $R/nersc/array.sh $R/nersc/<file> <name>
  ```
- Short tests: `-q debug -t 00:20:00` (starts at once; charges a whole node per task, so
  keep them minutes long).  Production: `-q shared` (charged per core).

## Lessons

- **Never edit the checkout that queued jobs run from.**  Each task loads the code from
  `SR_REPO` when it *starts*, so edits reach every task that has not started yet.  On
  2026-11-08 the speedup work was done in `~/stochastic-sph` for ~10 min while `first`
  was queued; tasks 7 and 8 started then (task 7 recorded `7b657c9+dirty`).  They were
  cancelled and rerun as job 61020605 from `nextstage`.  Development now happens in the
  worktree `~/stochastic-sph-speedup` (branch `speedup`); `~/stochastic-sph` stays on
  `nextstage` for the runs.  Check `repo` in the `run_seed*.txt` records.

- `array.sh` had three launch bugs, now fixed: it located the repository from `$0`
  (Slurm runs a spooled copy), did not load the Julia module, and split campaign lines
  on single spaces (aligned columns broke `env`).  Dry-run changes with a stand-in
  `julia` on PATH (prints its arguments) before submitting.
- **Threads barely help.**  Benchmark to t = 2 (231 steps), debug queue:

  | run | N | `-c 8` | `-c 2` | µs / particle / step (`-c 8`) |
  |---|---|---|---|---|
  | hamiltonian, h/a=2 | 672 | 15 s | 16 s | 97 (two stochastic half steps) |
  | equilibrium L=1.2 | 2688 | 25 s | 28 s | 40 (merged half steps) |
  | equilibrium L=2.4 | 10560 | 92 s | 102 s | 38 |

  A Slurm CPU on Perlmutter is a hardware thread (2 per physical core), so `-c 8` is 4
  physical cores and `-c 2` is one core with 2 threads; shared QOS charges per physical
  core.  Results are identical for any thread count.  Default is now `-c 2`.  Speed-up needs
  threading the serial parts (star rebuild, fits, smoothing, leapfrog), `docs/plan.tex`.
- Output of `stochrep_equilibrium.jl` is HDF5 (`equilibrium_<tag>_r<seed>.h5`):
  datasets `ks t ML MT S H KE`, parameters as root attributes (incl. `commit`).  Read
  with `read_equilibrium` (`src/measure.jl`) or h5py (arrays appear as (nt, nk)).
  `stochrep_hamiltonian.jl` still writes `.jls` (Julia `serialize`; needs the same
  `src/` definitions to load).
- **Throughput in `shared`: `MaxJobsAccruePU = 2`.**  Only two pending tasks gain
  priority at a time; array tasks start in pairs about every 5 min (≈ 24 per hour),
  whatever the `%` limit.  176 tasks take ≈ 7 h just to start.  For many long tasks,
  consider packing them into whole-node jobs (64 two-thread tasks per node) instead.
- Recording interval: the sound period at k h = 1.5 is 2π/(c_s k) ≈ 0.36; keep
  `rec_every × Δt ≲ 0.05` (Δt = 8.66e-3 at h/a = 2) if C_L is wanted at large k.
- Cost model: wall ≈ c × N × steps, steps = t_end / 8.66e-3 at h/a = 2.

## Campaigns

### `energystep` — 2026-11-09/10, branch `hamiltonian-energy-step`, done

- Run checkout `~/stochastic-sph-run5` (detached at `934aa86`).  Debug jobs 61055259 (t = 10,
  no crash) and **61055621** (L = 0.6, t = 200, 4 seeds: energy/smoothed, cold start and ideal
  step alone; `$SCRATCH/sph/energystep_debug2/`, `examples/energystep_equilibration.jl`).
- Campaign job **61067905** (`-q shared -c 8 -t 2:00:00`); tasks 1 and 3 timed out and resumed
  as **61073581**.  Output `$SCRATCH/sph/energystep/`.  Analysis `examples/energystep_compare.jl`
  against rigidity `ha2`.
- Outcome (docs/energystep.pdf, last section): the transverse oscillation is gone (c_t fit
  0.000–0.002, C_T ≈ e^{−νk²τ}, damping 1.0–1.3× input, C_T(k,0) 0.93–1.04); KE/eq 0.99, no
  heating; H falls linearly, −3.5e−5 per unit time, every seed.  Ideal step alone: KE/eq ≈ 0.67,
  no runaway (smoothed: 60–95).  Wall 4.5× longer than smoothed at L = 1.2, though 1.4×
  faster at t ≤ 10 on one node: the slowdown builds up during the run (being measured).

#### Original proposal

Note from the local session for the NERSC session.  Read `docs/energystep.pdf` first.
- **What it is.**  `SPH_HAMILTONIAN=energy` (default `smoothed`, so nothing else changes):
  H = Σ_a e_a, each particle's energy at its own momentum density m_a/V_a, V_a = s_a/S_a.
  ẋ_a = v_a (not smoothed), a pure SPH pressure force, a viscous step with no change of
  representation.  In `stochrep_step!` the gradient correction is off with `:energy` (an
  error if forced on); `stochrep_hamiltonian.jl` follows, output suffix `_energy`.
  `stochrep_equilibrium.jl` needs no change (it uses the defaults).
- **Checked locally.**  Hamilton's equations against finite differences of H (1e-6);
  momentum to 1e-16.  Deterministic shear wave (`examples/shearwave_decay.jl`,
  `shearwave_scan.jl`): the two Hamiltonians agree; the ideal step has the same transverse
  sound, c_t = 0.034 at h/a = 2 (C2), the static value of `rigidity.tex` — as expected,
  since H is the same function of the positions at m = 0.  It does **not** cure the
  rigidity by itself.
- **Stochastic step, one-step bookkeeping** (`examples/energystep_stoch_onestep.jl`,
  L = 0.6, h/a = 2, eq start, 20 half steps at fixed positions, 3 seeds):
  `:smoothed` (gradient map) ΔH/H per step rms 3.4e-5, random sign; `:energy` ΔH = Δ(Σe)
  exactly, rms 1.8e-7, but systematic: −1.8e-7 per half step (cooling, about −4e-5 per
  unit time).  Correlated (−0.8) with Σ(Teff−T)Δs but of opposite sign: not understood yet.
- **Proposed runs** (`nersc/campaign_energystep.txt`, 5 tasks, `-c 8`, the same settings
  as `rigidity` ha2 and nv_ha2): ≈ 0.15 node-hours, scaled from `rigidity` (28 tasks ≈
  0.8).  Compare C_T, C_L, ΔH/H(t), KE/eq and UT with ha2 / nv_ha2.  Ask the owner before
  submitting; run from a separate checkout of `hamiltonian-energy-step` at a fixed commit.
- **Next idea, not implemented** (`energystep.tex`, "volumes that add up to the area"):
  V_a = A s_a/S_a / Σ_c(s_c/S_c) makes Σ V_a = A, which removes the pressure-driven
  modulus G ~ p ε on a lattice; the force becomes the present one with p → p − p̄.  If it
  is implemented, the same two campaign lines with it are the test.

### `rigidity` — 2026-11-09, `nersc/campaign_rigidity.txt` (+ `_det`), branch `rigidity`

Where does the transverse rigidity come from (docs/campaigns.pdf)?  L = 1.2, t = 200, 16
realizations per set, displacements UL/UT recorded: h/a = 2, 2.5, 3; Wendland C4 at h/a = 2
(`SPH_KERNEL=wc4`); positions frozen (`SR_NOHAM=1`); λ = ∞ (`SR_LAMBDA=Inf`); ideal SPH alone
(`SR_NOVISC=1`, thermal start, L = 0.6, t = 50) at each h/a.  4 seeds per task
(`SR_SEEDSTRIDE=4`, nreal 4, `-c 8`).
- Job **61045626** (24 tasks, run checkout `~/stochastic-sph-run2` at `4663c3f`) and job
  **61045885** (λ = ∞, 4 tasks, `~/stochastic-sph-run3` at the next commit).  ≈ 0.8 node-hours.
  Output `$SCRATCH/sph/rigidity/`.  Analysis: `examples/rigidity_analysis.jl`.
- Job **61046892** (9 tasks, `~/stochastic-sph-run4`): C6 at h/a = 2, C4 at h/a = 2.5, ideal SPH C6.
- **Outcome** (all 37 tasks done 2026-11-09 19:35; docs/campaigns.pdf, "The rigidity campaign"):
  c_t (fit, kh = 0.37) = 0.022, 0.015, 0.012 at h/a = 2, 2.5, 3 (C2); C4 0.023, 0.016; C6 0.026;
  λ = ∞ 0.020; positions frozen 0.000–0.003 with the input viscosity.  Not the T = 0 phonon
  values (C4 lattice is unstable there).  Displacement estimate 25–35% below the fit.  Ideal
  SPH alone runs away at every h/a and kernel (KE/eq 40–90 by t = 50).  With
  `SPH_HAMILTONIAN=energy` the rigidity is gone (see `energystep`, branch `hamiltonian-energy-step`).
- Found while setting up (L = 0.6, interactive): ideal SPH alone does not hold the lattice
  (KE/eq 0.9 → 6.8 by t = 20 from the thermal start, H conserved); random positions release
  ≈ 400 T per particle (KE/eq → 420 by t = 5) and heat the full scheme (ΔH/H +0.64 by t = 10).

### `second` — 2026-11-09 (done 10:56, 64/64), `nersc/campaign_second.txt`, 64 tasks, ≈ 0.8 node-hours

The fast code (`main` at `f355cd0`: speedup merged after validation, equilibrium start,
defaults star / λ = 1 / hot start, checkpoints every 20 min).  eqL1.2 and eqL2.4, 32
realizations each, t = 400, SR_START=eq.  Compare with the same runs of `first`.

- Run checkout: the worktree **`~/stochastic-sph-run`**, a detached HEAD at `f355cd0`, edited
  by no one.  (`~/stochastic-sph` stays on `nextstage` until job 61035596 is done.)
- Submitted as job **61037106** (`-q shared -c 2 -t 8:00:00`, m3722).  Tasks 1–32 eqL1.2,
  33–64 eqL2.4.  Output `$SCRATCH/sph/second/`.  A task that times out leaves a `.chk` and
  a partial `.h5`; resubmitting the same array index resumes it.

**Outcome.**  All 64 completed (eqL2.4 ≈ 3.7 h per task).  Same as `first` within the spread:
ΔH/H(t=400) = 0.219 (L1.2), 0.221 (L2.4); KE/eq 1.31.  The transverse correlator is a damped
oscillation with c_t = 0.022 at all kh = 0.13–0.74 in both boxes; damping 2γ/νk² = 1.1–1.2 at
kh ≥ 0.5, rising to ≈ 2 at kh = 0.13 (extra rate ∝ k^{2/3}).  C_T(k,0) = 0.75 at kh = 0.13.
Report: `docs/campaigns.pdf` (figures `docs/figs/compare_*.png`, `transverse_*.png`).

**Validation of the fast code** (before `second`): 32 vs 32 realizations of the
equilibrium driver, L = 0.6, t = 50, S0 = 1e6, old (`nextstage`) vs new (`speedup`).  All
13 observables (KE/eq, ΔH/H, both acceptances, C_T(k,0), C_L(k,0), C_T(k,τ=5) in three
shells) agree within 1.3σ; the new code took 243 s against 502 s.  Speed per particle
and step at L = 2.4: 1 thread 52 → 27 μs, 2 threads 42 → 13, 8 threads 37 → 5.5.

**s1e5_hfix rerun (61035596)**: 12/16 failed again in `bond_weights_comoving` once the
lattice had melted (a particle with no neighbours; a superluminal smoothed velocity in
the comoving metric).  Recorded as: the energy fix does not prevent the blow-up at
δv = 0.14.  Not pursued further (owner, 2026-11-09).


### `first` — 2026-11-08, `nersc/campaign_first.txt`, 144 tasks, ≈ 2.1 node-hours

Stability scan in S0 (thermal velocity), the energy fix and h/a, to t = 200 (16 seeds
each); equilibrium ensembles L = 1.2 and 2.4 to t = 400 (32 seeds each).  Estimated
per task: hamiltonian ≈ 0.45 h, eqL1.2 ≈ 1.6 h, eqL2.4 ≈ 5.8 h (limit 8 h).
Output `$SCRATCH/sph/first/`.

- 2026-11-08 19:20: submitted as job **61019642** (`-q shared -c 2 -t 8:00:00`, m3722,
  from `$SCRATCH/sph`, commit `1834b6a`).  Logs `$SCRATCH/sph/slurm/sph-61019642_<task>.out`.
  Tasks 1–80 stability (ref 1–16, s3e5 17–32, s1e5 33–48, s1e5_hfix 49–64,
  s1e5_ha3 65–80), 81–112 eqL1.2, 113–144 eqL2.4.

### `first_det` — 2026-11-08, `nersc/campaign_first_det.txt`, 32 tasks, ≈ 0.12 node-hours

λ = ∞ (kind `det`) at S0 = 3e5 and 1e5, 16 seeds each, t = 200; same seeds as s3e5 and
s1e5 in `first`, so each pair starts from the same state.  Question: does λ = ∞ (cold
lattice modes) stay stable where λ = 1 melts?  If it does, the melting is fed through
the lattice modes; if not, look at the ideal step on a disordered arrangement.

- 2026-11-08 ~19:30: submitted as job **61019836** (`-q shared -c 2`, m3722, commit
  `7295450`).  Tasks 1–16 s3e5_det, 17–32 s1e5_det.  Output `$SCRATCH/sph/first_det/`.

- Tasks 7, 8 of 61019642 (ref seeds 7, 8) cancelled and rerun as job **61020605** (see
  Lessons); their cancelled records are kept as `*.cancelled.txt`.

**Outcome (2026-11-09 morning).**
- Stability scan complete except `s1e5_hfix`: all 16 tasks died with a SingularException in
  the 4×4 solve of `bond_weights_comoving` (melting lattice → nearly singular star).
  Fixed in `34c1c32` (fallback to Λ = 0); rerun as job **61035596**.
- eqL1.2: 32/32 done.  eqL2.4: only 12/32 done; 20 hit the 8 h limit (they needed more
  than 8 h on shared nodes, estimate was 5.8 h), and the .h5 is written only at the end,
  so they are lost.  To be rerun (see the speedup branch, validated first).
- Figures: `docs/figs/campaign_first_stability.png` (`examples/campaign_stability_plot.jl`),
  `docs/figs/correlators_L1.2.png` (`examples/stochrep_correlators.jl`).

**What the stability scan shows** (t = 200, 16 seeds per set, box 0.6, h/a = 2 unless noted):

| set | ΔH/H | KE/eq | disorder | melted |
|---|---|---|---|---|
| S0=1e6 (δv=0.043), λ=1 | +4.6e-2 | 1.06 | 0.31 a | 0/16 |
| S0=1e6, λ=∞ | +3.2e-3 | 0.05 | 0.19 a | 0/16 |
| S0=3e5 (δv=0.078), λ=1 | +0.47 | 1.66 | 0.83 a | 1/16 (t≈198) |
| S0=3e5, λ=∞ | +4.1e-2 | 0.06 | 0.38 a | 0/16 |
| S0=1e5 (δv=0.14), λ=1 | blow-up | 3e8 | 4.7 a | 16/16 (t≈28) |
| S0=1e5, λ=∞ | +0.40 | 0.09 | 0.93 a | 8/16 (t≈147) but no blow-up |
| S0=1e5, λ=1, h/a=3 | blow-up | 5e10 | 9.9 a | 16/16 (t≈11) |

- The heating grows about ×10–12 per factor 1.83 in δv, i.e. roughly as δv⁴, for λ = 1 and
  λ = ∞ alike; λ = 1 heats 10–14× faster than λ = ∞ at the same δv.
- The λ = 1 instability sets in between δv = 0.078 and 0.14 (h/a = 2).  λ = ∞ never blows
  up, even where its lattice disorders to ~1 a: the runaway goes through the lattice modes.
- h/a = 3 at the same S0 has more particles in the box, so a larger δv per particle: not a
  clean comparison at fixed δv.

**What the L = 1.2 correlators show** (32 realizations, t = 400, S0 = 1e6):
- Statics C_T(k,0), C_L(k,0) ≈ 1.04–1.10 at kh ≳ 0.5: 5–10% high, about what the slow
  heating does to wT over the run.
- **C_T is not diffusive: it is a damped oscillation** (transverse sound).  At kh = 0.26,
  0.37, 0.52 the first zero is at τ ≈ 11, 7.5, 5 and the first minimum (C ≈ −0.3) at τ ≈ 22,
  14.5, 10: ω ∝ k with c_t ≈ 0.029 (c_s = 0.577), i.e. a shear modulus G = w c_t² ≈ 0.003 p.
  The particle arrangement (disorder 0.3 a, never melts) responds as a weakly rigid
  solid.  The fitted "ν_fit/ν ≈ 2 at small k" is this oscillation, not a viscosity.
  Testing e^{−νk²τ} needs the lattice rigidity understood or removed first.

Next once `first` finishes: stability plots (`examples/stochrep_longruns_plot.jl`-style
over the seeds), `SR_DATADIR=$SCRATCH/sph/first/eqL1.2 julia --project
examples/stochrep_correlators.jl L1.2` (and L2.4, in an interactive job), then a short
note in `docs/` with the tables and figures.
