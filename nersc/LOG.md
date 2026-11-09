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
- Recording interval: the sound period at k h = 1.5 is 2π/(c_s k) ≈ 0.36; keep
  `rec_every × Δt ≲ 0.05` (Δt = 8.66e-3 at h/a = 2) if C_L is wanted at large k.
- Cost model: wall ≈ c × N × steps, steps = t_end / 8.66e-3 at h/a = 2.

## Campaigns

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

Next once `first` finishes: stability plots (`examples/stochrep_longruns_plot.jl`-style
over the seeds), `SR_DATADIR=$SCRATCH/sph/first/eqL1.2 julia --project
examples/stochrep_correlators.jl L1.2` (and L2.4, in an interactive job), then a short
note in `docs/` with the tables and figures.
