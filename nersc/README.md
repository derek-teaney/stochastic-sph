# Running stochastic SPH at NERSC (Perlmutter CPU nodes)

The runs are embarrassingly parallel: many independent realizations and parameter sets.
Each realization is one Julia process with a few threads; Slurm job arrays in the
`shared` QOS run many of them side by side, charged per core.

## One-time setup

1. **Julia.**  Install it with juliaup in your home directory (this gives Julia 1.12,
   which the code requires; check `module avail julia` too, but NERSC's module may be
   older):

   ```bash
   curl -fsSL https://install.julialang.org | sh -s -- --yes
   # log out and in, then
   juliaup add 1.12 && juliaup default 1.12
   julia --version
   ```

2. **Package depot.**  Julia keeps packages and compiled caches in a "depot".  Put it on
   the software file system, which compute nodes read fast (adjust the project name):

   ```bash
   mkdir -p /global/common/software/<project>/$USER/julia_depot
   echo 'export JULIA_DEPOT_PATH=/global/common/software/<project>/$USER/julia_depot' >> ~/.bashrc
   ```

   (Home also works for a start; it is just slower when hundreds of jobs start at once.)

3. **The code.**

   ```bash
   cd $HOME   # or /global/common/software/<project>/$USER
   git clone https://github.com/derek-teaney/stochastic-sph.git
   cd stochastic-sph && git checkout nextstage
   ```

4. **Install and precompile** the run-only environment `nersc/` (no plotting) on the
   login node, once:

   ```bash
   julia --project=nersc -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
   ```

5. **Smoke test** on the login node (a few seconds):

   ```bash
   SR_OUTDIR=$SCRATCH/sph/smoke julia -t 2 --project=nersc \
       examples/stochrep_equilibrium.jl 0.6 2.0 0.2 1 2 smoke
   ```

## Running a campaign

A campaign file lists runs, one per line, each with a number of seeds
(`nersc/campaign_example.txt`).  Every (run, seed) pair is one array task.

```bash
mkdir -p slurm
N=$(nersc/ntasks.sh nersc/campaign_example.txt)
sbatch --array=1-$N%64 -A <account> nersc/array.sh nersc/campaign_example.txt first
squeue --me                      # status
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

To keep the Slurm logs in scratch too, submit from there and point the job at the code:

```bash
R=$HOME/stochastic-sph
mkdir -p $SCRATCH/sph/slurm && cd $SCRATCH/sph
N=$($R/nersc/ntasks.sh $R/nersc/campaign_example.txt)
sbatch --array=1-$N%64 -A <account> --export=ALL,SR_REPO=$R \
       $R/nersc/array.sh $R/nersc/campaign_example.txt first
```

- Output: `$SCRATCH/sph/<campaign name>/<run>/`: `equilibrium_*.h5` (HDF5) or
  `stochrep_hamiltonian_*.jls`, one file per realization, with the job script, campaign file and a per-task record.
  `$SCRATCH` is purged after 8 weeks of no access: copy results worth keeping to CFS
  (`/global/cfs/cdirs/<project>`) or home.
- `%64` limits how many tasks run at once.
- Time per task: `t_end` × (steps per unit time) × (cost per step); see
  `docs/plan.tex`, "Where the time goes".  At `h/a = 2`, `N = 672`: about 8 minutes per
  realization to `t = 100`; 2 cores are as fast as 8 (`nersc/LOG.md`).  Raise `#SBATCH -t` for bigger boxes.

## Analysis

Plots need PyPlot, which the run environment leaves out.  Either copy the `.jls` files
back and run the plotting scripts locally, or install the full environment
(`julia --project=. -e 'using Pkg; Pkg.instantiate()'`, which sets up Python/matplotlib
through Conda) and run the analysis in an interactive job:

```bash
salloc -C cpu -q interactive -t 01:00:00 -c 16 -A <account>
```

Do not run heavy computations on the login nodes.
