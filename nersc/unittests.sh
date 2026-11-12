#!/bin/bash
# unittests.sh — test/run.sh (unit tests, load check, regression against $REF) on one node.
#   sbatch -A m3722 -q debug -C cpu -N 1 -t 00:30:00 --export=ALL,SR_REPO=<checkout> nersc/unittests.sh [out]
# MODE (default all: unit, load, regress) and REF (default main) from the environment.
#SBATCH -J sph_unittests
#SBATCH -o slurm/%x-%j.out
set -uo pipefail
R=${SR_REPO:?set SR_REPO};  OUT=${1:-$SCRATCH/sph/tests/$(date +%Y%m%d-%H%M%S)}
module load julia/1.12.1 python >/dev/null 2>&1
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/global/common/software/m3722/$USER/julia_depot} MPLBACKEND=Agg
export SR_TESTPROJECT=${SR_PLOTPROJECT:-$HOME/stochastic-sph}     # an environment with PyPlot
echo "unittests.sh: $(git -C "$R" rev-parse --short HEAD) on $(hostname), $(date)"
"$R/test/run.sh" "${MODE:-all}" -r "${REF:-main}" -o "$OUT" -j 64
