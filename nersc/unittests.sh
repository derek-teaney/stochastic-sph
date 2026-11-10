#!/bin/bash
# unittests.sh — every test/*.jl side by side on one node, one thread each (Agg backend),
# with the default scheme; testregress_step compares against $REF (smoothed path).
#   sbatch -A m3722 -q debug -C cpu -N 1 -t 00:30:00 --export=ALL,SR_REPO=<checkout>,REF=<file.jls> nersc/unittests.sh [out]
#SBATCH -J sph_unittests
#SBATCH -o slurm/%x-%j.out
set -uo pipefail
R=${SR_REPO:?set SR_REPO};  OUT=${1:-$SCRATCH/sph/unittests};  mkdir -p "$OUT"
module load julia/1.12.1 python >/dev/null 2>&1
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/global/common/software/m3722/$USER/julia_depot} MPLBACKEND=Agg
PROJ=${SR_PLOTPROJECT:-$HOME/stochastic-sph};  cd "$R"
echo "unittests.sh: $(git rev-parse --short HEAD) on $(hostname), $(date)"
for f in test/*.jl; do
    n=$(basename $f .jl);  args=""
    [ "$n" = testregress_step ] && args="compare ${REF:?set REF}"
    ( srun --exact -n1 -c 2 --cpu-bind=cores julia -t 1 --startup-file=no --project="$PROJ" $f $args > "$OUT/$n.log" 2>&1
      echo "$n exit $?" >> "$OUT/status.txt" ) &
done
wait;  echo "unittests.sh: done $(date)"
