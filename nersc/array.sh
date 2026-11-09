#!/bin/bash
# Slurm job array for a campaign file (see campaign_example.txt).  One array task per
# (line, seed); each task runs one realization with OMP-free Julia threads.
#
#   sbatch --array=1-<ntasks> -A <account> nersc/array.sh <campaign file> <campaign name>
#   (nersc/ntasks.sh <campaign file> prints <ntasks>)
#
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 06:00:00
#SBATCH -c 8
#SBATCH -J sph
#SBATCH -o slurm/%x-%A_%a.out
set -euo pipefail
CAMPAIGN=$1;  NAME=$2
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUTROOT=${SR_OUTROOT:-$SCRATCH/sph/$NAME}
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# map the array index to (line, seed)
mapfile -t LINES < <(grep -v '^\s*#' "$CAMPAIGN" | grep -v '^\s*$')
i=$((SLURM_ARRAY_TASK_ID - 1));  line=""
for l in "${LINES[@]}"; do
    n=$(echo "$l" | awk '{print $2}')
    if [ "$i" -lt "$n" ]; then line=$l; break; fi
    i=$((i - n))
done
[ -n "$line" ] || { echo "array index beyond campaign"; exit 1; }
run=$(echo "$line" | awk '{print $1}');  seed=$((i + 1))
envs=$(echo "$line" | sed 's/ -- .*//' | cut -d' ' -f3-)
cmd=$(echo "$line" | sed 's/.* -- //')
mkdir -p "$OUTROOT/$run"
echo "task $SLURM_ARRAY_TASK_ID: run $run seed $seed: $envs julia $cmd"
cd "$REPO"
env $envs SR_SEED0=$((seed - 1)) SR_OUTDIR="$OUTROOT/$run" \
    julia --project=nersc --startup-file=no $cmd
