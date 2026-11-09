#!/bin/bash
# Slurm job array for a campaign file (see campaign_example.txt).  One array task per
# (line, seed); each task runs one realization with 2 Julia threads
# (more threads barely help: 8 cores were only ~10% faster than 2, nersc/LOG.md).
#
#   sbatch --array=1-<ntasks> -A <account> nersc/array.sh <campaign file> <campaign name>
#   To submit from another directory (e.g. $SCRATCH/sph, where slurm/ logs then go):
#   sbatch --export=ALL,SR_REPO=$HOME/stochastic-sph ... $HOME/stochastic-sph/nersc/array.sh \
#          $HOME/stochastic-sph/nersc/<campaign file> <campaign name>
#   (nersc/ntasks.sh <campaign file> prints <ntasks>)
#
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 08:00:00
#SBATCH -c 2
#SBATCH -J sph
#SBATCH -o slurm/%x-%A_%a.out
set -euo pipefail
CAMPAIGN=$1;  NAME=$2
# sbatch runs a spooled copy of this script, so $0 does not locate the repository.
# Use SR_REPO, else the submit directory if it is the repository.
REPO=${SR_REPO:-$SLURM_SUBMIT_DIR}
[ -f "$REPO/nersc/array.sh" ] || { echo "set SR_REPO to the stochastic-sph checkout (got '$REPO')"; exit 1; }
OUTROOT=${SR_OUTROOT:-$SCRATCH/sph/$NAME}
command -v julia >/dev/null || module load ${SR_JULIA_MODULE:-julia/1.12.1}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/global/common/software/m3722/$USER/julia_depot}
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

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
envs=$(echo "$line" | sed 's/[[:space:]]--[[:space:]].*//' | awk '{$1=$2=""; print}')
cmd=$(echo "$line" | sed 's/.*[[:space:]]--[[:space:]]//')
mkdir -p "$OUTROOT/$run"
echo "task $SLURM_ARRAY_TASK_ID: run $run seed $seed: $envs julia $cmd"
cd "$REPO"

# provenance: the job script that ran ($0 is Slurm's copy), the campaign, and a per-task record
cp "$0" "$OUTROOT/jobscript_${SLURM_ARRAY_JOB_ID}.sh" 2>/dev/null || true
cp "$CAMPAIGN" "$OUTROOT/campaign_${SLURM_ARRAY_JOB_ID}.txt" 2>/dev/null || true
commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
git -C "$REPO" diff --quiet HEAD 2>/dev/null || commit="$commit+dirty"
cat > "$OUTROOT/$run/run_seed${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" <<EOF
job       $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date -Is)
sbatch    -A ${SLURM_JOB_ACCOUNT:-?} -q ${SLURM_JOB_QOS:-?} -c ${SLURM_CPUS_PER_TASK:-?}
repo      $REPO @ $commit
campaign  $CAMPAIGN
line      $line
command   cd $REPO && env $envs SR_SEED0=$((seed - 1)) SR_OUTDIR=$OUTROOT/$run julia --project=nersc --startup-file=no $cmd
EOF
env $envs SR_SEED0=$((seed - 1)) SR_OUTDIR="$OUTROOT/$run" \
    julia --project=nersc --startup-file=no $cmd
