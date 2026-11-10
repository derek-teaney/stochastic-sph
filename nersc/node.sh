#!/bin/bash
# node.sh — run the tasks of a campaign file packed onto whole nodes, instead of one Slurm
# job per task: inside one allocation, start each task as its own job step (srun --exact)
# with THREADS physical cores, keep NSLOT of them running, and start the next task as soon
# as one finishes.  Each task runs nersc/array.sh exactly as in an array job (same output,
# run records and checkpoints), with SLURM_ARRAY_TASK_ID set to the task number.
#
#   R=$HOME/stochastic-sph-run; cd $SCRATCH/sph
#   sbatch -A m3722 -q regular -C cpu -N 1 -t 04:00:00 --export=ALL,SR_REPO=$R,THREADS=4 \
#          $R/nersc/node.sh $R/nersc/<file> <name> [first last]
#
# THREADS = Julia threads (= physical cores) per task (default 4); NSLOT = 128·nodes/THREADS.
# Perlmutter CPU nodes have 128 physical cores with 2 hardware threads each, and Slurm's -c
# counts hardware threads: -c 2·THREADS with --cpu-bind=cores gives THREADS physical cores.
# A task that does not finish leaves its .chk; rerun the same command to resume.
set -uo pipefail
CAMPAIGN=$1;  NAME=$2
R=${SR_REPO:?set SR_REPO to the run checkout}
T=${THREADS:-4}
NT=$("$R/nersc/ntasks.sh" "$CAMPAIGN")
FIRST=${3:-1};  LAST=${4:-$NT}
NSLOT=$(( 128 * ${SLURM_NNODES:-1} / T ))
OUTROOT=${SR_OUTROOT:-$SCRATCH/sph/$NAME};  mkdir -p "$OUTROOT/slurm"
command -v julia >/dev/null || module load ${SR_JULIA_MODULE:-julia/1.12.1}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/global/common/software/m3722/$USER/julia_depot}
echo "node.sh: job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, tasks $FIRST-$LAST of $NT, $NSLOT slots x $T cores, $(date)"

running=0
for i in $(seq "$FIRST" "$LAST"); do
    if [ "$running" -ge "$NSLOT" ]; then wait -n; running=$((running - 1)); fi
    SLURM_ARRAY_TASK_ID=$i SLURM_ARRAY_JOB_ID=$SLURM_JOB_ID SR_THREADS=$T \
        srun --exact -N1 -n1 -c $((2 * T)) --cpu-bind=cores \
             bash "$R/nersc/array.sh" "$CAMPAIGN" "$NAME" > "$OUTROOT/slurm/${SLURM_JOB_ID}_$i.out" 2>&1 &
    running=$((running + 1))
    sleep 0.2                       # stagger the Julia start-ups a little
done
wait
echo "node.sh: done $(date)"
