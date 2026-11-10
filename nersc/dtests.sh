#!/bin/bash
# dtests.sh — the deterministic test suite (examples/test*.jl, shearwave_decay.jl,
# shearwave_scan.jl) in three configurations, side by side on one node, one thread each:
#   s = SPH_HAMILTONIAN=smoothed,  k = energy with the kernel volume,  n = energy with
#   SPH_VOLUME=number (volume_n.tex).  Figures (Agg) and logs in $OUT/<cfg>/.
#   R=$HOME/stochastic-sph-run; cd $SCRATCH/sph
#   sbatch -A m3722 -q debug -C cpu -N 1 -t 00:30:00 --export=ALL,SR_REPO=$R $R/nersc/dtests.sh [out]
#SBATCH -J sph_dtests
#SBATCH -o slurm/%x-%j.out
set -uo pipefail
R=${SR_REPO:?set SR_REPO}
OUT=${1:-$SCRATCH/sph/number_tests}
module load julia/1.12.1 python >/dev/null 2>&1
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/global/common/software/m3722/$USER/julia_depot}
PROJ=${SR_PLOTPROJECT:-$HOME/stochastic-sph}          # an environment with PyPlot
cd "$R"
echo "dtests.sh: $(git -C $R rev-parse --short HEAD) on $(hostname), $(date)"
run() {   # cfg name script [env...]
    local cfg=$1 name=$2 script=$3;  shift 3
    mkdir -p "$OUT/$cfg"
    case $cfg in
        s) local he="SPH_HAMILTONIAN=smoothed SPH_VOLUME=kernel";;
        k) local he="SPH_HAMILTONIAN=energy SPH_VOLUME=kernel";;
        n) local he="SPH_HAMILTONIAN=energy SPH_VOLUME=number";;
        *) local he="";;
    esac
    ( t0=$(date +%s)
      env $he SR_FIGDIR="$OUT/$cfg" "$@" srun --exact -n1 -c 2 --cpu-bind=cores \
          julia -t 1 --startup-file=no --project="$PROJ" "examples/$script" > "$OUT/$cfg/$name.log" 2>&1
      echo "$cfg $name exit $? $(( $(date +%s) - t0 )) s" >> "$OUT/status.txt" ) &
}
for cfg in s k n; do
    for t in testgaussian testgaussian_1d testsound testsound_viscous testsound_viscous_1d testdiffuse testshearwave; do
        run $cfg $t $t.jl
    done
done
run sw decay shearwave_decay.jl SW_HAMS=smoothed,energy,energy_n
run sw decay_eta0.002 shearwave_decay.jl SW_HAMS=smoothed,energy,energy_n SW_ETA=0.002 SW_OUT=shearwave_decay_eta0.002.png
run sw scan shearwave_scan.jl SC_HAMS=energy,energy_n SC_KERNELS=C2,C6 SC_HA=2,3
wait
echo "dtests.sh: done $(date)"
