#!/bin/bash
# N4 smoke runs, all 40 in one job step (the OS schedules them over the node's 256 CPUs; the earlier --overlap steps were stacked on a few
# cores).  Measure form (beta = D/2+1) and split, c_s^2 = 1/2, 1/4; the measure form with beta = b = 2 at c_s^2 = 1/4; s_a = 16, 54;
# v0 = 0, 0.8; t = 20; 2 seeds; equilibrium.jl L = 0.6, h/a = 2.  Pinned worktree ~/ssM2 (0c00611).
cd ~/ssM2
OUT=$SCRATCH/sph/ssM/n4
export JULIA_NUM_THREADS=6
run() {  # scheme-tag, c2, sa, v, seed, extra env
  local srel=$([ $3 = 16 ] && echo 2.956e4 || echo 9.977e4)
  local dtf=$([ $4 = 0.8 ] && echo "SR_DTFAC=0.6" || echo "")
  local tag=n4_$1_c$2_s$3_v$4_k$5
  env $6 SR_CS2=$2 SR_SREL0=$srel SR_BOOST=$4 $dtf SR_KHMAX=1.6 SR_SEEDOFFSET=$((900 + $5)) SR_OUTDIR=$OUT \
    julia --project=. examples/equilibrium.jl 0.6 2 20 1 4 $tag > $OUT/$tag.log 2>&1 &
}
for c2 in 0.5 0.25; do for sa in 16 54; do for v in 0 0.8; do for k in 0 1; do
  run sscheme $c2 $sa $v $k "SPH_RENORM=sscheme"
  run split   $c2 $sa $v $k "SPH_RENORM=split"
  [ $c2 = 0.25 ] && run sschemeb $c2 $sa $v $k "SPH_RENORM=sscheme SPH_SSCHEME_BETA=2.0"
done; done; done; done
wait
