#!/bin/bash
# X4 smoke runs of the exact form (docs/sscheme_plan.tex, "The exact form", X4), all in one job step on one node.  c_s^2 = 1/2; the exact form
# without and with omega (SPH_POSITIONAL=1), and split+omega as the reference; s_a = 16, 54; v0 = 0, 0.8; t = 20; 2 seeds; equilibrium.jl
# L = 0.6, h/a = 2 (24 runs x 6 threads); and hybrid Monte Carlo of the exact form+omega, 60 updates, s_a = 16, 54 at rest and v0 = 0.8 (8 runs).
# Pinned worktree $SSX (the commit of the tests).
cd ${SSX:?set SSX to the pinned worktree}
OUT=$SCRATCH/sph/ssX/x4;  mkdir -p $OUT
export JULIA_NUM_THREADS=6
run() {  # scheme-tag, sa, v, seed, extra env
  local srel=$([ $2 = 16 ] && echo 2.956e4 || echo 9.977e4)
  local dtf=$([ $3 = 0.8 ] && echo "SR_DTFAC=0.6" || echo "")
  local tag=x4_$1_s$2_v$3_k$4
  env $5 SR_SREL0=$srel SR_BOOST=$3 $dtf SR_KHMAX=1.6 SR_SEEDOFFSET=$((900 + $4)) SR_OUTDIR=$OUT \
    julia --project=. examples/equilibrium.jl 0.6 2 20 1 4 $tag > $OUT/$tag.log 2>&1 &
}
hmc() {  # sa, v, seed
  local srel=$([ $1 = 16 ] && echo 2.956e4 || echo 9.977e4)
  local tag=x4hq_ssxw_s$1_v$2_k$3
  env SPH_RENORM=sscheme SPH_SSCHEME_FORM=exact SPH_POSITIONAL=1 SR_SREL0=$srel SR_BOOST=$2 SR_KHMAX=1.6 SR_HMC_TAUJIT=0.5 SR_HMC_DTFAC=0.35 \
    SR_SEEDOFFSET=$((950 + $3)) SR_OUTDIR=$OUT julia --project=. examples/hmc_positions.jl 0.6 2 60 0 $tag > $OUT/$tag.log 2>&1 &
}
for sa in 16 54; do for v in 0 0.8; do for k in 0 1; do
  run ssx  $sa $v $k "SPH_RENORM=sscheme SPH_SSCHEME_FORM=exact"
  run ssxw $sa $v $k "SPH_RENORM=sscheme SPH_SSCHEME_FORM=exact SPH_POSITIONAL=1"
  run spw  $sa $v $k "SPH_RENORM=split SPH_POSITIONAL=1"
  hmc $sa $v $k
done; done; done
wait
