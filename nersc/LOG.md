# NERSC log

Running record of what has been set up and run at NERSC (Perlmutter CPU), newest first.
Agents: read this before submitting anything, and add an entry when you submit, finish,
or learn something about running the code here.

## Reply from the NERSC session (2026-11-15, evening)

**Campaign `moment` ran fine**: job 61359856 (4 debug nodes, node.sh, 8 min), 96/96 complete, at d1b5ee6 (before a1b1457; the 3e-14
roundoff of common does not matter here).  testmoment.jl PASS at NERSC.  Results in docs/inverse_moment.tex, new last section "Stage 1:
results from NERSC" (figures moment_nersc_*): acceptances identical to 1e-3, no guards, KE and C_AA/χ equal within the errors; one flag,
the shear damping along the flow at v0 = 0.8, kh = 0.52 (L = 0.6 fundamental): common 1.49(17), moment 0.86(14), 2.9σ.

**Problems found (scripts, not the scheme):**
1. examples/moment_det.jl and commonkernel_waves.jl: for δm along a strong flow (v0 = 0.8: long k∥x, shear k∥y) the deficit D0 < 0, so
   Γ_scheme = K/D < 0, t_end < 0, steps < 0: those cases never ran and showed as 0 in the plots (also in the earlier commonkernel tests).
   Fixed in df4dc51 (|K/D| sets the run length; "D0_negative" recorded).  Run, they are stable and identical for common and moment, but
   decay at 0.35–1.27 |K/D|: not a rate.
2. `quiet(f) = redirect_stdout(f, devnull)` inside Threads.@threads is not thread-safe: stdout is process-wide, so output printed by
   other threads (e.g. "viscous step alone, …") is lost at random.  Print to stderr or redirect outside the threaded loop.
3. `sbatch --export=ALL,SR_WAVES_SCHEMES=common,moment` splits at the comma (only "common" arrives): set it inside --wrap.  And
   commonkernel_waves.jl had no such option (it always ran particle, common): added in df4dc51.
4. moment_det.jl at NERSC (job 61360797) failed with InexactError Int64(NaN) in cell_index (NaN positions) after the |K/D| change; set A
   alone (job 61361103) runs clean for both schemes, so it comes from set B or C.  Not pursued (the laptop said to skip it here).

## Message from the laptop session (2026-11-15, evening) — read first

The owner says the NERSC session is having problems with campaign `moment`.  We cannot talk directly: **write what went wrong
here (the error, the job ID, the slurm/*.out lines) and push to `inverse-moment`**; the laptop session pulls it.  Known issues:

1. **Pull to `a1b1457` or later** before anything else (`git -C $R pull`).  Earlier commits of this branch changed `common` at
   roundoff (3e-14; harmless for the physics, but the regression suite fails), now fixed.
2. **The regression suite** (`test/run.sh`, `nersc/dtests.sh`): since the new defaults on `main` (b48106b), `test/regress/step.jl`
   stopped with "measure = :liouville is not available with λ = ∞" for the energy Hamiltonian.  Fixed on this branch (the script
   sets SPH_MEASURE=einstein).  Against a reference built from `main` the energy regression still fails for that reason: known, ignore.
3. **The smoke test** is `julia --project=nersc test/testmoment.jl` (no PyPlot).  `examples/moment_det.jl` and
   `examples/moment_gaussian.jl` plot at the end and need the plotting environment ($HOME/stochastic-sph); they are already done
   on the laptop (docs/inverse_moment.tex, figures moment_det.png, moment_gaussian.png) — **skip them at NERSC** unless asked.
4. **Debug queue**: 30 min.  The full runs (t = 100, L = 0.6) include 200 HMC updates first; if they time out, rerun the same
   node.sh command (they resume from .chk), or use an interactive allocation (`salloc -q interactive -N 4 -t 01:00:00`).  If even
   that is too slow, halve t (edit the t argument in a copy of the campaign file) rather than the statistics.
5. Only the 96 stochastic tasks of `nersc/campaign_moment.txt` are needed; the comparison is common against moment with shared
   seeds (analysis groups MOA, MO in examples/boost_analysis.jl).

## Next session at NERSC — campaign `moment` (written 2026-11-15 on the laptop; branch `inverse-moment`)

**What we are trying to do.**  The owner wants a cleaner viscous step, `SPH_DISSIPATION=moment` (docs/inverse_moment.tex; read
Sects. I–V and "Stage 1").  It is the common scheme without the per-star weight correction Λ_a: the faces carry the inverse bond
moment, A_ab = V_a w_ab 𝓜_a⁻¹ r_ab, which closes for any star; the residual is the least-squares residual with the weights c_ab; the
residual move is a Gaussian conditioned on zero virial.  Motivation: Λ_a falls back in 7% of the stars at s_a = 16 (faces that do
not close, small torques), and only with fixed faces can the Hastings correction (stage 2, not implemented) remove the measured bias
of the stochastic step at low s_a.  The owner expects `moment` to do as well as `common` and, if it does, to make it the default.
**We must not make anything worse**: this campaign is the check.  The default stays `common`; nothing here changes other runs.

**Done on the laptop** (docs/inverse_moment.tex, "Stage 1"): test/testmoment.jl passes (identical to common at rest on the lattice
to 4e-16; closure and zero virial to roundoff on jittered and Poisson stars; momentum conserved; K/K_target at v0 = 0.8 within 0.2%
of common); the stochastic step alone samples the exact distribution with frozen coefficients (examples/hastings_baseline.jl,
SPH_MEASURE=einstein) and has the same bias as common when they are rebuilt; the deterministic comparison examples/moment_det.jl
(viscous step alone moving and on disordered positions, full deterministic waves) — results and figure docs/figs/moment_det.png in
the note.

**To run** (worktree at the pushed tip of `inverse-moment`; NOT the `hbya` worktree while its tasks run).  Smoke test on the login
node: `julia --project=nersc test/testmoment.jl` (≈ 1 min).  Then, from `$SCRATCH/sph`:
```bash
R=$HOME/stochastic-sph-mo        # git worktree add ~/stochastic-sph-mo inverse-moment  (or update it to the tip)
# 96 tasks × 4 cores, packed on 4 debug nodes, < 30 min (≈ 2 node-hours); or the same with bash inside
# salloc -A m3722 -q interactive -C cpu -N 4 -t 01:00:00
sbatch -A m3722 -q debug -C cpu -N 4 -t 00:30:00 --export=ALL,SR_REPO=$R,THREADS=4 $R/nersc/node.sh $R/nersc/campaign_moment.txt moment
# the deterministic comparison at L = 1.2 with more threads (one debug node), as on the laptop:
# (it plots at the end, so it needs the plotting environment, $HOME/stochastic-sph, as for the analysis below)
sbatch -A m3722 -q debug -C cpu -N 1 -t 00:30:00 --wrap "module load julia/1.12.1; export JULIA_DEPOT_PATH=/global/common/software/m3722/$USER/julia_depot; cd $R && MD_OUT=$SCRATCH/sph/moment/det SR_FIGDIR=$SCRATCH/sph/moment/det julia -t 64 --project=$HOME/stochastic-sph examples/moment_det.jl"
```
If a full run (t = 100, L = 0.6) does not finish in 30 min, rerun the same node.sh command: the tasks resume from their .chk.
Record the job IDs here.

**Analysis** (the plotting environment `$HOME/stochastic-sph`, which has PyPlot; the scripts are taken from $R):
```bash
D=$SCRATCH/sph/moment
BOOST_GROUP=MOA julia --project=$HOME/stochastic-sph $R/examples/boost_analysis.jl $D    # stochastic step alone: common vs moment
BOOST_GROUP=MO  julia --project=$HOME/stochastic-sph $R/examples/boost_analysis.jl $D    # full step: C_MM, C_EE, shear damping
```
Compare, common against moment, line by line (each pair shares its seeds, so the noise is correlated):
1. acceptances of the stress and residual moves (the run records / logs);
2. stochastic step alone: ⟨δm²⟩ and the equal-time spectrum, at v = 0 and 0.8;
3. full step: equal-time C_MM/χ, C_EE/χ against kh, the shear damping Γ/Γ_hyd at kh = 0.52 and 1.05, the sound;
4. ΔH/H and the guards.
Pass: no difference beyond the statistics, except possibly through the residual at v = 0.8 (f_ab ≠ 1).  Write a short results
paragraph, tables and figures into docs/inverse_moment.tex as a **new last section, "Stage 1: results from NERSC"** (the laptop session
keeps editing the other sections of that note; stay inside this section so that the two do not conflict), figures in docs/figs/
with the prefix `moment_nersc_`; commit on `inverse-moment`, pull --rebase, push.  The owner wants many plots: the deterministic set
and the boost-style stochastic comparison (the MO/MOA figures), common and moment side by side.
Do not change the default; the owner decides after reading.

## 2026-11-15 (laptop): new defaults — read before the next campaign

`equilibrium` was merged into `main` (tag `notes-2026-11-15`, defaults unchanged).  Then, on `new-defaults`: the defaults are the
generalized leapfrog (`SPH_IDEAL=generalized`), the Liouville measure (`SPH_MEASURE=liouville`) and, in examples/equilibrium.jl,
the HMC start (`SR_POSITIONS=hmc`, `SR_HMC_N=200`, `SR_HMC_TAUJIT=0.5`, `SR_HMC_DTFAC=0.35`), i.e. the settings of `hbya`.
Campaign files that spell these out are unaffected; old campaigns relying on the old defaults must now set
`SPH_IDEAL=particle SPH_MEASURE=einstein SR_POSITIONS=lattice` to be reproduced.  Running jobs are unaffected (fixed worktree);
do not update the `hbya` worktree until its tasks are done.  Also measured: the stochastic step alone is biased at low s_a
(+6% in the momentum fluctuations at s_a = 54, +14% at 16; docs/commonkernel.tex) — keep in mind when comparing dynamics to HMC.

## Next session at NERSC — start now: campaigns `hmc_dyn` and `hmc_eq` (written 2026-11-14, on the laptop)

New on `equilibrium`: `SR_POSITIONS=hmc` in examples/equilibrium.jl (hybrid Monte Carlo from the lattice at the β^μ of the
run, at rest or boosted, then the present start), `SR_BOOST` in examples/hmc_positions.jl, the generalized-leapfrog ideal
step `SPH_IDEAL=generalized` (docs/equilibrium_positions.tex).  Checked on the laptop at v = 0.8: HMC acceptance 0.95–0.97,
a liquid; the HMC-started generalized-leapfrog run clean (ΔH/H 1e-5 over t = 5).  The owner asked to start the v = 0 runs now,
with a first boosted set at v = 0.8 alongside; a first look, statistics to be increased afterwards.  The sets form a 2×2 of
ideal step × start at each velocity, so that the effect of each change can be seen separately (owner, 2026-11-14).

Worktree: update the run worktree to the pushed tip of `equilibrium` if no task of `pairs_low` is pending; otherwise make
`~/stochastic-sph-eq2` at the tip and use it below.  Smoke test (login node): `julia --project=nersc test/testhmc.jl`.
```bash
R=$HOME/stochastic-sph-eq; cd $SCRATCH/sph
# hmc_dyn: 104 tasks, ≈ 2.2 node-hours (the t = 2000 runs ≈ 6 h each on 2 threads; they checkpoint, rerun to resume)
sbatch --array=1-104 -A m3722 -t 08:00:00 --export=ALL,SR_REPO=$R $R/nersc/array.sh $R/nersc/campaign_hmc_dyn.txt hmc_dyn
# hmc_eq: 72 tasks (64 at rest, 8 boosted at v = 0.8), ≈ 1.4 node-hours
sbatch --array=1-72 -A m3722 -t 08:00:00 --export=ALL,SR_REPO=$R $R/nersc/array.sh $R/nersc/campaign_hmc_eq.txt hmc_eq
```
Record both job IDs here.  First look (plotting environment), when the short sets are done:
```bash
D=$SCRATCH/sph/hmc_dyn;  E=$SCRATCH/sph/hmc_eq
ln -s $SCRATCH/sph/commonkernel/ck_v0 $D/ck_v0;  ln -s $SCRATCH/sph/commonkernel/ck_v0.8 $D/ck_v0.8   # the DKD + lattice cell
BOOST_GROUP=HD julia --project=$HOME/stochastic-sph examples/boost_analysis.jl $D          # C_EE, shear damping: HMC vs lattice start
SR_FIGDIR=$D/figs julia --project=$HOME/stochastic-sph examples/pair_analysis.jl $D/hd_hmc_v0 $D/hd_lat_v0 $D/hd_hmc_v0.8 $D/hd_long_s541
SR_FIGDIR=$E/figs12 julia --project=$HOME/stochastic-sph examples/hmc_analysis.jl $E/hq12_s54 $E/hq12_s135 $E/hq12_s541 $E/hq12_s541_b08
```
The questions, for a short section of docs/equilibrium_positions.tex: (1) does S(k) of `hd_long_*` approach the HMC value
of `hq12_*` (examples/hmc_analysis.jl gives it per band of kh); (2) do the HMC-started runs keep S(k) at the HMC value, and
how do C_EE(k,0)/χ and the shear damping compare across the 2×2 of ideal step (generalized leapfrog, GLF, or drift–kick–drift,
DKD) and start (HMC or lattice) — `hd_hmc_*`, `hd_lat_*`, `hd_dkdhmc_*` and `ck_*` — at v = 0 and 0.8; which of the two changes
removes the C_EE deficit along the flow; (3) at v = 0.8, g(r)
along and across the flow and S(k) from `hd_hmc_v0.8` against `hq12_s541_b08`, and C_EE along the flow (the deficit of
boost.tex).  Commit the figures and the section on `equilibrium`, push, and report the jobs, cost and outcome here.

**Submitted (2026-11-14, NERSC):** `pairs_low` was still running from `~/stochastic-sph-eq`, so a second worktree
`~/stochastic-sph-eq2` at `cc8e292`; `test/testhmc.jl` PASS (δt² ratios 4.02, 4.01; acceptance 1.00).  `hmc_dyn` array job
**61295240** (104 tasks), `hmc_eq` array job **61295241** (72 tasks), both from `~/stochastic-sph-eq2`, 8 h limit.

**2026-11-14 evening, first look written up** (`docs/equilibrium_positions.tex`, section "First look: HMC start and the
generalized leapfrog"; figs `boost_static_HD.png`, `boost_shear_HD.png`, `equilibrium_Sk_hmc_eq_first.png`).
`hmc_dyn`: 5 of 6 short cells complete (GLF + HMC at v0 = 0.8: 6 of 16); at v0 = 0.8, kh = 0.52 the C_EE deficit along the flow
(0.77) and the excess shear damping (1.42) appear only in DKD + lattice; HMC start or GLF each remove both.  `hmc_eq`: 37 of 72
chains (hq12_s135, hq12_s541, hq12_s541_b08); hq12_s54, hq24_* still running.  Analysis:
`BOOST_BURN=… julia --project=nersc examples/boost_analysis.jl HD`, `examples/hmc_analysis.jl`.  To do as they finish: the
rest of GLF + HMC v0.8, the hq24 chains (then replace Fig. Skk), the long runs' S(k) against HMC, g(r) along/across at v0 = 0.8.

**2026-11-14, campaign `hmc_liou`** (owner's request): the GLF + HMC cell with `SPH_MEASURE=liouville`, v = 0 and 0.8, 16 runs each,
`nersc/campaign_hmc_liou.txt`, array job **61308476** (32 tasks, ≈ 0.75 node-hours), from worktree `~/stochastic-sph-eq3` at `428ea25`
(`test/testmeasure.jl` PASS).  Same seeds as hd_hmc_v0/v0.8.  Output `$SCRATCH/sph/hmc_liou/hl_hmc_v0{,.8}`.
**Correction, same day:** with all 16 GLF + HMC runs and errors at three k, the C_EE deficit along the flow at v0 = 0.8 is not
removed by one change alone (DKD+HMC 0.86(4) at kh 0.79, GLF+HMC 0.77(10) at 0.26; only GLF + lattice consistent with 1) —
the first-look claim was from kh = 0.52 alone.  Section rewritten (997f7d4).

**2026-11-14 evening: the HMC start of hmc_dyn/hmc_liou was resonant.**  `SR_POSITIONS=hmc` used trajectories of τ = L/c_s (default),
exactly one period of the box's fundamental sound mode, so the kh = 0.26 density mode barely moves from the lattice: S(k)/(4/s_a) ≈ 3
at the start of the dynamics against ≈ 20 in equilibrium.  Debug job 61309290 (`$SCRATCH/sph/hmc_tau_test`, 3 × 4 chains from the
lattice, 60 updates): τ = 2.08 → kh 0.26 stuck (4.4 at updates 41–60); τ = 1.73 → 8.6, relaxing; τ = 1.04 = L/2c_s → kh 0.52 stuck
(3.3 against ~10).  Also the two HMC cells of hmc_dyn shared seeds, hence identical starts.  Fix: `SR_HMC_TAUJIT` (examples/equilibrium.jl,
3a84f31), random τ per update; `SR_SEEDOFFSET` per line for independent cells.  hmc_analysis' default HMC_BURN = 20 is too short at
kh ≲ 0.3 (S settles after ~100 updates): rerun with HMC_BURN=100.
**Campaign `hmc_liou2`** (owner: rerun with the Liouville measure, up to 24 node-hours): the 2×2 at v = 0, 0.8, 64 runs per cell, HMC start
τ ∈ [L/2c_s, L/c_s], 200 updates.  `nersc/campaign_hmc_liou2.txt`, array job **61309610** (512 tasks, ≈ 11.5 node-hours), worktree
`~/stochastic-sph-eq4` at `3a84f31`.  hmc_liou (61308476) has the old resonant start; kept running.

**2026-11-14 night: hmc_liou2 done** (512/512; analysis in debug job 61326832, `$SCRATCH/sph/hmc_liou2/analysis/hl2.txt`, `BOOST_GROUP=HL2`;
note the scratchpad /tmp is not visible from compute nodes — write analysis output to $SCRATCH).  The HMC start is in equilibrium (S at
kh 0.26 = 21 at t = 0 and through t = 300; HMC 20.3); the k_x/k_y test at rest passes (Σz²/n 0.62).  At v0 = 0.8 the C_EE deficit along
the flow is the DKD step's: 0.88(2) from lattice and from HMC; GLF 1.02(2) (lattice), 1.08(3) (HMC).  Shear damping along the flow 1.25–1.33
at kh 0.52 in all four cells (1.1 at rest).  KE from the lattice 19.5 (10% high) with either step; from HMC 17.6–17.7.
docs/equilibrium_positions.tex, Sect. "The 2×2 with the Liouville measure".

**2026-11-14 late: hmc_eq.**  hq24_s541 done (8 chains): agrees with L = 1.2 at kh ≥ 0.5; kh ≤ 0.2 not settled in 150 updates.
hq24_s54 (tasks 57–64) all TIMEOUT at 8 h with no output (hmc_positions.jl writes only at the end; ≈ 12.5 h needed on 2 threads,
scaled from hq12_s54) — resubmitted with -t 16:00:00 as job **61330757** from ~/stochastic-sph-eq2 (cc8e292), ≈ 1.6 node-hours.

**2026-11-15: campaign `hbya`** (owner: explore h/a, up to 48 node-hours, done today; docs/hbya.tex).  New option `SR_H` (kernel scale,
default 0.05) in examples/equilibrium.jl.  Timing test debug job 61342810 (`$SCRATCH/sph/hbya_test`).  h/a = 2, 3, 4 at fixed a = 0.025,
s_a = 135, L = 1.2; kernels C2/C4/C6 at v = 0 (32 runs), C2/C4 at v = 0.8 (16 runs); GLF + Liouville + HMC start (200 updates, TAUJIT 0.5).
Worktree ~/stochastic-sph-hb at 18eb717.  Arrays in shared: ha2 **61343006** (-c 2, 2 threads; an earlier -c 4 submission 61342998 was
cancelled before it started), ha3 **61342999** (-c 8, SR_THREADS=4), ha4 **61343000** (-c 16, SR_THREADS=8); 384 tasks, ≈ 37 node-hours.
ha4 moved (owner: pack onto whole nodes) to `node.sh` on 4 regular nodes, THREADS=4 (32 slots per node, all 128 at once), job
**61346451**, -t 09:00:00, ≈ 28 node-hours billed (the longest task, C4 at v = 0.8, ≈ 7 h at 4 threads).  Cancelling array 61343000 also
killed its 6 tasks that had just started (minutes in): check for running tasks right before cancelling an array.  Task logs of the
packed job: `$SCRATCH/sph/hbya/slurm/61346451_<task>.out`.
h/a = 2 complete (128/128, mean 2.2 h per task at -c 2).  ha3 in shared was slow (13 running, 108 pending): the pending tasks 21–128 were
cancelled (`scancel 61342999_[21-128]`, none started) and packed on 4 regular nodes, THREADS=4, job **61349267** (-t 05:00:00).  Packed
ha4 rates: C4 v = 0.8 ≈ 55 min HMC + 15 s per unit time → ≈ 4 h; C4 v = 0 ≈ 2.4 h.
**hbya complete** (2026-11-15 12:25): ha4 packed job 4 h 04 min (4 nodes, ≈ 16 node-hours), ha3 packed 2 h 13 min (≈ 9), ha2 + 20 ha3 in
shared (≈ 4); total ≈ 30 node-hours.  Analysis in debug jobs (`$SCRATCH/sph/hbya/analysis/hb{2,3,4}.txt`).  Result: the excess shear
damping at rest GROWS with h/a at fixed a: δ ≈ 0.09, 0.14, 0.2–0.28 at h/a = 2, 3, 4, all kernels (C6 lowest), flat in kh — an extra
viscosity ∝ h.  docs/hbya.tex.
Also cancelled at the owner's request: the held GPU jobs 58984xxx of the m2 campaign (other project, m4662_g / m3722_g).

**2026-11-15: hq24_s54 rerun (61330757) done**, 8/8 in 11.3–12.5 h, but HMC acceptance 0.16 (δt factor 0.1 is too large for N = 10752):
≈ 13 accepted moves per chain.  kh ≥ 0.5 agree with L = 1.2; kh ≤ 0.3 not converged.  In equilibrium_positions.tex with that caveat.

## Earlier: campaign `hmc_eq` instructions (written 2026-11-14, on the laptop; now submitted with `hmc_dyn` above)

The equilibrium S(k) of the positions from many independent hybrid Monte Carlo chains (docs/equilibrium_positions.tex,
"Hybrid Monte Carlo in the same box" and Fig. Skk; single chains on the laptop were too noisy: one mode of |N_k|² is
exponentially distributed and the chains are autocorrelated).  `nersc/campaign_hmc_eq.txt`, 64 tasks, ≈ 1.3 node-hours.
Run from a worktree at the pushed tip of `equilibrium` that nobody edits (e.g. update `~/stochastic-sph-eq` if no task of
`pairs_low` is pending, otherwise make `~/stochastic-sph-eq2`).  Smoke test first (login node, seconds):
`julia --project=nersc test/testhmc.jl` (PASS).
```bash
R=$HOME/stochastic-sph-eq; cd $SCRATCH/sph
sbatch --array=1-64 -A m3722 -t 08:00:00 --export=ALL,SR_REPO=$R $R/nersc/array.sh $R/nersc/campaign_hmc_eq.txt hmc_eq
```
Record the job ID here.  Analysis (plotting environment), one figure per box:
```bash
P=$SCRATCH/sph/hmc_eq
SR_FIGDIR=$P/figs12 julia --project=$HOME/stochastic-sph examples/hmc_analysis.jl $P/hq12_s54 $P/hq12_s135 $P/hq12_s541
SR_FIGDIR=$P/figs24 julia --project=$HOME/stochastic-sph examples/hmc_analysis.jl $P/hq24_s54 $P/hq24_s541
```
It prints S(k)/(4/s_a) per band of kh with the error from the spread between chains and τ_int.  Replace Fig. Skk
(`docs/figs/equilibrium_Sk_vs_k.png`) by `figs12/hmc_Sk_vs_k.png` with the L = 2.4 sets added, update the table and bullets
of "Hybrid Monte Carlo in the same box" (in particular the small-k plateau and its s_a dependence: S(k→0) ≈ 0.25 at
s_a = 54 and 0.11 at 541 from single chains), compare with the dynamics of `pairs` and `pairs_low`, commit, push, report here.

## Next session at NERSC — also: campaign `pairs_low` (written 2026-11-14, on the laptop)

Do **not** stop `pairs` if it is running, and do not delete its output or any figure: this is a second campaign.
The owner wants lower entropy per particle, but s_a ≥ 30 for now, by two routes that reach the same s_a: the fixed
kernel h/a = 2 at lower S0 (s_a = 135, 60, 34), and the continuum route of campaign `cl` (S0 = 1e6, h/a = 4, 6, 8 →
s_a = 135, 60, 34).  `nersc/campaign_pairs_low.txt`, 72 tasks, L = 0.6, estimate ≈ 1.1 node-hours (cost per step
∝ (h/a)^4, from `cl`).  From the same run worktree as `pairs` (update it to the pushed tip of `equilibrium` first if no
`pairs` task is still pending; otherwise make a second worktree, e.g. `~/stochastic-sph-eq2`):
```bash
R=$HOME/stochastic-sph-eq; cd $SCRATCH/sph
# tasks 1-56: h/a = 2 (three sets) and h/a = 4, one array task each, 2 threads
sbatch --array=1-56 -A m3722 -t 04:00:00 --export=ALL,SR_REPO=$R $R/nersc/array.sh $R/nersc/campaign_pairs_low.txt pairs_low
# tasks 57-72: h/a = 6 and 8, packed, 16 threads each (8 at a time on one node)
sbatch -A m3722 -q regular -C cpu -N 1 -t 04:00:00 --export=ALL,SR_REPO=$R,THREADS=16 \
       $R/nersc/node.sh $R/nersc/campaign_pairs_low.txt pairs_low 57 72
```
Record the job IDs here.  A task that times out leaves a `.chk`; rerun the same command to resume.  Analysis, two
figures (the combined figure takes up to four sets):
```bash
P=$SCRATCH/sph/pairs_low
SR_FIGDIR=$P/figs_ha2 julia --project=$HOME/stochastic-sph examples/pair_analysis.jl $P/pl_ha2_s135 $P/pl_ha2_s60 $P/pl_ha2_s34
SR_FIGDIR=$P/figs_cl  julia --project=$HOME/stochastic-sph examples/pair_analysis.jl $P/pl_ha4_s135 $P/pl_ha6_s60 $P/pl_ha8_s34
```
Copy the two `equilibrium_g_vs_t.png` to `docs/figs/equilibrium_g_vs_t_ha2.png` and `..._cl.png`, add a short paragraph
and table to docs/equilibrium_positions.tex (melting time, ψ6, first peak of g, S(k) at the lowest kh against 4/s_a; same
s_a by the two routes side by side), commit on `equilibrium`, push, and report the jobs, cost and outcome here.

**Submitted (2026-11-14, NERSC):** worktree `~/stochastic-sph-eq` updated to `96ee6f7` (no `pairs` task pending).  Tasks 1-56
array job **61291002**; tasks 57-72 packed job **61291004** (1 node, THREADS=16, 4 h limit).
**Done:** 72/72 complete (array tasks ≈ 0.5 h each; packed node 2 h 20 min; total ≈ 2.4 node-hours, above the 1.1 estimate: the h/a = 6, 8 runs cost ∝ (h/a)^4).  All liquids, melted within t ≈ 0.1 (the
continuum route before the first record); local ψ6 = Poisson.  g(0) ≈ 0.4-0.6 at h/a = 2, g ≈ 1 at all r on the continuum
route.  S(kh = 0.52) = 2.5, 3.3, 2.2 × 4/s_a (h/a = 2, s_a = 135, 60, 34) against 6.5, 4.5, 2.7 (h/a = 4, 6, 8).  Two figures
and a table in docs/equilibrium_positions.tex ("Toward lower entropy per particle").

## Next session at NERSC — start here: campaign `pairs` (written 2026-11-14, on the laptop)

**Branch `equilibrium`** (off `common-default`): the pair correlation of the particle positions as a standard
measurement (docs/equilibrium_positions.tex, stage 1).  Every run of `examples/equilibrium.jl` now also records g(r),
g along/across x, ψ6 (global, local) and N_k (datasets `part_g*`, `part_psi6*`, `part_Nk*` in the .h5).  Numerics
unchanged (test suite on the laptop: all pass; regression against common-default bit-identical in every old dataset).
1. `git -C ~/stochastic-sph fetch`; a fresh worktree detached at the pushed tip, nobody edits it:
   `git -C ~/stochastic-sph worktree add --detach ~/stochastic-sph-eq origin/equilibrium`; precompile there
   (`julia --project=nersc -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`).
2. Quick check, login node, seconds: `julia --project=nersc test/testpairs.jl` (PASS).
3. **Submit campaign `pairs`** (`nersc/campaign_pairs.txt`, 64 tasks, ~0.5 node-hours; within the budget rule):
   ```bash
   R=$HOME/stochastic-sph-eq; cd $SCRATCH/sph
   sbatch --array=1-$($R/nersc/ntasks.sh $R/nersc/campaign_pairs.txt) -A m3722 -t 04:00:00 \
          --export=ALL,SR_REPO=$R $R/nersc/array.sh $R/nersc/campaign_pairs.txt pairs
   ```
   (or packed: `sbatch -A m3722 -q regular -C cpu -N 1 -t 02:00:00 --export=ALL,SR_REPO=$R,THREADS=2 $R/nersc/node.sh
   $R/nersc/campaign_pairs.txt pairs`).  Record the job ID here.  The s_a = 54 runs (`pr_s54`) may blow up after
   t ≈ 45; that is expected and their record up to then is what is wanted.
4. **Analysis** when done (plotting environment):
   `SR_FIGDIR=$SCRATCH/sph/pairs/figs julia --project=$HOME/stochastic-sph examples/pair_analysis.jl
   $SCRATCH/sph/pairs/pr_s54 $SCRATCH/sph/pairs/pr_s541 $SCRATCH/sph/pairs/pr_s5413 $SCRATCH/sph/pairs/pr_s54000`.
   Copy `equilibrium_g_vs_t.png` to `docs/figs/` (it replaces the single-run figure of the note), update the table and
   text of Sect. "First look" in docs/equilibrium_positions.tex with the averaged numbers (melting time, ψ6, first
   peak, S(k) against 4/s_a), commit on `equilibrium` and push.  Report the job, cost and outcome in this log.

**Done (2026-11-14, NERSC):** worktree `~/stochastic-sph-eq` at `f682251`; `test/testpairs.jl` PASS.  Campaign `pairs` job
**61289130** (1 node, THREADS=2, 15 min, 0.25 node-hours, 64/64 complete; the s_a = 54 runs reached t = 200, ΔH/H ≈ 4e-2).
Melting (global ψ6 < 0.1) by t ≈ 0.07, 0.21, 1.5 at s_a = 54, 541, 5413; s_a = 54000 stays solid (ψ6 0.82).  Liquid local
ψ6 = the Poisson value; S(k), kh 0.5-1.5, t > 100 = 2.4, 3.4, 2.1 × 4/s_a (solid 1.4).  In the liquids S(k) at kh = 0.52
is still climbing at t = 200 (×2.8 at s_a = 541): the low dynamics/HMC ratio is a slow relaxation (equilibrium_positions.tex).

## Earlier: start here, common-default (written 2026-11-14, on the laptop)

**The default viscous step changes to the one-kernel step** (`SPH_DISSIPATION=common`, docs/commonkernel.tex) on
branch `common-default` (off `commonkernel`; not yet in `main`).  Runs that must reproduce earlier ones set
`SPH_DISSIPATION=particle`.  To do, from a run checkout that nobody edits (see Lessons):
1. `git fetch`; make a worktree detached at the tip of `origin/common-default`, e.g.
   `git -C ~/stochastic-sph worktree add --detach ~/stochastic-sph-ck origin/common-default`; precompile there.
2. **Test suite**, debug job: `sbatch -A m3722 -q debug -C cpu -N 1 -t 00:30:00
   --export=ALL,SR_REPO=$HOME/stochastic-sph-ck $HOME/stochastic-sph-ck/nersc/unittests.sh` from `$SCRATCH/sph`.
   Expected: unit, physics and load pass.  The regression against `main` is expected to differ for the default
   (energy, 2D) runs only; on the laptop the suite against `commonkernel` gave the list in the commit message of
   the default change.
3. **Campaign `ck_stats`** (`nersc/campaign_ck_stats.txt`, 128 tasks, about 2.3 node-hours): grid + one kernel at
   v = 0.3, 0.6, t = 1000, 64 runs each, to settle the slow late-time shear decay of docs/commonkernel.tex
   (Fig. ckgshear) against `grid_stats` at equal statistics.  Submit with `nersc/node.sh` or `array.sh` as for
   `grid_stats`; analysis: `BOOST_GROUP=CKSTATS julia examples/boost_analysis.jl $SCRATCH/sph/ck_stats`.
4. When 2 passes and 3 settles the question, `common-default` is merged into `main` (the owner decides).

**Done (2026-11-14, NERSC):** worktree `~/stochastic-sph-ck` at `89bc470`.  Test suite job **61283664**: unit, physics,
load pass; the regression against `main` differs exactly as listed in the commit message (energy/2D runs), the smoothed
runs bit-identical; the unit-output diffs are the banner, one more setting in the run record (`SR_POSITIONS`, from
`commonkernel`), the separating trajectories (checkpoint, eqstart, stochastic_guard) and roundoff in testdissipative.
Campaign `ck_stats` job **61283666** (4 nodes, 37 min, 128/128): the shear along the flow at v = 0.3, 0.6 is at the
hydrodynamic rate (0.95-0.99 ± 0.02-0.03), equal to `grid_stats` within errors; the 16-run low values were statistics
(commonkernel.tex, paragraph 9).  Lesson: three large `boost_analysis` runs at once on the login node — one was killed
silently (memory); run them one or two at a time.

## Earlier: start here (written 2026-11-12, on the laptop)

The code was reorganized on the laptop and merged into `main` (see "The code after the
cleanup" below).  Before any new campaign:
1. **Switch the checkouts to `main`.**  `~/stochastic-sph` and the run copies
   `~/stochastic-sph-run*` are on `nextstage`, which is fully contained in `main`:
   `git -C <checkout> status` (no local changes expected), then
   `git -C <checkout> switch main && git -C <checkout> pull`.
2. **Precompile** (login node): `julia --project=nersc -e 'using Pkg; Pkg.instantiate();
   Pkg.precompile()'`, and the same for the plotting environment `$HOME/stochastic-sph`.
   Smoke test, seconds: `julia --project=nersc -e 'include("src/load.jl")'`.
3. **Run the test suite once** in a debug job (it has only run on the laptop so far):
   `sbatch -A m3722 -q debug -C cpu -N 1 -t 00:30:00 --export=ALL,SR_REPO=<checkout>
   <checkout>/nersc/unittests.sh` from `$SCRATCH/sph`.  Expect PASS for every line:
   unit tests, physics (three wave tests, about a minute, four threads each), load, and
   the regression against `main` (built in a git worktree under the output directory).
   Read `<out>/results.toml` and the `slurm/` log; if anything fails, find the cause first.
4. **New campaign files use the new drivers**: `examples/equilibrium.jl`,
   `examples/stability.jl` (same arguments and SR_* settings as before).  Each run now
   writes `<run>.toml`; gather a campaign with `julia --project=nersc nersc/collect_runs.jl
   $SCRATCH/sph/<campaign>`.  The old `campaign_*.txt` files name the removed drivers
   (`stochrep_*`): do not resubmit them.
5. **Old runs on `$SCRATCH/sph/`**: the owner expects to discard them (the figures are in
   `docs/figs`).  Ask before deleting anything.
6. Budget rule unchanged: ask before more than ~20 node-hours, with the estimate.

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

- **What was run** (since 2026-11-12): every run of `equilibrium.jl` and `stability.jl` writes
  `<run>.toml`, generated from the values used: every field of SimParams, the EOS and
  StochOptions, every SR_*/SPH_* setting read (value, default, whether set), the start,
  commit (+dirty and a diff hash), host, Slurm job, and the outcomes (`src/runinfo.jl`).
  `julia --project=nersc nersc/collect_runs.jl $SCRATCH/sph/<campaign>` makes `runs.csv`
  for a spreadsheet; Python reads a record with `tomllib`.  The task log (`slurm/<job>_<task>.out`) has a monitor
  line every 5% of the run (acceptances, |v| ≥ 1 rejections, speeds) and a summary block.

- **Plotting scripts in batch jobs**: matplotlib needs no display with the Agg backend.
  The test scripts include `examples/plotsetup.jl` (Agg inside Slurm or without DISPLAY,
  figures to `SR_FIGDIR`) and run with the full environment (`--project=$HOME/stochastic-sph`,
  which has PyPlot), one thread each; `nersc/dtests.sh` runs the whole suite on one debug node.

- **Packing a whole node: `nersc/node.sh` (2026-11-10).**  Runs the tasks of a campaign file
  inside one allocation as job steps, `srun --exact -n1 -c 2T --cpu-bind=cores`, NSLOT =
  128/T at a time, refilling as tasks finish (`wait -n`); each task is `array.sh` as in an array
  job.  Two traps it avoids: Slurm's `-c` counts hardware threads (2 per core), so `-c 2T` for T
  cores; and `srun` resets `SLURM_CPUS_PER_TASK` in the step, so the thread count is passed as
  `SR_THREADS`.  Test (debug job 61088563): 64 tasks, T = 4: 32 ran at once, each on its own
  4 cores.  Demonstration (debug job 61089773): 40 steps of 4 cores, 32 at once, the rest as slots
  freed; core n's sibling is CPU n+128 (0-3,128-131).  `--exact` turned out not to be needed when
  `-c` is given (steps without it also ran in parallel); kept as harmless.
  Note: in the shared-queue arrays, `-c 8` is 4 physical cores with 8 Julia threads.

- **Type stability of the kernel (2026-11-10).**  Commit `037b464` (C6 kernel) made every run 5×
  slower: `SimParams.kernel` was typed `AbstractKernel`, and with a fourth kernel method Julia
  stopped union-splitting, so each kernel call in the neighbour loops became a dynamic
  dispatch with an allocation; with several realizations in one process the GC stalls them all.
  Affected: rigidity job 61046892 (C4/C6), campaign `energystep` (61067905, 61073581), and the
  debug jobs since 2026-11-09 11:00 — results unaffected, cost only.  Fixed by `SimParams{K}`
  (bit-identical, `testregress_step`).  Smoothed, L = 1.2, 4 seeds on 8 threads to t = 10:
  62 s before `037b464`, 315 s after, 64 s fixed (energy 59 s).  Check speed after touching
  types in `sph.jl`.
- **Threads vs realizations.**  One realization scales poorly beyond 4 threads (energy, 231
  steps: L = 1.2 13/5/4 s and L = 2.4 52/17/12 s at 1/4/16 threads); realizations side by side
  scale almost perfectly.  Use 1–4 threads per realization and many realizations per node.

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
- Output of `equilibrium.jl` is HDF5 (`equilibrium_<tag>_r<seed>.h5`):
  datasets `ks t ML MT S H KE`, parameters as root attributes (incl. `commit`).  Read
  with `read_equilibrium` (`src/measure.jl`) or h5py (arrays appear as (nt, nk)).
  `stability.jl` still writes `.jls` (Julia `serialize`; needs the same
  `src/` definitions to load).
- **Throughput in `shared`: `MaxJobsAccruePU = 2`.**  Only two pending tasks gain
  priority at a time; array tasks start in pairs about every 5 min (≈ 24 per hour),
  whatever the `%` limit.  176 tasks take ≈ 7 h just to start.  For many long tasks,
  consider packing them into whole-node jobs (64 two-thread tasks per node) instead.
- Recording interval: the sound period at k h = 1.5 is 2π/(c_s k) ≈ 0.36; keep
  `rec_every × Δt ≲ 0.05` (Δt = 8.66e-3 at h/a = 2) if C_L is wanted at large k.
- Cost model: wall ≈ c × N × steps, steps = t_end / 8.66e-3 at h/a = 2.

## The code after the cleanup (2026-11-12)

All campaigns before this date are finished and their runs will be discarded (the figures in
`docs/figs` stay).  The code was reorganized without any change to the numerics
(`docs/codeplan.tex`; bit-identical against the old `main`):
- Drivers renamed: `examples/stochrep_equilibrium.jl` → `examples/equilibrium.jl`,
  `examples/stochrep_hamiltonian.jl` → `examples/stability.jl`.  The old campaign files
  (`campaign_*.txt`) keep the old names; new campaign files use the new ones.
- Library: scripts include `src/load.jl`; the stochastic step is `stochastic_step!` in
  `src/stochastic.jl`; superseded code in `src/legacy/`.  Settings go through `setting()`;
  `SPH_DISSIPATION=changerep` is now `particle` (the old value still works).
- Run record `<run>.toml` (above) instead of `.json`/`.csv`.
- Tests: `test/run.sh all` (unit, physics, load, regression against `main`);
  `nersc/unittests.sh` runs it on one debug node.  Run it once at NERSC before the first
  new campaign.
- Scripts behind the older notes: git tag `notes-2026-11`.

## Campaigns

### `grid` — the grid step (gridstep.tex), 2026-11-12, branch `gridcode`

**Option (2026-11-13):** the grid step is `SimParams.ideal = :grid`, set by `SPH_IDEAL=grid` (default
`particle`), recorded in `[sim]` and `[settings]` of the run record.  The campaign files of 2026-11-12 used
`SR_GRID=1`, which now stops the run with a message: use `SPH_IDEAL=grid`.  `examples/equilibrium.jl` only.

`src/grid.jl`, `SR_GRID=1` in `examples/equilibrium.jl`.  Checkouts `~/stochastic-sph-run{,2,3,4}` (worktrees
at gridcode commits).  Deterministic tests: `examples/grid_waves.jl` (debug 61209546 ideal, 15 min;
viscous full set timed out 61209692, quick set 61211081), `examples/grid_gaussian.jl` (61209795),
smoke test `$SCRATCH/sph/grid_smoke` (61209993).  Kernel comparison C4/C6: 61211910/1.
- Campaign `nersc/campaign_grid.txt` (the boost sets with SR_GRID=1, + L = 0.6): job **61209985**, 3 nodes,
  12.5 min, 72/72 runs, no grid-step failure.  Output `$SCRATCH/sph/grid/`.
- Outcome: the excess shear damping with flow and the equal-time deficit along the flow of boost.tex are
  gone; ΔH/H ~1e-5 (particles 1e-3); phase errors κ(k)/k − 1 ~ (kh)² from the kernel (accepted).
- Lesson: debug jobs with Julia print tables at the end only when stdout is flushed; read the .toml.
- Longer runs (L = 2.4, v = 0.8 to t = 600): job **61212123** (1 h, 2 nodes).  Kernel C4 with the particle step
  (`nersc/campaign_kernel.txt`): job **61212224** — helps a little, not a fix.  Full-zone spectra
  (`nersc/campaign_grid_spectra.txt`): job **61213116** (debug).  Test suite on 4d5d9c2: job 61212619 (all pass).
- High statistics (`nersc/campaign_grid_stats.txt`, t = 1000, 64 runs per v at L = 1.2, 32 at L = 2.4): jobs
  **61212171** (2 nodes, 2.3 h) and **61212170** (3 nodes, 3.5 h), 352/352 complete, output `$SCRATCH/sph/grid_stats/`.
  Shear damping at the hydrodynamic rate at every v to a few percent; a uniform 2-7% long-wave excess in the
  equal-time values (also in the full-zone spectra); the near-null modes of the stencil poorly equilibrated.

### `boost` — moving fluid, 2026-11-11, `nersc/campaign_boost.txt`, docs/boost.tex

`SR_BOOST=v` (fac262c): uniform flow along x at rest-frame S0 = 1e6, equilibrium start at the
moving β^μ, E_k recorded.  Output `$SCRATCH/sph/boost/`.  Analysis `examples/boost_analysis.jl`.
- L = 1.2, v = 0, 0.3, 0.6, 0.8 × 16 (+4 at v = 0.8, Δt/2), t = 300: jobs **61156299** (tasks
  1-34) and **61156300** (35-68), checkout `~/stochastic-sph-run`, THREADS=4, -t 3 h.
- L = 2.4, v = 0, 0.6, 0.8 × 8, t = 300: job **61156353** (tasks 69-92), checkout
  `~/stochastic-sph-run2`, -t 8 h.
- v = 0.8 statistics: job **61158254** (tasks 93-124: 16 runs to t = 600, 16 at Δt/2), checkout
  `~/stochastic-sph-run`.  All runs of 61156299/300/353 and 61158254 completed, none broke.
- Outcome so far (docs/boost.pdf): susceptibilities, advection, Doppler-shifted sound, time-dilated
  damping all as predicted; but an excess shear damping with flow (10-20% at v ≤ 0.6, 40-50% along
  the flow at v = 0.8, at kh = 0.26 and 0.52), not removed by Δt/2.  Fit the shear-mode projection
  (left eigenvector of K), not C_xx, across the flow.
- Follow-up tasks 125-220 (η/s = 0.1, 0.4 and SR_NOVISC=1 at v = 0, 0.8): is the excess from the
  viscous or the ideal step?  Job **61160802** (17 min, 3 nodes); the v = 0 sets lacked SR_RECORD_E,
  rerun as tasks 221-268, job **61161590** (12 min, 2 nodes).  Lesson: v = 0 sets need SR_RECORD_E=1.
- Outcome: along the flow at v = 0.8 the excess is η-independent, ≈ 0.35 νk²/γ³ (η/s = 0.1, 0.2, 0.4)
  and equals the ideal-step-only rate: a numerical damping of the moving fluid (~η/s 0.015 in the lab).
  No excess at rest with viscosity on.  Total boost campaign 3.8 node-hours (sacct).

### `cl` — continuum limit, 2026-11-10, `nersc/campaign_cl.txt`, docs/continuum_limit.tex

n-scheme, h = 0.05, S0 = 1e6, η/s = 0.2; s_a = 2165/(h/a)² lowered stage by stage, equilibrium
start.  Output `$SCRATCH/sph/cl/`.  Each stage one regular node, 3 h, `node.sh` with a task range.
- Benchmark debug 61110291 (cost per step ∝ (h/a)^4 or faster); short runs 61110825
  (`$SCRATCH/sph/cl_short`); Δt scan 61111926/7 (`$SCRATCH/sph/cl_dtscan`).
- Stage 1 (tasks 1-32, THREADS=4) job **61110829**, checkout `~/stochastic-sph-run` (632f29f code).
- Stages 2 (33-40, THREADS=16) **61113602** and 3 (41-44, THREADS=32) **61113633**, checkout
  `~/stochastic-sph-run2`.  Stages 4, 5 after stage 3 is seen to be sane; stage 6 not run (the
  equilibrium draw fails at s_a = 2.1).
- Stages 4 (45-48) **61116238** and 5 (49-52) **61116239**, -t 4:00:00, THREADS=32, checkout `~/stochastic-sph-run2`.
- Δt criterion: residual acceptance ≈ 0.75 (f = 1, 1/2, 1/4, 1/16 at h/a = 8, 12, 16, 23).
- Resumes: stage 3 61124065 (timeout) + 61128803; stage 4 61127404.  Stage 5 stopped at t = 0.5
  of 1 (4 h gave ~870 of 1848 steps at ~16 s/step).  Total ≈ 19 node-hours.
- **Outcome** (docs/continuum_limit.pdf; `nersc/runs/cl_runs.csv`): stable down to s_a = 4 with the
  Δt criterion; equilibrium start fails at s_a = 2.1; |v| ≥ 1 rejections are the real equilibrium
  tail; equal-time correlators fluid-like; KE above the fixed-volume one-particle value (up to 15%);
  D_self ∝ (h/a)² and the transverse damping rises (1.1 → 1.8 by h/a = 12): no continuum limit of the
  transport at fixed h.  Cost per step ∝ (h/a)^4 or faster (4.5 s at h/a = 12, 9 s at 16, 16 s at 23
  on 32 threads).

### Default scheme changed — 2026-11-10, `main` at `91d9e66`

With no environment variables the code now runs the **n-scheme**: `SPH_HAMILTONIAN=energy`,
`SPH_VOLUME=number`.  Old schemes: `SPH_HAMILTONIAN=smoothed` (with `SPH_VOLUME=entropy` implied),
or `SPH_HAMILTONIAN=energy SPH_VOLUME=entropy`.  Earlier campaign files that relied on the
default (first, second, rigidity, frozen `s_*`) must now set `SPH_HAMILTONIAN=smoothed` to be
reproduced.  Unit tests: debug job 61109430 (`nersc/unittests.sh`), all pass after pinning
`testchangerep` to smoothed (61109670).

### `volume_n` and the deterministic suite — 2026-11-10, `main` at `0d3e3e9` / `e87d5d5`

New option `SPH_VOLUME=number` (V_a = 1/n_a) with `SPH_HAMILTONIAN=energy`; the old volume
V_a = s_a/S_a is now called `entropy` (the default; renamed from `kernel` after these runs).
- Campaign `nersc/campaign_volume_n.txt` (100 runs) packed on the interactive node
  61106091 (`node.sh` with SLURM_JOB_ID set from the login node, THREADS=2), 21 min.  Run
  checkout `~/stochastic-sph-run` at `0d3e3e9`.  Output `$SCRATCH/sph/volume_n/`.
- **Outcome** (docs/volume_n.pdf, `examples/volume_n_analysis.jl`): the back-map loss is
  gone (ΔH/H per unit time −3.45e−5 → +3.1e−6, halving with Δt); KE/eq 0.985, 1.008, 1.020
  at δv = 0.043, 0.078, 0.136 (entropy volume: 0.986, 0.812, 0.606); C_T and D_rel unchanged;
  the spread of s_a stays near √(3/s_a), the volumes spread by 3% at δv = 0.043.
- Deterministic suite: debug job **61106930** (`nersc/dtests.sh`, checkout
  `~/stochastic-sph-dt` at `e87d5d5`), 24 runs, 17 min, all pass; docs/number_tests.pdf.
  Output `$SCRATCH/sph/number_tests/{s,k,n,sw}` (k = entropy volume, then called kernel).

### `frozen` — 2026-11-10, `nersc/campaign_frozen.txt`, `main` at `1ffc82f`, docs/frozen.tex

Do the particles rearrange?  Pair separation D²(r) (new `ParticleRecorder`) with C_T:
`:energy` at S0 = 1e6, 1e7, 1e8 and `smoothed` at 1e6; L = 1.2, h/a = 2, t = 200, 16 seeds each.
- Job **61090342**: one regular node, `nersc/node.sh` THREADS=4 (32 at a time), `-t 1:30:00`,
  64 tasks (one realization each).  Run checkout `~/stochastic-sph-run2` (detached at `1ffc82f`).
  Output `$SCRATCH/sph/frozen/`.  Analysis `examples/frozen_analysis.jl`.  ≤ 1.5 node-hours.
- **Outcome** (done 07:19, 64/64 in 22 min, ≈ 0.4 node-hours; docs/frozen.pdf): `:energy` is a solid
  at S0 = 1e8 (D² plateau 0.016a², c_t = 0.031 ≈ static 0.034) and a fluid at 1e7 and 1e6 (c_t ≈ 0);
  `smoothed` at 1e6 is a solid (D² ≈ 0.3a², creeping).  At 1e7 the pair separation matches
  fluctuating hydrodynamics (D_rel/hydro 2.3 → 1.0 from r = a to 8a); at 1e6 particles walk
  independently, 2–5× faster than hydro.  The H loss of :energy falls ≈ δv^4.7.

### `energy2` — 2026-11-10, `nersc/campaign_energy2_{small,large}.txt`, `main` at `3c5f387`

Follow-up of `energystep`, all with `SPH_HAMILTONIAN=energy`, the kernel-speed fix in.  Run
checkout `~/stochastic-sph-run` (detached at `3c5f387`).  Output `$SCRATCH/sph/energy2/`.
- Job **61086213** (small, 80 tasks, `-c 2 -t 3:00:00`): the loss of H against Δt (SR_DTFAC =
  1, 0.5, 0.25) and with positions frozen, L = 0.6, t = 100, 8 seeds each; the stability scan of
  `first` (S0 = 1e6, 3e5, 1e5, λ = 1 and ∞), 16 seeds each, t = 200.
- Job **61086214** (large, 16 tasks, `-c 8 -t 6:00:00`, 4 seeds per task): h/a = 2.5, 3 at L = 1.2
  (16 seeds, t = 200); L = 2.4, h/a = 2 (32 seeds, t = 400).
- Estimate ≈ 2 node-hours.
- **Outcome** (all 96 tasks done by 10:53, no failures; docs/campaigns.pdf, section
  "loss of H, stability, h/a"; `examples/energy2_analysis.jl`): the loss of H does not fall with
  Δt (3.45, 4.15, 4.64e-5 per unit time at Δt, Δt/2, Δt/4) and is the same with positions frozen
  (3.70e-5) — it is made by the random residual moves of the stochastic step (λ = ∞ loses 5000×
  less).  No blow-up to δv = 0.136, but the system cools (KE/eq 0.99, 0.81, 0.61; ΔH/H −0.8%,
  −15%, −31% at t = 200).  No transverse sound at h/a = 2.5, 3; at L = 2.4 C_T is the fluid one
  down to kh = 0.13 (C_T(0) ≈ 1, damping 0.92–1.14×), the smoothed small-k anomalies are gone.
  Run checkout `~/stochastic-sph-run` removed.

### `energystep` — 2026-11-09/10, branch `hamiltonian-energy-step`, done

- Run checkout `~/stochastic-sph-run5` (detached at `934aa86`).  Debug jobs 61055259 (t = 10,
  no crash) and **61055621** (L = 0.6, t = 200, 4 seeds: energy/smoothed, cold start and ideal
  step alone; `$SCRATCH/sph/energystep_debug2/`, `examples/energystep_equilibration.jl`).
- Campaign job **61067905** (`-q shared -c 8 -t 2:00:00`); tasks 1 and 3 timed out and resumed
  as **61073581**.  Output `$SCRATCH/sph/energystep/`.  Analysis `examples/energystep_compare.jl`
  against rigidity `ha2`.
- Outcome (docs/energystep.pdf, last section): the transverse oscillation is gone (c_t fit
  0.000–0.002, C_T ≈ e^{−νk²τ}, damping 1.0–1.3× input, C_T(k,0) 0.93–1.04); KE/eq 0.99, no
  heating; H falls linearly, −3.5e−5 per unit time, every seed.  Ideal step alone: KE/eq ≈ 0.67,
  no runaway (smoothed: 60–95).  Wall 4.5× longer than smoothed at L = 1.2, though 1.4×
  faster at t ≤ 10 on one node: the slowdown builds up during the run (being measured).

#### Original proposal

Note from the local session for the NERSC session.  Read `docs/energystep.pdf` first.
- **What it is.**  `SPH_HAMILTONIAN=energy` (default `smoothed`, so nothing else changes):
  H = Σ_a e_a, each particle's energy at its own momentum density m_a/V_a, V_a = s_a/S_a.
  ẋ_a = v_a (not smoothed), a pure SPH pressure force, a viscous step with no change of
  representation.  In `stochrep_step!` the gradient correction is off with `:energy` (an
  error if forced on); `stochrep_hamiltonian.jl` follows, output suffix `_energy`.
  `stochrep_equilibrium.jl` needs no change (it uses the defaults).
- **Checked locally.**  Hamilton's equations against finite differences of H (1e-6);
  momentum to 1e-16.  Deterministic shear wave (`examples/shearwave_decay.jl`,
  `shearwave_scan.jl`): the two Hamiltonians agree; the ideal step has the same transverse
  sound, c_t = 0.034 at h/a = 2 (C2), the static value of `rigidity.tex` — as expected,
  since H is the same function of the positions at m = 0.  It does **not** cure the
  rigidity by itself.
- **Stochastic step, one-step bookkeeping** (`examples/energystep_stoch_onestep.jl`,
  L = 0.6, h/a = 2, eq start, 20 half steps at fixed positions, 3 seeds):
  `:smoothed` (gradient map) ΔH/H per step rms 3.4e-5, random sign; `:energy` ΔH = Δ(Σe)
  exactly, rms 1.8e-7, but systematic: −1.8e-7 per half step (cooling, about −4e-5 per
  unit time).  Correlated (−0.8) with Σ(Teff−T)Δs but of opposite sign: not understood yet.
- **Proposed runs** (`nersc/campaign_energystep.txt`, 5 tasks, `-c 8`, the same settings
  as `rigidity` ha2 and nv_ha2): ≈ 0.15 node-hours, scaled from `rigidity` (28 tasks ≈
  0.8).  Compare C_T, C_L, ΔH/H(t), KE/eq and UT with ha2 / nv_ha2.  Ask the owner before
  submitting; run from a separate checkout of `hamiltonian-energy-step` at a fixed commit.
- **Next idea, not implemented** (`energystep.tex`, "volumes that add up to the area"):
  V_a = A s_a/S_a / Σ_c(s_c/S_c) makes Σ V_a = A, which removes the pressure-driven
  modulus G ~ p ε on a lattice; the force becomes the present one with p → p − p̄.  If it
  is implemented, the same two campaign lines with it are the test.

### `rigidity` — 2026-11-09, `nersc/campaign_rigidity.txt` (+ `_det`), branch `rigidity`

Where does the transverse rigidity come from (docs/campaigns.pdf)?  L = 1.2, t = 200, 16
realizations per set, displacements UL/UT recorded: h/a = 2, 2.5, 3; Wendland C4 at h/a = 2
(`SPH_KERNEL=wc4`); positions frozen (`SR_NOHAM=1`); λ = ∞ (`SR_LAMBDA=Inf`); ideal SPH alone
(`SR_NOVISC=1`, thermal start, L = 0.6, t = 50) at each h/a.  4 seeds per task
(`SR_SEEDSTRIDE=4`, nreal 4, `-c 8`).
- Job **61045626** (24 tasks, run checkout `~/stochastic-sph-run2` at `4663c3f`) and job
  **61045885** (λ = ∞, 4 tasks, `~/stochastic-sph-run3` at the next commit).  ≈ 0.8 node-hours.
  Output `$SCRATCH/sph/rigidity/`.  Analysis: `examples/rigidity_analysis.jl`.
- Job **61046892** (9 tasks, `~/stochastic-sph-run4`): C6 at h/a = 2, C4 at h/a = 2.5, ideal SPH C6.
- **Outcome** (all 37 tasks done 2026-11-09 19:35; docs/campaigns.pdf, "The rigidity campaign"):
  c_t (fit, kh = 0.37) = 0.022, 0.015, 0.012 at h/a = 2, 2.5, 3 (C2); C4 0.023, 0.016; C6 0.026;
  λ = ∞ 0.020; positions frozen 0.000–0.003 with the input viscosity.  Not the T = 0 phonon
  values (C4 lattice is unstable there).  Displacement estimate 25–35% below the fit.  Ideal
  SPH alone runs away at every h/a and kernel (KE/eq 40–90 by t = 50).  With
  `SPH_HAMILTONIAN=energy` the rigidity is gone (see `energystep`, branch `hamiltonian-energy-step`).
- Found while setting up (L = 0.6, interactive): ideal SPH alone does not hold the lattice
  (KE/eq 0.9 → 6.8 by t = 20 from the thermal start, H conserved); random positions release
  ≈ 400 T per particle (KE/eq → 420 by t = 5) and heat the full scheme (ΔH/H +0.64 by t = 10).

### `second` — 2026-11-09 (done 10:56, 64/64), `nersc/campaign_second.txt`, 64 tasks, ≈ 0.8 node-hours

The fast code (`main` at `f355cd0`: speedup merged after validation, equilibrium start,
defaults star / λ = 1 / hot start, checkpoints every 20 min).  eqL1.2 and eqL2.4, 32
realizations each, t = 400, SR_START=eq.  Compare with the same runs of `first`.

- Run checkout: the worktree **`~/stochastic-sph-run`**, a detached HEAD at `f355cd0`, edited
  by no one.  (`~/stochastic-sph` stays on `nextstage` until job 61035596 is done.)
- Submitted as job **61037106** (`-q shared -c 2 -t 8:00:00`, m3722).  Tasks 1–32 eqL1.2,
  33–64 eqL2.4.  Output `$SCRATCH/sph/second/`.  A task that times out leaves a `.chk` and
  a partial `.h5`; resubmitting the same array index resumes it.

**Outcome.**  All 64 completed (eqL2.4 ≈ 3.7 h per task).  Same as `first` within the spread:
ΔH/H(t=400) = 0.219 (L1.2), 0.221 (L2.4); KE/eq 1.31.  The transverse correlator is a damped
oscillation with c_t = 0.022 at all kh = 0.13–0.74 in both boxes; damping 2γ/νk² = 1.1–1.2 at
kh ≥ 0.5, rising to ≈ 2 at kh = 0.13 (extra rate ∝ k^{2/3}).  C_T(k,0) = 0.75 at kh = 0.13.
Report: `docs/campaigns.pdf` (figures `docs/figs/compare_*.png`, `transverse_*.png`).

**Validation of the fast code** (before `second`): 32 vs 32 realizations of the
equilibrium driver, L = 0.6, t = 50, S0 = 1e6, old (`nextstage`) vs new (`speedup`).  All
13 observables (KE/eq, ΔH/H, both acceptances, C_T(k,0), C_L(k,0), C_T(k,τ=5) in three
shells) agree within 1.3σ; the new code took 243 s against 502 s.  Speed per particle
and step at L = 2.4: 1 thread 52 → 27 μs, 2 threads 42 → 13, 8 threads 37 → 5.5.

**s1e5_hfix rerun (61035596)**: 12/16 failed again in `bond_weights_comoving` once the
lattice had melted (a particle with no neighbours; a superluminal smoothed velocity in
the comoving metric).  Recorded as: the energy fix does not prevent the blow-up at
δv = 0.14.  Not pursued further (owner, 2026-11-09).


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

**Outcome (2026-11-09 morning).**
- Stability scan complete except `s1e5_hfix`: all 16 tasks died with a SingularException in
  the 4×4 solve of `bond_weights_comoving` (melting lattice → nearly singular star).
  Fixed in `34c1c32` (fallback to Λ = 0); rerun as job **61035596**.
- eqL1.2: 32/32 done.  eqL2.4: only 12/32 done; 20 hit the 8 h limit (they needed more
  than 8 h on shared nodes, estimate was 5.8 h), and the .h5 is written only at the end,
  so they are lost.  To be rerun (see the speedup branch, validated first).
- Figures: `docs/figs/campaign_first_stability.png` (`examples/campaign_stability_plot.jl`),
  `docs/figs/correlators_L1.2.png` (`examples/stochrep_correlators.jl`).

**What the stability scan shows** (t = 200, 16 seeds per set, box 0.6, h/a = 2 unless noted):

| set | ΔH/H | KE/eq | disorder | melted |
|---|---|---|---|---|
| S0=1e6 (δv=0.043), λ=1 | +4.6e-2 | 1.06 | 0.31 a | 0/16 |
| S0=1e6, λ=∞ | +3.2e-3 | 0.05 | 0.19 a | 0/16 |
| S0=3e5 (δv=0.078), λ=1 | +0.47 | 1.66 | 0.83 a | 1/16 (t≈198) |
| S0=3e5, λ=∞ | +4.1e-2 | 0.06 | 0.38 a | 0/16 |
| S0=1e5 (δv=0.14), λ=1 | blow-up | 3e8 | 4.7 a | 16/16 (t≈28) |
| S0=1e5, λ=∞ | +0.40 | 0.09 | 0.93 a | 8/16 (t≈147) but no blow-up |
| S0=1e5, λ=1, h/a=3 | blow-up | 5e10 | 9.9 a | 16/16 (t≈11) |

- The heating grows about ×10–12 per factor 1.83 in δv, i.e. roughly as δv⁴, for λ = 1 and
  λ = ∞ alike; λ = 1 heats 10–14× faster than λ = ∞ at the same δv.
- The λ = 1 instability sets in between δv = 0.078 and 0.14 (h/a = 2).  λ = ∞ never blows
  up, even where its lattice disorders to ~1 a: the runaway goes through the lattice modes.
- h/a = 3 at the same S0 has more particles in the box, so a larger δv per particle: not a
  clean comparison at fixed δv.

**What the L = 1.2 correlators show** (32 realizations, t = 400, S0 = 1e6):
- Statics C_T(k,0), C_L(k,0) ≈ 1.04–1.10 at kh ≳ 0.5: 5–10% high, about what the slow
  heating does to wT over the run.
- **C_T is not diffusive: it is a damped oscillation** (transverse sound).  At kh = 0.26,
  0.37, 0.52 the first zero is at τ ≈ 11, 7.5, 5 and the first minimum (C ≈ −0.3) at τ ≈ 22,
  14.5, 10: ω ∝ k with c_t ≈ 0.029 (c_s = 0.577), i.e. a shear modulus G = w c_t² ≈ 0.003 p.
  The particle arrangement (disorder 0.3 a, never melts) responds as a weakly rigid
  solid.  The fitted "ν_fit/ν ≈ 2 at small k" is this oscillation, not a viscosity.
  Testing e^{−νk²τ} needs the lattice rigidity understood or removed first.

Next once `first` finishes: stability plots (`examples/stochrep_longruns_plot.jl`-style
over the seeds), `SR_DATADIR=$SCRATCH/sph/first/eqL1.2 julia --project
examples/stochrep_correlators.jl L1.2` (and L2.4, in an interactive job), then a short
note in `docs/` with the tables and figures.
