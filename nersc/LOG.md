# NERSC log

Running record of what has been set up and run at NERSC (Perlmutter CPU), newest first.
Agents: read this before submitting anything, and add an entry when you submit, finish,
or learn something about running the code here.

**2026-11-16 (NERSC): hastings_low, s_a = 4 done** (61376873, 37 min): 49/128 tasks stopped in HMC (eos_newton, see below); survivors at rest
7 nh / 10 h / 5 chains, at v0 = 0.8 14 / 14 / 13.  Indicative: at rest residual acceptance 5% (nh) / 0.8% (h), light cone 90% of residual and 60%
of stress proposals, shear Γ/Γ_hyd ≈ 2.6–2.7 in both, KE/2VwT ≈ 0.6; spectra / HMC: nh +3–9% at rest and M_T +6–11% at v0 = 0.8, h ≈ 1.
**Note on the new defaults (a82ccdc, moment + Hastings by default):** the hastings campaign files' "nh" lines set SPH_DISSIPATION=moment but
not SPH_HASTINGS=0 — run at or after a82ccdc they would get the correction.  Add SPH_HASTINGS=0 before reusing them.  (campaign_hastings_dt.txt,
9e4bf68, written but NOT run — cancelled before it started; it needs that fix too.)

**2026-11-16 (NERSC): hastings_low, s_a = 12 and 8 done** (61376871: 128/128; 61376872: 96/96 dynamics, 29/32 HMC chains).  docs/inverse_moment.tex,
Stage 2, new subsection "Lower entropy per particle".  With the correction the dynamics matches HMC down to s_a = 8; without it the excess at rest
is +15–21% at s_a = 12 and +5–12% at 8 (smaller: at s_a = 8 the light cone rejects 65% of the residual and 31% of the stress proposals).  The
method degrades below s_a ≈ 12 (ΔH/H ~1e-2, advection 0.77 instead of 0.80 at s_a = 8).
**New failure at s_a ≤ 8 (≈ 40% of the s_a = 4 tasks so far, 3 HMC chains at s_a = 8):** `DomainError with -Inf: sqrt` in Trel (src/sph.jl:204)
from eos_newton (:359) — the Newton iteration runs to e = −∞.  Call chain: solve_eos (sph.jl:375–401) ← hmc.jl:61 ← :78 ← :104 ← :116
(hmc_update!), from equilibrium.jl:185 (the HMC start) and hmc_positions.jl:44 (the chains).  Needs a guard there (reject the HMC step).

**2026-11-16 (NERSC): campaign `hastings_low` submitted** (owner: "let's go wild"): the hastings runs at s_a = 12, 8, 4 with the γ cap,
nersc/campaign_hastings_low.txt (77cf74f), 384 tasks, worktree ~/stochastic-sph-hs.  Three packed 4-node regular jobs: s_a = 12 **61376871**
(tasks 1–128, -t 1:30), s_a = 8 **61376872** (129–256, -t 3:00), s_a = 4 **61376873** (257–384, -t 5:00); ≈ 20 node-hours at most.  Smoke test
at s_a = 4, v0 = 0.8, with the correction: runs, but dt = 8e-4 (≈ 4× smaller than at s_a = 16) and 51503 light-cone rejections in 125
steps.  Analysis: HA_SAS=12,8,4 examples/hastings_analysis.jl; boost groups HS12, HS8, HS4.

**2026-11-16 (NERSC): s_a = 16 rerun with the γ cap (dfa4834)** — campaign_hastings_s16.txt, job 61372840 (4 debug nodes, 29 min, all 96
complete), and the s_a = 16 HMC chains, campaign_hastings_hq16.txt, job 61374810 (1 debug node, 7 min).  Pre-cap output kept as *_precap.
**Not bit-identical**, for the corrected runs too: the start already differs (the HMC start / equilibrium draw at s_a = 16 produced states
beyond γ = 100) and so does dt (3.1e-3 → 5.4e-3 at rest).  So the hs16 *with* runs did reach γ > 100 (at least at the start).  Results with
the cap: at rest without the correction 13–19% above HMC in M_L, M_T, E (16 runs, 5–18σ), with it 0.98–1.02; v0 = 0.8 both ≈ 1.  New: at
s_a = 16 at rest the corrected dynamics damps the shear mode ≈ 15% faster (1.17–1.24 vs 1.02–1.09).  Residual log-ratio tails remain (max
~1e13).  docs/inverse_moment.tex, Stage 2 NERSC section updated; Fig. "divided by HMC" added for s_a = 16 and 54 (examples/equaltime_over_hmc.jl).

**2026-11-15 (NERSC): campaign `hastings` done** (61364382, 8 nodes, 18 min ≈ 2.4 node-hours).  docs/inverse_moment.tex, new last section
"Stage 2: results from NERSC" (figures hastings_nersc_*).  Answer: at rest the correction is needed — without it the full dynamics is ≈ 5%
(s_a = 54) and ≈ 20% (s_a = 16) above HMC of the same model in ⟨|M_L|²⟩, ⟨|M_T|²⟩, ⟨|E|²⟩; with it 1.00 ± 0.01–0.02.  At v0 = 0.8 no bias
visible in either (errors 2–5%).
**Two things for the laptop:**
1. **Crash without the correction at s_a = 16** (20 of 256 tasks: hs16_nh_v0 15/16, hs16_nh_alone_v0 4/8, hs16_nh_v0.8 1/16):
   `DomainError with -6.66e-16: sqrt` in `sqrt_comoving` (src/stochastic.jl:34) called from `build_stars` (:109) — a particle at |v| = 1
   to roundoff.  Needs `sqrt(max(1 - v2, 0))` or a guard in build_stars; the same seeds would stop again, so not rerun.
   (logs: $SCRATCH/sph/hastings/slurm/61364382_{3..16,129..136}.out)
2. **The residual log ratio has enormous tails**: typical windows fine (median rms 1.6–4.7), but windows with rms > 100: 0.4% at s_a = 54
   rest, 2% at 16/0.8, and 83% at s_a = 16, v0 = 0 (median rms 1.7e3, max ~1e18) — near-singular residual covariances; the spectra still
   agree with HMC there, but the residual move is then decided by q'/q.  Worth a regularization?
Also: at s_a = 16, v0 = 0.8 the advected shear is far from hydro in both versions (Γ/Γ_hyd ≈ 3, Ω/k = 0.82): the dynamics, not Hastings.

**2026-11-15 (NERSC): moment_a and moment_b done.**  moment_a (61361957, 8 min): 64 runs per scheme at v0 = 0.8, L = 0.6 — the kh = 0.52
difference along the flow is 1.0σ (a fluctuation).  moment_b (61361958, 42 min, ≈ 2.8 node-hours): L = 1.2, 32 runs per set — common and
moment agree within the errors at kh = 0.26, 0.52, 0.79, at rest and at v0 = 0.8.  docs/inverse_moment.tex, Stage 1 NERSC section,
figures moment_nersc_A_*, moment_nersc_B_*.

**2026-11-15 (NERSC): campaign `hastings` submitted** as written: job **61364382** (8 regular nodes, node.sh THREADS=4, -t 01:30:00),
from a fresh worktree ~/stochastic-sph-hs at 927e035 (testhastings.jl PASS there).  Output $SCRATCH/sph/hastings.  moment_b (61361958)
still running from ~/stochastic-sph-mo-run.

## 2026-11-15 (late night, laptop): NEW DEFAULTS on `main` — moment + Hastings

`inverse-moment` is merged into `main`.  The defaults are now `SPH_DISSIPATION=moment` with the Hastings correction
(`SPH_HASTINGS` on wherever it applies) and `SPH_GAMMA_MAX=100`, besides the generalized leapfrog, the Liouville measure and the
HMC start.  Campaign files that spell out `SPH_DISSIPATION=moment` and `SPH_HASTINGS=1` are unaffected; campaigns relying on the
previous defaults must now set `SPH_DISSIPATION=common SPH_HASTINGS=0` to be reproduced (CLAUDE.md).  Running jobs are unaffected
(fixed worktrees).  The h5 of every default run now carries hast_lq_{stress,resid}_{mean,rms}.  Next physics: the renormalization
(docs/helfand.tex, the Helfand separation), first at s_a ≈ 100, v0 = 0.

## From the laptop (2026-11-15, late night): γ_max = 100 — rerun the crashed tasks; add Fig. 37

`SPH_GAMMA_MAX` (default 100, src/sph.jl): states with γ > 100 (|v|² > 1 − 1e-4) have no weight (particle_entropy), and the comoving
structures cap the velocity there.  Bit-identical below the cap (step.jl, the equilibrium driver: 44 datasets identical).  This fixes the
`sqrt_comoving` DomainError of the 20 hs16_nh tasks (on the laptop the s_a = 16 run at rest now completes).  The large residual log ratios
at s_a = 16, v0 = 0 remain (median rms 2.6e3): particles at γ ~ 10–100 are common there; the correction stays exact.  The owner: s_a ≈ 16
is the edge of the method.

To do (worktree updated to the pushed tip; a few node-minutes):
1. **Rerun the 20 crashed tasks** of campaign `hastings` (hs16_nh_v0: 15, hs16_nh_alone_v0: 4, hs16_nh_v0.8: 1) — delete their partial
   output or keep it apart, then the same node.sh line restricted to those tasks (node.sh takes [first last]; or a short campaign file
   with just those lines).  Note: the hs16 *with* runs were done before the cap; they need no rerun (none reached γ > 100?  check the
   max |v| in their logs and say so).
2. **Keep Fig. 36** (equal-time C_AA/χ, s_a = 16) **and add a figure next to it, in the same layout, divided by HMC instead of χ**
   (the owner's request), and the same for s_a = 54.
3. Update "Stage 2: results from NERSC" with the s_a = 16 row without the correction, and push.

## Next session at NERSC — campaign `hastings` (written 2026-11-15, night, on the laptop; branch `inverse-moment`)

**Question** (owner): is the Hastings correction needed in practice?  The stochastic step alone is biased at low s_a (up to +17% in
the momentum fluctuations at s_a = 16) and the correction removes that exactly (docs/inverse_moment.tex, Stage 2 results).  But in the
full dynamics the ideal step also moves energy and positions, so the size of the bias there is not known.  The reference is HMC of the
same model and measure (your remark), not χ.

**Campaign** `nersc/campaign_hastings.txt`: L = 0.6, h/a = 2, current defaults, s_a = 16 and 54, v0 = 0 and 0.8; per (s_a, v0): the
full step without (`hs<sa>_nh_v<v>`) and with (`hs<sa>_h_v<v>`) the correction, 16 runs each, shared seeds, t = 100; the stochastic
step alone, 8 runs each; HMC chains `hq<sa>_v<v>`, 16 each, 300 updates (hmc_positions.jl now has SR_HMC_TAUJIT, nsteps = 0 for
τ = L/c_s, SR_RECORD_E).  256 tasks of 4 cores, packed on 8 nodes, ≈ 8–12 node-hours (within the 20 node-hour rule):
```bash
R=$HOME/stochastic-sph-mo      # update this worktree to the pushed tip of inverse-moment (moment_b must be finished, else a new one)
julia --project=nersc $R/test/testhastings.jl      # smoke test (login node, ≈ 1 min)
cd $SCRATCH/sph
sbatch -A m3722 -q regular -C cpu -N 8 -t 01:30:00 --export=ALL,SR_REPO=$R,THREADS=4 $R/nersc/node.sh $R/nersc/campaign_hastings.txt hastings
```
(or an interactive allocation, `salloc -A m3722 -q interactive -C cpu -N 8 -t 01:30:00`, then the node.sh line with bash).  Unfinished
tasks resume from .chk with the same command.  Note: the equilibrium start at s_a = 16, v0 = 0.8 could draw a particle beyond the light
cone and stopped with "unphysical draw" before this commit (src/measure.jl now redraws such particles; unchanged otherwise).

**Analysis** (plotting environment):
```bash
julia --project=$HOME/stochastic-sph $R/examples/hastings_analysis.jl $SCRATCH/sph/hastings     # spectra / HMC, table and figure
BOOST_GROUP=... (as for MO) for the shear damping if wanted; the h5 of the hs*_h_* runs carry hast_lq_{stress,resid}_{mean,rms}
```
Report in docs/inverse_moment.tex as a new last section "Stage 2: results from NERSC": the spectra ⟨|M_L|²⟩, ⟨|M_T|²⟩, ⟨|E|²⟩ of nh and h
against HMC per kh band; the acceptances; the log ratios; the shear damping nh vs h.  The answer to the owner's question in one line.

## Earlier (2026-11-15, night): SPH_HASTINGS available — superseded by the campaign above

`SPH_HASTINGS=1` (with `SPH_DISSIPATION=moment`) adds the Hastings correction to the stochastic step (docs/inverse_moment.tex,
"Stage 2" and its results).  Tested on the laptop: test/testhastings.jl (the ratios against direct densities), and the stochastic step
alone samples the exact distribution with Einstein and with Liouville, s_a = 16 and 54, v0 = 0 and 0.8, where without it the bias is up
to 17%.  Price: the residual acceptance falls (0.68 → 0.22 at s_a = 16, v0 = 0.8, Δt/2).  Recorded per window in the h5:
hast_lq_{stress,resid}_{mean,rms}.  A campaign with the HMC reference at s_a = 16 and 54 (the owner's request: compare to HMC of the
same model and measure, as you also suggested) will be written on the laptop; wait for it.

## Remarks for the Hastings work (NERSC session, 2026-11-15, evening)

The owner is starting on the Hastings correction (stage 2).  Two remarks from today's runs:
1. **Keep the HMC offset apart from the stochastic-step bias.**  Pure hybrid Monte Carlo, with no stochastic step at all, puts the
   equal-time momentum correlator above the continuum susceptibility by ≈ 8.5/s_a: C_MM/χ_MM = 1.16 (s_a = 54), 1.06 (135), 1.02 (541),
   flat in k and equal for the longitudinal and transverse components (hmc_eq chains, Einstein measure; docs/equilibrium_positions.tex,
   "The HMC equilibrium against the susceptibilities", branch `equilibrium`).  The measured bias of the stochastic step (+6% at s_a = 54,
   +14% at 16) is a different effect, but both raise C_MM at low s_a.  The clean reference for "exact" at finite s_a is the HMC value of
   the same model (and measure), not χ.
2. **`moment` is a safe base.**  Campaign `moment` (96 runs): acceptances identical to 1e-3, no guard fired, kinetic energy and every
   equal-time correlator, shear and sound at rest and at v0 = 0.8 equal to `common` within the statistics (docs/inverse_moment.tex,
   "Stage 1: results from NERSC").  So the fixed faces needed for the Hastings ratio cost nothing in the dynamics so far.  Running now:
   `moment_a` (v0 = 0.8, L = 0.6, 64 runs per scheme; job 61361957) and `moment_b` (L = 1.2, v0 = 0 and 0.8, 32 runs per set; job
   61361958), from ~/stochastic-sph-mo-run at 2b72675; results will be pushed into the same section.

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

### S-scheme measure form and the positional counterterm — 2026-11-25, branch `sscheme-measure` (docs/sscheme_plan.tex, "Level 1, amended", level 2; docs/split_dynamics.tex, last section)

All on interactive nodes with nersc/node.sh (salloc ... bash node.sh <campaign> <name>; at most 2 interactive jobs per user, up to 4 nodes each).
- `eos_measure` (job 62015696, c0ea823, 144 tasks, 2 nodes, 20 min): the measure form at rest, beta = D/2+1 and beta = b.  = split at c_s^2 = 1/2 and with beta = b.
- `eos_positional` (62017396, 67471dc, 192 tasks; tasks 107, 114 rerun at 4dda501): equilibrium with omega (SPH_POSITIONAL=1).  The positional error is gone at s_a >= 54.
- `hastings_positional` (62017421, 67471dc, 160 tasks, 4 nodes; tasks 8, 45 rerun at 4dda501): dynamics with omega at rest, and the approach from the lattice.
- `hastings_compare` (62019278, d3aacca, 384 tasks, 4 nodes, 35 min) + `hastings_compare_sp` (62020908, d7d26ad, 128 tasks, 15 min): split, split+omega, S-scheme,
  S-scheme+omega with the seeds of hastings_bath.  The split reference was rerun at the present code, because hastings_bath's split (3262b00) records 10-300x more change in H.
  CORRECTION (2026-11-26, debug 62046403, \$SCRATCH/sph/bisect_drift): that was the record, not the dynamics. One seed (hs16_sp_v0.8 r601) at
  every commit 3262b00..d7d26ad (worktrees ~/bisect/<commit>) reproduces both recorded dH/H (+1.99e-4, -1.48e-6). It changes at three commits:
  ab44bae (+1.14e-3) and 2bc03dd (-2.07e-4) change the start (nominal-bath T_rel 139.69 -> 140.38 from the EOS, KE normalization), so the
  trajectory and acceptance counts change too; 3fa7a3e (-1.48e-6) changes the record only, the trajectory bit-identical (all counts equal). Before 3fa7a3e the full-step record of H was sum s_a (H/S)_a with the labels after the stochastic step and H/S
  from before it (no solve in between); since then sum H_a. KE/eq likewise (V = s/S). Full-step dH/H and KE records before 3fa7a3e are
  wrong at the 1e-4..1e-3 level; runs with SR_NOVISC or SR_NOHAM resmooth before the record and are not affected.
- Lessons:
  - srun --overlap steps from the login node into one allocation all bind to the same few cores (32 runs at 25% of a CPU each).  Start many runs in ONE step
    (srun -c 256 --cpu-bind=none bash script-that-backgrounds-them), or use node.sh.
  - pdflatex is not on the default path: module load texlive/2024.
  - The table of I_0 must reach sigma -> 0, because split's vote Newton steps try labels u -> 0; it now starts at 1e-8.
- Node-hours this session: about 36 of the 100 granted.

### `eos_legendre` — 2026-11-22, the renormalized Hamiltonian as a Legendre transform — TO DO (note for the NERSC session)

**Plan: docs/legendre_plan.tex (read it first; it is self-contained).**  Implement `SPH_RENORM=legendre`: the particle Hamiltonian is
the Legendre transform of F_want = F_t + T δlnZ_1, for ConstantCs2EOS (any c_s^2), d = 2, the n-scheme.  Only the two particle-level maps
change (forward (s, m, V) -> state; inverse (e, m, V) -> entropy), each a 2x2 Newton solve with an analytic Jacobian and warm starts.
The owner wants, when back: the option implemented, tested (bit-identical with the switch off), timed, and demonstrated by repeating
the `eos_renorm` measurements at rest (c_s^2 = 1/2 and 1/4) and the same at v = 0.8 with the momentum and R = M/(v(E + T^yy)) - 1, and the full susceptibility matrix (dE/dT, dM/dT, dE/dv, dM/dv
from the canonical covariances of the total E and P; HMC unchanged, eos_measure.jl records the total momentum; plan Sect. 4).
Order of work (plan sections): A routing refactor (bit-identical), B src/renorm.jl, C the switch, D tests, E speed, F smoke runs,
then the campaign `eos_legendre` (~430 tasks, ~3 node-hours shared; recompute from the smoke runs and record here before submitting).
Branch: **`legendre`** (created by the owner's laptop session from `statics`, with the plan; `git fetch && git checkout legendre`).
Reference numbers: examples/renorm_quadrature_legendre.py.  Report in docs/legendre_plan.tex, "Results", and here.
**Owner's authorization (2026-11-22): work on this for the next two days; commit and push to `legendre` as results come in**
(code, tests, campaign files, figures, csv, notes; never .h5/.jls).  Budget for the campaign as in the plan (~3 node-hours; ask
before going over 20).

**Progress (NERSC session, 2026-11-22/23):**
- A (fe1ca71): the particle-level maps go through `particle_forward` / `particle_inverse` (src/pointwise.jl; target: `target_forward`,
  `target_inverse`, the arithmetic of solve_eos!/eos_from_energy).  Field-level uses left as they are: grid.jl (grid_forward/inverse),
  the starting guess of beta_from_averages, state_from_beta, particle_moments.  Regression against db139ef (debug 61821418 and the
  interactive node): bit-identical except the known hast_lq_* datasets.
- B–D (6b7a207, 186124b): src/renorm.jl — Newton in (ln T_rel, rapidity), analytic Jacobian, warm starts from pa.e (the target's rest
  energy density at T_rel) and pa.γ, cold starts by a fixed point on the shifted entropy/energy (contracts on the physical branch);
  counters legendre_counts() (solves, iterations, failures = no convergence, no state, cold).  Switch `SPH_RENORM=legendre`
  (sph.jl LEGENDRE, read through settings.jl setting_peek so that the record of SPH_RENORM is unchanged without it); check_legendre.
  test/testlegendre.jl: reference values to 3e-11 (the table's digits), Legendre identities to 1e-9, forward∘inverse to 1e-14 (incl.
  labels s < 0), counterterm off = the target to its own 1e-12, warm = cold, HMC energy error ∝ δt² (ratio 16.1), all pass.
- Lessons: (1) the label s = target entropy − (Δ + k) is often ≤ 0 at s_a ≲ 50 (Δ + k ≈ 25 at c_s² = 1/2, ≈ 45 at 1/4; the canonical
  spread of s is ±20 at c_s² = 1/4): s ≤ 0 and H ≤ 0 are legitimate states (the physical branch is det ∂(s,|m|)/∂(T,v) > 0, |m| ≥ 0);
  (2) HMC rejects a trajectory that reaches a state with no H (hmc.jl, LEGENDRE only); (3) a non-converged inverse with e < |m| is
  counted as "no state" (any Legendre state has H − |m| > −kT; these are 7σ momentum tails of the heat-bath proposals); (4) in motion
  the bath is the rest start's T_rel moving at SR_V (examples/eos_measure.jl), else σ moves with v; (5) beta_from_averages (measure.jl)
  halves a Newton step that leaves the light cone (never triggered before; needed in motion with the Legendre H).
- Failures (no convergence) in HMC: ≤ 3e-7 of the solves (c_s² = 1/4, s_a = 16, v = 0.8 the worst), all inverse, at extreme proposals.
- E, speed (debug 61824228, 8 threads, N = 9752, s_a = 54, c_s² = 1/2): ideal step (generalized leapfrog) 49 ms tree, 39 ms Legendre
  (forward solves 0–1 iterations warm: no cost); stochastic step 36 → 143 ms (1.25 M inverse solves per step, cold, 2.5 iterations,
  ~0.9 µs each against ~0.23 µs for the target): the votes need warm starts (to do); HMC update at N = 360: 0.32 → 0.37 s.
- F, smoke (interactive node 61821858, L = 0.45, h/a = 2, s_a = 54, 200 updates, 2 seeds): c_s² = 1/2 Legendre E/e−1 = −0.010,
  P/p−1 = −0.032 (σ = 22.7) against SPH_RENORM=1 −0.015, −0.039 (σ = 18.0): within the single-particle residual, both dominated
  by the positions; acceptance 0.985–0.995 for all; tree 0.134, 0.106.  Dynamics (stability.jl, S0 = 1e5, t = 20): ΔH/H = +4.8e-5
  Legendre, +4.2e-5 SPH_RENORM=1.
- Campaign file `nersc/campaign_eos_legendre.txt` (432 tasks, aa85f5c); smoke test of two tasks (60 updates): debug job 61824455 —
  eosL_leg_c025_s16_ha2_v08 seed 0 stuck (a cold inverse at v ≈ 0.997 stalled under step halving; one NaN particle then kept the
  heat bath from moving it): fixed in 3c6fd46 (full Newton steps unless the residual grows fourfold; 50 cold iterations; the heat bath
  leaves a state with no weight).  Again, debug job 61824956: both COMPLETED, acceptance 0.98 and 1.0, no failures; 0.75 s per update
  at h/a = 2 (N = 360), 3.5 s at h/a = 3 (N = 864), Legendre, 2 threads.
- Unit tests with the switch on (186124b): 14 pass; 9 fail by design — testeos, testeos2d, testeos_consistency, testchangerep (they
  compare the particle maps with the target's closed forms), testeqstart (the target's most probable state), testhastings (Einstein
  measure, refused), testdims (a different refusal message), teststochastic_guard (calls eos_from_energy directly on particle states;
  a Legendre particle may have e < |m|), testparticle_sweep (the legacy two-level dissipation); every script loads.  With the switch
  off: all unit, physics and load tests pass; regression against db139ef bit-identical except hast_lq_*.
- **Campaign submitted**: job **61825274** (432 tasks, shared, -c 2, -t 2 h), run checkout `~/stochastic-sph-legR` (worktree at
  3c6fd46), output `$SCRATCH/sph/eos_legendre/`.  Estimate 216 × 5 min + 216 × 19 min on 2 threads ≈ 1.4 node-hours.
- Clean switch-off regression of b474aef against db139ef (job 61825713, shared -c 64, 38 min): step.jl and stability drivers
  IDENTICAL; equilibrium drivers differ only in hast_lq_* (known, thread order); unit-test output differs only in testcheckpoint
  (wall-clock seconds) and testrunrecord (src/renorm.jl in the file list).  The suite reports FAIL for these reasons only.
- **Solver failures in the campaign** (2026-11-23): eosL_leg_c025_s16_ha2_v0 had E = NaN on ~1% of the configurations (all 8 seeds;
  acceptance 0.91), ~10 failed forward solves per seed; a few also at c_s² = 1/2, s_a = 16, at rest.  Reproduced (debug 61829998): the
  failing particles (labels s = −3.7 to −4.4, |m| = 7–13, γ = 12–41) have a physical state, but Newton from the warm and the cold start
  converged to the unphysical branch, and a warm start that converged to the wrong branch was not retried; an HMC update that then
  rejects restores the particle with no state.  Fixed in **b351f3a**: retry cold on a wrong-branch convergence (forward and inverse),
  and as the last resort bracket the physical branch (legendre_bracket_forward; ~0.6 ms, rarely called); testlegendre 5b with the three
  states; all pass (debug 61830861).  Only renorm.jl changed (switch-off numerics untouched).
- **Rerun**: pending/running c_s² = 1/4 Legendre tasks of 61825274 cancelled (array 289–384); old output of the 14 affected lines in
  `$SCRATCH/sph/eos_legendre/old_3c6fd46/`; `nersc/campaign_eos_legendre_rerun.txt` (14 lines, 112 tasks) submitted from worktree
  `~/stochastic-sph-legS` (b351f3a): job **61831354**, shared, -t 2 h (~0.7 node-hours).
- **Rerun 61831354 done (112/112)**, but c_s² = 1/4, s_a = 16 still had E = NaN, now mostly in motion (3–8% of the configurations at
  v = 0.8; acceptance 0.78–0.90).  Reproduced with b351f3a (debug 61836018, 61836717, the heat bath instrumented): the heat bath accepted
  proposals at e < 0 (e = −53, |m| = 31) for which the inverse, from the guess, converged to a point with det > 0 and |m| ≥ 0 but **v < 0**
  (the velocity against the momentum, |m|/v < 0): a label (s = −5.5 to −8.7) whose (s, m, V) has no state with γ ≤ 100.  Fixed in
  **34b8161**: the physical branch also requires v ≥ 0 (`legendre_physical`); test 5c.  Check (debug 61837367): testlegendre all pass;
  that line, 4 seeds × 300 updates: no NaN, acceptance 0.98–1.0 (was 0.90).  Such accepted proposals could also bias the other Legendre
  runs silently (a wrong weight where some other state exists), so **every Legendre line is rerun**: `nersc/campaign_eos_legendre_rerun2.txt`
  (24 lines, 192 tasks), worktree `~/stochastic-sph-legV` (34b8161), job **61838799**, shared, -t 2 h (~1.2 node-hours).  Earlier Legendre
  output moved to `$SCRATCH/sph/eos_legendre/old_legendre_v1/` (and `old_3c6fd46/`); the tree and SPH_RENORM=1 runs stand.
  2026-11-23 05:10: the shared queue ran only 2 tasks at a time (16 done in 3 h, ~29 h to go), so tasks 21–192 were held and packed
  on 3 regular nodes, `node.sh` THREADS=2 (job 61851444); the held array tasks were cancelled when it started.  It failed in 43 s: the
  campaign file is not in the run worktree (committed after 34b8161; give node.sh the main checkout's copy, as the array had it).
  Resubmitted as job **61852069** (-t 1:15, ≤ 3.75 node-hours), tasks 21–192; tasks 1–20 finished on shared.
- **eos_legendre done (2026-11-23 05:40).**  61852069: 172 tasks in 18 min on 3 nodes (0.92 node-hours).  Campaign total ≈ 2.1 node-hours
  (61825274 0.88, 61831354 0.24, 61838799 0.04, 61851444 0.04 (failed at start), 61852069 0.92).  All 24 Legendre lines have 8 seeds; 425 Newton
  failures in 4.6e10 solves (415 at c_s² = 1/4, s_a = 16); 4 kept configurations with E = NaN (c_s² = 1/4, s_a = 16, h/a = 3), which make the
  per-seed summary NaN: those two lines average 6 and 7 seeds.  Lesson: node.sh needs the campaign file of the main checkout when the run worktree
  predates it; the shared QOS can sit at 2 running tasks (packing 3 regular nodes took 18 min including the queue wait of 10).
  Results in docs/legendre_plan.tex, "The campaign" (figures docs/figs/legendre_*.png, table docs/figs/legendre_table.csv).  In short: the Legendre
  H removes the O(1/s) of the tree level; its single particle keeps a large O(1/s²) (E/e−1 = +0.03 at s = 28, c_s² = 1/2); at s_a = 541 the
  positions' part is the same for Legendre and SPH_RENORM=1 (E/e−1 −0.004 at h/a = 2, ∝ 1/(n̄h²)) and includes R ≈ +0.0008 in motion; at small s
  it grows and is not additive (R at s_a = 16 is the Legendre single particle's, +0.0032, and SPH_RENORM=1's is +0.0004 instead of one loop −0.006);
  c_s² = 1/4, s_a = 16, v = 0.8: E/e−1 ≈ +0.01–0.03 against the quadrature's +0.14, not understood; χ: errors 0.02–0.05, a positions' part ≈ −0.04
  (h/a = 2), −0.01 (h/a = 3) at s_a = 541.  Open for the owner: H two-valued at s < −5.65 (c_s² = 1/2).
- **Found while at it — the Legendre H is two-valued at very negative labels** (c_s² = 1/2): for s < −5.65 (independent of V) the curve
  s = const in (T_rel, rapidity) has two segments with det ∂(s,|m|)/∂(T,v) > 0 separated by a fold (det < 0, rapidity ≈ 1.2–1.8), and their
  |m| ranges overlap: two physical states (a hot slow one and a cold fast one, γ ~ 15–65) with the same (s, |m|, V).  At c_s² = 1/4 it is
  single-valued down to s = −10.  The warm-started Newton follows the sheet it is on; the bracket picks the slow one.  Deep in the tail
  (|m| ≈ 85 at V = 5.6e-4 at the onset), so negligible for the campaign's averages, but H(s, m, V) is not a function there: for the owner.

### `hastings_split` — 2026-11-23, branch `split` (docs/split_dynamics.tex)
The dynamics (moment + Hastings, the defaults) at the tree level (SPH_RENORM=0, "tr") and with SPH_RENORM=split ("sp"), each against HMC
of its own Hamiltonian; nersc/campaign_hastings_split.txt, s_a = 16, 54, v0 = 0, 0.8, 16 seeds, t = 100 / 300 HMC updates, conformal
c_s^2 = 1/2.  **The old `hastings` runs had c_s^2 = 1/3** (cs2_val = 0.333 in every .toml of $SCRATCH/sph/hastings: 927e035, e282211,
77cf74f, before the dimfix 8d50d04 made d = d_s), so they are not a reference; the tree lines are rerun here at c_s^2 = 1/2 (owner:
"the old jobs did not have any renormalization ... we should know what was run").  256 tasks × 4 cores, 8 regular nodes, one wave;
estimate ≈ 2–4 node-hours (smoke: split ≈ 2× the tree's cost per step), at most 8 (-t 1:00).  **Submitted** job 61879905 (8 regular nodes) from the pinned worktree ~/stochastic-sph-splitH (8e44845); output $SCRATCH/sph/hastings_split.
- Smoke (debug 61879181, s_a = 16, t = 3, HMC start of 50 updates; 20 split HMC updates): all four ran, no failures.  v0 = 0: stress /
  residual acceptance 0.93 / 0.51, ΔH/H = +3.6e-3 (short run, short HMC start); v0 = 0.8: 0.985 / 0.76, ΔH/H = −1.6e-4.  HMC: acceptance
  1.0, |ΔH/H| 5e-6 (v0 = 0), 4e-7 (0.8).

- **Done** (61879905, 8 nodes, 14 min ≈ 1.9 node-hours): 256/256 complete, cs2_val = 0.5 everywhere, SPH_RENORM recorded false / split.
  Split costs ≈ 1.6× the tree per run (rest 304 / 185 s, v0 = 0.8 488 / 289 s at s_a = 16).  **Dynamics against its own HMC: ≈ 1 for both
  Hamiltonians** (M_L, M_T, E at kh 0.52–1.56, within 2–5%): the Hastings correction works with the split Hamiltonian.
- **But the tree/split comparison is not at the same bath in motion.**  The HMC start (equilibrium_init!, beta_from_averages) fixes β^μ so
  that the single-particle canonical averages equal the set-up totals.  The set-up states are identical (E_tot 3.69e6 / 3.68e6 at s_a = 16,
  v0 = 0.8), but the bath is: tree T = 81.1, v = 0.794 (nominal 84.2, 0.8); split T = 158, v = 0.658.  Cause: particle_moments' quadrature
  (41³ points, ±9σ boxes along the axes) is under-resolved at s_a = 16 in motion: with nq = 81 the averages move (tree ⟨e⟩/ē − 1 +0.18 →
  +0.27, split +0.06 → −0.005), and d⟨e⟩/dθ⁰ differs from Cov(e, e) by up to 1.7 (split); the Newton iteration of beta_from_averages then
  oscillates and diverges for split (T 84 → 79 → 99 → 71 → 515 → …), and lands slightly off for the tree.  At rest split's bath is right
  (T = 139.5, nominal 140.4); the tree's is 120.5 (its O(1/s) error at fixed energy).  Pre-existing (not split code); every equilibrium.jl /
  hmc_positions.jl run with low s_a and v0 ≠ 0 has a bath from this solver.  The second draw (SR_START=eq after the HMC start) uses it too.
  So: split dynamics/HMC valid; split/tree at v0 = 0.8 not a comparison at the same (T, v); boost_analysis χ uses the nominal T, v.
- boost_analysis.jl now reads c_s² from the runs (was hard-wired to 1/3 and refused c_s² = 1/2; sound damping (1 − c_s²)νk²).
- **Final regressions at 9c01b6c** (61871467 switch off, all; 61871469 legendre): only the known differences — stability drivers IDENTICAL,
  equilibrium drivers only hast_lq_*, unit output only paths / wall clock, and testrunrecord now passes (one more setting, the
  renorm_split.jl line).  Split leaves the switch-off and SPH_RENORM=legendre paths bit-identical to 53d29e4.

- **The start was not at the requested state** (owner: "we are thermalizing via HMC where temperature and velocity are precisely
  defined").  HMC was given the β^μ solved from the set-up totals (`beta_from_averages`, a Newton solve on a 41³ grid of single-particle
  states), not the requested one.  From the run records ([start] T0, v0, E_total, M_total): split s_a = 16, v0 = 0.8 ran at T_rel = 208,
  v = 0.66 (70% more momentum than set up); split s_a = 54 at 293, 0.78; tree at 137–139, 0.784 (s_a = 16).  At rest all within ~1%
  of the totals.  The analysis compared with hydro at the set-up T_rel and the hard-wired v = 0.8: the split "×2.4 equal time" and
  "advection 0.66" were the wrong state, not split.  With the state from the conserved totals (boost_analysis, 3262b00): split v0 = 0.8
  equal time / χ = 1.0–1.15, Ω/k = 0.659 against hydro 0.658, Γ/hydro 3.6–4.2 (tree 3.6–4.8).
- **Fix (3262b00)**: `SR_BATH=nominal`, new default of equilibrium.jl and hmc_positions.jl: the HMC start and the draw at β⁰ = γ/T_rel0,
  β = γv/T_rel0 x̂ (T_rel0 of the rest state at S0).  `SR_BATH=totals` = before.  src default (bath = nothing) unchanged.  Smoke (debug
  61885718, s_a = 16): bath exactly 140.4, 0.800; the totals through the target EOS: split 139.4 / 142.6 (v0 0.8 / 0), tree 151.9 /
  155.5 — the tree's O(1/s) excess energy at a given T, removed by split.  boost_analysis uses the recorded bath when it is nominal.
- **Rerun** `hastings_bath` (nersc/campaign_hastings_bath.txt, same 256 tasks): job **61886625** (8 regular nodes, ≈ 2 node-hours) from
  ~/stochastic-sph-splitB (3262b00); tests (unit, load, regress against 86fbc35) debug 61886627.

- **hastings_bath done** (61886625, 8 nodes, 29 min ≈ 3.9 node-hours, 256/256).  Bath exactly as requested.  Dynamics / own HMC = 1
  within 2–5% for tree and split.  Equal time / χ(EOS at the bath): split 0.92–1.07, tree 1.2–1.4 (s_a = 16).  Tests (debug 61886627):
  unit, physics, load pass; equilibrium drivers differ (start changed, intended).  With SR_BATH=totals (debug 61888823) the drivers
  reproduce 86fbc35 except hast_lq_* (as before): the change is the start only.  (Both debug jobs hit the 30 min limit after these checks.)
- **The damping**: shear Γ/hydro 3–6 for tree and split at η/s = 0.2, c_s^2 = 1/2 (owner: "tree level was working pretty good").  The
  present analysis on the old c_s^2 = 1/3 hastings runs gives 0.94–1.23: not the analysis.  T_rel at the same S0 went 28 → 140
  (s = (4/3)T^3 → (3/2)T^2), so ν0 = (η/s)/T_rel is 5× smaller.  Campaign **damping_scan** (61889391, 2 nodes; tree at rest):
  η/s = 1.0 gives 1.09–1.20 × hydro; no viscous step gives 7.3–7.9 (s_a = 16), 5.8–6.2 (54) × the η/s = 0.2 rate, with H conserved to
  2e-6.  The decay is mode coupling to the fluctuations, not the viscous step: Γ = ν0 k^2 + (A/ν0) k^2, A fixed at η/s = 0.2 predicts 1.12
  at η/s = 1.0 (measured 1.09–1.16).  In d = 2 conformal A/ν0^2 ∝ 1/(η/s)^2, independent of s_a.  H(t) flat in all sets
  (examples/energy_time.jl).
- **cs13** (nersc/campaign_cs13.txt, 09c10bf: SR_CS2=1/3, SR_ZETAS=(η/s)/3, i.e. ζ_eff = η/3 as before 8d50d04): the hastings tree runs with
  the present code, job **61890225** (2 nodes) from ~/stochastic-sph-splitC13.

- **cs13 done** (61890225, 64/64): tree at c_s^2 = 1/3, ζ_eff = η/3 with the present code: shear Γ/hydro 0.93–0.98 (s_a = 16 rest),
  0.98–1.10 (54 rest), 1.01–1.26 (54, v = 0.8), as hastings (1.17–1.23, 0.94–1.10, 0.96–1.20).  The c_s^2 = 1/2 excess is the small ν0,
  not the code.  Equal time differs (cs13 1.5, hastings 0.85 at s_a = 16): hastings started at the right energy with a bath 14% cooler
  (T0 = 24.3 vs 28.1); (28.1/24.3)^4 = 1.79 vs measured 1.76; s_a = 54: 1.17 vs 1.18.  The tree's O(1/s) T–energy mismatch.

- **Default switch and merge (2026-11-23).**  SPH_RENORM=split is the default (SPH_RENORM=0 tree, legendre the alternative);
  unsupported schemes stop with a message.  Survey with split forced (61892182): 12 unit + 3 physics tests check tree-level machinery
  (d = 3, entropy volume, Einstein measure, target maps, waves at s_a ~ 1e-3 below the label boundary) — they now set SPH_RENORM=0
  themselves.  T_rel,0 of the nominal bath now from the EOS (trel_of_srel), not a set-up particle (Legendre's was T_rel ≈ 203; tree and
  split 140.437 vs exact 140.381).  SR_S0 retired → SR_SREL0 (hard stop; driver variable SREL0, record key srel0; srel0_of(m) reads
  either).  Full suite at the new defaults (61894594): unit, physics, load pass; regression (SPH_RENORM=0 SR_BATH=totals vs main 8e147db):
  step.jl and stability identical, equilibrium drivers only hast_lq_* (as before), unit output only HMC wall time (testcheckpoint) and
  the settings count / renorm files (testrunrecord).
- **Legendre dynamics smoke** (debug 61893274, then 61894564 with the EOS bath): runs, bath exact (T0 = 84.23 at v = 0.8), cost ≈ split.
  But at s_a = 16 at rest ΔH/H = +2.0e-2 in 5 time units, acceptances 0.83/0.45, 270k guard rejections (split: 3e-4, 0.97/0.58, 43k).
  Campaign hastings_legendre held until this is understood.
- **Legendre smoke at s_a = 54** (debug 61901626, `$SCRATCH/sph/lg_smoke54`, 5 time units): healthy.  At rest ΔH/H = +2.2e-4, acceptances
  0.987/0.880, 142 guard rejections, |v| > 0.9 in 1e-5 of samples (split at rest: −3.6e-4, 0.988/0.886, 2); v = 0.8: +2.1e-4, 0 rejections.
  The s_a = 16 trouble: the counterterm takes T(K − c_s²) = 7T (c_s² = 1/2) from the inertia |m|/v ≈ 16T at rest and 2.5T/V from the
  pressure, about half the tree values, so particles with low entropy content go to the light cone; at s_a = 54 it is ~14%.
  KE/eq at rest was 0.56 (s_a = 54) and 0.38 (s_a = 16) with Legendre: the normalization 2(N−1)VwT took w, T from a set-up particle,
  which under Legendre is hotter (wT 1.79×, 4.06×); corrected 1.00 (s_a = 54) and 1.05 → 1.55 (s_a = 16, heating).  Fixed in
  equilibrium.jl (SR_BATH=nominal: w0 and the KE normalization from the EOS at s_rel,0, T = T_rel,0/γ; e_of_srel in sph.jl); debug
  61903312, s_a = 54 at rest, 3 time units: KE/eq Legendre 1.001, split 1.011.
- **Campaign `hastings_legendre` dropped** (owner, 2026-11-25): superseded by the S-scheme (docs/renorm3.tex: the Legendre Hamiltonian moves
  the stationary point; docs/sscheme_plan.tex).  Never submitted; its campaign file was not committed.  The KE fix went to branch
  `sscheme-kefix` (merged into `sscheme`), since the S-scheme's set-up particle is likewise not at the target state.

### `eos_split` — 2026-11-23, branch `split` (from `legendre` at 53d29e4), docs/step5_plan.tex — Step 5, `SPH_RENORM=split`

Worktree `~/stochastic-sph-split`; output `$SCRATCH/sph/eos_split/`.  This section is appended to only (the later merge with `legendre`
is a concatenation).
- **A. ∂Δs_dyn/∂u** (examples/renorm_quadrature_split.py, new switch: model `step5-du0`, `SR_MODELS`; login node, ~1 min per c_s²): with
  and without the term T ∂Δs_dyn/∂u in E_dyn, s_a = 16, 54, 100, 200, 541, v = 0, 0.8.  Both residuals fall as 1/s²; the term changes the
  coefficient by up to ~2 either way (without it smaller at c_s² = 1/2 everywhere and at c_s² = 1/4 for s_a ≥ 100; with it 8× smaller at
  c_s² = 1/4, s_a = 54, v = 0.8, near a sign change of its residual).  Decision (agreed with the owner): drop it, E_dyn = T Δs_dyn —
  first derivatives of the target state only.
- **B–C.** `src/renorm_split.jl` (split_forward, split_inverse = Newton in u from the target's inversion, split_vote_entropy, check_split,
  V̄ = SPH_RENORM_VBAR, default L²/N); `SPH_RENORM=split` chosen at load time in sph.jl (`SPLIT`, `HAS_DOMAIN = LEGENDRE || SPLIT` for the
  rejections in hmc.jl), pointwise.jl, stochastic.jl.  Counters reported as legendre_* by eos_measure.jl, V̄ as split_vbar.
- **SPH_RENORM=1 at any c_s²** (owner, 2026-11-23): its conformal-only check removed (stochastic.jl, renorm_vote_entropy); its α/2 =
  (1 + d + 2d c_s²)/2 is split's b = 5/2 − 𝒩, so it is the label part alone.  Campaign `nersc/campaign_eos_ren1.txt` (144 tasks:
  c_s² = 1/4 at rest and v = 0.8, c_s² = 1/2 at rest), beside `nersc/campaign_eos_split.txt` (192 tasks).
- Note: an `srun` without `-n 1` on a debug node runs 32 copies (job 61861667, testsplit, wasted a few node-minutes).
- **D. testsplit** (debug 61861873 → fixes → 61862296): all 33 checks pass.  Two fixes on the way (878ddaa): the inverse asked for 1e-14,
  but H(u) through the target's forward solve (eos_newton, tol 1e-12) is resolved only to ~1e-13, so Newton stalled (now 1e-12 relative,
  or a step below 1e-13 u); and heat-bath proposals in the tail drove Newton below the label boundary b without converging (now: if an
  iterate falls below b and e ≤ H(b), no state; else bisect toward b).  HMC: |ΔH/H| ∝ δt² (ratio 16.1), 5/5 accepted at both c_s²;
  10080 inverse solves, 2.9 iterations per solve, 0 failures; ~7% of the heat-bath proposals have e < |m| (no fluid state; the tree
  level rejects them the same way).
- Regressions submitted against 53d29e4 from the pinned worktree `~/stochastic-sph-splitR` (4482c91): switch off, MODE=all (job
  61862061), and SPH_RENORM=legendre, MODE=regress (job 61862064); shared -c 64.  Smoke runs: debug 61862298
  (`$SCRATCH/sph/eos_split/smoke/smoke.sh`).
- **E. Smoke** (debug 61862298, 61863427; L = 0.45, h/a = 2, s_a = 54, 200 updates): c_s² = 1/4 split crashed at first in
  equilibrium_init (V̄ unset: check_split ran only in the HMC update; now called when SimParams is built, 08962be).  2 inverse solves
  in 2.6e5 did not converge (proposals with e < |m| whose raised-energy start fell below b; fixed, 0 since; probe on debug 61863010).
  Results: c_s² = 1/2 at rest E/e−1 = −0.015, P/p−1 = −0.039 (σ = 18.0; = SPH_RENORM=1 of legendre_plan.tex); c_s² = 1/4 at rest split
  −0.021, −0.047 against SPH_RENORM=1 −0.020, −0.004 (σ = 10.8): the difference −0.043 is the label-only pressure error 𝒩/σ = 0.046 that
  E_dyn removes.  In motion and at s_a = 16 no failures, acceptance 0.985–1.0.  HMC, 100 updates on 2 threads, same node: tree 77 s,
  Legendre 78 s, split 76 s.  Dynamics (stability.jl, S0 = 1e5, t = 20): ΔH/H = +4.8e-5 (split), +4.2e-5 (SPH_RENORM=1); the stochastic
  step with split costs 3.4× SPH_RENORM=1's (175 s against 51 s; the votes' Newton solves; Legendre 2.6× the tree).
- Regressions resubmitted at 08962be (the SimParams change touches sph.jl): 61863441 (switch off, all), 61863444 (legendre, regress).
- **F. Campaigns** from the pinned worktree `~/stochastic-sph-splitC` (08962be), packed with node.sh, THREADS = 2: eos_split (192 tasks)
  on 3 regular nodes and eos_ren1 (144) on 2, one wave each, ~20 min (h/a = 3 tasks ~15 min on dedicated nodes): estimate ≈ 1.7
  node-hours, at most 5 (-t 1:00).
- **Campaigns done (2026-11-23 08:53)**: eos_split job 61863928 (3 nodes, 19.8 min) and eos_ren1 job 61863936 (2 nodes, 20.3 min): 1.7
  node-hours.  All 42 lines with 8 seeds; no NaN; 0 failed inverse solves in 4.2e8 (split); acceptance 0.97–1.0 everywhere (c_s² = 1/4,
  s_a = 16 included).  Figures docs/figs/split_*.png, tables split_table.csv, split_compare_table.csv (examples/eos_compare.py); reading in
  docs/step5_plan.tex, "Results".  In short: split = Legendre where comparable (P at s_a = 541, c_s² = 1/4: −0.0168 / −0.0166; R at
  c_s² = 1/2: +0.0033 / +0.0032 at s_a = 16); SPH_RENORM=1 (label only) differs by its one-loop error (P: +𝒩/σ as predicted, 0.0046 and
  0.044 at s_a = 541, 54); positions' correction dominates at s_a ≥ 54.  Split: no solver trouble, stochastic step 3.4× label-only.
- **Faster inverse** (owner: make it faster and robust): a warm start from the particle's label, as Legendre's votes, did not help
  (iterations 2.9 → 3.8, and 4.5 in the heat bath): the target's closed-form inversion is already as good a start, and the cost was the
  forward solve (eos_newton) inside every Newton step in u.  Now Newton in the target's energy e_t, e_t + E_dyn(T, v, V) = e, each step
  the closed-form inversion at (e_t, m, V): 2.9 steps to rounding (a stop relative to e alone was early near the light cone, e/(Tu) ~ γ²;
  the step now runs until it stops shrinking), inverse to 8e-13; Newton in u stays as the fallback.  testsplit passes (debug 61868745,
  then the rounding fix).  Stochastic step, stability.jl t = 20, 4 threads, three runs on one node (debug 61868745, 61869262): split
  86 / 84 s, SPH_RENORM=1 71 / 51 s, Legendre 104 / 100 s (first inverse: 175 s): split 1.2–1.6× label-only, less than Legendre.
  The campaign results stand (same answers to 1e-12).
- **Regressions at 08962be against 53d29e4** (61863441 switch off, all; 61863444 SPH_RENORM=legendre, regress; both report FAIL, for
  these reasons only): switch off — step.jl and stability drivers IDENTICAL, equilibrium drivers differ only in hast_lq_* (known,
  thread order), unit output differs in testcheckpoint (wall clock), testlegendre (a path in a warning) and testrunrecord: SPH_RENORM_VBAR
  was read only with split on, but the record must list every setting of the library — fixed in 9c01b6c (read at load).  Legendre —
  stability drivers IDENTICAL, equilibrium drivers only hast_lq_*, step.jl refused in both (Einstein measure), unit-test output differs
  only in file paths of the refusal messages.  Final checks at 9c01b6c (after the faster inverse touched stochastic.jl/hmc.jl/pointwise.jl):
  jobs 61871467 (switch off, all), 61871469 (legendre, regress), worktree ~/stochastic-sph-splitR2.

### `eos_renorm` — 2026-11-21, `nersc/campaign_eos_renorm.txt`, branch `statics`, docs/gravity.tex — TO DO (note for the NERSC session)

The equation of state of the model with the tree-level and the renormalized Hamiltonian (`SPH_RENORM=1`, docs/partition_v2.tex
"A Hamiltonian that does it"), in canonical HMC ensembles (`examples/eos_measure.jl`): the energy density and the mechanical
pressure (kinetic + virial, `energy_pressure` in src/measure.jl) against the target e(T0), p(T0), the derivatives dE/dT, dP/dT and
c_s^2 at fixed density from the canonical fluctuations, and S(k).  480 tasks, 2 threads, estimate ~2 node-hours (longest task
~1-2 h; laptop 0.5 s per HMC update at N = 360).  Expected (docs/gravity.tex, "What to expect"): tree level E/e-1, P/p-1 = 2.5/sigma
plus the correction of the positions (+0.14, +0.11 at sigma = 16.4, h/a = 2); renormalized: only the correction of the positions
(-0.016, -0.040 there); c_s^2 = 1/2 at one loop for both.  Laptop smoke runs agree (commit message of a6992de).

**What to do (the owner asked the NERSC session to do all of it):**
1. `git pull` on `statics` (needs a6992de or later); precompile `julia --project=nersc`.
2. Smoke test in `-q debug`: one task of `eos_ren_s16_ha4` and one of `eos_tree_s541_ha3_L0.9` (check the .txt and .h5 appear).
3. Submit from `$SCRATCH/sph`:
   `sbatch --array=1-480 -A m3722 --export=ALL,SR_REPO=$HOME/stochastic-sph $HOME/stochastic-sph/nersc/array.sh $HOME/stochastic-sph/nersc/campaign_eos_renorm.txt eos_renorm`
   and record the job ID here.
4. When done: `module load python; python3 examples/eos_measure.py $SCRATCH/sph/eos_renorm docs/figs` (writes docs/figs/eos_EP.png,
   eos_trace.png, eos_derivs.png, eos_Sk.png, eos_table.csv).
5. Add a section "Results" to docs/gravity.tex (before "The equilibrium partition function as a generating functional"), with the
   four figures and a short table from eos_table.csv: E/e-1, P/p-1, the trace, dE/dT, dP/dT and c_s^2 for both Hamiltonians against
   the predictions of "What to expect"; say plainly where they agree and where not (the h/a dependence separates the positions'
   correction, which should scale as 1/(n h^2); the renormalized S(k) in the L = 0.9 boxes should rise at small k).
6. Commit the figures, the csv, gravity.tex and this log entry on `statics` (no .h5 or .jls), and push.

**Done so far (NERSC session, 2026-11-21):**
- Pulled `statics` at 1914653; `julia --project=nersc` precompiled (julia/1.12.1, depot /global/common/software/m3722).
- Smoke test, debug job **61747894** (`$SCRATCH/sph/eos_smoke`, the two runs cut to 24 updates, since a full L = 0.9, h/a = 3 task does
  not fit in 30 min): both COMPLETED, .txt and .h5 written, acceptance 0.96.  eos_ren_s16_ha4 (N = 1512, σ = 5.16 — h/a = 4 at
  s_a = 16 is not σ = 3.9): 11 s per update, E/e−1 = +0.039, P/p−1 = +0.030, c_s² = 0.49; eos_tree_s541_ha3_L0.9: 17 s per update,
  E/e−1 = +0.012, P/p−1 = +0.009, c_s² = 0.50 (24 updates, 8 kept after the burn: indicative only).  Output files are `<tag>_seed0..15`
  (SR_SEED0 = seed − 1).
- Campaign submitted from `$SCRATCH/sph`: job **61748587** (480 tasks, shared QOS, -c 2, -t 8 h).  Output `$SCRATCH/sph/eos_renorm/`.
- 13:20: pending tasks of 61748587 cancelled (tasks 25–122 left running) to pack the rest on 2 regular nodes (job 61751219, `node.sh`
  with the new `TASKS` list); it sat in the queue (Resources, no start estimate) and was cancelled unstarted.  Back to shared arrays,
  run checkout `~/stochastic-sph-eos` (worktree, detached at e3f4307, same code): the S(k) boxes L = 0.9 first, tasks 289–352, job
  **61751343**; the rest, tasks 123–288 and 353–480, job **61751345**.  Lesson: for 2-thread tasks of ≤ 2 h the shared QOS starts ~70
  at once within minutes; dedicated nodes only add queue wait.  Task times on 2 threads: h/a = 2 ~10 min, h/a = 3 ~38 min,
  h/a = 4 ~75 min, L = 0.9 h/a = 3 ~2 h (estimate).
- **Done 17:59**: all 480 tasks COMPLETED (61748587, 61751343, 61751345; the L = 0.9 h/a = 3 boxes took 4.4 h, h/a = 2 1 h).  Total
  ≈ 8 node-hours (shared, 2 cores).  docs/gravity.tex "Results" extended: full table (σ ≈ 17, 180 × h/a = 2, 3, 4, both Hamiltonians),
  finite differences, S(k) at h/a = 2, 3.  Outcome: renormalized E/e−1, P/p−1 = the correction of the positions alone, as predicted to
  ≤ 10%, ∝ 1/(n h²); tree = 2.5/σ + the same; the trace and the departure of c_s² from 1/2 are the positions' (same for both
  Hamiltonians, collapse with n h²; trace 9–19% above one loop at σ ≈ 17); renormalized S(k) rises at small k on Ornstein–Zernike
  with B = δB (no free parameter).  Open: renormalized at σ = 5.16, E/e−1 and P/p−1 → +0.048 at n h² → ∞ (a single-particle remainder
  beyond one loop; check by quadrature).
- Extension `eos_renorm_big` (`nersc/campaign_eos_renorm_big.txt`): S(k) in the box L = 1.8 = 36h (kh_min = 0.175), h/a = 2, ren and tree
  at s_a = 541, ren at s_a = 54, 16 seeds × 400 updates, SR_KHMAX = 1.6: job **61762047** (48 tasks, shared, -t 16 h, est. ~9 h per task,
  ~7 node-hours of the owner's extra 24).  Output into `$SCRATCH/sph/eos_renorm/*_L1.8`.  Question: does the renormalized S(k) at σ = 180
  turn over towards S(0) = 1/δB ≈ 1 (predicted 0.40 at kh_min, tree 0.20)?
- **eos_renorm_big done 2026-11-22 02:16**: 48/48 COMPLETED, 6.0–8.2 h per task.  Renormalized s_a = 54 on Ornstein–Zernike to kh = 0.175
  (B fitted 0.42 ± 0.03 vs δB = 0.38); tree s_a = 541 on the curve (3.8 ± 0.3 vs 3.45); renormalized s_a = 541 rises above tree but less than
  predicted, 0.30 ± 0.03 vs 0.40 at kh = 0.175, B = 1.65 ± 0.23 vs δB = 0.95 (one loop): δB from the positions is larger than one loop at
  σ = 180, h/a = 2; S(0) ≈ 0.6.  gravity.tex Results updated.
- (laptop session, 2026-11-21) docs/gravity.tex now has the section "Results" with the four figures and a table for the first run
  (tree, s_a = 16, h/a = 2): agreement to 0.2% with the exact single particle plus the positions.  When more runs finish, rerun
  eos_measure.py and **extend that section** (table rows, the bullet list), do not start a new one.

Known: the regression's equilibrium-driver comparison differs only in hast_lq_* (thread-order sums; also between two runs of the
same commit) — not a failure of SPH_RENORM.


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

## 2026-11-26: the exact form of the S-scheme (SPH_SSCHEME_FORM=exact; docs/sscheme_plan.tex "The exact form", X1-X6)
- Table of ln g (g = s_exact - s_min, A = V = 1) by examples/exact_count.py dyntable + checkpoints: debug 62047724 (1 node, ~10 min),
  $SCRATCH/sph/exact_table; src/tables/exact_lng_c05.csv (c_s^2 = 1/2). A first table of delta = s_exact - S_1loop (62047454) was dropped:
  at small S_t it cannot give D = ds/dS_t.  Step tests: debug 62048441 ($SCRATCH/sph/ssx_tests; run.sh all timed out at 30 min after
  unit, physics, load and the step regression passed; the driver regression differs only in the known records of 3fa7a3e).
- X4 smoke: debug 62049195, worktree ~/ssX at 14ce681, nersc/smoke_x4_sscheme_exact.sh, $SCRATCH/sph/ssX/x4.
- X5: interactive 62049406 (2 nodes, 12 min), nersc/campaign_eos_positional_ssx.txt -> $SCRATCH/sph/eos_positional/eosP_ssx_*.
- X6: interactive 62049405 (4 nodes, 23 min), nersc/campaign_hastings_compare_ssx.txt -> $SCRATCH/sph/hastings_compare/h{s,q}<sa>_ssxw_*;
  ~/ssX at 93e01a2 (code of 14ce681). Analysis: BOOST_GROUP=XCMP16/54 boost_analysis.jl; PD_SCH=":spw:split+ω,:ssxw:exact+ω"
  PD_TAG=hastings_compare_exact positional_dyn.jl; energy_time.jl ... hastings_compare_exact. Written up in docs/split_dynamics.tex sec:exact.
- Also today: debug 62046403 bisection ($SCRATCH/sph/bisect_drift, worktrees ~/bisect/<commit>), see the correction above.
- Node-hours today: ~0.3 (debug) + 1.5 (X6) + 0.4 (X5) + ~0.5 (tests, tables).

