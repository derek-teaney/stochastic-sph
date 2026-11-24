# hmc_positions.jl — hybrid Monte Carlo from the hexagonal lattice (src/hmc.jl, docs/equilibrium_positions.tex, stage 2):
# samples the joint equilibrium of positions, energies and momenta at the multipliers β^μ of the present start (at rest),
# recording after every update the pair observables of the equilibrium runs (g(r), ψ6, N_k), H and the acceptances.
# Writes equilibrium_<tag>_r<seed>.h5 in the format of examples/equilibrium.jl, with t = the update number times the
# trajectory length, so that examples/pair_analysis.jl reads it.
#   julia -t 4 --project examples/hmc_positions.jl L h/a nupdates nsteps [tag]
# Environment: SR_SREL0 (s_rel,0, the rest-frame entropy density, default 1e6), SR_SEED0 (seed, default 1), SR_HMC_DTFAC (δt / acoustic step,
# default 0.5), SR_OUTDIR (default data/hmc), SR_KHMAX (wave vectors of N_k, default 1.5), SR_BOOST (fluid velocity along x,
# default 0: the rest-frame entropy density is SREL0, the lab one γS0 and M = w0 γ² v, as in examples/equilibrium.jl).
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const LBOX = parse(Float64, ARGS[1]);  const HA = parse(Float64, ARGS[2])
const NUPD = parse(Int, ARGS[3]);  const NSTEPS_ARG = parse(Int, ARGS[4])
const TAG  = length(ARGS) > 4 ? ARGS[5] : "hmc_L$(LBOX)_ha$(HA)"
const SREL0  = setting("SR_SREL0", 1e6)          # s_rel,0, the rest-frame entropy density (was SR_S0)
const SEED   = setting("SR_SEED0", 1)
const DTFAC  = setting("SR_HMC_DTFAC", 0.5)
const OUTDIR = setting("SR_OUTDIR", joinpath(@__DIR__, "..", "data", "hmc"))
const KHMAX  = setting("SR_KHMAX", 1.5)                 # as examples/equilibrium.jl
const BOOST  = setting("SR_BOOST", 0.0);  const GAMMA = 1 / sqrt(1 - BOOST^2)
# SR_HMC_TAUJIT=j: each trajectory has a random length, uniform in [(1−j)τ, τ] (as in examples/equilibrium.jl; default 0, a fixed τ, which
# resonates with a box sound mode when τ = L/c_s).  nsteps = 0 on the command line: τ = L/c_s, as in examples/equilibrium.jl.
const TAUJIT = setting("SR_HMC_TAUJIT", 0.0)
const RECORD_E = setting("SR_RECORD_E", false) || BOOST != 0            # the energy modes E_k too
const BATH = setting("SR_BATH", :nominal)            # as examples/equilibrium.jl: nominal (the requested state) or totals (before 2026-11-23)
BATH in (:nominal, :totals) || error("SR_BATH must be nominal or totals")

eos = conformal_eos(default_phys_dim(); η_by_s = 0.2)
_, ps0, _, _ = setup_ic(0.05, HA, eos, (x, y) -> SREL0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex, seed = 1000SEED + 4)
const MX0 = (ps0[1].e + ps0[1].p) * GAMMA^2 * BOOST            # the momentum density of the boosted state
const TREL0 = trel_of_srel(eos, SREL0)                             # the temperature of the target fluid at s_rel,0 (the EOS; as equilibrium.jl)
params, ps, cl, _ = setup_ic(0.05, HA, eos, (x, y) -> GAMMA * SREL0, (x, y) -> SVector(MX0, 0.0); L = LBOX, lattice = :hex, seed = 1000SEED + 4)
start = equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1000SEED + 3),       # the present start, at the lattice
                          bath = BATH === :nominal ? (GAMMA / TREL0, SVector(GAMMA * BOOST / TREL0, 0.0)) : nothing)
tg = HMCTarget(start.β0, start.β, params.vol, eos)
dt = DTFAC * compute_dt(ps, params, eos; mode = :ideal)
const NSTEPS = NSTEPS_ARG > 0 ? NSTEPS_ARG : max(1, ceil(Int, params.L[1] / sqrt(cs2(eos)) / dt));  τ = NSTEPS * dt
@printf("HMC: N = %d, s_a = %.4g, T = %.5g, v = %.3g, δt = %.3e, %d steps per trajectory (τ = %.3g), %d updates\n",
        params.N, GAMMA * SREL0 * params.vol, start.T, BOOST, dt, NSTEPS, τ, NUPD)

ks = box_wavevectors(params, KHMAX)
rec = Recorder(ks; energies = RECORD_E);  prec = ParticleRecorder(ps, params; ks, cl)
Ht = Float64[];  acc = Float64[];  hb = Float64[];  dH = Float64[];  its = Float64[]
rng = Xoshiro(1000SEED + 7);  H0 = hmc_measure!(ps, params, cl, eos)[1];  t0 = time()
for n in 1:NUPD
    nst = TAUJIT > 0 ? max(1, round(Int, NSTEPS * (1 - TAUJIT * rand(rng)))) : NSTEPS
    u = hmc_update!(ps, params, cl, eos, tg, rng; nsteps = nst, dt)
    push!(acc, u.accepted);  push!(hb, u.heatbath);  push!(dH, u.dH);  push!(its, u.iters)
    record!(prec, ps, n * τ; cl);  record!(rec, ps, n * τ, eos)
    push!(Ht, hmc_measure!(ps, params, cl, eos)[1] / H0 - 1)
    (n % max(1, NUPD ÷ 20) == 0 || n == NUPD) &&
        @printf("update %5d  acc %.3f  heat bath %.3f  |ΔH/H| %.1e  iters/step %.1f  ψ6 %.3f (local %.3f)  H/H0−1 %+.2e  %.0f s\n",
                n, sum(acc) / n, sum(hb) / n, sum(abs, dH) / n, sum(its) / n, prec.psi6[end], prec.psi6loc[end], Ht[end], time() - t0)
end
mkpath(OUTDIR)
meta = (L = params.L, h = params.h, a = params.a, ha = HA, N = params.N, srel0 = SREL0, s_a = GAMMA * SREL0 * params.vol, T0 = start.T, Trel0 = TREL0, bath = string(BATH), eos_meta(params, eos)..., boost = BOOST,
        sampler = "hmc", nupdates = NUPD, nsteps = NSTEPS, taujit = TAUJIT, dt = dt, tau = τ, acceptance = sum(acc) / NUPD,
        heatbath = sum(hb) / NUPD, seed = SEED)
write_equilibrium_h5(joinpath(OUTDIR, "equilibrium_$(TAG)_r$(SEED).h5"), meta, rec, Ht, zeros(length(Ht)); prec,
                     extra = (hmc_acc = Float64.(acc), hmc_heatbath = hb, hmc_dH = dH, hmc_iters = its))
println("done")
