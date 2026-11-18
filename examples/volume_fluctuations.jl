# volume_fluctuations.jl — the particle volumes V_a = 1/n_a in the equilibrium of the positions (docs/equilibrium.tex):
# hybrid Monte Carlo from the hexagonal lattice, as examples/hmc_positions.jl, recording after every update the mean and
# variance of the volumes and the structure factor and g(r) of the positions.
#   julia -t 8 --project examples/volume_fluctuations.jl L h/a s_a nupdates [tag]
# s_a is the entropy per particle (the entropy density is s_a/V, V = √3 a²/2 at h = 0.05).  Writes
# data/volfluct/<tag>.h5 (the format of examples/equilibrium.jl plus the volume series) for examples/volume_fluctuations.py.
# Environment: SR_SEED0 (default 1), SR_HMC_DTFAC (default 0.35), SR_HMC_TAUJIT (default 0.5), SR_KHMAX (default 3).
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const LBOX = parse(Float64, ARGS[1]);  const HA = parse(Float64, ARGS[2]);  const SA = parse(Float64, ARGS[3])
const NUPD = parse(Int, ARGS[4]);  const TAG = length(ARGS) > 4 ? ARGS[5] : "vf_L$(LBOX)_ha$(HA)_s$(SA)"
const SEED   = setting("SR_SEED0", 1)
const DTFAC  = setting("SR_HMC_DTFAC", 0.35)
const TAUJIT = setting("SR_HMC_TAUJIT", 0.5)
const KHMAX  = setting("SR_KHMAX", 3.0)
const H = 0.05;  const A = H / HA;  const S0 = SA / (sqrt(3) / 2 * A^2)

eos = conformal_eos(default_phys_dim(); η_by_s = 0.2)
params, ps, cl, _ = setup_ic(H, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex, seed = 1000SEED + 4)
start = equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1000SEED + 3))
tg = HMCTarget(start.β0, start.β, params.vol, eos)
dt = DTFAC * compute_dt(ps, params, eos; mode = :ideal)
const NSTEPS = max(1, ceil(Int, params.L[1] / sqrt(cs2(eos)) / dt))
@printf("volume fluctuations: N = %d, h/a = %.3g, s_a = %.4g, %d steps per trajectory, %d updates, measure %s\n",
        params.N, HA, S0 * params.vol, NSTEPS, NUPD, params.measure)

ks = box_wavevectors(params, KHMAX)
rec = Recorder(ks);  prec = ParticleRecorder(ps, params; ks, cl)
vmean = Float64[];  vvar = Float64[];  vskew = Float64[];  acc = Float64[]
rng = Xoshiro(1000SEED + 7);  t0 = time();  nbar = params.N / prod(params.L)
for n in 1:NUPD
    nst = max(1, round(Int, NSTEPS * (1 - TAUJIT * rand(rng))))
    u = hmc_update!(ps, params, cl, eos, tg, rng; nsteps = nst, dt);  push!(acc, u.accepted)
    hmc_measure!(ps, params, cl, eos)
    V = [q.s / q.S_smooth for q in ps] .* nbar                     # V_a n̄
    μ = mean(V);  push!(vmean, μ);  push!(vvar, var(V) / μ^2);  push!(vskew, mean(((V .- μ) ./ std(V)) .^ 3))
    record!(prec, ps, n * NSTEPS * dt; cl);  record!(rec, ps, n * NSTEPS * dt, eos)
    (n % max(1, NUPD ÷ 10) == 0) && @printf("update %4d  acc %.2f  ⟨V⟩n̄ %.4f  var V/⟨V⟩² %.3e  skew %+.2f  %.0f s\n",
                                            n, mean(acc), μ, vvar[end], vskew[end], time() - t0)
end
outdir = joinpath(@__DIR__, "..", "data", "volfluct");  mkpath(outdir)
meta = (L = params.L, h = params.h, a = params.a, ha = HA, N = params.N, S0 = S0, s_a = S0 * params.vol, T0 = start.T, eos_meta(params, eos)...,
        sampler = "hmc", nupdates = NUPD, nsteps = NSTEPS, taujit = TAUJIT, dt = dt, acceptance = mean(acc), seed = SEED,
        measure = string(params.measure))
write_equilibrium_h5(joinpath(outdir, "$(TAG).h5"), meta, rec, vmean, zeros(NUPD); prec,
                     extra = (vol_mean = vmean, vol_var = vvar, vol_skew = vskew, hmc_acc = acc))
println("done")
