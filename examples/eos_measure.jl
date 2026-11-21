# eos_measure.jl — the equation of state of the model (docs/gravity.tex, "Measurement plan: flat space first"): hybrid Monte Carlo
# from the hexagonal lattice, as examples/volume_fluctuations.jl, recording after every update the energy density and the mechanical
# pressure (kinetic and virial parts, src/measure.jl energy_pressure), against the target e(T0), p(T0) at the bath temperature.
#   julia -t 8 --project examples/eos_measure.jl L h/a s_a nupdates [tag]
# s_a is the entropy per particle of the lattice start.  With SPH_RENORM=1 the renormalized Hamiltonian (docs/partition_v2.tex).
# Writes <outdir>/<tag>.txt (a summary, key = value) and <outdir>/<tag>.h5 (the format of examples/volume_fluctuations.jl: S(k), g(r),
# the volume series, plus the E and P series), outdir = SR_OUTDIR (default data/eos).
# Environment: SR_OUTDIR, SR_SEED0 (default 1), SR_HMC_DTFAC (default 0.35), SR_HMC_TAUJIT (default 0.5), SR_BURN (fraction discarded,
# default 1/3), SR_KHMAX (default 3).
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const LBOX = parse(Float64, ARGS[1]);  const HA = parse(Float64, ARGS[2]);  const SA = parse(Float64, ARGS[3])
const NUPD = parse(Int, ARGS[4])
const SEED   = setting("SR_SEED0", 1)
const DTFAC  = setting("SR_HMC_DTFAC", 0.35)
const TAUJIT = setting("SR_HMC_TAUJIT", 0.5)
const BURN   = setting("SR_BURN", 1 / 3)
const KHMAX  = setting("SR_KHMAX", 3.0)
const OUTDIR = setting("SR_OUTDIR", normpath(joinpath(@__DIR__, "..", "data", "eos")))
const TAG = (length(ARGS) > 4 ? ARGS[5] : "eos_L$(LBOX)_ha$(HA)_s$(SA)" * (RENORM ? "_renorm" : "_tree")) * "_seed$(SEED)"
const H = 0.05;  const A = H / HA;  const S0 = SA / (sqrt(3) / 2 * A^2)

eos = conformal_eos(default_phys_dim(); η_by_s = 0.2)
params, ps, cl, _ = setup_ic(H, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex, seed = 1000SEED + 4)
start = equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1000SEED + 3))
tg = HMCTarget(start.β0, start.β, params.vol, eos)
dt = DTFAC * compute_dt(ps, params, eos; mode = :ideal)
const NSTEPS = max(1, ceil(Int, params.L[1] / sqrt(cs2(eos)) / dt))

# the target at the bath temperature T0 (at rest)
T0 = start.T;  st = state_from_beta(start.β0, SVector(0.0, 0.0), eos)
e_t = st.ε;  p_t = pressure(eos, e_t);  nbar = params.N / prod(params.L);  σp = p_t / (T0 * nbar)
@printf("eos: N = %d, h/a = %.3g, s_a = %.4g, renorm = %s, T0 = %.6g, σ = %.4g, %d steps per trajectory, %d updates\n",
        params.N, HA, SA, RENORM, T0, σp, NSTEPS, NUPD)

ks = box_wavevectors(params, KHMAX)
rec = Recorder(ks);  prec = ParticleRecorder(ps, params; ks, cl)
E = Float64[];  P = Float64[];  Pk = Float64[];  Pv = Float64[];  acc = Float64[];  vmean = Float64[];  vvar = Float64[]
rng = Xoshiro(1000SEED + 7);  t0 = time()
for n in 1:NUPD
    nst = max(1, round(Int, NSTEPS * (1 - TAUJIT * rand(rng))))
    u = hmc_update!(ps, params, cl, eos, tg, rng; nsteps = nst, dt);  push!(acc, u.accepted)
    r = energy_pressure(ps, params, cl, eos)
    push!(E, r.E);  push!(P, r.P);  push!(Pk, r.Pkin);  push!(Pv, r.Pvir)
    Vn = [q.s / q.S_smooth for q in ps] .* nbar;  push!(vmean, mean(Vn));  push!(vvar, var(Vn) / mean(Vn)^2)
    record!(prec, ps, n * NSTEPS * dt; cl);  record!(rec, ps, n * NSTEPS * dt, eos)
    (n % max(1, NUPD ÷ 10) == 0) && @printf("update %4d  acc %.2f  E/e-1 %+.4f  P/p-1 %+.4f  %.0f s\n",
                                            n, mean(acc), E[end] / e_t - 1, P[end] / p_t - 1, time() - t0)
end

# means over the kept updates, errors from 8 blocks
keep = (floor(Int, BURN * NUPD) + 1):NUPD
function mean_err(x)
    y = x[keep];  nb = 8;  L = length(y) ÷ nb
    L == 0 && return (mean(y), NaN)
    bm = [mean(y[(i - 1) * L + 1:i * L]) for i in 1:nb]
    return (mean(y), std(bm) / sqrt(nb))
end
mE, dE = mean_err(E);  mP, dP = mean_err(P);  mPk, dPk = mean_err(Pk);  mPv, dPv = mean_err(Pv)
tr = P .- E ./ 2;  mT, dT = mean_err(tr)

# temperature derivatives at fixed volume and number of particles from the canonical fluctuations (docs/gravity.tex):
#   T² ∂E/∂T = A Var(E),  T² ∂P/∂T = A Cov(P, E),  c_s²|_n = Cov(P, E)/Var(E),  A the area;
# errors by the jackknife over 8 blocks.  Targets (conformal, d = 2): de/dT = 3e/T, dp/dT = 3p/T.
function fluct_derivs(Ek, Pk)
    Ar = prod(params.L)
    vE = var(Ek);  cPE = cov(Pk, Ek)
    return (dEdT = Ar * vE / T0^2, dPdT = Ar * cPE / T0^2, c2 = cPE / vE)
end
function jackknife(f, x, y; nb = 8)
    xk = x[keep];  yk = y[keep];  L = length(xk) ÷ nb
    full = f(xk, yk);  L < 2 && return full, map(_ -> NaN, full)
    reps = [f(vcat(xk[1:(i - 1) * L], xk[i * L + 1:end]), vcat(yk[1:(i - 1) * L], yk[i * L + 1:end])) for i in 1:nb]
    err = map(k -> sqrt((nb - 1) / nb * sum((r[k] - full[k])^2 for r in reps)), keys(full))
    return full, NamedTuple{keys(full)}(err)
end
fd, fde = jackknife(fluct_derivs, E, P)
dedT_t = 3e_t / T0;  dpdT_t = 3p_t / T0
summary = [
    "tag" => TAG, "renorm" => RENORM, "L" => LBOX, "ha" => HA, "nbar_h2" => nbar * H^2, "N" => params.N, "s_a_start" => SA,
    "T0" => T0, "sigma" => σp, "e_target" => e_t, "p_target" => p_t, "nupdates" => NUPD, "acceptance" => mean(acc),
    "E_over_e_minus1" => mE / e_t - 1, "dE_over_e" => dE / e_t,
    "P_over_p_minus1" => mP / p_t - 1, "dP_over_p" => dP / p_t,
    "Pkin_over_nT" => mPk / (nbar * T0), "dPkin_over_nT" => dPk / (nbar * T0),
    "Pvir_minus_p_over_nT" => (mPv - p_t) / (nbar * T0), "dPvir_over_nT" => dPv / (nbar * T0),
    "trace_over_nT" => mT / (nbar * T0), "dtrace_over_nT" => dT / (nbar * T0),
    "dEdT_over_target_minus1" => fd.dEdT / dedT_t - 1, "d_dEdT_over_target" => fde.dEdT / dedT_t,
    "dPdT_over_target_minus1" => fd.dPdT / dpdT_t - 1, "d_dPdT_over_target" => fde.dPdT / dpdT_t,
    "cs2_fixed_n" => fd.c2, "d_cs2_fixed_n" => fde.c2,
]
mkpath(OUTDIR)
meta = (L = params.L, h = params.h, a = params.a, ha = HA, N = params.N, S0 = S0, s_a = SA, T0 = T0, sigma = σp, renorm = RENORM,
        eos_meta(params, eos)..., sampler = "hmc", nupdates = NUPD, nsteps = NSTEPS, taujit = TAUJIT, dt = dt,
        acceptance = mean(acc), seed = SEED, measure = string(params.measure), e_target = e_t, p_target = p_t)
write_equilibrium_h5(joinpath(OUTDIR, "$(TAG).h5"), meta, rec, vmean, zeros(NUPD); prec,
                     extra = (E = E, P = P, Pkin = Pk, Pvir = Pv, vol_mean = vmean, vol_var = vvar, hmc_acc = acc))
open(joinpath(OUTDIR, "$(TAG).txt"), "w") do io
    for (k, v) in summary;  println(io, k, " = ", v);  end
end
for (k, v) in summary;  println(k, " = ", v);  end
