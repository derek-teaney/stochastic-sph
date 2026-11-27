# eos_measure.jl — the equation of state of the model (docs/gravity.tex, "Measurement plan: flat space first"): hybrid Monte Carlo
# from the hexagonal lattice, as examples/volume_fluctuations.jl, recording after every update the energy density and the mechanical
# pressure (kinetic and virial parts, src/measure.jl energy_pressure), against the target e(T0), p(T0) at the bath temperature.
#   julia -t 8 --project examples/eos_measure.jl L h/a s_a nupdates [tag]
# s_a is the entropy per particle of the lattice start.
# Writes <outdir>/<tag>.txt (a summary, key = value) and <outdir>/<tag>.h5 (the format of examples/volume_fluctuations.jl: S(k), g(r),
# the volume series, plus the E and P series), outdir = SR_OUTDIR (default data/eos).
# Environment: SR_OUTDIR, SR_SEED0 (default 1), SR_HMC_DTFAC (default 0.35), SR_HMC_TAUJIT (default 0.5), SR_BURN (fraction discarded,
# default 1/3), SR_KHMAX (default 3); SR_CS2 (the speed of sound squared of ConstantCs2EOS, default 1/2, the conformal fluid in d = 2)
# and SR_V (the velocity of the fluid along x, default 0; docs/legendre_plan.tex, "The campaign").  SPH_RENORM=split, sscheme or 0
# (the counters of the solves are reported under legendre_*, the names of the first campaign, and V̄ as split_vbar).
# Recorded after every update besides E and P: the stress T^yy = P − (T^xx − T^yy)/2 and the total momentum Σ m_a; the summary has
# the momentum density M, R = M/(v(E + T^yy)) − 1, and the susceptibility matrix from the canonical covariances of the totals
# ℰ = L² E and 𝒫 = Σ m_a at the multipliers β⁰ = 1/T, β = v/T (legendre_plan.tex, eqs. chifluct, chitarget):
#   ∂E/∂v|_T = Cov(ℰ, 𝒫_x)/(L² T),  ∂M/∂v|_T = Var(𝒫_x)/(L² T),  ∂E/∂T|_v = Cov(ℰ, ℰ − v𝒫_x)/(L² T²),  ∂M/∂T|_v = Cov(𝒫_x, ℰ − v𝒫_x)/(L² T²),
# and the transverse Var(𝒫_y)/(L² T) = w γ², against the target's entries, with jackknife errors.
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
const SCHEME = SSCHEME ? "sscheme" : SPLIT ? "split" : "tree"
const CS2 = setting("SR_CS2", 0.5);  const VFLUID = setting("SR_V", 0.0)
const TAG = (length(ARGS) > 4 ? ARGS[5] : "eos_L$(LBOX)_ha$(HA)_s$(SA)_$(SCHEME)") * "_seed$(SEED)"
const H = 0.05;  const A = H / HA;  const S0 = SA / (sqrt(3) / 2 * A^2)

eos = CS2 == 0.5 ? conformal_eos(default_phys_dim(); η_by_s = 0.2) : ConstantCs2EOS(cs2_val = CS2, η_by_s = 0.2)
params, ps, cl, _ = setup_ic(H, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex, seed = 1000SEED + 4)
# the bath: the target state (src/measure.jl target_state) at s_rel,0 = S0, at rest or moving at v = SR_V: β⁰ = γ/T_rel,0, β = v β⁰ x̂.
# The draw is at rest (the heat bath gives the particles their momenta at the first update; the burn-in absorbs the start).
# (Until 2026-11-27 the rest bath was solved from the totals of the lattice, beta_from_averages: docs/focus_plan.tex Q9.)
const NEXP = 1 + 1 / CS2
const TGT = target_state(S0, VFLUID, eos)
start = equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1000SEED + 3), bath = target_state(S0, 0.0, eos).bath)
const BATH = (β0 = TGT.β0, β = TGT.β, T = 1 / TGT.β0, v = SVector(VFLUID, 0.0))
tg = HMCTarget(BATH.β0, BATH.β, params.vol, eos)
dt = DTFAC * compute_dt(ps, params, eos; mode = :ideal)
const NSTEPS = max(1, ceil(Int, params.L[1] / sqrt(cs2(eos)) / dt))

# the target at the bath's multipliers: T0 = 1/β⁰ (the bare temperature), v = β/β⁰, T_rel = γ T0; e_t, p_t at rest (v = 0) or the
# lab energy density and the transverse pressure (v > 0), M_t = w γ² v
T0 = BATH.T;  vb = BATH.v[1];  γb = 1 / sqrt(1 - dot(BATH.v, BATH.v));  Tr0 = γb * T0
e_rest = state_from_beta(1 / Tr0, SVector(0.0, 0.0), eos).ε;  p_t = pressure(eos, e_rest);  w_t = e_rest + p_t
st = state_from_beta(BATH.β0, BATH.β, eos)
e_t = st.ε;  M_t = st.M[1];  nbar = params.N / prod(params.L);  σp = p_t / (Tr0 * nbar)
@printf("eos: N = %d, h/a = %.3g, s_a = %.4g, %s, c_s² = %.3g, v = %.4g, T0 = %.6g, σ = %.4g, %d steps per trajectory, %d updates\n",
        params.N, HA, SA, SCHEME, CS2, vb, T0, σp, NSTEPS, NUPD)
SPLIT && reset_split_counts!()
SSCHEME && reset_sscheme_counts!()

ks = box_wavevectors(params, KHMAX)
rec = Recorder(ks);  prec = ParticleRecorder(ps, params; ks, cl)
E = Float64[];  P = Float64[];  Pk = Float64[];  Pv = Float64[];  acc = Float64[];  vmean = Float64[];  vvar = Float64[]
Tyy = Float64[];  Px = Float64[];  Py = Float64[]
rng = Xoshiro(1000SEED + 7);  t0 = time()
hmc_equilibrate!(ps, params, cl, eos, tg, rng; nupdates = NUPD, nsteps = NSTEPS, dt, taujit = TAUJIT) do n, u
    push!(acc, u.accepted)
    r = energy_pressure(ps, params, cl, eos)
    push!(E, r.E);  push!(P, r.P);  push!(Pk, r.Pkin);  push!(Pv, r.Pvir);  push!(Tyy, r.P - r.Sxxyy)
    Ptot = sum(q.m for q in ps);  push!(Px, Ptot[1]);  push!(Py, Ptot[2])
    Vn = [q.V for q in ps] .* nbar;  push!(vmean, mean(Vn));  push!(vvar, var(Vn) / mean(Vn)^2)
    record!(prec, ps, n * NSTEPS * dt; cl);  record!(rec, ps, n * NSTEPS * dt, eos)
    (n % max(1, NUPD ÷ 10) == 0) && @printf("update %4d  acc %.2f  E/e-1 %+.4f  Tyy/p-1 %+.4f  %.0f s\n",
                                            n, mean(acc), E[end] / e_t - 1, Tyy[end] / p_t - 1, time() - t0)
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
dedT_t = NEXP * e_t / T0;  dpdT_t = NEXP * p_t / T0          # at rest; e, p ∝ T^n, n = 1 + 1/c_s²
# the susceptibility matrix from the covariances of the totals (Ar = L²), and the transverse momentum susceptibility
function chi(Ek, Xk, Yk)
    Ar = prod(params.L);  ℰ = Ar .* Ek;  𝒦 = ℰ .- vb .* Xk
    return (dEdv = cov(ℰ, Xk) / (Ar * T0), dMdv = var(Xk) / (Ar * T0), dEdT = cov(ℰ, 𝒦) / (Ar * T0^2), dMdT = cov(Xk, 𝒦) / (Ar * T0^2),
            chiyy = var(Yk) / (Ar * T0))
end
function jackknife3(f, x, y, z; nb = 8)
    xk = x[keep];  yk = y[keep];  zk = z[keep];  L = length(xk) ÷ nb
    full = f(xk, yk, zk);  L < 2 && return full, map(_ -> NaN, full)
    cut(u, i) = vcat(u[1:(i - 1) * L], u[i * L + 1:end])
    reps = [f(cut(xk, i), cut(yk, i), cut(zk, i)) for i in 1:nb]
    err = map(k -> sqrt((nb - 1) / nb * sum((r[k] - full[k])^2 for r in reps)), keys(full))
    return full, NamedTuple{keys(full)}(err)
end
χ, χe = jackknife3(chi, E, Px, Py)
# the target's entries (legendre_plan.tex, eq. chitarget), T = T0, w at T_rel
χt = (dEdT = (w_t / T0) * (NEXP * γb^2 - 1), dMdT = NEXP * γb^2 * vb * w_t / T0, dEdv = (NEXP + 2) * γb^4 * vb * w_t - γb^2 * vb * w_t,
      dMdv = w_t * γb^2 * (1 + (NEXP + 2) * γb^2 * vb^2), chiyy = w_t * γb^2)
Ar = prod(params.L);  Mx = Px ./ Ar
mM, dM = mean_err(Mx);  mTyy, dTyy = mean_err(Tyy)
# R = M/(v(E + T^yy)) − 1, its error from the blocks of the ratio's pieces (jackknife)
Rf(Ek, Mk, Tk) = (R = vb > 0 ? mean(Mk) / (vb * (mean(Ek) + mean(Tk))) - 1 : NaN,)
Rv, Re = jackknife3(Rf, E, Mx, Tyy)
lc = SPLIT ? split_counts() :
     SSCHEME ? (sc = sscheme_counts();  (solves = sc.solves, iters = sc.iters, fails = sc.fallbacks, nostate = sc.nostate, cold = 0)) :
     (solves = 0, iters = 0, fails = 0, nostate = 0, cold = 0)
summary = [
    "tag" => TAG, "renorm" => SCHEME != "tree", "scheme" => SCHEME, "cs2" => CS2, "v" => vb, "gamma" => γb, "w_target" => w_t, "M_target" => M_t, "L" => LBOX, "ha" => HA, "nbar_h2" => nbar * H^2, "N" => params.N, "s_a_start" => SA,
    "T0" => T0, "sigma" => σp, "e_target" => e_t, "p_target" => p_t, "nupdates" => NUPD, "acceptance" => mean(acc),
    "E_over_e_minus1" => mE / e_t - 1, "dE_over_e" => dE / e_t,
    "P_over_p_minus1" => mP / p_t - 1, "dP_over_p" => dP / p_t,
    "Pkin_over_nT" => mPk / (nbar * T0), "dPkin_over_nT" => dPk / (nbar * T0),
    "Pvir_minus_p_over_nT" => (mPv - p_t) / (nbar * T0), "dPvir_over_nT" => dPv / (nbar * T0),
    "trace_over_nT" => mT / (nbar * T0), "dtrace_over_nT" => dT / (nbar * T0),
    "dEdT_over_target_minus1" => fd.dEdT / dedT_t - 1, "d_dEdT_over_target" => fde.dEdT / dedT_t,
    "dPdT_over_target_minus1" => fd.dPdT / dpdT_t - 1, "d_dPdT_over_target" => fde.dPdT / dpdT_t,
    "cs2_fixed_n" => fd.c2, "d_cs2_fixed_n" => fde.c2,
    "Tyy_over_p_minus1" => mTyy / p_t - 1, "dTyy_over_p" => dTyy / p_t,
    "M_over_target_minus1" => vb > 0 ? mM / M_t - 1 : NaN, "dM_over_target" => vb > 0 ? dM / M_t : NaN, "M_over_w" => mM / w_t,
    "R" => Rv.R, "dR" => Re.R,
    "chi_dEdT_over_target_minus1" => χ.dEdT / χt.dEdT - 1, "d_chi_dEdT" => χe.dEdT / χt.dEdT,
    "chi_dMdv_over_target_minus1" => χ.dMdv / χt.dMdv - 1, "d_chi_dMdv" => χe.dMdv / χt.dMdv,
    "chi_dEdv" => χ.dEdv, "d_chi_dEdv" => χe.dEdv, "chi_dEdv_target" => χt.dEdv,
    "chi_dMdT" => χ.dMdT, "d_chi_dMdT" => χe.dMdT, "chi_dMdT_target" => χt.dMdT,
    "chi_yy_over_target_minus1" => χ.chiyy / χt.chiyy - 1, "d_chi_yy" => χe.chiyy / χt.chiyy,
    "legendre_solves" => lc.solves, "legendre_iters_per_solve" => lc.iters / max(1, lc.solves), "legendre_fails" => lc.fails,
    "legendre_nostate" => lc.nostate, "seconds" => time() - t0,
    "split_vbar" => SPLIT ? SPLIT_VBAR[] : NaN,
]
mkpath(OUTDIR)
meta = (L = params.L, h = params.h, a = params.a, ha = HA, N = params.N, S0 = S0, s_a = SA, T0 = T0, sigma = σp, renorm = SCHEME != "tree",
        scheme = SCHEME, v = vb, gamma = γb, w_target = w_t, M_target = M_t,
        eos_meta(params, eos)..., sampler = "hmc", nupdates = NUPD, nsteps = NSTEPS, taujit = TAUJIT, dt = dt,
        acceptance = mean(acc), seed = SEED, measure = "liouville", e_target = e_t, p_target = p_t)
write_equilibrium_h5(joinpath(OUTDIR, "$(TAG).h5"), meta, rec, vmean, zeros(NUPD); prec,
                     extra = (E = E, P = P, Pkin = Pk, Pvir = Pv, Tyy = Tyy, Px = Px, Py = Py, vol_mean = vmean, vol_var = vvar, hmc_acc = acc))
open(joinpath(OUTDIR, "$(TAG).txt"), "w") do io
    for (k, v) in summary;  println(io, k, " = ", v);  end
end
for (k, v) in summary;  println(k, " = ", v);  end
