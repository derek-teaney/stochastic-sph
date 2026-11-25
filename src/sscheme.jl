# sscheme.jl — the S-scheme, state counting at one loop (docs/renorm3.tex; docs/sscheme_plan.tex, "Level 1", item B).
#
# A particle has the label s (flat Liouville measure e^s ds d²m, conserved by the ideal step), the momentum m and the volume V.  Its
# target state is the temperature T_rel and velocity v that the input equation of state gives to a cell of volume V holding the particle's
# energy e and momentum m (e = V(wγ² − p), m = V w γ² v); its target entropy is S_t = γ s(T_rel) V.  The label is the target entropy less
# two counterterms evaluated at the target state (d = 2, D = d + 1 = 3, a = 1/c_s² − v²),
#   s = S_t − Δ_χ − Δ_ℓ,   Δ_χ = (D/2) ln(2π V γ w T_rel) + ½ ln a,   Δ_ℓ = 1 + ln V + ln γ − ln T_rel,
# and the particle's temperature, velocity and pressure are the derivatives of S(e, m, V) = s (renorm3.tex, eq. Tava; plan, eq. state1):
#   ε_p = Δ_p/(S_t a),  ε_∥ = Δ_θ/S_t,  𝒟 = 1 − (1 + v²) ε_p + v ε_∥,
#   T_a = ∂H/∂s = T/𝒟,   v_a = [v (1 − 2ε_p) + ε_∥]/𝒟 (along m),   P_a V/T_a = S_t/n − (D/2 + 1) + [(D/2)(2 + q) − 1]/n,
#   Δ_p = (D/2)(2 + q − v²) + v²(1 − v²)/a − (1 + v²),   Δ_θ = (D/2) v − v(1 − v²)/a + v,
# with T = T_rel/γ, q = 1/c_s², n = 1 + q.  The energy is the target's, H = V(wγ² − p).  The weight of a vote (Liouville measure, the
# variables (e, m)) is s − ln T_a and needs no solve.  A state has a Hamiltonian when 𝒟 > 0 and |v_a| < 1 (and the target's γ ≤ γ_max,
# SPH_GAMMA_MAX, as at the tree level); outside it the maps return ok = false, s = −Inf, weight −Inf.
#
# For a constant speed of sound (ConstantCs2EOS: p = A T_rel^n, A = c_s² T_scale^(−n)) both target inversions are one-dimensional:
#   from the energy:  the rest-frame energy density in closed form (pointwise.jl, rest_energy), as eos_from_energy;
#   from the entropy: with x = T_rel and γv = |m|/(S_t x),  (n A V)² x^(2n−2) [1 + (|m|/(S_t x))²] = S_t², monotonic in x: Newton in ln x.
# The map from the label iterates on S_t alone, S_t ← S_t − (s(S_t) − s)/𝒟 (𝒟 = ∂s/∂S_t at fixed m, V), from the particle's previous S_t,
# with a bisection on S_t as the fallback.

const SS_CNT = zeros(Int, 8, Threads.maxthreadid())     # rows: solves, iterations, fallbacks (bisection), no state
@inline sscnt!(i, x = 1) = (@inbounds SS_CNT[i, Threads.threadid()] += x;  nothing)
reset_sscheme_counts!() = fill!(SS_CNT, 0)
sscheme_counts() = (t = sum(SS_CNT; dims = 2);  (solves = t[1], iters = t[2], fallbacks = t[3], nostate = t[4]))

const SS_TOL = 1e-13          # relative change of S_t at which the map from the label stops
const SS_MAXIT = 30           # Newton steps before the bisection
const SS_D = 3                # D = d + 1 (d = 2)
const SS_SOUND = setting("SPH_SSCHEME_SOUND", 1.0)
const SS_CUT = setting("SPH_SSCHEME_CUT", 0.0)
const SS_G = setting("SPH_SSCHEME_G", 0.0)   # DIAGNOSTIC: X = ½ ln(a/q) times g(S_t) = 1 − h, h = 3u² − 2u³, u = b/S_t (c_s² = 1/2)   # DIAGNOSTIC: labels below s_min(rest) + CUT·½ ln(q/(q−1)) have no state   # DIAGNOSTIC: the factor of the sound term ½ ln a in Δ_χ (1 = the scheme)

struct SSchemeConsts
    c2 :: Float64;  q :: Float64;  n :: Float64;  A :: Float64
    Pconst :: Float64         # [(D/2)(2 + q) − 1]/n − (D/2 + 1): P_a V/T_a = S_t/n + Pconst
    sb :: Float64             # DIAGNOSTIC: the cut on the label (SPH_SSCHEME_CUT=1), −Inf without it
end
function SSchemeConsts(eos::ConstantCs2EOS)
    c2 = eos.cs2_val;  q = 1 / c2;  n = 1 + q
    c0 = SSchemeConsts(c2, q, n, c2 * eos.T_scale^(-n), ((SS_D / 2) * (2 + q) - 1) / n - (SS_D / 2 + 1), -Inf)
    SS_CUT > 0 || return c0
    abs(c2 - 0.5) < 1e-12 || error("SPH_SSCHEME_CUT: c_s^2 = 1/2 only (no V dependence of s_min)")
    # s_min at rest (V = 1; at c_s^2 = 1/2 it does not depend on V): golden section on ln S_t, the label without the domain test
    lab(St) = (r = sscheme_target_entropy(St, 0.0, 1.0, c0);  Tr = r[1];  p = c0.A * Tr^n;  w = n * p;
               St - ((SS_D / 2) * log(2π * w * Tr) + 0.5 * log(q)) - (1 - log(Tr)))
    lo, hi = log(0.5), log(20.0);  φ = (sqrt(5) - 1) / 2
    for _ in 1:200
        x1 = hi - φ * (hi - lo);  x2 = lo + φ * (hi - lo)
        lab(exp(x1)) < lab(exp(x2)) ? (hi = x2) : (lo = x1)
    end
    smin = lab(exp((lo + hi) / 2))
    return SSchemeConsts(c2, q, n, c0.A, c0.Pconst, smin + SS_CUT * 0.5 * log(q / (q - 1)))
end
SSchemeConsts(eos::AbstractEOS) = error("SPH_RENORM=sscheme: implemented for ConstantCs2EOS only (got $(typeof(eos)))")

const SS_NOSTATE = (ok = false, s = -Inf, St = NaN, H = NaN, Ta = NaN, va = NaN, Pa = NaN, D = NaN, weight = -Inf,
                    γ = NaN, Trel = NaN, T = NaN, v = NaN, erest = NaN)

# Everything about the particle at a target state: T_rel, uγ = γv ≥ 0 (along m) and V.
@inline function sscheme_state(Tr::Float64, uγ::Float64, V::Float64, c::SSchemeConsts)
    (; q, n, A) = c
    γ = sqrt(1 + uγ * uγ);  v = uγ / γ;  v2 = v * v
    (isfinite(Tr) && Tr > 0 && γ <= GAMMA_MAX) || return SS_NOSTATE
    a = q - v2
    p = A * Tr^n;  w = n * p;  St = γ * (w / Tr) * V
    Dχ = (SS_D / 2) * log(2π * V * γ * w * Tr) + SS_SOUND * 0.5 * log(a)
    Dℓ = 1 + log(V) + log(γ) - log(Tr)
    s = St - Dχ - Dℓ
    Dp = (SS_D / 2) * (2 + q - v2) + SS_SOUND * v2 * (1 - v2) / a - (1 + v2)
    Dθ = (SS_D / 2) * v - SS_SOUND * v * (1 - v2) / a + v
    Pg = 0.0
    if SS_G > 0                                                        # Δ → Δ − h X: s += h X, slopes and pressure from h(S_t), X(v)
        b = SS_D / 2 + 1 - (1 - 2 * c.c2);  u = b / St
        h = 3u^2 - 2u^3;  dh = -6u^2 * (1 - u)                         # dh/d ln S_t;  ∂ln S_t/∂(τ, θ, ln V) = (q, v, 1)
        X = 0.5 * log(a / q);  Xθ = -v * (1 - v2) / a                  # ∂X/∂θ
        s += h * X
        Dp += -dh * X * a + v * h * Xθ                                 # M = −h X:  ΔM_p = ∂M/∂τ − v ∂M/∂θ
        Dθ += -dh * v * X - h * Xθ
        Pg = dh * X / n                                                # P_a V/T_a += (∂M/∂τ)/n − ∂M/∂ln V
    end
    εp = Dp / (St * a);  εpar = Dθ / St
    D = 1 - (1 + v2) * εp + v * εpar
    T = Tr / γ
    Ta = T / D;  va = (v * (1 - 2εp) + εpar) / D
    ok = D > 0 && abs(va) < 1 && s > c.sb
    ok || return SS_NOSTATE
    Pa = (St / n + c.Pconst + Pg) * Ta / V
    return (ok = true, s = s, St = St, H = V * (w * γ * γ - p), Ta = Ta, va = va, Pa = Pa, D = D, weight = s - log(Ta),
            γ = γ, Trel = Tr, T = T, v = v, erest = p / c.c2)
end

# The target state of the energy e, momentum magnitude |m| and volume V: (T_rel, γv); (NaN, NaN) if no fluid has it (|M| ≥ ε).
@inline function sscheme_target_energy(e::Float64, mmag::Float64, V::Float64, c::SSchemeConsts)
    ε = e / V;  M = mmag / V;  c2 = c.c2
    (isfinite(ε) && ε > 0 && M < ε) || return (NaN, NaN)
    er = 2 * (ε^2 - M^2) / ((1 - c2) * ε + sqrt((1 + c2)^2 * ε^2 - 4c2 * M^2))     # rest_energy, conformal form
    v = M / (ε + c2 * er)
    Tr = (c2 * er / c.A)^(1 / c.n)
    return (Tr, v / sqrt(1 - v * v))
end

# The target state of the target entropy S_t, momentum |m| and volume V: Newton in y = ln T_rel on
#   f(y) = ln[(nAV)² (x^(2n−2) + k² x^(2n−4))] − 2 ln S_t,  k = |m|/S_t,  x = e^y   (f′ = [(2n−2) x² + (2n−4) k²]/(x² + k²) > 0).
@inline function sscheme_target_entropy(St::Float64, mmag::Float64, V::Float64, c::SSchemeConsts; y0::Float64 = NaN)
    (isfinite(St) && St > 0) || return (NaN, NaN)
    n = c.n;  nAV = n * c.A * V;  k = mmag / St
    y = isfinite(y0) ? y0 : log(St / nAV) / (n - 1)                    # the start: at rest, x^(n−1) = S_t/(nAV)
    for _ in 1:60
        x = exp(y);  x2 = x * x;  k2 = k * k
        f = 2 * log(nAV) + (2n - 4) * y + log(x2 + k2) - 2 * log(St)
        fp = ((2n - 2) * x2 + (2n - 4) * k2) / (x2 + k2)
        dy = f / fp;  y -= dy
        abs(dy) < 1e-15 && break
    end
    x = exp(y)
    return (x, k / x)
end

# The map from the energy: (e, m, V) → the state of the particle.  No iteration beyond the target's inversion.
@inline function sscheme_from_energy(e::Float64, m::SVector{2,Float64}, V::Float64, c::SSchemeConsts)
    Tr, uγ = sscheme_target_energy(e, norm(m), V, c)
    isfinite(Tr) || return SS_NOSTATE
    return sscheme_state(Tr, uγ, V, c)
end

# The map from the label: (s, m, V) → the state of the particle, by Newton's iteration on S_t from St0 (the particle's previous S_t),
# with a bisection on S_t when Newton leaves the domain or does not converge.  ok = false if no state has the label s.
function sscheme_from_label(s::Float64, m::SVector{2,Float64}, V::Float64, c::SSchemeConsts, St0::Float64)
    sscnt!(1)
    mmag = norm(m)
    st_at(St) = (r = sscheme_target_entropy(St, mmag, V, c);  isfinite(r[1]) ? sscheme_state(r[1], r[2], V, c) : SS_NOSTATE)
    St = (isfinite(St0) && St0 > 0) ? St0 : max(s, 1.0) + 10.0
    st = st_at(St)
    for _ in 1:SS_MAXIT
        st.ok || break
        sscnt!(2)
        d = (st.s - s) / st.D
        St -= d
        st = st_at(St)
        abs(d) <= SS_TOL * St && st.ok && return st
    end
    # fallback: bisection on S_t; s(S_t) increases on the domain, and a state outside it counts as "below" (the domain is S_t above a
    # boundary at fixed m, V).
    sscnt!(3)
    hi = max(St0, s, 1.0) + 10.0
    k = 0
    while !(st_at(hi).ok && st_at(hi).s >= s)
        hi *= 2;  k += 1
        k > 200 && (sscnt!(4);  return SS_NOSTATE)
    end
    lo = hi
    k = 0
    while true
        lo /= 2;  k += 1
        r = st_at(lo)
        (!r.ok || r.s < s) && break
        k > 2000 && (sscnt!(4);  return SS_NOSTATE)
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        r = st_at(mid)
        (r.ok && r.s >= s) ? (hi = mid) : (lo = mid)
        hi - lo <= SS_TOL * hi && break
    end
    st = st_at(hi)
    # no root: the label lies below every state of the domain at this m, V (the bisection then ends at the boundary)
    (st.ok && abs(st.s - s) <= 1e-9 * max(1.0, abs(s))) || (sscnt!(4);  return SS_NOSTATE)
    return st
end

# The particle's velocity vector: v_a along m (zero at m = 0).
@inline sscheme_velocity(st, m::SVector{2,Float64}) = (mm = norm(m);  mm > 0 ? (st.va / mm) * m : zero(m))

# ── The S-scheme in the code (sscheme_plan.tex, item C) ───────────────────────────────────────────────────────────────────────────────
# The particle fields: pa.s the label; pa.St the target entropy (the warm start of the next solve, and the entropy wherever the code needs a
# physical one, entropy_of); pa.V = 1/n_a (stored by the density pass); pa.H, pa.v (= v_a, along m), pa.T = T_a, pa.p = P_a the particle's
# own; pa.e and pa.γ the target's rest-frame energy density and Lorentz factor (the transport coefficients: target_trel, srel(eos, pa.e));
# pa.S_smooth = S_t/V, pa.M_smooth = m/V the target's densities.  H_over_S, F_over_S are not used (the energy Hamiltonian, the number volume).

# The state of the particle at its label, momentum and volume; NaN fields outside the domain (hybrid Monte Carlo rejects the trajectory).
function sscheme_solve!(pa::Particle, eos::AbstractEOS)
    c = SSchemeConsts(eos)
    st = sscheme_from_label(pa.s, pa.m, pa.V, c, pa.St)
    if !st.ok
        pa.e = NaN;  pa.p = NaN;  pa.T = NaN;  pa.γ = NaN;  pa.H = NaN;  pa.v = SVector(NaN, NaN)
        pa.v_over_S = SVector(NaN, NaN);  pa.H_over_S = NaN;  pa.F_over_S = NaN
        return nothing
    end
    pa.e = st.erest;  pa.p = st.Pa;  pa.T = st.Ta;  pa.γ = st.γ;  pa.H = st.H;  pa.St = st.St
    pa.v = sscheme_velocity(st, pa.m)
    pa.S_smooth = st.St / pa.V;  pa.M_smooth = pa.m / pa.V
    pa.v_over_S = pa.v / pa.S_smooth;  pa.H_over_S = NaN;  pa.F_over_S = NaN
    return nothing
end

# The set-up gives each particle its target entropy in pa.s (S_func V, setup.jl): at the smoothed volume, replace it by the label of the state
# with that target entropy and the particle's momentum.
function sscheme_labels!(particles, params, cl, eos)
    c = SSchemeConsts(eos)
    smooth_to_particles!(particles, params, cl)
    for pa in particles
        St = pa.s;  r = sscheme_target_entropy(St, norm(pa.m), pa.V, c)
        st = sscheme_state(r[1], r[2], pa.V, c)
        st.ok || error("SPH_RENORM=sscheme: the set-up state S_t = $St, |m| = $(norm(pa.m)), V = $(pa.V) is outside the domain")
        pa.s = st.s;  pa.St = St
    end
    return nothing
end

# The target's T_rel (the transport coefficients, the time step): pa.T γ at the tree level (and split, legendre, as before); with sscheme
# pa.T is T_a and the target's T_rel follows from its rest-frame energy density.
target_trel(pa, eos) = SSCHEME ? Trel(eos, pa.e) : pa.T * pa.γ
# The target's velocity of a particle with sscheme (the comoving structures of the viscous step, as star_coef)
function sscheme_target_velocity(pa, eos)
    mm = norm(pa.m);  mm > 0 || return zero(pa.m)
    _, uγ = sscheme_target_energy(pa.H, mm, pa.V, SSchemeConsts(eos))
    return (uγ / sqrt(1 + uγ * uγ) / mm) * pa.m
end

# The inverse (the votes, the heat bath, the star coefficients): (s, v_a, T_a) of the particle of energy e, momentum m and volume V.
function sscheme_inverse(e, m, V, eos)
    st = sscheme_from_energy(e, m, V, SSchemeConsts(eos))
    st.ok || return (-Inf, SVector(NaN, NaN), NaN)
    return (st.s, sscheme_velocity(st, m), st.Ta)
end
# The vote weight (Liouville measure): s − ln T_a, no solve.
sscheme_vote_entropy(e, m, V, eos) = sscheme_from_energy(e, m, V, SSchemeConsts(eos)).weight

# The schemes the S-scheme supports (as check_split).
function check_sscheme(params)
    SSCHEME || return nothing
    hint = "; set SPH_RENORM=split (the default) or SPH_RENORM=0"
    params.volume === :number && params.hamiltonian === :energy ||
        error("SPH_RENORM=sscheme needs hamiltonian = :energy and volume = :number" * hint)
    sim_dim(params) == 2 && params.phys_dim == 2 || error("SPH_RENORM=sscheme: implemented for d = 2" * hint)
    params.measure === :liouville || error("SPH_RENORM=sscheme needs the Liouville measure (SPH_MEASURE=liouville)" * hint)
    params.ideal === :grid && error("SPH_RENORM=sscheme is not available with the grid step (SPH_IDEAL=grid)" * hint)
    return nothing
end
