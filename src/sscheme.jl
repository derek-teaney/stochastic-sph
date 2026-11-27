# sscheme.jl — the S-scheme, state counting at one loop (docs/renorm3.tex; docs/sscheme_plan.tex, "Level 1", item B, and "Level 1, amended").
#
# The default form (SPH_SSCHEME_FORM=measure) moves bᵨ ln S_t of Δ into the measure of the labels: see SSchemeConsts below.  The header
# describes the one-loop label S_t − Δ_χ − Δ_ℓ that the measure form is built on (it was the full form, removed on 2026-11-26).
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

const SS_CNT = zeros(Int, 8, Threads.maxthreadid())     # rows: solves, iterations, fallbacks (bisection), no state, states from the energy,
                                                        # those refused at the wall only (measure form), solves with P_a < 0
@inline sscnt!(i, x = 1) = (@inbounds SS_CNT[i, Threads.threadid()] += x;  nothing)
reset_sscheme_counts!() = fill!(SS_CNT, 0)
sscheme_counts() = (t = sum(SS_CNT; dims = 2);  (solves = t[1], iters = t[2], fallbacks = t[3], nostate = t[4], fromenergy = t[5], wall = t[6], negP = t[7]))

const SS_TOL = 1e-13          # relative change of S_t at which the map from the label stops
const SS_MAXIT = 30           # Newton steps before the bisection
const SS_D = 3                # D = d + 1 (d = 2)

# The form of the scheme (sscheme_plan.tex, "Level 1, amended"; renorm3.tex, "The bounds, ..."), SPH_SSCHEME_FORM:
#   (full    s = S_t − Δ_χ − Δ_ℓ with the flat measure: removed on 2026-11-26, superseded by exact; its one-loop core stays, for measure)
#   measure  (default) bᵨ ln S_t of Δ moved into the measure of the labels (bᵨ = b_ρ, "the b of the measure ρ"; it was β until 2026-11-26): s = S_t − Δ^H, Δ^H = Δ − bᵨ ln S_t − const, measure
#            ρ(s) = e^s s^(−bᵨ) (1 − bᵨ/s) on s > bᵨ (the wall).  For a constant speed of sound
#              Δ^H = (D/2 + 1 − bᵨ) ln(S_t/S̄_t) − 𝒩 ln(s_rel/s̄_rel) + ½ ln(a/q),   𝒩 = 1 − d c_s²,
#            zero at the reference state of the run at rest (S̄_t = s̄_rel V̄); bᵨ = D/2 + 1 (SPH_SSCHEME_BRHO, default): Δ^H = −(q − d) ln(T_rel/T_0)
#            + ½ ln(a/q), the owner's eq. amend (F = 1), and only s̄_rel enters; bᵨ = b = D/2 + d c_s²: Δ^H = 𝒩 ln(γV/V̄) + ½ ln(a/q), the split of
#            SPH_RENORM=split, and only V̄ enters.  As a term added to Δ the change is M = −bᵨ ln S_t + C, with slopes (−bᵨq, −bᵨv, −bᵨ) along
#            (τ = ln T_rel, rapidity, ln V); it shifts Δ_p by −bᵨ(q − v²), Δ_θ by −bᵨv and P_a V/T_a by bᵨ(1 − q/n).
#   The reference (s̄_rel, V̄) is set from the initial condition (sscheme_reference!, called by sscheme_labels!): s̄_rel the mean rest-frame
#   entropy density of the targets, V̄ = L²/N, as split sets its V̄.  measure-f (the factor F of eq. schemeF) is reserved.
#   exact    s = S_1loop + δ(ln S_t, θ), the exact count of states of the particle, flat measure (sscheme_exact.jl; c_s² = 1/2 only).
const SS_FORM = setting("SPH_SSCHEME_FORM", :measure)
SS_FORM in (:measure, :exact) || error("SPH_SSCHEME_FORM = $SS_FORM: measure (the default) or exact.  The full form (flat measure, the one-loop label) " *
    "was removed on 2026-11-26: the exact form is the full form with the exact count (docs/sscheme_plan.tex, sec:prune; last commit with it 9846a6c)")
const SS_MEASURE = SS_FORM === :measure
const SS_BRHO = setting("SPH_SSCHEME_BRHO", SS_D / 2 + 1)
const SS_REF = Ref((NaN, NaN))          # (s̄_rel, V̄)

struct SSchemeConsts
    c2 :: Float64;  q :: Float64;  n :: Float64;  A :: Float64
    Pconst :: Float64         # P_a V/T_a = S_t/n + Pconst
    bᵨ :: Float64             # b_ρ, the exponent of the measure of the labels (0 in the exact form)
    C :: Float64              # the constant of M = −bᵨ ln S_t + C (Δ^H = 0 at the reference state)
end
# The label of the full form at rest, (S_t, V) (closed form: T_rel^q = S_t/(nAV))
function sscheme_label_full_rest(St, V, c2, q, n, A)
    Tr = (St / (n * A * V))^(1 / q);  w = n * A * Tr^n
    return St - (SS_D / 2) * log(2π * V * w * Tr) - 0.5 * log(q) - (1 + log(V) - log(Tr))
end
function SSchemeConsts(eos::ConstantCs2EOS)
    c2 = eos.cs2_val;  q = 1 / c2;  n = 1 + q;  A = c2 * eos.T_scale^(-n)
    Pc = ((SS_D / 2) * (2 + q) - 1) / n - (SS_D / 2 + 1)
    SS_FORM === :exact && check_sscheme_exact(eos)
    SS_MEASURE || return SSchemeConsts(c2, q, n, A, Pc, 0.0, 0.0)
    bᵨ = SS_BRHO;  N = 1 - (SS_D - 1) * c2;  k = SS_D / 2 + 1 - bᵨ
    srel, Vb = SS_REF[]
    if !(isfinite(srel) && isfinite(Vb))
        (abs(N) < 1e-14 && abs(k) < 1e-14) || error("SPH_SSCHEME_FORM=measure: the reference state is not set (sscheme_reference!, called by sscheme_labels!)")
        srel, Vb = 1.0, 1.0           # conformal with bᵨ = D/2 + 1: Δ^H = ½ ln(a/q), the reference does not enter
    end
    Stb = srel * Vb
    C = sscheme_label_full_rest(Stb, Vb, c2, q, n, A) + bᵨ * log(Stb) - Stb     # s(reference, rest) = S̄_t
    return SSchemeConsts(c2, q, n, A, Pc + bᵨ * (1 - q / n), bᵨ, C)
end
# The reference state of the measure form: s̄_rel (rest-frame entropy density) and V̄
sscheme_reference!(srel::Real, Vbar::Real) = (SS_REF[] = (Float64(srel), Float64(Vbar));  nothing)
# The logarithm of the measure of the labels: s (full), ln ρ(s) = s − bᵨ ln s + ln(1 − bᵨ/s), −Inf for s ≤ bᵨ (measure)
@inline ss_lnrho(s::Float64, c::SSchemeConsts) = !SS_MEASURE ? s : (s > c.bᵨ ? s - c.bᵨ * log(s) + log1p(-c.bᵨ / s) : -Inf)
SSchemeConsts(eos::AbstractEOS) = error("SPH_RENORM=sscheme: implemented for ConstantCs2EOS only (got $(typeof(eos)))")

const SS_NOSTATE = (ok = false, s = -Inf, St = NaN, H = NaN, Ta = NaN, va = NaN, Pa = NaN, D = NaN, weight = -Inf,
                    γ = NaN, Trel = NaN, T = NaN, v = NaN, erest = NaN)
const SS_WALL = merge(SS_NOSTATE, (D = Inf,))          # no state only because the label is at or below the wall (the counters)

# Everything about the particle at a target state: T_rel, uγ = γv ≥ 0 (along m) and V.
# cap = true: no state above the Lorentz factor γ_max (SPH_GAMMA_MAX), the map from the energy only (the votes, the heat bath), as at the tree level
# and split; the map from the label (the ideal step) has no cap, so that a particle the ideal step carries above γ_max keeps its state.
@inline function sscheme_state(Tr::Float64, uγ::Float64, V::Float64, c::SSchemeConsts; cap::Bool = true)
    (; q, n, A) = c
    γ = sqrt(1 + uγ * uγ);  v = uγ / γ;  v2 = v * v
    (isfinite(Tr) && Tr > 0 && (!cap || γ <= GAMMA_MAX)) || return SS_NOSTATE
    a = q - v2
    p = A * Tr^n;  w = n * p;  St = γ * (w / Tr) * V
    SS_FORM === :exact && return sscheme_state_exact(Tr, γ, v, uγ, a, p, w, St, V, c)
    Dχ = (SS_D / 2) * log(2π * V * γ * w * Tr) + 0.5 * log(a)
    Dℓ = 1 + log(V) + log(γ) - log(Tr)
    s = St - Dχ - Dℓ
    Dp = (SS_D / 2) * (2 + q - v2) + v2 * (1 - v2) / a - (1 + v2)
    Dθ = (SS_D / 2) * v - v * (1 - v2) / a + v
    if SS_MEASURE                                              # Δ → Δ^H = Δ + M, M = −bᵨ ln S_t + C
        s += c.bᵨ * log(St) - c.C
        Dp -= c.bᵨ * (q - v2);  Dθ -= c.bᵨ * v
    end
    Pω = 0.0
    if POS_ON                                                  # the positional counterterm, s → s + ω(S_t/n, V) (positional.jl)
        o = positional_omega(St / n, V)
        s += o.ω;  Dp -= a * o.ωσ;  Dθ -= v * o.ωσ;  Pω += o.ωσ / n + o.ωV
    end
    εp = Dp / (St * a);  εpar = Dθ / St
    D = 1 - (1 + v2) * εp + v * εpar
    T = Tr / γ
    Ta = T / D;  va = (v * (1 - 2εp) + εpar) / D
    (D > 0 && abs(va) < 1) || return SS_NOSTATE
    SS_MEASURE && s <= c.bᵨ && return SS_WALL                 # measure: the wall s > bᵨ
    Pa = (St / n + c.Pconst + Pω) * Ta / V
    return (ok = true, s = s, St = St, H = V * (w * γ * γ - p), Ta = Ta, va = va, Pa = Pa, D = D, weight = ss_lnrho(s, c) - log(Ta),
            γ = γ, Trel = Tr, T = T, v = v, erest = p / c.c2)
end

# The exact form (SPH_SSCHEME_FORM=exact, sscheme_exact.jl): s = s_min + ln A + g(ln S_t, θ) (+ ω), the state from the slopes of s directly.
@inline function sscheme_state_exact(Tr, γ, v, uγ, a, p, w, St, V, c::SSchemeConsts)
    n = c.n;  v2 = v * v
    g, Gx, Gθ = exact_g(log(St), asinh(uγ))
    s = ssx_table().smin + log(c.A) + g
    ωV = 0.0
    if POS_ON
        o = positional_omega(St / n, V)
        s += o.ω;  Gx += o.ωσ;  ωV = o.ωV
    end
    y = (a * Gx - v * Gθ) / (a * St);  z = (v * Gx + Gθ) / St
    D = (1 + v2) * y - v * z
    T = Tr / γ
    Ta = T / D;  va = (2v * y - z) / D
    (D > 0 && abs(va) < 1) || return SS_NOSTATE
    Pa = (Gx / n + ωV) * Ta / V
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
    st = sscheme_state(Tr, uγ, V, c)
    sscnt!(5);  st.D == Inf && sscnt!(6)
    return st
end

# The map from the label: (s, m, V) → the state of the particle, by Newton's iteration on S_t from St0 (the particle's previous S_t),
# with a bisection on S_t when Newton leaves the domain or does not converge.  ok = false if no state has the label s.
function sscheme_from_label(s::Float64, m::SVector{2,Float64}, V::Float64, c::SSchemeConsts, St0::Float64)
    sscnt!(1)
    mmag = norm(m)
    st_at(St) = (r = sscheme_target_entropy(St, mmag, V, c);  isfinite(r[1]) ? sscheme_state(r[1], r[2], V, c; cap = false) : SS_NOSTATE)
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
# The first 200 solves without a state: (s, m, V, S_t of the warm start), for diagnosis (sscheme_faillog())
const SS_FAILLOG = Any[];  const SS_FAILLOCK = ReentrantLock()
sscheme_faillog() = SS_FAILLOG
function sscheme_solve!(pa::Particle, eos::AbstractEOS)
    c = SSchemeConsts(eos)
    st = sscheme_from_label(pa.s, pa.m, pa.V, c, pa.St)
    !st.ok && lock(() -> length(SS_FAILLOG) < 200 && push!(SS_FAILLOG, (s = pa.s, m = pa.m, V = pa.V, St0 = pa.St, x = pa.x)), SS_FAILLOCK)
    if !st.ok
        pa.e = NaN;  pa.p = NaN;  pa.T = NaN;  pa.γ = NaN;  pa.H = NaN;  pa.v = SVector(NaN, NaN)
        pa.v_over_S = SVector(NaN, NaN);  pa.H_over_S = NaN;  pa.F_over_S = NaN
        return nothing
    end
    st.Pa < 0 && sscnt!(7)
    pa.e = st.erest;  pa.p = st.Pa;  pa.T = st.Ta;  pa.γ = st.γ;  pa.H = st.H;  pa.St = st.St
    pa.v = sscheme_velocity(st, pa.m)
    pa.S_smooth = st.St / pa.V;  pa.M_smooth = pa.m / pa.V
    pa.v_over_S = pa.v / pa.S_smooth;  pa.H_over_S = NaN;  pa.F_over_S = NaN
    return nothing
end

# The set-up gives each particle its target entropy in pa.s (S_func V, setup.jl): at the smoothed volume, replace it by the label of the state
# with that target entropy and the particle's momentum.
function sscheme_labels!(particles, params, cl, eos)
    smooth_to_particles!(particles, params, cl)
    if SS_MEASURE                     # the reference of the measure form: the mean rest-frame entropy density of the targets, V̄ = L²/N
        c0 = SSchemeConsts(eos.cs2_val, 1 / eos.cs2_val, 1 + 1 / eos.cs2_val, eos.cs2_val * eos.T_scale^(-1 - 1 / eos.cs2_val), 0.0, 0.0, 0.0)
        sr = 0.0
        for pa in particles
            Tr, _ = sscheme_target_entropy(pa.s, norm(pa.m), pa.V, c0);  sr += c0.n * c0.A * Tr^c0.q
        end
        sscheme_reference!(sr / length(particles), prod(params.L) / params.N)
        println(@sprintf("SPH_SSCHEME_FORM=measure, b_ρ = %.4g: reference s̄_rel = %.10g, V̄ = %.10g (S̄_t = %.6g)", SS_BRHO, SS_REF[]..., prod(SS_REF[])))
    end
    c = SSchemeConsts(eos)
    for pa in particles
        St = pa.s;  r = sscheme_target_entropy(St, norm(pa.m), pa.V, c)
        st = sscheme_state(r[1], r[2], pa.V, c)
        st.ok || error("SPH_RENORM=sscheme: the set-up state S_t = $St, |m| = $(norm(pa.m)), V = $(pa.V) is outside the domain")
        pa.s = st.s;  pa.St = St
    end
    return nothing
end

# The target's T_rel (the transport coefficients, the time step): pa.T γ at the tree level (and split, as before); with sscheme
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
# The vote weight: ln ρ(s) − ln T_a (s − ln T_a in the full form), no solve.
sscheme_vote_entropy(e, m, V, eos) = sscheme_from_energy(e, m, V, SSchemeConsts(eos)).weight

# The schemes the S-scheme supports (as check_split).
function check_sscheme(params)
    SSCHEME || return nothing
    positional_set!(params)
    hint = "; set SPH_RENORM=split (the default) or SPH_RENORM=0"
    params.phys_dim == 2 || error("SPH_RENORM=sscheme: implemented for d = 2" * hint)
    return nothing
end
