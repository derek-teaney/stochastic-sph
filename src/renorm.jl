# renorm.jl — the renormalized particle Hamiltonian as a Legendre transform (docs/legendre_plan.tex, SPH_RENORM=legendre).
#
# The Hamiltonian of a particle is the Legendre transform of the wanted single-particle free energy,
#   F_want(T, v, V) = −A V (γT)^n + T Δ,   Δ = (5/2) ln V + (5/2) ln γ + k ln(γT) + ½ ln(1 − c_s² v²) + C,
#   H(s, m, V) = F_want + T s + v |m|   at   s = −∂F_want/∂T,  |m| = −∂F_want/∂v,  m̂ = v̂,
# for the code's ConstantCs2EOS (p = A T_rel^n, n = 1 + 1/c_s², A = c_s² T_scale^(−n)), d = 2, neutral labels (B = 0), with
# k = 2 + 3/(2 c_s²), K = 5/2 + k and C = (3/2) ln(2π n A) − ½ ln c_s² + 1 (legendre_plan.tex, "The formulas"):
#   s   = n A V γ T_rel^(n−1) − Δ − k
#   |m| = n A V T_rel^n γ² v − T v [K γ² − c_s²/q],   q = 1 − c_s² v²
#   P   = −∂H/∂V = A T_rel^n − (5/2) T/V                (the particle's pressure; the forces use P V²)
# The unknowns of the solves are (λ, η) = (ln T_rel, the rapidity): γ = cosh η, γv = sinh η, T = T_rel/γ, so every Newton step stays
# in the domain and 1 − v² is never formed.  Forward: (s, |m|, V) → (T, v); inverse: (H, |m|, V) → (T, v), then s.  Each is a 2×2
# Newton solve with the analytic Jacobian; ∂(s, |m|)/∂(T, v) is the particle's susceptibility (symmetric: ∂s/∂v = ∂|m|/∂T), and
# ∂H/∂T = T ∂s/∂T + v ∂|m|/∂T, ∂H/∂v = T ∂s/∂v + v ∂|m|/∂v (dH = T ds + v d|m|).

# Solves, Newton iterations, failures (no convergence), states with no solution (no start: |M| ≥ ε or s ≤ 0; or the solve ends
# off the physical branch, det ∂(s,|m|)/∂(T,v) ≤ 0) and cold starts, summed over the run (legendre_counts(), reset_legendre_counts!()).
const LEG_SOLVES  = Threads.Atomic{Int}(0)
const LEG_ITERS   = Threads.Atomic{Int}(0)
const LEG_FAILS   = Threads.Atomic{Int}(0)
const LEG_NOSTATE = Threads.Atomic{Int}(0)
const LEG_COLD    = Threads.Atomic{Int}(0)
reset_legendre_counts!() = foreach(c -> (c[] = 0), (LEG_SOLVES, LEG_ITERS, LEG_FAILS, LEG_NOSTATE, LEG_COLD))
legendre_counts() = (solves = LEG_SOLVES[], iters = LEG_ITERS[], fails = LEG_FAILS[], nostate = LEG_NOSTATE[], cold = LEG_COLD[])

# the inputs of the last failed forward solve, for diagnosis: (s, S, M, e0, γ0)
const LEG_LASTFAIL = Ref{Any}(nothing)
const LEG_FAILLOG = Any[];  const LEG_FAILLOCK = ReentrantLock()      # the first 2000 failures, (:forward or :inverse, inputs, result)
logfail(x) = lock(() -> length(LEG_FAILLOG) < 2000 && push!(LEG_FAILLOG, x), LEG_FAILLOCK)
const LEG_TOL = 1e-14         # relative residual, both equations
const LEG_MAXIT = 8           # Newton iterations per solve from a warm start or a guess; a solve that fails then starts again cold
const LEG_MAXIT_COLD = 30     # from a cold start (near the light cone the iteration is slow: 8–10 steps at v ≈ 0.997)

# The constants of the model for an equation of state; an error for any other than ConstantCs2EOS.
struct LegendreConsts
    c2 :: Float64;  n :: Float64;  A :: Float64;  k :: Float64;  K :: Float64;  C :: Float64;  Ts :: Float64
end
function LegendreConsts(eos::ConstantCs2EOS)
    c2 = eos.cs2_val;  n = 1 + 1 / c2;  A = c2 * eos.T_scale^(-n);  k = 2 + 1.5 / c2
    return LegendreConsts(c2, n, A, k, 2.5 + k, 1.5 * log(2π * n * A) - 0.5 * log(c2) + 1, eos.T_scale)
end
LegendreConsts(eos::AbstractEOS) = error("SPH_RENORM=legendre: implemented for ConstantCs2EOS only (got $(typeof(eos)))")

# The particle at (λ, η) = (ln T_rel, rapidity) and volume V: s, |m|, H, P, T, v, γ and the derivatives of s, |m|, H with respect
# to (λ, η).  `counter = false` in the forward map of F_t alone (the tests: the counterterm off).
@inline function legendre_point(λ::Float64, η::Float64, V::Float64, c::LegendreConsts; counter::Bool = true)
    (; c2, n, A, k, K, C) = c
    Tr = exp(λ);  γ = cosh(η);  sh = sinh(η);  v = sh / γ;  T = Tr / γ
    q = 1 - c2 * v * v;  Trn = exp(n * λ)                      # T_rel^n
    a0 = n * A * V * Trn                                        # n A V T_rel^n = w V
    f = counter ? 1.0 : 0.0
    Δ = 2.5 * log(V) + 2.5 * log(γ) + k * λ + 0.5 * log(q) + C
    s  = a0 * γ / Tr - f * (Δ + k)
    bk = K * γ * γ - c2 / q
    mv = a0 * γ * γ - f * T * bk                                # |m|/v
    m  = mv * v
    F  = -A * V * Trn + f * T * Δ
    H  = F + T * s + v * m
    P  = A * Trn - f * 2.5 * T / V
    # derivatives in (T, v) (legendre_plan.tex), γ^n T^(n−2) = γ² T_rel^(n−2), T^(n−1) γ^(n+2) = γ³ T_rel^(n−1), T^n γ^(n+2) = γ² T_rel^n
    sT = n * (n - 1) * A * V * γ * γ * Trn / (Tr * Tr) - f * k / T
    sv = n * n * A * V * γ^3 * Trn / Tr * v - f * (K * γ * γ * v - c2 * v / q)              # = ∂|m|/∂T
    mvv = a0 * γ * γ * (1 + (n + 2) * sh * sh) - f * T * (K * (γ * γ + 2 * γ^2 * sh * sh) - c2 * (1 + c2 * v * v) / (q * q))
    HT = T * sT + v * sv;  Hv = T * sv + v * mvv
    # to (λ, η): ∂T/∂λ = T, ∂v/∂λ = 0, ∂T/∂η = −T v, ∂v/∂η = 1/γ²
    ig2 = 1 / (γ * γ)
    J = (sλ = T * sT, sη = -T * v * sT + sv * ig2,
         mλ = T * sv, mη = -T * v * sv + mvv * ig2,
         Hλ = T * HT, Hη = -T * v * HT + Hv * ig2,
         det_Tv = sT * mvv - sv * sv)                            # the particle's susceptibility determinant, > 0 on the physical branch
    return (s = s, m = m, H = H, P = P, T = T, v = v, γ = γ, Tr = Tr, J = J)
end

# Newton on (λ, η) for the residuals (x(λ, η) − x0, |m|(λ, η) − m0), x = s (forward) or H (inverse), with step halving.
@inline function legendre_newton(λ, η, x0, m0, V, c, inverse::Bool, maxit::Int; counter::Bool = true)
    # scales of the residuals; the label s may be small or negative (it is the target's entropy less Δ + k), so its scale has a floor
    xs = abs(x0) + (inverse ? 10.0 * exp(λ) : 10.0);  ms = m0 + (inverse ? xs : xs * exp(λ))
    pt = legendre_point(λ, η, V, c; counter)
    r1 = (inverse ? pt.H : pt.s) - x0;  r2 = pt.m - m0
    res = abs(r1) / xs + abs(r2) / ms
    its = 0
    while !(abs(r1) <= LEG_TOL * xs && abs(r2) <= LEG_TOL * ms)
        its >= maxit && return (λ, η, pt, its, false)
        its += 1
        a = inverse ? pt.J.Hλ : pt.J.sλ;  b = inverse ? pt.J.Hη : pt.J.sη
        cc = pt.J.mλ;  d = pt.J.mη;  det = a * d - b * cc
        (isfinite(det) && det != 0) || return (λ, η, pt, its, false)
        dλ = (d * r1 - b * r2) / det;  dη = (a * r2 - cc * r1) / det
        t = 1.0
        while true
            λn = λ - t * dλ;  ηn = η - t * dη
            ptn = legendre_point(λn, ηn, V, c; counter)
            r1n = (inverse ? ptn.H : ptn.s) - x0;  r2n = ptn.m - m0
            resn = abs(r1n) / xs + abs(r2n) / ms
            if (isfinite(resn) && resn <= res) || t < 1e-3 || (abs(t * dλ) + abs(t * dη) < 1e-15)
                λ, η, pt, r1, r2, res = λn, ηn, ptn, r1n, r2n, resn
                break
            end
            t /= 2
        end
        # at roundoff: the step no longer changes the point
        abs(dλ) + abs(dη) < 4e-16 * (1 + abs(λ) + abs(η)) && res <= 1e-12 && return (λ, η, pt, its, true)
    end
    return (λ, η, pt, its, true)
end

# converged: the Newton solve converged; ok: and on the physical branch
@inline function legendre_count(its, converged, ok, cold)
    Threads.atomic_add!(LEG_SOLVES, 1);  Threads.atomic_add!(LEG_ITERS, its)
    converged || Threads.atomic_add!(LEG_FAILS, 1);  converged && !ok && Threads.atomic_add!(LEG_NOSTATE, 1)
    cold && Threads.atomic_add!(LEG_COLD, 1)
end
@inline legendre_count_nostate() = (Threads.atomic_add!(LEG_SOLVES, 1);  Threads.atomic_add!(LEG_NOSTATE, 1))

# The start of a solve from a state (T, v⃗) of the particle (a previous solve), or nothing.
start_from(T, v) = (vm = norm(v);  γ = 1 / sqrt(1 - vm * vm);  (log(γ * T), asinh(γ * vm)))

# Forward: the state of the particle of entropy s, momentum m = M V and volume V = s/S (S, M the densities, as solve_eos!).  The label s
# may be negative: it is the target's entropy less Δ + k (≈ 25 at c_s² = 1/2, 45 at 1/4), and the canonical ensemble reaches it in the
# tails at s_a ≲ 50; the n-scheme keeps V = s/S = 1/n_a > 0 (S_smooth = s n_a).
# Warm start from the stored state (e0 = the target's rest-frame energy density at T_rel, γ0), or guess = (T, v⃗); cold start from
# the target's state at the entropy and momentum shifted by the counterterm.  Returns the fields of particle_forward (pointwise.jl);
# a failed solve returns NaN, and is counted.
function legendre_forward(s, S_in, M_vec, e0, γ0, eos; guess = nothing, counter::Bool = true)
    c = LegendreConsts(eos)
    nan = (e = NaN, p = NaN, T = NaN, γ = NaN, v = SVector(NaN, NaN), HS = NaN, FS = NaN)
    (isfinite(s) && s != 0 && isfinite(S_in) && s / S_in > 0 && all(isfinite, M_vec)) ||
        (legendre_count_nostate();  logfail((:forward_input, s, S_in, M_vec));  return nan)
    V = s / S_in;  m = M_vec * V;  mm = norm(m)
    cold = false
    if guess !== nothing
        λ, η = start_from(guess[1], guess[2])
    elseif e0 > 0 && isfinite(e0) && γ0 >= 1
        λ = log(Trel(eos, e0));  η = acosh(γ0)
    else
        cold = true
        λ, η = legendre_cold_forward(s, S_in, M_vec, mm, V, c, eos; counter)
    end
    λ, η, pt, its, ok = legendre_newton(λ, η, s, mm, V, c, false, cold ? LEG_MAXIT_COLD : LEG_MAXIT; counter)
    if !ok && !cold                                   # a poor warm start: once more from the cold start
        λ, η = legendre_cold_forward(s, S_in, M_vec, mm, V, c, eos; counter)
        λ, η, pt, its2, ok = legendre_newton(λ, η, s, mm, V, c, false, LEG_MAXIT_COLD; counter);  its += its2;  cold = true
    end
    conv = ok;  ok = ok && pt.J.det_Tv > 0 && pt.m >= 0
    legendre_count(its, conv, ok, cold)
    ok || (LEG_LASTFAIL[] = (s, S_in, M_vec, e0, γ0, conv, pt.J.det_Tv, pt.m);
           logfail((conv ? :forward_branch : :forward, s, s / S_in, norm(M_vec) * s / S_in, pt.T, pt.v, pt.J.det_Tv, e0, γ0, cold));  return nan)
    v = mm > 0 ? (pt.v / mm) * m : zero(SVector{2,Float64})
    return (e = (pt.Tr / c.Ts)^c.n, p = pt.P, T = pt.T, γ = pt.γ, v = v, HS = pt.H / s,
            FS = dot(v, M_vec / S_in) / S_in - pt.P / S_in^2)
end

# The cold start of the forward map.  On the physical branch the target's entropy and momentum of the state are s_t = s + Δ + k and
# |m|_t = |m| + T v [Kγ² − c_s²/q], with Δ logarithmic in T_rel; the fixed point s_t ← s + (Δ + k)(T(s_t)) (T from the target's state at
# (s_t, |m|_t)) contracts where (n − 1) s_t > k, the physical branch, so a few iterations from a hot start reach it (also for labels
# s ≤ 0, at small s_a).
function legendre_cold_forward(s, S_in, M_vec, mm, V, c, eos; counter::Bool = true)
    if !counter
        st = target_forward(s, S_in, M_vec, S_in, eos)
        return log(st.T * st.γ), asinh(st.γ * norm(st.v))
    end
    floor_t = 2 * c.k / (c.n - 1)                      # well inside the physical branch
    st_ = max(s + 2c.k, floor_t);  mt = mm;  e0 = 0.0;  λ = η = 0.0
    for _ in 1:6
        M2 = mm > 0 ? M_vec * (mt / mm) : M_vec
        st = target_forward(st_, st_ / V, M2, e0 > 0 ? e0 : st_ / V, eos);  e0 = st.e
        λ = log(st.T * st.γ);  η = asinh(st.γ * norm(st.v))
        pt = legendre_point(λ, η, V, c)
        Δk = c.n * c.A * V * pt.γ * exp((c.n - 1) * λ) - pt.s           # = Δ + k there
        dm = pt.T * pt.v * (c.K * pt.γ^2 - c.c2 / (1 - c.c2 * pt.v^2))
        st_new = max(s + Δk, floor_t / 2);  mt = mm + dm
        abs(st_new - st_) < 1e-3 * abs(st_) && break
        st_ = st_new
    end
    return λ, η
end

# Inverse: (s, v⃗, T) of the particle of energy e, momentum m and volume V; s = −Inf for a state with no solution (the solve fails or
# leaves the physical branch) or with γ > γ_max (SPH_GAMMA_MAX).  Start from guess = (T, v⃗), else from the target's inversion.
function legendre_inverse(e, m, V, eos; guess = nothing, counter::Bool = true)
    c = LegendreConsts(eos)
    fail = (-Inf, SVector(NaN, NaN), NaN)
    (isfinite(e) && isfinite(V) && V > 0) || return fail          # e ≤ 0 is possible: H = E_target − T(k + …) at small labels
    mm = norm(m)
    cold = guess === nothing
    if cold
        λ, η = legendre_cold_inverse(e, m, V, eos)
        isfinite(λ) || (legendre_count_nostate();  return fail)
    else
        λ, η = start_from(guess[1], guess[2])
    end
    λ, η, pt, its, ok = legendre_newton(λ, η, e, mm, V, c, true, cold ? LEG_MAXIT_COLD : LEG_MAXIT; counter)
    if !ok && !cold                                   # a poor guess: once more from the target's inversion
        λ, η = legendre_cold_inverse(e, m, V, eos)
        isfinite(λ) || (legendre_count(its, false, false, true);  return fail)
        λ, η, pt, its2, ok = legendre_newton(λ, η, e, mm, V, c, true, LEG_MAXIT_COLD; counter);  its += its2;  cold = true
    end
    # a solve that does not converge with e < |m| is counted as a state with no solution: any Legendre state has H − |m| > −kT (from
    # H = E_t − T(k + v²b), |m| = |m|_t − T v b, E_t > |m|_t), and these are proposals in the far tail (|m| ~ 7σ of the momentum)
    conv = ok || e < mm;  ok = ok && pt.J.det_Tv > 0 && pt.m >= 0
    legendre_count(its, conv, ok, cold)
    conv || logfail((:inverse, e, V, mm, pt.T, pt.v, pt.J.det_Tv, pt.s))
    ok || return fail
    pt.γ <= GAMMA_MAX || return fail
    v = mm > 0 ? (pt.v / mm) * m : zero(SVector{2,Float64})
    return (pt.s, v, pt.T)
end

# The cold start of the inverse map.  H = E_target(T, v) − T (k + v² [Kγ² − c_s²/q]) and |m| = |m|_target − T v [Kγ² − c_s²/q] at the
# same (T, v): the target's state at the energy and momentum raised by these terms, iterated (as legendre_cold_forward); without
# the cap on γ (the counterterm moves γ).  NaN if the target has no state at (e, m).
function legendre_cold_inverse(e, m, V, eos)
    c = LegendreConsts(eos);  mm = norm(m)
    et = max(e, 0.0) + mm + 1e-3 * (abs(e) + mm);  mt = mm;  λ = η = NaN     # a start the target can invert (ε > |M|)
    for _ in 1:6
        ε = et / V;  M = (mm > 0 ? m * (mt / mm) : m) / V
        (isfinite(ε) && ε > 0 && dot(M, M) < ε^2) || return (λ, η)
        S, v, T = eos_from_energy(ε, M, eos)
        (isfinite(T) && T > 0 && dot(v, v) < 1) || return (λ, η)
        λ, η = start_from(T, v)
        vv = norm(v);  γ = cosh(η);  bk = c.K * γ^2 - c.c2 / (1 - c.c2 * vv^2)
        etn = max(e + T * (c.k + vv^2 * bk), (mm + T * vv * bk) * (1 + 1e-9));  mt = mm + T * vv * bk
        abs(etn - et) < 1e-3 * abs(et) && break
        et = etn
    end
    return λ, η
end

# The state at (T, |v|, V) in the plan's variables, for the tests: (s, |m|, H, P).
function legendre_at(T, v, V, eos; counter::Bool = true)
    γ = 1 / sqrt(1 - v^2);  pt = legendre_point(log(γ * T), atanh(v), V, LegendreConsts(eos); counter)
    return (s = pt.s, m = pt.m, H = pt.H, P = pt.P)
end

# Errors for the schemes the Legendre Hamiltonian does not support (legendre_plan.tex, C).
function check_legendre(params)
    LEGENDRE || return nothing
    params.volume === :number && params.hamiltonian === :energy ||
        error("SPH_RENORM=legendre needs hamiltonian = :energy and volume = :number (in the entropy-volume scheme H/s serves as ∂H/∂s)")
    sim_dim(params) == 2 && params.phys_dim == 2 || error("SPH_RENORM=legendre: implemented for d = 2")
    params.measure === :liouville || error("SPH_RENORM=legendre needs the Liouville measure (SPH_MEASURE=liouville)")
    params.ideal === :grid && error("SPH_RENORM=legendre is not available with the grid step (SPH_IDEAL=grid)")
    return nothing
end
