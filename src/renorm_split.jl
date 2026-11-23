# renorm_split.jl — the split renormalization, Step 5 (docs/step5_plan.tex, SPH_RENORM=split; docs/partition.tex, "The label s_eff").
#
# The one-loop counterterm of the single particle, at the target state (T, v) of the label u, the momentum m and the volume V,
#   δln Z_1 = Δs_lab(u) + Δs_dyn(T, v, V),   Δs_lab = b ln u + const,   b = 5/2 − 𝒩,   𝒩 = 1 − 2 c_s²,
#   Δs_dyn = 𝒩 ln(γ V / V̄) + ½ ln(1 − c_s² v²),
# is split: the label part stays in the measure, as in SPH_RENORM=1 (flat in s_a = u − b ln u, with the Jacobian 1 − b/u), and the rest
# goes into the Hamiltonian as an energy,
#   H(u, m, V) = e_t(u, m, V) + E_dyn,   E_dyn = T Δs_dyn
# (the term T ∂Δs_dyn/∂u of the plan is left out: step5_plan.tex, "Results", A).  pa.s is the label u (the target's entropy), as with
# SPH_RENORM=1.  The forward map fills (step5_plan.tex, eq. fields)
#   pa.T = ∂H/∂u = T + ∂E_dyn/∂u,   ẋ = ∂H/∂m = (v + ∂E_dyn/∂|m|) m̂,   pa.p = −∂H/∂V = P_t − ∂E_dyn/∂V,   H = e_t + E_dyn,
# the derivatives at fixed particle variables by the chain rule through the target's (T, v):
#   ∂g/∂u|_{m,V} = (g_T m_v − g_v u_v)/det,   ∂g/∂|m||_{u,V} = (g_v u_T − g_T u_v)/det,   ∂g/∂V|_{u,m} = g_V − (u ∂g/∂u + |m| ∂g/∂|m|)/V,
# u_T = (n−1) u/T, u_v = m_T = n γ² v u, m_v = n A V γ^(n+2) T^n (1 + (n+2) γ² v²), det = u_T m_v − u_v² (ConstantCs2EOS: p = A T_rel^n,
# n = 1 + 1/c_s²).  A state with ∂H/∂u ≤ 0 or |ẋ| ≥ 1 has no Hamiltonian (NaN fields; hybrid Monte Carlo rejects the trajectory).
# Inverse (the votes, the heat bath): (e, m, V) → u by Newton in u alone, H(u, m, V) = e, from the target's inversion at e.
# Vote weight (Liouville measure): u − b ln u − ln(∂H/∂u / (1 − b/u)), −∞ for u ≤ b (the label boundary).

const SPLIT_CNT = zeros(Int, 8, Threads.maxthreadid())    # rows: solves, iterations, failures (no convergence), no state, cold retries
@inline splitcnt!(i, x = 1) = (@inbounds SPLIT_CNT[i, Threads.threadid()] += x;  nothing)
reset_split_counts!() = fill!(SPLIT_CNT, 0)
split_counts() = (t = sum(SPLIT_CNT; dims = 2);  (solves = t[1], iters = t[2], fails = t[3], nostate = t[4], cold = t[5]))

# V̄, the reference volume of Δs_dyn (SPH_RENORM_VBAR; default the mean volume L²/N of the run, set by check_split)
const SPLIT_VBAR = Ref(NaN)
const SPLIT_FAILLOG = Any[];  const SPLIT_FAILLOCK = ReentrantLock()   # the first 200 solves that did not converge: (e, m, V, u, H − e, ∂H/∂u)
splitlog(x) = lock(() -> length(SPLIT_FAILLOG) < 200 && push!(SPLIT_FAILLOG, x), SPLIT_FAILLOCK)
const SPLIT_TOL = 1e-12      # relative residual of the inverse's Newton solve in u: the accuracy of the target's forward solve
                             # (eos_newton, tol 1e-12), below which H(u) does not resolve u (the steps stall at ~1e-13)
const SPLIT_MAXIT = 20

struct SplitConsts
    c2 :: Float64;  n :: Float64;  A :: Float64;  N :: Float64;  b :: Float64;  lVbar :: Float64
end
function SplitConsts(eos::ConstantCs2EOS)
    c2 = eos.cs2_val;  n = 1 + 1 / c2;  N = 1 - 2c2
    N != 0 && !isfinite(SPLIT_VBAR[]) && error("SPH_RENORM=split: V̄ not set (check_split(params), called when SimParams is built)")
    return SplitConsts(c2, n, c2 * eos.T_scale^(-n), N, 2.5 - N, log(SPLIT_VBAR[]))
end
SplitConsts(eos::AbstractEOS) = error("SPH_RENORM=split: implemented for ConstantCs2EOS only (got $(typeof(eos)))")

# E_dyn and its derivatives at fixed particle variables, at the target state (T, |v|, γ) of the label u, |m| and V.
# sound = false drops the sound term ½ ln(1 − c_s² v²) (the tests).
@inline function split_dyn(T, v, γ, u, mm, V, c::SplitConsts; sound::Bool = true)
    (; c2, n, A, N) = c
    q = 1 - c2 * v * v;  sf = sound ? 1.0 : 0.0
    Δ = (N == 0 ? 0.0 : N * (log(γ * V) - c.lVbar)) + sf * 0.5 * log(q)     # 𝒩 = 0 (c_s² = 1/2): V̄ does not enter
    gT = Δ;  gv = T * (N * γ * γ * v - sf * c2 * v / q);  gV = T * N / V
    uT = (n - 1) * u / T;  uv = n * γ * γ * v * u
    mv = n * A * V * γ^(n + 2) * T^n * (1 + (n + 2) * γ * γ * v * v)
    det = uT * mv - uv * uv
    gu = (gT * mv - gv * uv) / det;  gm = (gv * uT - gT * uv) / det
    return (E = T * Δ, Eu = gu, Em = gm, EV = gV - (u * gu + mm * gm) / V, det = det)
end

# The state of the particle of label u (= s) at the entropy density S = u/V and momentum density M (warm start e0 of the target).
function split_forward(s, S_in, M_vec, e0, eos; M_mag = sqrt(M_vec[1]^2 + M_vec[2]^2), sound::Bool = true,
                       tol = 1e-12, maxiter = 50, verbose = false)
    c = SplitConsts(eos)
    st = target_forward(s, S_in, M_vec, e0, eos; M_mag, tol, maxiter, verbose)
    V = s / S_in;  vt = norm(st.v);  T = st.T
    d = split_dyn(T, vt, st.γ, s, M_mag * V, V, c; sound)
    TH = T + d.Eu;  xd = vt + d.Em;  p = st.p - d.EV
    if !(TH > 0 && abs(xd) < 1 && isfinite(p))
        return (e = NaN, p = NaN, T = NaN, γ = NaN, v = SVector(NaN, NaN), HS = NaN, FS = NaN)
    end
    v = vt > 0 ? st.v * (xd / vt) : st.v
    return (e = st.e, p = p, T = TH, γ = st.γ, v = v, HS = st.HS + d.E / s, FS = dot(v, M_vec / S_in) / S_in - p / S_in^2)
end

# H(u, m, V), ∂H/∂u and ẋ at the label u (the inverse's Newton steps), from a target warm start e0
@inline function split_H(u, mm, mvec, V, e0, eos, c; sound::Bool = true)
    S = u / V;  M = mvec / V
    st = target_forward(u, S, M, e0, eos)
    vt = norm(st.v)
    d = split_dyn(st.T, vt, st.γ, u, mm, V, c; sound)
    return (H = u * st.HS + d.E, TH = st.T + d.Eu, xd = vt + d.Em, v = st.v, e = st.e, γ = st.γ)
end

# Newton in u for H(u, m, V) = e from (u, e0) (e0 the target's rest-frame energy density, the warm start of its forward solve).
# Returns (status, u, r, its): status :conv (r the state at the last iterate, u the converged label), :nostate (the solution, if any,
# is below the label boundary b, where the weight is zero) or :fail (no convergence).
@inline function split_newton(u, e0, e, m, mm, V, eos, c; sound::Bool = true)
    its = 0;  r = split_H(u, mm, m, V, e0, eos, c; sound)
    for it in 1:SPLIT_MAXIT
        it > 1 && (r = split_H(u, mm, m, V, e0, eos, c; sound))
        its += 1;  e0 = r.e
        (isfinite(r.H) && r.TH > 0) || return (:fail, u, r, its)
        du = (r.H - e) / r.TH
        un = u - du
        if un <= c.b                                           # below the label boundary b: the solution, if any, is above it
            rb = split_H(c.b, mm, m, V, e0, eos, c; sound)    # H is increasing in u: e ≤ H(b) has no state with weight
            (isfinite(rb.H) && rb.H < e) || return (:nostate, u, r, its)
            un = (u + c.b) / 2
        end
        if abs(r.H - e) <= SPLIT_TOL * (abs(e) + abs(r.H)) || abs(du) <= 1e-13 * abs(u)
            return (:conv, un, r, its)
        end
        u = un
    end
    return (:fail, u, r, its)
end

# The cold start: the target's label at the same energy; where the target has no state there (E_dyn < 0 lowers H below the target's
# energy, so e can be below what the target needs at this momentum: |M| ≥ ε), at an energy raised step by step until it has one;
# a start at or below the label boundary (where ∂H/∂u can be < 0) moves to 2b, or there is no state.  Returns (u, e0) or (NaN, NaN).
function split_cold_start(e, m, mm, V, eos, c; sound::Bool = true)
    u, v0, T0 = target_inverse(e, m, V, eos)
    δe = 1e-3 * (abs(e) + mm)
    for _ in 1:40
        (isfinite(u) && u > 0) && break
        u, v0, T0 = target_inverse(max(e, mm) + δe, m, V, eos);  δe *= 2
    end
    (isfinite(u) && u > 0) || return (NaN, NaN)
    if u <= c.b
        rb = split_H(c.b, mm, m, V, 1.0, eos, c; sound)
        (isfinite(rb.H) && rb.H < e) || return (NaN, NaN)
        return (2 * c.b, rb.e)
    end
    return (u, (T0 / sqrt(1 - dot(v0, v0)) / eos.T_scale)^(1 + 1 / eos.cs2_val))
end

# The fast inverse: Newton in the target's energy e_t for e_t + E_dyn = e, each step the target's closed-form inversion at (e_t, m, V)
# (eos_from_energy; no forward solve) and E_dyn at its (T, v): g(e_t) = e_t + E_dyn − e, dg/de_t = 1 + (∂E_dyn/∂u)/T (∂e_t/∂u = T at
# fixed m, V).  Starts from e_t = e − Eg.  Returns (status, u, T, v⃗, d, its).
@inline function split_newton_et(e, m, mm, V, eos, c, Eg; sound::Bool = true)
    et = e - Eg;  its = 0;  δprev = Inf
    for it in 1:SPLIT_MAXIT
        u, v, T = target_inverse(et, m, V, eos)
        (isfinite(u) && u > 0 && T > 0) || return (:fail, NaN, NaN, v, nothing, its)
        vt = norm(v);  γ = 1 / sqrt(1 - vt * vt)
        d = split_dyn(T, vt, γ, u, mm, V, c; sound);  its += 1
        g = et + d.E - e
        dg = 1 + d.Eu / T
        (isfinite(g) && dg > 0) || return (:fail, u, T, v, d, its)
        δ = g / dg
        # every step is closed form, so the iteration runs to rounding: until the step is at the rounding of e_t, or stops shrinking
        # (a test relative to e alone stops early near the light cone, where e/(T u) ~ γ²)
        if g == 0 || abs(δ) <= 4e-16 * (abs(e) + abs(et)) || (abs(δ) >= abs(δprev) && abs(δ) <= 1e-10 * (abs(e) + abs(et)))
            return (:conv, u, T, v, d, its)
        end
        δprev = δ;  et -= δ
    end
    return (:fail, NaN, NaN, SVector(NaN, NaN), nothing, its)
end

# Inverse: (u, ẋ, ∂H/∂u) of the particle of energy e, momentum m and volume V; u = −Inf for a state with no solution, below the label
# boundary, or with γ > γ_max.  First the fast Newton in the target's energy (split_newton_et); where it fails (the target has no
# state at an iterate: E_dyn < 0 can put e_t below what the target needs at this momentum), the Newton in u (split_newton) from the
# cold start.  guess (the particle's label and target energy density, set by the votes and the heat bath) is accepted and not used.
function split_inverse(e, m, V, eos; guess = nothing, sound::Bool = true)
    c = SplitConsts(eos)
    fail = (-Inf, SVector(NaN, NaN), NaN)
    splitcnt!(1)
    (isfinite(e) && isfinite(V) && V > 0 && all(isfinite, m)) || (splitcnt!(4);  return fail)
    mm = norm(m)
    st, u, T, v, d, its = split_newton_et(e, m, mm, V, eos, c, 0.0; sound)     # from e_t = e (the guess is not needed: the start is closed form)
    if st === :conv
        splitcnt!(2, its)
        u > c.b || (splitcnt!(4);  return fail)
        TH = T + d.Eu;  vt = norm(v);  xd = vt + d.Em
        (TH > 0 && abs(xd) < 1 && 1 / sqrt(1 - vt * vt) <= GAMMA_MAX) || (splitcnt!(4);  return fail)
        return (u, vt > 0 ? v * (xd / vt) : v, TH)
    end
    # fallback: Newton in u from the cold start
    splitcnt!(5)
    u0, e0 = split_cold_start(e, m, mm, V, eos, c; sound)
    isfinite(u0) || (splitcnt!(2, its);  splitcnt!(4);  return fail)
    st, u, r, its2 = split_newton(u0, e0, e, m, mm, V, eos, c; sound);  its += its2
    splitcnt!(2, its)
    st === :nostate && (splitcnt!(4);  return fail)
    st === :conv || (splitcnt!(3);  splitlog((e, m, V, u, r.H - e, r.TH));  return fail)
    r = split_H(u, mm, m, V, r.e, eos, c; sound)
    (r.TH > 0 && abs(r.xd) < 1 && r.γ <= GAMMA_MAX) || (splitcnt!(4);  return fail)
    vt = norm(r.v)
    return (u, vt > 0 ? r.v * (r.xd / vt) : r.v, r.TH)
end

# The weight of a state in the votes: s_a(u) = u − b ln u, and with the Liouville measure − ln(∂H/∂u / (1 − b/u)) (∂H/∂s_a).
@noinline function split_vote_entropy(e, m, V, eos, liouville::Bool; guess = nothing, sound::Bool = true)
    u, _, TH = split_inverse(e, m, V, eos; guess, sound)
    (isfinite(u) && TH > 0) || return -Inf
    b = 2.5 - (1 - 2 * eos.cs2_val)
    u > b || return -Inf                                       # the label boundary
    s = u - b * log(u)
    return liouville ? s - log(TH / (1 - b / u)) : s
end

# The schemes Step 5 supports (as check_legendre), and V̄.
function check_split(params)
    SPLIT || return nothing
    params.volume === :number && params.hamiltonian === :energy ||
        error("SPH_RENORM=split needs hamiltonian = :energy and volume = :number")
    sim_dim(params) == 2 && params.phys_dim == 2 || error("SPH_RENORM=split: implemented for d = 2")
    params.measure === :liouville || error("SPH_RENORM=split needs the Liouville measure (SPH_MEASURE=liouville)")
    params.ideal === :grid && error("SPH_RENORM=split is not available with the grid step (SPH_IDEAL=grid)")
    isfinite(SPLIT_VBAR[]) || (SPLIT_VBAR[] = setting("SPH_RENORM_VBAR", prod(params.L) / params.N))
    return nothing
end
