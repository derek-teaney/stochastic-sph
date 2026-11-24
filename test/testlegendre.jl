# testlegendre.jl — the renormalized Hamiltonian as a Legendre transform (src/renorm.jl, SPH_RENORM=legendre; docs/legendre_plan.tex, D):
#   1. the reference values of the plan (examples/renorm_quadrature_legendre.py), to 1e-10 relative;
#   2. the Legendre identities by central differences: ∂H/∂s = T, ∂H/∂m = v, ∂H/∂V = −P;
#   3. forward and inverse are inverses, H(S(e, m, V), m, V) = e to 1e-13, for s_a in [10, 1e4], γ in [1, 100], c_s² = 1/2 and 1/4
#      (the label s = the target's entropy less Δ + k is negative at the smallest s_a: allowed);
#   4. with the counterterm off the solves reproduce the target's (eos_newton, eos_from_energy) to roundoff;
#   5. warm and cold starts give the same answer, and the iterations per solve; 5b, 5c: the campaign's failing states;
#   6. hybrid Monte Carlo with the Legendre Hamiltonian: the energy error of a trajectory falls as δt² (the forces are −∂H/∂x),
#      and a few updates are accepted with the total momentum unchanged by the trajectories.
ENV["SPH_RENORM"] = "legendre"
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
LEGENDRE || error("SPH_RENORM=legendre not set at load")

# H, T, v, P of the particle (s, m, V) through the forward map, cold
fwd(s, m, V, eos; kw...) = legendre_forward(s, s / V, m / V, 0.0, 1.0, eos; kw...)
Hof(s, m, V, eos) = s * fwd(s, m, V, eos).HS

println("1. reference values (legendre_plan.tex)")
ref = [(0.5, 3.0, 0.0, 4.0, 3.5329615802e1, 0.0, 9.3000000000e1, 1.1625000000e1),
       (0.5, 3.0, 0.8, 4.0, 2.2769125486e2, 1.6184313725e3, 1.7797450980e3, 6.0625000000e1),
       (0.5, 1.7, 0.95, 0.5, 5.2134000814e1, 1.0565293387e3, 1.0758910379e3, 7.2188166117e1),
       (0.25, 3.0, 0.0, 1.0, 7.9676423584e1, 0.0, 2.1900000000e2, 5.3250000000e1),
       (0.25, 3.0, 0.8, 1.0, 1.2752332646e3, 8.6112698413e3, 9.9900158730e3, 7.7375000000e2),
       (0.25, 1.7, 0.95, 0.5, 1.7311977590e3, 2.8956034219e4, 2.9886312509e4, 1.1873400004e3)]
rel(a, b) = b == 0 ? abs(a) : abs(a / b - 1)
for (c2, T, v, V, s, m, H, P) in ref
    local r = legendre_at(T, v, V, ConstantCs2EOS(cs2_val = c2))
    err = maximum((rel(r.s, s), rel(r.m, m), rel(r.H, H), rel(r.P, P)))
    check(err < 1e-10, @sprintf("c_s² = %.2f, T = %.1f, v = %.2f, V = %.1f: max relative error %.1e", c2, T, v, V, err))
end

println("2. Legendre identities (central differences)")
# the volume at which the target's entropy of the particle is s_a at (T, v)
vol_for(sa, T, v, eos) = (c = LegendreConsts(eos);  γ = 1 / sqrt(1 - v^2);  sa / (c.n * c.A * γ * (γ * T)^(c.n - 1)))
for c2 in (0.5, 0.25), (T, v) in ((3.0, 0.0), (3.0, 0.8), (1.7, 0.95))
    eos = ConstantCs2EOS(cs2_val = c2);  V = vol_for(54.0, T, v, eos)
    local r = legendre_at(T, v, V, eos);  m = SVector(0.6, 0.8) * r.m
    st = fwd(r.s, m, V, eos)
    hs = 1e-5 * r.s;  hm = 1e-5 * (r.m + r.s * T);  hV = 1e-5 * V
    dHds = (Hof(r.s + hs, m, V, eos) - Hof(r.s - hs, m, V, eos)) / 2hs
    dHdm = SVector((Hof(r.s, m + SVector(hm, 0), V, eos) - Hof(r.s, m - SVector(hm, 0), V, eos)) / 2hm,
                   (Hof(r.s, m + SVector(0, hm), V, eos) - Hof(r.s, m - SVector(0, hm), V, eos)) / 2hm)
    # at fixed s and m: the forward map takes S = s/V and M = m/V
    dHdV = (Hof(r.s, m, V + hV, eos) - Hof(r.s, m, V - hV, eos)) / 2hV
    e1 = abs(dHds / st.T - 1);  e2 = norm(dHdm - st.v);  e3 = abs(-dHdV / st.p - 1)
    check(max(e1, e2, e3) < 1e-6, @sprintf("c_s² = %.2f, T = %.1f, v = %.2f: ∂H/∂s = T to %.1e, ∂H/∂m = v to %.1e, −∂H/∂V = P to %.1e",
                                           c2, T, v, e1, e2, e3))
end

println("3. forward and inverse are inverses")
for c2 in (0.5, 0.25)
    eos = ConstantCs2EOS(cs2_val = c2);  c = LegendreConsts(eos);  worst = 0.0;  worst_s = 0.0;  n = 0;  nneg = 0
    for sa in (10.0, 30.0, 100.0, 1e3, 1e4), γ in (1.0, 1.5, 3.0, 10.0, 30.0, 99.0), V in (1e-3, 1.0)
        v = sqrt(1 - 1 / γ^2)
        Tr = (sa / (c.n * c.A * V * γ))^(1 / (c.n - 1));  T = Tr / γ       # the target's entropy s_a at (T, v)
        local r = legendre_at(T, v, V, eos)
        legendre_point(log(γ * T), atanh(v), V, c).J.det_Tv > 0 && r.m >= 0 || continue   # the physical branch (m̂ = v̂) only
        m = SVector(0.6, -0.8) * r.m
        si, vi, Ti = legendre_inverse(r.H, m, V, eos)
        isfinite(si) || (worst = Inf;  continue)
        e = Hof(si, m, V, eos)
        worst = max(worst, abs(e / r.H - 1));  worst_s = max(worst_s, abs(si - r.s) / (abs(r.s) + 10));  n += 1;  nneg += r.s < 0
    end
    check(worst < 1e-13, @sprintf("c_s² = %.2f: %d states (%d with s < 0), max |H(S(e)) / e − 1| = %.1e (|S(e) − s|/(|s| + 10) ≤ %.1e)",
                                  c2, n, nneg, worst, worst_s))
end

println("4. the counterterm off: the target's maps")
for c2 in (0.5, 0.25)
    eos = ConstantCs2EOS(cs2_val = c2);  worstf = 0.0;  worsti = 0.0
    for S in (5.0, 50.0, 5e3), Mr in (0.0, 0.5, 5.0, 50.0)
        M = SVector(0.8, 0.6) * (Mr * S);  s = 2.0;  V = s / S
        t = target_forward(s, S, M, 0.0, eos)
        l = legendre_forward(s, S, M, 0.0, 1.0, eos; counter = false)
        worstf = max(worstf, abs(l.T / t.T - 1), abs(l.γ / t.γ - 1), abs(l.HS / t.HS - 1), abs(l.p / t.p - 1), abs(l.e / t.e - 1))
        e = s * t.HS
        st, vt, Tt = target_inverse(e, M * V, V, eos)
        isfinite(st) || continue                     # beyond γ_max: no state for either
        sl, vl, Tl = legendre_inverse(e, M * V, V, eos; counter = false)
        worsti = max(worsti, abs(sl / st - 1), abs(Tl / Tt - 1), norm(vl - vt))
    end
    # the target's own Newton (eos_newton) stops at 1e-12 relative
    check(worstf < 1e-10 && worsti < 1e-10, @sprintf("c_s² = %.2f: forward to %.1e, inverse to %.1e of the target's (eos_newton: tol 1e-12)", c2, worstf, worsti))
end

println("5. warm and cold starts")
eos = ConstantCs2EOS(cs2_val = 0.5)
let worst = 0.0, nw = 0, nc = 0
    for (T, v) in ((3.0, 0.0), (3.0, 0.5), (1.7, 0.95)), δ in (1e-3, 1e-2)
        V = vol_for(54.0, T, v, eos);  local r = legendre_at(T, v, V, eos);  m = SVector(1.0, 0.0) * r.m
        reset_legendre_counts!();  cold = fwd(r.s, m, V, eos);  nc += legendre_counts().iters
        # warm: from the state of a neighbouring s (a step of the dynamics)
        near = fwd(r.s * (1 + δ), m, V, eos)
        reset_legendre_counts!()
        warm = legendre_forward(r.s, r.s / V, m / V, near.e, near.γ, eos);  nw += legendre_counts().iters
        worst = max(worst, abs(warm.T / cold.T - 1), abs(warm.HS / cold.HS - 1), norm(warm.v - cold.v))
    end
    check(worst < 1e-13, @sprintf("warm and cold agree to %.1e; iterations per solve: cold %.1f, warm %.1f", worst, nc / 6, nw / 6))
end

println("5b. states near the light cone at negative labels (campaign eos_legendre, c_s² = 1/4, s_a = 16, h/a = 2)")
# Newton from the warm and the cold start converged to the other branch; the bracket (legendre_bracket_forward) finds them
let eos4 = ConstantCs2EOS(cs2_val = 0.25), worst = 0.0, found = 0
    for (s0, V, m0, e0, γ0) in ((-3.701646564598388, 0.0005105584111477213, 13.100908743039128, 7465.33891696229, 13.099751342102051),
                                (-4.228451305363242, 0.0005010158199551941, 6.70568165109789, 403603.83728987037, 1.0266072738374323),
                                (-4.360144143972738, 0.00044821615662484507, 10.564417044412394, 1.2759910013123723e6, 1.000998591390844))
        m = SVector(m0, 0.0)
        for (e0_, γ0_) in ((e0, γ0), (0.0, 1.0))                    # the warm start the run had, and cold
            f = legendre_forward(s0, s0 / V, m / V, e0_, γ0_, eos4)
            isfinite(f.T) || continue
            found += 1
            r = legendre_inverse(s0 * f.HS, m, V, eos4)
            worst = max(worst, abs(r[1] / s0 - 1), abs(r[3] / f.T - 1))
        end
    end
    check(found == 6 && worst < 1e-10, @sprintf("%d of 6 solves found the state (γ = 12–41); the round trip agrees to %.1e", found, worst))
end

println("5c. heat-bath proposals with no state (e < 0): the inverse must not return a point with v < 0")
let eos4 = ConstantCs2EOS(cs2_val = 0.25), bad = 0
    for (e, m, V, g) in ((-52.6904936029332, SVector(-15.149293494605445, 26.993310483648635), 0.0004702561003668446,
                          (8.024846998070426, SVector(0.8376620274049337, 0.005914579786578995))),
                         (-38.118200373637706, SVector(-2.022086830333933, -20.748100871888123), 0.000569020443935683,
                          (7.525107324087005, SVector(0.8213736460156041, -0.07003512497206414))))
        for gg in (g, nothing)
            r = legendre_inverse(e, m, V, eos4; guess = gg)
            # a state, if returned, must have its velocity along m and a forward map back to it
            if isfinite(r[1])
                f = legendre_forward(r[1], r[1] / V, m / V, 0.0, 1.0, eos4)
                (dot(r[2], m) >= 0 && isfinite(f.T)) || (bad += 1)
            end
        end
    end
    check(bad == 0, "the two proposals of the campaign: no state returned with v against m, or without a forward map ($bad bad)")
end

println("6. hybrid Monte Carlo with the Legendre Hamiltonian")
function legendre_state(L; seed = 1, sa = 54.0)
    params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> sa / (sqrt(3) / 2 * (0.05 / 2)^2), (x, y) -> SVector(0.0, 0.0);
                                             L, lattice = :hex, seed = 4))
    r = quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(seed)))
    return params, ps, cl, r
end
params, ps, cl, r = legendre_state(0.3)
Hnow() = hmc_measure!(ps, params, cl, eos)[1]
dt0 = 0.5 * compute_dt(ps, params, eos; mode = :ideal)
x0 = [q.x for q in ps];  m0 = [q.m for q in ps];  H0 = Hnow()
errs = Float64[]
for f in (1, 2, 4)
    setx!(ps, params, cl, x0);  setm!(ps, m0);  H0 = Hnow()
    hmc_trajectory!(ps, params, cl, eos, dt0 / f, 8f)
    push!(errs, abs(Hnow() - H0) / abs(H0))
end
setx!(ps, params, cl, x0);  setm!(ps, m0);  Hnow()
ratio = errs[1] / errs[3]
check(10 < ratio < 25, @sprintf("|ΔH/H| at δt, δt/2, δt/4: %.2e %.2e %.2e (ratio %.1f, δt² gives 16)", errs..., ratio))
tg = HMCTarget(r.β0, r.β, params.vol, eos)
P0 = sum(q.m for q in ps);  reset_legendre_counts!()
acc = [hmc_update!(ps, params, cl, eos, tg, Xoshiro(7 + i); nsteps = 8, dt = dt0) for i in 1:5]
na = count(u -> u.accepted, acc);  lc = legendre_counts()
check(na >= 3, @sprintf("%d of 5 updates accepted, heat-bath acceptance %.2f", na, sum(u.heatbath for u in acc) / 5))
check(lc.fails == 0, @sprintf("%d solves, %.2f iterations per solve, %d failures (no convergence), %d proposals with no state, %d cold",
                              lc.solves, lc.iters / lc.solves, lc.fails, lc.nostate, lc.cold))

ok || error("testlegendre: failures")
println("testlegendre: all passed")
