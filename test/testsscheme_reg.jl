# testsscheme_reg.jl — the regularization of the S-scheme near the boundary (SPH_SSCHEME_REG=1; src/sscheme.jl; sscheme_fold.tex):
#   1. ∂H/∂s = T_a, ∂H/∂m = v_a, ∂H/∂V = −P_a by central differences of H(s, m, V), near the boundary and away from it, c_s² = 1/2 and 1/4;
#   2. the smallest label over S_t, on a grid of γv and V, is never below s_b and equals it at rest at V = V̄ (the boundary does not move);
#   3. hybrid Monte Carlo at s_a = 16 (the start of equilibrium.jl, L = 0.6): no trajectory leaves the domain, acceptance ≥ 0.9.
ENV["SPH_RENORM"] = "sscheme";  ENV["SPH_SSCHEME_REG"] = "1"
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
d4(f, x, h) = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
eos_A1(c2) = ConstantCs2EOS(c2, c2^(1 / (1 + 1 / c2)), 0.0, 0.0)

println("1. the derivatives of H with the regularization")
SS_VBAR[] = 1.0
for c2 in (0.5, 0.25)
    c = SSchemeConsts(eos_A1(c2))
    check(c.gvel == (c2 > 1 / 3) && c.gvol == (c2 < 0.5), @sprintf("c_s² = %.2f: g on the velocity part %s, on the volume part %s, s_b = %.4f", c2, c.gvel, c.gvol, c.sb))
    for (St, m, V) in ((3.2, 0.0, 1.0), (3.2, 1.5, 0.7), (4.0, 6.0, 0.8), (6.0, 10.0, 1.6), (16.0, 9.0, 1.3), (54.0, 300.0, 0.8))
        r = sscheme_target_entropy(St, m, V, c);  st = sscheme_state(r[1], r[2], V, c)
        st.ok || (println(@sprintf("  (S_t = %.1f, |m| = %.1f, V = %.1f: outside the domain, skipped)", St, m, V));  continue)
        s = st.s;  mv = SVector(m, 0.0);  H(ss, mm, VV) = sscheme_from_label(ss, mm, VV, c, St).H
        eT = abs(d4(x -> H(x, mv, V), s, 1e-4) / st.Ta - 1)
        ev = abs(d4(x -> H(s, SVector(x, 0.0), V), m, 1e-4 * (1 + m)) - st.va)
        eP = abs(-d4(x -> H(s, mv, x), V, 1e-5 * V) / st.Pa - 1)
        check(max(eT, ev, eP) < 1e-8, @sprintf("c_s² = %.2f, S_t = %4.1f, |m| = %5.1f, V = %.1f (𝒟 = %.3f): ∂H/∂s %.0e, ∂H/∂m %.0e, ∂H/∂V %.0e", c2, St, m, V, st.D, eT, ev, eP))
    end
end

println("2. the smallest label does not depend on m and V")
for c2 in (0.5, 0.25)
    c = SSchemeConsts(eos_A1(c2))
    worst = -Inf;  atrest = NaN
    for uγ in (0.0, 0.5, 1.0, 3.0, 10.0, 50.0), V in (0.4, 0.7, 1.0, 1.5, 1.8)
        smin = minimum(sscheme_label(St, uγ, V, c) for St in exp.(range(log(c.b) + 1e-9, log(c.b) + 3, length = 3000)))
        worst = max(worst, smin - c.sb);  (uγ == 0 && V == 1.0) && (atrest = smin - c.sb)
    end
    check(worst < 1e-6 && abs(atrest) < 1e-6, @sprintf("c_s² = %.2f: max over (γv, V ≤ 1.8 V̄) of s_min − s_b = %+.1e (≤ 0: every label above s_b has a state), at rest at V̄ %+.1e (0)", c2, worst, atrest))
end

get(ENV, "SR_TEST_QUICK", "0") == "1" && (ok || error("testsscheme_reg: failures"); println("testsscheme_reg: 1–2 passed (3 skipped, SR_TEST_QUICK=1)"); exit(0))
println("3. hybrid Monte Carlo at s_a = 16")
for c2 in (0.5, 0.25)
    global eos = ConstantCs2EOS(cs2_val = c2, η_by_s = 0.2)
    h = 0.6 / sqrt(672 / (2 / sqrt(3))) * 2;  S0 = 16 / (sqrt(3) / 2 * (h / 2)^2)
    SS_VBAR[] = NaN
    params, ps, cl, _ = quiet(() -> setup_ic(h, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex, seed = 5))
    Tr0 = trel_of_srel(eos, S0)
    r = quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1), bath = (1 / Tr0, SVector(0.0, 0.0))))
    tg = HMCTarget(r.β0, r.β, params.vol, eos)
    dt = 0.35 * compute_dt(ps, params, eos; mode = :ideal);  ns = round(Int, 0.6 / sqrt(c2) / dt)
    nacc = 0;  nexit = 0
    for i in 1:20
        u = hmc_update!(ps, params, cl, eos, tg, Xoshiro(100i); nsteps = ns, dt = dt)
        nacc += u.accepted;  nexit += !isfinite(u.dH)
    end
    check(nexit == 0 && nacc >= 18, @sprintf("c_s² = %.2f: %d of 20 accepted, %d left the domain (V̄ = %.3e, s_b = %.3f)", c2, nacc, nexit, SS_VBAR[], SSchemeConsts(eos).sb))
end

ok || error("testsscheme_reg: failures")
println("testsscheme_reg: all passed")
