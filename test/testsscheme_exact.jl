# testsscheme_exact.jl — the exact form of the S-scheme, SPH_SSCHEME_FORM=exact (src/sscheme_exact.jl; docs/sscheme_plan.tex, "The exact form",
# step X2).  The label is the exact count of states, s = s_min + ln A + g(ln S_t, θ), with ℓ = ln g from the table of examples/exact_count.py (c_s² = 1/2).
#   1. the interpolant against the exact count (mpmath) at 85 points off the grid (src/tables/exact_lng_c05_check.csv): g, g_x, g_θ;
#   2. the edges of the table (ℓ linear outside) are continuous; g_θ = 0 at θ = 0; above S_t = 600 the label is S_1loop + c(a)/S_t;
#   3. ∂H/∂s = T_a, ∂H/∂m = v_a, ∂H/∂V = −P_a by central differences of H(s, m, V), S_t = 0.5 to 2000, at rest and in motion;
#   4. the map from the energy inverts the map from the label, and the reverse;
#   5. no fold: 𝒟 = ∂s/∂S_t > 0 and |v_a| < 1 on a grid S_t = 0.5 to 10⁵, θ ≤ 5.2 (the one loop folds at S_t ≲ 3);
#   6. the floor: a state at every S_t > 0, its label above ln(A K(n/2)) − 1;
#   7. at m = 0 the particle has no velocity.
ENV["SPH_SSCHEME_FORM"] = "exact"
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "0")     # the maps are called directly
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
rel(a, b) = b == 0 ? abs(a) : abs(a / b - 1)
d4(f, x, h) = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)

eos_A1(c2) = ConstantCs2EOS(c2, c2^(1 / (1 + 1 / c2)), 0.0, 0.0)        # A = 1
c = SSchemeConsts(eos_A1(0.5));  q = c.q
state_St(St, m, V) = (r = sscheme_target_entropy(St, m, V, c);  sscheme_state(r[1], r[2], V, c))
# the state at target entropy S_t and rapidity θ: |m| = S_t T_rel γv, solved by bisection in |m| (the target map takes |m|)
function state_Stθ(St, θ, V)
    lo, hi = 0.0, 1.0
    f(m) = (r = sscheme_target_entropy(St, m, V, c);  asinh(r[2]))
    while f(hi) < θ;  hi *= 2;  end
    for _ in 1:200;  mid = (lo + hi) / 2;  f(mid) < θ ? (lo = mid) : (hi = mid);  end
    return state_St(St, (lo + hi) / 2, V), (lo + hi) / 2
end
Hof(s, m, V, St0) = sscheme_from_label(s, m, V, c, St0).H

println("1. the interpolant against the exact count")
let f = joinpath(@__DIR__, "..", "src", "tables", "exact_lng_c05_check.csv")
    isfile(f) || (check(false, "no $f (python3 examples/exact_count.py checkpoints)");  @goto done1)
    rows = [parse.(Float64, split(l, ",")) for l in readlines(f)[2:end]]
    e0 = e1 = e2 = 0.0
    for r in rows
        g, gx, gt = exact_g(r[1], r[2])
        sc = max(1.0, r[3])                                  # absolute below g = 1 (S_t ≲ 1.5), relative above
        e0 = max(e0, abs(g - r[3]) / sc);  e1 = max(e1, abs(gx - r[4]) / max(1.0, r[4]));  e2 = max(e2, abs(gt - r[5]) / max(1.0, r[4]))
    end
    check(e0 < 1e-6 && e1 < 2e-5 && e2 < 2e-5, @sprintf("%d points, S_t = 10⁻³ to 600: g = s_exact − s_min to %.1e, g_x to %.1e, g_θ to %.1e", length(rows), e0, e1, e2))
    @label done1
end

println("2. the edges")
let t = ssx_table(), ε = 1e-9
    jv = 0.0;  jx = 0.0
    for x in (t.x0, t.x0 + (t.nx - 1) * t.dx), θ in 0.0:0.5:6.0
        a = exact_g(x - ε, θ);  b = exact_g(x + ε, θ)
        jv = max(jv, abs(a[1] / b[1] - 1));  jx = max(jx, abs(a[2] / b[2] - 1))
    end
    check(jv < 1e-8 && jx < 1e-8, @sprintf("at S_t = 10⁻³ and 10⁶ (ℓ = ln g linear outside): jump of g %.0e, of g_x %.0e (relative)", jv, jx))
    z = maximum(abs(exact_g(x, 0.0)[3]) for x in range(t.x0, t.x0 + (t.nx - 1) * t.dx, length = 400))
    check(z == 0.0, @sprintf("g_θ at θ = 0: %.0e", z))
    # the asymptotic rows: s_exact against S_1loop + c(a)/S_t at S_t = 10⁴ (the full form's label, A = 1)
    for θ in (0.0, 2.0)
        st, _ = state_Stθ(1e4, θ, 1.0)
        ENV_s1 = 1e4 - (1.5 * log(2π * st.γ * 3 * st.Trel^3 * st.Trel) + 0.5 * log(2 - st.v^2)) - (1 + log(st.γ) - log(st.Trel))
        cc, _ = ssx_c2(2.0, 2 - st.v^2)
        check(abs(st.s - ENV_s1 - cc / 1e4) < 1e-6, @sprintf("θ = %.0f, S_t = 10⁴: s_exact − S_1loop − c(a)/S_t = %.1e", θ, st.s - ENV_s1 - cc / 1e4))
    end
end

println("3. the derivatives of H(s, m, V)")
for (St, θ, V) in ((0.5, 0.0, 1.0), (0.5, 1.5, 0.7), (1.0, 0.0, 1.0), (1.0, 2.0, 1.3), (3.0, 0.5, 0.9), (16.0, 0.0, 1.0), (16.0, 1.0, 1.3),
                   (54.0, 0.3, 0.8), (54.0, 3.0, 0.8), (541.0, 1.0, 1.1), (2000.0, 0.7, 1.0), (2000.0, 4.0, 2.0))
    st, m = state_Stθ(St, θ, V);  s = st.s;  mv = SVector(m, 0.0)
    hs = 1e-4 * max(1e-2, st.D) * max(1.0, St / 50);  hm = 1e-4 * (1 + m)    # the step grows with S_t: the solves stop at 1e-13 S_t
    eT = abs(d4(x -> Hof(x, mv, V, St), s, hs) / st.Ta - 1)
    vx = d4(x -> Hof(s, SVector(x, 0.0), V, St), m, hm)
    vy = d4(x -> Hof(s, SVector(m, x), V, St), 0.0, hm)
    ev = max(abs(vx - st.va), abs(vy))
    eP = abs(-d4(x -> Hof(s, mv, x, St), V, 1e-4 * V) / st.Pa - 1)
    check(max(eT, ev, eP) < 1e-6, @sprintf("S_t = %6.2f θ = %.1f V = %.1f (𝒟 = %.3f, v_a = %.4f): ∂H/∂s/T_a − 1 %.0e, |∂H/∂m − v_a| %.0e, −∂H/∂V/P_a − 1 %.0e",
                                         St, θ, V, st.D, st.va, eT, ev, eP))
end

println("4. the two maps invert each other")
for (St, θ, V) in ((0.5, 0.0, 1.0), (1.0, 2.0, 1.3), (16.0, 1.0, 1.3), (54.0, 3.0, 0.8), (2000.0, 4.0, 2.0))
    st, m = state_Stθ(St, θ, V);  mv = SVector(m, 0.0)
    back = sscheme_from_energy(st.H, mv, V, c)
    fwd = sscheme_from_label(st.s, mv, V, c, 1.3St)
    check(abs(back.s - st.s) < 1e-12 * max(1, abs(st.s)) && rel(fwd.St, St) < 1e-11 && rel(fwd.H, st.H) < 1e-12,
          @sprintf("S_t = %6.2f θ = %.1f: label from the energy %.0e, S_t from the label (start 1.3 S_t) %.0e, H %.0e",
                   St, θ, abs(back.s - st.s), rel(fwd.St, St), rel(fwd.H, st.H)))
end

println("5. no fold")
let Dmin = Inf, vmax = 0.0, bad = 0, n = 0, where = (0.0, 0.0)
    for x in range(log(0.5), log(1e5), length = 400), θ in range(0, 5.2, length = 53)      # θ ≤ 5.2: γ ≤ 100 (SPH_GAMMA_MAX)
        st, _ = state_Stθ(exp(x), θ, 1.0);  n += 1
        st.ok || (bad += 1;  continue)
        st.D < Dmin && (Dmin = st.D;  where = (exp(x), θ))
        vmax = max(vmax, abs(st.va))
    end
    check(bad == 0 && Dmin > 0 && vmax < 1, @sprintf("%d states, S_t = 0.5 to 10⁵, θ ≤ 5.2: no state %d, smallest 𝒟 = ∂s/∂S_t %.3e (S_t = %.3g, θ = %.1f), largest |v_a| %.6f",
                                                    n, bad, Dmin, where..., vmax))
end

println("6. the floor")
let t = ssx_table(), s0 = t.smin + log(c.A)
    r = [state_St(St, 0.0, 1.0) for St in (1e-4, 1e-2, 1.0)]
    check(all(x -> x.ok, r) && 0 < r[1].s - s0 < r[2].s - s0 < r[3].s - s0,
          @sprintf("s − [ln(A/2π) − 1] = %.1e, %.1e, %.1e at S_t = 10⁻⁴, 10⁻², 1 (a state at every S_t > 0, the label above the floor)", (x.s - s0 for x in r)...))
end

println("7. at rest")
let st = state_St(16.0, 0.0, 1.0), st2 = state_St(1.0, 0.0, 1.0)
    check(st.va == 0.0 && st2.va == 0.0, @sprintf("v_a at m = 0: %.0e (S_t = 16), %.0e (S_t = 1)", st.va, st2.va))
end

ok || error("testsscheme_exact: failures")
println("testsscheme_exact: all passed")
