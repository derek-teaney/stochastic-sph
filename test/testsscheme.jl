# testsscheme.jl — the maps of the S-scheme (src/sscheme.jl; docs/sscheme_plan.tex, item D, tests 1–6; test 7, hybrid Monte Carlo, comes
# with the steps, item C):
#   1. ∂H/∂s = T_a, ∂H/∂m = v_a, ∂H/∂V = −P_a by central differences of H(s, m, V) (the map from the label), to 1e-9, at rest and in motion,
#      c_s² = 1/2 and 1/4, S_t = 16, 54, 541;
#   2. the map from the energy inverts the map from the label (the label to 1e-13 relative), and the reverse (from a start 30% off);
#   3. reference values of s, H, T_a, v_a, P_a from examples/sscheme_ref.py (mpmath, A = 1), to 1e-12 relative;
#   4. Δ_χ + Δ_ℓ = b ln S_t + 𝒩 ln(γV) + ½ ln(1 − c_s² v²) + const (the relation to the split scheme), to 1e-12;
#   5. conformal (c_s² = 1/2) at rest: the vote weight differs from that of SPH_RENORM=1 (stochastic.jl, renorm_vote_entropy) by a constant;
#   6. a state outside the domain has no Hamiltonian from either map, and its weight is −Inf.
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "0")     # the maps are called directly; renorm_vote_entropy (test 5) needs the target's inverse
ENV["SPH_SSCHEME_REG"] = "0"                                # tests 1–6: one loop as in renorm3.tex (the regularization: testsscheme_reg.jl)
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
rel(a, b) = b == 0 ? abs(a) : abs(a / b - 1)

eos_A1(c2) = ConstantCs2EOS(c2, c2^(1 / (1 + 1 / c2)), 0.0, 0.0)        # A = c_s² T_scale^(−n) = 1, as the reference script
# the state with target entropy S_t, momentum (|m|, 0) and volume V
state_St(St, m, V, c) = (r = sscheme_target_entropy(St, m, V, c);  sscheme_state(r[1], r[2], V, c))
Hof(s, m, V, c, St0) = sscheme_from_label(s, m, V, c, St0).H
# fourth-order central difference
d4(f, x, h) = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)

println("1. the derivatives of H(s, m, V)")
for c2 in (0.5, 0.25), (St, m, V) in ((16.0, 0.0, 1.0), (16.0, 9.0, 1.3), (54.0, 40.0, 0.8), (541.0, 900.0, 1.1), (54.0, 300.0, 0.8))
    c = SSchemeConsts(eos_A1(c2));  st = state_St(St, m, V, c);  s = st.s;  mv = SVector(m, 0.0)
    eT = abs(d4(x -> Hof(x, mv, V, c, St), s, 1e-3) / st.Ta - 1)
    vx = d4(x -> Hof(s, SVector(x, 0.0), V, c, St), m, 1e-3 * (1 + m))
    vy = d4(x -> Hof(s, SVector(m, x), V, c, St), 0.0, 1e-3 * (1 + m))
    ev = max(abs(vx - st.va), abs(vy))
    eP = abs(-d4(x -> Hof(s, mv, x, c, St), V, 1e-4 * V) / st.Pa - 1)
    check(max(eT, ev, eP) < 1e-9, @sprintf("c_s² = %.2f, S_t = %3.0f, |m| = %5.1f (v = %.3f, v_a = %.3f): ∂H/∂s/T_a − 1 %.0e, |∂H/∂m − v_a| %.0e, −∂H/∂V/P_a − 1 %.0e",
                                         c2, St, m, st.v, st.va, eT, ev, eP))
end

println("2. the two maps invert each other")
for c2 in (0.5, 0.25), (St, m, V) in ((16.0, 9.0, 1.3), (54.0, 40.0, 0.8), (541.0, 900.0, 1.1), (54.0, 300.0, 0.8))
    c = SSchemeConsts(eos_A1(c2));  st = state_St(St, m, V, c);  mv = SVector(m, 0.0)
    back = sscheme_from_energy(st.H, mv, V, c)
    fwd = sscheme_from_label(st.s, mv, V, c, 1.3St)
    check(rel(back.s, st.s) < 1e-13 && rel(fwd.St, St) < 1e-13 && rel(fwd.H, st.H) < 1e-13,
          @sprintf("c_s² = %.2f, S_t = %3.0f, |m| = %5.1f: label from the energy %.0e, S_t from the label (start 1.3 S_t) %.0e, H %.0e",
                   c2, St, m, rel(back.s, st.s), rel(fwd.St, St), rel(fwd.H, st.H)))
end

println("3. reference values (examples/sscheme_ref.py)")
# (c_s², S_t, |m|, V, s, H, T_a, v_a, P_a)
ref = [(0.5, 16, 0, 1, 6.063751293174666, 24.633611485424034, 2.7370679428248925, 0.0, 12.316805742712017),
       (0.5, 16, 9, 1.3, 6.082656042606672, 22.84284180104857, 2.2750486385177937, 0.2630448949320473, 7.875168364100055),
       (0.5, 54, 40, 0.8, 41.028822296760225, 173.87672966682916, 4.88525410168171, 0.15374180439392754, 104.82941093192002),
       (0.5, 541, 900, 1.1, 522.2658773783289, 4676.277945575474, 12.701786313848459, 0.12926295433764193, 2072.7005848507256),
       (0.5, 30, 200, 0.7, 18.79743256579471, 203.21150353452464, 0.6611134444803707, 0.9554554519285892, 8.657437963433425),
       (0.25, 16, 0, 1, 6.809578731563525, 17.119751807396405, 1.5285492685175361, 0.0, 3.515663317590333),
       (0.25, 16, 9, 1.3, 6.652734744023954, 17.981559418077648, 1.26383383705748, 0.4251800492259858, 2.2360137117170797),
       (0.25, 54, 40, 0.8, 42.46942047844257, 90.3284599451199, 1.8176312989740981, 0.36787494769493556, 22.493187324804463),
       (0.25, 541, 900, 1.1, 524.6846294802541, 1589.2093897366417, 2.6852339190693715, 0.47926280425992074, 261.9323631964941),
       (0.25, 30, 200, 0.7, 18.95492376131146, 201.5789218086538, 0.180838987151729, 0.9876406424822753, 1.317541192105454)]
for (c2, St, m, V, s, H, Ta, va, Pa) in ref
    c = SSchemeConsts(eos_A1(c2));  st = state_St(float(St), float(m), float(V), c)
    err = maximum((rel(st.s, s), rel(st.H, H), rel(st.Ta, Ta), abs(st.va - va), rel(st.Pa, Pa)))
    check(err < 1e-12, @sprintf("c_s² = %.2f, S_t = %3d, |m| = %3d, V = %.1f: largest relative difference %.0e", c2, St, m, V, err))
end

println("4. the relation to the split scheme")
for c2 in (0.5, 0.25)
    c = SSchemeConsts(eos_A1(c2));  N = 1 - 2c2;  b = 1.5 + 1 - N
    vals = Float64[]
    for (St, m, V) in ((16.0, 9.0, 1.3), (54.0, 40.0, 0.8), (541.0, 900.0, 1.1), (30.0, 200.0, 0.7))
        st = state_St(St, m, V, c)
        Δ = st.St - st.s
        push!(vals, Δ - b * log(st.St) - N * log(st.γ * V) - 0.5 * log(1 - c2 * st.v^2))
    end
    check(maximum(vals) - minimum(vals) < 1e-12 * maximum(abs, vals) + 1e-12,
          @sprintf("c_s² = %.2f: Δ − [b ln S_t + 𝒩 ln(γV) + ½ ln(1 − c_s² v²)] = %.12f, spread %.0e", c2, vals[1], maximum(vals) - minimum(vals)))
end

println("5. conformal at rest: the weight against SPH_RENORM=1")
let c2 = 0.5, eos = eos_A1(0.5), c = SSchemeConsts(eos_A1(0.5)), d = Float64[]
    for (St, V) in ((16.0, 1.0), (54.0, 0.8), (541.0, 1.1), (30.0, 2.5))
        st = state_St(St, 0.0, V, c)
        push!(d, st.weight - renorm_vote_entropy(st.H, SVector(0.0, 0.0), V, eos, true))
    end
    check(maximum(d) - minimum(d) < 1e-11 * maximum(abs, d), @sprintf("weight − weight(SPH_RENORM=1) = %.12f, spread %.0e", d[1], maximum(d) - minimum(d)))
end

println("6. outside the domain")
let c = SSchemeConsts(eos_A1(0.5))
    st = state_St(2.0, 0.0, 1.0, c)                                       # S_t = 2 < 5/2: 𝒟 < 0 at rest
    r = sscheme_target_entropy(2.0, 0.0, 1.0, c);  H = c.A * r[1]^c.n * (c.n - 1) * 1.0      # the target's energy there, e = V p/c_s²
    fe = sscheme_from_energy(H, SVector(0.0, 0.0), 1.0, c)
    fl = sscheme_from_label(-50.0, SVector(0.0, 0.0), 1.0, c, 10.0)      # a label below every state of the domain
    check(!st.ok && !fe.ok && fe.weight == -Inf && !fl.ok && fl.s == -Inf,
          @sprintf("S_t = 2 at rest: state ok = %s, from the energy ok = %s (weight %s); label −50: from the label ok = %s", st.ok, fe.ok, fe.weight, fl.ok))
    n = sscheme_counts()
    println(@sprintf("  counters: %d solves, %.1f Newton steps per solve, %d fallbacks, %d without a state", n.solves, n.iters / max(n.solves, 1), n.fallbacks, n.nostate))
end

ok || error("testsscheme: failures")
println("testsscheme: all passed")
