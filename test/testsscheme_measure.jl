# testsscheme_measure.jl — the measure form of the S-scheme, SPH_SSCHEME_FORM=measure (docs/sscheme_plan.tex, "Level 1, amended", N3;
# src/sscheme.jl): s = S_t − Δ^H, measure ρ(s) = e^s s^(−β)(1 − β/s) on s > β, β = D/2 + 1.
#   1. ∂H/∂s = T_a, ∂H/∂m = v_a, ∂H/∂V = −P_a by central differences of H(s, m, V), to 1e-9;
#   2. the map from the energy inverts the map from the label, and the reverse;
#   3. reference values of s, H, T_a, v_a, P_a and the weight ln ρ(s) − ln T_a from state_m of examples/sscheme_ref.py (mpmath, A = 1,
#      F = 1, T_0 at s_rel = 16), to 1e-12 relative;
#   4. s − (S_t − Δ_χ − Δ_ℓ + β ln S_t) is one constant for every state;
#   5. (N3 i) on a grid of (|m|, V): the label increases with S_t on the domain, the domain is one interval, and the label β + 1e-6 (just
#      above the wall) has a state;
#   6. the weight is −Inf at and below the wall, and a label below it has no state.
ENV["SPH_SSCHEME_FORM"] = "measure"
haskey(ENV, "SPH_SSCHEME_BETA") && delete!(ENV, "SPH_SSCHEME_BETA")
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "0")
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
rel(a, b) = b == 0 ? abs(a) : abs(a / b - 1)
eos_A1(c2) = ConstantCs2EOS(c2, c2^(1 / (1 + 1 / c2)), 0.0, 0.0)        # A = 1, as the reference script
sscheme_reference!(16.0, 1.0)                                          # s̄_rel = 16 (T_0 of the reference values); V̄ does not enter at β = D/2 + 1
state_St(St, m, V, c) = (r = sscheme_target_entropy(St, m, V, c);  sscheme_state(r[1], r[2], V, c))
Hof(s, m, V, c, St0) = sscheme_from_label(s, m, V, c, St0).H
d4(f, x, h) = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
println("β = ", SS_BETA)

println("1. the derivatives of H(s, m, V)")
for c2 in (0.5, 0.25), (St, m, V) in ((16.0, 0.0, 1.0), (16.0, 9.0, 1.3), (54.0, 40.0, 0.8), (541.0, 900.0, 1.1), (54.0, 300.0, 0.8), (4.0, 2.0, 1.0))
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
for c2 in (0.5, 0.25), (St, m, V) in ((16.0, 9.0, 1.3), (54.0, 40.0, 0.8), (541.0, 900.0, 1.1), (54.0, 300.0, 0.8), (4.0, 2.0, 1.0))
    c = SSchemeConsts(eos_A1(c2));  st = state_St(St, m, V, c);  mv = SVector(m, 0.0)
    back = sscheme_from_energy(st.H, mv, V, c)
    fwd = sscheme_from_label(st.s, mv, V, c, 1.3St)
    check(rel(back.s, st.s) < 1e-13 && rel(fwd.St, St) < 1e-13 && rel(fwd.H, st.H) < 1e-13,
          @sprintf("c_s² = %.2f, S_t = %3.0f, |m| = %5.1f: label from the energy %.0e, S_t from the label (start 1.3 S_t) %.0e, H %.0e",
                   c2, St, m, rel(back.s, st.s), rel(fwd.St, St), rel(fwd.H, st.H)))
end

println("3. reference values (examples/sscheme_ref.py, state_m)")
# (c_s², S_t, |m|, V, s, H, T_a, v_a, P_a, weight)
ref = [(0.5, 16, 0, 1, 16.0, 24.633611485424034, 2.309401076758503, 0.0, 12.316805742712017, 8.061640940819313),
       (0.5, 16, 9, 1.3, 16.018904749432007, 22.84284180104857, 1.9183333087881456, 0.2645133156268539, 7.870085369387263, 8.263343565789725),
       (0.5, 54, 40, 0.8, 54.00605931439679, 173.87672966682916, 4.65900924069681, 0.15380992504359972, 104.82770791567822, 42.44711908121232),
       (0.5, 541, 900, 1.1, 541.004202476671, 4676.277945575474, 12.643089081589412, 0.12926350016543614, 2072.700361557537, 522.7288923652686),
       (0.5, 30, 200, 0.7, 30.305202921175983, 203.21150353452464, 0.6056501704014059, 0.9554925006324826, 8.652145291448656, 22.192260876260193),
       (0.5, 4, 2, 1, 4.044618939730518, 3.5022292765237637, 1.03631514348561, 0.36936111361440194, 1.38175352464748, -0.44713080427682916),
       (0.25, 16, 0, 1, 16.0, 17.119751807396405, 1.2969508944997277, 0.0, 3.631462504599237, 8.638613113820483),
       (0.25, 16, 9, 1.3, 15.84315601246043, 17.981559418077648, 1.0730938675217057, 0.4240800407545701, 2.31127909927752, 8.694032032278614),
       (0.25, 54, 40, 0.8, 54.70083005769028, 90.3284599451199, 1.7343268622228907, 0.3677952188050466, 22.546249208897578, 44.09873299734082),
       (0.25, 541, 900, 1.1, 542.6771271402074, 1589.2093897366417, 2.672838776696165, 0.4792618129687365, 261.93820011622415, 525.9480822511764),
       (0.25, 30, 200, 0.7, 29.71686667830387, 201.5789218086538, 0.16639098607901015, 0.9875949087416297, 1.3311278886320814, 22.943116728312003),
       (0.25, 4, 2, 1, 3.2711599073822972, 3.5294389682089076, 0.7000610835455972, 0.5046095337223789, 0.2800244334182389, -0.7801181495628776)]
for (c2, St, m, V, s, H, Ta, va, Pa, wt) in ref
    c = SSchemeConsts(eos_A1(c2));  st = state_St(float(St), float(m), float(V), c)
    err = maximum((abs(st.s - s) / max(1, abs(s)), rel(st.H, H), rel(st.Ta, Ta), abs(st.va - va), rel(st.Pa, Pa), abs(st.weight - wt) / max(1, abs(wt))))
    check(err < 1e-12, @sprintf("c_s² = %.2f, S_t = %3d, |m| = %3d, V = %.1f: largest relative difference %.0e", c2, St, m, V, err))
end

println("4. the relation to the full form: s − (S_t − Δ_χ − Δ_ℓ + β ln S_t) is a constant")
for c2 in (0.5, 0.25)
    c = SSchemeConsts(eos_A1(c2));  vals = Float64[]
    for (St, m, V) in ((16.0, 9.0, 1.3), (54.0, 40.0, 0.8), (541.0, 900.0, 1.1), (30.0, 200.0, 0.7), (4.0, 2.0, 1.0))
        st = state_St(St, m, V, c);  γ = st.γ;  v = st.v;  Tr = st.Trel;  w = c.n * c.A * Tr^c.n
        full = St - (1.5 * log(2π * V * γ * w * Tr) + 0.5 * log(c.q - v^2)) - (1 + log(V) + log(γ) - log(Tr))
        push!(vals, st.s - full - c.β * log(St))
    end
    check(maximum(vals) - minimum(vals) < 1e-11, @sprintf("c_s² = %.2f: the constant %.12f, spread %.0e", c2, vals[1], maximum(vals) - minimum(vals)))
end

println("5. (N3 i) the label increases with S_t, and every label above the wall has a state")
for c2 in (0.5, 0.25)
    c = SSchemeConsts(eos_A1(c2));  nonmono = 0;  nogap = 0;  noroot = 0;  ncase = 0;  Dmin = Inf
    for mm in (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0), V in (0.25, 0.5, 1.0, 2.0, 4.0)
        ncase += 1
        Sts = exp.(range(log(0.3), log(2000.0), length = 4000))
        sts = [state_St(St, mm, V, c) for St in Sts]
        oks = [st.ok for st in sts];  i0 = findfirst(oks)
        i0 === nothing && (noroot += 1;  continue)
        all(oks[i0:end]) || (nogap += 1)                                           # the domain is one interval [S*, ∞)
        ss = [st.s for st in sts[i0:end] if st.ok]
        all(diff(ss) .> 0) || (nonmono += 1)
        Dmin = min(Dmin, minimum(st.D for st in sts[i0:end] if st.ok))
        r = sscheme_from_label(c.β + 1e-6, SVector(mm, 0.0), V, c, 10.0)          # just above the wall
        r.ok || (noroot += 1)
    end
    check(nonmono == 0 && nogap == 0 && noroot == 0,
          @sprintf("c_s² = %.2f, %d cases (|m| ≤ 100, 1/4 ≤ V ≤ 4): not increasing %d, domain not one interval %d, label β + 1e-6 without a state %d; smallest 𝒟 on the domain %.3f",
                   c2, ncase, nonmono, nogap, noroot, Dmin))
end

println("6. the wall")
let c = SSchemeConsts(eos_A1(0.25))
    fl = sscheme_from_label(c.β - 1e-3, SVector(1.0, 0.0), 1.0, c, 10.0)
    check(ss_lnrho(c.β, c) == -Inf && ss_lnrho(c.β - 0.1, c) == -Inf && isfinite(ss_lnrho(c.β + 1e-9, c)) && !fl.ok && fl.weight == -Inf,
          @sprintf("ln ρ(β) = %s, ln ρ(β + 1e-9) = %.3f; the label β − 1e-3: ok = %s", ss_lnrho(c.β, c), ss_lnrho(c.β + 1e-9, c), fl.ok))
    n = sscheme_counts()
    println(@sprintf("  counters: %d solves, %.1f Newton steps per solve, %d fallbacks, %d without a state", n.solves, n.iters / max(n.solves, 1), n.fallbacks, n.nostate))
end

ok || error("testsscheme_measure: failures")
println("testsscheme_measure: all passed")
