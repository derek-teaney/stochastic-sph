# testpositional.jl — the positional counterterm ω (src/positional.jl; docs/sscheme_plan.tex, "Level 2", items H, I), SPH_POSITIONAL=1, a₂ = 0.3.
# Run once per scheme: SPH_RENORM=sscheme (default here; ω in the label) or SPH_RENORM=split (ω in the energy):
#   julia --project=. test/testpositional.jl            (sscheme)      SPH_RENORM=split julia --project=. test/testpositional.jl
#   0. the table: I₀ at the reference values of item H; the interpolant's derivative against a central difference;
#   1. ∂H/∂(label) = T_a, ∂H/∂m = v_a, −∂H/∂V = P_a by central differences of H with ω on, c_s² = 1/2, 1/4, S_t = 4 to 541, in motion;
#   2. the inverse (from the energy) of the map from the label, with ω on;
#   3. (sscheme) N3 (i) with ω: the label increases with S_t on the domain, and the label β + 1e-6 has a state.
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "sscheme")
ENV["SPH_POSITIONAL"] = "1";  ENV["SPH_POSITIONAL_A2"] = "0.3"
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
rel(a, b) = b == 0 ? abs(a) : abs(a / b - 1)
d4(f, x, h) = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
POS_H[] = 2.0                                        # V/h² = 1/4 at V = 1, as in the runs (V/h² = 1/N_h = 0.22 at h/a = 2)
println("scheme: ", SSCHEME ? "sscheme (form $(SS_FORM))" : "split", ";  a₂ = ", POS_A2, ", h = ", POS_H[])

println("0. the table of I₀")
let ref = [(3.89, -0.9791), (16.4, -3.6480), (18.0, -3.8777), (60.0, -7.7547), (179.0, -13.3240)]
    e = maximum(abs(i0_eval(s)[1] - r) for (s, r) in ref)
    ed = maximum(abs(d4(x -> i0_eval(exp(x))[1], log(s), 1e-4) - i0_eval(s)[2]) for s in (0.05, 0.7, 3.3, 47.0, 2e4))
    check(e < 2e-4 && ed < 1e-9, @sprintf("I₀ at the reference values (examples/i0_table.py) to %.0e; dI₀/dln σ of the interpolant to %.0e", e, ed))
end

if SSCHEME
    eos_A1(c2) = ConstantCs2EOS(c2, c2^(1 / (1 + 1 / c2)), 0.0, 0.0)
    sscheme_reference!(16.0, 1.0)
    state_St(St, m, V, c) = (r = sscheme_target_entropy(St, m, V, c);  sscheme_state(r[1], r[2], V, c))
    Hof(s, m, V, c, St0) = sscheme_from_label(s, m, V, c, St0).H
    println("1. the derivatives of H(s, m, V) with ω")
    for c2 in (0.5, 0.25), (St, m, V) in ((16.0, 0.0, 1.0), (16.0, 9.0, 1.3), (54.0, 40.0, 0.8), (541.0, 900.0, 1.1), (54.0, 300.0, 0.8), (4.0, 2.0, 1.0))
        c = SSchemeConsts(eos_A1(c2));  st = state_St(St, m, V, c);  s = st.s;  mv = SVector(m, 0.0)
        st.ok || (println(@sprintf("  (c_s² = %.2f, S_t = %.0f: no state, the label is at or below the wall)", c2, St));  continue)
        eT = abs(d4(x -> Hof(x, mv, V, c, St), s, 1e-3) / st.Ta - 1)
        vx = d4(x -> Hof(s, SVector(x, 0.0), V, c, St), m, 1e-3 * (1 + m));  vy = d4(x -> Hof(s, SVector(m, x), V, c, St), 0.0, 1e-3 * (1 + m))
        eP = abs(-d4(x -> Hof(s, mv, x, c, St), V, 1e-4 * V) / st.Pa - 1)
        o = positional_omega(St / c.n, V)
        check(max(eT, abs(vx - st.va), abs(vy), eP) < 1e-8,
              @sprintf("c_s² = %.2f, S_t = %3.0f, |m| = %5.1f (ω = %+.3f, ν_* = %.3f): T_a %.0e, v_a %.0e, P_a %.0e", c2, St, m, o.ω, o.ν, eT, max(abs(vx - st.va), abs(vy)), eP))
    end
    println("2. the inverse")
    for c2 in (0.5, 0.25), (St, m, V) in ((16.0, 9.0, 1.3), (54.0, 40.0, 0.8), (541.0, 900.0, 1.1), (4.0, 2.0, 1.0))
        c = SSchemeConsts(eos_A1(c2));  st = state_St(St, m, V, c);  mv = SVector(m, 0.0)
        st.ok || continue
        back = sscheme_from_energy(st.H, mv, V, c);  fl = sscheme_from_label(st.s, mv, V, c, 1.3St)
        check(abs(back.s - st.s) < 1e-12 * max(1, abs(st.s)) && rel(fl.St, St) < 1e-12, @sprintf("c_s² = %.2f, S_t = %3.0f: label %.0e, S_t %.0e", c2, St, abs(back.s - st.s), rel(fl.St, St)))
    end
    println("3. (N3 i) with ω: the label increases with S_t, and every label above the wall has a state")
    for c2 in (0.5, 0.25)
        c = SSchemeConsts(eos_A1(c2));  bad = 0;  Dmin = Inf;  n = 0
        for mm in (0.0, 1.0, 10.0, 100.0), V in (0.25, 1.0, 4.0)
            n += 1;  sts = [state_St(St, mm, V, c) for St in exp.(range(log(0.5), log(2000.0), length = 3000))]
            i0 = findfirst(st -> st.ok, sts);  i0 === nothing && (bad += 1;  continue)
            ss = [st.s for st in sts[i0:end] if st.ok]
            (all(st -> st.ok, sts[i0:end]) && all(diff(ss) .> 0)) || (bad += 1)
            Dmin = min(Dmin, minimum(st.D for st in sts[i0:end] if st.ok))
            SS_MEASURE && (sscheme_from_label(c.β + 1e-6, SVector(mm, 0.0), V, c, 10.0).ok || (bad += 1))
        end
        check(bad == 0, @sprintf("c_s² = %.2f: %d cases, %d failing; smallest 𝒟 %.3f", c2, n, bad, Dmin))
    end
else
    SPLIT_VBAR[] = 1.0
    fwd(u, m, V, eos) = split_forward(u, u / V, m / V, 1.0, eos)
    Hof(u, m, V, eos) = u * fwd(u, m, V, eos).HS
    function target_um(T, v, V, eos)
        c = SplitConsts(eos);  γ = 1 / sqrt(1 - v^2)
        return c.n * c.A * V * γ^c.n * T^(c.n - 1), c.n * c.A * V * γ^(c.n + 2) * T^c.n * v
    end
    println("1. the fields of the forward map with ω (central differences of H)")
    for c2 in (0.5, 0.25), (T, v, V) in ((3.0, 0.0, 4.0), (3.0, 0.8, 4.0), (1.7, 0.95, 0.5), (2.0, 0.5, 1.3), (0.6, 0.3, 1.0))
        eos = ConstantCs2EOS(cs2_val = c2)
        u, mm = target_um(T, v, V, eos);  m = SVector(mm * 0.6, mm * 0.8);  st = fwd(u, m, V, eos)
        hu = 1e-5 * u;  hm = 1e-5 * (norm(m) + u * T * 1e-2);  hV = 1e-5 * V
        dHu = (Hof(u + hu, m, V, eos) - Hof(u - hu, m, V, eos)) / (2hu)
        dHm = SVector((Hof(u, m + SVector(hm, 0.0), V, eos) - Hof(u, m - SVector(hm, 0.0), V, eos)) / (2hm),
                      (Hof(u, m + SVector(0.0, hm), V, eos) - Hof(u, m - SVector(0.0, hm), V, eos)) / (2hm))
        dHV = (Hof(u, m, V + hV, eos) - Hof(u, m, V - hV, eos)) / (2hV)
        eT = rel(st.T, dHu);  ex = norm(st.v - dHm) / max(norm(dHm), 1e-3);  eP = abs(st.p + dHV) / abs(dHV)
        o = positional_omega(u / (1 + 1 / c2), V)
        check(max(eT, ex, eP) < 1e-7, @sprintf("c_s² = %.2f, u = %6.1f, v = %.2f (ω = %+.3f, ν_* = %.3f): ∂H/∂u = T to %.1e, ∂H/∂m = ẋ to %.1e, −∂H/∂V = P to %.1e",
                                               c2, u, v, o.ω, o.ν, eT, ex, eP))
    end
    println("2. the inverse of the forward map with ω")
    for c2 in (0.5, 0.25)
        eos = ConstantCs2EOS(cs2_val = c2);  worst = 0.0;  nst = 0;  rng = Xoshiro(3)
        for sa in (8.0, 16.0, 54.0, 541.0), γ in (1.0, 1.5, 4.0)
            v = sqrt(1 - 1 / γ^2);  T = 1.0;  c = SplitConsts(eos);  V = sa / (c.n * c.A * γ^c.n * T^(c.n - 1))
            u, mm = target_um(T, v, V, eos);  ϕ = 2π * rand(rng);  m = mm * SVector(cos(ϕ), sin(ϕ))
            st = fwd(u, m, V, eos);  isfinite(st.T) || continue
            ui, xd, TH = split_inverse(u * st.HS, m, V, eos)
            worst = max(worst, abs(ui - u) / u);  nst += 1
        end
        check(worst < 1e-10, @sprintf("c_s² = %.2f: %d states, max |u(H)/u − 1| = %.1e", c2, nst, worst))
    end
end
ok || error("testpositional: failures")
println("testpositional: all passed")
