# testsscheme.jl — the one-loop core of the S-scheme (src/sscheme.jl), checked through the measure form, s = S_t − Δ^H with
# Δ^H = Δ_χ + Δ_ℓ − b_ρ ln S_t − C (docs/sscheme_plan.tex, "Level 1, amended").  The full form, s = S_t − Δ_χ − Δ_ℓ with the flat measure, was
# removed on 2026-11-26 (sec:prune); its tests 1–3 and 6 are those of testsscheme_measure.jl for the measure form.
#   1. the relation to the split scheme: S_t − s + b_ρ ln S_t = Δ_χ + Δ_ℓ − C = b ln S_t + 𝒩 ln(γV) + ½ ln(1 − c_s² v²) + const, to 1e-12,
#      c_s² = 1/2 and 1/4, at rest and in motion;
#   2. conformal (c_s² = 1/2) at rest: Δ^H = 0 and 𝒟 = 1, so the vote weight ln ρ(s) − ln T_a is that of the entropy shift u − (5/2) ln u
#      (the removed SPH_RENORM=1), exactly.
ENV["SPH_SSCHEME_FORM"] = "measure"
haskey(ENV, "SPH_SSCHEME_BRHO") && delete!(ENV, "SPH_SSCHEME_BRHO")
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "0")     # the maps are called directly
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)

eos_A1(c2) = ConstantCs2EOS(c2, c2^(1 / (1 + 1 / c2)), 0.0, 0.0)        # A = c_s² T_scale^(−n) = 1, as the reference script
sscheme_reference!(16.0, 1.0)                                          # s̄_rel = 16 (T_0 off conformal); V̄ does not enter at b_ρ = D/2 + 1
# the state with target entropy S_t, momentum (|m|, 0) and volume V
state_St(St, m, V, c) = (r = sscheme_target_entropy(St, m, V, c);  sscheme_state(r[1], r[2], V, c))

println("1. the relation to the split scheme")
for c2 in (0.5, 0.25)
    c = SSchemeConsts(eos_A1(c2));  N = 1 - 2c2;  b = 1.5 + 1 - N
    vals = Float64[]
    for (St, m, V) in ((16.0, 9.0, 1.3), (54.0, 40.0, 0.8), (541.0, 900.0, 1.1), (30.0, 200.0, 0.7), (54.0, 0.0, 1.0))
        st = state_St(St, m, V, c)
        Δ = st.St - st.s + c.bᵨ * log(st.St)                             # Δ_χ + Δ_ℓ − C
        push!(vals, Δ - b * log(st.St) - N * log(st.γ * V) - 0.5 * log(1 - c2 * st.v^2))
    end
    check(maximum(vals) - minimum(vals) < 1e-12 * maximum(abs, vals) + 1e-12,
          @sprintf("c_s² = %.2f (b_ρ = %.2f): S_t − s + b_ρ ln S_t − [b ln S_t + 𝒩 ln(γV) + ½ ln(1 − c_s² v²)] = %.12f, spread %.0e",
                   c2, c.bᵨ, vals[1], maximum(vals) - minimum(vals)))
end

println("2. conformal at rest: the weight of the entropy shift u − (5/2) ln u (the removed SPH_RENORM=1)")
let c = SSchemeConsts(eos_A1(0.5)), d = Float64[]
    for (St, V) in ((16.0, 1.0), (54.0, 0.8), (541.0, 1.1), (30.0, 2.5))
        st = state_St(St, 0.0, V, c)
        u = st.St;  w1 = u - 2.5 * log(u) + log(1 - 2.5 / u) - log(st.T)  # at rest the target's entropy is S_t and its temperature T
        push!(d, abs(st.weight - w1) / abs(w1))
    end
    check(maximum(d) < 1e-13, @sprintf("|weight − weight(entropy shift)| / |weight| ≤ %.0e at S_t = 16, 54, 541, 30", maximum(d)))
end

ok || error("testsscheme: failures")
println("testsscheme: all passed")
