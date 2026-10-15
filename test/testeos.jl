include(joinpath(@__DIR__, "..", "src", "sph.jl"))

# ── Helpers ────────────────────────────────────────────────────────────────────

function make_particle(; S_smooth, Mx_smooth=0.0, My_smooth=0.0)
    pa = Particle(SVector(0.0, 0.0), 1.0)
    pa.S_smooth = S_smooth
    pa.M_smooth = SVector(Mx_smooth, My_smooth)
    return pa
end

function residual(pa::Particle, eos::AbstractEOS)
    e = pa.e
    w = e + pressure(eos, e)
    M = sqrt(pa.M_smooth[1]^2 + pa.M_smooth[2]^2)
    u = M / w
    γ = sqrt(1.0 + u^2)
    return w * γ / Trel(eos, e) - pa.S_smooth
end

const eos = ConformalEOS(1.0/3.0, 1.0, 0.0, 0.0)

# ── Test 1: ConformalEOS interface ─────────────────────────────────────────────

println("=" ^ 55)
println("Test 1: ConformalEOS interface")
println("=" ^ 55)

let
    for e_test in [0.1, 1.0, 5.0, 100.0]
        p_val  = pressure(eos, e_test)
        T_val  = Trel(eos, e_test)
        c2_val = cs2(eos, e_test)
        @assert p_val  ≈ (1.0/3.0) * e_test  "pressure wrong at e=$e_test"
        @assert T_val  ≈ e_test^(1.0/4.0)    "Trel wrong at e=$e_test"
        @assert c2_val ≈ 1.0/3.0             "cs2 wrong at e=$e_test"
        println("  e=$e_test  p=$(round(p_val,digits=4))  Trel=$(round(T_val,digits=4))  cs2=$c2_val")
    end
end
println("PASS")

# ── Test 2: Thermodynamic identity  dTrel/de = cs2 * Trel / (e+p) ─────────────

println("\n", "=" ^ 55)
println("Test 2: dTrel/de thermodynamic identity (finite diff)")
println("=" ^ 55)

let δ = 1e-7
    for e_test in [0.5, 1.0, 4.0]
        dT_fd  = (Trel(eos, e_test + δ) - Trel(eos, e_test - δ)) / (2δ)
        dT_ana = cs2(eos, e_test) * Trel(eos, e_test) / (e_test + pressure(eos, e_test))
        err    = abs(dT_fd - dT_ana) / abs(dT_ana)
        println("  e=$e_test  fd=$(round(dT_fd,sigdigits=6))  ana=$(round(dT_ana,sigdigits=6))  rel_err=$(round(err,sigdigits=2))")
        @assert err < 1e-6  "thermodynamic identity violated at e=$e_test, rel_err=$err"
    end
end
println("PASS")

# ── Test 3: solve_eos! at rest — exact analytic answer ─────────────────────────
#
# For ConformalEOS(1/3) with M=0, γ=1:
#   (4/3)*e^(3/4) = S  →  e_exact = (0.75 * S)^(4/3)

println("\n", "=" ^ 55)
println("Test 3: solve_eos! at rest  (M=0, analytic answer)")
println("=" ^ 55)

let
    for S_in in [0.5, 1.0, 2.0, 10.0]
        pa      = make_particle(S_smooth=S_in)
        solve_eos!(pa, eos)
        e_exact = (0.75 * S_in)^(4/3)
        p_exact = e_exact / 3.0
        T_exact = (e_exact + p_exact) / S_in
        e_err   = abs(pa.e - e_exact) / e_exact
        p_err   = abs(pa.p - p_exact) / p_exact
        T_err   = abs(pa.T - T_exact) / T_exact
        res     = abs(residual(pa, eos)) / S_in
        println("  S=$S_in  e=$(round(pa.e,sigdigits=6)) (exact $(round(e_exact,sigdigits=6)))" *
                "  T=$(round(pa.T,sigdigits=5))  res=$(round(res,sigdigits=2))")
        @assert e_err < 1e-10  "e wrong at S=$S_in, rel_err=$e_err"
        @assert p_err < 1e-10  "p wrong at S=$S_in"
        @assert T_err < 1e-10  "T wrong at S=$S_in"
        @assert res   < 1e-11  "residual too large at S=$S_in"
    end
end
println("PASS")

# ── Test 4: solve_eos! residual check — relativistic (M ≠ 0) ──────────────────
#
# Physical constraint for ConformalEOS(1/3): a root exists iff M/S^(4/3) < 0.625.

println("\n", "=" ^ 55)
println("Test 4: solve_eos! residual — relativistic M≠0")
println("=" ^ 55)

let test_cases = [
        (S_smooth=1.0,  Mx=0.3,  My=0.0),   # M/S^(4/3) ≈ 0.30
        (S_smooth=1.0,  Mx=0.4,  My=0.3),   # M/S^(4/3) ≈ 0.50
        (S_smooth=2.0,  Mx=0.8,  My=0.6),   # M/S^(4/3) ≈ 0.40
        (S_smooth=0.5,  Mx=0.05, My=0.08),  # M/S^(4/3) ≈ 0.24
    ]
    for tc in test_cases
        pa  = make_particle(S_smooth=tc.S_smooth, Mx_smooth=tc.Mx, My_smooth=tc.My)
        solve_eos!(pa, eos)
        res = abs(residual(pa, eos)) / tc.S_smooth
        M   = sqrt(tc.Mx^2 + tc.My^2)
        w   = pa.e + pa.p
        u   = M / w
        v2  = u^2 / (1.0 + u^2)
        println("  S=$(tc.S_smooth)  |M|=$(round(M,sigdigits=3))  e=$(round(pa.e,sigdigits=5))" *
                "  v²=$(round(v2,sigdigits=4))  res=$(round(res,sigdigits=2))")
        @assert res < 1e-11  "residual too large: S=$(tc.S_smooth), M=($(tc.Mx),$(tc.My))"
        @assert v2  < 1.0    "superluminal: v²=$v2"
        @assert pa.p > 0.0   "negative pressure"
        @assert pa.T > 0.0   "negative temperature"
    end
end
println("PASS")

# ── Test 5: post-convergence expressions ──────────────────────────────────────

println("\n", "=" ^ 55)
println("Test 5: post-convergence  F/S and H/S")
println("=" ^ 55)

let
    pa  = make_particle(S_smooth=1.5, Mx_smooth=0.6, My_smooth=0.4)
    solve_eos!(pa, eos)
    S   = pa.S_smooth
    M   = pa.M_smooth
    w   = pa.e + pa.p
    u_v = M / w
    u2  = u_v[1]^2 + u_v[2]^2
    γ   = sqrt(1.0 + u2)
    v   = u_v / γ
    F_over_S_check = dot(v, M / S) / S - pa.p / S^2
    H_over_S_check = (w * γ^2 - pa.p) / S
    T_check        = w / S
    @assert abs(pa.F_over_S - F_over_S_check) < 1e-14  "F/S mismatch"
    @assert abs(pa.H_over_S - H_over_S_check) < 1e-14  "H/S mismatch"
    @assert abs(pa.T        - T_check)        < 1e-14  "T mismatch"
    println("  F/S = $(round(pa.F_over_S, sigdigits=6))  ✓")
    println("  H/S = $(round(pa.H_over_S, sigdigits=6))  ✓")
    println("  T   = $(round(pa.T,        sigdigits=6))  ✓")
end
println("PASS")

# ── Test 6: warm start with 10% perturbation ──────────────────────────────────

println("\n", "=" ^ 55)
println("Test 6: warm start — perturb e by 10%, verify same result")
println("=" ^ 55)

let
    pa = make_particle(S_smooth=1.0, Mx_smooth=0.5, My_smooth=0.3)
    solve_eos!(pa, eos)
    e1, p1, T1 = pa.e, pa.p, pa.T
    println("  first call:   e = $(round(e1, sigdigits=10))")

    pa.e *= 1.10                        # perturb warm-start by +10%
    solve_eos!(pa, eos; verbose=true)   # should re-converge to same answer
    e2, p2, T2 = pa.e, pa.p, pa.T
    println("  second call:  e = $(round(e2, sigdigits=10))")

    @assert abs(e1 - e2) / e1 < 1e-10  "perturbed warm start gave different e: $e1 vs $e2"
    @assert abs(p1 - p2) / p1 < 1e-10  "perturbed warm start gave different p"
    @assert abs(T1 - T2) / T1 < 1e-10  "perturbed warm start gave different T"
end
println("PASS")

# ── Test 7: quadratic convergence ─────────────────────────────────────────────

println("\n", "=" ^ 55)
println("Test 7: Newton convergence history  (verbose, cold start)")
println("=" ^ 55)

let
    pa = make_particle(S_smooth=1.0, Mx_smooth=0.4, My_smooth=0.3)
    solve_eos!(pa, eos; verbose=true)
end
println("PASS")

# ── Test 8: dsrel/de = srel/(e+p)  (thermodynamic identity) ──────────────────
#
# From srel = (e+p)/Trel and dTrel/de = cs2·Trel/(e+p):
#   dsrel/de = (1+cs2)/Trel − cs2/Trel = 1/Trel = srel/(e+p).
# Checked numerically at cs2=0.2, T_scale=7.1.

println("\n", "=" ^ 55)
println("Test 8: dsrel/de = srel/(e+p)  (finite-difference check)")
println("=" ^ 55)

let
    eos_b = ConformalEOS(0.2, 7.1, 0.0, 0.0)
    δ = 1e-7
    for e_test in [0.1, 1.0, 3.0, 10.0]
        ds_fd  = (srel(eos_b, e_test + δ) - srel(eos_b, e_test - δ)) / (2δ)
        ds_ana = srel(eos_b, e_test) / (e_test + pressure(eos_b, e_test))
        err    = abs(ds_fd - ds_ana) / abs(ds_ana)
        println("  e=$e_test  fd=$(round(ds_fd,sigdigits=6))  ana=$(round(ds_ana,sigdigits=6))  rel_err=$(round(err,sigdigits=2))")
        @assert err < 1e-6  "dsrel/de identity violated at e=$e_test, rel_err=$err"
    end
end
println("PASS")

println("\nAll EOS tests passed.")
