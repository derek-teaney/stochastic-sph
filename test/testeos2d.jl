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

# 2D conformal fluid: cs2 = 1/2 (vs 1/3 in testeos.jl).  T_scale=1.0 is an
# arbitrary unit choice (no 2D Stefan-Boltzmann constant exists yet — see
# sph.jl's T_scale_gluon, which is 3D-specific); this file only checks the
# EOS-solver mechanics at cs2=1/2, not a physically normalized 2D gas.
const cs2_val = 0.5
const Tscale  = 1.0
const eos = ConformalEOS(cs2_val=cs2_val, T_scale=Tscale)

# ── Helper: causality bound M_crit(S) ──────────────────────────────────────────
# solve_eos! has a real root for M < M_crit(S), none for M > M_crit(S) (found by
# bisecting on the minimum, over e, of w·γ/Trel(e) − S).  Recomputed from
# cs2_val/Tscale so test cases below stay valid if those constants change.
function critical_M(S::Float64; e_max=50.0, n=20_000)
    function fmin(M)
        best = Inf
        for e in range(1e-6, e_max, length=n)
            w = e + pressure(eos, e)
            u = M / w
            γ = sqrt(1.0 + u^2)
            g = w * γ / Trel(eos, e) - S
            g < best && (best = g)
        end
        best
    end
    lo, hi = 0.0, max(1.0, S^(1.0 + cs2_val))
    while fmin(hi) < 0.0
        hi *= 2
    end
    for _ in 1:50
        mid = (lo + hi) / 2
        fmin(mid) > 0.0 ? (hi = mid) : (lo = mid)
    end
    return (lo + hi) / 2
end

# (Mx, My) at a target fraction of the causality bound, along direction θ.
function momentum_at(S::Float64, frac::Float64, θ::Float64)
    m = frac * critical_M(S)
    return m * cos(θ), m * sin(θ)
end

# ── Test 1: ConformalEOS interface ─────────────────────────────────────────────

println("=" ^ 55)
println("Test 1: ConformalEOS interface (cs2_val=$cs2_val, Tscale=$Tscale)")
println("=" ^ 55)

let
    for e_test in [0.1, 1.0, 5.0, 100.0]
        p_val  = pressure(eos, e_test)
        T_val  = Trel(eos, e_test)
        c2_val = cs2(eos, e_test)
        @assert p_val  ≈ cs2_val * e_test                                "pressure wrong at e=$e_test"
        @assert T_val  ≈ Tscale * e_test^(cs2_val / (1.0 + cs2_val))     "Trel wrong at e=$e_test"
        @assert c2_val ≈ cs2_val                                         "cs2 wrong at e=$e_test"
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
# General conformal-fluid identity at rest (M=0, γ=1):
#   S = (1+cs2)/T_scale * e^{1/(1+cs2)}  ⟹  e = (T_scale*S/(1+cs2))^{1+cs2}

println("\n", "=" ^ 55)
println("Test 3: solve_eos! at rest  (M=0, analytic answer)")
println("=" ^ 55)

let
    for S_in in [0.5, 1.0, 2.0, 10.0]
        pa      = make_particle(S_smooth=S_in)
        solve_eos!(pa, eos)
        e_exact = (Tscale * S_in / (1.0 + cs2_val))^(1.0 + cs2_val)
        p_exact = cs2_val * e_exact
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
# Physical constraint: a root exists only below a causality bound M_crit(S)
# (larger cs2 leaves less room for relativistic momentum before solve_eos!
# has no real solution — see critical_M above). Test cases below are chosen
# as fractions of that bound, recomputed live from cs2_val/Tscale.

println("\n", "=" ^ 55)
println("Test 4: solve_eos! residual — relativistic M≠0")
println("=" ^ 55)

let
    θ34 = atan(3.0, 4.0)
    Mx1, My1 = momentum_at(1.0, 0.35, 0.0)
    Mx2, My2 = momentum_at(1.0, 0.66, θ34)
    Mx3, My3 = momentum_at(2.0, 0.55, θ34)
    Mx4, My4 = momentum_at(0.5, 0.33, θ34)
    test_cases = [
        (S_smooth=1.0, Mx=Mx1, My=My1),
        (S_smooth=1.0, Mx=Mx2, My=My2),
        (S_smooth=2.0, Mx=Mx3, My=My3),
        (S_smooth=0.5, Mx=Mx4, My=My4),
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
    Mx5, My5 = momentum_at(1.5, 0.4, atan(3.0, 4.0))
    pa  = make_particle(S_smooth=1.5, Mx_smooth=Mx5, My_smooth=My5)
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
    Mx6, My6 = momentum_at(1.0, 0.66, atan(3.0, 4.0))
    pa = make_particle(S_smooth=1.0, Mx_smooth=Mx6, My_smooth=My6)
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
    Mx7, My7 = momentum_at(1.0, 0.66, atan(3.0, 4.0))
    pa = make_particle(S_smooth=1.0, Mx_smooth=Mx7, My_smooth=My7)
    solve_eos!(pa, eos; verbose=true)
end
println("PASS")

# ── Test 8: dsrel/de = srel/(e+p)  (thermodynamic identity) ──────────────────
#
# From srel = (e+p)/Trel and dTrel/de = cs2·Trel/(e+p):
#   dsrel/de = (1+cs2)/Trel − cs2/Trel = 1/Trel = srel/(e+p).
# Checked here at cs2_val/Tscale (the same eos used throughout this file).

println("\n", "=" ^ 55)
println("Test 8: dsrel/de = srel/(e+p)  (finite-difference check)")
println("=" ^ 55)

let δ = 1e-7
    for e_test in [0.1, 1.0, 3.0, 10.0]
        ds_fd  = (srel(eos, e_test + δ) - srel(eos, e_test - δ)) / (2δ)
        ds_ana = srel(eos, e_test) / (e_test + pressure(eos, e_test))
        err    = abs(ds_fd - ds_ana) / abs(ds_ana)
        println("  e=$e_test  fd=$(round(ds_fd,sigdigits=6))  ana=$(round(ds_ana,sigdigits=6))  rel_err=$(round(err,sigdigits=2))")
        @assert err < 1e-6  "dsrel/de identity violated at e=$e_test, rel_err=$err"
    end
end
println("PASS")

println("\nAll 2D EOS tests passed.")
