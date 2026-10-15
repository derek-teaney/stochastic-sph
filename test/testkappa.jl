include(joinpath(@__DIR__, "..", "src", "sph.jl"))

# ── Helpers ───────────────────────────────────────────────────────────────────

# Build a Particle with all point-wise thermodynamic fields set from (vx, vy, T_rel, S).
# Bypasses solve_eos! so tests don't depend on EOS Newton convergence.
# S is treated as the rest-frame entropy density (= srel); pa.e is obtained by inverting
# srel(e) = (1+cs²)/T_scale · e^(1/(1+cs²)) = S  →  e = (S·T_scale/(1+cs²))^(1+cs²).
# Fields left unevaluated (require kernel sums): x, s, M, dotx, Teff.
function make_test_particle(vx, vy, T_rel, S, eos::ConformalEOS)
    v2  = vx^2 + vy^2
    γ   = 1 / sqrt(1 - v2)
    v   = SVector(vx, vy)
    e   = (S * eos.T_scale / (1 + eos.cs2_val))^(1 + eos.cs2_val)  # srel(eos,e) = S
    p   = pressure(eos, e)
    w   = e + p                                    # enthalpy density
    pa  = Particle(SVector(0.0, 0.0), 1.0)         # x, s placeholders
    pa.S_smooth  = S
    pa.M_smooth  = w * γ^2 .* v                    # momentum density M^i = wγ²v^i
    pa.e         = e
    pa.p         = p
    pa.T         = T_rel / γ
    pa.γ         = γ
    pa.v_over_S  = v / S
    pa.F_over_S  = dot(v, pa.M_smooth / S) / S - p / S^2
    pa.H_over_S  = (w * γ^2 - p) / S
    return pa
end

# Rotate a 2D vector by angle θ.
rot2(v, θ) = SVector(cos(θ)*v[1] - sin(θ)*v[2],  sin(θ)*v[1] + cos(θ)*v[2])

# Apply rotation R (2×2 matrix given as SMatrix) to a rank-4 tensor K.
function rotate_kappa(K, R)
    out = zeros(2, 2, 2, 2)
    for i in 1:2, j in 1:2, l in 1:2, m in 1:2
        s = 0.0
        for a in 1:2, b in 1:2, c in 1:2, d in 1:2
            s += R[i,a] * R[j,b] * K[a,b,c,d] * R[l,c] * R[m,d]
        end
        out[i,j,l,m] = s
    end
    return out
end

# ── Tests ─────────────────────────────────────────────────────────────────────

cs2v   = 1.0/3.0
η_by_s = 0.2
ζ_by_s = 0.0
eos    = ConformalEOS(cs2v, 1.0, η_by_s, ζ_by_s)
T_rel  = 1.5
S      = 2.0
η      = η_by_s * S   # = 0.4
tol    = 1e-12

println("=" ^ 60)
println("testkappa.jl — κ^{ijlm} tensor tests (d=3 formula, 2D motion)")
println("=" ^ 60)

# ── Test 1: Rest-frame values ──────────────────────────────────────────────
# At v=0: T_rel κ^{xxxx} = 4η/3, κ^{xxyy} = −2η/3, κ^{xyxy} = η.
println("\nTest 1: rest-frame values (v=0)")
let
    pa = make_test_particle(0.0, 0.0, T_rel, S, eos)
    K  = kappa_tensor(pa, eos, Kappa3D())

    expected_xxxx = T_rel * (4η/3)
    expected_xxyy = T_rel * (-2η/3)
    expected_xyxy = T_rel * η

    err_xxxx = abs(K[1,1,1,1] - expected_xxxx)
    err_xxyy = abs(K[1,1,2,2] - expected_xxyy)
    err_xyxy = abs(K[1,2,1,2] - expected_xyxy)

    @assert err_xxxx < tol "κ^{xxxx}|_0 error = $err_xxxx"
    @assert err_xxyy < tol "κ^{xxyy}|_0 error = $err_xxyy"
    @assert err_xyxy < tol "κ^{xyxy}|_0 error = $err_xyxy"
    println("  κ[1,1,1,1] = $(K[1,1,1,1])  (expected $(expected_xxxx))  ✓")
    println("  κ[1,1,2,2] = $(K[1,1,2,2])  (expected $(expected_xxyy))  ✓")
    println("  κ[1,2,1,2] = $(K[1,2,1,2])  (expected $(expected_xyxy))  ✓")
end

# ── Test 2: 1D analytic formula ───────────────────────────────────────────
# For vy=0, vx=v:  T_rel κ^{xxxx} = (4/3) T_rel η / (γ^4 (1−cs²vx²)²)
println("\nTest 2: 1D analytic formula (vy=0, vx=0.7)")
let
    vx  = 0.7
    v2  = vx^2
    γ   = 1 / sqrt(1 - v2)
    pa  = make_test_particle(vx, 0.0, T_rel, S, eos)
    K   = kappa_tensor(pa, eos, Kappa3D())

    expected = (4/3) * T_rel * η / (γ^4 * (1 - cs2v * vx^2)^2)
    err      = abs(K[1,1,1,1] - expected)
    @assert err < tol * abs(expected) "κ^{xxxx} 1D error = $err"
    println("  κ[1,1,1,1] = $(K[1,1,1,1])")
    println("  analytic   = $expected")
    println("  rel error  = $(err / abs(expected))  ✓")
end

# ── Test 3: Index symmetry ─────────────────────────────────────────────────
# κ[i,j,l,m] = κ[j,i,l,m] = κ[i,j,m,l] = κ[l,m,i,j]
println("\nTest 3: index symmetry at (vx, vy) = (0.5, 0.3)")
let
    pa = make_test_particle(0.5, 0.3, T_rel, S, eos)
    K  = kappa_tensor(pa, eos, Kappa3D())
    fails = 0
    for i in 1:2, j in 1:2, l in 1:2, m in 1:2
        abs(K[i,j,l,m] - K[j,i,l,m]) > tol && (fails += 1)  # i↔j
        abs(K[i,j,l,m] - K[i,j,m,l]) > tol && (fails += 1)  # l↔m
        abs(K[i,j,l,m] - K[l,m,i,j]) > tol && (fails += 1)  # (ij)↔(lm)
    end
    @assert fails == 0 "$fails symmetry violations"
    println("  all 16×3 symmetry checks passed  ✓")
end

# ── Test 4: Rotation invariance ────────────────────────────────────────────
# κ(R·v) = R⊗R⊗R⊗R · κ(v) for any rotation R.
println("\nTest 4: rotation invariance (θ = 0.7 rad, v=0.6 in x)")
let
    v0 = 0.6
    θ  = 0.7
    c  = cos(θ);  s = sin(θ)
    R  = SMatrix{2,2,Float64}(c, s, -s, c)   # column-major: R[:,1]=(c,s), R[:,2]=(-s,c)

    pa0 = make_test_particle(v0, 0.0, T_rel, S, eos)
    K0  = kappa_tensor(pa0, eos, Kappa3D())

    vrot = rot2(SVector(v0, 0.0), θ)
    par  = make_test_particle(vrot[1], vrot[2], T_rel, S, eos)
    Kr   = kappa_tensor(par, eos, Kappa3D())

    K0_rotated = rotate_kappa(K0, R)

    max_err = maximum(abs(K0_rotated[i,j,l,m] - Kr[i,j,l,m])
                      for i in 1:2, j in 1:2, l in 1:2, m in 1:2)
    rtol = max_err / maximum(abs.(Kr))
    @assert rtol < 1e-10 "rotation invariance max rel error = $rtol"
    println("  max relative error = $rtol  ✓")
end

# ── Test 5: Bulk tensor factorization ─────────────────────────────────────
# κ_bulk^{ijlm} = L^{ij}L^{lm} × T_rel ζ / (1−cs²v²)²  where L^{ij} = δ^{ij}−v^iv^j.
# Verify every element at a non-trivial velocity.
println("\nTest 5: bulk tensor = L^{ij}L^{lm} × T_rel ζ / (1−cs²v²)² (vx=0.5, vy=0.3)")
let
    ζ_test = 0.3
    eos_b  = ConformalEOS(cs2v, 1.0, 0.0, ζ_test)
    vx, vy = 0.5, 0.3
    pa     = make_test_particle(vx, vy, T_rel, S, eos_b)

    v2  = vx^2 + vy^2
    ζ   = ζ_test * S
    K   = kappa_tensor(pa, eos_b, Kappa3D())

    # Analytic expected value element by element
    Lmat = [1-vx^2  -vx*vy; -vx*vy  1-vy^2]
    prefac = T_rel * ζ / (1 - cs2v * v2)^2

    max_err = 0.0
    for i in 1:2, j in 1:2, l in 1:2, m in 1:2
        max_err = max(max_err, abs(K[i,j,l,m] - Lmat[i,j] * Lmat[l,m] * prefac))
    end
    @assert max_err < tol "bulk factorization max error = $max_err"
    println("  max element error = $max_err  ✓")
end

# ── Test 6: bulk_2d A–F identity with kappa_bulk_geom_3d ──────────────────
# bulk_2d.txt gives A–F directly; confirm they assemble to the same tensor
# as kappa_bulk_geom_3d, which derives them from L^{ij} = δ^{ij} − v^iv^j.
println("\nTest 6: bulk_2d A–F formulas match kappa_bulk_geom_3d (vx=0.5, vy=0.3)")
let
    vx, vy = 0.5, 0.3
    vx2 = vx^2;  vy2 = vy^2

    A = (-1+vx2)^2
    B = vx*vy*(-1+vx2)
    C = (-1+vx2)*(-1+vy2)
    D = vx2*vy2
    E = vx*vy*(-1+vy2)
    F = (-1+vy2)^2

    K_2d = SArray{Tuple{2,2,2,2},Float64}(
        A, B, B, C,
        B, D, D, E,
        B, D, D, E,
        C, E, E, F)

    K_3d = kappa_bulk_geom_3d(vx, vy)

    max_err = maximum(abs(K_2d[i,j,l,m] - K_3d[i,j,l,m])
                      for i in 1:2, j in 1:2, l in 1:2, m in 1:2)
    @assert max_err < tol "bulk_2d vs kappa_bulk_geom_3d max error = $max_err"
    println("  max element error = $max_err  ✓")
end

println("\n" * "=" ^ 60)
println("All testkappa.jl tests passed.")
println("=" ^ 60)
