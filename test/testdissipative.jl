using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

# ── Shared resolution ─────────────────────────────────────────────────────────
const h_test  = 0.05
const ha_test = 8.0
const L_test  = 1.0
const cs2v    = 1.0 / 3.0
const s0      = 1.0     # entropy density (per unit area)

# Build a shear-wave IC (k=4π/L) with given M_amp for a dissipative test.
function make_test_ic(M_amp; η_by_s=0.0, jitter=0.0)
    params, particles, cl, eos, _, _ = setup_shear_wave(h_test, ha_test;
        L=L_test, M_amp, s0, cs2v, η_by_s, jitter)
    return params, particles, cl, eos
end

println("=" ^ 60)
println("testdissipative.jl — GENERIC dissipative step tests")
println("=" ^ 60)

# ── Test 1: Zero viscosity → no force, entropy conserved ─────────────────────
println("\nTest 1: zero viscosity (η=ζ=0) → zero force everywhere")
let
    params, particles, cl, eos = make_test_ic(0.1; η_by_s=0.0)
    N    = params.N
    dotm = zeros(SVector{2,Float64}, N)
    dissipative_force!(dotm, particles, params, cl, eos)

    max_force = maximum(norm(f) for f in dotm)
    @assert max_force < 1e-14 "zero-viscosity force = $max_force"

    s_before = sum(pa.s for pa in particles)
    dissipative_step!(particles, params, cl, 0.01, eos)
    s_after = sum(pa.s for pa in particles)
    Δs = s_after - s_before
    @assert abs(Δs) < 1e-14 "entropy changed by $Δs with zero viscosity"
    println("  max force = $max_force  ✓")
    println("  Δs_total  = $Δs  ✓")
end

# ── Test 2: Entropy production ≥ 0 for every particle ────────────────────────
println("\nTest 2: entropy production ≥ 0 for all particles (nonzero η, v≠0)")
let
    η_by_s = 0.3
    params, particles, cl, eos = make_test_ic(0.2; η_by_s)
    N    = params.N
    dotm = zeros(SVector{2,Float64}, N)
    dissipative_force!(dotm, particles, params, cl, eos)

    dt        = 0.001
    neg_count = 0
    for a in eachindex(particles)
        Δm   = dt .* dotm[a]
        Δs_a = -dot(particles[a].dotx / particles[a].Teff, Δm)
        Δs_a < -1e-14 && (neg_count += 1)
    end
    @assert neg_count == 0 "$neg_count particles with Δs < 0"
    println("  neg-entropy particles = $neg_count / $N  ✓")
end

# ── Test 3: Momentum conservation (Newton's 3rd law) ─────────────────────────
println("\nTest 3: total momentum conserved (Σ dotm = 0)")
let
    η_by_s = 0.2
    params, particles, cl, eos = make_test_ic(0.15; η_by_s)
    N    = params.N
    dotm = zeros(SVector{2,Float64}, N)
    dissipative_force!(dotm, particles, params, cl, eos)

    total     = sum(dotm)
    err       = sqrt(total[1]^2 + total[2]^2)
    sum_norms = sum(norm(f) for f in dotm)
    rel_err   = err / sum_norms
    @assert rel_err < 1e-14 "total viscous force rel. error = $rel_err (should be < 1e-14)"
    println("  |Σ dotm|        = $err")
    println("  Σ |dotm|        = $sum_norms")
    println("  rel. error           = $rel_err  ✓")
end

# ── Test 4: κ tensor spot-check on a few particles ───────────────────────────
println("\nTest 4: κ^{ijlm} tensor components at two sample particles")
let
    η_by_s = 0.2
    params, particles, cl, eos = make_test_ic(0.15; η_by_s, jitter=0.6)

    for idx in [1, params.N ÷ 2]
        pa    = particles[idx]
        K     = kappa_tensor(pa, eos, Kappa3D())
        vx    = pa.v_over_S[1] * pa.S_smooth
        vy    = pa.v_over_S[2] * pa.S_smooth
        v2    = vx^2 + vy^2
        T_rel = pa.T * pa.γ
        sr    = srel(eos, pa.e)
        # For shear wave (vx=0): κ[x,y,x,y] = T_rel·srel·η/s·(1−v²) exactly.
        expected = T_rel * sr * η_by_s * (1 - v2)
        xyxy_err = abs(K[1,2,1,2] - expected) / abs(expected)
        println("  particle $idx:  x=$(round.(pa.x, sigdigits=4))" *
                "  v=($(round(vx,sigdigits=4)), $(round(vy,sigdigits=4)))" *
                "  T_rel=$(round(T_rel, sigdigits=5))  srel=$(round(sr, sigdigits=5))")
        println("    κ[x,x,x,x] = $(round(K[1,1,1,1], sigdigits=6))")
        println("    κ[x,x,y,y] = $(round(K[1,1,2,2], sigdigits=6))")
        println("    κ[x,y,x,y] = $(round(K[1,2,1,2], sigdigits=6))" *
                "  expected T_rel·srel·η/s·(1−v²) = $(round(expected, sigdigits=6))" *
                "  rel_err=$(round(xyxy_err, sigdigits=2))")
        println("    κ[y,y,y,y] = $(round(K[2,2,2,2], sigdigits=6))")
        @assert xyxy_err < 1e-12 "κ[x,y,x,y] mismatch at particle $idx: rel_err=$xyxy_err"
    end
end

println("\n" * "=" ^ 60)
println("All testdissipative.jl tests passed.")
println("=" ^ 60)
