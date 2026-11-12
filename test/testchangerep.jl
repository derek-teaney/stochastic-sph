# testchangerep.jl — the viscous step in the particle representation (docs/changerep.tex),
# params.dissipation = :particle.
#
#   1  conversion map: S_a(E_a(s_a, m_a), m_a) = s_a for a moving fluid
#   2  1D lattice: the rate with the particle β is the bond response B(k); one step
#      multiplies a mode by 1 − Γ_phys B Δt; the first-order step produces ΣΔs = 2KΔt
#   3  smooth shear flow (2D): the gradient energy Σg_a accounts for the excess Σe − H,
#      and the gradient-corrected step changes H much less than the exact step
#   4  random momentum field (2D): the default step conserves momentum and produces entropy
#      (at the half step length generic_step! uses)
#   5  the default time step of compute_dt: every mode decays monotonically
#
# Run:  julia --project test/testchangerep.jl
using StaticArrays, LinearAlgebra, Printf, Random
# The change of representation belongs to the smoothed Hamiltonian (with :energy, Σe = H and
# there is no excess to account for), so this test runs it whatever the default scheme.
ENV["SPH_HAMILTONIAN"] = "smoothed"
include(joinpath(@__DIR__, "..", "src", "load.jl"))

Hof(ps, params, cl, eos) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
                            sum(p.s * p.H_over_S for p in ps))
println("=" ^ 60, "\ntestchangerep.jl — viscous step in the particle representation\n", "=" ^ 60)

# ── 1 ────────────────────────────────────────────────────────────────────────
println("\nTest 1: conversion map round trip, S_a(E_a(s_a, m_a), m_a) = s_a")
let
    eos = ConformalEOS(1/3, 1.0, 0.1, 0.0)
    params, ps, cl, eos = setup_ic(0.08, 2.5, eos, (x, y) -> 1.0,
                                   (x, y) -> SVector(0.4 * sin(2π * x), 0.3 * cos(2π * y)); L=1.0, lattice=:hex)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    err = maximum(begin
                      V = p.s / p.S_smooth
                      abs(V * eos_from_energy(particle_energy(p, eos) / V, p.m / V, eos)[1] - p.s) / p.s
                  end for p in ps)
    vmax = maximum(norm(p.v_over_S) * p.S_smooth for p in ps)
    @printf("  |v| up to %.2f:  max relative error %.1e\n", vmax, err)
    @assert err < 1e-10 "conversion map round trip failed: $err"
    println("  ✓")
end

# ── 2 ────────────────────────────────────────────────────────────────────────
println("\nTest 2: 1D lattice — rates B(k), one-step factor 1 − ΓBΔt, ΣΔs = 2KΔt")
let
    KER = WendlandC2_1D()
    Wp(r, h) = (q = abs(r) / h; q >= 2 ? 0.0 : sign(r) * (-5 * 3 / (4h)) * q * (1 - q / 2)^3 / h)
    ψ1(r, h) = r == 0 ? 0.0 : -Wp(r, h) / r
    lat(h, a) = [m * a for m in -ceil(Int, 2h / a):ceil(Int, 2h / a)]
    Bk(k, h, a) = sum(ψ1(r, h) * 2 * (1 - cos(k * r)) for r in lat(h, a)) /
                  (k^2 * sum(ψ1(r, h) * r^2 for r in lat(h, a)))
    eos = ConformalEOS(1/3, 1.0, 0.1, 0.0)
    params, ps, cl, eos = setup_ic(0.02, 4.0, eos, (x, y) -> 2.0, (x, y) -> SVector(0.0, 0.0);
                                   Lx=1.0, sim_dim=1)
    params.dissipation = :particle
    a, N, L, h = params.a, params.N, params.L[1], params.h
    w = ps[1].e + ps[1].p;  κ = 4/3 * 0.1 * srel(eos, ps[1].e)
    s0 = [p.s for p in ps];  f = zeros(SVector{2,Float64}, N)
    dt = 0.5 * compute_dt(ps, params, eos; mode=:viscous)
    erate = 0.0;  eg = 0.0;  eS = 0.0
    for n in (3, 17, 40, 100)
        k = 2π * n / L
        set!() = for (i, p) in enumerate(ps); p.s = s0[i]; p.m = SVector(1e-5 * w * a * cos(k * (p.x[1] - a / 2)), 0.0); end
        proj() = sum(p.m[1] * cos(k * (p.x[1] - a / 2)) for p in ps)
        set!();  thermalize!(ps, params, cl, eos)
        dissipative_force!(f, ps, params, cl, eos)
        Γ = -sum(f[i][1] * ps[i].m[1] for i in 1:N) / sum(p.m[1]^2 for p in ps)
        erate = max(erate, abs(Γ / (κ / w * k^2) - Bk(k, h, a)))
        set!();  A0 = proj()
        viscous_step_exact!(ps, params, cl, dt, eos)
        eg = max(eg, abs(proj() / A0 - (1 - κ / w * k^2 * Bk(k, h, a) * dt)))
        set!()
        K, dS = viscous_step_first!(ps, params, cl, dt, eos)
        eS = max(eS, abs(dS - 2K * dt) / (2K * dt))
    end
    @printf("  max |Γ/Γ_phys − B| = %.1e   max |g − (1 − ΓBΔt)| = %.1e   max |ΣΔs/(2KΔt) − 1| = %.1e\n",
            erate, eg, eS)
    @assert erate < 1e-8 && eg < 1e-8 && eS < 1e-10 "1D rates or step factor wrong"
    println("  ✓")
end

# ── 3 ────────────────────────────────────────────────────────────────────────
println("\nTest 3: smooth shear flow — gradient energy and the corrected step")
let
    eos = ConformalEOS(1/3, 1.0, 0.1, 0.0)
    params, ps, cl, eos = setup_ic(0.05, 2.5, eos, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                   L=1.0, lattice=:hex)
    params.dissipation = :particle
    s0 = [p.s for p in ps]
    set!() = for (i, p) in enumerate(ps); p.s = s0[i]; p.m = SVector(0.0, 1e-3 * sin(2π * p.x[1]) * p.s); end
    set!()
    H = Hof(ps, params, cl, eos);  E = sum(particle_energy(p, eos) for p in ps)
    μ = [p.m / p.s for p in ps]
    g = sum(gradient_energy(ps[a].s, inverse_susceptibility(ps[a], eos),
                            star_fit(kernel_star(a, ps, params, cl), μ)...) for a in eachindex(ps))
    ratio = Dict{Symbol,Float64}()
    for (kind, step!) in ((:exact, viscous_step_exact!), (:grad, viscous_step_grad!))
        set!();  H0 = Hof(ps, params, cl, eos)
        K, dS = step!(ps, params, cl, 1e-4, eos)
        ratio[kind] = (Hof(ps, params, cl, eos) - H0) / (dS * ps[1].T)
    end
    @printf("  Σg/(Σe − H) = %.3f   ΔH/(TΔS): exact %.4f, gradient-corrected %.4f\n",
            g / (E - H), ratio[:exact], ratio[:grad])
    @assert 0.9 < g / (E - H) < 1.1 "gradient energy does not account for the excess"
    @assert abs(ratio[:grad]) < 0.1 * abs(ratio[:exact]) "gradient correction ineffective"
    println("  ✓")
end

# ── 4 ────────────────────────────────────────────────────────────────────────
println("\nTest 4: random momentum field — momentum conservation and entropy production")
let
    eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    params, ps, cl, eos = setup_ic(0.05, 2.5, eos, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                   L=1.0, lattice=:hex)
    @assert params.dissipation === :particle "default dissipation is not :particle"
    rng = Xoshiro(1)
    for p in ps; p.m = 1e-3 * p.s * SVector(randn(rng), randn(rng)); end
    P0 = sum(p.m for p in ps);  S0 = sum(p.s for p in ps)
    # half the viscous Δt, as generic_step! uses for each viscous half step: with the
    # particle β the shortest modes are damped strongly, and the full Δt makes the explicit
    # step unstable in 2D (ΓΔt > 2).
    dt = 0.5 * compute_dt(ps, params, eos; mode=:viscous)
    for _ in 1:10; viscous_step!(ps, params, cl, dt, eos); end
    dP = norm(sum(p.m for p in ps) - P0) / sum(norm(p.m) for p in ps)
    dS = sum(p.s for p in ps) - S0
    @printf("  relative momentum change %.1e   ΔS = %.3e\n", dP, dS)
    @assert dP < 1e-13 "momentum not conserved"
    @assert dS > 0 "no entropy produced"
    println("  ✓")
end

# ── 5 ────────────────────────────────────────────────────────────────────────
println("\nTest 5: default time step — every mode decays monotonically in each viscous half step")
let
    # 1D: the per-step factor of a mode, measured, lies in [0, 1] for all k
    eos = ConformalEOS(1/3, 1.0, 0.1, 0.0)
    params, ps, cl, eos = setup_ic(0.02, 4.0, eos, (x, y) -> 2.0, (x, y) -> SVector(0.0, 0.0);
                                   Lx=1.0, sim_dim=1)
    a, N, L = params.a, params.N, params.L[1];  w = ps[1].e + ps[1].p;  s0 = [p.s for p in ps]
    thermalize!(ps, params, cl, eos)
    dt = compute_dt(ps, params, eos; mode=:viscous)
    gs = Float64[]
    for n in 1:N ÷ 2
        k = 2π * n / L
        for (i, p) in enumerate(ps); p.s = s0[i]; p.m = SVector(1e-6 * w * a * cos(k * (p.x[1] - a / 2)), 0.0); end
        proj() = sum(p.m[1] * cos(k * (p.x[1] - a / 2)) for p in ps)
        A0 = proj();  viscous_step!(ps, params, cl, dt / 2, eos);  push!(gs, proj() / A0)
    end
    @printf("  1D: per-step factors in [%.3f, %.3f]\n", minimum(gs), maximum(gs))
    @assert all(0 .<= gs .<= 1) "a mode is not damped monotonically"
    # 2D: random field, kinetic energy decreases and entropy increases at every half step
    eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    params, ps, cl, eos = setup_ic(0.05, 2.5, eos, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                   L=1.0, lattice=:hex)
    rng = Xoshiro(2)
    for p in ps; p.m = 1e-3 * p.s * SVector(randn(rng), randn(rng)); end
    thermalize!(ps, params, cl, eos)
    dt = compute_dt(ps, params, eos; mode=:viscous)
    KE() = sum(dot(p.m, p.m) / p.s for p in ps)
    ok = true;  rmax = 0.0
    for _ in 1:8
        k0 = KE();  S0 = sum(p.s for p in ps)
        viscous_step!(ps, params, cl, dt / 2, eos)
        rmax = max(rmax, KE() / k0)
        ok &= KE() < k0 && sum(p.s for p in ps) > S0
    end
    @printf("  2D: largest kinetic-energy ratio per half step %.3f, entropy increasing: %s\n", rmax, ok)
    @assert ok "kinetic energy or entropy not monotone in 2D"
    println("  ✓")
end

println("\nAll testchangerep.jl tests passed.")
