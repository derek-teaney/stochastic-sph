# testeqstart.jl — the equilibrium start with energy fluctuations (docs/eqstart.tex,
# equilibrium_init! in src/measure.jl).
#
#   A. One particle: the sampler against direct quadrature of exp[S_a(e,m) − β⁰e + β·m],
#      at rest and moving, at small and large entropy per particle; the most probable
#      state is stationary.
#   B. A lattice with exact_totals = true: totals E, M exact; T and v recovered; spread over
#      particles = the exact single-particle covariance.
#   E. Canonical: over many configurations the totals (E, P) fluctuate about the target
#      with the covariance N·C_a (the global susceptibility).
#   C. Consistency with the SPH representation after re-smoothing (reported).
#   D. A short coupled run from this start (reported).
#
# Run:  julia --project test/testeqstart.jl
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "src", "load.jl"))
check(ok, msg) = (println(ok ? "  ✓ " : "  ✗ FAIL: ", msg); ok || error(msg))
const eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)

# ── A. One particle ──────────────────────────────────────────────────────────
function one_particle(ε̄, v, V; n = 40000, nq = 61, rng = Xoshiro(1))
    # rest-frame state with velocity v and energy density ε̄ in the frame of the box
    γ2 = 1 / (1 - v^2)
    e_r = find_zero_e(ε̄, v)                        # rest-frame e with wγ² − p = ε̄
    w = e_r * (1 + 1/3);  M̄ = SVector(w * γ2 * v, 0.0)
    _, vv, T = eos_from_energy(ε̄, M̄, eos)
    β0 = 1 / T;  β = vv / T;  ē = V * ε̄;  m̄ = V * M̄
    # stationarity: ∂S_a = (β⁰, −β) at the most probable state
    δ = 1e-4 * sqrt(ē * T)
    g = [(particle_entropy(ē + δ, m̄, V, eos) - particle_entropy(ē - δ, m̄, V, eos)) / 2δ,
         (particle_entropy(ē, m̄ + SVector(δ, 0), V, eos) - particle_entropy(ē, m̄ - SVector(δ, 0), V, eos)) / 2δ,
         (particle_entropy(ē, m̄ + SVector(0, δ), V, eos) - particle_entropy(ē, m̄ - SVector(0, δ), V, eos)) / 2δ]
    sa = particle_entropy(ē, m̄, V, eos)
    @printf("  s_a = %.0f, v = %.2f:  ∂S/∂e·T = %.6f,  ∂S/∂m·T = (%.6f, %.6f),  expected 1, (−%.2f, 0)\n",
            sa, v, g[1] * T, g[2] * T, g[3] * T, v)
    check(abs(g[1] / β0 - 1) < 1e-5 && abs(g[2] + β[1]) < 1e-5 * β0 && abs(g[3]) < 1e-6 * β0, "most probable state is stationary")
    # quadrature on a grid of ±7 Gaussian standard deviations
    H = entropy_hessian(ē, m̄, V, eos, 0.02 * sqrt(ē * T));  Hn = SMatrix{3,3,Float64}(-H)
    C = inv(Matrix(Hn));  L = SMatrix{3,3,Float64}(cholesky(Symmetric(C)).L);  sd = sqrt.(diag(C))
    xs = [range(-7sd[i], 7sd[i], length = nq) for i in 1:3]
    lp = [particle_entropy(ē + x, m̄ + SVector(y, z), V, eos) - β0 * x + β[1] * y for x in xs[1], y in xs[2], z in xs[3]]
    wq = exp.(lp .- maximum(filter(isfinite, lp)));  wq[.!isfinite.(wq)] .= 0;  Z = sum(wq)
    X = [SVector(x, y, z) for x in xs[1], y in xs[2], z in xs[3]]
    μq = sum(wq .* X) / Z;  Cq = sum(wq .* [(x - μq) * (x - μq)' for x in X]) / Z
    # sampler
    R = [draw_particle(ē, m̄, V, β0, β, eos, L, Hn, rng) for _ in 1:n];  D = first.(R)
    @printf("    sampler acceptance %.3f\n", sum(last.(R)) / (30n))
    μs = mean(D);  Cs = sum((x - μs) * (x - μs)' for x in D) / (n - 1)
    @printf("    quadrature: mean/sd = (%+.3f, %+.3f, %+.3f)  var/Gaussian = (%.4f, %.4f, %.4f)\n",
            (μq ./ sd)..., (diag(Cq) ./ diag(C))...)
    @printf("    sampler:    mean/sd = (%+.3f, %+.3f, %+.3f)  var/Gaussian = (%.4f, %.4f, %.4f)\n",
            (μs ./ sd)..., (diag(Cs) ./ diag(C))...)
    check(all(abs.(μs - μq) ./ sqrt.(diag(Cq) ./ n) .< 4.5), "sampler mean = quadrature mean (within 4.5σ)")
    check(all(abs.(diag(Cs) ./ diag(Cq) .- 1) .< 4.5 * sqrt(2 / n)), "sampler variances = quadrature variances (within 4.5σ)")
    check(abs(Cs[1, 2] - Cq[1, 2]) / sqrt(Cq[1, 1] * Cq[2, 2]) < 4.5 / sqrt(n), "sampler e–m_x covariance = quadrature")
    return C, T, ē, V
end
# rest-frame e for box energy density ε̄ at speed v (conformal): ε̄ = (4/3) e γ² − e/3
find_zero_e(ε̄, v) = ε̄ / ((4/3) / (1 - v^2) - 1/3)

println("A. One particle: sampler vs quadrature")
for (ε̄, v, V) in ((1.0e8, 0.0, 5.4e-4), (1.0e8, 0.5, 5.4e-4), (3.2e6, 0.0, 5.4e-4), (3.2e6, 0.5, 5.4e-4))
    C, T, ē, Vp = one_particle(ε̄, v, V)
    if v == 0
        w = ε̄ * 4/3
        @printf("    at rest: var(e)/(4 V ε T) = %.4f,  var(m)/(V w T) = %.4f  (Gaussian, from the Hessian)\n",
                C[1, 1] / (4 * Vp * ε̄ * T), C[2, 2] / (Vp * w * T))
        check(abs(C[1, 1] / (4 * Vp * ε̄ * T) - 1) < 1e-3 && abs(C[2, 2] / (Vp * w * T) - 1) < 1e-3,
              "at rest: var(e) = V T² c_v, var(m) = V w T")
    end
end

# ── B. A lattice ─────────────────────────────────────────────────────────────
println("\nB. Lattice, h/a = 2, box 0.6")
# a uniform lattice moving along x: setup_ic takes the momentum density M(x, y), here
# w γ² v with w of the state at rest (the speed comes out close to v); vbg is the speed
# of the uniform state before the draw
function lattice(v; S0 = 1e6, seed = 1, exact_totals = true)
    _, p0, _, _ = setup_ic(0.05, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex)
    Mx = (p0[1].e + p0[1].p) * v / (1 - v^2)
    params, ps, cl, _ = setup_ic(0.05, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(Mx, 0.0); L = 0.6, lattice = :hex)
    thermalize!(ps, params, cl, eos)
    vbg = ps[1].v_over_S * ps[1].S_smooth
    E0 = sum(particle_energy(p, eos) for p in ps);  M0 = sum(p.m for p in ps)
    r = equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(seed), exact_totals)
    return params, ps, cl, r, E0, M0, vbg
end
for v in (0.0, 0.3)
    params, ps, cl, r, E0, M0, vbg = lattice(v)
    N = length(ps);  m = [p.m for p in ps]
    @printf("  v = %.1f: N = %d, acceptance %.3f, T_rel = %.4g, v = (%.4f, %.4f)\n", v, N, r.acc, r.T * sqrt(1 / (1 - dot(r.v, r.v))), r.v...)
    check(abs(sum(r.e) / E0 - 1) < 1e-12, "Σ e_a = E exactly")
    check(norm(sum(m) - M0) < 1e-10 * norm(sum(abs.(p.m) for p in ps)), "Σ m_a = M exactly")
    avg = sum(particle_moments(V_, r.β0, r.β, eos).mean for V_ in r.V)
    @printf("    background speed %.6f;  v = β/β⁰ of the most probable state %.6f (differs at O(1/s_a))\n", vbg[1], r.v[1])
    check(abs(avg[1] / E0 - 1) < 1e-10 && norm(SVector(avg[2], avg[3]) - M0) < 1e-10 * E0,
          "the averages at β reproduce the totals: Σ⟨e_a⟩ = E, Σ⟨m_a⟩ = M")
    check(r.acc > 0.4, "acceptance > 0.4 (30 independence steps: unmixed chains ~ 0.6^30)")
    # spread over particles against the single-particle covariance
    C = particle_moments(r.V[1], r.β0, r.β, eos).cov            # exact single-particle covariance
    X = [SVector(r.e[a], m[a][1], m[a][2]) for a in 1:N];  μ = mean(X)
    Cs = sum((x - μ) * (x - μ)' for x in X) / (N - 1)
    @printf("    spread over particles / C: e %.3f, m_x %.3f, m_y %.3f  (statistical ± %.3f)\n",
            Cs[1, 1] / C[1, 1], Cs[2, 2] / C[2, 2], Cs[3, 3] / C[3, 3], sqrt(2 / N))
    check(all(abs.(diag(Cs) ./ diag(C) .- 1) .< 4.5 * sqrt(2 / N)), "spread over particles = single-particle covariance")
end

# ── E. Canonical: fluctuations of the totals ─────────────────────────────────
println("\nE. Canonical ensemble: totals over many configurations (lattice volumes, N = 672)")
for (S0, v) in ((1e6, 0.0), (1e6, 0.3), (1e5, 0.0))
    _, p0, _, _ = setup_ic(0.05, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex)
    Mx = (p0[1].e + p0[1].p) * v / (1 - v^2)
    params, ps, cl, _ = setup_ic(0.05, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(Mx, 0.0); L = 0.6, lattice = :hex)
    V = [p.s / p.S_smooth for p in ps];  N = length(V)
    E = sum(particle_energy(p, eos) for p in ps);  M = sum(p.m for p in ps)
    β0, β, mom = beta_from_averages(V, E, M, eos)
    K = 3000;  rng = Xoshiro(21)
    tot = [sum(draw_configuration(V, β0, β, mom, eos, rng)[1]) for _ in 1:K]
    μ = mean(tot);  Ct = sum((t - μ) * (t - μ)' for t in tot) / (K - 1)
    Cpred = sum(mom(v_).cov for v_ in V)                          # N · C_a (equal volumes)
    sd = sqrt.(diag(Cpred))
    # thermodynamics: Var E = V_tot T² c_v, Var P = V_tot w T at rest (bare T = T_rel at rest)
    T = 1 / β0;  Vt = sum(V);  st = state_from_beta(β0, β, eos)
    @printf("  s_a = %.0f, v = %.2f: β⁰ T_most-probable-guess shift %+.2e;  ⟨E⟩/E − 1 = %+.1e (±%.1e),  ⟨P_x⟩−M_x = %+.2f σ\n",
            p0[1].s, β[1] / β0, β0 * (eos_from_energy(E / Vt, M / Vt, eos)[3]) - 1, μ[1] / E - 1, sd[1] / sqrt(K) / E, (μ[2] - M[1]) / (sd[2] / sqrt(K)))
    @printf("      Cov(totals)/(N C_a): EE %.3f  PxPx %.3f  PyPy %.3f  corr(E,Px) measured %+.3f predicted %+.3f   (statistical ± %.3f)\n",
            Ct[1, 1] / Cpred[1, 1], Ct[2, 2] / Cpred[2, 2], Ct[3, 3] / Cpred[3, 3],
            Ct[1, 2] / sqrt(Ct[1, 1] * Ct[2, 2]), Cpred[1, 2] / sqrt(Cpred[1, 1] * Cpred[2, 2]), sqrt(2 / K))
    if v == 0
        w = 4/3 * (st.ε)
        @printf("      thermodynamics (Gaussian): Var E/(V_tot T² c_v) = %.3f,  Var P_x/(V_tot w T) = %.3f\n",
                Ct[1, 1] / (Vt * T^2 * 4 * st.ε / T), Ct[2, 2] / (Vt * w * T))
    end
    check(all(abs.(μ - SVector(E, M[1], M[2])) ./ (sd ./ sqrt(K)) .< 4.5), "average totals = (E, M) within 4.5σ")
    check(all(abs.(diag(Ct) ./ diag(Cpred) .- 1) .< 4.5 * sqrt(2 / K)), "Cov(totals) = N C_a (global susceptibility)")
    check(abs(Ct[1, 2] / sqrt(Ct[1, 1] * Ct[2, 2]) - Cpred[1, 2] / sqrt(Cpred[1, 1] * Cpred[2, 2])) < 4.5 / sqrt(K), "E–P_x correlation as predicted")
end

# ── C. Consistency with the SPH representation ───────────────────────────────
println("\nC. After re-smoothing: energies and volumes recomputed from the SPH state")
let (params, ps, cl, r, E0, M0, _) = lattice(0.0; exact_totals = false)
    e′ = [particle_energy(p, eos) for p in ps];  V′ = [p.s / p.S_smooth for p in ps]
    s = [p.s for p in ps]
    H = sum(p.s * p.H_over_S for p in ps)
    @printf("  var(e′)/var(e drawn) = %.3f   corr(e′, e) = %.3f   rms(V′/V − 1) = %.3f   rms(s/s̄ − 1) = %.3f\n",
            var(e′) / var(r.e), cor(e′, r.e), sqrt(mean((V′ ./ r.V .- 1) .^ 2)), std(s) / mean(s))
    @printf("  Σe drawn/E − 1 = %+.2e   Σe′/E − 1 = %+.2e   H/E − 1 = %+.2e\n", sum(r.e) / E0 - 1, sum(e′) / E0 - 1, H / E0 - 1)
end

# ── D. A short coupled run from this start ───────────────────────────────────
println("\nD. Coupled run (leapfrog + stochastic step), t = 3: particle-energy variance over time")
for start in (:thermal_init, :equilibrium_init)
    params, ps, cl, _ = setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex)
    thermalize!(ps, params, cl, eos)
    E0 = sum(particle_energy(p, eos) for p in ps)
    r = equilibrium_init!(deepcopy(ps), params, deepcopy(cl), eos; rng = Xoshiro(9))   # for the reference variance
    vref = var(r.e)
    start === :thermal_init ? thermal_init!(ps, params, cl, eos; rng = Xoshiro(3)) :
                              equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(3), exact_totals = true)
    N = params.N;  rngs = make_phys_rng(N, 4);  rng = Xoshiro(5);  dotm = zeros(SVector{2,Float64}, N)
    dt = 0.5 * compute_dt(ps, params, eos; mode = :viscous);  nsteps = ceil(Int, 3.0 / dt)
    out = String[]
    for n in 0:nsteps
        if n > 0
            leapfrog_step!(ps, params, cl, dt, eos, dotm)
            stochastic_step!(ps, params, cl, dt, eos; rng, rngs, vote = :star, flip = isodd(n))
        end
        if n in (0, 10, 30, 100, nsteps)
            smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
            e = [particle_energy(p, eos) for p in ps]
            push!(out, @sprintf("t=%.2f: var(e)/var_eq %.3f", n * dt, var(e) / vref))
        end
    end
    @printf("  %-17s %s\n", start, join(out, "   "))
end
println("\nAll testeqstart.jl checks passed.")
