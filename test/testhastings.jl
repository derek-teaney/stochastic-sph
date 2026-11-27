# testhastings.jl — the Hastings correction of the stochastic step (SPH_HASTINGS=1, dissipation = :moment; docs/inverse_moment.tex,
# stage 2):
#   1. star_coef from the state of a particle reproduces the coefficients of build_stars (moving, jittered lattice);
#   2. the log ratio of the stress move, eq. qstress2, against the ratio of the two Gaussian densities of ξ computed directly;
#   3. the log ratio of the residual move, eq. qresid, against the ratio of the densities of the conditioned draws on the zero-virial
#      subspace of the 2n bond components (from their covariance, projected on an orthonormal basis of the subspace);
#   4. a stochastic step with the correction runs, conserves momentum, and leaves the entropies S_a(e_a, m_a).
using StaticArrays, LinearAlgebra, Printf, Random
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "0")   # tree level (split, the default, is tested in testsplit.jl): set before src/load.jl
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
eos = ConstantCs2EOS(1/3, 1.0, 0.2, 0.0)

params, ps, cl = let
    v0 = 0.5;  γ = 1 / sqrt(1 - v0^2)
    _, q, _, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 3e4, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex))
    w0 = q[1].e + q[1].p
    P, p, c, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> γ * 3e4, (x, y) -> SVector(w0 * γ^2 * v0, 0.0); L = 0.6, lattice = :hex,
                                       jitter = 0.3, seed = 4))
    quiet(() -> equilibrium_init!(p, P, c, eos; rng = Xoshiro(2)))           # thermal momenta (s_a ≈ 16)
    smooth_to_particles!(p, P, c);  update_thermodynamics!(p, eos);  update_velocities!(p, P, c)
    (P, p, c)
end
V = [p.s / p.S_smooth for p in ps];  e = [particle_energy(p, eos) for p in ps];  m = [p.m for p in ps]
stars = build_stars(ps, params, cl, eos, V);  d = params.phys_dim

println("1. star_coef against build_stars")
let err = 0.0
    for a in 1:50
        c = star_coef(e[a], m[a], V[a], eos, d);  st = stars[a]
        err = max(err, abs(c.T / st.T - 1), abs(c.κs / st.κs - 1), abs(c.κb / st.κb - 1), norm(c.ℓ - st.ℓ),
                  maximum(norm(c.B[k] - st.B[k]) / norm(st.B[k]) for k in 1:3),
                  maximum(abs(resid_c(c, st, k, V[a], params) / st.c[k] - 1) for k in eachindex(st.nb)))
    end
    @printf("  max relative difference over 50 stars: %.1e\n", err)
    check(err < 1e-10, "the coefficients of the particle's state are those of build_stars")
end

dt = 2.0 * compute_dt(ps, params, eos; mode = :viscous)            # a large step, so that the ratios are far from 1
println("2. the stress move: eq. qstress2 against the Gaussian densities")
let worst = 0.0
    for a in 1:20
        st = stars[a];  n = length(st.nb);  δ = Vector{SVector{2,Float64}}(undef, n)
        c = star_coef(e[a], m[a], V[a], eos, d)
        lq = propose_stress_hastings!(δ, st, c, e[a], m[a], V[a], dt, eos, d, Xoshiro(a))
        ξ = sum(δ[k] * st.r[k]' for k in 1:n) / V[a]                       # the closure: Σ_b ξ A_ab r_abᵀ = V_a ξ
        c′ = star_coef(e[a], m[a] - sum(δ), V[a], eos, d)
        Cm = (J -> J * J')(stress_J(c, V[a], dt));  Cp = (J -> J * J')(stress_J(c′, V[a], dt))
        x = symvec(ξ);  lN(C, y) = -0.5 * dot(y, C \ y) - 0.5 * log(det(C))
        direct = lN(Cp, -x) - lN(Cm, x)
        worst = max(worst, abs(lq - direct))
        a <= 3 && @printf("  star %d: ln q ratio %+.6f, direct %+.6f\n", a, lq, direct)
    end
    check(worst < 1e-9, @sprintf("eq. qstress2 = the direct ratio (max difference %.1e)", worst))
end

println("3. the residual move: eq. qresid against the density on the zero-virial subspace")
let worst = 0.0
    for a in 1:10
        st = stars[a];  n = length(st.nb);  δ = Vector{SVector{2,Float64}}(undef, n)
        c = star_coef(e[a], m[a], V[a], eos, d)
        lq = propose_resid_hastings!(δ, st, c, e[a], m[a], V[a], dt, eos, d, params, Xoshiro(100 + a))
        c′ = star_coef(e[a], m[a] - sum(δ), V[a], eos, d)
        # the 2n-dimensional Gaussian of the draws and the four virial constraints
        Σof(cc) = (S = zeros(2n, 2n);  for k in 1:n
                       S[2k-1:2k, 2k-1:2k] = 2dt * resid_c(cc, st, k, V[a], params) * Matrix(cc.ℓ);  end;  S)
        C = zeros(4, 2n)
        for k in 1:n, i in 1:2, j in 1:2
            C[2(i-1)+j, 2(k-1)+i] = st.r[k][j]                                # (Σ_k p_k r_kᵀ)_{ij}
        end
        Q = nullspace(C)                                                     # orthonormal basis of the zero-virial subspace
        lN(S, y) = (Ss = Q' * (S - S * C' * ((C * S * C') \ (C * S))) * Q;  Ss = (Ss + Ss') / 2;
                    -0.5 * dot(Q' * y, Ss \ (Q' * y)) - 0.5 * logdet(Ss))
        p = reduce(vcat, [collect(δ[k]) for k in 1:n])
        vir = norm(C * p) / norm(p)
        direct = lN(Σof(c′), -p) - lN(Σof(c), p)
        worst = max(worst, abs(lq - direct) / max(1, abs(direct)))
        a <= 3 && @printf("  star %d (n = %d, virial %.0e): ln q ratio %+.6f, direct %+.6f\n", a, n, vir, lq, direct)
    end
    check(worst < 1e-6, @sprintf("eq. qresid = the direct ratio (max relative difference %.1e; the direct side inverts a ~100-dimensional covariance)", worst))
end

println("4. a stochastic step with the correction")
withenv("SPH_HASTINGS" => "1") do
    p2 = deepcopy(ps);  P0 = sum(q.m for q in p2)
    opts = StochOptions(params)
    a = stochastic_step!(p2, params, cl, 0.5 * compute_dt(p2, params, eos; mode = :viscous), eos, opts;
                         rng = Xoshiro(6), rngs = make_phys_rng(params.N, 5))
    smooth_to_particles!(p2, params, cl);  update_thermodynamics!(p2, eos)
    err = maximum(abs(p2[i].s - particle_entropy(e[i], p2[i].m, V[i], eos)) / p2[i].s for i in eachindex(p2))
    dP = norm(sum(q.m for q in p2) - P0) / sum(norm(q.m) for q in p2)
    check(opts.hastings && a.stress > 0.5 && a.resid > 0.3 && dP < 1e-13 && err < 1e-10,
          @sprintf("runs: acceptances %.3f %.3f, |ΔP| %.0e, entropies S_a(e_a, m_a) to %.0e", a.stress, a.resid, dP, err))
end
println(ok ? "testhastings: PASS" : "testhastings: FAIL");  ok || exit(1)
