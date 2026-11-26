# testpositional_steps.jl — the steps with the positional counterterm (SPH_POSITIONAL=1, a₂ = SPH_POSITIONAL_A2, default 0), for the scheme
# of SPH_RENORM (sscheme, the default here, or split), c_s² = 1/2, 1/4, at s_a = 16, h/a = 2:
#   1. N3 (ii): the Jacobian determinant of two steps of the generalized leapfrog at fixed labels is one (168 particles, after the equilibrium
#      draw): the forces with ω are the derivatives of one Hamiltonian;
#   2. N3 (iii): 20 hybrid Monte Carlo updates of length 0.6/c_s (N = 672): most accepted, none leaves the domain.
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "sscheme")
ENV["SPH_POSITIONAL"] = "1"
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
println("scheme ", SSCHEME ? "sscheme ($(SS_FORM), β = $(SS_BETA))" : SPLIT ? "split" : "other", ", ω on, a₂ = ", POS_A2)

println("1. the Jacobian determinant of the ideal step")
for c2 in (0.5, 0.25)
    global eos = ConstantCs2EOS(cs2_val = c2, η_by_s = 0.2)
    h = 0.033;  S0 = 16 / (sqrt(3) / 2 * (h / 2)^2)
    params, ps, cl, _ = quiet(() -> setup_ic(h, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.2, lattice = :hex, seed = 3))
    quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(2), bath = (1 / trel_of_srel(eos, S0), SVector(0.0, 0.0))))
    N = length(ps);  x0 = [q.x for q in ps];  m0 = [q.m for q in ps]
    dt = 0.5 * compute_dt(ps, params, eos; mode = :ideal)
    msc = sqrt(sum(norm(m)^2 for m in m0) / N)
    function flow(x, m)
        setx!(ps, params, cl, x);  setm!(ps, m)
        hmc_trajectory!(ps, params, cl, eos, dt, 2; tol = 1e-14, maxit = 200)
        return vcat(reduce(vcat, [minimum_image(q.x - x0[a], params.L) for (a, q) in enumerate(ps)]), reduce(vcat, [q.m for q in ps]))
    end
    unpack(z) = ([x0[a] + SVector(z[2a-1], z[2a]) for a in 1:N], [SVector(z[2N+2a-1], z[2N+2a]) for a in 1:N])
    z = vcat(zeros(2N), reduce(vcat, m0));  J = zeros(4N, 4N)
    for k in 1:4N
        δ = k <= 2N ? 1e-6 * params.a : 1e-6 * msc;  e = zeros(4N);  e[k] = δ
        J[:, k] = (flow(unpack(z + e)...) - flow(unpack(z - e)...)) / (2δ)
    end
    ld, sg = logabsdet(J)
    check(sg > 0 && abs(ld) < 1e-6, @sprintf("c_s² = %.2f: N = %d, ln det J = %.1e", c2, N, ld))
end

println("2. hybrid Monte Carlo at s_a = 16")
for c2 in (0.5, 0.25)
    global eos = ConstantCs2EOS(cs2_val = c2, η_by_s = 0.2)
    h = 0.6 / sqrt(672 / (2 / sqrt(3))) * 2;  S0 = 16 / (sqrt(3) / 2 * (h / 2)^2)
    params, ps, cl, _ = quiet(() -> setup_ic(h, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex, seed = 5))
    r = quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1), bath = (1 / trel_of_srel(eos, S0), SVector(0.0, 0.0))))
    tg = HMCTarget(r.β0, r.β, params.vol, eos)
    dt = 0.35 * compute_dt(ps, params, eos; mode = :ideal);  ns = round(Int, 0.6 / sqrt(c2) / dt)
    nacc = 0;  nexit = 0
    for i in 1:20
        u = hmc_update!(ps, params, cl, eos, tg, Xoshiro(100i); nsteps = ns, dt = dt)
        nacc += u.accepted;  nexit += !isfinite(u.dH)
    end
    ν = [positional_omega(q.St / (1 + 1 / c2), q.V).ν for q in ps if SSCHEME]
    check(nexit == 0 && nacc >= 16, @sprintf("c_s² = %.2f: %d of 20 accepted, %d left the domain%s", c2, nacc, nexit,
                                             SSCHEME ? @sprintf("; ν_* of the particles %.3f to %.3f, smallest label %.3f", minimum(ν), maximum(ν), minimum(q.s for q in ps)) : ""))
end
ok || error("testpositional_steps: failures")
println("testpositional_steps: all passed")
