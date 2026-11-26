# testsscheme_jacobian.jl — N3 (ii) of docs/sscheme_plan.tex, "Level 1, amended": the ideal step (the generalized leapfrog, src/hmc.jl) at
# fixed labels preserves the volume of (x, m) space.  The Jacobian of a few steps, by central differences in every coordinate, has
# determinant one.  The step is symplectic only if v_a and the force (from P_a) are the derivatives of one Hamiltonian H(s, m, V), so this
# tests the state of the particle in the steps, whatever the form (SPH_SSCHEME_FORM, default measure).  A small system at s_a = 16 after
# the equilibrium draw (thermal momenta), c_s² = 1/2 and 1/4.
ENV["SPH_RENORM"] = "sscheme"
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)

println("N3 (ii): the Jacobian determinant of the ideal step at fixed labels (form $(SS_FORM))")
for c2 in (0.5, 0.25)
    global eos = ConstantCs2EOS(cs2_val = c2, η_by_s = 0.2)
    h = 0.033;  S0 = 16 / (sqrt(3) / 2 * (h / 2)^2)
    params, ps, cl, _ = quiet(() -> setup_ic(h, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.2, lattice = :hex, seed = 3))
    quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(2), bath = (1 / trel_of_srel(eos, S0), SVector(0.0, 0.0))))
    N = length(ps);  x0 = [q.x for q in ps];  m0 = [q.m for q in ps]
    dt = 0.5 * compute_dt(ps, params, eos; mode = :ideal);  nsteps = 2
    msc = sqrt(sum(norm(m)^2 for m in m0) / N)
    function flow(x, m)
        setx!(ps, params, cl, x);  setm!(ps, m)
        hmc_trajectory!(ps, params, cl, eos, dt, nsteps; tol = 1e-14, maxit = 200)
        xs = [minimum_image(q.x - x0[a], params.L) for (a, q) in enumerate(ps)]     # displacement from x0, no wrap jumps
        return vcat(reduce(vcat, xs), reduce(vcat, [q.m for q in ps]))
    end
    z = vcat(reduce(vcat, [zero(x) for x in x0]), reduce(vcat, m0))
    zf = flow(x0, m0)
    J = zeros(4N, 4N)
    for k in 1:4N
        δ = k <= 2N ? 1e-6 * params.a : 1e-6 * msc
        e = zeros(4N);  e[k] = δ
        unpack(zz) = ([x0[a] + SVector(zz[2a-1], zz[2a]) for a in 1:N], [SVector(zz[2N+2a-1], zz[2N+2a]) for a in 1:N])
        J[:, k] = (flow(unpack(z + e)...) - flow(unpack(z - e)...)) / (2δ)
    end
    ld, sg = logabsdet(J)
    move = maximum(abs, zf[1:2N]) / params.a
    check(sg > 0 && abs(ld) < 1e-6, @sprintf("c_s² = %.2f: N = %d, %d steps of δt = %.2e (largest displacement %.2f a), ‖J − 1‖ = %.2f: ln det J = %.1e, smallest label %.3f",
                                             c2, N, nsteps, dt, move, opnorm(J - I), ld, minimum(q.s for q in ps)))
end
ok || error("testsscheme_jacobian: failures")
println("testsscheme_jacobian: all passed")
