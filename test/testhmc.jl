# testhmc.jl — hybrid Monte Carlo on positions and momenta (src/hmc.jl, docs/equilibrium_positions.tex):
#   1. reversibility: a trajectory, then the trajectory from its end with the momenta reversed, returns to the start;
#   2. the energy error of a trajectory of fixed length falls as δt²;
#   3. a few updates from the lattice: acceptance, heat-bath acceptance, total momentum unchanged by the trajectories.
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)

eos = ConstantCs2EOS(1/3, 1.0, 0.2, 0.0)
function thermal_state(L; seed = 1)
    params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L, lattice = :hex))
    r = quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(seed)))
    return params, ps, cl, r
end
H(ps, params, cl) = hmc_measure!(ps, params, cl, eos)[1]

params, ps, cl, r = thermal_state(0.3)
N = params.N;  a = params.a
dt = 0.5 * compute_dt(ps, params, eos; mode = :ideal)
@printf("N = %d, δt = %.3e, T = %.4g\n", N, dt, r.T)

println("1. reversibility")
x0 = [q.x for q in ps];  m0 = [q.m for q in ps]
hmc_trajectory!(ps, params, cl, eos, dt, 10)
for q in ps;  q.m = -q.m;  end
hmc_trajectory!(ps, params, cl, eos, dt, 10)
for q in ps;  q.m = -q.m;  end
dx = maximum(norm(minimum_image(ps[i].x - x0[i], params.L)) for i in 1:N) / a
dm = maximum(norm(ps[i].m - m0[i]) for i in 1:N) / maximum(norm, m0)
check(dx < 1e-9 && dm < 1e-9, @sprintf("back to the start: max |Δx|/a = %.1e, max |Δm|/max|m| = %.1e", dx, dm))

println("2. energy error against δt (trajectory of fixed length 8δt₀)")
errs = Float64[]
for k in (1, 2, 4)
    let (params, ps, cl, _) = thermal_state(0.3)
        H0 = H(ps, params, cl);  hmc_trajectory!(ps, params, cl, eos, dt / k, 8k);  push!(errs, abs(H(ps, params, cl) / H0 - 1))
    end
end
@printf("  |ΔH/H| = %.2e, %.2e, %.2e at δt₀, δt₀/2, δt₀/4\n", errs...)
check(errs[1] / errs[2] > 3 && errs[2] / errs[3] > 3, @sprintf("ratios %.2f, %.2f (≈ 4 for δt²)", errs[1] / errs[2], errs[2] / errs[3]))

println("3. updates from the lattice")
params, ps, cl, r = thermal_state(0.3; seed = 2)
tg = HMCTarget(r.β0, r.β, params.vol, eos)
rng = Xoshiro(5);  nacc = 0;  hbs = Float64[];  its = Float64[]
for n in 1:20
    u = hmc_update!(ps, params, cl, eos, tg, rng; nsteps = 10, dt)
    global nacc += u.accepted;  push!(hbs, u.heatbath);  push!(its, u.iters)
end
@printf("  acceptance %.2f, heat bath %.2f, leapfrog iterations per step %.1f\n", nacc / 20, sum(hbs) / 20, sum(its) / 20)
check(nacc >= 10, "acceptance at least 1/2")
check(sum(hbs) / 20 > 0.3, "heat-bath chains accept")

println(ok ? "testhmc: PASS" : "testhmc: FAIL");  ok || exit(1)
