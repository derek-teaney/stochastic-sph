# bench_legendre.jl — the cost of the Legendre Hamiltonian (docs/legendre_plan.tex, E): one step of the ideal integrator (generalized
# leapfrog) and one stochastic step at N ≈ 10⁴, s_a = 54, c_s² = 1/2, and one hybrid Monte Carlo update at N = 378 (L = 0.45, h/a = 2);
# the time in the solves (update_thermodynamics!, the forward map of every particle; the inverse map timed per call), and the
# Newton iterations per solve.  Run with and without SPH_RENORM=legendre:
#   julia -t 8 --project=nersc examples/bench_legendre.jl              (the target, tree level)
#   SPH_RENORM=legendre julia -t 8 --project=nersc examples/bench_legendre.jl
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
counts() = LEGENDRE ? legendre_counts() : (solves = 0, iters = 0, fails = 0, nostate = 0, cold = 0)
reset!() = LEGENDRE && reset_legendre_counts!()
scheme = LEGENDRE ? "legendre" : "tree"
eos = conformal_eos(2; η_by_s = 0.2)
state(L, sa) = quiet() do
    h = 0.05;  a = h / 2;  S0 = sa / (sqrt(3) / 2 * a^2)
    params, ps, cl, _ = setup_ic(h, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L, lattice = :hex, seed = 4)
    r = equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(3))
    params, ps, cl, r
end
med(f, n) = median([@elapsed(f()) for _ in 1:n])
@printf("bench_legendre: %s, %d threads\n", scheme, Threads.nthreads())

# 1. N ≈ 10⁴
params, ps, cl, r = state(2.3, 54.0)
N = params.N;  dotm = zeros(SVector{2,Float64}, N);  dt = 0.35 * compute_dt(ps, params, eos; mode = :ideal)
rngs = make_phys_rng(N, 1);  rng = Xoshiro(2);  cache = StochCache();  opts = StochOptions(params)
dts = 0.5 * compute_dt(ps, params, eos; mode = :viscous)
for _ in 1:2;  leapfrog_step!(ps, params, cl, dt, eos, dotm);  stochastic_step!(ps, params, cl, dts, eos, opts; rng, rngs, cache);  end
reset!();  t_ideal = med(() -> leapfrog_step!(ps, params, cl, dt, eos, dotm), 5);  c1 = counts()
reset!();  t_thermo = med(() -> update_thermodynamics!(ps, eos), 20);  c2 = counts()
reset!();  t_stoch = med(() -> stochastic_step!(ps, params, cl, dts, eos, opts; rng, rngs, cache), 5);  c3 = counts()
# one inverse solve (a vote), cold and with a guess
V = ps[1].s / ps[1].S_smooth;  e1 = particle_energy(ps[1], eos);  m1 = ps[1].m;  g = (ps[1].T, ps[1].v_over_S * ps[1].S_smooth)
ninv = 20000
t_inv = @elapsed (for i in 1:ninv;  vote_entropy(e1 * (1 + 1e-6i / ninv), m1, V, eos, true);  end)
t_invg = LEGENDRE ? (@elapsed (for i in 1:ninv;  particle_inverse(e1 * (1 + 1e-6i / ninv), m1, V, eos; guess = g);  end)) : NaN
@printf("N = %d, s_a = 54, c_s² = 1/2\n", N)
@printf("  ideal step (generalized leapfrog)   %8.1f ms   forward solves per step %d, %.2f iterations each\n",
        1000t_ideal, c1.solves ÷ 5, c1.iters / max(1, c1.solves))
@printf("  update_thermodynamics! (N forward)  %8.2f ms   (%.0f ns per particle, %.2f iterations)\n",
        1000t_thermo, 1e9t_thermo / N, c2.iters / max(1, c2.solves))
@printf("  stochastic step                     %8.1f ms   solves per step %d, %.2f iterations each, %d failures, %d with no state\n",
        1000t_stoch, c3.solves ÷ 5, c3.iters / max(1, c3.solves), c3.fails, c3.nostate)
@printf("  one inverse solve (vote, cold)      %8.0f ns;  with a guess %8.0f ns\n", 1e9t_inv / ninv, 1e9t_invg / ninv)

# 2. hybrid Monte Carlo, N = 378
params, ps, cl, r = state(0.45, 54.0)
tg = HMCTarget(r.β0, r.β, params.vol, eos);  dth = 0.35 * compute_dt(ps, params, eos; mode = :ideal)
nst = max(1, ceil(Int, params.L[1] / sqrt(cs2(eos)) / dth));  rng = Xoshiro(5)
hmc_update!(ps, params, cl, eos, tg, rng; nsteps = nst, dt = dth)
reset!();  us = [(t = @elapsed(u = hmc_update!(ps, params, cl, eos, tg, rng; nsteps = nst, dt = dth)); (t, u.accepted)) for _ in 1:10];  c4 = counts()
@printf("HMC, N = %d, %d steps per trajectory: %.3f s per update, acceptance %.1f, %d solves per update, %.2f iterations each\n",
        params.N, nst, median(first.(us)), mean(last.(us)), c4.solves ÷ 10, c4.iters / max(1, c4.solves))
