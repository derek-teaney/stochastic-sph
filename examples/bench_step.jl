# bench_step.jl — time the parts of one coupled time step (docs/plan.tex, "Where the
# time goes").  Run before and after every performance change:
#   julia -t 1 --project examples/bench_step.jl
using StaticArrays, LinearAlgebra, Printf, Random
const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "src", "sph.jl")); include(joinpath(ROOT, "src", "load.jl"))
for ha in (2.0, 2.5)
eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
params, ps, cl, eos = setup_ic(0.05, ha, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L=0.6, lattice=:hex)
N = params.N;  rngs = make_phys_rng(N, 1);  rng = Xoshiro(2);  dotm = zeros(SVector{2,Float64}, N)
dt = 0.5 * compute_dt(ps, params, eos; mode=:viscous)
for n in 1:30   # warm up and thermalize a bit
    stochrep_step!(ps, params, cl, dt/2, eos; rng, rngs, vote=:star); leapfrog_step!(ps, params, cl, dt, eos, dotm)
end
T = Dict{String,Float64}();  add(k, t) = (T[k] = get(T, k, 0.0) + t)
nrep = 10
for n in 1:nrep
    add("leapfrog (ideal SPH)", @elapsed leapfrog_step!(ps, params, cl, dt, eos, dotm))
    add("  smooth+thermo+velocities", @elapsed (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos); update_velocities!(ps, params, cl)))
    V = [pa.s / pa.S_smooth for pa in ps]
    add("  particle energies e_a (deepcopy+solve_eos)", @elapsed (e = [particle_energy(pa, eos) for pa in ps]))
    add("  build_stars (comoving weights, 4x4 solve)", @elapsed (stars = build_stars(ps, params, cl, eos, V)))
    add("  gradient: kernel_star", @elapsed (kst = [kernel_star(a, ps, params, cl) for a in 1:N]))
    add("  gradient: C, star_fit, g (x2)", @elapsed begin
        Cs = [inverse_susceptibility(pa, eos) for pa in ps];  μ = [pa.m / pa.s for pa in ps]
        g = [gradient_energy(ps[a].s, Cs[a], star_fit(kst[a], μ)...) for a in 1:N]
        g = [gradient_energy(ps[a].s, Cs[a], star_fit(kst[a], μ)...) for a in 1:N] end)
    m = [pa.m for pa in ps];  Sc = [particle_entropy(e[a], m[a], V[a], eos) for a in 1:N]
    groups = color_groups(cl);  cells = shifted_cells(ps, cl, SVector(0.01, 0.02))
    add("  stress sweep", @elapsed color_sweep!(m, Sc, e, V, stars, groups, cells, :stress, dt/2, 1.0, false, eos, rngs, rng, :star))
    add("  residual sweep", @elapsed color_sweep!(m, Sc, e, V, stars, groups, cells, :resid, dt/2, 1.0, false, eos, rngs, rng, :star))
    add("  residual sweep, bond vote", @elapsed color_sweep!(copy(m), copy(Sc), e, V, stars, groups, cells, :resid, dt/2, 1.0, false, eos, rngs, rng, :bonds))
    add("full stochrep_step! (one half step)", @elapsed stochrep_step!(ps, params, cl, dt/2, eos; rng, rngs, vote=:star))
end
ne = 200000;  e1 = ps[1].e;  tE = @elapsed (for i in 1:ne; eos_from_energy(1e8, SVector(1e3*rand(), 0.0), eos); end)
@printf("\nh/a = %.1f, N = %d, %d threads, ms per call (avg of %d)\n", ha, N, Threads.nthreads(), nrep)
for k in sort(collect(keys(T)), by = k -> -T[k]); @printf("  %-48s %8.1f ms\n", k, 1000T[k]/nrep); end
@printf("  one eos_from_energy call: %.2f μs;  bonds per star ≈ %d\n", 1e6tE/ne, round(Int, sum(length(s.nb) for s in build_stars(ps, params, cl, eos, [pa.s / pa.S_smooth for pa in ps]))/N))
end
