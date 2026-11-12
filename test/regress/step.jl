# step.jl — fixed-seed regression (test/run.sh regress): the full coupled
# step (thermal start, leapfrog + stochastic step, gradient-corrected map) for λ = 1 and
# λ = ∞, on a small box.  Saves or compares the final particle states.
#
#   julia -t <n> --project test/regress/step.jl save    <file.jls>
#   julia -t <n> --project test/regress/step.jl compare <file.jls> [tol]
#
# `compare` passes if every x, m, s agrees with the saved state to the relative
# tolerance tol (default 0: bit-identical).  Results must not depend on the number of
# threads, so compare across -t 1 and -t 4 as well as across code versions.
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
# The reference path is the smoothed Hamiltonian with the gradient-corrected map (the default
# until 2026-11-10); set SPH_HAMILTONIAN=energy to check the n-scheme instead.
ENV["SPH_HAMILTONIAN"] = get(ENV, "SPH_HAMILTONIAN", "smoothed")
const ROOT = joinpath(@__DIR__, "..", "..")
include(joinpath(ROOT, "src", "load.jl"))

function run_case(λ; L = 0.6, nsteps = 12, cache = get(ENV, "SR_NOCACHE", "0") == "1" ? nothing : StochCache())
    eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    params, ps, cl, eos = setup_ic(0.05, 2.0, eos, (x, y) -> 1e5, (x, y) -> SVector(0.0, 0.0); L, lattice=:hex)
    N = params.N;  rngs = make_phys_rng(N, 11);  rng = Xoshiro(12);  dotm = zeros(SVector{2,Float64}, N)
    thermal_init!(ps, params, cl, eos; rng = Xoshiro(13))
    dt = 0.5 * compute_dt(ps, params, eos; mode=:viscous)
    reset_guards!()
    acc = Float64[]
    stochastic_step!(ps, params, cl, dt / 2, eos; λ, rng, rngs, vote=:star, cache)
    for n in 1:nsteps
        leapfrog_step!(ps, params, cl, dt, eos, dotm)
        a = stochastic_step!(ps, params, cl, n == nsteps ? dt / 2 : dt, eos; λ, rng, rngs, vote=:star, flip=isodd(n), cache)
        push!(acc, a.stress);  isinf(λ) || push!(acc, a.resid)
    end
    return (x = [p.x for p in ps], m = [p.m for p in ps], s = [p.s for p in ps], acc = acc, guards = guards())
end

reldiff(a, b) = maximum(norm(a[i] - b[i]) / max(norm(a[i]), norm(b[i]), 1e-300) for i in eachindex(a))

mode = ARGS[1];  file = ARGS[2];  tol = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.0
t0 = time()
res = Dict(λ => run_case(λ) for λ in (1.0, Inf))
@printf("%d threads, %.1f s\n", Threads.nthreads(), time() - t0)
if mode == "save"
    serialize(file, res);  println("saved ", file)
else
    ref = deserialize(file);  global ok = true
    for λ in (1.0, Inf)
        a, b = res[λ], ref[λ]
        dx = reldiff(a.x, b.x);  dm = reldiff(a.m, b.m);  ds = reldiff(a.s, b.s);  da = reldiff(a.acc, b.acc)
        pass = max(dx, dm, ds, da) <= tol && a.guards == b.guards
        global ok &= pass
        @printf("λ = %-4s  max rel diff: x %.1e  m %.1e  s %.1e  acc %.1e  guards %s  %s\n",
                string(λ), dx, dm, ds, da, a.guards == b.guards ? "same" : "DIFFER", pass ? "PASS" : "FAIL")
    end
    println(ok ? "PASS" : "FAIL");  ok || exit(1)
end
