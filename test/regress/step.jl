# step.jl — fixed-seed regression (test/run.sh regress): the full coupled
# step (equilibrium start, leapfrog + stochastic step), on a small box.
# Saves or compares the final particle states.  test/run.sh runs this file on both sides (copied
# into the reference worktree, with the reference's library), so it uses only what both have.
# The λ = ∞ case was removed with the deterministic residual (docs/focus_plan.tex, Q1).
#
#   julia -t <n> --project test/regress/step.jl save    <file.jls>
#   julia -t <n> --project test/regress/step.jl compare <file.jls> [tol]
#
# `compare` passes if every x, m, s agrees with the saved state to the relative
# tolerance tol (default 0: bit-identical).  Results must not depend on the number of
# threads, so compare across -t 1 and -t 4 as well as across code versions.
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
# The n-scheme (energy Hamiltonian, number volume: the default of the reference, the only scheme since docs/focus_plan.tex Q7).
# The measure is the Liouville measure, the only one since 2026-11-27 (docs/focus_plan.tex, Q2; the reference has it as the default of
# the energy Hamiltonian).  The smoothed Hamiltonian, which had the Einstein measure, is no longer run here.
const ROOT = joinpath(@__DIR__, "..", "..")
include(joinpath(ROOT, "src", "load.jl"))

function run_case(; L = 0.6, nsteps = 12, cache = get(ENV, "SR_NOCACHE", "0") == "1" ? nothing : StochCache())
    eos = conformal_eos(default_phys_dim(); η_by_s = 0.2)
    params, ps, cl, eos = setup_ic(0.05, 2.0, eos, (x, y) -> 1e5, (x, y) -> SVector(0.0, 0.0); L, lattice=:hex)
    N = params.N;  rngs = make_phys_rng(N, 11);  rng = Xoshiro(12);  dotm = zeros(SVector{2,Float64}, N)
    # the equilibrium draw at the requested state, totals exact (the thermal start was removed, docs/focus_plan.tex Q9)
    equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(13), exact_totals = true, bath = (1 / trel_of_srel(eos, 1e5), SVector(0.0, 0.0)))
    dt = 0.5 * compute_dt(ps, params, eos; mode=:viscous)
    reset_guards!()
    acc = Float64[]
    stochastic_step!(ps, params, cl, dt / 2, eos; rng, rngs, cache)
    for n in 1:nsteps
        leapfrog_step!(ps, params, cl, dt, eos, dotm)
        a = stochastic_step!(ps, params, cl, n == nsteps ? dt / 2 : dt, eos; rng, rngs, flip=isodd(n), cache)
        push!(acc, a.stress);  push!(acc, a.resid)
    end
    return (x = [p.x for p in ps], m = [p.m for p in ps], s = [p.s for p in ps], acc = acc, guards = guards())
end

reldiff(a, b) = maximum(norm(a[i] - b[i]) / max(norm(a[i]), norm(b[i]), 1e-300) for i in eachindex(a))

mode = ARGS[1];  file = ARGS[2];  tol = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.0
t0 = time()
res = run_case()
@printf("%d threads, %.1f s\n", Threads.nthreads(), time() - t0)
if mode == "save"
    serialize(file, res);  println("saved ", file)
else
    a, b = res, deserialize(file)
    dx = reldiff(a.x, b.x);  dm = reldiff(a.m, b.m);  ds = reldiff(a.s, b.s);  da = reldiff(a.acc, b.acc)
    ga = a.guards.rejected == b.guards.rejected                     # the guards both versions count
    ok = max(dx, dm, ds, da) <= tol && ga
    @printf("max rel diff: x %.1e  m %.1e  s %.1e  acc %.1e  rejected %s\n", dx, dm, ds, da, ga ? "same" : "DIFFER")
    println(ok ? "PASS" : "FAIL");  ok || exit(1)
end
