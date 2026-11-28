# teststart.jl — the start of a run (docs/focus_plan.tex, Q9; src/measure.jl target_state, initial_state!;
# src/hmc.jl hmc_equilibrate!):
#   1. target_state: T_rel,0 and w0 from the equation of state at s_rel,0, S = γ s_rel,0, M0 = w0 γ² v, β⁰ = γ/T_rel,0, β = γ v/T_rel,0;
#   2. cold: the lattice as set up, the totals unchanged;
#   3. hot: the draw at the target's β^μ, the totals equal to the cold ones to rounding;
#   4. the removed starts eq and thermal stop.
using StaticArrays, LinearAlgebra, Printf, Random
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "0")   # tree level (split, the default, is tested in testsplit.jl): set before src/load.jl
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
eos = conformal_eos(2; η_by_s = 0.2)
const SREL0 = 1e6                                                     # s_a ≈ 540 at h/a = 2
setup(tgt; seed = 4) = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> tgt.S, (x, y) -> tgt.M0; L = 0.6, lattice = :hex, seed))
totals(ps) = (E = sum(particle_energy(p, eos) for p in ps), P = sum(p.m for p in ps))
reldiff(a, b) = max(abs(a.E / b.E - 1), norm(a.P - b.P) / abs(b.E))

println("1. target_state")
for v in (0.0, 0.6)
    t = target_state(SREL0, v, eos);  γ = 1 / sqrt(1 - v^2)
    e0 = e_of_srel(eos, SREL0);  Tr = trel_of_srel(eos, SREL0)
    good = t.Trel0 == Tr && t.w0 == e0 + pressure(eos, e0) && t.S == γ * SREL0 && t.M0 == SVector(t.w0 * γ^2 * v, 0.0) &&
           t.β0 == γ / Tr && t.β == SVector(γ * v / Tr, 0.0) && t.bath == (t.β0, t.β) && abs(srel(eos, e0) / SREL0 - 1) < 1e-12
    check(good, @sprintf("v = %.1f: T_rel,0 = %.6g, w0, S, M0, β⁰, β from the equation of state", v, Tr))
end

for v in (0.0, 0.6)
    tgt = target_state(SREL0, v, eos)
    println("v = $v")
    params, ps, cl, _ = setup(tgt);  cold = cold_totals(ps, params, cl, eos)
    println("2. cold")
    quiet(() -> initial_state!(ps, params, cl, eos, :cold))
    check(reldiff(totals(ps), cold) == 0, "the cold start keeps the lattice and its totals")
    println("3. hot")
    params, ps, cl, _ = setup(tgt)
    r = quiet(() -> initial_state!(ps, params, cl, eos, :hot; rng = Xoshiro(3), bath = tgt.bath, totals = cold))
    d = reldiff(totals(ps), cold)
    check(d < 1e-13, @sprintf("hot: totals = the cold totals to %.1e (draw acceptance %.2f)", d, r.mh_acceptance))
end

println("4. the removed starts stop")
let tgt = target_state(SREL0, 0.0, eos), (params, ps, cl, _) = setup(tgt)
    stops(s) = try initial_state!(ps, params, cl, eos, s; bath = tgt.bath); false catch err; occursin("removed", sprint(showerror, err)) end
    check(stops(:eq) && stops(:thermal), "SR_START=eq and thermal stop with a message")
end

println(ok ? "teststart: PASS" : "teststart: FAIL");  ok || exit(1)
