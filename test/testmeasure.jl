# testmeasure.jl — the integration measure, Liouville (docs/equilibrium_positions.tex, Appendix A):
#   1. the vote entropy is S_a − ln T_a, T_a = ∂e_a/∂s_a the bare temperature (finite differences); flat_weight is S_a;
#   2. the default ideal step: the generalized leapfrog in 2D with the energy Hamiltonian, else :particle;
#   4. a stochastic step runs and leaves the true entropies S_a(e_a, m_a) in the particles.
# (Test 3 compared the Einstein and the Liouville starts; the Einstein measure was removed on 2026-11-27, docs/focus_plan.tex Q2.)
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "0")   # tree level (split, the default, is tested in testsplit.jl): set before src/load.jl
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
eos = ConstantCs2EOS(1/3, 1.0, 0.2, 0.0)

println("1. vote entropy")
e, m, V = 2.0, SVector(0.3, -0.1), 0.7
S = particle_entropy(e, m, V, eos);  δ = 1e-6
T = 2δ / (particle_entropy(e + δ, m, V, eos) - particle_entropy(e - δ, m, V, eos))     # ∂e/∂s at fixed m, V
check(abs(vote_entropy(e, m, V, eos) - (S - log(T))) < 1e-8, @sprintf("S − ln T: %.10f vs %.10f", vote_entropy(e, m, V, eos), S - log(T)))
check(flat_weight(e, m, V, eos) == S, "flat_weight (the proposal of the HMC heat bath): the entropy itself")

println("2. default")
p0, _, _, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 3e4, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex))
check(p0.ideal === :generalized, "SimParams.ideal defaults to :generalized in 2D with the energy Hamiltonian (since 2026-11-15)")
p1 = withenv("SPH_HAMILTONIAN" => "smoothed") do
    quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 3e4, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex))[1]
end
check(p1.ideal === :particle, "with the smoothed Hamiltonian: :particle")

println("4. a stochastic step")
let
    params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 3e4, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex))
    quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(4)))
    rngs = make_phys_rng(params.N, 5)
    dt = 0.5 * compute_dt(ps, params, eos; mode = :viscous)
    a = stochastic_step!(ps, params, cl, dt, eos, StochOptions(params); rng = Xoshiro(6), rngs)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    V = [q.s / q.S_smooth for q in ps];  e = [particle_energy(q, eos) for q in ps]
    err = maximum(abs(ps[i].s - particle_entropy(e[i], ps[i].m, V[i], eos)) / ps[i].s for i in eachindex(ps))
    check(a.stress > 0.5 && a.resid > 0.3, @sprintf("acceptances %.3f, %.3f", a.stress, a.resid))
    check(err < 1e-10, @sprintf("particle entropies are S_a(e_a, m_a): max relative difference %.1e", err))
end

println(ok ? "testmeasure: PASS" : "testmeasure: FAIL");  ok || exit(1)
