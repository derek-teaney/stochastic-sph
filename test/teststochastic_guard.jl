# teststochastic_guard.jl — the superluminal guard of the stochastic step (src/stochastic.jl).
#   1  particle_entropy is −Inf for states no fluid can have, finite otherwise
#   2  a stochastic move that would make a particle superluminal is rejected; with
#      huge proposals the guard fires, momentum is conserved and every particle stays
#      physical
#   3  in normal runs the guard never fires and the results are unchanged
# Run:  julia --project test/teststochastic_guard.jl
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))

println("Test 1: particle_entropy on unphysical states")
let eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    e, V = 1.0, 0.1
    @assert isfinite(particle_entropy(e, SVector(0.5e, 0.0), V, eos))
    @assert particle_entropy(e, SVector(1.0e, 0.0), V, eos) == -Inf
    @assert particle_entropy(e, SVector(5.0e, 0.0), V, eos) == -Inf
    @assert particle_entropy(-1.0, SVector(0.0, 0.0), V, eos) == -Inf
    println("  ✓")
end

println("Test 2: huge proposals — the guard fires, momentum conserved, all states physical")
let eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    # low entropy per particle (s_a ≈ 0.2) and a fluid moving at v = 0.6: proposals reach |v| ≥ 1
    params, ps, cl, eos = setup_ic(0.05, 2.0, eos, (x, y) -> 3e2, (x, y) -> SVector(0.0, 0.0); L=0.6, lattice=:hex)
    w0 = ps[1].e + ps[1].p;  u = 0.6
    for p in ps; p.m = (p.s / p.S_smooth) * w0 / (1 - u^2) * SVector(u, 0.0); end
    thermalize!(ps, params, cl, eos)
    N = params.N;  rngs = make_phys_rng(N, 3);  rng = Xoshiro(4)
    P0 = sum(p.m for p in ps);  reset_guards!()
    dt = 50 * compute_dt(ps, params, eos; mode=:viscous)
    for n in 1:5
        stochastic_step!(ps, params, cl, dt, eos; rng, rngs, vote=:bonds)
    end
    g = guards()
    vmax = maximum(norm(eos_from_energy(particle_energy(p, eos) / (p.s / p.S_smooth),
                                        p.m / (p.s / p.S_smooth), eos)[2]) for p in ps)
    dP = norm(sum(p.m for p in ps) - P0) / norm(P0)
    @printf("  guard events: %s   max |v| = %.4f   |ΔP|/|P| = %.1e\n", g, vmax, dP)
    @assert g.rejected > 0 "guard never fired"
    @assert vmax < 1 && dP < 1e-12
    println("  ✓")
end

println("Test 3: normal run — the guard never fires")
let eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    params, ps, cl, eos = setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L=0.6, lattice=:hex)
    N = params.N;  rngs = make_phys_rng(N, 5);  rng = Xoshiro(6);  reset_guards!()
    dt = 0.5 * compute_dt(ps, params, eos; mode=:viscous);  dotm = zeros(SVector{2,Float64}, N)
    for n in 1:20
        stochastic_step!(ps, params, cl, dt / 2, eos; rng, rngs, vote=:star)
        leapfrog_step!(ps, params, cl, dt, eos, dotm)
    end
    @printf("  guard events: %s\n", guards())
    @assert all(==(0), values(guards()))
    println("  ✓")
end
println("All teststochastic_guard.jl tests passed.")
