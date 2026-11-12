# multistep_dS.jl — rms sizes of the entropy changes in one star or residual move versus h/a,
# at fixed S0 and Δt, frozen lattice, equilibrium start: ΔS_a (the centre), one ΔS_b (a
# neighbour), Σ_b ΔS_b, and the total.  No accept/reject: proposals only.
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const h = 0.05;  const S0 = 1e6;  const DT = 8.66e-3
eos = ConformalEOS(1/3, 1.0, 0.2, 0.0);  rng = Xoshiro(5)
println(" h/a  bonds | move    rms ΔS_a    rms ΔS_b    rms ΣΔS_b   rms total   √N·rms ΔS_b")
for ha in (2.0, 3.0, 4.0, 6.0)
    params, ps, cl, _ = setup_ic(h, ha, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 6.2h, lattice = :hex)
    initial_state!(ps, params, cl, eos, :eq; rng = Xoshiro(3))
    fz = stochastic_frozen(ps, params, cl, eos);  V, e, stars = fz.V, fz.e, fz.stars
    m = [p.m for p in ps];  Sc = [particle_entropy(e[i], m[i], V[i], eos) for i in eachindex(ps)]
    for kind in (:star, :resid)
        A = Float64[];  B = Float64[];  ΣB = Float64[];  T = Float64[]
        for rep in 1:20, a in eachindex(ps)
            st = stars[a];  n = length(st.nb);  δ = Vector{SVector{2,Float64}}(undef, n)
            kind === :star ? propose_stress!(δ, st, DT, false, rng) : propose_resid!(δ, st, DT, 1.0, false, rng)
            dSa = particle_entropy(e[a], m[a] - sum(δ), V[a], eos) - Sc[a]
            dSb = [particle_entropy(e[st.nb[k]], m[st.nb[k]] + δ[k], V[st.nb[k]], eos) - Sc[st.nb[k]] for k in 1:n]
            push!(A, dSa);  append!(B, dSb);  push!(ΣB, sum(dSb));  push!(T, dSa + sum(dSb))
        end
        nb = length(stars[1].nb);  r(x) = sqrt(mean(abs2, x))
        @printf("%4.0f %5d  | %-5s  %.2e    %.2e    %.2e    %.2e    %.2e\n", ha, nb, kind, r(A), r(B), r(ΣB), r(T), sqrt(nb) * r(B))
    end
end
