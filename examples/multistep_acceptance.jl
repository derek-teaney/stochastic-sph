# multistep_acceptance.jl — acceptance of the star and residual moves versus h/a at fixed
# S0 (so fixed S0·πh²) and fixed Δt, on a frozen hexagonal lattice in a box of 6h, from the
# equilibrium start.  Votes: :star (one vote per star, the default) and :bonds (factorized).
# Run:  julia -t 8 --project examples/multistep_acceptance.jl
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure"); include(joinpath(@__DIR__, "..", "src", "$f.jl")); end
const h = 0.05;  const S0 = 1e6;  const DT = 8.66e-3;  const NST = 10
eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
@printf("S0 = %.0e (S0·πh² = %.0f), Δt = %.2e, box 6h, frozen lattice, λ = 1\n", S0, S0 * π * h^2, DT)
println(" h/a     N  bonds   s_a  |  vote   acc(star)  acc(resid)")
for ha in (2.0, 3.0, 4.0, 6.0)
    params, ps, cl, _ = setup_ic(h, ha, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 6.2h, lattice = :hex)
    N = params.N
    initial_state!(ps, params, cl, eos, :eq; rng = Xoshiro(3))
    fz = stochrep_frozen(ps, params, cl, eos);  nb = length(fz.stars[1].nb)
    for vote in (:star, :bonds)
        rngs = make_phys_rng(N, 11);  rng = Xoshiro(12);  as = Float64[];  ar = Float64[]
        for n in 1:NST
            acc = stochrep_step!(ps, params, cl, DT, eos; rng, rngs, frozen = fz, vote, flip = isodd(n))
            push!(as, acc.stress);  push!(ar, acc.resid)
        end
        @printf("%4.0f %5d %5d %6.1f  |  %-6s  %.3f      %.3f\n", ha, N, nb, ps[1].s, vote, mean(as), mean(ar))
    end
end
