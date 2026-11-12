# multistep_kick.jl — decay of a single-particle momentum excess on a frozen lattice, single
# vote vs bond vote, versus h/a at fixed S0 and Δt.  Each seed is run twice on identical random
# numbers, with and without the kick (1.5× the thermal momentum along x), and the difference is
# the response.  Prediction: P(t) = (1/N) Σ_k exp(−acc Γ_xx(k) t), Γ_xx from Kmode with ê = x̂.
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "stochrep_thermalize_common.jl"))          # Kmode
const h = 0.05;  const S0 = 1e6;  const DT = 8.66e-3;  const NST = 16;  const NSEED = 24
eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
println(" h/a  vote    acc_r  |  P(t) at steps 2, 4, 8, 16: measured / predicted")
for ha in (2.0, 4.0, 6.0)
    params, ps0, cl, _ = setup_ic(h, ha, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 6.2h, lattice = :hex)
    N = params.N;  a0 = 1;  Lx, Ly = params.L
    for vote in (:star, :bonds)
        P = zeros(NST + 1);  ar = Float64[];  as = Float64[];  st0 = nothing;  w0 = 0.0
        for seed in 1:NSEED
            runs = map((0.0, 1.0)) do kick
                ps = deepcopy(ps0)
                initial_state!(ps, params, cl, eos, :eq; rng = Xoshiro(100seed))
                fz = stochrep_frozen(ps, params, cl, eos);  st = fz.stars[a0];  w = ps[a0].e + ps[a0].p
                st0 = st;  w0 = w
                Δm = 1.5 * sqrt(st.V * w * st.T);  ps[a0].m += SVector(kick * Δm, 0.0)
                rngs = make_phys_rng(N, 7seed);  rng = Xoshiro(8seed);  tr = [ps[a0].m[1]]
                for n in 1:NST
                    acc = stochrep_step!(ps, params, cl, DT, eos; rng, rngs, frozen = fz, vote, flip = isodd(n))
                    kick > 0 && (push!(ar, acc.resid); push!(as, acc.stress));  push!(tr, ps[a0].m[1])
                end
                (tr, Δm)
            end
            P .+= (runs[2][1] .- runs[1][1]) ./ runs[2][2]
        end
        P ./= NSEED
        # Green's function prediction on the box k-grid (hex lattice in a rectangular periodic box)
        Nx = round(Int, Lx / params.a);  Ny = N ÷ Nx;  ê = SVector(1.0, 0.0)
        Γk = Float64[];  Γrk = Float64[]
        for i in 0:Nx-1, j in 0:Ny-1
            k = SVector(2π * i / Lx, 2π * j / Ly);  K, Kr = Kmode(st0, k, ê, false)
            push!(Γk, 2K / (st0.V * w0 * st0.T));  push!(Γrk, 2Kr / (st0.V * w0 * st0.T))
        end
        Ppred(t) = mean(exp.(-(mean(as) .* Γrk .+ mean(ar) .* (Γk .- Γrk)) .* t))
        @printf("%4.0f  %-6s  %.3f  | ", ha, vote, mean(ar))
        for n in (2, 4, 8, 16); @printf(" %.3f/%.3f ", P[n+1], Ppred(n * DT)); end
        println()
    end
end
