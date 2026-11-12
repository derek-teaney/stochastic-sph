# boosted_1d.jl — the one-level bond form on a MOVING 1D background
# (docs/bond_dissipation.tex).
#  (i)  the code's T κ^{xxxx}(v) (kappa_tensor) against the closed form
#       T κ0 (1−v²)² / (1−cs² v²)²,   κ0 = 2η(d−1)/d + ζ
#  (ii) the bond dissipative function of a small wave δβ = A sin(kx) on the moving
#       background, K_bond / K_cont  with  K_cont = ½ Σ V T κ(v) (∂_x δβ)²,
#       against the lattice symbol B_v(k) = Σψ(γr)2(1−cos kr) / (k² Σψ(γr) r²).
# Run:  julia --project examples/boosted_1d.jl
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))

ηs, ζs, d, cs2v = 0.1, 0.05, 3, 1/3
h, ha = 0.02, 4.0
ψ(ρ) = (q = ρ / h; q < 2 ? (1 - q / 2)^3 : 0.0)
for M0 in (0.0, 0.8, 1.15, 1.35)
    eos = ConformalEOS(cs2v, 1.0, ηs, ζs)
    params, ps, cl, eos = setup_ic(h, ha, eos, (x, y) -> 2.0, (x, y) -> SVector(M0, 0.0); Lx=1.0, sim_dim=1, d=d)
    params.dissipation = :bond
    a, N = params.a, params.N
    p1 = ps[1];  v = p1.v_over_S[1] * p1.S_smooth;  γ = 1 / sqrt(1 - v^2)
    T  = p1.T * p1.γ                                     # rest-frame temperature
    sr = srel(eos, p1.e);  κ0 = 2ηs * sr * (d - 1) / d + ζs * sr
    Tκ_code = kappa_tensor(p1, eos, Kappa3D())[1, 1, 1, 1]
    Tκ_form = T * κ0 * (1 - v^2)^2 / (1 - cs2v * v^2)^2
    b0 = [p.dotx / p.Teff for p in ps]
    lat = [j * a for j in -40:40 if j != 0]
    @printf("v=%.3f γ=%.3f  Tκxxxx: code %.6e  closed form %.6e  (rel %.1e)\n",
            v, γ, Tκ_code, Tκ_form, abs(Tκ_code / Tκ_form - 1))
    for n in (2, 10, 30, 60)
        k = 2π * n / params.L[1];  A = 1e-3 * norm(b0[1]) + 1e-6
        bv = [b0[i] + SVector(A * sin(k * ps[i].x[1]), 0.0) for i in 1:N]
        δb = [SVector(A * sin(k * ps[i].x[1]), 0.0) for i in 1:N]   # K depends only on differences of β
        K  = bond_dissipation(ps, params, cl, eos, δb)
        Kc = sum(0.5 * (p.s / p.S_smooth) * Tκ_form * (A * k * cos(k * p.x[1]))^2 for p in ps)
        Bv = sum(ψ(γ * abs(r)) * 2 * (1 - cos(k * r)) for r in lat) / (k^2 * sum(ψ(γ * abs(r)) * r^2 for r in lat))
        @printf("     ka=%.3f  K_bond/K_cont = %.5f   B_v(k) = %.5f\n", k * a, K / Kc, Bv)
    end
end
