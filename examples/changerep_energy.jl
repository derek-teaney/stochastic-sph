# changerep_energy.jl — energy bookkeeping of the exact viscous step in the particle
# representation (viscous_step_exact!, docs/changerep.tex).
#
# A momentum wave of wavenumber k on a frozen 1D lattice at rest is damped to (nearly)
# zero by repeated steps.  Each step holds the particle energies e_a fixed, so Σe_a is
# conserved, while the smoothed energy H grows.  Prediction:
#   Σe − H = (1 − R_k²) KE ,   KE = Σ_a |m_a|²/(2 w V)   (initially, quadratic order),
# and once the wave has decayed, H_end − H_0 = Σe − H_0.
#
# Run:  julia --project examples/changerep_energy.jl
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const KER = WendlandC2_1D()
W(r, h)  = wendland_c2(abs(r), h, KER)
lat(h, a) = [m * a for m in -ceil(Int, 2h / a):ceil(Int, 2h / a)]
Rk(k, h, a) = sum(a * W(r, h) * cos(k * r) for r in lat(h, a))

ηs  = 0.1
eos = ConformalEOS(1/3, 1.0, ηs, 0.0)
h, ha = 0.02, 4.0
params, ps, cl, eos = setup_ic(h, ha, eos, (x, y) -> 2.0, (x, y) -> SVector(0.0, 0.0);
                               Lx=1.0, sim_dim=1)
params.dissipation = :bond
a, N, L = params.a, params.N, params.L[1]
w  = ps[1].e + ps[1].p
s0 = [p.s for p in ps]
dt = 0.5 * compute_dt(ps, params, eos; mode=:viscous)     # all factors 1 − ΓΔt > 0
Hof(ps) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
           sum(p.s * p.H_over_S for p in ps))

println("  ka     R²      (1−R²)KE     Σe−H₀       H_end−H₀    |ΔΣe|/Σe   steps  ΣΔs·T/KE   2KΔt−ΔS (1st step)")
for (n, A_rel) in ((5, 1e-2), (20, 1e-2), (40, 1e-2), (70, 1e-2), (100, 1e-2), (40, 3e-2))
    k = 2π * n / L
    for (i, p) in enumerate(ps)
        p.s = s0[i]
        p.m = SVector(A_rel * w * a * cos(k * (p.x[1] - a / 2)), 0.0)
    end
    H0  = Hof(ps)
    V   = [p.s / p.S_smooth for p in ps]
    KE  = sum(dot(p.m, p.m) / (2w * V[i]) for (i, p) in enumerate(ps))
    e0  = sum(particle_energy(p, eos) for p in ps)
    T0  = ps[1].T
    amp(ps) = sqrt(sum(dot(p.m, p.m) for p in ps))
    A0  = amp(ps)
    nst = 0;  ΣS = 0.0;  Σe = e0;  first_def = NaN
    while amp(ps) > 1e-4 * A0 && nst < 200_000
        K, dS, Σe = viscous_step_exact!(ps, params, cl, dt, eos)
        nst == 0 && (first_def = (2K * dt - dS) / (2K * dt))
        ΣS += dS;  nst += 1
    end
    Hend = Hof(ps)
    @printf("%6.3f  %.4f  %.4e  %.4e  %.4e  %.1e  %6d  %.4f   %.2e\n",
            k * a, Rk(k, h, a)^2, (1 - Rk(k, h, a)^2) * KE, e0 - H0, Hend - H0,
            abs(Σe - e0) / e0, nst, ΣS * T0 / KE, first_def)
end
