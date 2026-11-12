# damping_rate.jl — viscous damping rate of a shear wave vs wavenumber.
#
# Physics: a transverse wave  m_y ∝ cos(k x)  decays at  Γ_phys(k) = (η/w) k²  —
# the shortest waves die fastest.
#
# Scheme: put that pattern (tiny amplitude) on the frozen hex lattice, apply the
# deterministic two-level dissipative_force! (= the mean drift of the stochastic
# sweep, testparticle_sweep.jl (vi)), and read off
#
#     Γ_SPH(k) = −Σ_a f_a·m_a / Σ_a m_a·m_a .
#
# Run:  julia --project examples/damping_rate.jl   → damping_rate.png

using PyPlot
using StaticArrays
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))

ηs  = 10.0 / sqrt(2.0)
eos = ConformalEOS(0.5, T_scale_2d_lq1, ηs, 0.0)
params, particles, cl, eos = setup_ic(15.0, 2.55, eos, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                      L=150.0, lattice=:hex, d=2)
h, a, V, L, N = params.h, params.a, params.vol, params.L, params.N
w  = particles[1].e + particles[1].p
η  = ηs * srel(eos, particles[1].e)
Γphys(k) = (η / w) * k^2

# smoothing response R(k) along x (as in smoothing_response.jl)
p1 = particles[1]
R(k) = V * sum(wendland_c2(norm(minimum_image(pb.x - p1.x, L)), h, params.kernel) *
               cos(k * minimum_image(pb.x - p1.x, L)[1])
               for pb in particles if norm(minimum_image(pb.x - p1.x, L)) < 2h)

A    = 1e-4 * w * V           # momentum amplitude per particle (v ~ 1e-4)
dotm = zeros(SVector{2,Float64}, N)
ns   = 1:17                   # k = 2π n / Lx, up to the zone corner 4π/(3a) ≈ 0.71
ks   = [2π * n / L[1] for n in ns]
Γ    = Float64[]
for k in ks
    for p in particles; p.m = SVector(0.0, A * cos(k * p.x[1])); end
    thermalize!(particles, params, cl, eos)
    fill!(dotm, zero(eltype(dotm)))
    dissipative_force!(dotm, particles, params, cl, eos)
    push!(Γ, -sum(dot(f, p.m) for (f, p) in zip(dotm, particles)) / sum(dot(p.m, p.m) for p in particles))
end

println("   k      R(k)     Γ_SPH/Γ_phys    R(k)^4")
for (k, g) in zip(ks, Γ)
    @printf("  %.3f   %7.4f   %10.3e   %10.3e\n", k, R(k), g / Γphys(k), R(k)^4)
end

fig, (ax1, ax2) = subplots(1, 2, figsize=(12, 4.8))
kf = range(0, maximum(ks), length=300)
ax1.plot(kf, Γphys.(kf), "k--", lw=1.2, label="physics  (η/w) k²")
ax1.plot(ks, Γ, "C3o-", lw=1.5, ms=4, label="SPH (two-level force)")
ax1.set_xlabel("k"); ax1.set_ylabel("damping rate Γ(k)")
ax1.set_title("shear-wave damping rate"); ax1.legend()
for ax in (ax1, ax2)
    ax.axvline(1 / h, color="0.5", ls=":", lw=1)
    ax.axvline(π / a, color="0.5", ls="--", lw=1)
end
ax1.text(1 / h, ax1.get_ylim()[2] * 0.95, " 1/h", color="0.4")
ax1.text(π / a, ax1.get_ylim()[2] * 0.95, " π/a", color="0.4")

ax2.semilogy(ks, Γ ./ Γphys.(ks), "C3o-", ms=4, label="Γ_SPH / Γ_phys")
ax2.semilogy(kf, max.(R.(kf), 1e-12) .^ 4, "C0-", lw=1, label="R(k)⁴")
ax2.semilogy(kf, max.(R.(kf), 1e-12) .^ 2, "C0:", lw=1, label="R(k)²")
ax2.set_ylim(1e-8, 2); ax2.set_xlabel("k"); ax2.set_ylabel("ratio")
ax2.set_title("suppression relative to physics"); ax2.legend()

tight_layout()
out = joinpath(@__DIR__, "damping_rate.png")
savefig(out, dpi=120)
println("Saved ", out)
