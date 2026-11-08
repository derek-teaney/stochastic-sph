# boosted_damping.jl — damping of a small shear wave on a MOVING background.
#
# Frozen hex lattice, uniform background momentum M0 (|v| ≈ 0.5) either along the
# wave vector (k ∥ v) or across it (k ⊥ v), plus a small transverse wave
# δm ∝ cos(k·x).  On a moving background δm and δb are not parallel, so the rate is
# the Rayleigh quotient  Γ(k) = −Σ δb·f / Σ δm·δb  (numerator = 2K(δb) ≥ 0, denominator
# > 0 by convexity), for the two-level and the comoving bond dissipative function.
#
# Run:  julia --project examples/boosted_damping.jl   → boosted_damping.png

using PyPlot
using StaticArrays
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

ηs  = 1.0
# Only δm ⊥ v: for k ⊥ v a transverse wave has δm ∥ v, which changes the energy
# density (hence T) at first order, so a pure-momentum perturbation is not a mode and
# Σ δm·δb has no fixed sign.  The covariance of K itself is tested exactly in
# test/testdissipative.jl (Test 6).
cases = (("v = 0", SVector(0.0, 0.0)),
         ("|v| = 0.3, k ∥ v", SVector(0.25, 0.0)),
         ("|v| = 0.5, k ∥ v", SVector(0.46, 0.0)))

function rates(M0, scheme)
    eos = ConformalEOS(0.5, T_scale_2d_lq1, ηs, 0.0)
    params, ps, cl, eos = setup_ic(15.0, 2.55, eos, (x, y) -> 1.0, (x, y) -> M0; L=150.0, lattice=:hex, d=2)
    params.dissipation = scheme
    N  = params.N;  vol = params.vol
    m0 = [p.m for p in ps]
    b0 = [p.dotx / p.Teff for p in ps]
    v  = norm(ps[1].v_over_S * ps[1].S_smooth)
    ks = [2π * n / params.L[1] for n in 1:17]
    f  = zeros(SVector{2,Float64}, N)
    Γ  = Float64[]
    for k in ks
        A = 1e-4 * vol
        for (i, p) in enumerate(ps); p.m = m0[i] + SVector(0.0, A * cos(k * p.x[1])); end
        thermalize!(ps, params, cl, eos)
        dissipative_force!(f, ps, params, cl, eos)
        δm = [p.m - m0[i] for (i, p) in enumerate(ps)]
        δb = [p.dotx / p.Teff - b0[i] for (i, p) in enumerate(ps)]
        push!(Γ, -sum(dot(f[i], δb[i]) for i in 1:N) / sum(dot(δm[i], δb[i]) for i in 1:N))
    end
    return v, ks, Γ
end

fig, (ax1, ax2) = subplots(1, 2, figsize=(12, 4.8))
for (j, (lab, M0)) in enumerate(cases)
    v, ks, Γ2 = rates(M0, :two_level)
    _, _,  Γb = rates(M0, :bond)
    @printf("%-20s |v|=%.3f   Γ(k_min): two-level %.4e  bond %.4e   Γ(k=0.6): two-level %.2e  bond %.2e\n",
            lab, v, Γ2[1], Γb[1], Γ2[14], Γb[14])
    ax1.semilogy(ks, Γ2, "--", color="C$(j-1)", lw=1.3, label="two-level, " * lab)
    ax1.semilogy(ks, Γb, "-",  color="C$(j-1)", lw=1.5, label="bond, " * lab)
    ax2.plot(ks[1:6], Γb[1:6] ./ Γ2[1:6], "o-", color="C$(j-1)", label=lab)
end
ax1.set_xlabel("k"); ax1.set_ylabel("Γ(k)"); ax1.set_ylim(1e-12, 1)
ax1.set_title("shear-wave damping on a moving background"); ax1.legend(fontsize=7)
ax2.set_xlabel("k"); ax2.set_ylabel("Γ_bond / Γ_two-level")
ax2.set_title("long wavelengths: same κ(v), different R-factors"); ax2.legend(fontsize=8)
tight_layout()
out = joinpath(@__DIR__, "boosted_damping.png")
savefig(out, dpi=120)
println("Saved ", out)
