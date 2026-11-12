# damping_1d.jl — 1D fluid at rest: damping rate of a longitudinal momentum wave
# for the two-level and one-level (bond) dissipative functions, measured with the
# code and compared with the closed-form lattice symbols below (the bond form is
# that of docs/bond_dissipation.tex).
#
#   m_a = A cos(k x_a) on a uniform 1D lattice (spacing a), frozen positions,
#   Γ(k) = −Σ f_a m_a / Σ m_a²,  f = dissipative_force!  (no ideal step, no noise)
#
# Prediction:
#   Γ_2L / Γ_phys = R(k)² D(k)² ,  Γ_1L / Γ_phys = R(k)² B(k) ,  Γ_phys = κ k² / w
#   R(k) = Σ_r a W(r) cos kr          (smoothing response, b = R² m /(V w T))
#   D(k) = −(1/k) Σ_r a W'(r) sin kr  (SPH gradient response, ε = i k D b)
#   B(k) = Σ_r ψ(r) 2(1 − cos kr) / (k² Σ_r ψ(r) r²)   (bond response), ψ = −W'/r
#
# Run:  julia --project examples/damping_1d.jl   → docs/figs/damping_1d.png
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const KER = WendlandC2_1D()
W(r, h)  = wendland_c2(abs(r), h, KER)
Wp(r, h) = (q = abs(r) / h; q >= 2 ? 0.0 : sign(r) * (-5 * 3 / (4h)) * q * (1 - q / 2)^3 / h)
ψ(r, h)  = r == 0 ? 0.0 : -Wp(r, h) / r
lat(h, a) = [m * a for m in -ceil(Int, 2h / a):ceil(Int, 2h / a)]
Rk(k, h, a) = sum(a * W(r, h) * cos(k * r) for r in lat(h, a))
Dk(k, h, a) = -sum(a * Wp(r, h) * sin(k * r) for r in lat(h, a)) / k
Bk(k, h, a) = sum(ψ(r, h) * 2 * (1 - cos(k * r)) for r in lat(h, a)) /
              (k^2 * sum(ψ(r, h) * r^2 for r in lat(h, a)))

function measure(scheme, h, ha; Lx=1.0, ηs=0.1, ζs=0.0, pointwise=false)
    eos = ConformalEOS(1/3, 1.0, ηs, ζs)
    params, ps, cl, eos = setup_ic(h, ha, eos, (x, y) -> 2.0, (x, y) -> SVector(0.0, 0.0);
                                   Lx, sim_dim=1)
    params.dissipation = scheme
    a, N, L = params.a, params.N, params.L[1]
    w  = ps[1].e + ps[1].p
    κ  = (4/3 * ηs + ζs) * srel(eos, ps[1].e)            # κ^{xxxx} at rest, d = 3
    A  = 1e-5 * w * a
    f  = zeros(SVector{2,Float64}, N)
    nmax = N ÷ 2
    ns = unique(round.(Int, exp.(range(0, log(nmax), length=40))))
    ks, Γ = Float64[], Float64[]
    for n in ns
        k = 2π * n / L
        for p in ps; p.m = SVector(A * cos(k * (p.x[1] - a / 2)), 0.0); end
        thermalize!(ps, params, cl, eos)
        if pointwise          # β_a = v_a/T_a from the particle's own m_a (linear, at rest)
            T = ps[1].T * ps[1].γ
            bond_dissipative_force!(f, ps, params, cl, eos, [p.m / (a * w * T) for p in ps])
        else
            dissipative_force!(f, ps, params, cl, eos)
        end
        push!(ks, k)
        push!(Γ, -sum(f[i][1] * ps[i].m[1] for i in 1:N) / sum(ps[i].m[1]^2 for i in 1:N))
    end
    return ks, Γ ./ (κ / w .* ks .^ 2), a
end

fig, axs = subplots(1, 3, figsize=(16, 4.6))
for (j, ha) in enumerate((2.0, 4.0, 6.0))
    ax = axs[j];  h = 0.02
    k2, r2, a = measure(:two_level, h, ha)
    k1, r1, _ = measure(:bond, h, ha)
    kp, rp, _ = measure(:bond, h, ha; pointwise=true)
    kf = range(1e-3, π / a, length=400)
    R2 = Rk.(kf, h, a) .^ 2
    ax.semilogy(kf .* a, R2 .* Dk.(kf, h, a) .^ 2 .+ 1e-30, "C0-", lw=1, alpha=0.6)
    ax.semilogy(kf .* a, R2 .* Bk.(kf, h, a) .+ 1e-30, "C3-", lw=1, alpha=0.6)
    ax.semilogy(kf .* a, R2, "k:", lw=1, label="R² (floor from smoothed b)")
    ax.semilogy(kf .* a, Bk.(kf, h, a), "C2-", lw=1, alpha=0.6)
    ax.semilogy(kp .* a, rp, "C2^", ms=3.5, label="one-level, pointwise β (code)  vs  B (line)")
    ax.semilogy(k2 .* a, max.(r2, 1e-30), "C0o", ms=3.5, label="two-level (code)  vs  R²D² (line)")
    ax.semilogy(k1 .* a, max.(r1, 1e-30), "C3s", ms=3.5, label="one-level (code)  vs  R²B (line)")
    ax.axvline(a / h, color="0.5", ls="--", lw=0.8);  ax.text(a / h, 2, " k=1/h", color="0.4")
    ax.set_ylim(1e-8, 5);  ax.set_xlim(0, π)
    ax.set_xlabel("k a   (π = zigzag mode)");  ax.set_ylabel("Γ(k) / Γ_phys(k)")
    ax.set_title(@sprintf("1D at rest, h/a = %g", ha));  ax.legend(fontsize=8, loc="lower left")
    # consistency check: code vs formula
    e2 = maximum(abs.(r2 .- [Rk(k, h, a)^2 * Dk(k, h, a)^2 for k in k2]))
    e1 = maximum(abs.(r1 .- [Rk(k, h, a)^2 * Bk(k, h, a) for k in k1]))
    @printf("h/a=%g: pointwise max|code−B| %.2e\n", ha, maximum(abs.(rp .- Bk.(kp, h, a))))
    @printf("h/a=%g: max|code−formula| two-level %.2e  one-level %.2e;  at ka=π: R²D²=%.2e  R²B=%.2e  B=%.3f\n",
            ha, e2, e1, Rk(π / a, h, a)^2 * Dk(π / a, h, a)^2, Rk(π / a, h, a)^2 * Bk(π / a, h, a), Bk(π / a, h, a))
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "damping_1d.png")
savefig(out, dpi=130)
println("Saved ", out)
