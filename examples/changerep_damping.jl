# changerep_damping.jl — damping of a momentum wave on a frozen lattice at rest, with the
# viscous step done in the smoothed representation (b = ẋ/Teff) and in the particle
# representation (pointwise β_a from the particle's own m_a, src/changerep.jl).
#
#   Γ(k) = −Σ F_a·m_a / Σ m_a·m_a ,   m_a = A ê cos(k x_a)
#
# Expected (docs/changerep.tex):  smoothed b → Γ/Γ_phys = R²B ;  pointwise β → B.
# Panel 2 checks the full time step (viscous_step_first!): one step multiplies
# the mode by g = 1 − Γ Δt, with Γ = Γ_phys B.
#
# Run:  julia --project examples/changerep_damping.jl  → docs/figs/changerep_damping.png
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "..", "src", "legacy", "load.jl"))   # two_level, particle_sweep (superseded)

# ── 1D lattice symbols (as in damping_1d.jl) ──────────────────────────────────
const KER = WendlandC2_1D()
W(r, h)  = kernel_W(abs(r), h, KER)
Wp(r, h) = (q = abs(r) / h; q >= 2 ? 0.0 : sign(r) * (-5 * 3 / (4h)) * q * (1 - q / 2)^3 / h)
ψ1(r, h) = r == 0 ? 0.0 : -Wp(r, h) / r
lat(h, a) = [m * a for m in -ceil(Int, 2h / a):ceil(Int, 2h / a)]
Rk(k, h, a) = sum(a * W(r, h) * cos(k * r) for r in lat(h, a))
Dk(k, h, a) = -sum(a * Wp(r, h) * sin(k * r) for r in lat(h, a)) / k
Bk(k, h, a) = sum(ψ1(r, h) * 2 * (1 - cos(k * r)) for r in lat(h, a)) /
              (k^2 * sum(ψ1(r, h) * r^2 for r in lat(h, a)))

rate(f, ps) = -sum(dot(f[i], ps[i].m) for i in eachindex(ps)) / sum(dot(p.m, p.m) for p in ps)

# ── 1D: instantaneous rates ───────────────────────────────────────────────────
ηs1 = 0.1
eos1 = ConformalEOS(1/3, 1.0, ηs1, 0.0)
h1, ha1 = 0.02, 4.0
params1, ps1, cl1, eos1 = setup_ic(h1, ha1, eos1, (x, y) -> 2.0, (x, y) -> SVector(0.0, 0.0);
                                   Lx=1.0, sim_dim=1)
a1, N1, L1 = params1.a, params1.N, params1.L[1]
w1 = ps1[1].e + ps1[1].p
κ1 = 4/3 * ηs1 * srel(eos1, ps1[1].e)
Γp1(k) = κ1 / w1 * k^2
A1 = 1e-5 * w1 * a1
f1 = zeros(SVector{2,Float64}, N1)
β1 = zeros(SVector{2,Float64}, N1)
s0 = [p.s for p in ps1]
ns1 = unique(round.(Int, exp.(range(0, log(N1 ÷ 2), length=40))))
k1 = [2π * n / L1 for n in ns1]
r_two, r_b, r_β = Float64[], Float64[], Float64[]
for k in k1
    for (i, p) in enumerate(ps1); p.s = s0[i]; p.m = SVector(A1 * cos(k * (p.x[1] - a1 / 2)), 0.0); end
    thermalize!(ps1, params1, cl1, eos1)
    params1.dissipation = :two_level; dissipative_force!(f1, ps1, params1, cl1, eos1)
    push!(r_two, rate(f1, ps1) / Γp1(k))
    params1.dissipation = :bond;      dissipative_force!(f1, ps1, params1, cl1, eos1)
    push!(r_b, rate(f1, ps1) / Γp1(k))
    pointwise_beta!(β1, ps1, eos1)
    bond_dissipative_force!(f1, ps1, params1, cl1, eos1, β1)
    push!(r_β, rate(f1, ps1) / Γp1(k))
end
@printf("1D h/a=%g: max|pointwise − B| = %.2e,  max|smoothed − R²B| = %.2e\n", ha1,
        maximum(abs.(r_β .- Bk.(k1, h1, a1))),
        maximum(abs.(r_b .- [Rk(k, h1, a1)^2 * Bk(k, h1, a1) for k in k1])))

# ── 1D: full time step in the particle representation ─────────────────────────
# One explicit step multiplies the mode by g = 1 − Γ Δt.  Measure g for each k at the
# default viscous Δt and at half of it, and check ΣΔs = 2K Δt.
params1.dissipation = :bond
dt1  = compute_dt(ps1, params1, eos1; mode=:viscous)
k_ev = [2π * n / L1 for n in unique(round.(Int, range(1, N1 ÷ 2, length=25)))]
g_ev = Dict(f => Float64[] for f in (1.0, 0.5))
dS_err = 0.0
for fdt in (1.0, 0.5), k in k_ev
    for (i, p) in enumerate(ps1); p.s = s0[i]; p.m = SVector(A1 * cos(k * (p.x[1] - a1 / 2)), 0.0); end
    proj() = sum(p.m[1] * cos(k * (p.x[1] - a1 / 2)) for p in ps1)
    amp0 = proj()
    K, ΣΔs = viscous_step_first!(ps1, params1, cl1, fdt * dt1, eos1)
    push!(g_ev[fdt], proj() / amp0)
    global dS_err = max(dS_err, abs(ΣΔs - 2K * fdt * dt1) / (2K * fdt * dt1))
end
gpred(k, fdt) = 1 - Γp1(k) * Bk(k, h1, a1) * fdt * dt1
@printf("1D time step: max Γ_phys B Δt = %.3f at the default viscous Δt\n",
        maximum(Γp1(k) * Bk(k, h1, a1) for k in k1) * dt1)
@printf("1D time step: max |g − (1 − Γ_phys B Δt)| = %.2e\n",
        maximum(abs(g_ev[f][i] - gpred(k, f)) for f in (1.0, 0.5) for (i, k) in enumerate(k_ev)))
@printf("1D time step: entropy production ΣΔs vs 2KΔt, max rel. error = %.2e\n", dS_err)

# ── 2D hex lattice at rest: instantaneous rates for the mean-field bond form ──
const ηs2, d2 = 1.0, 3
eos2 = ConformalEOS(1/3, 1.0, ηs2, 0.0)
params2, ps2, cl2, eos2 = setup_ic(15.0, 2.55, eos2, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                   L=150.0, lattice=:hex, d=d2)
h2, a2, V2, L2, N2 = params2.h, params2.a, params2.vol, params2.L, params2.N
w2 = ps2[1].e + ps2[1].p
η2 = ηs2 * srel(eos2, ps2[1].e)
ΓT(k) = η2 * k^2 / w2
ΓL(k) = 2η2 * (d2 - 1) / d2 * k^2 / w2
f2 = zeros(SVector{2,Float64}, N2)
β2 = zeros(SVector{2,Float64}, N2)
k2 = [2π * n / L2[1] for n in 1:17]
res2 = Dict{Tuple{Symbol,Symbol},Vector{Float64}}()
for pol in (:T, :L)
    ê  = pol === :T ? SVector(0.0, 1.0) : SVector(1.0, 0.0)
    Γp = pol === :T ? ΓT : ΓL
    for key in (:two, :b, :β); res2[(pol, key)] = Float64[]; end
    for k in k2
        for p in ps2; p.m = 1e-5 * w2 * V2 * cos(k * p.x[1]) * ê; end
        thermalize!(ps2, params2, cl2, eos2)
        params2.dissipation = :two_level; dissipative_force!(f2, ps2, params2, cl2, eos2)
        push!(res2[(pol, :two)], rate(f2, ps2) / Γp(k))
        params2.dissipation = :bond;      dissipative_force!(f2, ps2, params2, cl2, eos2)
        push!(res2[(pol, :b)], rate(f2, ps2) / Γp(k))
        pointwise_beta!(β2, ps2, eos2)
        bond_dissipative_force!(f2, ps2, params2, cl2, eos2, β2)
        push!(res2[(pol, :β)], rate(f2, ps2) / Γp(k))
    end
end

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axs = subplots(1, 4, figsize=(22, 4.8))
kf = range(1e-3, π / a1, length=400)
global ax = axs[1]
ax.semilogy(kf .* a1, Rk.(kf, h1, a1) .^ 2 .* Dk.(kf, h1, a1) .^ 2 .+ 1e-30, "C0-", lw=1, alpha=0.6)
ax.semilogy(kf .* a1, Rk.(kf, h1, a1) .^ 2 .* Bk.(kf, h1, a1), "C3-", lw=1, alpha=0.6)
ax.semilogy(kf .* a1, Bk.(kf, h1, a1), "C2-", lw=1, alpha=0.6)
ax.semilogy(k1 .* a1, max.(r_two, 1e-30), "C0o", ms=3.5, label="two-level, smoothed b  (line R²D²)")
ax.semilogy(k1 .* a1, r_b, "C3s", ms=3.5, label="bond, smoothed b  (line R²B)")
ax.semilogy(k1 .* a1, r_β, "C2^", ms=3.5, label="bond, pointwise β  (line B)")
ax.axvline(a1 / h1, color="0.5", ls=":", lw=0.8)
ax.set_ylim(1e-8, 5); ax.set_xlim(0, π)
ax.set_xlabel("k a   (π = zigzag)"); ax.set_ylabel("Γ(k) / Γ_phys(k)")
ax.set_title(@sprintf("1D at rest, h/a = %g", ha1)); ax.legend(fontsize=8, loc="lower left")

global ax = axs[2]
for (fdt, c) in ((1.0, "C2"), (0.5, "C4"))
    ax.plot(kf .* a1, gpred.(kf, fdt), c * "-", lw=1)
    ax.plot(k_ev .* a1, g_ev[fdt], c * "^", ms=5, label=@sprintf("Δt = %.1f × default viscous Δt", fdt))
end
ax.axhline(0, color="k", lw=0.6);  ax.axhline(-1, color="k", ls="--", lw=0.6)
ax.set_xlim(0, π); ax.set_ylim(-1.1, 1.05)
ax.set_xlabel("k a"); ax.set_ylabel("per-step factor g")
ax.set_title("1D: one step of viscous_step_first!  (line: 1 − Γ_phys B Δt)")
ax.legend(fontsize=8, loc="lower left")

for (j, (pol, ttl)) in enumerate(((:T, "transverse"), (:L, "longitudinal")))
    local ax = axs[2 + j]
    ax.semilogy(k2 .* a2, max.(res2[(pol, :two)], 1e-14), "C0o-", ms=3.5, lw=1, label="two-level, smoothed b")
    ax.semilogy(k2 .* a2, res2[(pol, :b)], "C3s-", ms=3.5, lw=1, label="bond, smoothed b")
    ax.semilogy(k2 .* a2, res2[(pol, :β)], "C2^-", ms=3.5, lw=1, label="bond, pointwise β")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_ylim(1e-8, 4); ax.set_xlabel("k a   (zone edge 4π/3 ≈ 4.19)")
    ax.set_ylabel("Γ(k) / Γ_phys(k)"); ax.set_title("2D hex, h/a = 2.55: " * ttl * " wave")
    ax.legend(fontsize=8, loc="lower left")
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "changerep_damping.png")
savefig(out, dpi=130); println("Saved ", out)

println("  ka    pol  two-level  bond_b    bond_β")
for (i, k) in enumerate(k2), pol in (:T, :L)
    i in (1, 3, 6, 10, 14, 17) || continue
    @printf("%6.3f  %s  %9.3e %9.3e %9.3e\n", k * a2, pol,
            res2[(pol, :two)][i], res2[(pol, :b)][i], res2[(pol, :β)][i])
end
