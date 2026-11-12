# damping_2d.jl — 2D fluid at rest (hex lattice, physical d = 3): damping rates of
# transverse (shear) and longitudinal momentum waves, k ∥ x, for
#   two-level                 (code, smoothed b)
#   plain bond, no mean field (this script, smoothed b and pointwise β)
#   mean-field bond           (code :bond, smoothed b and pointwise β)
# relative to physics  Γ_T = η k²/w ,  Γ_L = (2η(d−1)/d + ζ) k²/w.
#   Γ(k) = −Σ f_a·m_a / Σ m_a·m_a ,  m_a = A ê cos(k x_a)    (forms: docs/bond_dissipation.tex)
#
# Run:  julia --project examples/damping_2d.jl   → docs/figs/damping_2d.png
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "..", "src", "legacy", "load.jl"))   # two_level, particle_sweep (superseded)

const ηs, ζs, d = 1.0, 0.0, 3
eos = ConformalEOS(1/3, 1.0, ηs, ζs)
params, ps, cl, eos = setup_ic(15.0, 2.55, eos, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                               L=150.0, lattice=:hex, d=d)
h, a, V, L, N = params.h, params.a, params.vol, params.L, params.N
w  = ps[1].e + ps[1].p
T  = ps[1].T * ps[1].γ
η  = ηs * srel(eos, ps[1].e);  ζ = ζs * srel(eos, ps[1].e)
ΓT(k) = η * k^2 / w
ΓL(k) = (2η * (d - 1) / d + ζ) * k^2 / w

# plain bond force (no mean field):  f_a = Σ_b (α_a+α_b) V_a V_b ψ_ab (β_b − β_a),
# α = T η / n,  n 𝟙 = Σ_b V_b ψ r rᵀ (hex lattice: isotropic).
nbrs = [[(b, minimum_image(ps[b].x - ps[a].x, L)) for b in 1:N
         if b != a && norm(minimum_image(ps[b].x - ps[a].x, L)) < 2h] for a in 1:N]
ψw(r) = bond_weight(norm(r), h)
nmom  = [sum(V * ψw(r) * dot(r, r) for (b, r) in nbrs[a]) / 2 for a in 1:N]
αp    = [T * η / nmom[a] for a in 1:N]
function plain_force!(f, β)
    for a in 1:N
        acc = zero(SVector{2,Float64})
        for (b, r) in nbrs[a]
            acc += (αp[a] + αp[b]) * V * V * ψw(r) * (β[b] - β[a])
        end
        f[a] = acc
    end
end

ns = 1:17                                # k up to the K point 4π/(3a) of the hex zone
ks = [2π * n / L[1] for n in ns]
f  = zeros(SVector{2,Float64}, N)
A  = 1e-5 * w * V
rate(f) = -sum(dot(f[i], ps[i].m) for i in 1:N) / sum(dot(p.m, p.m) for p in ps)
res = Dict{Tuple{Symbol,Symbol},Vector{Float64}}()
for pol in (:T, :L)
    ê = pol === :T ? SVector(0.0, 1.0) : SVector(1.0, 0.0)
    Γp = pol === :T ? ΓT : ΓL
    for key in (:two_level, :plain_b, :plain_β, :mf_b, :mf_β)
        res[(pol, key)] = Float64[]
    end
    for k in ks
        for p in ps; p.m = A * cos(k * p.x[1]) * ê; end
        thermalize!(ps, params, cl, eos)
        bsm = [p.dotx / p.Teff for p in ps]                 # smoothed b
        βpt = [p.m / (V * w * T) for p in ps]               # pointwise β (linear, at rest)
        params.dissipation = :two_level;  dissipative_force!(f, ps, params, cl, eos)
        push!(res[(pol, :two_level)], rate(f) / Γp(k))
        plain_force!(f, bsm);  push!(res[(pol, :plain_b)], rate(f) / Γp(k))
        plain_force!(f, βpt);  push!(res[(pol, :plain_β)], rate(f) / Γp(k))
        params.dissipation = :bond
        bond_dissipative_force!(f, ps, params, cl, eos, bsm);  push!(res[(pol, :mf_b)], rate(f) / Γp(k))
        bond_dissipative_force!(f, ps, params, cl, eos, βpt);  push!(res[(pol, :mf_β)], rate(f) / Γp(k))
    end
end

lab = Dict(:two_level => "two-level (smoothed b)", :plain_b => "plain bond, smoothed b",
           :plain_β => "plain bond, pointwise β", :mf_b => "mean-field bond, smoothed b",
           :mf_β => "mean-field bond, pointwise β")
sty = Dict(:two_level => "C0o-", :plain_b => "C2s--", :plain_β => "C2^-",
           :mf_b => "C3s--", :mf_β => "C3^-")
fig, axs = subplots(1, 2, figsize=(14, 5))
for (j, (pol, ttl)) in enumerate(((:T, "transverse (shear) wave"), (:L, "longitudinal wave")))
    ax = axs[j]
    for key in (:two_level, :plain_b, :plain_β, :mf_b, :mf_β)
        ax.semilogy(ks .* a, max.(res[(pol, key)], 1e-12), sty[key], ms=3.5, lw=1.2, label=lab[key])
    end
    ax.axhline(1, color="k", lw=0.6)
    ax.axvline(a / h, color="0.5", ls=":", lw=0.8);  ax.text(a / h, 2.2, " k=1/h", color="0.4")
    ax.set_ylim(1e-8, 4);  ax.set_xlabel("k a   (k ∥ x; zone edge 4π/3 ≈ 4.19)")
    ax.set_ylabel("Γ(k) / Γ_phys(k)");  ax.set_title("2D hex lattice at rest, h/a = 2.55: " * ttl)
    ax.legend(fontsize=8, loc="lower left")
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "damping_2d.png")
savefig(out, dpi=130);  println("Saved ", out)

println("  ka    pol  two-level  plain_b   plain_β   mf_b      mf_β")
for (i, k) in enumerate(ks), pol in (:T, :L)
    i in (1, 3, 6, 10, 14, 17) || continue
    @printf("%6.3f  %s  %9.3e %9.3e %9.3e %9.3e %9.3e\n", k * a, pol,
            res[(pol, :two_level)][i], res[(pol, :plain_b)][i], res[(pol, :plain_β)][i],
            res[(pol, :mf_b)][i], res[(pol, :mf_β)][i])
end
