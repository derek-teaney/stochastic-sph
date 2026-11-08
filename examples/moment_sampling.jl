# moment_sampling.jl — what makes the bond moment M_a = Σ_b ψ(|r_ab|) r_ab r_abᵀ anisotropic?
# (docs/particle_metrics.tex, "What sets the bond moment").
#
#   anisotropy A = |M − ½ tr M 𝟙|_F / (½ tr M)
#
# Compared: perfect hex lattice, a 10% sheared lattice (weights at the PHYSICAL bond
# length, as in the code, and at the REFERENCE-LATTICE length, as on a mesh), and jittered lattices
# (each particle displaced uniformly in a square of side j·a).  Bond weight
# ψ(r) ∝ (1 − r/2h)³ (= −W'/r of the Wendland kernel), support 2h.
#
# Run:  julia --project examples/moment_sampling.jl   → docs/figs/moment_sampling.png
using PyPlot, LinearAlgebra, Random, Printf, Statistics

h  = 1.0
ψ(r) = (q = r / h; q < 2 ? (1 - q / 2)^3 : 0.0)
aniso(M) = norm(M - tr(M) / 2 * I) / (tr(M) / 2)
hexlat(a, R) = [[a * (i + j / 2), a * j * sqrt(3) / 2] for i in -R:R for j in -R:R]
E = [1.1 0.06; 0.06 0.95]                             # 10% deformation gradient

# moment of the star around a centre particle c among points p (weights at physical length)
function moment(p, c)
    M = zeros(2, 2)
    for x in p
        r = x - c;  d = norm(r)
        (d == 0 || d >= 2h) && continue
        M .+= ψ(d) .* (r * r')
    end
    M
end

# ── statistics vs h/a ───────────────────────────────────────────────────────
rng  = Xoshiro(7)
has  = [2.0, 2.55, 3.0, 4.0, 6.0, 8.0]
jits = [0.1, 0.3, 1.0]
stats = Dict{Tuple{Float64,Float64},Tuple{Float64,Float64,Float64}}()   # (ha,j) → mean, p95, max
shear_phys = Float64[];  shear_lab = Float64[];  nnb = Float64[];  pois = Tuple{Float64,Float64}[]
# fully random (Poisson) particles at the same mean density, with a particle at the origin
function poisson_star(rng, a)
    ρ = 2 / (sqrt(3) * a^2);  Lb = 5h;  n = round(Int, ρ * (2Lb)^2)
    vcat([[0.0, 0.0]], [Lb .* (2 .* rand(rng, 2) .- 1) for _ in 1:n])
end
for ha in has
    a = h / ha;  R = ceil(Int, 3ha)
    L0 = hexlat(a, R)
    push!(shear_phys, aniso(moment([E * v for v in L0], [0.0, 0.0])))
    push!(shear_lab, aniso(sum(ψ(norm(v)) * (E * v) * (E * v)' for v in L0 if 0 < norm(v) < 2h)))
    push!(nnb, count(v -> 0 < norm(v) < 2h, L0))
    for j in jits
        A = Float64[]
        for t in 1:2000
            p = [v .+ j * a .* (rand(rng, 2) .- 0.5) for v in L0]
            c = p[findmin(norm.(L0))[2]]                       # the particle at the origin
            push!(A, aniso(moment(p, c)))
        end
        stats[(ha, j)] = (mean(A), quantile(A, 0.95), maximum(A))
    end
    AP = [aniso(moment(poisson_star(rng, a), [0.0, 0.0])) for t in 1:2000]
    push!(pois, (mean(AP), maximum(AP)))
end

println("anisotropy |dev M|/(½trM) of the bond moment (2000 random stars per entry)")
println("(jitter j: each coordinate displaced uniformly in ±j·a/2)")
@printf("%6s %6s | %-22s | %-22s | %-22s | %-22s | %9s %9s\n", "h/a", "N_nb", "jitter 0.1a mean/max",
        "jitter 0.3a mean/max", "jitter 1.0a mean/max", "Poisson mean/max", "shear,phys", "shear,lab")
for (k, ha) in enumerate(has)
    s1, s3, s5 = stats[(ha, 0.1)], stats[(ha, 0.3)], stats[(ha, 1.0)]
    @printf("%6.2f %6d | %9.1e / %9.1e  | %9.1e / %9.1e  | %9.1e / %9.1e  | %9.1e / %9.1e  | %9.1e %9.1e\n",
            ha, nnb[k], s1[1], s1[3], s3[1], s3[3], s5[1], s5[3], pois[k][1], pois[k][2], shear_phys[k], shear_lab[k])
end
mp = [x[1] for x in pois]
@printf("Poisson: mean anisotropy ∝ (h/a)^%.2f\n", (log(mp[end]) - log(mp[2])) / (log(has[end]) - log(has[2])))
m1 = [stats[(ha, 1.0)][1] for ha in has]
@printf("jitter 1.0a: mean anisotropy ∝ (h/a)^%.2f\n", (log(m1[end]) - log(m1[2])) / (log(has[end]) - log(has[2])))
m3 = [stats[(ha, 0.3)][1] for ha in has]
slope = (log(m3[end]) - log(m3[2])) / (log(has[end]) - log(has[2]))
@printf("jitter 0.3a: mean anisotropy ∝ (h/a)^%.2f between h/a = %.2f and %.0f\n", slope, has[2], has[end])

# ── figure ──────────────────────────────────────────────────────────────────
fig = figure(figsize=(19, 9.5))
function star!(ax, p, c, title; lab=nothing)
    ws = [ψ(norm(x - c)) for x in p]
    inside = [0 < norm(x - c) < 2h for x in p]
    ax.scatter([x[1] for x in p], [x[2] for x in p], s=6, c="0.75", zorder=1)
    sc = ax.scatter([x[1] for (x, f) in zip(p, inside) if f], [x[2] for (x, f) in zip(p, inside) if f],
                    s=22, c=[w for (w, f) in zip(ws, inside) if f], cmap="viridis", vmin=0, vmax=1, zorder=2)
    ax.plot(c[1], c[2], "r*", ms=14, zorder=3)
    θ = range(0, 2π, length=200)
    ax.plot(c[1] .+ 2h .* cos.(θ), c[2] .+ 2h .* sin.(θ), "k:", lw=1)
    M = lab === nothing ? moment(p, c) : lab
    n = tr(M) / 2;  F = eigen(Symmetric(M ./ n))
    # ellipse of the normalized moment; deviations from the circle exaggerated ×kx
    kx  = min(10.0, 0.6 / max(maximum(abs.(F.values .- 1)), 1e-12))
    ell = [F.vectors * (sqrt.(1 .+ kx .* (F.values .- 1)) .* [cos(t), sin(t)]) for t in θ]
    ax.plot(c[1] .+ 1.2h .* [e[1] for e in ell], c[2] .+ 1.2h .* [e[2] for e in ell], "r-", lw=1.8)
    ax.plot(c[1] .+ 1.2h .* cos.(θ), c[2] .+ 1.2h .* sin.(θ), "r:", lw=0.8)
    ax.set_xlim(c[1] - 2.4h, c[1] + 2.4h);  ax.set_ylim(c[2] - 2.4h, c[2] + 2.4h);  ax.set_aspect("equal")
    ax.set_title(title * @sprintf("\nanisotropy %.1e  (ellipse: deviation ×%.0f)", aniso(M), kx), fontsize=10)
    ax.set_xticks([]);  ax.set_yticks([])
    sc
end
a  = h / 2.55;  L0 = hexlat(a, 10)
rngp = Xoshiro(3)
pj = [v .+ 0.3a .* (rand(rngp, 2) .- 0.5) for v in L0];  cj = pj[findmin(norm.(L0))[2]]
pJ = [v .+ 1.0a .* (rand(rngp, 2) .- 0.5) for v in L0];  cJ = pJ[findmin(norm.(L0))[2]]
pP = poisson_star(rngp, a)
ps = [E * v for v in L0]
Mlab = sum(ψ(norm(v)) * (E * v) * (E * v)' for v in L0 if 0 < norm(v) < 2h)
ax1 = subplot(2, 4, 1);  star!(ax1, L0, [0.0, 0.0], "perfect hex lattice, h/a = 2.55")
ax2 = subplot(2, 4, 2);  star!(ax2, pj, cj, "jitter 0.3a (shifts ≤ ±0.15a)")
ax3 = subplot(2, 4, 3);  star!(ax3, pJ, cJ, "jitter 1.0a (shifts ≤ ±0.5a)")
ax6 = subplot(2, 4, 4);  star!(ax6, pP, [0.0, 0.0], "random (Poisson) particles")
ax4 = subplot(2, 4, 5);  star!(ax4, ps, [0.0, 0.0], "10% shear, physical-length weights")
ax7 = subplot(2, 4, 6);  sc = star!(ax7, ps, [0.0, 0.0], "10% shear, reference-lattice weights"; lab=Mlab)
cax = fig.add_axes([0.13, 0.04, 0.33, 0.015])
colorbar(sc, cax=cax, orientation="horizontal", label="bond weight ψ")
ax5 = subplot(2, 4, (7, 8))
for (j, c) in zip(jits, ("C0", "C1", "C2"))
    ax5.loglog(has, [stats[(ha, j)][1] for ha in has], "o-", color=c, label=@sprintf("jitter %.1fa: mean", j))
    ax5.loglog(has, [stats[(ha, j)][3] for ha in has], "^:", color=c, ms=4, label=@sprintf("jitter %.1fa: max", j))
end
ax5.loglog(has, 0.4 .* 0.3 ./ has .^ 2, "C1-", lw=4, alpha=0.25, label="0.4·j·(a/h)², j = 0.3")
ax5.loglog(has, [x[1] for x in pois], "mo-", label="Poisson: mean")
ax5.loglog(has, [x[2] for x in pois], "m^:", ms=4, label="Poisson: max")
ax5.loglog(has, shear_phys, "ks-", label="10% shear, physical-length weights")
ax5.loglog(has, shear_lab, "kD--", label="10% shear, reference-lattice weights")
ax5.set_xlabel("h/a");  ax5.set_ylabel("anisotropy |dev M| / (½ tr M)")
ax5.set_title("sampling error of the bond moment");  ax5.legend(fontsize=7, loc="lower left", ncol=2)
ax5.set_ylim(1e-7, 2)
ax5.set_xticks(has);  ax5.set_xticklabels([string(x) for x in has])
out = joinpath(@__DIR__, "..", "docs", "figs", "moment_sampling.png")
savefig(out, dpi=130, bbox_inches="tight");  println("Saved ", out)
