# gaussian_compare_plot.jl — plots for the two-level vs bond Gaussian comparison.
#   julia --project examples/gaussian_compare_plot.jl <eta_by_s> <h> [<h_ref>]
# Reads examples/gaussian_cmp/*.txt written by gaussian_compare_run.jl and writes
# gaussian_compare_eta<eta>.png next to this script.
using PyPlot, DelimitedFiles, Printf, Statistics

ηs   = parse(Float64, ARGS[1])
h    = parse(Float64, ARGS[2])
href = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : nothing
dir  = joinpath(@__DIR__, "gaussian_cmp")
load(s, hh) = (f = joinpath(dir, @sprintf("%s_h%g_eta%g.txt", s, hh, ηs)); isfile(f) ? readdlm(f) : nothing)
loadp(s, hh) = (f = joinpath(dir, @sprintf("%s_h%g_eta%g_particles.txt", s, hh, ηs)); isfile(f) ? readdlm(f) : nothing)

T2, B = load("two_level", h), load("bond", h)
(T2 === nothing || B === nothing) && error("runs for η/s=$ηs, h=$h not finished")
R2 = href === nothing ? nothing : load("two_level", href)
RB = href === nothing ? nothing : load("bond", href)

fig, axs = subplots(1, 3, figsize=(16, 4.6))
for (j, (col, lab)) in enumerate(((1, "x"), (3, "y")))
    ax = axs[j]
    ax.plot(T2[:, col], T2[:, col+1], "C0-",  lw=1.6, label=@sprintf("two-level, h=%g", h))
    ax.plot(B[:, col],  B[:, col+1],  "C3-",  lw=1.6, label=@sprintf("bond, h=%g", h))
    R2 === nothing || ax.plot(R2[:, col], R2[:, col+1], "C0:", lw=1.2, label=@sprintf("two-level, h=%g (ref)", href))
    RB === nothing || ax.plot(RB[:, col], RB[:, col+1], "k--", lw=1.2, label=@sprintf("bond, h=%g (ref)", href))
    ax.set_xlabel(lab);  ax.set_ylabel("energy density e")
    ax.set_title(@sprintf("e along %s through the centre, t = 2.2  (η/s = %g)", lab, ηs))
    ax.legend(fontsize=8)
end
# difference of the two schemes along both slices, and vs the reference
ax = axs[3]
ax.plot(T2[:, 1], B[:, 2] .- T2[:, 2], "C2-", lw=1.4, label="bond − two-level (x slice)")
ax.plot(T2[:, 3], B[:, 4] .- T2[:, 4], "C4-", lw=1.4, label="bond − two-level (y slice)")
if RB !== nothing
    ax.plot(T2[:, 1], T2[:, 2] .- RB[:, 2], "C0:", lw=1.2, label="two-level − ref (x)")
    ax.plot(T2[:, 1], B[:, 2]  .- RB[:, 2], "C3:", lw=1.2, label="bond − ref (x)")
    @printf("η/s=%g  rms deviation from h=%g bond reference (x slice): two-level %.3e   bond %.3e\n",
            ηs, href, sqrt(mean((T2[:, 2] .- RB[:, 2]).^2)), sqrt(mean((B[:, 2] .- RB[:, 2]).^2)))
end
ax.axhline(0, color="0.5", lw=0.5)
ax.set_xlabel("x or y");  ax.set_ylabel("Δe");  ax.set_title("differences");  ax.legend(fontsize=8)
@printf("η/s=%g  h=%g  max |bond − two-level| on slices: %.3e   (peak e ≈ %.3f)\n",
        ηs, h, maximum(abs.(vcat(B[:, 2] .- T2[:, 2], B[:, 4] .- T2[:, 4]))), maximum(T2[:, 2]))
tight_layout()
out = joinpath(@__DIR__, @sprintf("gaussian_compare_eta%g.png", ηs))
savefig(out, dpi=130)
println("Saved ", out)

# 2D maps of e for both schemes and their difference (particle scatter)
P2, PB = loadp("two_level", h), loadp("bond", h)
fig2, ax2 = subplots(1, 3, figsize=(16, 4.8))
for (k, (P, lab)) in enumerate(((P2, "two-level"), (PB, "bond")))
    sc = ax2[k].scatter(P[:, 1], P[:, 2], c=P[:, 3], s=2, cmap="viridis")
    colorbar(sc, ax=ax2[k]);  ax2[k].set_aspect("equal");  ax2[k].set_title("e, " * lab * @sprintf(", h=%g, t=2.2", h))
end
d  = PB[:, 3] .- P2[:, 3]       # same particle labels in both runs
m  = maximum(abs, d)
sc = ax2[3].scatter(PB[:, 1], PB[:, 2], c=d, s=2, cmap="RdBu_r", vmin=-m, vmax=m)
colorbar(sc, ax=ax2[3]);  ax2[3].set_aspect("equal");  ax2[3].set_title("e(bond) − e(two-level), by particle")
tight_layout()
out2 = joinpath(@__DIR__, @sprintf("gaussian_maps_eta%g.png", ηs))
savefig(out2, dpi=130)
println("Saved ", out2)
