# changerep_convergence_plot.jl — redraw the H(t), S(t) panels of changerep_convergence.jl
# from its saved curves (examples/compare/conv_<scheme>_h<h>.txt), with each panel on its
# own vertical scale.
#
# Run:  julia --project examples/changerep_convergence_plot.jl  → docs/figs/changerep_convergence.png
using PyPlot, DelimitedFiles, Printf
hs    = (0.045, 0.03, 0.0225, 0.015)
kinds = ((:two_level, "two-level form"), (:smoothed, "bond form on the smoothed b"),
         (:exact, "particle representation, exact back map"),
         (:grad, "particle representation, gradient-corrected"))
load(k, h) = readdlm(joinpath(@__DIR__, "compare", @sprintf("conv_%s_h%g.txt", k, h)))
fig, axs = subplots(2, 4, figsize=(26, 9.5))
for (j, (k, title)) in enumerate(kinds)
    for (i, h) in enumerate(hs)
        d = load(k, h);  c = get_cmap("viridis")((i - 1) / (length(hs) - 1))
        axs[1, j].plot(d[:, 1], d[:, 2], "-", color=c, lw=1.2, label=@sprintf("h = %g", h))
        axs[2, j].plot(d[:, 1], d[:, 3], "-", color=c, lw=1.2, label=@sprintf("h = %g", h))
    end
    if k !== :two_level
        d = load(:two_level, hs[end])
        axs[1, j].plot(d[:, 1], d[:, 2], "k:", lw=1, label=@sprintf("two-level, h = %g", hs[end]))
        axs[2, j].plot(d[:, 1], d[:, 3], "k:", lw=1, label=@sprintf("two-level, h = %g", hs[end]))
    end
    axs[1, j].axhline(0, color="0.5", lw=0.6)
    axs[1, j].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axs[1, j].set_title(title); axs[1, j].set_xlabel("t"); axs[1, j].set_ylabel("H(t)/H(0) − 1")
    axs[2, j].set_xlabel("t"); axs[2, j].set_ylabel("S(t)/S(0) − 1"); axs[2, j].set_ylim(0, 0.047)
    axs[1, j].legend(fontsize=8); axs[2, j].legend(fontsize=8)
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "changerep_convergence.png")
savefig(out, dpi=130); println("Saved ", out)
