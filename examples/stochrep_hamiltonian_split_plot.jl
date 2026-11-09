# stochrep_hamiltonian_split_plot.jl — where the energy goes: cumulative changes of H and Σe
# from the stochastic half steps and from the leapfrog (stochrep_hamiltonian.jl ... split),
# for the star vote at Δt and Δt/2, λ = 1, h/a = 2.
# Run:  julia --project examples/stochrep_hamiltonian_split_plot.jl → docs/figs/stochrep_hamiltonian_split.png
using PyPlot, Printf, Serialization, StaticArrays
f(s) = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_hamiltonian_real_ha2.0_t30_star$(s).jls")
# runs to compare: suffixes from the command line (default: Δt and Δt/2 without the gradient correction)
sufs = length(ARGS) > 0 ? ARGS : ["", "_dt0.5"]
names = Dict("" => "Δt", "_dt0.5" => "Δt/2", "_grad" => "Δt, gradient-corrected map",
             "_dt0.5_grad" => "Δt/2, gradient-corrected map")
runs = [(get(names, x, x), deserialize(f(x)), "C$(i-1)") for (i, x) in enumerate(sufs)]
fig, axs = subplots(1, 5, figsize=(29, 5.5))
for (lab, res, c) in runs, (i, ((kind, seed), r)) in enumerate(sort(collect(res), by = x -> x[1][2]))
    kind === :full || continue
    t = r.split_t;  S = reduce(hcat, [collect(x) for x in r.split])'
    l = i == 1 ? lab : nothing
    axs[1].plot(t, S[:, 1], color=c, lw=1, label=l)
    axs[2].plot(t, S[:, 2], color=c, lw=1, label=l)
    axs[3].plot(t, S[:, 1] .+ S[:, 2], color=c, lw=1, label=l)
    axs[4].plot(t, S[:, 3], color=c, lw=1, label=l)
    axs[5].plot(t, S[:, 4], color=c, lw=1, label=l)
end
titles = ("ΔH from the stochastic half steps", "ΔH from the leapfrog", "ΔH total",
          "ΔΣe from the stochastic half steps (e_a recomputed from the smoothed state)",
          "ΔΣe from the leapfrog")
for (ax, tt) in zip(axs, titles)
    ax.set_title(tt, fontsize=10);  ax.set_xlabel("t");  ax.axhline(0, color="k", lw=0.5);  ax.legend(fontsize=8)
end
axs[1].set_ylabel("cumulative change / H₀ (or Σe₀)")
suptitle("2D uniform fluid, Hamiltonian step on, h/a = 2, λ = 1, star vote: where the energy changes (4 realizations each)")

tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_hamiltonian_split$(join(sufs, "")).png")
savefig(out, dpi=110);  println("Saved ", out)
