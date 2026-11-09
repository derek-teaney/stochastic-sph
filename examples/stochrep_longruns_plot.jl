# stochrep_longruns_plot.jl — the long runs (t = 100) of stochrep_hamiltonian.jl compared:
# kinetic energy, particle energies, smoothed energy and local disorder versus time, the
# four realizations of each set shown as a band (min to max) around their mean.
# (The S = 10⁵ set, which blows up, has its own figure: stochrep_hamiltonian_real_plot.jl.)
# Run:  julia --project examples/stochrep_longruns_plot.jl → docs/figs/stochrep_longruns.png
using PyPlot, Printf, Serialization, StaticArrays
f(s) = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_hamiltonian_real_$(s).jls")
sets = [("reference: λ = 1, single vote, gradient map, Δt", "ha2.0_t100_star_grad", :full, "C0"),
        ("λ = ∞", "ha2.0_t100_star_grad", :det, "C2"),
        ("Δt/2", "ha2.0_t100_star_dt0.5_grad", :full, "C1"),
        ("per-bond vote", "ha2.0_t100_grad", :full, "C3"),
        ("plain map (no gradient correction)", "ha2.0_t100_star", :full, "C4"),
        ("η/s = 0.1", "ha2.0_t100_star_grad_eta0.1", :full, "C5"),
        ("h/a = 2.5", "ha2.5_t100_star_grad", :full, "C7")]
fig, axs = subplots(1, 4, figsize=(26, 6.5))
rows = String[]
for (lab, suf, kind, c) in sets
    isfile(f(suf)) || (println("missing ", suf); continue)
    res = deserialize(f(suf))
    rs = [r for ((k, s), r) in res if k === kind]
    isempty(rs) && continue
    t = rs[1].ts
    for (ax, fld) in zip(axs, (:KE, :Es, :Hs, :loc))
        Y = reduce(hcat, [getfield(r, fld) for r in rs])
        ax.fill_between(t, minimum(Y; dims=2)[:], maximum(Y; dims=2)[:], color=c, alpha=0.2)
        ax.plot(t, sum(Y; dims=2)[:] ./ size(Y, 2), color=c, lw=1.2, label=lab)
    end
    end_(fld) = sum(getfield(r, fld)[end] for r in rs) / length(rs)
    push!(rows, @sprintf("%-45s KE/eq %.3f   ΔΣe/Σe %+.2e   ΔH/H %+.2e   local disorder %.3f a",
                         lab, end_(:KE), end_(:Es), end_(:Hs), end_(:loc)))
end
axs[1].axhline(1, color="k", lw=0.5);  axs[1].set_ylim(0.85, 1.25)
axs[1].set_title("kinetic energy / equilibrium (λ = ∞ off scale, ≈ 0.05)")
axs[2].set_title("particle energies Σe/Σe₀ − 1");  axs[3].set_title("smoothed energy H/H₀ − 1")
axs[4].set_title("local disorder / a")
for ax in axs; ax.set_xlabel("t"); end
axs[2].legend(fontsize=8, loc="upper left")
suptitle("2D uniform fluid at rest, Hamiltonian step on, to t = 100: four realizations per set (band: min to max)")
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_longruns.png")
savefig(out, dpi=110);  println("Saved ", out)
foreach(println, rows)
