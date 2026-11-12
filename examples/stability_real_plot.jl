# stability_real_plot.jl — all realizations of stability.jl (NREAL > 1).
# Run:  julia --project examples/stability_real_plot.jl [h/a]
#   → docs/figs/stochrep_hamiltonian_real_ha<h/a>.png
using PyPlot, Printf, Serialization, StaticArrays
const HA = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 2.0
const TE = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 15
const SUF = length(ARGS) > 2 ? "_" * ARGS[3] : ""            # e.g. star
res = deserialize(joinpath(@__DIR__, "..", "docs", "figs", "stochrep_hamiltonian_real_ha$(HA)_t$(TE)$(SUF).jls"))
lab = Dict(:full => "stress + residual (λ = 1)", :det => "stress + deterministic residual (λ = ∞)",
           :stress => "stress only")
col = Dict(:full => "C0", :det => "C2", :stress => "C3")
fig, axs = subplots(1, 4, figsize=(25, 5.5))
seen = Set{Symbol}()
for ((kind, seed), r) in sort(collect(res), by = x -> (string(x[1][1]), x[1][2]))
    l = kind in seen ? nothing : lab[kind];  push!(seen, kind)
    axs[1].plot(r.ts, r.KE, color=col[kind], lw=1, label=l)
    axs[2].plot(r.ts, r.Es, color=col[kind], lw=1, label=l)
    axs[3].plot(r.ts, r.Hs, color=col[kind], lw=1, label=l)
    axs[4].plot(r.ts, r.loc, color=col[kind], lw=1, label=l)
end
axs[1].axhline(1, color="k", lw=0.5);  axs[1].set_yscale("log")
axs[1].set_xlabel("t");  axs[1].set_ylabel("Σ|m|² / (2(N−1) V w T)");  axs[1].set_title("kinetic energy / equilibrium")
axs[2].set_xlabel("t");  axs[2].set_ylabel("Σe/Σe₀ − 1");  axs[2].set_title("particle energies Σe")
axs[3].set_xlabel("t");  axs[3].set_ylabel("H/H₀ − 1");  axs[3].set_title("smoothed energy H")
axs[4].set_xlabel("t");  axs[4].set_ylabel("rms change of neighbour distances / a");  axs[4].set_title("local disorder")
for ax in axs; ax.legend(fontsize=8); end
n = length(res) ÷ length(unique(first.(collect(keys(res)))))
suptitle(@sprintf("2D uniform fluid, Hamiltonian step on, h/a = %.1f: %d realizations of each variant", HA, n))
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_hamiltonian_real_ha$(HA)_t$(TE)$(SUF).png")
savefig(out, dpi=110);  println("Saved ", out)
