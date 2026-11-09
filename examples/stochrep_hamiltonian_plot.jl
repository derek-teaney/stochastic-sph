# stochrep_hamiltonian_plot.jl — plots for stochrep_hamiltonian.jl.
# Run:  julia --project examples/stochrep_hamiltonian_plot.jl [h/a]  → docs/figs/stochrep_hamiltonian_ha<h/a>.png
using PyPlot, Printf, Serialization, StaticArrays
const HA = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 2.0
const TE = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 15
const SUF = length(ARGS) > 2 ? "_" * ARGS[3] : ""            # e.g. star
res = deserialize(joinpath(@__DIR__, "..", "docs", "figs", "stochrep_hamiltonian_ha$(HA)_t$(TE)$(SUF).jls"))
lab = Dict(:full => "stress + residual (λ = 1)", :det => "stress + deterministic residual (λ = ∞)",
           :stress => "stress only")
col = Dict(:full => "C0", :det => "C2", :stress => "C3")
fig, axs = subplots(2, 3, figsize=(20, 10))
for kind in filter(k -> haskey(res, k), [:full, :det, :stress])
    r = res[kind];  kh = r.ks .* r.h
    edges = range(0, maximum(kh) * 1.0001, length=26);  idx = [searchsortedlast(edges, x) for x in kh]
    used = sort(unique(idx));  bx = [sum(kh[idx .== b]) / count(==(b), idx) for b in used]
    bav(v) = [sum(v[idx .== b]) / count(==(b), idx) for b in used]
    axs[1, 1].plot(bx, bav(r.late[1]), "o-", ms=3, color=col[kind], label=lab[kind] * ", longitudinal")
    axs[1, 1].plot(bx, bav(r.late[2]), "s--", ms=3, mfc="none", color=col[kind], label=lab[kind] * ", transverse")
    axs[1, 2].plot(r.ts, r.Hs, color=col[kind], label=lab[kind])
    axs[1, 3].plot(r.ts, r.Es, color=col[kind], label=lab[kind])
    axs[2, 1].semilogx(r.ts, r.KE, color=col[kind], label=lab[kind])
    axs[2, 2].plot(r.ts, r.dis, "--", color=col[kind], lw=1)
    axs[2, 2].plot(r.ts, r.loc, color=col[kind], label=lab[kind])
    axs[2, 3].plot(r.ts, r.acc, color=col[kind], label=lab[kind])
end
axs[1, 1].axhline(1, color="k", lw=0.5);  axs[1, 1].set_xlabel("k h");  axs[1, 1].set_ylabel("⟨|m_k|²⟩ / (N V w T)")
axs[1, 1].set_title("spectrum, second half of the run");  axs[1, 1].legend(fontsize=7)
axs[1, 2].set_xlabel("t");  axs[1, 2].set_ylabel("H/H₀ − 1");  axs[1, 2].set_title("smoothed energy H")
axs[1, 3].set_xlabel("t");  axs[1, 3].set_ylabel("Σe/Σe₀ − 1");  axs[1, 3].set_title("particle energies Σe")
axs[2, 1].axhline(1, color="k", lw=0.5);  axs[2, 1].set_xlabel("t");  axs[2, 1].set_ylabel("Σ|m|² / (2(N−1) V w T)")
axs[2, 1].set_title("kinetic energy / equilibrium")
axs[2, 2].set_xlabel("t");  axs[2, 2].set_ylabel("/ a")
axs[2, 2].set_title("local disorder: rms change of neighbour distances (solid);\nrms displacement from the initial site (dashed)", fontsize=10)
axs[2, 3].set_xlabel("t");  axs[2, 3].set_ylabel("acceptance");  axs[2, 3].set_title("stress-move acceptance")
for ax in axs[[2, 3, 4, 5, 6]]; ax.legend(fontsize=7); end
r = first(values(res))
suptitle(@sprintf("2D uniform fluid at rest, Hamiltonian step on, from m = 0:  h/a = %.1f, N = %d, Δt = %.2e", HA, r.N, r.dt))
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_hamiltonian_ha$(HA)_t$(TE)$(SUF).png")
savefig(out, dpi=110);  println("Saved ", out)
