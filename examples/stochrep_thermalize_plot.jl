# stochrep_thermalize_plot.jl — plots for stochrep_thermalize.jl (reads the saved data).
#
# Run:  julia --project examples/stochrep_thermalize_plot.jl [1|2]
#   → docs/figs/stochrep_thermalize_1d.png (or _2d.png)
using PyPlot, Printf, Serialization, StaticArrays, LinearAlgebra
const DIM  = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
const LAMS = (1.0, 4.0, Inf)
const COL  = Dict(1.0 => "C0", 4.0 => "C1", Inf => "C3")
lab(λ) = λ == 1 ? "λ = 1 (detailed balance)" : isinf(λ) ? "λ = ∞ (deterministic residual)" : "λ = $(round(Int, λ)) (cold residual)"

d = deserialize(joinpath(@__DIR__, "..", "docs", "figs", "stochrep_thermalize_$(DIM)d.jls"))
ks, eq, dt, snaps, a = d[:ks], d[:eq], d[:dt], d[:snaps], d[:a]
pred = d[:pred]
oneD = DIM == 1
pols = oneD ? (:L,) : (:L, :T)
x = ks .* a ./ π                                   # k a / π
Teff(λ, s) = pred[s].R .+ (1 .- pred[s].R) ./ λ
ens(λ, f) = sum(f(r) for r in d[λ]) ./ length(d[λ])
spec_at(λ, n, i) = ens(λ, r -> r.Ssnap[n][i]) ./ eq
late(λ, i) = ens(λ, r -> r.avg[i]) ./ eq

# 2D: bin |k|
if oneD
    bins = nothing;  binx = x
    binavg(v) = v
else
    edges = range(0, maximum(x) * 1.0001, length=26)
    idx = [searchsortedlast(edges, xi) for xi in x]
    used = sort(unique(idx))
    binx = [sum(x[idx .== b]) / count(==(b), idx) for b in used]
    binavg(v) = [sum(v[idx .== b]) / count(==(b), idx) for b in used]
end

fig, axs = subplots(2, 2, figsize=(14, 10))
# (a) approach to equilibrium, λ = 1
ax = axs[1, 1]
sel = snaps[round.(Int, range(1, length(snaps), length=8))]
cm = get_cmap("viridis")
for (j, n) in enumerate(sel)
    for (i, s) in enumerate(pols)
        ax.plot(binx, binavg(spec_at(1.0, n, i)), i == 1 ? "-" : "--", color=cm((j - 1) / (length(sel) - 1)),
                label=i == 1 ? @sprintf("t = %.3g", n * dt) : nothing)
    end
end
ax.axhline(1, color="k", lw=0.8)
ax.set_xlabel("k a / π");  ax.set_ylabel("⟨|m_k|²⟩ / (N V w T)")
ax.set_title("λ = 1: spectrum during thermalization" * (oneD ? "" : "  (— longitudinal, -- transverse)"))
ax.legend(fontsize=8)
# (b) late-time spectra vs T_eff(k)/T
ax = axs[1, 2]
for λ in LAMS, (i, s) in enumerate(pols)
    ax.plot(binx, binavg(late(λ, i)), (i == 1 ? "o" : "s"), ms=4, color=COL[λ], mfc=(i == 1 ? COL[λ] : "none"),
            label=i == 1 ? lab(λ) : nothing)
    ax.plot(binx, binavg(Teff(λ, s)), i == 1 ? "-" : "--", color=COL[λ], lw=1)
end
ax.set_ylim(0, 1.3);  ax.axhline(1, color="k", lw=0.5)
ax.set_xlabel("k a / π");  ax.set_ylabel("late-time ⟨|m_k|²⟩ / (N V w T)")
ax.set_title("equilibrium spectrum (points) vs T_eff(k)/T (lines)")
ax.legend(fontsize=8)
# (c) mode histories, λ = 1, vs 1 − e^{−2Γt}
ax = axs[2, 1]
ts = snaps .* dt
modes = oneD ? [1, 3, 8, 20, length(ks)] : begin
    o = sortperm(ks); o[round.(Int, range(1, length(o), length=6))][1:5]
end
for (j, i) in enumerate(modes)
    s = :L
    hist = [spec_at(1.0, n, 1)[i] for n in snaps]
    ax.semilogx(ts, hist, "o", ms=3, color="C$j")
    tt = exp10.(range(log10(dt), log10(ts[end]), length=200))
    ax.semilogx(tt, Teff(1.0, s)[i] .* (1 .- exp.(-2 .* pred[s].Γ[i] .* tt)), "-", color="C$j",
                label=@sprintf("k a/π = %.2f", x[i]))
end
ax.set_xlabel("t");  ax.set_ylabel("⟨|m_k|²⟩ / (N V w T)" * (oneD ? "" : " (longitudinal)"))
ax.set_title("λ = 1: single modes (points) vs 1 − exp(−2Γ_k t) (lines)")
ax.legend(fontsize=8)
# (d) kinetic energy
ax = axs[2, 2]
for λ in LAMS
    KE = ens(λ, r -> r.KE)
    t = (1:length(KE)) .* dt
    ax.semilogx(t, KE, color=COL[λ], label=lab(λ) * @sprintf("  acc %.2f / %.2f",
                ens(λ, r -> r.acc[1]), ens(λ, r -> r.acc[2])))
end
ax.axhline(1, color="k", lw=0.5)
ax.set_xlabel("t");  ax.set_ylabel("Σ_a |m_a|² / (d N V w T)")
ax.set_title("kinetic energy (legend: acceptance stress / residual)")
ax.legend(fontsize=8)
suptitle(@sprintf("%dD frozen lattice, constant e_a, m_a = 0 at t = 0  (Δt = %.2e, h/a = %.1f)", DIM, dt, d[:h] / a))
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_thermalize_$(DIM)d.png")
savefig(out, dpi=120);  println("Saved ", out)
