# campaign_compare_plot.jl — equilibrium ensembles of two campaigns side by side (e.g.
# `first`: thermal start, old code; `second`: hot start, fast code), one figure per box.
#   C_T(k, τ) to long lags in the smallest shells (the transverse oscillation), the
#   statics C_T(k,0) and C_L(k,0) against kh, and ΔH/H(t) and KE/eq(t) (mean and spread).
#
# Run:  julia --project examples/campaign_compare_plot.jl <run> [campaigns...]
#   e.g. examples/campaign_compare_plot.jl eqL1.2 first second
#   reads $SR_DATADIR/<campaign>/<run>/equilibrium_*.h5 (SR_DATADIR default $SCRATCH/sph)
#   → docs/figs/compare_<run>.png
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const ROOT  = get(ENV, "SR_DATADIR", joinpath(get(ENV, "SCRATCH", "."), "sph"))
const RUN   = ARGS[1]
const CAMPS = length(ARGS) > 1 ? ARGS[2:end] : ["first", "second"]
const TBURN = 2.0;  const MAXLAG = 40.0

function load(camp)
    d = joinpath(ROOT, camp, RUN)
    fs = filter(f -> endswith(f, ".h5") && startswith(f, "equilibrium_"), readdir(d))
    runs = [read_equilibrium(joinpath(d, f)) for f in fs]
    filter(r -> !haskey(r.meta, :complete) || r.meta.complete, runs)
end

fig, axs = subplots(2, 2, figsize = (15, 10));  axs = vec(permutedims(axs))
styles = ["-", "--", ":"];  rows = String[]
for (ic, camp) in enumerate(CAMPS)
    runs = load(camp);  isempty(runs) && continue
    m = runs[1].meta;  norm0 = m.N * m.V * m.w * m.T;  ν = m.etas / m.Trel
    start = haskey(m, :start) ? m.start : "thermal"
    per = [shell_correlators(r.rec, :MT; t0 = TBURN, maxlag_t = MAXLAG) for r in runs]
    ks, τ = per[1][1], per[1][2]
    C = mean(cat([p[3] for p in per]...; dims = 3); dims = 3)[:, :, 1] ./ norm0
    E = std(cat([p[3] for p in per]...; dims = 3); dims = 3)[:, :, 1] ./ norm0 ./ sqrt(length(runs))
    perL = [shell_correlators(r.rec, :ML; t0 = TBURN, maxlag_t = 0.1) for r in runs]
    CL0 = mean(cat([p[3][:, 1] for p in perL]...; dims = 2); dims = 2)[:, 1] ./ norm0
    lab = @sprintf("%s (%s start, %d runs)", camp, start, length(runs))
    for j in 1:3
        c = "C$(j-1)"
        axs[1].fill_between(τ, C[j, :] .- E[j, :], C[j, :] .+ E[j, :], color = c, alpha = 0.2)
        axs[1].plot(τ, C[j, :], color = c, ls = styles[ic], lw = 1.2,
                    label = ic == 1 ? @sprintf("kh = %.2f", ks[j] * m.h) : nothing)
        ic == 1 && axs[1].plot(τ, exp.(-ν * ks[j]^2 .* τ), color = c, lw = 0.6, alpha = 0.6)
    end
    axs[2].plot(ks .* m.h, C[:, 1], "s", ls = styles[ic], color = "C0", ms = 4, label = "C_T(k,0), " * lab)
    axs[2].plot(ks .* m.h, CL0, "^", ls = styles[ic], color = "C1", ms = 4, label = "C_L(k,0), " * lab)
    t = runs[1].rec.t;  n = minimum(length(r.H) for r in runs)
    Hm = reduce(hcat, [r.H[1:n] for r in runs]);  Km = reduce(hcat, [r.KE[1:n] for r in runs])
    for (ax, Y) in ((axs[3], Hm), (axs[4], Km))
        ax.fill_between(t[1:n], minimum(Y; dims = 2)[:], maximum(Y; dims = 2)[:], color = "C$(ic-1)", alpha = 0.2)
        ax.plot(t[1:n], mean(Y; dims = 2)[:], color = "C$(ic-1)", lw = 1.2, label = lab)
    end
    i5 = findfirst(>=(5.0), τ)
    push!(rows, @sprintf("%-8s %-7s %2d runs  ΔH/H(end) %.3f ± %.3f  KE/eq(end) %.3f  C_T(0) kh=%.2f: %.3f  C_T(τ=%.0f): %+.3f %+.3f %+.3f",
                         camp, start, length(runs), mean(Hm[end, :]), std(Hm[end, :]), mean(Km[end, :]),
                         ks[1] * m.h, C[1, 1], τ[i5], C[1, i5], C[2, i5], C[3, i5]))
end
axs[1].axhline(0, color = "k", lw = 0.5);  axs[1].set_xlabel("τ");  axs[1].set_ylabel("C_T(k, τ) / (N V w T)")
axs[1].set_title("transverse correlator, smallest shells (thin lines: e^{−νk²τ}; solid / dashed: campaigns)")
axs[1].legend(fontsize = 8)
axs[2].axhline(1, color = "k", lw = 0.5);  axs[2].set_xlabel("k h");  axs[2].set_title("statics");  axs[2].legend(fontsize = 7)
axs[3].set_xlabel("t");  axs[3].set_title("smoothed energy H/H₀ − 1");  axs[3].legend(fontsize = 8)
axs[4].set_xlabel("t");  axs[4].set_title("kinetic energy / equilibrium");  axs[4].axhline(1, color = "k", lw = 0.5)
suptitle("$RUN: " * join(CAMPS, " vs "))
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "compare_$(RUN).png")
savefig(out, dpi = 110);  println("Saved ", out)
foreach(println, rows)
