# energy_time.jl — the energy of the equilibrium runs against time: ΔH/H = H(t)/H(0) − 1 (H the total energy Σ_a s_a H_a/S_a,
# recorded with the Fourier modes) and the momentum fluctuations KE/(2(N−1)VwT) (1 in equilibrium at rest), each run thin, the
# mean over runs thick.  One column per (s_a, v0); the sets of a column overlaid.
#   julia --project examples/energy_time.jl <dir> <tag> set1,set2,... [set1,... for the next column] ...
# e.g.  examples/energy_time.jl $SCRATCH/sph/hastings_bath hastings_bath hs16_tr_v0,hs16_sp_v0 hs16_tr_v0.8,hs16_sp_v0.8 ...
# Writes docs/figs/energy_time_<tag>.png.
using PyPlot, Printf, HDF5, Serialization, StaticArrays, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const DIR = ARGS[1];  const TAG = ARGS[2];  const COLS = [split(c, ",") for c in ARGS[3:end]]
fig, axs = subplots(2, length(COLS), figsize = (4.6 * length(COLS), 7), squeeze = false)
for (ci, sets) in enumerate(COLS), (si, set) in enumerate(sets)
    d = joinpath(DIR, set);  isdir(d) || continue
    fs = [joinpath(d, f) for f in readdir(d) if endswith(f, ".h5")]
    rs = [read_equilibrium(f) for f in fs];  rs = filter(r -> get(r.meta, :complete, true) && length(r.H) > 10, rs)
    isempty(rs) && continue
    m = rs[1].meta;  sa = m.S0 * m.a^2 * sqrt(3) / 2;  col = "C$(si - 1)"
    n = minimum(length(r.H) for r in rs);  t = rs[1].rec.t[1:n]
    for (row, key) in ((1, :H), (2, :KE))
        ax = axs[row, ci];  ys = [getfield(r, key)[1:n] for r in rs]
        for y in ys;  ax.plot(t, y, color = col, lw = 0.3, alpha = 0.35);  end
        ax.plot(t, sum(ys) / length(ys), color = col, lw = 2,
                label = @sprintf("%s (%d runs, η/s=%.2g%s)", set, length(rs), m.etas, get(m, :novisc, false) ? ", no viscous step" : ""))
    end
    axs[1, ci].set_title(@sprintf("s_a = %.0f, v0 = %.1f, T_rel(set-up) = %.1f, c_s² = %.2f", sa, get(m, :boost, 0.0), m.Trel, m.cs2), fontsize = 10)
    @printf("%-16s s_a=%.0f  ΔH/H at t = %.0f: mean %+.2e, runs %+.2e … %+.2e;  KE(fluct) second half %.3f\n", set, sa, t[end],
            sum(r.H[n] for r in rs) / length(rs), minimum(r.H[n] for r in rs), maximum(r.H[n] for r in rs),
            sum(sum(r.KE[n÷2:n]) / (n - n÷2 + 1) for r in rs) / length(rs))
end
for ci in eachindex(COLS)
    axs[1, ci].axhline(0, color = "k", lw = 0.5);  axs[1, ci].set_ylabel("ΔH/H = H(t)/H(0) − 1");  axs[1, ci].legend(fontsize = 7)
    axs[2, ci].set_ylabel("KE(fluct) / 2(N−1)VwT");  axs[2, ci].set_xlabel("t");  axs[2, ci].legend(fontsize = 7)
end
fig.text(0.995, 0.003, "runs: " * DIR, ha = "right", va = "bottom", fontsize = 8, color = "0.3")
tight_layout(rect = (0, 0.02, 1, 1));  out = joinpath(@__DIR__, "..", "docs", "figs", "energy_time_$(TAG).png")
savefig(out, dpi = 100);  println("written ", out)
