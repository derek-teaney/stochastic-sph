# grid_smoke_plot.jl — the stochastic smoke test of the grid step (docs/gridstep.tex): L = 0.6,
# h/a = 2, η/s = 0.2, S0 = 1e6, t = 20, 4 seeds each; grid (SR_GRID=1) and particles, at rest and
# at v0 = 0.8.  Series from the .h5 of examples/equilibrium.jl, averaged over the seeds.
#   julia --project examples/grid_smoke_plot.jl [dir]    (dir default $SCRATCH/sph/grid_smoke)
# → docs/figs/grid_smoke.png, and a table on stdout.
using PyPlot, HDF5, Printf
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "grid_smoke")
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
SETS = [("sm_g_v0", "grid, v0 = 0", "C0", "-"), ("sm_p_v0", "particles, v0 = 0", "C0", "--"),
        ("sm_g_v0.8", "grid, v0 = 0.8", "C3", "-"), ("sm_p_v0.8", "particles, v0 = 0.8", "C3", "--")]
series(f, name) = haskey(f, name) ? read(f[name]) : Float64[]
mean_(xs) = (n = minimum(length, xs);  n == 0 ? Float64[] : sum(x[1:n] for x in xs) / length(xs))

fig, axs = subplots(2, 3, figsize = (16, 8.5))
panels = [("KE", "fluctuation KE / (2VwT)"), ("H", "ΔH/H"), ("acc", "acceptance: stress (thick), residual (thin)"),
          ("grid_dST", "grid: Σ ΔS_T / S (entropy made by the return)"), ("grid_hkm", "grid: power of m at kh > 1 (fraction)"),
          ("vmax", "max |v| over the particles")]
@printf("%-12s %5s %9s %10s %8s %8s %11s %9s\n", "set", "runs", "KE/eq", "ΔH/H", "acc_s", "acc_r", "ΣΔS_T/S", "hk(m)")
for (d, lab, col, ls) in SETS
    fs = filter(f -> endswith(f, ".h5"), readdir(joinpath(DIR, d); join = true))
    isempty(fs) && continue
    D = [h5open(f) do h
            (t = read(h["t"]), KE = read(h["KE"]), H = read(h["H"]), acc_s = series(h, "acc_s"), acc_r = series(h, "acc_r"),
             dST = series(h, "grid_dST"), hkm = series(h, "grid_hkm"), vmax = series(h, "vmax"))
         end for f in fs]
    t = D[1].t;  n = minimum(length(x.KE) for x in D);  t = t[1:n]
    for (ax, (key, ttl)) in zip(vec(permutedims(axs)), panels)
        if key == "KE";  ax.plot(t, mean_([x.KE for x in D]), color = col, ls = ls, label = lab)
        elseif key == "H";  ax.plot(t, mean_([x.H for x in D]), color = col, ls = ls, label = lab)
        elseif key == "acc"
            ax.plot(t, mean_([x.acc_s for x in D])[1:n], color = col, ls = ls, lw = 2, label = lab)
            ax.plot(t, mean_([x.acc_r for x in D])[1:n], color = col, ls = ls, lw = 0.8)
        elseif key == "grid_dST";  y = mean_([x.dST for x in D]);  isempty(y) || ax.plot(t, y[1:n], color = col, ls = ls, label = lab)
        elseif key == "grid_hkm";  y = mean_([x.hkm for x in D]);  isempty(y) || ax.plot(t, y[1:n], color = col, ls = ls, label = lab)
        else;  ax.plot(t, mean_([x.vmax for x in D])[1:n], color = col, ls = ls, label = lab)
        end
        ax.set_title(ttl, fontsize = 10);  ax.set_xlabel("t")
    end
    m(v) = isempty(v) ? NaN : sum(v[end÷2+1:end]) / length(v[end÷2+1:end])
    @printf("%-12s %5d %9.3f %+10.2e %8.3f %8.3f %+11.2e %9.3f\n", d, length(D), m(mean_([x.KE for x in D])),
            mean_([x.H for x in D])[end], m(mean_([x.acc_s for x in D])), m(mean_([x.acc_r for x in D])),
            isempty(D[1].dST) ? NaN : mean_([x.dST for x in D])[end], m(mean_([x.hkm for x in D])))
end
axs[1, 1].legend(fontsize = 8);  axs[1, 1].set_yscale("log")   # v0 = 0.8: ≈ 18 (boost.tex)
tight_layout();  savefig(joinpath(FIGD, "grid_smoke.png"), dpi = 100);  close()
println("wrote grid_smoke.png")
