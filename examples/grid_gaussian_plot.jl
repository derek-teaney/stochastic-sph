# grid_gaussian_plot.jl — figures of examples/grid_gaussian.jl.
#   julia --project examples/grid_gaussian_plot.jl [dir]     (dir default $SCRATCH/sph/grid/gaussian)
# docs/figs/grid_gaussian_slices.png: rest-frame energy density along x and y through the centre at
#   t = 0.55, 1.1, 2.2, grid (η/s = 0.002), particles (η/s = 0.002), grid without viscosity.
# docs/figs/grid_gaussian_maps.png: the energy density on the sites/particles at t = 2.2, and
#   ΔH/H, ΔS/S against t.
using PyPlot, Serialization, Printf, StaticArrays
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "grid", "gaussian")
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
res = deserialize(joinpath(DIR, "grid_gaussian.jls"))
lab(r) = r.scheme == "grid" ? (r.etas > 0 ? "grid, η/s = $(r.etas)" : "grid, no viscosity") : "particles, η/s = $(r.etas)"
sty = Dict(1 => ("C0", "-", 2.0), 2 => ("C1", "--", 1.5), 3 => ("C3", ":", 1.5))

fig, axs = subplots(2, 3, figsize = (16, 8.5))
for (j, t) in enumerate((0.55, 1.1, 2.2)), (i, dim) in enumerate((:x, :y))
    ax = axs[i, j]
    for (k, r) in enumerate(res)
        haskey(r.snaps, t) || continue
        s = r.snaps[t];  c, ls, lw = sty[k]
        ax.plot(dim === :x ? s.xs : s.ys, dim === :x ? s.ex : s.ey, color = c, ls = ls, lw = lw, label = lab(r))
    end
    ax.set_title(@sprintf("t = %.2f, slice along %s through the centre", t, dim), fontsize = 10)
    ax.set_xlabel(string(dim));  j == 1 && ax.set_ylabel("e (rest frame)")
    i == 1 && j == 1 && ax.legend(fontsize = 8)
end
tight_layout();  savefig(joinpath(FIGD, "grid_gaussian_slices.png"), dpi = 100);  close()

fig, axs = subplots(1, 4, figsize = (19, 4.6))
emin = minimum(minimum(r.snaps[2.2].e) for r in res if haskey(r.snaps, 2.2))
emax = maximum(maximum(r.snaps[2.2].e) for r in res if haskey(r.snaps, 2.2))
for (k, r) in enumerate(res)
    ax = axs[k];  haskey(r.snaps, 2.2) || continue
    s = r.snaps[2.2]
    sc = ax.scatter([x[1] for x in s.x], [x[2] for x in s.x], c = s.e, s = 1.5, cmap = "viridis", vmin = emin, vmax = emax)
    ax.set_aspect("equal");  ax.set_title(lab(r) * ", t = 2.2", fontsize = 10);  colorbar(sc, ax = ax, fraction = 0.046)
end
ax = axs[4]
for (k, r) in enumerate(res)
    c, ls, _ = sty[k]
    ax.plot(r.ts, r.Ht, color = c, ls = ls, label = "ΔH/H, " * lab(r));  ax.plot(r.ts, r.St, color = c, ls = ls, alpha = 0.5, lw = 3, label = "ΔS/S, " * lab(r))
end
ax.axhline(0, color = "k", lw = 0.5);  ax.set_xlabel("t");  ax.legend(fontsize = 7);  ax.set_title("energy and entropy", fontsize = 10)
tight_layout();  savefig(joinpath(FIGD, "grid_gaussian_maps.png"), dpi = 100);  close()
println("wrote grid_gaussian_slices.png, grid_gaussian_maps.png")
