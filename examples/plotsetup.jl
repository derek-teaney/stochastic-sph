# plotsetup.jl — PyPlot for the example and test scripts, safe at NERSC.
# In a batch job or without a display, matplotlib uses the Agg backend and only writes
# files (no window, so no crash on a compute node).  Figures go to SR_FIGDIR (default:
# examples/), so several configurations can run side by side.  Run the plotting scripts with
# the full environment (--project=., which has PyPlot), not the run-only one (--project=nersc).
(haskey(ENV, "DISPLAY") && !haskey(ENV, "SLURM_JOB_ID")) || (ENV["MPLBACKEND"] = "Agg")
using PyPlot, Random
figpath(name) = (d = get(ENV, "SR_FIGDIR", @__DIR__); mkpath(d); joinpath(d, name))

# Quick-look plots of a particle state (moved from src/setup.jl, which needs no PyPlot).

# 1D line plot of field(p) vs p.x[1] for a 1D simulation.
function plot_line(particles, params;
                   field=p->p.s, ylabel="field", title="1D profile", ax=nothing)
    xs = [pa.x[1] for pa in particles]
    fs = [field(pa) for pa in particles]
    standalone = ax === nothing
    if standalone
        _, ax = subplots(1, 1, figsize=(8, 4))
    end
    ax.plot(xs, fs, "o", markersize=4)
    ax.set_xlim(0, params.L[1])
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    standalone && tight_layout()
    return ax
end

# 2D scatter of particle positions, colored by field(p) (default: p.s).
# Three particles chosen with a local RNG get a circle of radius 2h.
# Pass an existing PyPlot axis via `ax`; omit it for a standalone figure.
function plot_particles(particles, params;
                        field=p->p.s, clabel="field", title="Particles", ax=nothing)
    L = params.L
    h = params.h

    xs = [pa.x[1]   for pa in particles]
    ys = [pa.x[2]   for pa in particles]
    cs = [field(pa) for pa in particles]

    standalone = ax === nothing
    if standalone
        _, ax = subplots(1, 1, figsize=(6, 6))
    end

    sc = ax.scatter(xs, ys, c=cs, s=6, cmap="viridis")
    colorbar(sc, ax=ax, label=clabel)

    # Pick 3 particles with a local RNG so we don't disturb global state.
    rng  = MersenneTwister(42)
    idxs = randperm(rng, length(particles))[1:3]
    for (color, idx) in zip(("C1", "C2", "C3"), idxs)
        cx, cy = particles[idx].x
        θ = range(0.0, 2π, length=200)
        ax.plot(cx .+ 2h .* cos.(θ), cy .+ 2h .* sin.(θ), color=color, linewidth=1.2)
        ax.plot([cx], [cy], "+", color=color, markersize=7, markeredgewidth=1.5)
    end

    ax.set_xlim(0, params.L[1])
    ax.set_ylim(0, params.L[2])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    standalone && tight_layout()
    return ax
end
