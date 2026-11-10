# plotsetup.jl — PyPlot for the example and test scripts, safe at NERSC.
# In a batch job or without a display, matplotlib uses the Agg backend and only writes
# files (no window, so no crash on a compute node).  Figures go to SR_FIGDIR (default:
# examples/), so several configurations can run side by side.  Run the plotting scripts with
# the full environment (--project=., which has PyPlot), not the run-only one (--project=nersc).
(haskey(ENV, "DISPLAY") && !haskey(ENV, "SLURM_JOB_ID")) || (ENV["MPLBACKEND"] = "Agg")
using PyPlot
figpath(name) = (d = get(ENV, "SR_FIGDIR", @__DIR__); mkpath(d); joinpath(d, name))
