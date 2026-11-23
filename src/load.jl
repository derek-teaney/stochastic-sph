# load.jl — the whole library, in order.  Scripts do
#     include(joinpath(@__DIR__, "..", "src", "load.jl"))
# Superseded code (src/legacy/) is not loaded here; include it after this file.
for f in ("settings", "sph", "celllist", "ideal", "pointwise", "renorm", "viscous", "setup", "rng", "stochastic", "grid",
          "measure", "hmc", "monitor", "runinfo")
    include(joinpath(@__DIR__, "$f.jl"))
end
