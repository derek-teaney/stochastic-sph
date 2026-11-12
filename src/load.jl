# load.jl — the whole library, in order.  Scripts do
#     include(joinpath(@__DIR__, "..", "src", "load.jl"))
# Superseded code (src/legacy/) is not loaded here; include it after this file.
for f in ("sph", "celllist", "ideal", "pointwise", "viscous", "setup", "rng", "stochastic",
          "measure", "monitor", "runinfo")
    include(joinpath(@__DIR__, "$f.jl"))
end
