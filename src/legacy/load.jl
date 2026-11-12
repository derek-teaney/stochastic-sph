# legacy/load.jl — superseded code, kept so the older notes' figures can be made again.
# Include after src/load.jl:
#     include(joinpath(@__DIR__, "..", "src", "legacy", "load.jl"))
#   two_level.jl       the two-level strain form of the viscous force (sph2.tex)
#   particle_sweep.jl  the old stochastic step: particle sweep with the noise tensor
for f in ("two_level", "particle_sweep")
    include(joinpath(@__DIR__, "$f.jl"))
end
