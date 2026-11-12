# changerep_gaussian.jl — the standard 2D Gaussian test (testgaussian.jl) run
# twice: with the viscous step on the smoothed b (generic_step!), and in the particle
# representation with the exact back map (docs/changerep.tex).  Plots go to
# examples/compare/changerep_{smoothed,particle}/.
#
# Run:  julia --project examples/changerep_gaussian.jl
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
# The "smoothed" runs use generic_step! with the bond form on the smoothed b; the
# particle-representation runs call their steps directly.
ENV["SPH_DISSIPATION"] = "bond"

function particle_step!(ps, params, cl, dt, eos, dotm)
    dissipative_step_changerep_exact!(ps, params, cl, dt / 2, eos)
    leapfrog_step!(ps, params, cl, dt, eos, dotm)
    dissipative_step_changerep_exact!(ps, params, cl, dt / 2, eos)
end

src = read(joinpath(@__DIR__, "testgaussian.jl"), String)
src = replace(src, r"^using PyPlot\n"m => "", r"^include\(.*\)\n"m => "",
              "generic_step!(particles, params, cl, dt, eos, dotm)" => "STEP(particles, params, cl, dt, eos, dotm)",
              "joinpath(@__DIR__, \"" => "joinpath(OUTDIR, \"")
for (tag, f) in (("smoothed", generic_step!), ("particle", particle_step!))
    global STEP = f
    global OUTDIR = joinpath(@__DIR__, "compare", "changerep_" * tag)
    mkpath(OUTDIR)
    println("\n══ viscous step: ", tag, " ══")
    include_string(Main, src)
    close("all")
end
