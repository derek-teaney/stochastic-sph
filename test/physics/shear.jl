# shear.jl — the shear wave, η/s = 0.02, at kh = 1.4, 0.7, 0.35, h/a = 3 (lib/waves.jl).
# Hydrodynamics: Γ = (η/s) k²/T_rel.  The SPH wave damps faster, by 21%, 8.5% and 3.4% on
# 2026-11-12 (the error falls by 2.5 per halving of kh); the tolerances hold that, with a
# margin.
include(joinpath(@__DIR__, "..", "..", "src", "load.jl"))
include(joinpath(@__DIR__, "lib", "waves.jl"))
rs = wave_report("shear", [wave_run(:shear; kh, etas = 0.02) for kh in (1.4, 0.7, 0.35)])
eΓ = [abs(r.Gamma / r.Gamma_hydro - 1) for r in rs]
check(eΓ[end] < 5e-2, "damping at kh = 0.35 within 5% ($(round(eΓ[end], sigdigits=2)))")
check(eΓ[1] / eΓ[2] > 2 && eΓ[2] / eΓ[3] > 2, "damping error falls by > 2 per halving of kh (ratios $(round.(eΓ[1:2] ./ eΓ[2:3], digits=1)))")
finish("shear.jl")
