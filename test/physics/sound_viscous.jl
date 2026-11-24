# sound_viscous.jl — the damped sound wave, η/s = 0.02, at kh = 1.4, 0.7, 0.35, h/a = 3
# (lib/waves.jl).  Hydrodynamics: ω = √(c_s²k² − (Γ_s k²/2)²), Γ = Γ_s k²/2.
# Tolerances from the conformal fluid in d = 2, cs² = 1/2 (dimension.tex; 2026-11-18): ω/ω_hyd − 1 = −3.1e-3,
# Γ/Γ_hyd − 1 = +2.0e-2 at kh = 0.35; the damping error falls by 3.0 and 3.6 between successive kh, with a margin.
# (At cs² = 1/3, d = 3, before dimfix: −1.5e-3 and +7.7e-3; the faster sound wave has the larger error at the same kh.)
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "0")   # tree level: deterministic waves at s_a ≪ 1, below the label boundary of split
include(joinpath(@__DIR__, "..", "..", "src", "load.jl"))
include(joinpath(@__DIR__, "lib", "waves.jl"))
rs = wave_report("sound_viscous", [wave_run(:sound; kh, etas = 0.02) for kh in (1.4, 0.7, 0.35)])
eω = [abs(r.omega / r.omega_hydro - 1) for r in rs];  eΓ = [abs(r.Gamma / r.Gamma_hydro - 1) for r in rs]
check(eω[end] < 5e-3, "frequency at kh = 0.35 within 0.5% ($(round(eω[end], sigdigits=2)))")
check(eΓ[end] < 3e-2, "damping at kh = 0.35 within 3% ($(round(eΓ[end], sigdigits=2)))")
check(eΓ[1] / eΓ[2] > 2 && eΓ[2] / eΓ[3] > 2, "damping error falls by > 2 per halving of kh (ratios $(round.(eΓ[1:2] ./ eΓ[2:3], digits=1)))")
finish("sound_viscous.jl")
