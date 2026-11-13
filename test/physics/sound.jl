# sound.jl — the ideal sound wave (η = 0) at kh = 1.4, 0.7, 0.35, h/a = 3 (lib/waves.jl).
# Hydrodynamics: ω = c_s k, no damping.  The SPH wave is slow by a dispersion error that
# falls faster than (kh)², and damps numerically at a rate that falls faster still.
# Tolerances are set from the results of 2026-11-12 (c/c_s − 1 = −1.0e-3, Γ/ω = 1.6e-5 at
# kh = 0.35; error ratios 5.3 and 13 between successive kh), with a margin.
include(joinpath(@__DIR__, "..", "..", "src", "load.jl"))
include(joinpath(@__DIR__, "lib", "waves.jl"))
rs = wave_report("sound", [wave_run(:sound; kh, etas = 0.0) for kh in (1.4, 0.7, 0.35)])
err = [abs(r.omega / r.omega_hydro - 1) for r in rs]
check(err[end] < 3e-3, "speed at kh = 0.35 within 0.3% of c_s ($(round(err[end], sigdigits=2)))")
check(rs[end].Gamma / rs[end].omega < 1e-4, "numerical damping Γ/ω < 1e-4 at kh = 0.35 ($(round(rs[end].Gamma / rs[end].omega, sigdigits=2)))")
check(err[1] / err[2] > 3.5 && err[2] / err[3] > 3.5, "speed error falls at least like (kh)² (ratios $(round.(err[1:2] ./ err[2:3], digits=1)))")
finish("sound.jl")
