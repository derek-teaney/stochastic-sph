# dt_scan.jl — does the sweep-only run converge as dt → 0?
#
# Particles frozen on the lattice (no leapfrog), cold start, stochastic sweep only.
# Same physical time for every dt.  Plots rms|m| and H/H0 − 1 vs t.
#
# Run:  julia --project examples/dt_scan.jl     → writes dt_scan.png

using PyPlot
using StaticArrays
using Random
using LinearAlgebra
using Statistics
using Printf
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
include(joinpath(@__DIR__, "..", "src", "stochastic.jl"))

t_max     = 848.0                       # physical run time (= 200 steps at the base dt)
dt_factors = (1.0, 0.5, 0.25, 0.125)    # dt = factor × CFL dt

energy(ps) = sum(p.s * p.H_over_S for p in ps)

function run_sweep_only(dt_factor; seed=1)
    eos = ConformalEOS(0.5, T_scale_2d_lq1, 10.0 / sqrt(2.0), 0.0)
    params, particles, cl, eos = setup_ic(15.0, 2.55, eos,
                                          (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                          L=150.0, lattice=:hex, d=2)
    dt   = dt_factor * compute_dt(particles, params, eos; mode=:ideal, CFL=0.2)
    rngs = make_phys_rng(params.N, seed)
    H0   = energy(particles)

    t = Float64[];  rms_m = Float64[];  dH = Float64[]
    for step in 1:round(Int, t_max / dt)
        thermalize!(particles, params, cl, eos)           # refresh fields (positions fixed)
        particle_sweep!(particles, params, cl, eos, dt, rngs)
        apply_sweep!(particles)

        push!(t, step * dt)
        push!(rms_m, sqrt(mean(norm(p.m)^2 for p in particles)))
        push!(dH, energy(particles) / H0 - 1)   # H_over_S is from the last refresh
    end
    return dt, t, rms_m, dH
end

fig, (ax1, ax2) = subplots(1, 2, figsize=(12, 4.5))
for (i, f) in enumerate(dt_factors)
    dt, t, rms_m, dH = run_sweep_only(f)
    @printf("dt = %.3f   final rms|m| = %.2f   H/H0-1 = %+.4f\n", dt, rms_m[end], dH[end])
    ax1.plot(t, rms_m, color="C$(i-1)", lw=1.2, label=@sprintf("Δt = %.2f", dt))
    ax2.plot(t, dH,    color="C$(i-1)", lw=1.2, label=@sprintf("Δt = %.2f", dt))
end

ax1.set_xlabel("t"); ax1.set_ylabel("rms |m|")
ax1.set_title("momentum keeps growing — same curve for every Δt")
ax1.legend()
ax2.set_xlabel("t"); ax2.set_ylabel("H / H₀ − 1")
ax2.set_title("energy grows linearly — same rate for every Δt")
ax2.legend()

tight_layout()
out = joinpath(@__DIR__, "dt_scan.png")
savefig(out, dpi=120)
println("Saved ", out)
