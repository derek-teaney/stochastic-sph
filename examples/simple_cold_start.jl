# simple_cold_start.jl — the most basic stochastic run.
#
# 1. Put particles on a hex lattice, uniform entropy, all at rest (cold start).
# 2. Take steps: ideal leapfrog step, then one stochastic (Metropolis) sweep.
# 3. Plot the particles and their momenta at a few times.
#
# Run:  julia --project examples/simple_cold_start.jl
# Writes simple_cold_start.png next to this script.

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

# ── Parameters (same as teststochastic_run.jl) ──────────────────────────────
h       = 15.0                 # SPH smoothing length
h_over_a = 2.55                # h / lattice spacing
L       = 150.0                # box size
ηs      = 10.0 / sqrt(2.0)    # η/s
n_steps = 200                  # number of time steps
snap_steps = (0, 50, 200)      # steps at which to plot the particles
do_leapfrog = false            # false: particles frozen, viscous/stochastic sweep only

# ── Make the particles: uniform entropy density 1, zero momentum ────────────
eos = ConformalEOS(0.5, T_scale_2d_lq1, ηs, 0.0)     # 2D conformal, cs² = 1/2
params, particles, cl, eos = setup_ic(h, h_over_a, eos,
                                      (x, y) -> 1.0,               # entropy density
                                      (x, y) -> SVector(0.0, 0.0); # momentum density
                                      L, lattice=:hex, d=2)

dt   = compute_dt(particles, params, eos; mode=:ideal, CFL=0.2)
rngs = make_phys_rng(params.N, 1)             # one random stream per particle
dotm = zeros(SVector{2,Float64}, params.N)    # scratch for leapfrog

@printf("N = %d particles,  s per particle = %.1f,  dt = %.3g\n",
        params.N, particles[1].s, dt)

energy(ps) = sum(p.s * p.H_over_S for p in ps)
H0 = energy(particles)

# ── Plot helper: particle positions, arrows = momentum ──────────────────────
function plot_snapshot(ax, particles, step)
    x  = [p.x[1] for p in particles];  y  = [p.x[2] for p in particles]
    mx = [p.m[1] for p in particles];  my = [p.m[2] for p in particles]
    # dots coloured by smoothed entropy density (uniform IC ⇒ should all be equal)
    ax.scatter(x, y, s=6, c=[p.S_smooth for p in particles], cmap="viridis_r", vmin=0.8, vmax=1.0)
    # cell-list cells (widths cl.cell_size = (Lx/ncx, Ly/ncy)); the box edge is drawn in black
    for i in 0:cl.ncx; ax.axvline(i * cl.cell_size[1], color="0.6", lw=0.8); end
    for j in 0:cl.ncy; ax.axhline(j * cl.cell_size[2], color="0.6", lw=0.8); end
    ax.axhline(params.L[2], color="k", lw=1.5)
    ax.quiver(x, y, mx, my, color="C3", angles="xy")
    # circles of radius h (solid) and 2h = kernel support (dashed) around 3 particles
    θ = range(0, 2π, length=200)
    for (c, (cx, cy)) in zip(("C0", "C2", "C1"), ((40.0, 40.0), (100.0, 60.0), (70.0, 110.0)))
        a = argmin([norm(p.x - SVector(cx, cy)) for p in particles])   # nearest particle
        px, py = particles[a].x
        ax.plot(px .+ h .* cos.(θ),  py .+ h .* sin.(θ),  color=c, lw=1.5)
        ax.plot(px .+ 2h .* cos.(θ), py .+ 2h .* sin.(θ), color=c, lw=1.0, ls="--")
        ax.plot([px], [py], "o", color=c, ms=5)
    end
    ax.set_xlim(0, params.L[1]); ax.set_ylim(0, cl.ncy * cl.cell_size[2])
    ax.set_aspect("equal")
    ax.set_title(@sprintf("step %d   rms|m| = %.3g", step,
                          sqrt(mean(norm(p.m)^2 for p in particles))))
end

fig, axs = subplots(1, length(snap_steps) + 1, figsize=(5 * (length(snap_steps) + 1), 5))
panel = 1
plot_snapshot(axs[panel], particles, 0); panel += 1

# ── Time loop ───────────────────────────────────────────────────────────────
rms_m = Float64[];  dH = Float64[]
for step in 1:n_steps
    global panel
    # (a) ideal (non-dissipative) step: moves particles, updates m
    if do_leapfrog
        leapfrog_step!(particles, params, cl, dt, eos, dotm)
        rebuild_cell_list!(cl, particles)
    end
    thermalize!(particles, params, cl, eos)      # smoothed fields, EOS, velocities

    # (b) stochastic sweep: noise + friction, then fold into (m, s)
    particle_sweep!(particles, params, cl, eos, dt, rngs)
    apply_sweep!(particles)

    push!(rms_m, sqrt(mean(norm(p.m)^2 for p in particles)))
    push!(dH, energy(particles) / H0 - 1)

    if step in snap_steps
        plot_snapshot(axs[panel], particles, step); panel += 1
    end
end

# ── Last panel: how rms momentum and energy evolve ──────────────────────────
ax = axs[end]
ax.plot(1:n_steps, rms_m ./ maximum(rms_m), label="rms|m| (scaled to max)")
ax.plot(1:n_steps, dH, label="H/H₀ − 1")
ax.set_xlabel("step"); ax.legend(); ax.set_title("history")

tight_layout()
out = joinpath(@__DIR__, "simple_cold_start.png")
savefig(out, dpi=120)
@printf("final rms|m| = %.3g,  H/H0 - 1 = %+.3g\nSaved %s\n", rms_m[end], dH[end], out)
