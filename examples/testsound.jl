using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

# ── Initialization ────────────────────────────────────────────────────────────

function initialize_sound_wave(h, ha, L, s0, A, cs2v, nx, ny)
    cs    = sqrt(cs2v)
    kx    = 2π / L * nx
    ky    = 2π / L * ny
    k     = sqrt(kx^2 + ky^2)
    knorm = sqrt(Float64(nx^2 + ny^2))
    khat  = SVector(Float64(nx), Float64(ny)) / knorm
    T_period = L / (cs * knorm)

    eos = ConformalEOS(cs2v, 1.0, 0.0, 0.0)

    # Uniform background → read off T_rel_0 for the entropy IC closure.
    _, bg, _, _ = setup_ic(h, ha, eos, (x, y) -> s0, (x, y) -> SVector(0.0, 0.0); L)
    T_rel_0 = bg[1].T * bg[1].γ
    @printf("T_rel_0 = %.8g\n", T_rel_0)

    # Right-moving mode: δS = A/T_rel_0 cos(k⃗·x),  δM⃗ = A cs k̂ cos(k⃗·x)
    params, particles, cl, eos = setup_ic(h, ha, eos,
        (x, y) -> s0 + (A / T_rel_0) * cos(kx*x + ky*y),
        (x, y) -> (A * cs) * khat * cos(kx*x + ky*y); L)

    @printf("k⃗=2π/L·(%d,%d)  k=%.4g  cs=%.4g  T_period=%.4g\n",
            nx, ny, k, cs, T_period)

    return params, particles, cl, eos, kx, ky, khat, k, cs, T_period
end

# ── IC plot ───────────────────────────────────────────────────────────────────

function plot_ic_sound_wave(particles, params; nx, ny, title="Particles")
    fig, (ax1, ax2) = subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title)
    plot_particles(particles, params;
                  field=p -> p.M_smooth[1], clabel="Mx density",
                  title="Mx density", ax=ax1)
    plot_particles(particles, params;
                  field=p -> p.M_smooth[2], clabel="My density",
                  title="My density", ax=ax2)
    tight_layout()
    return fig
end

# ── Time evolution ────────────────────────────────────────────────────────────

function run_sound_wave!(particles, params, cl, eos, t_end; CFL=0.2)
    dt      = compute_dt(particles, params, eos; mode=:ideal, CFL)
    n_steps = ceil(Int, t_end / dt)
    dt      = t_end / n_steps
    @printf("dt=%.4g  n_steps=%d\n", dt, n_steps)

    s_init = sum(pa.s for pa in particles)
    dotm   = zeros(SVector{2,Float64}, params.N)
    for step in 1:n_steps
        leapfrog_step!(particles, params, cl, dt, eos, dotm)
        step % 20 == 0 && @printf("  step %d / %d  t=%.4g\n", step, n_steps, step*dt)
    end
    s_final = sum(pa.s for pa in particles)

    return dt, n_steps, s_init, s_final
end

# ── Final comparison plot ─────────────────────────────────────────────────────

function plot_final_sound_wave(particles, params, cl,
                                t_end, A, cs, k, kx, ky, khat;
                                nx, ny, T_period)
    xs_f    = [pa.x[1] for pa in particles]
    ys_f    = [pa.x[2] for pa in particles]
    Mx_sph  = [pa.M_smooth[1] for pa in particles]
    Mx_anal_pts = [A * cs * khat[1] * cos(kx*x + ky*y - cs*k*t_end)
                   for (x, y) in zip(xs_f, ys_f)]
    rms = sqrt(sum((s - a)^2 for (s, a) in zip(Mx_sph, Mx_anal_pts)) / length(Mx_sph))
    @printf("Mx residual: rms=%.3g  amplitude=%.3g\n", rms, A * cs * abs(khat[1]))

    fig, ax = subplots(1, 1, figsize=(7, 4.5))
    fig.suptitle("Sound wave  nx=$nx ny=$ny  t=$(round(t_end,digits=3)) = $(round(t_end/T_period,digits=2)) T_period  rms=$(round(rms,sigdigits=3))")

    xs_sl, fs_sl = plot_slice(particles, params, cl;
                               field=p -> p.M_smooth[1], dim=:x)
    ax.plot(xs_sl, fs_sl, color="C0", label="SPH")
    ax.plot(xs_sl,
            [A * cs * khat[1] * cos(kx*x + ky*(params.L[2]/2) - cs*k*t_end) for x in xs_sl],
            "--", color="C1", linewidth=1.5, label="analytic (y=L/2)")
    ax.set_xlabel("x"); ax.set_ylabel("Mx density")
    ax.set_title("Final: SPH vs analytic")
    ax.legend(fontsize=8)

    tight_layout()
    return fig
end

# ── Main ──────────────────────────────────────────────────────────────────────

h            = 0.1
ha           = 1.5
L            = 1.0
s0           = 2.0
A            = 0.01        # perturbation amplitude (keep ≪ s0)
cs2v         = 1.0 / 3.0
nx           = 2
ny           = 1
params, particles, cl, eos, kx, ky, khat, k, cs, T_period =
    initialize_sound_wave(h, ha, L, s0, A, cs2v, nx, ny)

fig_ic = plot_ic_sound_wave(particles, params; nx, ny, title = "Initial Particles")
savefig(joinpath(@__DIR__, "testsound_ic.png"), dpi=150)
println("Saved testsound_ic.png")

t_end = T_period
@printf("t_end=%.4g (= 1 T_period)\n", t_end)

dt, n_steps, s_init, s_final =
    run_sound_wave!(particles, params, cl, eos, t_end)
@printf("Entropy: Σs_0=%.10g  Σs_f=%.10g  rel.err=%.3g\n",
        s_init, s_final, abs(s_final - s_init) / s_init)

fig_final = plot_final_sound_wave(particles, params, cl,
                                   t_end, A, cs, k, kx, ky, khat;
                                   nx, ny, T_period)
savefig(joinpath(@__DIR__, "testsound_final.png"), dpi=150)
println("Saved testsound_final.png")

fig_fc = plot_ic_sound_wave(particles, params; nx, ny, title = "Final Particles")
savefig(joinpath(@__DIR__, "testsound_fc.png"), dpi=150)
println("Saved testsound_fc.png")

show()
