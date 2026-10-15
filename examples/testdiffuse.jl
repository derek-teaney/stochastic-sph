using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

# ── IC test ───────────────────────────────────────────────────────────────────
# Compute the instantaneous SPH force density ∂_t M^y(x) and compare to the
# analytic shear-wave decay rate at t=0.
#
#   ∂_t M^y ≈ -(η/s_rel) / T_rel * k² M_amp sin(kx)
#
function run_diffusion_test_ic(params, particles, cl, eos, k, M_amp)
    L      = params.L
    h      = params.h
    ncx    = cl.ncx;  ncy = cl.ncy
    η_by_s = eta_by_s(eos)

    dotm = zeros(SVector{2,Float64}, params.N)
    dissipative_force!(dotm, particles, params, cl, eos)

    # SPH kernel estimate of ∂_t M^y on a 1D grid at y = L[2]/2
    Ng = 400
    xs_grid    = LinRange(0.0, L[1], Ng)
    dMydt_grid = zeros(Ng)
    for (i, xg) in enumerate(xs_grid)
        ix, iy = cell_index(SVector(xg, L[2]/2), cl)
        for ddx in -1:1, ddy in neighbor_range(ncy)
            jx = mod(ix + ddx - 1, ncx) + 1
            jy = mod(iy + ddy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                xygrid = SVector(xg, L[2]/2)
                rvec = minimum_image(xygrid - particles[b].x, L)
                r = norm(rvec)
                r >= 2h && continue
                dMydt_grid[i] += dotm[b][2] * wendland_c2(r, h, params.kernel)
            end
        end
    end

    xs         = [pa.x[1]                              for pa in particles]
    My_density = [pa.M_smooth[2]                       for pa in particles]
    dotx_y     = [pa.dotx[2]                            for pa in particles]
    v_y        = [pa.v_over_S[2] * pa.S_smooth          for pa in particles]
    b_y        = [pa.dotx[2] / pa.Teff                  for pa in particles]
    My_ep      = [pa.M_smooth[2] / (pa.T * pa.S_smooth) for pa in particles]

    T_rel_bg = particles[1].T * particles[1].γ
    My_anal  = @. M_amp * sin(k * xs_grid)
    dM_anal  = @. -(η_by_s / T_rel_bg) * k^2 * M_amp * sin(k * xs_grid)

    return xs, My_density, xs_grid, dMydt_grid, My_anal, dM_anal,
           dotx_y, v_y, b_y, My_ep
end

function plot_diffusion_ic(params, particles, cl, eos, k, M_amp)
    h      = params.h
    η_by_s = eta_by_s(eos)
    label  = @sprintf("h=%.3g  h/a=%.3g  η/s=%.3g", h, h/params.a, η_by_s)
    @printf("IC test: %s\n", label)

    xs, My_d, xs_grid, dMydt_grid, My_anal, dM_anal,
        dotx_y, v_y, b_y, My_ep =
        run_diffusion_test_ic(params, particles, cl, eos, k, M_amp)

    _, (ax1, ax2, ax3) = subplots(3, 1, figsize=(9, 10), sharex=true)

    ax1.scatter(xs, My_d,    s=8, alpha=0.4, color="C0", label="SPH M_smooth^y")
    ax1.plot(xs_grid, My_anal, color="C0", linewidth=1.2, label="analytic")
    ax1.set_ylabel("M^y density")
    ax1.set_title("Shear wave IC: SPH (dots) vs analytic  — $label")
    ax1.legend(fontsize=8)

    ax2.plot(xs_grid, dMydt_grid, color="C1", linewidth=1.2, label="SPH ∂_t M^y")
    ax2.plot(xs_grid, dM_anal,   "--", color="C1", linewidth=1.0, label="analytic")
    ax2.set_ylabel("∂_t M^y density")
    ax2.set_title("Viscous force density vs analytic  [−(η/s)/T_rel k² M_amp sin(kx)]")
    ax2.legend(fontsize=8)

    ax3.scatter(xs, dotx_y, s=10, alpha=0.5, color="C0", marker="o", label="ẋ_y")
    ax3.scatter(xs, v_y,    s=10, alpha=0.5, color="C1", marker="s", label="v_y")
    ax3.scatter(xs, b_y,    s=10, alpha=0.5, color="C2", marker="^", label="b_y = ẋ_y/Teff")
    ax3.scatter(xs, My_ep,  s=10, alpha=0.5, color="C3", marker="D", label="M_y/(T·S)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("velocity-like quantity")
    ax3.set_title("Diagnostics: ẋ_y  v_y  b_y=ẋ/Teff  M_y/(T·S)")
    ax3.legend(fontsize=8)

    tight_layout()
    savefig(joinpath(@__DIR__, "testdiffuse.png"), dpi=150)
    println("Saved testdiffuse.png")
    show()
end

# ── Time evolution ─────────────────────────────────────────────────────────────
# Evolve the shear wave using dissipative_step! alone and record the amplitude.
# dt is chosen from the diffusive CFL: dt = 0.1 * h² * T_rel / (η/s).
# Runs for n_decay decay times τ_k = T_rel / ((η/s) k²).
function run_diffusion_evolve!(particles, params, cl, eos, k; n_decay=3)
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)

    T_rel_bg = particles[1].T * particles[1].γ
    γ_k      = eta_by_s(eos) * k^2 / T_rel_bg
    t_final  = n_decay / γ_k
    dt       = compute_dt(particles, params, eos; mode=:viscous, CFL=0.1, T0=T_rel_bg)
    n_steps  = ceil(Int, t_final / dt)
    dt       = t_final / n_steps          # adjust to land exactly on t_final

    @printf("Evolution: γ_k=%.4g  τ_k=%.4g  t_final=%.4g  dt=%.4g  n_steps=%d\n",
            γ_k, 1/γ_k, t_final, dt, n_steps)

    t_vals   = zeros(n_steps + 1)
    amp_vals = zeros(n_steps + 1)
    amp_vals[1] = maximum(abs(pa.M_smooth[2]) for pa in particles)

    s_init = sum(pa.s         for pa in particles)
    H_init = sum(pa.s * pa.H_over_S for pa in particles)
    @printf("  %6s  %8s  %10s  %12s  %12s  %12s  %12s\n",
            "step", "t·γ_k", "amp/amp₀", "ΔH/H₀", "ΔS/S₀", "Px", "Py")

    for n in 1:n_steps
        dissipative_step!(particles, params, cl, dt, eos)
        t_vals[n+1]   = n * dt
        amp_vals[n+1] = maximum(abs(pa.M_smooth[2]) for pa in particles)
        if n % 20 == 0 || n == n_steps
            H_now = sum(pa.s * pa.H_over_S for pa in particles)
            s_now = sum(pa.s         for pa in particles)
            P_now = sum(pa.m         for pa in particles)
            @printf("  %6d  %8.4f  %10.6f  %+12.4e  %+12.4e  %+12.4e  %+12.4e\n",
                    n, n*dt*γ_k, amp_vals[n+1]/amp_vals[1],
                    (H_now - H_init)/abs(H_init), (s_now - s_init)/s_init,
                    P_now[1], P_now[2])
        end
    end

    return t_vals, amp_vals, γ_k
end

function plot_diffusion_evolve(t_vals, amp_vals, γ_k, M_amp, params, eos)
    dt    = t_vals[2] - t_vals[1]
    label = @sprintf("h=%.3g  h/a=%.3g  η/s=%.3g  dt/τ_k=%.3g",
                      params.h, params.h/params.a, eta_by_s(eos), dt*γ_k)

    _, ax = subplots(1, 1, figsize=(8, 5))
    ax.semilogy(t_vals .* γ_k, amp_vals ./ M_amp, "o-", ms=3, label="SPH")
    t_line = [0.0, maximum(t_vals) * γ_k]
    ax.semilogy(t_line, exp.(-t_line), "--", color="k", label="\$e^{-\\gamma_k t}\$")
    ax.set_xlabel("\$t \\, / \\, \\tau_k\$  (decay times)")
    ax.set_ylabel("\$M^y\\!\\!/M_{\\rm amp}\$  (normalized amplitude)")
    ax.set_title("Shear wave decay — $label")
    ax.legend()
    tight_layout()
    savefig(joinpath(@__DIR__, "testdiffuse_evolve.png"), dpi=150)
    println("Saved testdiffuse_evolve.png")
    show()
end

function plot_diffusion_final(particles, params, cl, k, M_amp, t_end, γ_k)
    xs_sl, My_sl = plot_slice(particles, params, cl;
                               field=p -> p.M_smooth[2], dim=:x)
    M_anal = @. M_amp * sin(k * xs_sl) * exp(-γ_k * t_end)
    rms = sqrt(sum((s - a)^2 for (s, a) in zip(My_sl, M_anal)) / length(My_sl))
    @printf("M^y residual at t_end: rms=%.3g  amplitude=%.3g\n", rms, M_amp*exp(-γ_k*t_end))

    _, ax = subplots(1, 1, figsize=(7, 4.5))
    ax.plot(xs_sl, My_sl, color="C0", label="SPH")
    ax.plot(xs_sl, M_anal, "--", color="C1", linewidth=1.5,
            label="analytic  M_amp·sin(kx)·e^{-γ_k t}")
    ax.set_xlabel("x")
    ax.set_ylabel("\$M^y\$ density")
    ax.set_title(@sprintf("Shear wave at t=%.4g (%.2f τ_k)  rms=%.3g",
                           t_end, t_end*γ_k, rms))
    ax.legend(fontsize=9)
    tight_layout()
    savefig(joinpath(@__DIR__, "testdiffuse_final.png"), dpi=150)
    println("Saved testdiffuse_final.png")
    show()
end

# ── Run ───────────────────────────────────────────────────────────────────────

h  = 0.02
ha = 2
params, particles, cl, eos, k, M_amp =
    setup_shear_wave(h, ha; L=1.4, s0=2.3, M_amp=0.02, cs2v=1.0/3.0,
                     η_by_s=0.1, lattice=:hex, jitter=0.0)

# (1) IC test: instantaneous viscous force vs analytic
plot_diffusion_ic(params, particles, cl, eos, k, M_amp)

# (2) Evolution: shear wave amplitude decay vs analytic exponential
#     Note: run_diffusion_evolve! mutates particles in place
t_vals, amp_vals, γ_k =
    run_diffusion_evolve!(particles, params, cl, eos, k; n_decay=3)
plot_diffusion_evolve(t_vals, amp_vals, γ_k, M_amp, params, eos)

# (3) Final spatial profile: SPH M^y(x) vs analytic decayed sine
plot_diffusion_final(particles, params, cl, k, M_amp, t_vals[end], γ_k)
