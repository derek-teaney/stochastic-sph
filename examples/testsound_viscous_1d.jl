using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

# ── Analytic solution ─────────────────────────────────────────────────────────
# Right-moving eigenmode: ω = ωR - i Γ_s k²/2, |ω/k| = cs (exact identity).
# M⃗(t,x) = A [(ωR/k)cos(kx−ωR t) + (Γ_s k/2)sin(kx−ωR t)] k̂ e^{−Γ_s k²t/2}

function analytic_M(t, x, y, A, k, kx, ky, khat, ωR, Γ_s)
    phase = kx*x + ky*y - ωR*t
    decay = exp(-Γ_s * k^2 * t / 2)
    return A * ((ωR/k)*cos(phase) + (Γ_s*k/2)*sin(phase)) * decay * khat
end

analytic_Mx(t, x, y, A, k, kx, ky, khat1, ωR, Γ_s) =
    analytic_M(t, x, y, A, k, kx, ky, SVector(khat1, 0.0), ωR, Γ_s)[1]

# ── Initialization ────────────────────────────────────────────────────────────

function initialize_viscous_sound_1d(h, ha, Lx, s0, A, cs2v, nx, η_by_s, ζ_by_s)
    cs   = sqrt(cs2v)
    kx   = 2π / Lx * nx
    ky   = 0.0
    k    = kx
    khat = SVector(1.0, 0.0)

    eos = ConformalEOS(cs2v, 1.0, η_by_s, ζ_by_s)

    # Uniform background → T_rel_0 for IC closure and Γ_s.
    _, bg, _, _ = setup_ic(h, ha, eos, (x, y) -> s0, (x, y) -> SVector(0.0, 0.0);
                            Lx, sim_dim=1)
    T_rel_0 = bg[1].T * bg[1].γ

    Γ_s = (4/3 * η_by_s + ζ_by_s) / T_rel_0

    # Exact dispersion: ω = ωR - i Γ_s k²/2,  ωR = (k/2)√(4cs²−Γ_s²k²)
    ωR         = (k/2) * sqrt(max(0.0, 4*cs^2 - (Γ_s*k)^2))
    τ_s        = 2 / (Γ_s * k^2)
    overdamped = ωR == 0.0
    T_period   = overdamped ? τ_s : 2π / ωR
    overdamped && @printf("WARNING: overdamped regime (Γ_s·k=%.4g ≥ 2cs=%.4g) — no propagating mode; T_period→τ_s\n",
                           Γ_s*k, 2*cs)

    @printf("T_rel_0=%.6g  η/s=%.4g  ζ/s=%.4g  Γ_s=%.4g\n", T_rel_0, η_by_s, ζ_by_s, Γ_s)
    @printf("k = 2π/Lx·%d = %.4g  cs=%.4g  ωR/k=%.4g  (shift=%.4g%%)\n",
            nx, k, cs, ωR/k, 100*(cs - ωR/k)/cs)
    @printf("T_period=%.4g  τ_s=%.4g=%.2f T_period  Γ_s·k/cs=%.4g\n",
            T_period, τ_s, τ_s/T_period, Γ_s*k/cs)
    @printf("φ = atan(Γ_s k²/2ωR) = %.4g rad\n", atan(Γ_s*k^2/2, ωR))

    # IC: δS = A/T_rel_0 cos(kx),  M⃗ = analytic_M(t=0, ...)
    params, particles, cl, eos = setup_ic(h, ha, eos,
        (x, y) -> s0 + (A / T_rel_0) * cos(kx*x),
        (x, y) -> analytic_M(0.0, x, 0.0, A, k, kx, 0.0, khat, ωR, Γ_s);
        Lx, sim_dim=1)

    return params, particles, cl, eos, kx, ky, khat, k, cs, ωR, T_period, T_rel_0, Γ_s
end

# ── IC plot ───────────────────────────────────────────────────────────────────

function plot_ic_viscous_sound_1d(particles, params; nx)
    fig, ax = subplots(1, 1, figsize=(8, 4))
    fig.suptitle("Viscous sound 1D IC  nx=$nx")
    plot_line(particles, params;
              field=p -> p.M_smooth[1], ylabel="\$M^x\$ density",
              title="\$M^x\$ density", ax=ax)
    tight_layout()
    return fig
end

# ── Evolution movie ───────────────────────────────────────────────────────────
# mp = NamedTuple: A, k, kx, ky, khat, ωR, Γ_s, T_period, η_by_s, ζ_by_s, cs2v, n_movie

function evolution_plot_1d!(fig_ax, particles, params, cl, step, t, n_steps, mp; flag::Symbol)
    cmap  = get_cmap("plasma")
    color = cmap(clamp(step / max(n_steps, 1), 0.05, 0.95))

    if flag == :beginning
        fig, ax = subplots(1, 1, figsize=(9, 5))
        ax.set_xlabel("x")
        ax.set_ylabel("\$M^x\$ density")
        ax.set_title(@sprintf(
            "Viscous sound 1D  η/s=%.3g  ζ/s=%.3g  h=%.3g  Γ_s·k/cs=%.3g",
            mp.η_by_s, mp.ζ_by_s, params.h, mp.Γ_s * mp.k / sqrt(mp.cs2v)))

        xs_sl, Mx_sl = plot_slice(particles, params, cl;
                                   field=p -> p.M_smooth[1], dim=:x)
        Mx_ic = [analytic_Mx(0.0, x, 0.0, mp.A, mp.k, mp.kx, 0.0,
                              mp.khat[1], mp.ωR, mp.Γ_s) for x in xs_sl]
        ax.plot(xs_sl, Mx_sl, color=color, linewidth=1.2, label="SPH t=0")
        ax.plot(xs_sl, Mx_ic, "k:", linewidth=0.8, alpha=0.6)
        tight_layout()
        pause(0.001)
        return fig, ax

    elseif flag == :middle || flag == :end
        fig, ax = fig_ax
        xs_sl, Mx_sl = plot_slice(particles, params, cl;
                                   field=p -> p.M_smooth[1], dim=:x)
        Mx_anal = [analytic_Mx(t, x, 0.0, mp.A, mp.k, mp.kx, 0.0,
                                mp.khat[1], mp.ωR, mp.Γ_s) for x in xs_sl]

        lw = flag == :end ? 1.5 : 1.0
        ax.plot(xs_sl, Mx_sl, color=color, linewidth=lw,
                label=@sprintf("SPH t=%.2f T", t / mp.T_period))
        ax.plot(xs_sl, Mx_anal, "k:", linewidth=0.7, alpha=0.5)

        fig.canvas.draw_idle()
        pause(0.001)

        if flag == :end
            ax.legend(fontsize=7, loc="upper right", ncol=2)
            tight_layout()
            savefig(joinpath(@__DIR__, "testsound_viscous_1d_evolution.png"), dpi=150)
            println("Saved testsound_viscous_1d_evolution.png")
        end
        return fig, ax
    end
end

# ── Time evolution ─────────────────────────────────────────────────────────────

function run_viscous_sound!(particles, params, cl, eos, t_end, ωR;
                             CFL=0.2, movie=nothing)
    T_period     = ωR > 0 ? 2π / ωR : Inf
    dt           = compute_dt(particles, params, eos; mode=:viscous, CFL)
    n_steps      = ceil(Int, t_end / dt)
    dt           = t_end / n_steps
    @printf("dt=%.4g  n_steps=%d\n", dt, n_steps)

    s_init   = sum(pa.s               for pa in particles)
    H_init   = sum(pa.s * pa.H_over_S for pa in particles)
    amp_init = maximum(abs(pa.M_smooth[1]) for pa in particles)

    t_vals   = zeros(n_steps + 1)
    amp_vals = zeros(n_steps + 1)
    amp_vals[1] = amp_init

    dotm   = zeros(SVector{2,Float64}, params.N)
    fig_ax = movie !== nothing ?
        evolution_plot_1d!(nothing, particles, params, cl, 0, 0.0, n_steps, movie;
                           flag=:beginning) : nothing

    @printf("  %6s  %8s  %10s  %12s  %12s\n",
            "step", "t/T_per", "amp/amp₀", "ΔH/H₀", "ΔS/S₀")

    for step in 1:n_steps
        generic_step!(particles, params, cl, dt, eos, dotm)
        t_vals[step+1]   = step * dt
        amp_vals[step+1] = maximum(abs(pa.M_smooth[1]) for pa in particles)

        if step % 20 == 0 || step == n_steps
            H_now = sum(pa.s * pa.H_over_S for pa in particles)
            s_now = sum(pa.s               for pa in particles)
            @printf("  %6d  %8.4f  %10.6f  %+12.4e  %+12.4e\n",
                    step, step*dt/T_period, amp_vals[step+1]/amp_init,
                    (H_now - H_init)/abs(H_init), (s_now - s_init)/s_init)
        end

        if movie !== nothing && (step % movie.n_movie == 0 || step == n_steps)
            mflag = step == n_steps ? :end : :middle
            fig_ax = evolution_plot_1d!(fig_ax, particles, params, cl,
                                        step, step*dt, n_steps, movie; flag=mflag)
        end
    end

    s_final = sum(pa.s               for pa in particles)
    H_final = sum(pa.s * pa.H_over_S for pa in particles)
    @printf("Entropy: Σs_0=%.10g  Σs_f=%.10g  rel.Δ=%+.3g\n",
            s_init, s_final, (s_final - s_init) / s_init)
    @printf("Energy:  H_0=%.10g   H_f=%.10g   rel.Δ=%+.3g\n",
            H_init, H_final, (H_final - H_init) / abs(H_init))

    return t_vals, amp_vals
end

# ── Amplitude decay plot ──────────────────────────────────────────────────────

function plot_amplitude_decay(t_vals, amp_vals, Γ_s, k, params, η_by_s, ζ_by_s)
    τ_s   = 2 / (Γ_s * k^2)
    x     = t_vals ./ τ_s
    label = @sprintf("η/s=%.3g  ζ/s=%.3g  h=%.3g  h/a=%.3g",
                      η_by_s, ζ_by_s, params.h, params.h/params.a)

    _, ax = subplots(1, 1, figsize=(8, 5))
    ax.semilogy(x, amp_vals ./ amp_vals[1], "o-", ms=2, label="SPH")
    x_line = [0.0, x[end]]
    ax.semilogy(x_line, exp.(-x_line), "--", color="k",
                label="\$e^{-\\Gamma_s k^2 t/2}\$")
    ax.set_xlabel("\$t \\,/\\, \\tau_s = t\\,\\Gamma_s k^2 / 2\$")
    ax.set_ylabel("amplitude / initial")
    ax.set_title("Viscous sound 1D attenuation — $label")
    ax.legend()
    tight_layout()
    savefig(joinpath(@__DIR__, "testsound_viscous_1d_decay.png"), dpi=150)
    println("Saved testsound_viscous_1d_decay.png")
end

# ── Final spatial profile ─────────────────────────────────────────────────────

function plot_final_viscous_sound_1d(particles, params, cl,
                                      t_end, A, k, kx, khat, ωR, Γ_s;
                                      nx, T_period)
    xs_sl, Mx_sl = plot_slice(particles, params, cl;
                               field=p -> p.M_smooth[1], dim=:x)
    Mx_anal = [analytic_Mx(t_end, x, 0.0, A, k, kx, 0.0, khat[1], ωR, Γ_s)
               for x in xs_sl]
    Mx_ic   = [analytic_Mx(0.0,   x, 0.0, A, k, kx, 0.0, khat[1], ωR, Γ_s)
               for x in xs_sl]
    rms = sqrt(sum((s - a)^2 for (s, a) in zip(Mx_sl, Mx_anal)) / length(Mx_sl))
    amp_anal = A * sqrt(ωR^2/k^2 + (Γ_s*k/2)^2) * abs(khat[1]) * exp(-Γ_s*k^2*t_end/2)
    @printf("Mx residual: rms=%.3g  analytic amplitude=%.3g\n", rms, amp_anal)

    _, ax = subplots(1, 1, figsize=(7, 4.5))
    ax.plot(xs_sl, Mx_ic,   "--", color="gray", linewidth=1.0, alpha=0.7, label="IC (t=0)")
    ax.plot(xs_sl, Mx_anal, "--", color="C1",   linewidth=1.5,            label="analytic")
    ax.plot(xs_sl, Mx_sl,         color="C0",   linewidth=1.5,            label="SPH")
    ax.set_xlabel("x"); ax.set_ylabel("\$M^x\$ density")
    ax.set_title(@sprintf("Viscous sound 1D  nx=%d  t=%.3g = %.2f T_period  rms=%.3g",
                           nx, t_end, t_end/T_period, rms))
    ax.legend(fontsize=9)
    tight_layout()
    savefig(joinpath(@__DIR__, "testsound_viscous_1d_final.png"), dpi=150)
    println("Saved testsound_viscous_1d_final.png")
end

# ── Main ──────────────────────────────────────────────────────────────────────

h      = 0.005
ha     = 4.0
Lx     = 1.0
s0     = 2.0
A      = 0.02
cs2v   = 1.0 / 3.0
nx     = 2
η_by_s = 0.01
ζ_by_s = 0.0

params, particles, cl, eos, kx, ky, khat, k, cs, ωR, T_period, T_rel_0, Γ_s =
    initialize_viscous_sound_1d(h, ha, Lx, s0, A, cs2v, nx, η_by_s, ζ_by_s)

fig_ic = plot_ic_viscous_sound_1d(particles, params; nx)
savefig(joinpath(@__DIR__, "testsound_viscous_1d_ic.png"), dpi=150)
println("Saved testsound_viscous_1d_ic.png")

t_end = 1 * T_period
@printf("t_end=%.4g (= 1 T_period)\n", t_end)

movie = (A=A, k=k, kx=kx, ky=0.0, khat=khat, ωR=ωR, Γ_s=Γ_s,
         T_period=T_period, n_movie=40, η_by_s=η_by_s, ζ_by_s=ζ_by_s, cs2v=cs2v)

t_vals, amp_vals =
    run_viscous_sound!(particles, params, cl, eos, t_end, ωR; movie)

plot_amplitude_decay(t_vals, amp_vals, Γ_s, k, params, η_by_s, ζ_by_s)

plot_final_viscous_sound_1d(particles, params, cl,
                             t_end, A, k, kx, khat, ωR, Γ_s;
                             nx, T_period)

show()
