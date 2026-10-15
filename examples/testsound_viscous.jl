using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

# ── Analytic solution ─────────────────────────────────────────────────────────
# Right-moving eigenmode: ω = ωR - i Γ_s k²/2, |ω/k| = cs (exact identity).
# M⃗(t,x,y) = A [(ωR/k)cos(k⃗·x−ωR t) + (Γ_s k/2)sin(k⃗·x−ωR t)] k̂ e^{−Γ_s k²t/2}
#           = A cs cos(k⃗·x − ωR t − φ) k̂ e^{−Γ_s k²t/2},  φ=atan(Γ_s k²/2ωR)

function analytic_M(t, x, y, A, k, kx, ky, khat, ωR, Γ_s)
    phase = kx*x + ky*y - ωR*t
    decay = exp(-Γ_s * k^2 * t / 2)
    return A * ((ωR/k)*cos(phase) + (Γ_s*k/2)*sin(phase)) * decay * khat
end

# x-component only (scalar), convenient for slices and plots
analytic_Mx(t, x, y, A, k, kx, ky, khat1, ωR, Γ_s) =
    analytic_M(t, x, y, A, k, kx, ky, SVector(khat1, 0.0), ωR, Γ_s)[1]

# ── Initialization ─────────────────────────────────────────────────────────────

function initialize_viscous_sound(h, ha, Lx, Ly, s0, A, cs2v, nx, ny, η_by_s, ζ_by_s)
    cs    = sqrt(cs2v)
    kx    = 2π / Lx * nx
    ky    = 2π / Ly * ny
    k     = sqrt(kx^2 + ky^2)
    khat  = SVector(kx, ky) / k

    eos = ConformalEOS(cs2v, 1.0, η_by_s, ζ_by_s)

    # Uniform background → T_rel_0 for IC closure and Γ_s.
    _, bg, _, _ = setup_ic(h, ha, eos, (x, y) -> s0, (x, y) -> SVector(0.0, 0.0); Lx, Ly)
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
    @printf("k⃗=(2π/Lx·%d, 2π/Ly·%d)  k=%.4g  cs=%.4g  ωR/k=%.4g  (shift=%.4g%%)\n",
            nx, ny, k, cs, ωR/k, 100*(cs - ωR/k)/cs)
    @printf("T_period=%.4g  τ_s=%.4g=%.2f T_period  Γ_s·k/cs=%.4g\n",
            T_period, τ_s, τ_s/T_period, Γ_s*k/cs)
    @printf("φ = atan(Γ_s k²/2ωR) = %.4g rad\n", atan(Γ_s*k^2/2, ωR))

    # IC: δS = A/T_rel_0 cos(k⃗·x),  M⃗ = analytic_M(t=0, ...)
    params, particles, cl, eos = setup_ic(h, ha, eos,
        (x, y) -> s0 + (A / T_rel_0) * cos(kx*x + ky*y),
        (x, y) -> analytic_M(0.0, x, y, A, k, kx, ky, khat, ωR, Γ_s); Lx, Ly)

    return params, particles, cl, eos, kx, ky, khat, k, cs, ωR, T_period, T_rel_0, Γ_s
end

# ── Evolution movie ─────────────────────────────────────────────────────────────
# Call with flag=:beginning before the loop (creates figure, plots IC),
#           flag=:middle during the loop (adds a colored SPH+analytic curve),
#           flag=:end after the loop (adds legend, saves to disk).
# mp = NamedTuple: A, k, kx, ky, khat, ωR, Γ_s, T_period, η_by_s, ζ_by_s, cs2v, n_movie

function evolution_plot!(fig_ax, particles, params, cl, step, t, n_steps, mp; flag::Symbol)
    cmap  = get_cmap("plasma")
    color = cmap(clamp(step / max(n_steps, 1), 0.05, 0.95))
    L     = params.L

    if flag == :beginning
        fig, ax = subplots(1, 1, figsize=(9, 5))
        ax.set_xlabel("x")
        ax.set_ylabel("\$M^x\$ density")
        ax.set_title(@sprintf(
            "Viscous sound  η/s=%.3g  ζ/s=%.3g  h=%.3g  Γ_s·k/cs=%.3g",
            mp.η_by_s, mp.ζ_by_s, params.h, mp.Γ_s * mp.k / sqrt(mp.cs2v)))

        xs_sl, Mx_sl = plot_slice(particles, params, cl;
                                   field=p -> p.M_smooth[1], dim=:x)
        Mx_ic = [analytic_Mx(0.0, x, L[2]/2, mp.A, mp.k, mp.kx, mp.ky,
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
        Mx_anal = [analytic_Mx(t, x, L[2]/2, mp.A, mp.k, mp.kx, mp.ky,
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
            savefig(joinpath(@__DIR__, "testsound_viscous_evolution.png"), dpi=150)
            println("Saved testsound_viscous_evolution.png")
        end
        return fig, ax
    end
end

# ── Time evolution ─────────────────────────────────────────────────────────────
# movie keyword: pass a NamedTuple (A,k,kx,ky,khat,ωR,Γ_s,T_period,
#                                    η_by_s,ζ_by_s,cs2v,n_movie) or nothing.

function run_viscous_sound!(particles, params, cl, eos, t_end, ωR;
                             CFL=0.2, movie=nothing)
    T_period     = ωR > 0 ? 2π / ωR : Inf   # Inf → diagnostic column shows 0 (harmless)
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
        evolution_plot!(nothing, particles, params, cl, 0, 0.0, n_steps, movie;
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
            fig_ax = evolution_plot!(fig_ax, particles, params, cl,
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

# ── Amplitude decay plot ────────────────────────────────────────────────────────
# Amplitude tracked as max_a |M_smooth^x(x_a)| — the peak of the SPH-smoothed
# M^x field over all particle positions.  Normalized to 1 at t=0.

function plot_amplitude_decay(t_vals, amp_vals, Γ_s, k, params, η_by_s, ζ_by_s)
    τ_s   = 2 / (Γ_s * k^2)
    x     = t_vals ./ τ_s          # dimensionless: t Γ_s k²/2
    label = @sprintf("η/s=%.3g  ζ/s=%.3g  h=%.3g  h/a=%.3g",
                      η_by_s, ζ_by_s, params.h, params.h/params.a)

    _, ax = subplots(1, 1, figsize=(8, 5))
    ax.semilogy(x, amp_vals ./ amp_vals[1], "o-", ms=2, label="SPH")
    x_line = [0.0, x[end]]
    ax.semilogy(x_line, exp.(-x_line), "--", color="k",
                label="\$e^{-\\Gamma_s k^2 t/2}\$")
    ax.set_xlabel("\$t \\,/\\, \\tau_s = t\\,\\Gamma_s k^2 / 2\$")
    ax.set_ylabel("amplitude / initial")
    ax.set_title("Viscous sound attenuation — $label")
    ax.legend()
    tight_layout()
    savefig(joinpath(@__DIR__, "testsound_viscous_decay.png"), dpi=150)
    println("Saved testsound_viscous_decay.png")
    show()
end

# ── Final spatial profile ───────────────────────────────────────────────────────

function plot_final_viscous_sound(particles, params, cl,
                                   t_end, A, k, kx, ky, khat, ωR, Γ_s;
                                   nx, ny, T_period)
    xs_sl, Mx_sl = plot_slice(particles, params, cl;
                               field=p -> p.M_smooth[1], dim=:x)
    Mx_anal = [analytic_Mx(t_end, x, params.L[2]/2, A, k, kx, ky, khat[1], ωR, Γ_s)
               for x in xs_sl]
    Mx_ic   = [analytic_Mx(0.0,   x, params.L[2]/2, A, k, kx, ky, khat[1], ωR, Γ_s)
               for x in xs_sl]
    rms = sqrt(sum((s - a)^2 for (s, a) in zip(Mx_sl, Mx_anal)) / length(Mx_sl))
    amp_anal = A * sqrt(ωR^2/k^2 + (Γ_s*k/2)^2) * abs(khat[1]) * exp(-Γ_s*k^2*t_end/2)
    @printf("Mx residual: rms=%.3g  analytic amplitude=%.3g (=A·cs·decay)\n", rms, amp_anal)

    _, ax = subplots(1, 1, figsize=(7, 4.5))
    # ax.plot(xs_sl, Mx_ic,   "--", color="gray", linewidth=1.0, alpha=0.7, label="IC (t=0)")
    ax.plot(xs_sl, Mx_anal, "--", color="C1",   linewidth=1.5,            label="analytic")
    ax.plot(xs_sl, Mx_sl,         color="C0",   linewidth=1.5,            label="SPH")
    ax.set_xlabel("x"); ax.set_ylabel("\$M^x\$ density")
    ax.set_title(@sprintf("Viscous sound  nx=%d ny=%d  t=%.3g = %.2f T_period  rms=%.3g",
                           nx, ny, t_end, t_end/T_period, rms))
    ax.legend(fontsize=9)
    tight_layout()
    savefig(joinpath(@__DIR__, "testsound_viscous_final.png"), dpi=150)
    println("Saved testsound_viscous_final.png")
    show()
end

# ── IC particle plot ───────────────────────────────────────────────────────────

function plot_ic_viscous_sound(particles, params; nx, ny)
    fig, (ax1, ax2) = subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Viscous sound IC  nx=$nx ny=$ny")
    plot_particles(particles, params;
                   field=p -> p.M_smooth[1], clabel="Mx density",
                   title="Mx density", ax=ax1)
    plot_particles(particles, params;
                   field=p -> p.M_smooth[2], clabel="My density",
                   title="My density", ax=ax2)
    tight_layout()
    return fig
end

# ── Main ───────────────────────────────────────────────────────────────────────

h      = 0.03
ha     = 2.0
Lx     = 1.2
Ly     = 1.6
s0     = 2.0
A      = 0.02        # perturbation amplitude (keep ≪ s0)
cs2v   = 1.0 / 3.0
nx     = 2
ny     = 1
η_by_s = 0.02
ζ_by_s = 0.0

params, particles, cl, eos, kx, ky, khat, k, cs, ωR, T_period, T_rel_0, Γ_s =
    initialize_viscous_sound(h, ha, Lx, Ly, s0, A, cs2v, nx, ny, η_by_s, ζ_by_s)

fig_ic = plot_ic_viscous_sound(particles, params; nx, ny)
savefig(joinpath(@__DIR__, "testsound_viscous_ic.png"), dpi=150)
println("Saved testsound_viscous_ic.png")

t_end = 1 * T_period
@printf("t_end=%.4g (= 2 T_period)\n", t_end)

movie = (A=A, k=k, kx=kx, ky=ky, khat=khat, ωR=ωR, Γ_s=Γ_s,
         T_period=T_period, n_movie=40, η_by_s=η_by_s, ζ_by_s=ζ_by_s, cs2v=cs2v)

t_vals, amp_vals =
    run_viscous_sound!(particles, params, cl, eos, t_end, ωR; movie)

plot_amplitude_decay(t_vals, amp_vals, Γ_s, k, params, η_by_s, ζ_by_s)

plot_final_viscous_sound(particles, params, cl,
                          t_end, A, k, kx, ky, khat, ωR, Γ_s;
                          nx, ny, T_period)

show()
