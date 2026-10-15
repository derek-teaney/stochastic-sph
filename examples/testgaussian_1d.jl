using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
# ── Initialization ────────────────────────────────────────────────────────────

function initialize_gaussian_1d(h, ha;
                                 Lx=1.0, A=1.0, delta=1e-1,
                                 sigma=0.1, x0=0.0, n_images=1,
                                 cs2v=1.0/3.0, η_by_s=0.002, ζ_by_s=0.0)
    eos = ConformalEOS(cs2v, T_scale_gluon, η_by_s, ζ_by_s)

    function e_ic(x, y)
        e = delta
        for mx in -n_images:n_images
            e += A * exp(-((x - x0 - mx*Lx)^2) / (2*sigma^2))
        end
        return e
    end
    S_func = (x, y) -> srel(eos, e_ic(x, y))

    params, particles, cl, eos = setup_ic(h, ha, eos, S_func,
                                           (x, y) -> SVector(0.0, 0.0); Lx, sim_dim=1)
    cs = sqrt(cs2v)
    @printf("Gaussian 1D: A=%.4g  δ=%.4g  σ=%.4g  center=%.3g  n_img=%d  cs=%.4g  η/s=%.4g\n",
            A, delta, sigma, x0, n_images, cs, η_by_s)
    return params, particles, cl, eos, cs
end

# ── Plotting ──────────────────────────────────────────────────────────────────

function plot_gaussian_1d(particles, params, cl; title_prefix="")
    fig, ax = subplots(1, 1, figsize=(8, 4))
    isempty(title_prefix) || fig.suptitle(title_prefix)

    xs_sl, fs_sl = plot_slice(particles, params, cl; field=p -> p.S_smooth * p.H_over_S, dim=:x)
    ax.plot(xs_sl, fs_sl, color="C0", label="SPH")
    xs_p = [pa.x[1] for pa in particles]
    es_p = [pa.e    for pa in particles]
    ax.scatter(xs_sl, fs_sl, s=4, color="C1", label="T00")
    ax.set_xlabel("x"); ax.set_ylabel("e density")
    ax.set_title("e density")
    ax.legend(fontsize=8)

    tight_layout()
    return fig
end

# ── Time evolution ────────────────────────────────────────────────────────────

function run_gaussian!(particles, params, cl, eos, t_end;
                       CFL=0.2)
    s_init = sum(pa.s for pa in particles)
    H_init = sum(pa.s * pa.H_over_S for pa in particles)
    dotm   = zeros(SVector{2,Float64}, params.N)

    t          = 0.0
    step       = 0

    while t < t_end
        dt = compute_dt(particles, params, eos; mode=:viscous, CFL)
        dt = min(dt, t_end - t)

        generic_step!(particles, params, cl, dt, eos, dotm)
        t    += dt
        step += 1

        if step % 20 == 0
            H = sum(pa.s * pa.H_over_S for pa in particles)
            @printf("  step %d  t=%.4g / %.4g  dt=%.4g  H=%.10g\n", step, t, t_end, dt, H)
        end

    end
    @printf("  done: %d steps  t=%.6g\n", step, t)

    s_final = sum(pa.s for pa in particles)
    H_final = sum(pa.s * pa.H_over_S for pa in particles)
    return step, s_init, s_final, H_init, H_final
end

# ── Main ──────────────────────────────────────────────────────────────────────

h        = 0.01
ha       = 5.0
Lx       = 1.0
A        = 1.0
delta    = 1.e-1
sigma    = 0.1
x0       = Lx / 2.
n_images = 1
η_by_s   = 0.002
t_end    = 2.2

params, particles, cl, eos, cs =
    initialize_gaussian_1d(h, ha; Lx, A, delta, sigma, x0, n_images, η_by_s)

fig_ic = plot_gaussian_1d(particles, params, cl;
                           title_prefix="IC: Gaussian 1D  A=$A  σ=$sigma")
savefig(joinpath(@__DIR__, "testgaussian_1d_ic.png"), dpi=150)
println("Saved testgaussian_1d_ic.png")

@printf("t_end=%.4g\n", t_end)

n_steps, s_init, s_final, H_init, H_final =
    run_gaussian!(particles, params, cl, eos, t_end)

@printf("Entropy: Σs_0=%.10g  Σs_f=%.10g  rel.err=%.3g\n",
        s_init, s_final, abs(s_final - s_init) / s_init)
@printf("Energy:  H_0=%.10g  H_f=%.10g  rel.err=%.3g\n",
        H_init, H_final, abs(H_final - H_init) / abs(H_init))

fig_final = plot_gaussian_1d(particles, params, cl;
                              title_prefix="Final: Gaussian 1D  t=$(round(t_end, digits=3))")
savefig(joinpath(@__DIR__, "testgaussian_1d_final.png"), dpi=150)
println("Saved testgaussian_1d_final.png")

show()
