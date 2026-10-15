using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
# ── Initialization ────────────────────────────────────────────────────────────

function initialize_gaussian(h, ha;
                              L=1.0, A=1.0, delta=1e-4,
                              sigmax=0.1, sigmay=0.14,
                              x0=0.0, y0=0.0, n_images=1,
                              cs2v=1.0/3.0, η_by_s=0.002, ζ_by_s=0.0)
    eos = ConformalEOS(cs2v, T_scale_gluon, η_by_s, ζ_by_s)

    # Energy density: small floor + Gaussian at (x0,y0) plus periodic images.
    # sigmay=Inf → flat in y (no y-images needed).
    flat_y = !isfinite(sigmay)
    my_range = flat_y ? (0:0) : (-n_images:n_images)
    function e_ic(x, y)
        e = delta
        for mx in -n_images:n_images, my in my_range
            cx = x0 + mx * L
            cy = y0 + my * L
            fy = flat_y ? 1.0 : exp(-((y - cy)^2) / (2*sigmay^2))
            e += A * exp(-((x - cx)^2) / (2*sigmax^2)) * fy
        end
        return e
    end
    S_func = (x, y) -> srel(eos, e_ic(x, y))

    params, particles, cl, eos = setup_ic(h, ha, eos, S_func,
                                           (x, y) -> SVector(0.0, 0.0); L)
    cs = sqrt(cs2v)
    sigmay_str = flat_y ? "Inf" : @sprintf("%.4g", sigmay)
    @printf("Gaussian: A=%.4g  δ=%.4g  σx=%.4g  σy=%s  center=(%.3g,%.3g)  n_img=%d  cs=%.4g  η/s=%.4g\n",
            A, delta, sigmax, sigmay_str, x0, y0, n_images, cs, η_by_s)
    return params, particles, cl, eos, cs
end

# ── Plotting ──────────────────────────────────────────────────────────────────

# Plots particle scatter colored by entropy density and the x-slice kernel average.
# s0 and s0+A are shown as reference lines on the slice panel.
function plot_gaussian(particles, params, cl; title_prefix="")
    fig, (ax1, ax2) = subplots(1, 2, figsize=(11, 4.5))
    isempty(title_prefix) || fig.suptitle(title_prefix)

    plot_particles(particles, params;
                  field=p -> p.e, clabel="e density",
                  title="e density", ax=ax1)

    xs, fs = plot_slice(particles, params, cl; field=p -> p.e, dim=:x)
    ax2.plot(xs, fs, color="C0", label="SPH")
    ax2.set_xlabel("x"); ax2.set_ylabel("e density")
    ax2.set_title("e density  x-slice at y=L/2")
    ax2.legend(fontsize=8)

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
        dt = min(dt, t_end - t)          # don't overshoot end

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
ha       = 2.0
L        = 1.0
A        = 1.0      # Gaussian amplitude (energy density units)
delta    = 1e-1     # floor energy density
sigmax   = 0.1
sigmay   = 0.14     # = sigmax * 1.4
#sigmay   = Inf     # = sigmax * 1.4

x0       = L/2     # Gaussian centre — corner of the box
y0       = L/2
n_images = 1        # ±1 periodic images in each direction (3×3 total)
η_by_s   = 0.002
t_end    = 2.2

params, particles, cl, eos, cs =
    initialize_gaussian(h, ha; L, A, delta, sigmax, sigmay,
                        x0, y0, n_images, η_by_s)

fig_ic = plot_gaussian(particles, params, cl;
                       title_prefix="IC: Gaussian  A=$A  σx=$sigmax  σy=$sigmay")
savefig(joinpath(@__DIR__, "testgaussian_ic.png"), dpi=150)
println("Saved testgaussian_ic.png")

@printf("t_end=%.4g\n", t_end)

n_steps, s_init, s_final, H_init, H_final =
    run_gaussian!(particles, params, cl, eos, t_end)

@printf("Entropy: Σs_0=%.10g  Σs_f=%.10g  rel.err=%.3g\n",
        s_init, s_final, abs(s_final - s_init) / s_init)
@printf("Energy:  H_0=%.10g  H_f=%.10g  rel.err=%.3g\n",
        H_init, H_final, abs(H_final - H_init) / abs(H_init))

fig_final = plot_gaussian(particles, params, cl;
                          title_prefix="Final: Gaussian stripe  t=$(round(t_end,digits=3))")
savefig(joinpath(@__DIR__, "testgaussian_final.png"), dpi=150)
println("Saved testgaussian_final.png")

show()
