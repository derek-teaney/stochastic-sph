# changerep_gaussian_lines.jl — the 2D Gaussian of testgaussian.jl, evolved with the viscous
# step on the smoothed b (generic_step!) and in the particle representation
# (dissipative_step_changerep_exact!, docs/changerep.tex).  Slices of the energy density
# through the centre, along x and y, at several times; both schemes on the same axes.
#
# Run:  julia --project examples/changerep_gaussian_lines.jl  → docs/figs/changerep_gaussian_lines.png
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
let src = read(joinpath(@__DIR__, "testgaussian.jl"), String)
    body = split(src, "# ── Main")[1]
    body = replace(body, r"^using PyPlot\n"m => "", r"^include\(.*\)\n"m => "")
    include_string(Main, body)
end

function particle_step!(ps, params, cl, dt, eos, dotm)
    dissipative_step_changerep_grad!(ps, params, cl, dt / 2, eos)
    leapfrog_step!(ps, params, cl, dt, eos, dotm)
    dissipative_step_changerep_grad!(ps, params, cl, dt / 2, eos)
end

h, ha  = 0.015, 2.0
η_by_s = 0.002
times  = (0.0, 0.2, 0.4, 0.7, 1.0, 1.5, 2.2)

function evolve(stepfn)
    params, ps, cl, eos, cs = initialize_gaussian(h, ha; L=1.0, A=1.0, delta=1e-1, sigmax=0.1,
                                                  sigmay=0.14, x0=0.5, y0=0.5, n_images=1, η_by_s)
    params.dissipation = :bond
    dotm = zeros(SVector{2,Float64}, params.N)
    slices = Dict{Float64,Any}()
    snap!(t) = (thermalize!(ps, params, cl, eos);
                slices[t] = (plot_slice(ps, params, cl; field=p -> p.e, dim=:x),
                             plot_slice(ps, params, cl; field=p -> p.e, dim=:y)))
    t = 0.0;  snap!(0.0)
    for tn in times[2:end]
        while t < tn - 1e-12
            dt = min(compute_dt(ps, params, eos; mode=:viscous, CFL=0.2), tn - t)
            stepfn(ps, params, cl, dt, eos, dotm)
            t += dt
        end
        snap!(tn);  @printf("  t = %.2f\n", tn)
    end
    return slices
end

println("smoothed b");             res_b  = evolve(generic_step!)
println("particle representation"); res_pr = evolve(particle_step!)

fig, axs = subplots(2, 2, figsize=(15, 9))
cols = get_cmap("viridis")
for (j, (dim, lab)) in enumerate(((1, "x  (y = L/2)"), (2, "y  (x = L/2)")))
    for (i, t) in enumerate(times)
        c = cols((i - 1) / (length(times) - 1))
        xb, eb = res_b[t][dim];  xp, ep = res_pr[t][dim]
        axs[1, j].plot(xb, eb, "-", color=c, lw=1.4, label=@sprintf("t = %.1f", t))
        axs[1, j].plot(xp, ep, "--", color=c, lw=1.4)
        i > 1 && axs[2, j].plot(xb, ep .- eb, "-", color=c, lw=1.2, label=@sprintf("t = %.1f", t))
    end
    axs[1, j].set_xlabel(lab); axs[1, j].set_ylabel("e")
    axs[1, j].set_title("energy density along " * lab * ":  solid smoothed b,  dashed particle rep. (gradient-corrected)")
    axs[1, j].legend(fontsize=8)
    axs[2, j].axhline(0, color="k", lw=0.6)
    axs[2, j].set_xlabel(lab); axs[2, j].set_ylabel("e(particle rep.) − e(smoothed b)")
    axs[2, j].set_title("difference along " * lab)
    axs[2, j].legend(fontsize=8)
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "changerep_gaussian_lines.png")
savefig(out, dpi=130); println("Saved ", out)
