# changerep_dt.jl — dependence of the energy gain of the particle representation on the
# time step (docs/changerep.tex).  The 2D Gaussian at h = 0.03 is evolved to t = 3 with
# the default viscous Δt (CFL = 0.2) and with Δt halved twice; H(t) and S(t) are recorded
# after each full step for the three viscous steps.
#
# Run:  julia --project examples/changerep_dt.jl  → docs/figs/changerep_dt.png
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
let src = read(joinpath(@__DIR__, "testgaussian.jl"), String)
    body = split(src, "# ── Main")[1]
    body = replace(body, r"^using PyPlot\n"m => "", r"^include\(.*\)\n"m => "")
    include_string(Main, body)
end

function full_step!(kind, ps, params, cl, dt, eos, dotm)
    kind === :smoothed && return generic_step!(ps, params, cl, dt, eos, dotm)
    D! = kind === :first ? viscous_step_first! : viscous_step_exact!
    D!(ps, params, cl, dt / 2, eos)
    leapfrog_step!(ps, params, cl, dt, eos, dotm)
    D!(ps, params, cl, dt / 2, eos)
end
Hof(ps, params, cl, eos) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
                            sum(p.s * p.H_over_S for p in ps))

h, t_end, t_late = 0.03, 3.0, 2.5
cfls  = (0.2, 0.1, 0.05)
kinds = ((:smoothed, "bond form, smoothed b"), (:first, "particle rep., first order"),
         (:exact, "particle rep., exact back map"))
res = Dict{Tuple{Symbol,Float64},Any}()
for (kind, lab) in kinds, cfl in cfls
    params, ps, cl, eos, cs = initialize_gaussian(h, 2.0; L=1.0, A=1.0, delta=1e-1, sigmax=0.1,
                                                  sigmay=0.14, x0=0.5, y0=0.5, n_images=1, η_by_s=0.002)
    params.dissipation = :bond
    dotm = zeros(SVector{2,Float64}, params.N)
    H0 = Hof(ps, params, cl, eos);  S0 = sum(p.s for p in ps)
    ts, dH, dS = [0.0], [0.0], [0.0];  t = 0.0
    while t < t_end - 1e-12
        dt = min(compute_dt(ps, params, eos; mode=:viscous, CFL=cfl), t_end - t)
        full_step!(kind, ps, params, cl, dt, eos, dotm);  t += dt
        push!(ts, t);  push!(dH, Hof(ps, params, cl, eos) / H0 - 1);  push!(dS, sum(p.s for p in ps) / S0 - 1)
    end
    late = ts .> t_late
    res[(kind, cfl)] = (ts, dH, dS)
    @printf("%-30s CFL = %.3f  %5d steps   late-time ΔH/H = %+.3e   ΔS/S = %+.3e\n",
            lab, cfl, length(ts) - 1, sum(dH[late]) / count(late), dS[end])
end

fig, axs = subplots(2, 3, figsize=(20, 9))
for (j, (kind, lab)) in enumerate(kinds), (i, cfl) in enumerate(cfls)
    ts, dH, dS = res[(kind, cfl)]
    c = get_cmap("viridis")((i - 1) / (length(cfls) - 1))
    axs[1, j].plot(ts, dH, "-", color=c, lw=1.2, label=@sprintf("Δt × %g", cfl / 0.2))
    axs[2, j].plot(ts, dS, "-", color=c, lw=1.2, label=@sprintf("Δt × %g", cfl / 0.2))
    if i == length(cfls)
        axs[1, j].axhline(0, color="0.5", lw=0.6);  axs[1, j].set_ylim(-0.002, 0.016)
        axs[1, j].set_title(lab * @sprintf(", h = %g", h));  axs[1, j].set_ylabel("H(t)/H(0) − 1")
        axs[2, j].set_ylabel("S(t)/S(0) − 1");  axs[2, j].set_xlabel("t")
        axs[1, j].legend(fontsize=8);  axs[2, j].legend(fontsize=8)
    end
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "changerep_dt.png")
savefig(out, dpi=130); println("Saved ", out)
