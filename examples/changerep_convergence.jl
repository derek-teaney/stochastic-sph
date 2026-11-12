# changerep_convergence.jl — convergence in h of the energy gain of the particle
# representation (docs/changerep.tex, "Long-time energy").  The 2D Gaussian of
# testgaussian.jl is evolved to t = 3 at several h; H(t) is recorded after each full
# step, and the late-time mean of ΔH/H (t > 2.5, after the gain has levelled off) is
# compared across h.  If the gain is a smooth-gradient effect it should scale as h².
#
# Run:  julia --project examples/changerep_convergence.jl  → docs/figs/changerep_convergence.png
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
let src = read(joinpath(@__DIR__, "testgaussian.jl"), String)
    body = split(src, "# ── Main")[1]
    body = replace(body, r"^using PyPlot\n"m => "", r"^include\(.*\)\n"m => "")
    include_string(Main, body)
end

function full_step!(kind, ps, params, cl, dt, eos, dotm)
    if kind === :smoothed || kind === :two_level
        params.dissipation = kind === :two_level ? :two_level : :bond
        return generic_step!(ps, params, cl, dt, eos, dotm)
    end
    params.dissipation = :bond
    D! = kind === :first ? viscous_step_first! :
         kind === :grad  ? viscous_step_grad! : viscous_step_exact!
    D!(ps, params, cl, dt / 2, eos)
    leapfrog_step!(ps, params, cl, dt, eos, dotm)
    D!(ps, params, cl, dt / 2, eos)
end
Hof(ps, params, cl, eos) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
                            sum(p.s * p.H_over_S for p in ps))

t_end, t_late = 3.0, 2.5
hs    = (0.045, 0.03, 0.0225, 0.015)
kinds = ((:two_level, "two-level", "C0"), (:smoothed, "bond, smoothed b", "C3"),
         (:exact, "particle, exact back map", "C2"), (:grad, "particle, gradient-corrected", "C4"))
curves  = Dict{Tuple{Symbol,Float64},Any}()
plateau = Dict{Tuple{Symbol,Float64},Float64}()
for h in hs, (kind, lab, c) in kinds
    params, ps, cl, eos, cs = initialize_gaussian(h, 2.0; L=1.0, A=1.0, delta=1e-1, sigmax=0.1,
                                                  sigmay=0.14, x0=0.5, y0=0.5, n_images=1, η_by_s=0.002)
    params.dissipation = :bond
    dotm = zeros(SVector{2,Float64}, params.N)
    H0 = Hof(ps, params, cl, eos);  S0 = sum(p.s for p in ps)
    ts, dH, dS = [0.0], [0.0], [0.0];  t = 0.0
    while t < t_end - 1e-12
        dt = min(compute_dt(ps, params, eos; mode=:viscous, CFL=0.2), t_end - t)
        full_step!(kind, ps, params, cl, dt, eos, dotm);  t += dt
        push!(ts, t);  push!(dH, Hof(ps, params, cl, eos) / H0 - 1);  push!(dS, sum(p.s for p in ps) / S0 - 1)
    end
    late = ts .> t_late
    curves[(kind, h)]  = (ts, dH, dS)
    plateau[(kind, h)] = sum(dH[late]) / count(late)
    @printf("h = %.4f  N = %6d  %-26s late-time ΔH/H = %+.3e   ΔS/S(t_end) = %+.4e\n", h, params.N, lab, plateau[(kind, h)], dS[end])
end

using DelimitedFiles
for (k, v) in curves
    writedlm(joinpath(@__DIR__, "compare", @sprintf("conv_%s_h%g.txt", k[1], k[2])), hcat(v[1], v[2], v[3]))
end
fig, axs = subplots(2, 4, figsize=(26, 9.5))
for (j, kind) in enumerate((:two_level, :smoothed, :exact, :grad))
    ax = axs[1, j];  axS = axs[2, j]
    for (i, h) in enumerate(hs)
        ts, dH, dS = curves[(kind, h)]
        axS.plot(ts, dS, "-", color=get_cmap("viridis")((i - 1) / (length(hs) - 1)), lw=1.2,
                 label=@sprintf("h = %g", h))
        ax.plot(ts, dH, "-", color=get_cmap("viridis")((i - 1) / (length(hs) - 1)), lw=1.2,
                label=@sprintf("h = %g", h))
    end
    if kind !== :two_level
        ts, dH, dS = curves[(:two_level, hs[end])]
        ax.plot(ts, dH, "k:", lw=1, label=@sprintf("two-level, h = %g", hs[end]))
        axS.plot(ts, dS, "k:", lw=1, label=@sprintf("two-level, h = %g", hs[end]))
    end
    axS.set_xlabel("t"); axS.set_ylabel("S(t)/S(0) − 1"); axS.set_ylim(0, 0.07); axS.legend(fontsize=8)
    ax.axhline(0, color="0.5", lw=0.6);  ax.set_ylim(-0.002, 0.016)
    ax.set_xlabel("t"); ax.set_ylabel("H(t)/H(0) − 1")
    ax.set_title(Dict(:two_level => "two-level form", :smoothed => "bond form on the smoothed b",
                      :grad => "particle representation, gradient-corrected",
                      :exact => "particle representation, exact back map")[kind])
    ax.legend(fontsize=8)
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "changerep_convergence.png")
savefig(out, dpi=130); println("Saved ", out)

# late-time energy gain and final entropy against h
fig2, ax2 = subplots(1, 3, figsize=(21, 5))
ax = ax2[1]
for (kind, lab, c) in kinds
    y = [plateau[(kind, h)] for h in hs]
    ax.loglog(collect(hs), abs.(y), "o-", color=c, label=lab)
end
href = collect(hs)
ax.loglog(href, plateau[(:exact, hs[1])] .* (href ./ hs[1]) .^ 2, "k--", lw=1, label="∝ h²")
ax.loglog(href, plateau[(:exact, hs[1])] .* (href ./ hs[1]), "k:", lw=1, label="∝ h")
ax.set_xlabel("h"); ax.set_ylabel(@sprintf("late-time |ΔH/H|  (t > %g)", t_late))
ax.set_title("energy gain against resolution"); ax.legend(fontsize=8)
ax = ax2[2]
for (kind, lab, c) in kinds
    ax.plot(collect(hs), [curves[(kind, h)][3][end] for h in hs], "o-", color=c, label=lab)
end
ax.set_xlabel("h"); ax.set_ylabel(@sprintf("S(t = %g)/S(0) − 1", t_end))
ax.set_title("entropy produced against resolution"); ax.legend(fontsize=8)
ax = ax2[3]
for (kind, lab, c) in kinds
    kind === :two_level && continue
    sl = Float64[]
    for h in hs
        ts, dH, dS = curves[(kind, h)];  tr, dHr, _ = curves[(:two_level, h)]
        x = dS[ts .> 0.5];  y = (dH .- [dHr[argmin(abs.(tr .- t))] for t in ts])[ts .> 0.5]
        push!(sl, sum((x .- sum(x)/length(x)) .* (y .- sum(y)/length(y))) / sum((x .- sum(x)/length(x)).^2))
    end
    ax.loglog(collect(hs), abs.(sl), "o-", color=c, label=lab)
    @printf("%-28s slope of (ΔH/H − two-level) vs ΔS/S: %s\n", lab, join([@sprintf("%.3e", x) for x in sl], "  "))
end
ax.loglog(href, 0.11 .* (href ./ hs[1]) .^ 2, "k--", lw=1, label="∝ h²")
ax.set_xlabel("h"); ax.set_ylabel("|d(ΔH/H − two-level)/d(ΔS/S)|")
ax.set_title("energy excess per entropy produced"); ax.legend(fontsize=8)
tight_layout()
out2 = joinpath(@__DIR__, "..", "docs", "figs", "changerep_convergence_h.png")
savefig(out2, dpi=130); println("Saved ", out2)
