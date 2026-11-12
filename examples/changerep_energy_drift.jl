# changerep_energy_drift.jl — long-time behaviour of the smoothed energy H(t) for the 2D
# Gaussian of testgaussian.jl, measured after every full step D(Δt/2) H(Δt) D(Δt/2).
# The leapfrog conserves a nearby (shadow) Hamiltonian, so H oscillates; the question is
# whether it drifts secularly.  Cases: ideal (η = 0), viscous step on the smoothed b,
# particle representation to first order, and with the exact back map (docs/changerep.tex).
#
# Run:  julia --project examples/changerep_energy_drift.jl  → docs/figs/changerep_energy_drift.png
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
let src = read(joinpath(@__DIR__, "testgaussian.jl"), String)
    body = split(src, "# ── Main")[1]
    body = replace(body, r"^using PyPlot\n"m => "", r"^include\(.*\)\n"m => "")
    include_string(Main, body)
end

function full_step!(kind, ps, params, cl, dt, eos, dotm)
    if kind === :two_level || kind === :smoothed
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

h, t_end = 0.03, 4.0
cases = ((:two_level, "two-level", "C0"), (:smoothed, "bond, smoothed b", "C3"),
         (:exact, "particle rep., exact back map", "C2"), (:grad, "particle rep., gradient-corrected", "C4"))
res = Dict{Symbol,Any}()
for (kind, lab, c) in cases
    params, ps, cl, eos, cs = initialize_gaussian(h, 2.0; L=1.0, A=1.0, delta=1e-1, sigmax=0.1,
                                                  sigmay=0.14, x0=0.5, y0=0.5, n_images=1,
                                                  η_by_s=0.002)
    params.dissipation = :bond
    dotm = zeros(SVector{2,Float64}, params.N)
    H0 = Hof(ps, params, cl, eos);  S0 = sum(p.s for p in ps)
    ts, dH, dS = [0.0], [0.0], [0.0];  t = 0.0
    while t < t_end - 1e-12
        dt = min(compute_dt(ps, params, eos; mode=:viscous, CFL=0.2), t_end - t)
        full_step!(kind, ps, params, cl, dt, eos, dotm);  t += dt
        push!(ts, t);  push!(dH, Hof(ps, params, cl, eos) / H0 - 1);  push!(dS, sum(p.s for p in ps) / S0 - 1)
        abs(dH[end]) > 0.05 && (@printf("  %s: unstable at t = %.3f, stopped\n", lab, t); break)
    end
    res[kind] = (ts, dH, dS)
    n = length(ts);  late = ts .> t_end / 2
    @printf("%-30s %5d steps   ΔH/H at end %+.3e   max|ΔH/H| %.3e   mean ΔH/H (t>%.0f) %+.3e   ΔS/S %+.3e\n",
            lab, n - 1, dH[end], maximum(abs.(dH)), t_end / 2, sum(dH[late]) / max(count(late), 1), dS[end])
end

fig, axs = subplots(1, 2, figsize=(14, 4.8))
for (kind, lab, c) in cases
    ts, dH, dS = res[kind]
    axs[1].plot(ts, dH, "-", color=c, lw=1.2, label=lab)
    axs[2].plot(ts, dS, "-", color=c, lw=1.2, label=lab)
end
axs[1].axhline(0, color="0.5", lw=0.6)
axs[1].set_xlabel("t"); axs[1].set_ylabel("H(t)/H(0) − 1")
axs[1].set_title(@sprintf("2D Gaussian, h = %g: smoothed energy after each full step", h))
axs[1].legend(fontsize=8)
axs[2].set_xlabel("t"); axs[2].set_ylabel("S(t)/S(0) − 1"); axs[2].set_title("total entropy")
axs[2].legend(fontsize=8)
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "changerep_energy_drift.png")
savefig(out, dpi=130); println("Saved ", out)
