# changerep_onestep.jl — what a single viscous step does to the energy, as a function of
# its length (docs/changerep.tex).  The 2D Gaussian at h = 0.03 is evolved with the
# smoothed-b scheme to t = 0.3, where the flow is strong.  From that state, one viscous
# step of length τ is applied with each scheme, at fixed positions, for several τ, and
#   ΔH   (smoothed energy),  ΔΣe (particle energies),  ΔS (total entropy)
# are recorded.  A scheme that conserves H to first order has ΔH = O(τ²).
#
# Run:  julia --project examples/changerep_onestep.jl  → docs/figs/changerep_onestep.png
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "..", "src", "legacy", "load.jl"))   # two_level, particle_sweep (superseded)
let src = read(joinpath(@__DIR__, "testgaussian.jl"), String)
    body = split(src, "# ── Main")[1]
    body = replace(body, r"^using PyPlot\n"m => "", r"^include\(.*\)\n"m => "")
    include_string(Main, body)
end
Hof(ps, params, cl, eos) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
                            sum(p.s * p.H_over_S for p in ps))
Eof(ps, params, cl, eos) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
                            sum(particle_energy(p, eos) for p in ps))

params, ps0, cl, eos, cs = initialize_gaussian(0.03, 2.0; L=1.0, A=1.0, delta=1e-1, sigmax=0.1,
                                               sigmay=0.14, x0=0.5, y0=0.5, n_images=1, η_by_s=0.002)
params.dissipation = :bond
dotm = zeros(SVector{2,Float64}, params.N)
t = 0.0
while t < 0.3 - 1e-12
    dt = min(compute_dt(ps0, params, eos; mode=:viscous, CFL=0.2), 0.3 - t)
    generic_step!(ps0, params, cl, dt, eos, dotm);  global t += dt
end
dt0 = compute_dt(ps0, params, eos; mode=:viscous, CFL=0.2)
@printf("state at t = %.3f, default Δt = %.3e\n", t, dt0)

function apply!(kind, ps, τ)
    if kind === :two_level || kind === :smoothed
        params.dissipation = kind === :two_level ? :two_level : :bond
        viscous_step!(ps, params, cl, τ, eos)
    elseif kind === :first
        params.dissipation = :bond;  viscous_step_first!(ps, params, cl, τ, eos)
    else
        params.dissipation = :bond;  viscous_step_exact!(ps, params, cl, τ, eos)
    end
end

kinds = ((:two_level, "two-level", "C0"), (:smoothed, "bond, smoothed b", "C3"),
         (:first, "particle, first order", "C1"), (:exact, "particle, exact back map", "C2"))
τs = dt0 / 2 .* 2.0 .^ (0:-1:-6)                    # the half step of generic_step!, then halved
H0 = Hof(deepcopy(ps0), params, cl, eos)
E0 = Eof(deepcopy(ps0), params, cl, eos)
S0 = sum(p.s for p in ps0)
@printf("Σe − H = %.4e  (relative %.3e)\n", E0 - H0, (E0 - H0) / H0)
res = Dict{Symbol,Any}()
for (kind, lab, c) in kinds
    dH, dE, dS = Float64[], Float64[], Float64[]
    for τ in τs
        ps = deepcopy(ps0)
        apply!(kind, ps, τ)
        push!(dH, Hof(ps, params, cl, eos) - H0)
        push!(dE, Eof(deepcopy(ps), params, cl, eos) - E0)
        push!(dS, sum(p.s for p in ps) - S0)
    end
    res[kind] = (dH, dE, dS)
    println(lab)
    for (i, τ) in enumerate(τs)
        @printf("   τ/Δt = %.4f   ΔH/H = %+.3e   ΔΣe/H = %+.3e   ΔS/S = %+.3e   T·ΔS/|ΔH| = %.3g\n",
                τ / dt0, dH[i] / H0, dE[i] / H0, dS[i] / S0, abs(dS[i]) * ps0[1].T / max(abs(dH[i]), 1e-300))
    end
end

fig, axs = subplots(1, 3, figsize=(20, 5))
for (kind, lab, c) in kinds
    dH, dE, dS = res[kind]
    axs[1].loglog(τs ./ dt0, abs.(dH) ./ H0, "o-", color=c, label=lab)
    axs[2].loglog(τs ./ dt0, abs.(dE) ./ H0, "o-", color=c, label=lab)
    axs[3].loglog(τs ./ dt0, abs.(dS) ./ S0, "o-", color=c, label=lab)
end
x = τs ./ dt0
for ax in axs
    ax.loglog(x, 1e-4 .* x, "k:", lw=1, label="∝ τ")
    ax.loglog(x, 1e-4 .* x .^ 2, "k--", lw=1, label="∝ τ²")
    ax.set_xlabel("step length τ / Δt"); ax.legend(fontsize=8)
end
axs[1].set_ylabel("|ΔH|/H");  axs[1].set_title("smoothed energy, one viscous step")
axs[2].set_ylabel("|ΔΣe|/H"); axs[2].set_title("particle energies, one viscous step")
axs[3].set_ylabel("ΔS/S");    axs[3].set_title("entropy, one viscous step")
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "changerep_onestep.png")
savefig(out, dpi=130); println("Saved ", out)
