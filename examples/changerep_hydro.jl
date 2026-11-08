# changerep_hydro.jl — full deterministic hydro, D(Δt/2) H(Δt) D(Δt/2), with the viscous
# step D done either with the bond form on the smoothed b (generic_step!) or in the
# particle representation with the exact back map (dissipative_step_changerep_exact!).
# docs/changerep.tex.
#
#   Part 1  1D viscous sound wave (testsound_viscous_1d.jl setup): amplitude, H, S.
#   Part 2  1D random momentum field on a uniform fluid: spectrum P(k,t)/P(k,0) against
#           the physical e^{−Γ_s k² t}, and the energy bookkeeping.
#
# Run:  julia --project examples/changerep_hydro.jl  → docs/figs/changerep_hydro.png
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
let src = read(joinpath(@__DIR__, "testsound_viscous_1d.jl"), String)
    body = split(src, "# ── Main")[1]
    body = replace(body, r"^using PyPlot\n"m => "", r"^include\(.*\)\n"m => "")
    include_string(Main, body)
end

function step!(scheme, ps, params, cl, dt, eos, dotm)
    if scheme === :smoothed
        generic_step!(ps, params, cl, dt, eos, dotm)
    else
        D! = scheme === :grad ? dissipative_step_changerep_grad! : dissipative_step_changerep_exact!
        D!(ps, params, cl, dt / 2, eos)
        leapfrog_step!(ps, params, cl, dt, eos, dotm)
        D!(ps, params, cl, dt / 2, eos)
    end
end
Hof(ps, params, cl, eos) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
                            sum(p.s * p.H_over_S for p in ps))
Sof(ps) = sum(p.s for p in ps)
const SCHEMES = ((:smoothed, "bond, smoothed b", "C3"), (:changerep, "bond, particle rep (exact)", "C2"),
                  (:grad, "bond, particle rep (gradient-corrected)", "C4"))

fig, axs = subplots(1, 4, figsize=(22, 4.8))

# ══ Part 1: viscous sound wave ═══════════════════════════════════════════════
println("Part 1: 1D viscous sound wave")
for (scheme, lab, c) in SCHEMES
    params, ps, cl, eos, kx, ky, khat, k, cs, ωR, T_period, T_rel_0, Γ_s =
        initialize_viscous_sound_1d(0.01, 4.0, 1.0, 2.0, 0.02, 1/3, 2, 0.02, 0.0)
    params.dissipation = :bond
    dt = compute_dt(ps, params, eos; mode=:viscous)
    nst = ceil(Int, 2T_period / dt);  dt = 2T_period / nst
    dotm = zeros(SVector{2,Float64}, params.N)
    H0 = Hof(ps, params, cl, eos);  S0 = Sof(ps)
    amp0 = maximum(abs(p.M_smooth[1]) for p in ps)
    ts, amps, dH = Float64[0.0], Float64[1.0], Float64[0.0]
    for n in 1:nst
        step!(scheme, ps, params, cl, dt, eos, dotm)
        H = Hof(ps, params, cl, eos)
        push!(ts, n * dt / T_period); push!(amps, maximum(abs(p.M_smooth[1]) for p in ps) / amp0)
        push!(dH, (H - H0) / H0)
    end
    axs[1].plot(ts, amps, "-", color=c, lw=1.5, label=lab)
    axs[2].plot(ts, dH, "-", color=c, lw=1.5, label=lab)
    @printf("  %-28s amp(2T)/amp0 = %.5f  analytic %.5f   ΔH/H = %+.2e   ΔS/S = %+.2e   (%d steps)\n",
            lab, amps[end], exp(-Γ_s * k^2 * 2T_period / 2), dH[end], (Sof(ps) - S0) / S0, nst)
    if scheme === :changerep
        tt = range(0, 2, length=200)
        axs[1].plot(tt, exp.(-Γ_s * k^2 .* tt .* T_period ./ 2), "k--", lw=1, label="analytic")
    end
end
axs[1].set_xlabel("t / T_period"); axs[1].set_ylabel("max|M_x| / initial")
axs[1].set_title("1D viscous sound wave (h = 0.01, h/a = 4, η/s = 0.02)"); axs[1].legend(fontsize=8)
axs[2].set_xlabel("t / T_period"); axs[2].set_ylabel("(H − H₀)/H₀")
axs[2].set_title("sound wave: smoothed energy H"); axs[2].legend(fontsize=8)

# ══ Part 2: random momentum field ════════════════════════════════════════════
println("\nPart 2: 1D random momentum field")
ηs = 0.1
snaps = (0.5, 2.0, 8.0)                        # in units of 1/Γ_s(k = 1/h)
res = Dict{Symbol,Any}()
for (scheme, lab, c) in SCHEMES
    eos = ConformalEOS(1/3, 1.0, ηs, 0.0)
    params, ps, cl, eos = setup_ic(0.02, 4.0, eos, (x, y) -> 2.0, (x, y) -> SVector(0.0, 0.0);
                                   Lx=1.0, sim_dim=1)
    params.dissipation = :bond
    N, a, h = params.N, params.a, params.h
    rng = Xoshiro(3)
    w = ps[1].e + ps[1].p
    for p in ps; p.m = SVector(1e-3 * w * a * randn(rng), 0.0); end
    sh = sum(p.m for p in ps) / N;  for p in ps; p.m -= sh; end
    thermalize!(ps, params, cl, eos)
    Γs = 4/3 * ηs / (ps[1].T * ps[1].γ)            # Γ_s = κ/w
    τ  = h^2 / Γs                                  # 1/Γ_s at k = 1/h
    ns = 1:N ÷ 2;  ks = 2π .* ns ./ params.L[1]
    P(ps) = [abs2(sum(p.m[1] * cis(-k * p.x[1]) for p in ps)) for k in ks]
    P0 = P(ps);  H0 = Hof(ps, params, cl, eos);  S0 = Sof(ps);  T0 = ps[1].T
    KE0 = sum(p.m[1]^2 / (2w * a) for p in ps)
    dt = compute_dt(ps, params, eos; mode=:viscous)
    dotm = zeros(SVector{2,Float64}, N)
    out = Dict{Float64,Vector{Float64}}();  t = 0.0
    for n in 1:round(Int, maximum(snaps) * τ / dt)
        step!(scheme, ps, params, cl, dt, eos, dotm);  t += dt
        for sn in snaps
            abs(t - sn * τ) < dt / 2 && (out[sn] = P(ps) ./ P0)
        end
    end
    H = Hof(ps, params, cl, eos)
    res[scheme] = (ks, out, Γs, τ, a)
    @printf("  %-28s t = %g τ:  ΔH = %.3e   T ΔS = %.3e   KE₀ = %.3e   KE left = %.3e\n",
            lab, maximum(snaps), H - H0, T0 * (Sof(ps) - S0), KE0,
            sum(p.m[1]^2 / (2w * a) for p in ps))
end
# binned spectra at the first and last snapshots
nb = 25
for (j, sn) in enumerate((snaps[1], snaps[end]))
    ax = axs[2 + j]
    for (scheme, lab, c) in SCHEMES
        ks, out, Γs, τ, a = res[scheme]
        edges = range(0, π / a, length=nb + 1)
        kc = (edges[1:end-1] .+ edges[2:end]) ./ 2
        b = [mean(out[sn][(ks .>= edges[i]) .& (ks .< edges[i+1])]) for i in 1:nb]
        ax.semilogy(kc .* a, b, "o-", color=c, ms=3, lw=1.2, label=lab)
    end
    ks, out, Γs, τ, a = res[:changerep]
    kf = range(0, π / a, length=200)
    ax.semilogy(kf .* a, exp.(-Γs .* kf .^ 2 .* sn .* τ), "k--", lw=1, label="physics e^{−Γ_s k² t}")
    ax.set_ylim(1e-8, 3); ax.set_xlim(0, π)
    ax.set_xlabel("k a"); ax.set_ylabel("P(k,t) / P(k,0)")
    ax.set_title(@sprintf("random momentum, t = %g h²/Γ_s", sn)); ax.legend(fontsize=8, loc="lower left")
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "changerep_hydro.png")
savefig(out, dpi=130); println("Saved ", out)
