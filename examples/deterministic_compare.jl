# deterministic_compare.jl — two-level vs comoving bond dissipation in full
# deterministic hydro (generic_step!: ideal leapfrog + viscous step, no noise).
#
#   Part 1  viscous sound wave (testsound_viscous.jl setup) at two resolutions:
#           amplitude vs time against the analytic decay.
#   Part 2  relaxation of a random momentum field on a uniform fluid: the transverse
#           (shear) spectrum P_T(k,t)/P_T(k,0) against the physical e^{−2Γ_phys t},
#           Γ_phys = (η/w) k², for every wavelength at once.
#
# Run:  julia --project examples/deterministic_compare.jl
#       → deterministic_sound.png, deterministic_noise.png

using PyPlot
using StaticArrays
using LinearAlgebra
using Random
using Statistics
using Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))

# reuse the sound-wave initialization and time loop (everything above "# ── Main")
let src = read(joinpath(@__DIR__, "testsound_viscous.jl"), String)
    body = split(src, "# ── Main")[1]
    body = replace(body, r"^using PyPlot\n"m => "", r"^include\(.*\)\n"m => "")
    include_string(Main, body)
end

const SCHEMES = ((:two_level, "two-level", "C0"), (:bond, "comoving bond", "C3"))

# ══════════════════════════════════════════════════════════════ Part 1 ══════
println("Part 1: viscous sound wave")
fig1, axs1 = subplots(1, 2, figsize=(12, 4.6))
for (j, h) in enumerate((0.03, 0.06))
    ax = axs1[j]
    local kk, Γs, Tp
    for (scheme, lab, c) in SCHEMES
        params, particles, cl, eos, kx, ky, khat, k, cs, ωR, T_period, T_rel_0, Γ_s =
            initialize_viscous_sound(h, 2.0, 1.2, 1.6, 2.0, 0.02, 1/3, 2, 1, 0.02, 0.0)
        params.dissipation = scheme
        t, amp = run_viscous_sound!(particles, params, cl, eos, 2T_period, ωR)
        ax.plot(t ./ T_period, amp ./ amp[1], "-", color=c, lw=1.5, label=lab)
        kk, Γs, Tp = k, Γ_s, T_period
        @printf("  h=%.2f (kh=%.2f)  %-14s amp(2T)/amp0 = %.4f\n", h, k * h, lab, amp[end] / amp[1])
    end
    tt = range(0, 2Tp, length=200)
    ana = exp.(-Γs * kk^2 .* tt ./ 2)
    ax.plot(tt ./ Tp, ana, "k--", lw=1.2, label="analytic e^{−Γ_s k² t/2}")
    @printf("  h=%.2f  analytic amp(2T)/amp0 = %.4f\n", h, ana[end])
    ax.set_xlabel("t / T_period");  ax.set_ylabel("max|M_x| / initial")
    ax.set_title(@sprintf("viscous sound wave, h = %.2f  (kh = %.2f)", h, kk * h))
    ax.legend(fontsize=8)
end
tight_layout()
savefig(joinpath(@__DIR__, "deterministic_sound.png"), dpi=120)
println("Saved deterministic_sound.png")

# ══════════════════════════════════════════════════════════════ Part 2 ══════
println("\nPart 2: relaxation of a random momentum field")
ηs     = 1.0
snaps  = (25.0, 100.0, 400.0)

function noise_run(scheme; ηs=ηs, seed=3)
    eos = ConformalEOS(0.5, T_scale_2d_lq1, ηs, 0.0)
    params, ps, cl, eos = setup_ic(15.0, 2.55, eos, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                   L=150.0, lattice=:hex, d=2)
    params.dissipation = scheme
    N = params.N
    rng = Xoshiro(seed)
    amp = 1e-3 * (ps[1].e + ps[1].p) * params.vol          # v ~ 1e-3 per particle
    for p in ps; p.m = amp * SVector(randn(rng), randn(rng)); end
    shift = sum(p.m for p in ps) / N
    for p in ps; p.m -= shift; end
    thermalize!(ps, params, cl, eos)
    w  = ps[1].e + ps[1].p;  η = ηs * srel(eos, ps[1].e)
    dt = compute_dt(ps, params, eos; mode=:viscous, CFL=0.2)
    dotm = zeros(SVector{2,Float64}, N)
    # wavevectors inside the first zone, |k| ≤ 0.6 (no aliasing)
    L  = params.L
    ks = [SVector(2π * nx / L[1], 2π * ny / L[2]) for nx in -15:15, ny in -15:15]
    ks = [k for k in ks if 0 < norm(k) <= 0.6]
    PT(ps) = [begin
                  t̂ = SVector(-k[2], k[1]) / norm(k)
                  abs2(sum(dot(t̂, p.m) * cis(-dot(k, p.x)) for p in ps))
              end for k in ks]
    P0 = PT(ps);  out = Dict{Float64,Vector{Float64}}()
    t = 0.0;  nsteps = round(Int, maximum(snaps) / dt)
    for step in 1:nsteps
        generic_step!(ps, params, cl, dt, eos, dotm)
        rebuild_cell_list!(cl, ps)
        t += dt
        for ts in snaps
            abs(t - ts) < dt / 2 && (out[ts] = PT(ps) ./ P0)
        end
    end
    return ks, out, η / w, dt
end

edges = range(0, 0.6, length=21);  kc = (edges[1:end-1] .+ edges[2:end]) ./ 2
function binned(ks, y)
    km = norm.(ks)
    [(s = (km .>= edges[i]) .& (km .< edges[i+1]); any(s) ? mean(y[s]) : NaN) for i in 1:length(kc)]
end

res = Dict(scheme => noise_run(scheme) for (scheme, _, _) in SCHEMES)
fig2, axs2 = subplots(1, length(snaps), figsize=(15, 4.6), sharey=true)
for (j, ts) in enumerate(snaps)
    ax = axs2[j]
    for (scheme, lab, c) in SCHEMES
        ks, out, ηw, dt = res[scheme]
        ax.semilogy(kc, binned(ks, out[ts]), "o-", color=c, ms=3, lw=1.4, label=lab)
    end
    ηw = res[:bond][3]
    kf = range(0, 0.6, length=200)
    ax.semilogy(kf, exp.(-2ηw .* kf .^ 2 .* ts), "k--", lw=1.2, label="physics e^{−2(η/w)k²t}")
    ax.axvline(1 / 15.0, color="0.5", ls=":", lw=1)
    ax.set_ylim(1e-6, 3);  ax.set_xlabel("|k|")
    ax.set_title(@sprintf("t = %g   (Γ_phys(k=0.2)·t = %.2f)", ts, ηw * 0.04 * ts))
    j == 1 && ax.set_ylabel("P_T(k,t) / P_T(k,0)  (transverse momentum)")
    ax.legend(fontsize=8, loc="lower left")
    for (scheme, lab, _) in SCHEMES
        ks, out, _, _ = res[scheme]
        b = binned(ks, out[ts])
        @printf("  t=%5g  %-14s P_T ratio at |k|≈0.105: %.3e   0.195: %.3e   0.315: %.3e   0.495: %.3e\n",
                ts, lab, b[4], b[7], b[11], b[17])
    end
    @printf("  t=%5g  physics                        %.3e          %.3e          %.3e          %.3e\n",
            ts, exp(-2ηw * kc[4]^2 * ts), exp(-2ηw * kc[7]^2 * ts), exp(-2ηw * kc[11]^2 * ts), exp(-2ηw * kc[17]^2 * ts))
end
@printf("  dt = %.3g (viscous CFL), η/w = %.3g\n", res[:bond][4], res[:bond][3])
tight_layout()
savefig(joinpath(@__DIR__, "deterministic_noise.png"), dpi=120)
println("Saved deterministic_noise.png")
