using PyPlot
using StaticArrays
using Random
using LinearAlgebra
using Statistics
using Printf
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
include(joinpath(@__DIR__, "..", "src", "stochastic.jl"))

# ═════════════════════════════════════════════════════════════════════════════
# Step 4: ideal symplectic step + one per-particle Metropolis sweep per step, on
# a QUIESCENT uniform background.  2D conformal fluid (cs² = 1/2, Kappa2D).  No
# free amplitude — the covariance is forced — so this is a pass/fail stability +
# FDT test, not a calibration:
#
#   • no blow-up: (e, T, γ, Teff) finite and sub-luminal for the whole run,
#     including a deliberate deep-tail single-particle kick (stress_probe)
#   • Σ_a m_a conserved to machine precision
#   • H = Σ s_a ℰ_a drifts up only slowly — the linearized-Δs heating,
#     rate ~ 1/(s_a τ_visc) — not a fast runaway
#   • Σ s_a non-decreasing
#   • started AT the FDT equilibrium, rms(M) stays there (stationarity)
#   • started from m=0, rms(M) climbs but its growth decelerates — no unbounded
#     √t grid-scale (odd-even) runaway
#   • the m field stays isotropic (no lattice / scan-order imprint)
#
# Parameter scheme (sph2 §"First stochastic steps"), units ℓ_Q = T_0 = 1:
#   ℓ_Q = 1/√s0  (interparticle / quantum length; s0 = entropy density)
#   ℓ   = √2 (η/s)/T_rel   (shear length / mean free path, cs = 1/√2)
#   η̂   = ℓ/ℓ_Q            (dimensionless viscosity, "go crazy ~ 10")
#   a ~ few·ℓ_Q ≲ h ~ ℓ ≲ L ,  so h/a ~ η̂
# Per particle s_a = s0·vol; thermal velocity ~ 1/√s_a → need s_a ≳ 10.
# ═════════════════════════════════════════════════════════════════════════════

# ── physical parameters (dimensionless, ℓ_Q = T_0 = 1) ──────────────────────
const CS2   = 1.0 / 2.0
const S0    = 1.0                       # entropy density  ⇒  ℓ_Q = 1
const ηHAT  = 10.0                      # ℓ/ℓ_Q
const ηBYS  = ηHAT / sqrt(2.0)          # η̂ = √2 (η/s) at T_rel = 1, s0 = 1
const ℓ     = ηHAT                      # = η̂·ℓ_Q
# numerical: a ≈ 5.9 ℓ_Q gives s_a ≈ s0·a²·(√3/2) ≈ 30; h/a ≈ 2.55 keeps SPH sane
const HSPH  = 15.0                      # ≈ 1.5 ℓ  (short modes mildly over-damped)
const HARAT = 2.55                      # h/a  ⇒  a ≈ 5.9, s_a ≈ 30
const LBOX  = 150.0                     # L/h = 10, L/ℓ = 15  (needs ncx ≥ 3)

function build_uniform(; h=HSPH, ha=HARAT, L=LBOX, s0=S0,
                         cs2v=CS2, η_by_s=ηBYS, ζ_by_s=0.0)
    eos0 = ConformalEOS(cs2v, T_scale_2d_lq1, η_by_s, ζ_by_s)
    params, particles, cl, eos = setup_ic(h, ha, eos0,
        (x, y) -> s0, (x, y) -> SVector(0.0, 0.0); L, lattice=:hex, d=2)
    thermalize!(particles, params, cl, eos)
    return params, particles, cl, eos
end

# per-component FDT rms of m_a, and rms(|m_a|)/vol for the M-density diagnostic
function fdt_scales(particles, params)
    p1  = particles[1]
    σm  = sqrt((p1.e + p1.p) * p1.T * params.vol)
    return σm, sqrt(2) * σm / params.vol
end

state_ok(particles) = all(p -> isfinite(p.e) && p.e > 0 && isfinite(p.T) &&
                               p.T > 0 && p.γ < 20 && isfinite(p.Teff), particles)

function run_uniform(; n_steps, CFL=0.2, seed=1, m_init=:zero, move=true,
                       verbose=false, kw...)
    params, particles, cl, eos = build_uniform(; kw...)
    dt   = compute_dt(particles, params, eos; mode=:ideal, CFL)
    vol  = params.vol
    dotm = zeros(SVector{2,Float64}, params.N)
    σm, fdt_rmsM = fdt_scales(particles, params)
    ℓ_Q = 1 / sqrt(params.vol * (params.N / prod(params.L)))   # ≈ 1/√s0

    if m_init === :fdt
        r0 = Xoshiro(hash((seed, :init)))
        for p in particles
            p.m = SVector(σm * randn(r0), σm * randn(r0))
        end
        shift = sum(p.m for p in particles) / params.N        # zero total momentum
        for p in particles; p.m = p.m - shift; end
        thermalize!(particles, params, cl, eos)
    end

    rngs = make_phys_rng(params.N, seed)                       # advances across sweeps
    Σm0  = sum(p.m for p in particles)
    H0   = sum(p.s * p.H_over_S for p in particles)
    S0v  = sum(p.s for p in particles)

    H = (t=Float64[], acc=Float64[], dΣm=Float64[], ΔH=Float64[], ΔS=Float64[],
         rmsM=Float64[], rmsMx=Float64[], rmsMy=Float64[], maxv=Float64[],
         blew=Ref(false), blewstep=Ref(0))

    t = 0.0
    for step in 1:n_steps
        if move
            leapfrog_step!(particles, params, cl, dt, eos, dotm)   # ideal only
            rebuild_cell_list!(cl, particles)
        end
        smooth_to_particles!(particles, params, cl)            # refresh + freeze
        update_thermodynamics!(particles, eos)
        update_velocities!(particles, params, cl)

        if !state_ok(particles)
            H.blew[] = true; H.blewstep[] = step
            verbose && @printf("  *** blew up at step %d (t=%.4g)\n", step, t)
            break
        end

        nacc, ntry, _ = particle_sweep!(particles, params, cl, eos, dt, rngs)
        apply_sweep!(particles)
        t += dt

        mx = Float64[p.m[1] for p in particles];  my = Float64[p.m[2] for p in particles]
        push!(H.t, t)
        push!(H.acc,  nacc / max(ntry, 1))
        push!(H.dΣm,  norm(sum(p.m for p in particles) - Σm0) / (σm * sqrt(params.N)))
        push!(H.ΔH,   (sum(p.s * p.H_over_S for p in particles) - H0) / abs(H0))
        push!(H.ΔS,   (sum(p.s for p in particles) - S0v) / S0v)
        push!(H.rmsM,  sqrt(mean(mx.^2 .+ my.^2)) / vol)
        push!(H.rmsMx, sqrt(mean(mx.^2)) / vol)
        push!(H.rmsMy, sqrt(mean(my.^2)) / vol)
        push!(H.maxv,  maximum(norm(p.v_over_S) * p.S_smooth for p in particles))

        verbose && step % 500 == 0 && @printf(
            "  step %5d  t=%7.1f  acc=%.4f  |ΔΣm|=%.1e  ΔH/H=%+.2e  ΔS/S=%+.2e  rms(M)/FDT=%.3f  max|v|=%.3f\n",
            step, t, H.acc[end], H.dΣm[end], H.ΔH[end], H.ΔS[end],
            H.rmsM[end] / fdt_rmsM, H.maxv[end])
    end
    return (; H, params, particles, cl, dt, vol, σm, fdt_rmsM, eos, ℓ_Q)
end

# ── stress probe: how big a single-particle kick can the scheme absorb? ──────
# randn essentially never exceeds ~6σ; over a full run (~N·n_steps ≈ 3e6 draws)
# the largest draw is ~5.3σ.  Require survival well past that (≤ 20σ); report the
# rest — where |v|→1 the EOS legitimately has no sub-luminal solution.
function stress_probe(; Kσ_list=(6.0, 12.0, 20.0, 40.0, 100.0),
                        n_relax=300, CFL=0.2, seed=7, kw...)
    for Kσ in Kσ_list
        params, particles, cl, eos = build_uniform(; kw...)
        dt   = compute_dt(particles, params, eos; mode=:ideal, CFL)
        dotm = zeros(SVector{2,Float64}, params.N)
        σm, _ = fdt_scales(particles, params)

        k = params.N ÷ 2 + 3
        particles[k].m = particles[k].m + SVector(Kσ * σm, 0.4 * Kσ * σm)
        smooth_to_particles!(particles, params, cl)
        update_thermodynamics!(particles, eos)
        update_velocities!(particles, params, cl)
        ok0  = state_ok(particles)
        v_k0 = maximum(norm(p.v_over_S) * p.S_smooth for p in particles)

        rngs = make_phys_rng(params.N, seed)
        blew = !ok0;  blewstep = 0
        if ok0
            for step in 1:n_relax
                leapfrog_step!(particles, params, cl, dt, eos, dotm)
                rebuild_cell_list!(cl, particles)
                smooth_to_particles!(particles, params, cl)
                update_thermodynamics!(particles, eos)
                update_velocities!(particles, params, cl)
                if !state_ok(particles); blew = true; blewstep = step; break; end
                particle_sweep!(particles, params, cl, eos, dt, rngs)
                apply_sweep!(particles)
            end
        end
        maxv1 = state_ok(particles) ? maximum(norm(p.v_over_S) * p.S_smooth for p in particles) : NaN
        @printf("  K=%-7g  max|v|: kick=%.3f end=%.3f   EOS ok=%-5s  →  %s\n",
                Kσ, v_k0, maxv1, ok0,
                blew ? "BLEW (step $blewstep)" : "survived $n_relax steps")
        Kσ <= 20.0 && @assert !blew  "scheme blew up on a $(Kσ)σ kick (randn never gets there)"
    end
    println("  stress probe: absorbs realistic-tail kicks (≤20σ) without blow-up  ✓")
end

# ═════════════════════════════════════════════════════════════════════════════

println("=" ^ 70)
println("teststochastic_run.jl — Step 4: uniform-background stability (2D fluid)")
println("=" ^ 70)
let
    p, pa, _, _ = build_uniform()
    @printf("params:  cs²=%.3g  η̂=%.3g  η/s=%.3g  s0=%.3g  ℓ=%.3g  h=%.3g  a=%.3g  h/a=%.3g\n",
            CS2, ηHAT, ηBYS, S0, ℓ, p.h, p.a, p.h/p.a)
    @printf("         N=%d  L=%.3g  vol=%.4g  s_a=%.3g  T_rel=%.4g  e+p=%.4g  v_th~%.3f\n",
            p.N, p.L[1], p.vol, S0*p.vol, pa[1].T*pa[1].γ, pa[1].e+pa[1].p,
            1/sqrt(S0*p.vol))
end

println("\n[stress] deep-tail single-particle kick scan")
stress_probe()

println("\n[A] started AT the FDT equilibrium — does rms(M) stay put?")
rA = run_uniform(; n_steps=3000, m_init=:fdt, seed=11, verbose=true)
@assert !rA.H.blew[]  "FDT-start run blew up at step $(rA.H.blewstep[])"
let H = rA.H, i1 = length(rA.H.t) ÷ 2
    r_mean = mean(H.rmsM[i1:end]) / rA.fdt_rmsM
    r_x, r_y = mean(H.rmsMx[i1:end]), mean(H.rmsMy[i1:end])
    aniso = abs(r_x - r_y) / ((r_x + r_y) / 2)
    @printf("  rms(M)/FDT (2nd half) = %.3f   anisotropy = %.3f   acc = %.4f\n",
            r_mean, aniso, mean(H.acc[i1:end]))
    @printf("  max|Σm−Σm0|/(σm√N) = %.2e   max|ΔH/H| = %.2e   ΔS/S final = %+.2e   max|v| = %.3f\n",
            maximum(H.dΣm), maximum(abs, H.ΔH), H.ΔS[end], maximum(H.maxv))
    @assert 0.7 < r_mean < 1.4     "rms(M) left the FDT equilibrium (ratio $r_mean)"
    @assert aniso < 0.15          "m field anisotropic (lattice / scan imprint)"
    @assert maximum(H.dΣm) < 1e-10 "total momentum not conserved"
    @assert maximum(abs, H.ΔH) < 0.05  "energy drift > 5%"
    @assert H.ΔS[end] > -1e-9     "entropy decreased"
    @assert maximum(H.maxv) < 0.9 "went near-luminal"
    println("  [A] FDT equilibrium is stationary  ✓")
end

println("\n[B] started from m=0 — climbs, then growth decelerates (no √t runaway)")
rB = run_uniform(; n_steps=4000, m_init=:zero, seed=22, verbose=true)
@assert !rB.H.blew[]  "m=0 run blew up at step $(rB.H.blewstep[])"
let H = rB.H, n = length(rB.H.t)
    early = mean(H.rmsM[n÷8  : n÷4]) / rB.fdt_rmsM
    mid   = mean(H.rmsM[3n÷8 : n÷2]) / rB.fdt_rmsM
    late  = mean(H.rmsM[7n÷8 : n  ]) / rB.fdt_rmsM
    @printf("  rms(M)/FDT  early=%.3f  mid=%.3f  late=%.3f   late/mid=%.3f\n",
            early, mid, late, late / mid)
    @printf("  acc(2nd half)=%.4f   max|ΔH/H|=%.2e   max|v|=%.3f\n",
            mean(H.acc[n÷2:end]), maximum(abs, H.ΔH), maximum(H.maxv))
    @assert all(isfinite, H.rmsM)  "non-finite momentum field"
    @assert late > early           "rms(M) never grew from 0"
    @assert late / mid < 1.30      "rms(M) still on the √t runaway (late/mid=$(late/mid))"
    @assert late < 2.0             "rms(M) overshot FDT badly (late/FDT=$late)"
    @assert maximum(H.maxv) < 0.9  "went near-luminal"
    println("  [B] grid-scale runaway suppressed — growth decelerates  ✓")
end

println("\n" * "=" ^ 70)
println("teststochastic_run.jl — Step 4 passed.")
println("=" ^ 70)

# ── plots ────────────────────────────────────────────────────────────────────
fig, ax = subplots(2, 2, figsize=(11, 8))
fig.suptitle(@sprintf("Step 4 — 2D uniform background (η̂=%.2g, s_a=%.2g, dt=%.3g, N=%d)",
                      ηHAT, S0*rB.params.vol, rB.dt, rB.params.N))

for (r, lab, c) in ((rA, "FDT start", "C0"), (rB, "m=0 start", "C1"))
    ax[1,1].plot(r.H.t, r.H.rmsM ./ r.fdt_rmsM, color=c, label=lab)
end
ax[1,1].axhline(1.0, color="k", lw=1.0, ls=":", label="FDT")
ax[1,1].set_xlabel("t"); ax[1,1].set_ylabel("rms(M) / FDT")
ax[1,1].set_title("stationarity / approach to equilibrium"); ax[1,1].legend(fontsize=8)

ax[1,2].plot(rA.H.t, rA.H.acc, "C0"); ax[1,2].plot(rB.H.t, rB.H.acc, "C1")
ax[1,2].set_ylim(0.9, 1.005)
ax[1,2].set_xlabel("t"); ax[1,2].set_ylabel("acceptance"); ax[1,2].set_title("sweep acceptance")

ax[2,1].plot(rA.H.t, rA.H.ΔH, "C0", label="ΔH/H (FDT start)")
ax[2,1].plot(rB.H.t, rB.H.ΔH, "C1", label="ΔH/H (m=0 start)")
ax[2,1].plot(rB.H.t, rB.H.ΔS, "C1--", lw=0.8, label="ΔS/S (m=0)")
ax[2,1].axhline(0, color="k", lw=0.5)
ax[2,1].set_xlabel("t"); ax[2,1].set_ylabel("relative change")
ax[2,1].set_title("H drift (linearized-Δs heating), S ↑"); ax[2,1].legend(fontsize=8)

ax[2,2].semilogy(rA.H.t, max.(rA.H.dΣm, 1e-20), "C0")
ax[2,2].semilogy(rB.H.t, max.(rB.H.dΣm, 1e-20), "C1")
ax[2,2].set_xlabel("t"); ax[2,2].set_ylabel("|Σm(t) − Σm(0)| / (σm √N)")
ax[2,2].set_title("total momentum conserved")

tight_layout()
savefig(joinpath(@__DIR__, "teststochastic_run.png"), dpi=150)
println("Saved teststochastic_run.png")
