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
# Staged tests for the stochastic SPH sweep (per-particle Metropolis scheme).
#   Step 0 — accumulator fields + apply_sweep!
#   Step 1 — the cell-list neighbor stencil (the traversal used by the strain
#            pass and the sweep) vs O(N²) brute force
#   Step 2 — noise covariance: draw_xi = 2 T_rel κ  (continuum FDT, δ's stripped),
#            for both d=2 (deployed 2D fluid) and d=3
#   Step 3 — particle_sweep!: conservation, ΔS identity, order independence,
#            mean drift = two-level dissipative_force!, entropy production > 0
# (The per-bond metropolis_sweep! was deleted — its mean is the single-level
#  force, never the two-level one.)
# Run the file, or call step0_accumulators() / step1_stencil() / step2_noise_cov()
# / step3() individually.
# ═════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════ Step 0 ══════
# Particle.Δs / Particle.Δm accumulators + apply_sweep!

function step0_accumulators()
    println("\n[Step 0] Particle.Δs / Particle.Δm accumulators")

    pa = Particle(SVector(0.3, 0.4), 2.0, SVector(0.1, -0.2))
    @assert pa.Δs == 0.0
    @assert pa.Δm == zero(SVector{2,Float64})
    println("  fresh particle: Δs=$(pa.Δs)  Δm=$(pa.Δm)  ✓")

    particles = [pa, Particle(SVector(0.6, 0.1), 1.5)]
    particles[1].Δs = 0.05;  particles[1].Δm = SVector(0.01, -0.03)
    particles[2].Δs = -0.02; particles[2].Δm = SVector(-0.01, 0.03)

    s_before = [p.s for p in particles]
    m_before = [p.m for p in particles]
    Σm_before = sum(m_before)

    apply_sweep!(particles)

    @assert particles[1].s ≈ s_before[1] + 0.05
    @assert particles[2].s ≈ s_before[2] - 0.02
    @assert particles[1].m ≈ m_before[1] + SVector(0.01, -0.03)
    @assert particles[2].m ≈ m_before[2] + SVector(-0.01, 0.03)
    @assert all(p.Δs == 0.0 for p in particles)
    @assert all(p.Δm == zero(SVector{2,Float64}) for p in particles)
    @assert sum(p.m for p in particles) ≈ Σm_before   # the two Δm here cancel
    println("  apply_sweep! folds Δ into s,m and resets; Σm preserved  ✓")
end

# ═══════════════════════════════════════════════════════════════ Step 1 ══════
# The exact traversal used by compute_strain / dissipative_force! /
# particle_sweep!: cell_index → 3×3 stencil → particles_in_cell, cutoff r < 2h
# (minimum image).  This is the only check of the neighbor machinery that is
# independent of it (the sweep's other checks compare stencil against stencil).
# Requires ncx, ncy ≥ 3 — the production stencil double-counts on degenerate
# grids, which the deterministic code has always assumed away.

# neighbor multiset of particle a via the production stencil
function stencil_neighbors(a, particles, params, cl)
    ncx = cl.ncx;  ncy = cl.ncy
    L   = params.L
    r2c = (2 * params.h)^2
    out = Int[]
    ix, iy = cell_index(particles[a].x, cl)
    for dx in -1:1, dy in neighbor_range(ncy)
        jx = mod(ix + dx - 1, ncx) + 1
        jy = mod(iy + dy - 1, ncy) + 1
        for b in particles_in_cell(cl, jx, jy)
            b == a && continue
            rv = minimum_image(particles[a].x - particles[b].x, L)
            rv[1]^2 + rv[2]^2 < r2c && push!(out, b)
        end
    end
    return sort!(out)
end

# reference: same predicate, no cell list
function neighbors_bruteforce(a, particles, params)
    L   = params.L
    r2c = (2 * params.h)^2
    out = Int[]
    for b in eachindex(particles)
        b == a && continue
        rv = minimum_image(particles[a].x - particles[b].x, L)
        rv[1]^2 + rv[2]^2 < r2c && push!(out, b)
    end
    return out
end

# degree counting neighbors strictly inside the cutoff shell (r < 2h·(1−ϵ)):
# on an exact lattice a whole shell can sit at r = 2h where FP rounding makes
# the raw degree wobble — physically irrelevant (W, ∇W = 0 there).
function interior_degree(a, particles, params, nbrs)
    L   = params.L
    cut = ((2 * params.h) * (1 - 1e-9))^2
    count(nbrs) do b
        rv = minimum_image(particles[a].x - particles[b].x, L)
        rv[1]^2 + rv[2]^2 < cut
    end
end

function check_stencil(tag, particles, params, cl)
    rebuild_cell_list!(cl, particles)
    @assert cl.ncx >= 3 && cl.ncy >= 3  "stencil test needs ncx,ncy ≥ 3 ($tag)"
    degi = zeros(Int, length(particles))
    nnbr = 0
    for a in eachindex(particles)
        got = stencil_neighbors(a, particles, params, cl)
        @assert allunique(got)  "stencil visits a neighbor twice ($tag, a=$a)"
        @assert got == neighbors_bruteforce(a, particles, params)  "neighbor set mismatch ($tag, a=$a)"
        degi[a] = interior_degree(a, particles, params, got)
        nnbr   += length(got)
    end
    @printf("  %-22s N=%-6d ⟨nbrs⟩=%.1f  interior-deg %d–%d\n",
            tag, length(particles), nnbr/length(particles),
            minimum(degi), maximum(degi))
    return degi
end

function step1_stencil()
    println("\n[Step 1] neighbor stencil (as used by strain/sweep) vs O(N²) brute force")
    eos = ConformalEOS(1.0/3.0, 1.0, 0.0, 0.0)

    # (a) exact hex lattice, small 2D box — interior degree must be uniform
    params, particles, cl, _ = setup_ic(0.1, 2.0, eos,
        (x,y)->1.0, (x,y)->SVector(0.0,0.0); L=1.0, lattice=:hex)
    degi = check_stencil("hex exact", particles, params, cl)
    @assert all(==(degi[1]), degi)  "hex interior degree not uniform: $(extrema(degi))"
    println("  hex lattice interior degree uniform ($(degi[1]))  ✓")

    # (b) square lattice, rectangular box
    params, particles, cl, _ = setup_ic(0.08, 2.5, eos,
        (x,y)->1.0, (x,y)->SVector(0.0,0.0); Lx=1.3, Ly=0.9, lattice=:square)
    degi = check_stencil("square exact", particles, params, cl)
    @assert all(==(degi[1]), degi)  "square interior degree not uniform: $(extrema(degi))"
    println("  square lattice interior degree uniform ($(degi[1]))  ✓")

    # (c) jittered lattice — degree no longer uniform, but must match brute force
    params, particles, cl, _ = setup_ic(0.1, 2.0, eos,
        (x,y)->1.0, (x,y)->SVector(0.0,0.0); L=1.0, lattice=:hex, jitter=0.8)
    check_stencil("hex jitter=0.8", particles, params, cl)

    # (d) after a leapfrog_step! the stencil must still match brute force
    params, particles, cl, eosv = setup_ic(0.1, 2.0, ConformalEOS(1.0/3.0, 1.0, 0.05, 0.0),
        (x,y)->1.0 + 0.1*cos(2π*x), (x,y)->SVector(0.0, 0.02*sin(2π*x)); L=1.0, lattice=:hex)
    dotm = zeros(SVector{2,Float64}, params.N)
    leapfrog_step!(particles, params, cl, 0.01, eosv, dotm)
    check_stencil("after leapfrog_step!", particles, params, cl)
    println("  post-move rebuild consistent  ✓")
end

# ═══════════════════════════════════════════════════════════════ Step 2 ══════
# kappa_tensor(pa,eos) returns T_rel κ, so 2·kappa_tensor = 2 T_rel κ.
#   generate_noise = ξ̂ :  ⟨ξ̂ ξ̂⟩ = κ        = kappa_tensor / T_rel
#   draw_xi        = √(2 T_rel) ξ̂ :  ⟨ξ ξ⟩ = 2 T_rel κ = 2·kappa_tensor
# Run for both kinematic dimensions — the deployed 2D fluid uses d=2 / Kappa2D
# / cs²=1/2, and the d=2 noise path (generate_landau_noise d=2 branch) must
# reproduce kappa_shear_geom_2d just as the d=3 path reproduces the 3D one.

# sample covariance Ĉ[i,j,l,m] = ⟨ξ[i,j] ξ[l,m]⟩ over nsamp draws
function sample_cov(draw, nsamp)
    C = zeros(2,2,2,2)
    for _ in 1:nsamp
        ξ = draw()
        for i in 1:2, j in 1:2, l in 1:2, m in 1:2
            C[i,j,l,m] += ξ[i,j] * ξ[l,m]
        end
    end
    return C ./ nsamp
end

const IDX6 = ((1,1,1,1,"xxxx"), (1,1,1,2,"xxxy"), (1,1,2,2,"xxyy"),
              (1,2,1,2,"xyxy"), (1,2,2,2,"xyyy"), (2,2,2,2,"yyyy"))

# a uniform-flow state (all particles identical) with 2-momentum (Mx,My)
function uniform_flow_particle(eos, Mx, My; d=3)
    _, particles, _, _ = setup_ic(0.1, 3.0, eos,
        (x,y) -> 1.0, (x,y) -> SVector(Mx, My); L=1.0, lattice=:hex, d)
    return particles[1]
end

function check_noise_cov(; d, cs2v, kmodel, nsamp=4_000_000, tol=0.01)
    η_by_s = 0.15;  ζ_by_s = 0.04
    eos    = ConformalEOS(cs2v, 1.0, η_by_s, ζ_by_s)
    rng    = MersenneTwister(20240827)
    @printf("  d=%d  cs²=%.3g  (%s)\n", d, cs2v, nameof(typeof(kmodel)))
    for (Mx, My) in ((0.0, 0.0), (0.6, 0.0), (2.0, 0.0), (1.2, 0.8))
        pa   = uniform_flow_particle(eos, Mx, My; d)
        vx   = pa.v_over_S[1] * pa.S_smooth
        vy   = pa.v_over_S[2] * pa.S_smooth
        Trel = pa.T * pa.γ

        K    = 2 .* kappa_tensor(pa, eos, kmodel)          # FDT target 2 T_rel κ
        Craw = sample_cov(() -> generate_noise(rng, vx, vy, pa.γ,
                    η_by_s*srel(eos,pa.e), ζ_by_s*srel(eos,pa.e), cs2v; d), nsamp)
        Cxi  = sample_cov(() -> draw_xi(rng, pa, eos, cs2v, d), nsamp)

        Knorm = sqrt(sum(K[i,j,l,m]^2 for i in 1:2, j in 1:2, l in 1:2, m in 1:2))
        raw_ratio = Float64[];  xi_err = 0.0
        for (i,j,l,m,_) in IDX6
            abs(K[i,j,l,m]) > 1e-10 && push!(raw_ratio, Craw[i,j,l,m]/K[i,j,l,m])
            xi_err = max(xi_err, abs(Cxi[i,j,l,m] - K[i,j,l,m]))
        end
        raw_spread = maximum(raw_ratio) - minimum(raw_ratio)
        @printf("    v=(%.2f,%.2f) Trel=%.3f | raw C/2κ = %.4f±%.4f (1/2Trel=%.4f) | draw_xi max|C−2κ|/‖2κ‖=%.4f\n",
                vx, vy, Trel, sum(raw_ratio)/length(raw_ratio), raw_spread/2,
                1/(2Trel), xi_err/Knorm)
        @assert raw_spread < 3tol   "raw covariance not ∝ one constant (d=$d, v=($vx,$vy))"
        @assert abs(sum(raw_ratio)/length(raw_ratio) - 1/(2Trel)) < tol/Trel  "raw constant ≠ 1/(2Trel) (d=$d)"
        @assert xi_err / Knorm < tol  "draw_xi covariance ≠ 2·kappa_tensor (d=$d, v=($vx,$vy))"
    end
end

function step2_noise_cov()
    println("\n[Step 2] noise covariance:  ⟨draw_xi²⟩ = 2 T_rel κ   (d = 2 and 3)")
    check_noise_cov(; d=3, cs2v=1.0/3.0, kmodel=Kappa3D())
    check_noise_cov(; d=2, cs2v=1.0/2.0, kmodel=Kappa2D())
    println("  draw_xi covariance = 2·kappa_tensor for d=2 and d=3  ✓")
end

# ═══════════════════════════════════════════════════════════════ Step 3 ══════
# particle_sweep! (per-particle accept/reject, the deployed scheme).
# particle_sweep! zeros the Δm, Δs accumulators at entry, so no caller-side
# reset is needed between sweeps; none of the checks below call apply_sweep!,
# so the shear state is frozen throughout and can be shared.

# frozen shear-wave state: uniform s0, sinusoidal M^y — a velocity gradient so
# the accept/reject has something to bite on.  d=2 / Kappa2D (the 2D fluid).
function shear_state(; cs2v=1.0/2.0, M_amp=0.05, s0=2.0, η_by_s=0.2, ζ_by_s=0.0, d=2)
    params, particles, cl, eos, _, _ = setup_shear_wave(0.12, 2.5;
        L=1.0, M_amp, s0, cs2v, η_by_s, ζ_by_s, d)
    rebuild_cell_list!(cl, particles)
    return (; params, particles, cl, eos)
end

# (i) zero viscosity ⇒ zero kicks
function step3_zero_viscosity(dt)
    st = shear_state(; η_by_s=0.0, ζ_by_s=0.0)
    _, _, dS = particle_sweep!(st.particles, st.params, st.cl, st.eos, dt,
                               make_phys_rng(st.params.N, 1))
    maxΔ = maximum(max(abs(p.Δs), norm(p.Δm)) for p in st.particles)
    @assert maxΔ == 0.0  "zero-viscosity sweep moved something: $maxΔ"
    @assert dS == 0.0
    println("  (i)   η=ζ=0 ⇒ Δm=Δs=0, ΔS=0  ✓")
end

# (ii) momentum conservation, (iii) GENERIC degeneracy,
# (v) ΔS identity:  Σ accepted (Δt V_a ξ:ε_a)  ==  Σ_c Δs_c accumulated
function step3_conservation(st, dt)
    p = st.particles
    nacc, ntry, dS = particle_sweep!(p, st.params, st.cl, st.eos, dt,
                                     make_phys_rng(st.params.N, 7))
    ΣΔm    = sum(q.Δm for q in p)
    mscale = sum(norm(q.Δm) for q in p)
    degen  = sum(q.Teff * q.Δs + dot(q.dotx, q.Δm) for q in p)
    escale = sum(abs(q.Teff * q.Δs) + abs(dot(q.dotx, q.Δm)) for q in p)
    ΣΔs    = sum(q.Δs for q in p)
    @printf("  (ii)  |ΣΔm| / Σ|Δm| = %.2e\n", norm(ΣΔm) / mscale)
    @assert norm(ΣΔm) / mscale < 1e-12
    @printf("  (iii) |Σ(T·Δs + ẋ·Δm)| / scale = %.2e   (acceptance %d/%d = %.2f)\n",
            abs(degen) / escale, nacc, ntry, nacc / ntry)
    @assert abs(degen) / escale < 1e-12
    @printf("  (v)   |ΔS_tot − ΣΔs| / |ΔS_tot| = %.2e   (ΔS_tot=%+.3e)\n",
            abs(dS - ΣΔs) / abs(dS), dS)
    @assert abs(dS - ΣΔs) / abs(dS) < 1e-10  "strain-formula ΔS ≠ −Σ b·Δm over kicks"
    println("  (ii),(iii),(v)  ✓")
end

# (iv) reproducibility + order independence: a fresh seed-7 stream vector
# reproduces (Δm,Δs) bit-for-bit, and so does a shuffled visit order (each
# particle draws from rngs[a] regardless of when it is visited).
function step3_order_independence(st, dt)
    p, params, eos = st.particles, st.params, st.eos
    particle_sweep!(p, params, st.cl, eos, dt, make_phys_rng(params.N, 7))
    Δm_ref = [q.Δm for q in p];  Δs_ref = [q.Δs for q in p]

    particle_sweep!(p, params, st.cl, eos, dt, make_phys_rng(params.N, 7))
    @assert all(p[i].Δm == Δm_ref[i] for i in eachindex(p))
    @assert all(p[i].Δs == Δs_ref[i] for i in eachindex(p))

    perm = shuffle(MersenneTwister(3), collect(1:params.N))
    pp   = p[perm]
    clp  = CellList2D(params.L, params.h, params.N);  rebuild_cell_list!(clp, pp)
    rp   = make_phys_rng(params.N, 7)[perm]           # rngs[a] follows particle a
    particle_sweep!(pp, params, clp, eos, dt, rp)
    invp   = invperm(perm)
    mscale = maximum(norm, Δm_ref);  sscale = maximum(abs, Δs_ref)
    dm_e   = maximum(norm(pp[invp[i]].Δm - Δm_ref[i]) for i in eachindex(p))
    ds_e   = maximum(abs(pp[invp[i]].Δs  - Δs_ref[i]) for i in eachindex(p))
    @printf("  (iv) shuffled order: max|ΔΔm|/scale=%.1e  max|ΔΔs|/scale=%.1e\n",
            dm_e/mscale, ds_e/sscale)
    @assert dm_e/mscale < 1e-12 && ds_e/sscale < 1e-12  "sweep is order-dependent"
    println("  (iv)  reproducible + visit-order independent  ✓")
end

# Antithetic drift estimator: the sampled accept/reject is too noisy for the
# mean-drift check (signal/noise ∝ σ_Y per move).  Weight each move's kicks by
# w(Y) = ½[min(1,e^Y) − min(1,e^{−Y})] — the SAME mean as sampled accept/reject
# (E_u[accept] = min(1,e^Y), and the ±ξ pair average kills the O(1) noise),
# variance down by ~1/σ_Y².  The kick loop mirrors particle_sweep!; its ΔS/kick
# consistency is pinned by check (v).
function drift_sweep!(D, particles, params, cl, eos, dt, rngs)
    cs2v = cs2(eos)
    ε    = compute_strain(particles, params, cl)
    ncx = cl.ncx;  ncy = cl.ncy
    h = params.h;  L = params.L;  kern = params.kernel
    for a in eachindex(particles)
        pa  = particles[a]
        rng = rngs[a]
        Va  = pa.s / pa.S_smooth
        ξa  = draw_xi(rng, pa, eos, cs2v, params.d) ./ sqrt(dt * Va)
        Y   = dt * Va * sum(ξa[i,j] * ε[a][j,i] for i in 1:2, j in 1:2)
        w   = 0.5 * (min(1.0, exp(Y)) - min(1.0, exp(-Y)))
        coeff = dt * pa.s / pa.S_smooth^2
        ix, iy = cell_index(pa.x, cl)
        for dx in -1:1, dy in neighbor_range(ncy)
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                b == a && continue
                rv = minimum_image(pa.x - particles[b].x, L)
                r  = sqrt(rv[1]^2 + rv[2]^2)
                r >= 2h && continue
                grad = wendland_c2_grad(rv, r, h, kern)
                f = (w * coeff * particles[b].s) .* (ξa * grad)
                D[a] += f;  D[b] -= f
            end
        end
    end
end

# (vi) mean drift = two-level dissipative_force!  (the decisive test)
function step3_mean_drift(st, dt; nsweep=200)
    p, params, cl, eos = st.particles, st.params, st.cl, st.eos
    dotm = zeros(SVector{2,Float64}, params.N)
    dissipative_force!(dotm, p, params, cl, eos)
    fnorm2 = sum(dot(f, f) for f in dotm)

    slopes = zeros(nsweep)
    accm   = [zero(SVector{2,Float64}) for _ in 1:params.N]
    D      = zeros(SVector{2,Float64}, params.N)
    rngs   = make_phys_rng(params.N, 40_000)          # advances across sweeps
    for j in 1:nsweep
        fill!(D, zero(SVector{2,Float64}))
        drift_sweep!(D, p, params, cl, eos, dt, rngs)
        slopes[j] = sum(dot(D[i], dotm[i]) for i in 1:params.N) / (dt * fnorm2)
        for i in 1:params.N;  accm[i] += D[i];  end
    end
    m     = sum(slopes) / nsweep
    se    = std(slopes) / sqrt(nsweep)
    resid = sqrt(sum(norm(accm[i]/(nsweep*dt) - dotm[i])^2 for i in 1:params.N) / fnorm2)
    @printf("  (vi)  slope ⟨Δm·f⟩/(Δt|f|²) = %.4f ± %.4f  (want 1)\n", m, se)
    @printf("        per-particle residual ‖⟨Δm⟩/Δt − f_det‖/‖f_det‖ = %.3f (statistical)\n", resid)
    @assert abs(m - 1) < 0.1  "mean drift ≠ dissipative_force! (slope $m)"
    println("  (vi)  mean drift = two-level dissipative_force!  ✓")
end

# (vii) real accept/reject path: ⟨ΔS_tot⟩ > 0 at a dt large enough that
# rejections actually happen (acceptance < 1).  In this frozen-state test dt is
# a pure parameter — it only sets σ(ΔS_a) ∝ √dt; the scheme's identities hold at
# any dt.  dt=200 ⇒ σ(ΔS_a) ≈ 0.09, acceptance ≈ 0.97.
function step3_entropy_production(st; dtb=200.0, nS=150)
    p, params = st.particles, st.params
    xs = zeros(nS)
    ptot = 0;  ttot = 0
    rngv = make_phys_rng(params.N, 90_000)           # advances across sweeps
    for j in 1:nS
        na, nt, s = particle_sweep!(p, params, st.cl, st.eos, dtb, rngv)
        ptot += na;  ttot += nt;  xs[j] = s
    end
    mS  = sum(xs) / nS
    seS = std(xs) / sqrt(nS)
    @printf("  (vii) dt=%.2g: acceptance=%.3f  ⟨ΔS_tot⟩=%+.3e ± %.1e  (%.1fσ > 0)\n",
            dtb, ptot/ttot, mS, seS, mS/seS)
    @assert ptot < ttot   "no rejections at dt=$dtb — test not probing accept/reject"
    @assert mS > 3 * seS  "entropy production not clearly > 0"
    println("  (vii) accept/reject dissipation: ⟨ΔS⟩ > 0  ✓")
end

function step3(; dt=0.002)
    println("\n[Step 3] particle_sweep! unit checks")
    st = shear_state()
    @printf("  N=%d  dt=%.4g  (d=%d, Kappa2D)\n", st.params.N, dt, st.params.d)
    step3_zero_viscosity(dt)
    step3_conservation(st, dt)
    step3_order_independence(st, dt)
    step3_mean_drift(st, dt)
    step3_entropy_production(st)
end

# ═══════════════════════════════════════════════════════════════════════════
function run_all()
    println("=" ^ 70)
    println("teststochastic.jl")
    println("=" ^ 70)
    #step0_accumulators()
    # step1_stencil()
    # step2_noise_cov()
    step3()
    # println("\n" * "=" ^ 70)
    # println("teststochastic.jl — Steps 0–3 passed.")
    # println("=" ^ 70)
end

run_all()
