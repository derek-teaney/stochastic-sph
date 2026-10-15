using Random
using LinearAlgebra
using StaticArrays
using Printf

include("rng.jl")

# ── Steps 1+2: lab-frame Landau noise tensor ──────────────────────────────────
#
# Returns the Landau frame noise ξ^{μν}_L (SMatrix{3,3}, indices t=1,x=2,y=3)
# with the covariance
#
#   ⟨ξ^{μν}_L ξ^{αβ}_L⟩ = 2η(Δ^{(μα}Δ^{νβ)} − (1/d) Δ^{μν}Δ^{αβ}) + ζ Δ^{μν}Δ^{αβ}
#
# vz=0 assumed; z row/col are identically zero and omitted.
# d=3 (3+1D kinematics) or d=2 (strictly 2D).
#
# Inputs
#   rng       — random number generator
#   vx, vy    — 3-velocity components
#   γ         — Lorentz factor
#   eta, zeta — shear/bulk transport coefficients (eta_by_s*srel, zeta_by_s*srel)
#   d         — spatial dimension (2 or 3, default 3)

function generate_landau_noise(rng::AbstractRNG,
                               vx::Float64, vy::Float64, γ::Float64,
                               eta::Float64, zeta::Float64;
                               d::Int=3)::SMatrix{3,3,Float64}

    # ── Step 1: LRF noise, xy-block only (z components identically zero) ──────
    ξ1 = randn(rng)
    ξ2 = randn(rng)

    # uniform random unit vector on S^{d-1}
    if d == 2
        θ = 2π * rand(rng)
        nx, ny = cos(θ), sin(θ)
    else  # d == 3: uniform on S²
        n_raw = SVector(randn(rng), randn(rng), randn(rng))
        n     = n_raw / norm(n_raw)
        nx, ny = n[1], n[2]
    end

    a = sqrt(d * (d + 2) * eta) * ξ1    # sqrt(d(d+2)η)
    b = sqrt(zeta) * ξ2

    # 2×2 xy-block of xi_lrf
    xi_xy = SMatrix{2,2,Float64}(
        a*(nx*nx - 1/d)+b,  a*nx*ny,
        a*nx*ny,            a*(ny*ny - 1/d)+b)

    # ── Step 2: Lorentz boost to lab frame ────────────────────────────────────
    # Λ_s is 3×2 (rows: t,x,y; cols: x,y).  z-col omitted (vz=0 ⟹ Λ^μ_z=0 for μ∈{t,x,y}).
    v2  = vx^2 + vy^2
    fac = v2 > 1e-14 ? (γ - 1.0) / v2 : 0.5   # (γ−1)/v², stable as v→0

    Λx  = SVector(γ*vx,  1.0 + fac*vx*vx,  fac*vx*vy)
    Λy  = SVector(γ*vy,       fac*vx*vy,   1.0 + fac*vy*vy)
    Λ_s = hcat(Λx, Λy)           # SMatrix{3,2}

    return Λ_s * xi_xy * Λ_s'    # SMatrix{3,3}  ξ^{μν}_L, μν ∈ {t,x,y}
end

# ── Step 3: apply projection κ^{ij}_{μν} to get density-frame noise ξ^{ij} ───
#
# Projection tensors:
#   e^i_μ  = δ^i_μ − v^i δ^0_μ                        (2×3, i∈{x,y}, μ∈{t,x,y})
#   h^{ij} = δ^{ij} − v^iv^j                           (2×2 spatial metric)
#   u_μ    = (−γ, γvx, γvy)                            (covariant, metric −+++)
#   Δ^0_μ  = δ^0_μ + u^0 u_μ = (1−γ², γ²vx, γ²vy)
#   κ^{ij}_{μν} = e^i_{(μ}e^j_{ν)} + h^{ij}·fac·(Δ^0_μΔ^0_ν/γ² − u_μu_ν)
#              where fac = cs²/(1 − cs²v²)
#
# Since ξ^{μν}_L is symmetric the full contraction reduces to:
#   ξ^{ij} = (e·ξ_L·eᵀ)^{ij} + h^{ij}·fac·(Δ0ᵀ·ξ_L·Δ0/γ² − uᵀ·ξ_L·u)

function apply_kappa_projector(xi_L::SMatrix{3,3,Float64},
                               vx::Float64, vy::Float64, γ::Float64,
                               cs2v::Float64)::SMatrix{2,2,Float64}
    v2  = vx^2 + vy^2

    # e^i_μ  (2×3, column-major: t-col, x-col, y-col)
    e   = SMatrix{2,3,Float64}(-vx, -vy,   1.0, 0.0,   0.0, 1.0)

    # h^{ij} = δ^{ij} − v^iv^j  (2×2)
    h   = SMatrix{2,2,Float64}(1.0-vx^2, -vx*vy, -vx*vy, 1.0-vy^2)

    # covariant four-velocity and Δ^0_μ projector component (3-vectors: t,x,y)
    u   = SVector(-γ,       γ*vx,      γ*vy)
    Δ0  = SVector(1.0-γ^2, γ^2*vx,    γ^2*vy)   # = (−γ²v², γ²vx, γ²vy)

    fac = cs2v / (1.0 - cs2v * v2)

    # spatial part: Σ_{μν} e^i_μ e^j_ν ξ^{μν}_L  =  e · ξ_L · eᵀ
    spatial = e * xi_L * e'

    # scalar bulk corrections
    bulk_scalar = fac * (dot(Δ0, xi_L * Δ0) / γ^2 - dot(u, xi_L * u))

    return spatial + h * bulk_scalar
end

# ── Combined wrapper: steps 1–3 ───────────────────────────────────────────────
#
# Returns the unit-normalized density-frame noise ξ̂:
# ⟨ξ̂^{ij} ξ̂^{lm}⟩ = κ^{ijlm}, the kappa tensor built from 2η(…)+ζ(…).
# Note: no T_rel here — the physical FDT noise is ξ = √(2 T_rel) ξ̂ (draw_xi).

function generate_noise(rng::AbstractRNG,
                        vx::Float64, vy::Float64, γ::Float64,
                        eta::Float64, zeta::Float64, cs2v::Float64;
                        d::Int=3)::SMatrix{2,2,Float64}
    xi_L = generate_landau_noise(rng, vx, vy, γ, eta, zeta; d)
    return apply_kappa_projector(xi_L, vx, vy, γ, cs2v)
end

# ─────────────────────────────────────────────────────────────────────────────
# Stochastic SPH sweep (per-particle Metropolis scheme).
# Requires the Particle type from sph.jl and the kernel/cell-list machinery
# from celllist.jl — include this file after those two.
# ─────────────────────────────────────────────────────────────────────────────

# Fold the per-sweep accumulators into the primitive variables and reset them.
# Call once, after a full Metropolis sweep, before refreshing the smoothed
# fields for the next step.
function apply_sweep!(particles::Vector{Particle})
    for pa in particles
        pa.s += pa.Δs
        pa.m += pa.Δm
        pa.Δs = 0.0
        pa.Δm = zero(SVector{2,Float64})
    end
end

# ── Per-particle physical noise stress ξ_a ───────────────────────────────────
# The FDT noise:
#
#     ξ_a = √(2 T_rel,a) · generate_noise(…) ,
#     ⟨ξ_a^{ij} ξ_a^{lm}⟩ = 2 T_rel,a κ^{ijlm}(v_a) = 2 · kappa_tensor(pa, eos)
#
# (generate_noise is unit-normalized, no T_rel; verified by the MC audit in
# teststochastic.jl Step 2.)  This is the continuum covariance (δ-functions
# stripped); a discrete step needs a further 1/√(Δt V_a) — the caller supplies
# that.  Requires pa's smoothed/EOS fields current
# (smooth_to_particles! → update_thermodynamics!).
function draw_xi(rng::AbstractRNG, pa::Particle, eos::AbstractEOS, cs2v::Float64,
                d::Int=3)
    vx   = pa.v_over_S[1] * pa.S_smooth
    vy   = pa.v_over_S[2] * pa.S_smooth
    Trel = pa.T * pa.γ
    sr   = srel(eos, pa.e)
    η    = eta_by_s(eos)  * sr
    ζ    = zeta_by_s(eos) * sr
    return sqrt(2 * Trel) .* generate_noise(rng, vx, vy, pa.γ, η, ζ, cs2v; d)
end

# ── One per-particle Metropolis sweep ────────────────────────────────────────
# One accept/reject per particle a: draw a single stress ξ_a with the
# discretized-FDT covariance
#
#     ⟨ξ_a^{ij} ξ_a^{lm}⟩ = 2 T_rel,a κ^{ijlm}|_a / (Δt V_a),   V_a = s_a/S_a,
#
# i.e. ξ_a = draw_xi / √(Δt V_a).  The proposal kicks all of a's bonds with the
# particle-a stress piece of the two-level force,
#
#     Δm_a += Δt (s_a s_b/S_a²) ξ_a·D(a,b),   Δm_b -= same,
#
# and its total entropy change collapses onto the strain:
#
#     ΔS_a = −Σ_c b_c·Δm_c = Δt V_a ξ_a^{ij} ε_{ji}(x_a).
#
# Accept with min(1, exp(ΔS_a)); the accept test needs only ξ_a and ε_a, so the
# bond loop runs for accepted particles only.  Mean drift over the sweep =
# dissipative_force! exactly (no free amplitude).
#
# Preconditions: cell list current; smoothed/EOS/velocity fields current and
# treated as frozen coefficients.  The Δs, Δm accumulators are zeroed here at
# entry, then filled with the accepted increments — call apply_sweep! afterwards
# to fold them into (s, m).  All randomness for particle a (noise ξ_a, accept
# uniform) is drawn from rngs[a] — one independent Xoshiro per particle (rng.jl),
# so the result is independent of visit order and thread chunking.  Size and
# seed `rngs` once per run with seed_phys!(N, seed); the streams advance across
# sweeps.  Returns (n_acc, n_try, ΔS_accepted).
function particle_sweep!(particles::Vector{Particle}, params::SimParams,
                         cl::CellList2D, eos::AbstractEOS,
                         dt::Float64, rngs::Vector{Xoshiro} = PHYS_RNG)
    @assert length(rngs) == length(particles) "rng vector length $(length(rngs)) ≠ particle count $(length(particles)) — call seed_phys!(N, seed)"
    ncx  = cl.ncx;  ncy = cl.ncy
    h    = params.h
    L    = params.L
    kern = params.kernel
    cs2v = cs2(eos)

    # frozen coefficients: strain (dissipative_force! Pass 1) and b = ẋ/T
    ε     = compute_strain(particles, params, cl)
    b_vec = [pa.dotx / pa.Teff for pa in particles]

    # start clean — a particle is written as a neighbor of others' moves, so
    # this must be a full pass before any kick lands
    for pa in particles
        pa.Δm = zero(SVector{2,Float64})
        pa.Δs = 0.0
    end

    n_acc  = 0
    n_try  = 0
    ΔS_tot = 0.0
    for a in eachindex(particles)
        pa  = particles[a]
        rng = rngs[a]
        Va  = pa.s / pa.S_smooth
        ξa  = draw_xi(rng, pa, eos, cs2v, params.d) ./ sqrt(dt * Va)  # discretized-FDT covariance
        u   = rand(rng)
        n_try += 1

        # ΔS_a = Δt V_a ξ^{ij} ε_{ji};  ε stored as ε[j,i] (gradient index first)
        ΔS = dt * Va * sum(ξa[i,j] * ε[a][j,i] for i in 1:2, j in 1:2)

        u < exp(ΔS) || continue                              # reject: nothing lands

        # accepted: distribute the kicks over a's bonds
        coeff = dt * pa.s / pa.S_smooth^2
        ix, iy = cell_index(pa.x, cl)
        for dx in -1:1, dy in neighbor_range(ncy)
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                b == a && continue
                pb = particles[b]
                rv = minimum_image(pa.x - pb.x, L)
                r  = sqrt(rv[1]^2 + rv[2]^2)
                r >= 2h && continue
                grad = wendland_c2_grad(rv, r, h, kern)      # D_j(x_a, x_b)
                f    = (coeff * pb.s) .* (ξa * grad)         # SVector{2}
                pa.Δm += f;   pa.Δs -= dot(b_vec[a], f)
                pb.Δm -= f;   pb.Δs += dot(b_vec[b], f)
            end
        end
        n_acc  += 1
        ΔS_tot += ΔS
    end
    return n_acc, n_try, ΔS_tot
end


