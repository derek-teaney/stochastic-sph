using StaticArrays
using Random
using LinearAlgebra
using Printf

const RNG = MersenneTwister(0)

abstract type AbstractKernel end
struct WendlandC2_2D <: AbstractKernel end   # sim dimension = 2 (default)
struct WendlandC2_1D <: AbstractKernel end   # sim dimension = 1

mutable struct SimParams
    L       :: SVector{2,Float64}  # (Lx, Ly) after snapping to lattice
    a       :: Float64             # lattice constant (= h/ha_ratio); uniform in x and y
    h       :: Float64             # smoothing length
    N       :: Int                 # total particle count
    vol     :: Float64             # volume (area in 2D, length in 1D) per particle
    seed    :: Int
    d       :: Int                 # physical dimension (2 or 3); drives cs²=1/d and kappa
    lattice :: Symbol              # :square or :hex
    kernel  :: AbstractKernel      # WendlandC2_2D() or WendlandC2_1D()
end

# Primary constructor: fixes a = h/ha_ratio, then snaps Lx and Ly to the nearest
# integer multiple of the lattice row spacing (a for square, a√3/2 in y for hex).
# sim_dim=1: single row of particles along x; Ly=a, vol=a, kernel=WendlandC2_1D().
function SimParams(Lx_t::Real, Ly_t::Real, h, ha_ratio; seed=42, d=3, lattice=:square, sim_dim=2)
    a = h / ha_ratio
    if sim_dim == 1
        Nx      = round(Int, Lx_t / a)
        Lx      = Nx * a;  Ly = a
        N       = Nx
        vol     = a
        lattice = :square
        kernel  = WendlandC2_1D()
    elseif lattice == :hex
        dy = a * sqrt(3) / 2
        Nx = round(Int, Lx_t / a)
        Ny = 2 * round(Int, Ly_t / (2*dy))   # force even for periodic offset rows
        Lx = Nx * a;  Ly = Ny * dy
        N  = Nx * Ny
        vol = a^2 * sqrt(3) / 2
        kernel  = WendlandC2_2D()
    else
        Nx = round(Int, Lx_t / a)
        Ny = round(Int, Ly_t / a)
        Lx = Nx * a;  Ly = Ny * a
        N  = Nx * Ny
        vol = a^2
        kernel  = WendlandC2_2D()
    end
    L = SVector(Lx, Ly)
    dim_str = kernel isa WendlandC2_1D ? "1D" : "2D"
    @printf("SimParams: Lx=%-6g  Ly=%-6g  a=%-8g  h=%-8g  N=%-6d  h/a=%.3g  vol=%.4g  lattice=%s  sim=%s\n",
            Lx, Ly, a, h, N, h/a, vol, lattice, dim_str)
    h/a < 0.5 && @warn "h/a = $(h/a) < 0.5: particles barely overlap, shot noise will dominate"
    h/a > 10  && @warn "h/a = $(h/a) > 10: very diffuse smoothing, resolution is poor"
    SimParams(L, a, h, N, vol, seed, d, lattice, kernel)
end

# Square-box convenience wrapper: SimParams(L, h, ha_ratio; kw...) → Lx=Ly=L.
SimParams(L::Real, h, ha_ratio; kw...) = SimParams(L, L, h, ha_ratio; kw...)

# ── Equation of state ─────────────────────────────────────────────────────────

abstract type AbstractEOS end

struct ConformalEOS <: AbstractEOS
    cs2_val :: Float64   # sound speed squared (1/3 for ultrarelativistic)
    T_scale :: Float64   # T_rel = T_scale * e^(cs2/(1+cs2)); sets temperature units
    η_by_s  :: Float64   # specific shear viscosity η/s (constant)
    ζ_by_s  :: Float64   # specific bulk viscosity  ζ/s (0 for conformal EOS)
end

# Gluon-gas Stefan-Boltzmann scale: e = (16π²/30) T⁴  →  T_scale = (30/(16π²))^(1/4)
const T_scale_gluon = (30.0 / (16π^2))^(1/4)

# Natural-units scale (ℓ_Q = T_0 = 1) for the 2D conformal fluid, cs² = 1/2:
# s_rel = (3/2) T_rel²/T_scale³, so s_rel(T_rel=1) = 1 ⟹ T_scale = (3/2)^(1/3).
# (For cs² = 1/3 the analogous value is (4/3)^(1/4).)
const T_scale_2d_lq1 = (3.0 / 2.0)^(1/3)

# Keyword constructor: cs2 = 1/d for a conformal fluid; gluon-gas T_scale by default.
ConformalEOS(; d=3, cs2_val=1.0/d, T_scale=T_scale_gluon, η_by_s=0.0, ζ_by_s=0.0) =
    ConformalEOS(cs2_val, T_scale, η_by_s, ζ_by_s)

pressure(eos::ConformalEOS, e::Float64) = eos.cs2_val * e

# Relativistic temperature
Trel(eos::ConformalEOS, e::Float64) = eos.T_scale * e^(eos.cs2_val / (1.0 + eos.cs2_val))

# Sound speed squared (two-arg form for generic EOS; one-arg shortcut for ConformalEOS)
cs2(eos::ConformalEOS,  e::Float64) = eos.cs2_val
cs2(eos::ConformalEOS)              = eos.cs2_val

# Specific transport coefficients
eta_by_s(eos::ConformalEOS)  = eos.η_by_s
zeta_by_s(eos::ConformalEOS) = eos.ζ_by_s

# Relativistic entropy density: s_rel = (e+p)/T_rel  (thermodynamic identity)
srel(eos::ConformalEOS, e::Float64) = (e + pressure(eos, e)) / Trel(eos, e)

# ─────────────────────────────────────────────────────────────────────────────

mutable struct Particle
    # primitive variables:
    x        :: SVector{2, Float64}   # position
    s        :: Float64                # entropy per particle
    m        :: SVector{2, Float64}   # momentum per particle
    
    # Thermodynamic fields computed by smooth_to_particles! 
    S_smooth :: Float64                # smoothed entropy density at x_a
    M_smooth :: SVector{2, Float64}   # smoothed momentum density at x_a
    
    # Thermodynamic fields computed by solve_eos!
    e        :: Float64                # internal energy density (Newton warm start)
    p        :: Float64                # pressure
    T        :: Float64                # T = T_rel/γ = (e+p)/S  (density-frame temperature)
    γ        :: Float64                # Lorentz factor γ = sqrt(1 + u²)
    v_over_S :: SVector{2, Float64}   # (v/S)(x_a)
    F_over_S :: Float64                # (F/S)(x_a)
    H_over_S :: Float64                # ℰ = energy per entropy; H = Σ s_a ℰ(x_a)

    # Particle-level derivatives of the Hamiltonian (smoothed sums)
    dotx     :: SVector{2, Float64}   # ẋ_a = Σ_b s_b (v/S)_b W  [smoothed velocity]
    Teff     :: Float64                # H_over_S_a − Σ_b s_b (F/S)_b W

    # Stochastic-sweep accumulators (per-particle Metropolis sweep).
    # During a Metropolis sweep s and m are held frozen as coefficients; the
    # accepted increments land here and are folded in by apply_sweep! afterwards.
    Δs       :: Float64                # Σ over accepted bonds of −b_a·δ_ab
    Δm       :: SVector{2, Float64}   # Σ over accepted bonds of ±δ_ab
end

Particle(x::SVector{2,Float64}, s::Float64) =
    Particle(x, s,
             zero(SVector{2,Float64}),   # m
             0.0,                         # S_smooth
             zero(SVector{2,Float64}),   # M_smooth
             0.0, 0.0, 0.0, 1.0,         # e, p, T, γ
             zero(SVector{2,Float64}),   # v_over_S
             0.0, 0.0,                    # F_over_S, H_over_S
             zero(SVector{2,Float64}), 0.0,   # dotx, Teff
             0.0, zero(SVector{2,Float64}))   # Δs, Δm

Particle(x::SVector{2,Float64}, s::Float64, m::SVector{2,Float64}) =
    Particle(x, s, m,
             0.0,                         # S_smooth
             zero(SVector{2,Float64}),   # M_smooth
             0.0, 0.0, 0.0, 1.0,         # e, p, T, γ
             zero(SVector{2,Float64}),   # v_over_S
             0.0, 0.0,                    # F_over_S, H_over_S
             zero(SVector{2,Float64}), 0.0,   # dotx, Teff
             0.0, zero(SVector{2,Float64}))   # Δs, Δm

function wendland_c2(r::Float64, h::Float64, ::WendlandC2_2D)::Float64
    q = r / h
    q >= 2.0 && return 0.0
    return (7.0 / (4π * h^2)) * (1.0 - q/2.0)^4 * (1.0 + 2.0*q)
end

function wendland_c2(r::Float64, h::Float64, ::WendlandC2_1D)::Float64
    q = r / h
    q >= 2.0 && return 0.0
    return (3.0 / (4h)) * (1.0 - q/2.0)^4 * (1.0 + 2.0*q)
end

wendland_c2(r::Float64, h::Float64) = wendland_c2(r, h, WendlandC2_2D())

function wendland_c2_grad(r_vec::SVector{2,Float64},
                          r::Float64,
                          h::Float64, ::WendlandC2_2D)
    q = r / h
    (q >= 2.0 || r < 1e-10) && return SVector(0.0, 0.0)
    C    = 7.0 / (4π * h^2)
    dWdr = -5C * q * (1 - q/2)^3 / h
    return dWdr / r .* r_vec
end

function wendland_c2_grad(r_vec::SVector{2,Float64},
                          r::Float64,
                          h::Float64, ::WendlandC2_1D)
    q = r / h
    (q >= 2.0 || r < 1e-10) && return SVector(0.0, 0.0)
    C    = 3.0 / (4h)
    dWdr = -5C * q * (1 - q/2)^3 / h
    return dWdr / r .* r_vec
end

wendland_c2_grad(rv, r, h) = wendland_c2_grad(rv, r, h, WendlandC2_2D())

function minimum_image(r::SVector{2,Float64}, L::SVector{2,Float64})
    r .- L .* round.(r ./ L)
end

function solve_eos!(pa::Particle, eos::AbstractEOS; tol=1e-12, maxiter=50, verbose=false)
    S_in  = pa.S_smooth
    M_vec = pa.M_smooth
    M_mag = sqrt(M_vec[1]^2 + M_vec[2]^2)

    # warm start from stored e; cold start falls back to e ~ S_in
    e = pa.e > 0.0 ? pa.e : S_in

    for iter in 1:maxiter
        w  = e + pressure(eos, e)
        u  = M_mag / w
        γ  = sqrt(1.0 + u^2)
        T  = Trel(eos, e)
        f  = w * γ / T - S_in
        verbose && @printf("  iter %2d:  e = %+.12e   f = %+.4e\n", iter, e, f)
        abs(f) < tol * S_in && break
        fp = (1.0 - cs2(eos, e) * u^2) / (T * γ)   # §II.4 analytic derivative
        Δe = f / fp
        while e - Δe ≤ 0.0          # keep e positive; bisect toward zero if needed
            Δe *= 0.5
        end
        e -= Δe
    end

    # post-convergence expressions (§II.4)
    w   = e + pressure(eos, e)
    u_v = M_vec / w
    u2  = u_v[1]^2 + u_v[2]^2
    γ   = sqrt(1.0 + u2)
    v   = u_v / γ
    p_v = pressure(eos, e)
    S   = S_in

    pa.e        = e
    pa.p        = p_v
    pa.T        = w / S        # T = T_rel/γ = (e+p)/S
    pa.γ        = γ
    pa.v_over_S = v / S
    pa.F_over_S = dot(v, M_vec / S) / S - p_v / S^2
    pa.H_over_S = (w * γ^2 - p_v) / S
end

function update_thermodynamics!(particles::Vector{Particle}, eos::AbstractEOS)
    for pa in particles
        solve_eos!(pa, eos)
    end
end

# ── κ^{ijlm} coupling tensor ──────────────────────────────────────────────────
# See sph1.tex §"The κ tensor" for the physics.  Dispatch on the model type
# allows future addition of a true d=2 formula.

abstract type AbstractKappaModel end
struct Kappa3D <: AbstractKappaModel end   # d=3 formula, 2D motion (current)
struct Kappa2D <: AbstractKappaModel end   # d=2 formula (future)

# Dimensionless shear geometric tensor (d=3 formula, 2D spatial indices).
# Translated from VischydroNode_inc.hpp (kxx_shear).
# C flat index k = 8i+4j+2l+m  ↔  Julia κ[i+1,j+1,l+1,m+1].
# Only 6 distinct values (A…F) due to full index symmetry.
function kappa_shear_geom_3d(vx::Float64, vy::Float64, v2::Float64, cs2v::Float64)
    vx2 = vx^2;  vy2 = vy^2

    A = (-4*(vx2-1)*(1 - vx2 + 3*cs2v*(cs2v*vx2-1)*vy2 + 3*cs2v^2*vy2^2)) / 3

    B = (2*(2 + 3*cs2v*(cs2v*v2-1))*vx*(vx2-1)*vy) / 3 -
        2*cs2v*(cs2v*v2-1)*vx*vy^3

    C = (-2*(vx2-1)*(3*cs2v*vx2-1) +
          2*(1 + 3*cs2v + 2*(1-6*cs2v)*vx2 + 6*cs2v^2*vx2^2)*vy2 +
          6*cs2v*(2*cs2v*vx2-1)*vy2^2) / 3

    D = -(cs2v^2*vx^6) - (vy2-1)*(cs2v*vy2-1)^2 +
         cs2v*vx^4*(2 + cs2v + cs2v*vy2) +
         (vx2*(-3 + 4*vy2 + 3*cs2v*(-2 + cs2v*vy2*(2+vy2)))) / 3

    E = (2*vx*vy*(-3*cs2v*(cs2v*v2-1)*(1+vx2-vy2) + 2*(vy2-1))) / 3

    F = (-4*(1 + 3*cs2v*(cs2v*v2-1)*vx2 - vy2)*(vy2-1)) / 3

    # Column-major SArray{Tuple{2,2,2,2}}: first index (i) fastest
    # Storage order: [1,1,1,1],[2,1,1,1],[1,2,1,1],[2,2,1,1],
    #                [1,1,2,1],[2,1,2,1],[1,2,2,1],[2,2,2,1],
    #                [1,1,1,2],[2,1,1,2],[1,2,1,2],[2,2,1,2],
    #                [1,1,2,2],[2,1,2,2],[1,2,2,2],[2,2,2,2]
    return SArray{Tuple{2,2,2,2},Float64}(
        A, B, B, C,
        B, D, D, E,
        B, D, D, E,
        C, E, E, F)
end

# Dimensionless shear geometric tensor (d=2 formula, 2D spatial indices).
# Translated from shear_2d.txt (C++ source).  Same index-symmetry pattern as 3D.
function kappa_shear_geom_2d(vx::Float64, vy::Float64, v2::Float64, cs2v::Float64)
    vx2 = vx^2;  vy2 = vy^2

    A = -((-1+vx2)*(vx2*(-1+4*cs2v^2*vy2)+(1-2*cs2v*vy2)^2))

    B = vx*vy*(-1+vx2+2*cs2v*(-1+cs2v*v2)*(-1+vx2-vy2))

    C = -((-1+vy2)*(-1+2*cs2v*vy2)) +
         2*cs2v*(-1+2*cs2v*vy2)*vx^4 +
         vx2*(1+vy2+2*cs2v*(1-4*vy2+2*cs2v*vy^4))

    D = (-1+vx2+cs2v*(-2+cs2v*vx2*(2+vx2)))*vy2 +
         cs2v*(2+cs2v+cs2v*vx2)*vy^4 -
         cs2v^2*vy^6 -
         (-1+vx2)*(cs2v*vx2-1)^2

    E = vx*vy*(-1-2*cs2v*(-1+cs2v*v2)*(1+vx2-vy2)+vy2)

    F = -((1+4*cs2v*(-1+cs2v*v2)*vx2-vy2)*(-1+vy2))

    return SArray{Tuple{2,2,2,2},Float64}(
        A, B, B, C,
        B, D, D, E,
        B, D, D, E,
        C, E, E, F)
end

# Dimensionless bulk geometric tensor (d=3): kxx_bulk^{ijlm} = L^{ij} L^{lm}
# where L^{ij} = δ^{ij} - v^iv^j.  Identical index symmetry pattern to shear.
function kappa_bulk_geom_3d(vx::Float64, vy::Float64)
    Lxx =  1 - vx^2
    Lyy =  1 - vy^2
    Lxy = -vx * vy   # = L^{xy} = L^{yx}

    A = Lxx * Lxx
    B = Lxx * Lxy   # = Lxy * Lxx  (all four xxxy variants equal)
    C = Lxx * Lyy   # = Lyy * Lxx  ([xxyy] = [yyxx])
    D = Lxy * Lxy
    E = Lxy * Lyy   # = Lyy * Lxy  (all four xyyy variants equal)
    F = Lyy * Lyy

    return SArray{Tuple{2,2,2,2},Float64}(
        A, B, B, C,
        B, D, D, E,
        B, D, D, E,
        C, E, E, F)
end

# Returns T_rel κ^{ijlm} as SArray{Tuple{2,2,2,2},Float64}.
# κ[i,j,l,m], i,j,l,m ∈ {1=x, 2=y}.
# Physics: Π^{ij} = −(T_rel κ^{ijlm}) ∂_{(l}β_m), β_m = γv_m/T_rel.
# Normalization matches C++ vhnode_kappa: kxx = kxx_geom × T_rel·(η or ζ)/(1−cs²v²)².
function kappa_tensor(pa::Particle, eos::AbstractEOS, ::Kappa3D)
    vx     = pa.v_over_S[1] * pa.S_smooth
    vy     = pa.v_over_S[2] * pa.S_smooth
    v2     = vx^2 + vy^2
    cs2v   = cs2(eos)
    T_rel  = pa.T * pa.γ          # pa.T = T_rel/γ
    sr     = srel(eos, pa.e)
    η      = eta_by_s(eos)  * sr
    ζ      = zeta_by_s(eos) * sr
    denom  = (1 - cs2v * v2)^2
    Teta   = T_rel * η / denom
    Tzeta  = T_rel * ζ / denom
    return Teta .* kappa_shear_geom_3d(vx, vy, v2, cs2v) .+
           Tzeta .* kappa_bulk_geom_3d(vx, vy)
end

# d=2 formula: shear geometry differs from 3D; bulk geometry is identical (L^{ij}L^{lm}).
function kappa_tensor(pa::Particle, eos::AbstractEOS, ::Kappa2D)
    vx     = pa.v_over_S[1] * pa.S_smooth
    vy     = pa.v_over_S[2] * pa.S_smooth
    v2     = vx^2 + vy^2
    cs2v   = cs2(eos)
    T_rel  = pa.T * pa.γ
    sr     = srel(eos, pa.e)
    η      = eta_by_s(eos)  * sr
    ζ      = zeta_by_s(eos) * sr
    denom  = (1 - cs2v * v2)^2
    Teta   = T_rel * η / denom
    Tzeta  = T_rel * ζ / denom
    # Note: we reuse the 3D bulk geometry since L^{ij}L^{lm} is the same in 2D and 3D (no z-dependence).
    return Teta .* kappa_shear_geom_2d(vx, vy, v2, cs2v) .+
           Tzeta .* kappa_bulk_geom_3d(vx, vy)
end

# smooth_to_grid is defined in celllist.jl (depends on CellList2D)
