# legacy/kappa.jl — the κ^{ijlm} noise tensors of the old particle-sweep stochastic step (moved from src/sph.jl on
# branch dimfix, docs/dimension.tex): Kappa3D is the d = 3 formula with 2D motion, Kappa2D the d = 2 formula.

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
