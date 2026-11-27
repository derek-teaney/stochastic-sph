# viscous.jl — the deterministic viscous step on the particle variables of the energy Hamiltonian: the inverse-moment bond
# dissipation (docs/inverse_moment.tex; docs/commonkernel.tex for the resolved part, docs/bond_dissipation.tex for the notation).
#   r_ab = x_b − x_a,  Δb_ab = b_b − b_a,  b_a = ẋ_a/T_a
#   ψ_ab        the bond weight of the SPH kernel at the lab distance, −W′(r_ab)/r_ab;  w_ab = V_b ψ_ab / n_a
#   G_a         = (∇b)_a, the least-squares fit with the weights ψ_ab;  θ_a = ℓ_a:G_a / ⟨Pℓ⟩_a;  Y_a = sym G_a − θ_a P_a
#   K_a = V_a T_a [η tr(ℓYℓY) + ½ ζ_eff θ²] + ½ Σ_b c_ab δ_ab·ℓ δ_ab,  δ_ab = Δb_ab − X_a r_ab (the residual of the fit X_a
#               with the weights c_ab = 2 V_a T_a η w_ab (r̂·ℓ r̂)²)
# The force is F_a = −∂K/∂b_a: the residual tractions and, through the fit G_a, the resolved stress E_a = ∂K_a/∂G_a.
# The simulation is two-dimensional: P and the subtracted dilation use d_s = 2, and the shear of the d − 2 unresolved
# directions of the d-dimensional fluid goes into ζ_eff = ζ + 2η (1/2 − 1/d).
# Removed on 2026-11-27 (docs/focus_plan.tex, Q3 and Q5; the tag pre-prune has them): the change of representation of the
# smoothed Hamiltonian (changerep.tex), the dissipations :particle, :common, :bond (the comoving weights with Λ_a, bond_h) and
# the 1D bonds.

# comoving metric ℓ = δ − v vᵀ, its inverse δ + γ² v vᵀ, and the scalar projector
# P (2×2 block) and ⟨Pℓ⟩ for physical dimension d.
function comoving_structures(v::SVector{2,Float64}, cs2v::Float64, d::Int)
    v   = cap_velocity(v)                               # |v|² ≤ 1 − 1/γ_max² (sph.jl)
    vv  = v * v'
    γ2  = 1 / (1 - dot(v, v))
    ℓ   = SMatrix{2,2,Float64}(I) - vv
    ℓi  = SMatrix{2,2,Float64}(I) + γ2 * vv
    P   = -cs2v * γ2 * vv + (SMatrix{2,2,Float64}(I) + γ2 * vv) / d
    Pℓ  = 1 - cs2v * dot(v, v)
    return ℓ, ℓi, P, Pℓ
end

# The bond weight of the SPH kernel itself (commonkernel.tex): ψ(r) = −W′(r; h)/r for params.kernel, so that
# ψ_ab r_ab = D_ab, the kernel gradient of the ideal step.  Its normalization is divided out by n_a.
function common_bond_weight(params::SimParams)
    h = params.h;  K = params.kernel
    return r -> r > 0 ? -kernel_gradW(SVector(r, 0.0), r, h, K)[1] / r : 0.0
end
# The lab weights without the correction Λ_a (inverse_moment.tex): ψ_ab = ψfun(r_ab), and n_a with
# tr Σ_b V_b ψ_ab r_ab r_abᵀ = 2 n_a, so that the normalized moment 𝓜_a = Σ_b w_ab r rᵀ has trace 2 (𝟙 on a perfect lattice).
function bond_weights_lab(nb, particles, ψfun;
                          ψ0 = Vector{Float64}(undef, length(nb)), Vb = Vector{Float64}(undef, length(nb)))
    resize!(ψ0, length(nb));  resize!(Vb, length(nb))
    tr2 = 0.0
    for (k, (b, r, rr)) in enumerate(nb)
        ψ0[k] = ψfun(rr);  Vb[k] = particles[b].V
        tr2 += Vb[k] * ψ0[k] * rr^2
    end
    return ψ0, tr2 / 2
end
# the factor of the residual of bond r in the metric ℓ, (r̂·ℓ r̂)² (commonkernel.tex, eq. fab)
resid_factor(params::SimParams, ℓ, r) = (dot(r, ℓ * r) / dot(r, r))^2

# Optional bv: force for a given b-field (default b = ẋ/Teff); used by the tests.
# bond_dissipation(...) returns K itself (same arguments).  Kvec, if given, receives
# each particle's share of K (its bonds plus its bulk term).
function bond_dissipative_force!(dotm::Vector{SVector{2,Float64}},
                                 particles::Vector{Particle},
                                 params::SimParams, cl::CellList2D,
                                 eos::AbstractEOS,
                                 bv::AbstractVector = [pa.dotx / pa.Teff for pa in particles])
    return _bond_dissipation!(dotm, particles, params, cl, eos, bv, nothing)
end

bond_dissipation(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                 eos::AbstractEOS, bv::AbstractVector = [pa.dotx / pa.Teff for pa in particles];
                 Kvec = nothing) =
    _bond_dissipation!(nothing, particles, params, cl, eos, bv, Kvec)

# Shared kernel: returns K; if dotm !== nothing also writes F = −∂K/∂b into it.
# One loop over particles a; each builds its star, weights and mean field, and adds its
# contributions to g = ∂K/∂b over the star (bond_dissipation.tex, Sect. "Implementation").
function _bond_dissipation!(dotm, particles::Vector{Particle},
                            params::SimParams, cl::CellList2D,
                            eos::AbstractEOS, bv::AbstractVector, Kvec)
    N      = length(particles)
    ncx    = cl.ncx;  ncy = cl.ncy
    h      = params.h
    L      = params.L
    d      = params.phys_dim
    cs2v   = cs2(eos)

    g  = zeros(SVector{2,Float64}, N)                     # g_a = ∂K/∂b_a
    K  = 0.0
    Kvec === nothing || fill!(Kvec, 0.0)

    nb = Tuple{Int,SVector{2,Float64},Float64}[]          # star: (b, r_ab, |r_ab|), |r_ab| < 2h
    for a in 1:N
        pa = particles[a]
        empty!(nb)
        ix, iy = cell_index(pa.x, cl)
        for dx in -1:1, dy in neighbor_range(ncy)
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                b == a && continue
                r_vec = minimum_image(particles[b].x - pa.x, L)
                r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
                r_mag < 2h && push!(nb, (b, r_vec, r_mag))
            end
        end

        # local data
        Va   = pa.V
        Ta   = target_trel(pa, eos)
        sr   = srel(eos, pa.e)
        η    = eta_by_s(eos) * sr
        ζ    = zeta_by_s(eos) * sr
        va   = pa.dotx                 # coarse-grained particle velocity ẋ_a (update_velocities!)
        ℓ, ℓi, P, Pℓ = comoving_structures(va, cs2v, 2)          # P with d_s = 2

        # weights ψ_ab (moment-corrected) and n_a
        ψs, n_a = bond_weights_lab(nb, particles, common_bond_weight(params))
        n_a > 0 || continue

        # fit G_a = (Σ_b V_b ψ_ab Δb rᵀ) M_a⁻¹,  M_a = Σ_b V_b ψ_ab r rᵀ  (= n_a ℓ_a)
        M = zero(SMatrix{2,2,Float64});  Gt = zero(SMatrix{2,2,Float64})
        for (k, (b, r, rr)) in enumerate(nb)
            cb  = ψs[k] * particles[b].V
            M  += cb * (r * r')
            Gt += cb * ((bv[b] - bv[a]) * r')
        end
        Mi    = inv(M)
        G     = Gt * Mi                             # G[i,j] = ∂_j b_i
        θa    = sum(ℓ .* G) / Pℓ                    # comoving dilation
        cbond = 2 * Ta * η * Va / n_a               # 2 V_a T_a η_a w_ab = cbond V_b ψ_ab

        # commonkernel.tex, eq. Knew: K_a = V_a T_a [η tr(ℓYℓY) + ½ ζ_eff θ² + η Σ_b w_ab δ_ab·ℓ δ_ab],
        # Y = sym G − θ P.  The residual (inverse_moment.tex, eq. Kresid): tractions t_ab = −c_ab ℓ δ_ab, f_ab = (r̂·ℓ r̂)² in c_ab,
        # δ_ab = Δb_ab − X_a r_ab with the least-squares fit X_a = Σ_b c_ab Δb rᵀ R_a⁻¹, R_a = Σ_b c_ab r rᵀ, whose derivative
        # through X_a vanishes; through the fit G_a, E_a = ∂K_a/∂G_a is the resolved part.
        ζeff = ζ + 2η * (1 / 2 - 1 / d)
        Y    = (G + G') / 2 - θa * P
        Ka   = Va * Ta * (η * tr(ℓ * Y * ℓ * Y) + 0.5 * ζeff * θa^2)
        Eres = zero(SMatrix{2,2,Float64})           # (zero; kept in E so that the arithmetic is that of the regression's reference)
        R = zero(SMatrix{2,2,Float64});  Xt = zero(SMatrix{2,2,Float64})
        for (k, (b, r, rr)) in enumerate(nb)
            ψ = ψs[k];  ψ == 0 && continue
            c = cbond * (particles[b].V) * ψ * resid_factor(params, ℓ, r)
            R += c * (r * r');  Xt += c * ((bv[b] - bv[a]) * r')
        end
        X = Xt * inv(R)
        for (k, (b, r, rr)) in enumerate(nb)
            ψ = ψs[k];  ψ == 0 && continue
            c = cbond * (particles[b].V) * ψ * resid_factor(params, ℓ, r)
            δ = (bv[b] - bv[a]) - X * r
            t = -c * (ℓ * δ)
            Ka   -= 0.5 * dot(δ, t)
            g[b] -= t
            g[a] += t
        end
        LYL  = ℓ * Y * ℓ
        E    = 2 * Va * Ta * η * (LYL - (sum(P .* LYL) / Pℓ) * ℓ) + (Va * Ta * ζeff * θa / Pℓ) * ℓ + Eres
        # q_ab = V_b ψ_ab E_a M_a⁻¹ r_ab:  ∂K/∂b_b += q_ab,  ∂K/∂b_a −= q_ab
        qsum = zero(SVector{2,Float64})
        for (k, (b, r, rr)) in enumerate(nb)
            q = (ψs[k] * particles[b].V) * (E * (Mi * r))
            g[b] += q
            qsum += q
        end
        g[a] -= qsum
        K += Ka
        Kvec === nothing || (Kvec[a] = Ka)
    end

    if dotm !== nothing
        for a in 1:N
            dotm[a] = -g[a]
        end
    end
    return K
end

# One dissipative half/full step at fixed positions (viscous_step_energy!).  compute_dt bounds the damping rate of the
# shortest modes (viscous_rate_bound) so that each viscous half step of generic_step!, of length Δt/2, decays every mode
# monotonically.
function viscous_step!(particles::Vector{Particle}, params::SimParams,
                            cl::CellList2D, dt::Float64, eos::AbstractEOS)
    eta_by_s(eos) == 0.0 && zeta_by_s(eos) == 0.0 && return
    viscous_step_energy!(particles, params, cl, dt, eos)
    return
end

# The n-scheme (energystep.tex, eq. viscstep): the viscous step on the particle
# variables, with no change of representation.  β_a = v_a/Teff_a, Teff_a = ∂H/∂s_a;
#   Δm_a = −Δt ∂K/∂β_a  (bond form, changerep.tex),   Δs_a = −β_a·Δm_a ,
# so ΔH = 0 to first order in Δt and ΣΔs_a = 2K Δt ≥ 0.  Returns (K, ΣΔs).
function viscous_step_energy!(particles::Vector{Particle}, params::SimParams,
                                  cl::CellList2D, dt::Float64, eos::AbstractEOS)
    N    = length(particles)
    dotm = zeros(SVector{2,Float64}, N)
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)
    β = [pa.dotx / pa.Teff for pa in particles]
    K = bond_dissipation(particles, params, cl, eos, β)
    bond_dissipative_force!(dotm, particles, params, cl, eos, β)
    ΣΔs = 0.0
    for a in 1:N
        Δm = dt .* dotm[a]
        Δs = -dot(β[a], Δm)
        particles[a].m += Δm
        particles[a].s += Δs
        ΣΔs += Δs
    end
    return K, ΣΔs
end

# Full GENERIC step: Strang splitting D(dt/2) · H(dt) · D(dt/2).
function generic_step!(particles::Vector{Particle}, params::SimParams,
                        cl::CellList2D, dt::Float64, eos::AbstractEOS,
                        dotm::Vector{SVector{2,Float64}})
    viscous_step!(particles, params, cl, dt/2, eos)
    leapfrog_step!(particles, params, cl, dt, eos, dotm)
    viscous_step!(particles, params, cl, dt/2, eos)
end
