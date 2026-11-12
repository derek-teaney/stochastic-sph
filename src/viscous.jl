# viscous.jl — the deterministic viscous step: the bond dissipation (bond_dissipation.tex),
# the two-level strain form (sph2.tex), and the step in the particle representation with
# its three back maps (changerep.tex).

# ── Dissipative (GENERIC) functions ───────────────────────────────────────────

# Two-level dissipative force (see sph2.tex).
#
# Pass 1 builds the SPH strain tensor at every particle:
#   ε_{ji}(x_a) = (1/S_a) Σ_b s_b (β_{i,b} − β_{i,a}) D_j(x_a, x_b)
# where D_j = ∂W/∂x_a^j and β_a = dotx_a / Teff_a.
#
# Pass 2 assembles the viscous force from the stored strains:
#   f^i(x_a) = Σ_b s_a s_b D_j(x_a,x_b)
#              × [ (T_rel κ^{ijlm}/S²)_a ε_{lm}(x_a)
#                + (T_rel κ^{ijlm}/S²)_b ε_{lm}(x_b) ]
#
# Newton's 3rd law holds because D_j(x_b,x_a) = −D_j(x_a,x_b).

# A[i,j] = Σ_{l,m} K[i,j,l,m] ε[l,m]   (4-tensor contracted with 2×2 matrix)
function kappa_dot_strain(K::SArray{Tuple{2,2,2,2},Float64,4,16},
                           ε::SMatrix{2,2,Float64,4})
    SMatrix{2,2,Float64,4}(
        sum(K[1,1,l,m]*ε[l,m] for l in 1:2, m in 1:2),
        sum(K[2,1,l,m]*ε[l,m] for l in 1:2, m in 1:2),
        sum(K[1,2,l,m]*ε[l,m] for l in 1:2, m in 1:2),
        sum(K[2,2,l,m]*ε[l,m] for l in 1:2, m in 1:2),
    )
end

# ── Pass 1: strain tensor ε[j,i] = (1/S_a) Σ_{b≠a} s_b (β_b−β_a)_i D_j(a,b) ──
# (Shared by dissipative_force! and particle_sweep!.)
# Requires up-to-date dotx, Teff, S_smooth (from smooth_to_particles! +
# update_thermodynamics! + update_velocities!).
# Optional bv: strain of a given b-field (default b = ẋ/Teff from the particles).
function compute_strain(particles::Vector{Particle}, params::SimParams,
                        cl::CellList2D,
                        bv::AbstractVector = [p.dotx / p.Teff for p in particles])
    N   = length(particles)
    ncx = cl.ncx;  ncy = cl.ncy
    h      = params.h
    L      = params.L
    kernel = params.kernel
    ε = Vector{SMatrix{2,2,Float64,4}}(undef, N)
    for a in 1:N
        ε_acc = zero(SMatrix{2,2,Float64})
        β_a   = bv[a]
        S_a   = particles[a].S_smooth
        ix, iy = cell_index(particles[a].x, cl)
        for dx in -1:1, dy in neighbor_range(ncy)
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                b == a && continue           # skip self; include all other neighbors
                r_vec = minimum_image(particles[a].x - particles[b].x, L)
                r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
                r_mag >= 2h && continue
                grad  = wendland_c2_grad(r_vec, r_mag, h, kernel)
                Δβ    = bv[b] - β_a
                # outer product: (grad * Δβ')[j,i] = D_j * Δβ_i  →  ε_{ji}
                ε_acc += particles[b].s * (grad * Δβ')
            end
        end
        ε[a] = ε_acc / S_a
    end
    return ε
end

# Viscous force F = −∂K/∂b (b = ẋ/T), written into dotm.  The discretization of the
# dissipative function K is chosen by params.dissipation:
#   :two_level  two-level SPH strain form (sph2.tex)
#   :bond       one-level bond form with mean-field rotation/dilation removed
#               (bond_dissipation.tex)
#   :particle  the bond form evaluated on the particle force β_a (changerep.tex);
#               the full viscous step is viscous_step_grad!
# Requires up-to-date dotx, Teff, S_smooth (from smooth_to_particles! +
# update_thermodynamics! + update_velocities!).
function dissipative_force!(dotm::Vector{SVector{2,Float64}},
                             particles::Vector{Particle},
                             params::SimParams, cl::CellList2D,
                             eos::AbstractEOS)
    if params.dissipation === :bond
        bond_dissipative_force!(dotm, particles, params, cl, eos)
    elseif params.dissipation === :particle
        β = zeros(SVector{2,Float64}, length(particles))
        bond_dissipative_force!(dotm, particles, params, cl, eos, pointwise_beta!(β, particles, eos))
    else
        two_level_dissipative_force!(dotm, particles, params, cl, eos)
    end
end

function two_level_dissipative_force!(dotm::Vector{SVector{2,Float64}},
                                      particles::Vector{Particle},
                                      params::SimParams, cl::CellList2D,
                                      eos::AbstractEOS)
    N   = length(particles)
    ncx = cl.ncx;  ncy = cl.ncy
    h      = params.h
    L      = params.L
    kernel = params.kernel

    # Pre-compute T_rel κ^{ijlm} (without 1/S²) at every particle.
    km = params.d == 2 ? Kappa2D() : Kappa3D()
    κ  = [kappa_tensor(pa, eos, km) for pa in particles]

    # ── Pass 1 ──
    ε = compute_strain(particles, params, cl)

    # ── Pass 2: viscous force from stored strains (pair loop, b > a) ──────────
    # Pre-compute A[a] = (T_rel κ / S²)[a] · ε[a]  (a 2×2 matrix per particle)
    A = [(kappa_dot_strain(κ[a], ε[a]) / particles[a].S_smooth^2) for a in 1:N]

    fill!(dotm, zero(SVector{2,Float64}))
    for a in 1:N
        sa = particles[a].s
        ix, iy = cell_index(particles[a].x, cl)
        for dx in -1:1, dy in neighbor_range(ncy)
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                b <= a && continue
                r_vec = minimum_image(particles[a].x - particles[b].x, L)
                r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
                r_mag >= 2h && continue
                grad  = wendland_c2_grad(r_vec, r_mag, h, kernel)
                sb    = particles[b].s
                # f^i(a←b) = s_a s_b D_j [(T_rel κ/S²)_a ε_a + (T_rel κ/S²)_b ε_b]^{ij}
                f_ab  = (sa * sb) .* ((A[a] + A[b]) * grad)
                dotm[a] += f_ab
                dotm[b] -= f_ab    # Newton's 3rd law: D_j(b,a) = −D_j(a,b)
            end
        end
    end
end

# ── One-level (bond) dissipation ──────────────────────────────────────────────
#
# docs/bond_dissipation.tex.  Notation follows the note:
#   r_ab = x_b − x_a,  Δb_ab = b_b − b_a,  V_a = s_a/S_a
#   ψ_ab        bond weight ψ(ρ_ab) at the comoving distance ρ_ab² = r·ℓ_a⁻¹·r,
#               moment-corrected so that Σ_b V_b ψ_ab r rᵀ = n_a ℓ_a  exactly
#   w_ab        = V_b ψ_ab / n_a                              (normalized weight)
#   G_a         = (∇b)_a,  G[i,j] = ∂_j b_i                   (least-squares fit)
#   Ω_a, θ_a    = antisym(G_a),  ℓ_a:G_a / ⟨Pℓ⟩_a              (rotation, comoving dilation)
#   D_ab        = Δb_ab − (Ω_a + θ_a P_a) r_ab                (covariant bond difference)
#   K = Σ_a V_a T_a η_a Σ_b w_ab D_ab·ℓ_a·D_ab + ½ Σ_a V_a T_a ζ_eff θ_a²
#   t_ab        = −2 V_a T_a η_a w_ab ℓ_a D_ab                (traction: momentum a gives b)
#   V_a Π_a     = Σ_b t_ab r_abᵀ                              (stress from the virial of the faces)
#   E_a         = ∂K/∂G_a = e_a ℓ_a + antisym(V_a Π_a)        (mean-field stress)
# The force is F_a = −∂K/∂b_a: bond part from the tractions, mean-field part through the fit.
# The simulation is two-dimensional: P and the subtracted dilation use d_s = 2, and
# the shear of the d − 2 unresolved directions of the d-dimensional fluid goes into
#   ζ_eff = ζ + 2η (1/2 − 1/d)                                 (note, Appendix B).
# The antisym(V_a Π_a) term of E_a vanishes when the Λ correction succeeds (the rotation
# is then variational); it keeps the force exact on fallback particles.
# 1D simulation: plain bond term with the full T κ^{xxxx}(v) on the bonds, no mean field.

# Bond weight ψ(ρ) ∝ −W′(ρ; h_ψ)/ρ for the Wendland C2 kernel, support ρ < 2h_ψ, with
# h_ψ = params.bond_h ≤ h.  Its normalization is irrelevant (divided out by n_a).
bond_weight(ρ::Float64, hψ::Float64) =
    (q = ρ / hψ; q < 2.0 ? (1 - q / 2)^3 / hψ^2 : 0.0)

# comoving metric ℓ = δ − v vᵀ, its inverse δ + γ² v vᵀ, and the scalar projector
# P (2×2 block) and ⟨Pℓ⟩ for physical dimension d.
function comoving_structures(v::SVector{2,Float64}, cs2v::Float64, d::Int)
    vv  = v * v'
    γ2  = 1 / (1 - dot(v, v))
    ℓ   = SMatrix{2,2,Float64}(I) - vv
    ℓi  = SMatrix{2,2,Float64}(I) + γ2 * vv
    P   = -cs2v * γ2 * vv + (SMatrix{2,2,Float64}(I) + γ2 * vv) / d
    Pℓ  = 1 - cs2v * dot(v, v)
    return ℓ, ℓi, P, Pℓ
end

# Moment-corrected comoving bond weights for one particle (particle_metrics.tex, Λ_a).
# Returns (ψ_b, n_a) with
#   Σ_b V_b ψ_b r_b r_bᵀ = n_a ℓ   exactly,   ψ_b = ψ(ρ_b) (1 + r_bᵀ Λ r_b),  ℓ:Λ = 0.
# Falls back to Λ = 0 (n_a = ½ tr(ℓ⁻¹ M)) if a corrected weight would be negative.
# Optional buffers ψ0, Vb, ψ (resized here) avoid allocation; the returned weights may
# then alias one of them, valid until the next call with the same buffers.
function bond_weights_comoving(nb, particles, ℓ, ℓi, hψ;
                               ψ0 = Vector{Float64}(undef, length(nb)),
                               Vb = Vector{Float64}(undef, length(nb)),
                               ψ  = Vector{Float64}(undef, length(nb)))
    resize!(ψ0, length(nb));  resize!(Vb, length(nb));  resize!(ψ, length(nb))
    for (k, (b, r, rr)) in enumerate(nb)
        ψ0[k] = bond_weight(sqrt(dot(r, ℓi * r)), hψ)
        Vb[k] = particles[b].s / particles[b].S_smooth
    end
    # unknowns z = (Λxx, Λxy, Λyy, n):  M + N:Λ − n ℓ = 0  (3 eqs),  ℓ:Λ = 0
    # (static 4×4 system: no allocation)
    A3 = zero(SMatrix{3,3,Float64});  r3 = zero(SVector{3,Float64})
    for (k, (b, r, rr)) in enumerate(nb)
        c = Vb[k] * ψ0[k];  c == 0 && continue
        x, y = r[1], r[2]
        q = SVector(x^2, 2x*y, y^2)               # ∂(rᵀΛr)/∂(Λxx, Λxy, Λyy)
        e = SVector(x^2, x*y, y^2)                # components (xx, xy, yy) of r rᵀ
        r3 -= c * e
        A3 += c * (e * q')
    end
    A = SMatrix{4,4,Float64}(A3[1,1], A3[2,1], A3[3,1], ℓ[1,1],
                             A3[1,2], A3[2,2], A3[3,2], 2ℓ[1,2],
                             A3[1,3], A3[2,3], A3[3,3], ℓ[2,2],
                             -ℓ[1,1], -ℓ[1,2], -ℓ[2,2], 0.0)
    rhs = SVector(r3[1], r3[2], r3[3], 0.0)
    # a nearly singular A (a star distorted by a melting lattice) can still make the LU
    # solve throw: take the fallback below (Λ = 0) instead
    z = abs(det(A)) > 0 ? (try A \ rhs catch err; err isa SingularException || rethrow();
                                fill(NaN, SVector{4,Float64}) end) :
                          fill(NaN, SVector{4,Float64})
    Λ = SMatrix{2,2,Float64}(z[1], z[2], z[2], z[3])
    for (k, (b, r, rr)) in enumerate(nb)
        ψ[k] = ψ0[k] * (1 + dot(r, Λ * r))
    end
    if all(>=(0), ψ) && isfinite(z[4]) && z[4] > 0
        return ψ, z[4]
    end
    M = sum((Vb[k] * ψ0[k]) * (r * r') for (k, (b, r, rr)) in enumerate(nb))
    return ψ0, tr(ℓi * M) / 2
end

function bond_weights_1d(nb, particles, ℓi, hψ)
    ψ0 = [bond_weight(sqrt(dot(r, ℓi * r)), hψ) for (b, r, rr) in nb]
    n  = sum((ψ0[k] * particles[b].s / particles[b].S_smooth) * r[1]^2 for (k, (b, r, rr)) in enumerate(nb); init=0.0)
    return ψ0, n
end

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
    hψ     = params.bond_h
    d      = params.d
    cs2v   = cs2(eos)
    oneD   = params.kernel isa WendlandC2_1D

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
        Va   = pa.s / pa.S_smooth
        Ta   = pa.T * pa.γ
        sr   = srel(eos, pa.e)
        η    = eta_by_s(eos) * sr
        ζ    = zeta_by_s(eos) * sr
        va   = pa.dotx                 # coarse-grained particle velocity ẋ_a (update_velocities!)
        # 1D: P with the physical d (it enters only κ^{xxxx}); 2D: P with d_s = 2
        ℓ, ℓi, P, Pℓ = comoving_structures(va, cs2v, oneD ? d : 2)

        # weights ψ_ab (moment-corrected) and n_a
        ψs, n_a = oneD ? bond_weights_1d(nb, particles, ℓi, hψ) :
                         bond_weights_comoving(nb, particles, ℓ, ℓi, hψ)
        n_a > 0 || continue

        Mi = zero(SMatrix{2,2,Float64});  θa = 0.0
        if oneD
            # T κ^{xxxx}(v): the closed form with S = e_x e_x
            θx   = ℓ[1,1] / Pℓ
            Sx   = SMatrix{2,2,Float64}(1.0, 0.0, 0.0, 0.0) - θx * P
            Tκxx = Ta * (2η * (tr(ℓ * Sx * ℓ * Sx) + (d - 2) * θx^2 / d^2) + ζ * θx^2)
            cbond = Tκxx * Va / n_a                     # bond coefficient: c_ab = cbond V_b ψ_ab
            Q    = zero(SMatrix{2,2,Float64})          # no mean-field subtraction in 1D
            ℓm   = SMatrix{2,2,Float64}(1.0, 0.0, 0.0, 0.0)   # 1D: only the x component
        else
            # fit G_a = (Σ_b V_b ψ_ab Δb rᵀ) M_a⁻¹,  M_a = Σ_b V_b ψ_ab r rᵀ  (= n_a ℓ_a)
            M = zero(SMatrix{2,2,Float64});  Gt = zero(SMatrix{2,2,Float64})
            for (k, (b, r, rr)) in enumerate(nb)
                cb  = ψs[k] * particles[b].s / particles[b].S_smooth
                M  += cb * (r * r')
                Gt += cb * ((bv[b] - bv[a]) * r')
            end
            Mi    = inv(M)
            G     = Gt * Mi                             # G[i,j] = ∂_j b_i
            θa    = sum(ℓ .* G) / Pℓ                    # comoving dilation
            Q     = (G - G') / 2 + θa * P               # D_ab = Δb_ab − Q r_ab
            cbond = 2 * Ta * η * Va / n_a               # 2 V_a T_a η_a w_ab = cbond V_b ψ_ab
            ℓm    = ℓ
        end

        # bond forces: traction t_ab = −c_ab ℓ D_ab, a gives t_ab to b
        VΠ = zero(SMatrix{2,2,Float64})                 # V_a Π_a = Σ_b t_ab r_abᵀ
        Ka = 0.0
        for (k, (b, r, rr)) in enumerate(nb)
            ψ = ψs[k];  ψ == 0 && continue
            c = cbond * (particles[b].s / particles[b].S_smooth) * ψ
            D = (bv[b] - bv[a]) - Q * r
            t = -c * (ℓm * D)
            Ka   -= 0.5 * dot(D, t)
            g[b] -= t
            g[a] += t
            VΠ   += t * r'
        end
        if !oneD
            # mean-field force through the fit: E_a = ∂K/∂G_a
            ζeff = ζ + 2η * (1 / 2 - 1 / d)
            Ka  += 0.5 * Va * Ta * ζeff * θa^2
            e_a  = (Va * Ta * ζeff * θa + sum(VΠ .* P)) / Pℓ
            E   = e_a * ℓ + (VΠ - VΠ') / 2
            # q_ab = V_b ψ_ab E_a M_a⁻¹ r_ab:  ∂K/∂b_b += q_ab,  ∂K/∂b_a −= q_ab
            qsum = zero(SVector{2,Float64})
            for (k, (b, r, rr)) in enumerate(nb)
                q = (ψs[k] * particles[b].s / particles[b].S_smooth) * (E * (Mi * r))
                g[b] += q
                qsum += q
            end
            g[a] -= qsum
        end
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

# One dissipative half/full step: smooth → EOS → viscous force → update (m, s).
# Positions are unchanged; entropy grows by Δs_a = −b_a · Δm_a ≥ 0.
# For :particle the shortest modes are damped at a rate of order (κ/w)/h²; compute_dt
# bounds this rate (viscous_rate_bound) so that each viscous half step of
# generic_step!, of length Δt/2, decays every mode monotonically.
function viscous_step!(particles::Vector{Particle}, params::SimParams,
                            cl::CellList2D, dt::Float64, eos::AbstractEOS)
    if params.hamiltonian === :energy
        eta_by_s(eos) == 0.0 && zeta_by_s(eos) == 0.0 && return
        viscous_step_energy!(particles, params, cl, dt, eos)
        return
    end
    if params.dissipation === :particle
        # no viscosity: nothing to do (the conversion maps would only round-trip s)
        eta_by_s(eos) == 0.0 && zeta_by_s(eos) == 0.0 && return
        viscous_step_grad!(particles, params, cl, dt, eos)
        return
    end
    N    = length(particles)
    dotm = zeros(SVector{2,Float64}, N)

    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)
    dissipative_force!(dotm, particles, params, cl, eos)

    for a in eachindex(particles)
        Δm = dt .* dotm[a]
        particles[a].m += Δm
        particles[a].s -= dot(particles[a].dotx / particles[a].Teff, Δm)   # Δs = −b·Δm ≥ 0
    end
end

# hamiltonian = :energy (energystep.tex, eq. viscstep): the viscous step on the particle
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

# ── The step in the particle representation (docs/changerep.tex) ───────────────

# One viscous step in the particle representation, at fixed positions.
# Requires params.dissipation = :bond (the bond force takes an arbitrary b-field).
# Returns (K, ΣΔs): K evaluated with the pointwise β, and the entropy produced.
function viscous_step_first!(particles::Vector{Particle}, params::SimParams,
                                     cl::CellList2D, dt::Float64, eos::AbstractEOS)
    params.dissipation in (:bond, :particle) ||
        error("viscous_step_first! uses the bond force (params.dissipation = :particle or :bond)")
    N    = length(particles)
    dotm = zeros(SVector{2,Float64}, N)
    β    = zeros(SVector{2,Float64}, N)

    smooth_to_particles!(particles, params, cl)     # S_a, hence V_a = s_a/S_a
    update_thermodynamics!(particles, eos)          # warm start for the pointwise solves
    update_velocities!(particles, params, cl)        # ẋ_a, the velocity of the comoving metric
    pointwise_beta!(β, particles, eos)
    K = bond_dissipation(particles, params, cl, eos, β)
    bond_dissipative_force!(dotm, particles, params, cl, eos, β)

    ΣΔs = 0.0
    for a in eachindex(particles)
        Δm = dt .* dotm[a]
        Δs = -dot(β[a], Δm)                             # Δs = −β·Δm, so ΣΔs = 2K Δt ≥ 0
        particles[a].m += Δm
        particles[a].s += Δs
        ΣΔs += Δs
    end
    return K, ΣΔs
end

# One viscous step with the exact back map.  Returns (K, ΣΔs, Σe) with ΣΔs the exact
# change of Σ_a S_a(e_a, m_a) and Σe the (conserved) total particle energy.
function viscous_step_exact!(particles::Vector{Particle}, params::SimParams,
                                           cl::CellList2D, dt::Float64, eos::AbstractEOS)
    params.dissipation in (:bond, :particle) ||
        error("viscous_step_exact! uses the bond force (params.dissipation = :particle or :bond)")
    N    = length(particles)
    dotm = zeros(SVector{2,Float64}, N)
    β    = zeros(SVector{2,Float64}, N)

    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)        # ẋ_a, the velocity of the comoving metric
    V = [pa.s / pa.S_smooth for pa in particles]               # held fixed
    e = [particle_energy(pa, eos) for pa in particles]         # held fixed
    pointwise_beta!(β, particles, eos)
    K = bond_dissipation(particles, params, cl, eos, β)
    bond_dissipative_force!(dotm, particles, params, cl, eos, β)

    ΣΔs = 0.0
    for a in eachindex(particles)
        pa = particles[a]
        pa.m += dt .* dotm[a]
        Snew, _, _ = eos_from_energy(e[a] / V[a], pa.m / V[a], eos)
        snew = V[a] * Snew
        ΣΔs += snew - pa.s
        pa.s = snew
    end
    return K, ΣΔs, sum(e)
end

# One viscous step with the gradient-corrected map.  Returns (K, ΣΔs, Σe′).
function viscous_step_grad!(particles::Vector{Particle}, params::SimParams,
                                          cl::CellList2D, dt::Float64, eos::AbstractEOS)
    params.dissipation in (:bond, :particle) ||
        error("viscous_step_grad! uses the bond force (params.dissipation = :particle or :bond)")
    N    = length(particles)
    dotm = zeros(SVector{2,Float64}, N)
    β    = zeros(SVector{2,Float64}, N)

    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)        # ẋ_a, the velocity of the comoving metric
    V     = [pa.s / pa.S_smooth for pa in particles]
    stars = [kernel_star(a, particles, params, cl) for a in 1:N]
    C     = [inverse_susceptibility(pa, eos) for pa in particles]
    μ     = [pa.m / pa.s for pa in particles]
    gQ    = [star_fit(stars[a], μ) for a in 1:N]
    g     = [gradient_energy(particles[a].s, C[a], gQ[a][1], gQ[a][2]) for a in 1:N]
    e′    = [particle_energy(particles[a], eos) - g[a] for a in 1:N]        # held fixed
    pointwise_beta!(β, particles, eos)
    K = bond_dissipation(particles, params, cl, eos, β)
    bond_dissipative_force!(dotm, particles, params, cl, eos, β)

    for a in 1:N
        particles[a].m += dt .* dotm[a]
    end
    μ′ = [pa.m / pa.s for pa in particles]                                  # s still old
    ΣΔs = 0.0
    for a in 1:N
        pa = particles[a]
        G′, Q = star_fit(stars[a], μ′)
        e_new = e′[a] + gradient_energy(pa.s, C[a], G′, Q)
        Snew, _, _ = eos_from_energy(e_new / V[a], pa.m / V[a], eos)
        snew = V[a] * Snew
        ΣΔs += snew - pa.s
        pa.s = snew
    end
    return K, ΣΔs, sum(e′)
end
