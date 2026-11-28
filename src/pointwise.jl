# pointwise.jl — the particle variables: the pointwise state of a particle and the maps
# between particle energies and entropies.  Used by the viscous step (viscous.jl), the
# stochastic step and the equilibrium start.
#
# The thermodynamic force of a particle is its own (pointwise) one,
#
#   β_a = v_a / T_a ,   (v_a, T_a) from the EOS at entropy density S_a and momentum
#                       density m_a / V_a ,   V_a = s_a / S_a ,
#
# where T_a is the code's density-frame temperature (e+p)/S = ∂h/∂S.
#
# β_a is the derivative of the particle energy E_a = V_a h(s_a/V_a, m_a/V_a) at fixed V_a:
#   ∂E_a/∂m_a = v_a ,  ∂E_a/∂s_a = T_a ,  so  β_a = (∂E_a/∂m_a)/(∂E_a/∂s_a).
# A step Δm_a = Δt F_a, Δs_a = −β_a·Δm_a then keeps each E_a fixed to first order.

# Pointwise state of particle a: solve the EOS at (S_a, m_a/V_a).
# Returns (β_a, e_a, T_a).  Uses a scratch copy so the smoothed fields are untouched.
function pointwise_state(pa::Particle, eos::AbstractEOS)
    q = deepcopy(pa)
    q.M_smooth = pa.m * (pa.S_smooth / pa.s)          # m_a / V_a
    solve_eos!(q, eos)
    v = q.v_over_S * q.S_smooth
    return v / q.T, q.e, q.T
end

function pointwise_beta!(β::AbstractVector, particles::Vector{Particle}, eos::AbstractEOS)
    for (a, pa) in enumerate(particles)
        β[a] = pointwise_state(pa, eos)[1]
    end
    return β
end

# ── Particle energies and the back map ─────────────────────────────────────────
#
# Particle a has volume V_a = s_a/S_a (held fixed), energy e_a = V_a h(S_a, m_a/V_a)
# (held fixed), and entropy S_a(e_a, m_a) = V_a σ(e_a/V_a, m_a/V_a).
#
# The viscous step changes m_a; the back map sets s_a' = S_a(e_a, m_a').

# State from the energy density ε = wγ² − p and the density-frame momentum density
# M = wγ²v = (ε + p) v, as in solve_eos!.  The rest-frame energy density is
# e = ε − M·v = ε − |M|²/(ε + p(e)) (a fixed point; a contraction for |v| < 1).
# Returns (S, v, T) with S = w γ / T_rel the entropy density and T = (e+p)/S.
function eos_from_energy(ε::Float64, M::SVector{2,Float64}, eos::AbstractEOS;
                         tol=1e-15, maxiter=500)
    M2 = dot(M, M)
    e  = rest_energy(ε, M2, eos; tol, maxiter)
    w = e + pressure(eos, e)
    v = M / (ε + pressure(eos, e))
    γ = 1 / sqrt(1 - dot(v, v))
    S = w * γ / Trel(eos, e)
    return S, v, w / S
end

# Rest-frame energy density from ε and |M|²: fixed point of e = ε − |M|²/(ε + p(e)).
function rest_energy(ε::Float64, M2::Float64, eos::AbstractEOS; tol=1e-15, maxiter=500)
    e = ε
    for _ in 1:maxiter
        e_new = ε - M2 / (ε + pressure(eos, e))
        done  = abs(e_new - e) <= tol * ε
        e = e_new
        done && break
    end
    return e
end
# Conformal, p = c² e: the fixed point solves c² e² + (1 − c²) ε e + |M|² − ε² = 0.
# The positive root, written without cancellation (agrees with the iteration to roundoff).
function rest_energy(ε::Float64, M2::Float64, eos::ConstantCs2EOS; tol=1e-15, maxiter=500)
    c2 = eos.cs2_val
    return 2 * (ε^2 - M2) / ((1 - c2) * ε + sqrt((1 + c2)^2 * ε^2 - 4c2 * M2))
end

# ── The two particle-level maps (docs/legendre_plan.tex, A) ──────────────────
# Every particle-level use of the equation of state goes through these two.  The
# field-level uses, which are not about one particle, do not: state_from_beta and
# particle_moments (measure.jl; the targets and the proposals).
#
#   forward  particle_forward(s, S, M, e0, eos; M_mag, γ0, guess, tol, maxiter, verbose):
#            the state of a particle of entropy s at the entropy density S = s/V and
#            momentum density M = m/V (warm start e0, the rest-frame energy density, and γ0);
#            returns (e = rest-frame energy density, p = the particle's pressure,
#            T = ∂H/∂s, γ, v, HS = H/s, FS)
#   inverse  particle_inverse(e, m, V, eos; guess): (s, v, T) of the particle of energy e,
#            momentum m and volume V; s = −Inf for a state no fluid can have (the inversion
#            fails, |M| ≥ ε, or γ > γ_max; sph.jl, SPH_GAMMA_MAX)
#
# The target's (tree level): the arithmetic of solve_eos! and eos_from_energy.
# SPH_RENORM=split: renorm_split.jl; sscheme: sscheme.jl.
@inline function target_forward(s, S_in, M_vec, e0, eos::AbstractEOS; M_mag = sqrt(M_vec[1]^2 + M_vec[2]^2),
                                tol = 1e-12, maxiter = 50, verbose = false)
    e = eos_newton(S_in, M_mag, e0, eos; tol, maxiter, verbose)
    w = e + pressure(eos, e)
    G, _ = eos_wgamma(w, M_mag)
    γ = G / w
    v = M_vec * (w / G^2)            # M = w γ² v
    p = pressure(eos, e)
    return (e = e, p = p, T = w / S_in, γ = γ, v = v, HS = (w * γ^2 - p) / S_in, FS = dot(v, M_vec / S_in) / S_in - p / S_in^2)
end
@inline function target_inverse(e, m, V, eos::AbstractEOS)
    ε = e / V;  M = m / V
    (isfinite(ε) && ε > 0 && dot(M, M) < ε^2) || return (-Inf, SVector(NaN, NaN), NaN)     # |M| < ε is necessary (p ≤ e)
    r = try
        eos_from_energy(ε, M, eos)
    catch
        return (-Inf, SVector(NaN, NaN), NaN)
    end
    S, v, T = r
    return ((isfinite(S) && dot(v, v) <= V2MAX) ? V * S : -Inf, v, T)
end
# Conformal EOS: the closed-form inversion cannot throw once |M| < ε (all square roots have
# positive arguments), so the try block, about 17% of a call, is not needed.
# Same arithmetic: bit-identical results.
@inline function target_inverse(e, m, V, eos::ConstantCs2EOS)
    ε = e / V;  M = m / V
    (isfinite(ε) && ε > 0 && dot(M, M) < ε^2) || return (-Inf, SVector(NaN, NaN), NaN)
    S, v, T = eos_from_energy(ε, M, eos)
    return ((isfinite(S) && dot(v, v) <= V2MAX) ? V * S : -Inf, v, T)
end
# The methods are chosen at load time, so that at the tree level the code is exactly
# target_forward and target_inverse.
#
# The S-scheme has no forward map in (s, S, M): solve_eos! and particle_energy call
# sscheme_from_label (sscheme.jl).
if SSCHEME
    particle_forward(s, S_in, M_vec, e0, eos; kw...) = error("SPH_RENORM=sscheme: particle_forward is not used (sscheme_from_label)")
    particle_inverse(e, m, V, eos; guess = nothing) = sscheme_inverse(e, m, V, eos)
elseif SPLIT
    particle_forward(s, S_in, M_vec, e0, eos; M_mag = sqrt(M_vec[1]^2 + M_vec[2]^2), γ0 = 1.0, guess = nothing,
                     tol = 1e-12, maxiter = 50, verbose = false) = split_forward(s, S_in, M_vec, e0, eos; M_mag, tol, maxiter, verbose)
    particle_inverse(e, m, V, eos; guess = nothing) = split_inverse(e, m, V, eos; guess)
else
    @inline particle_forward(s, S_in, M_vec, e0, eos; M_mag = sqrt(M_vec[1]^2 + M_vec[2]^2), γ0 = 1.0, guess = nothing,
                             tol = 1e-12, maxiter = 50, verbose = false) = target_forward(s, S_in, M_vec, e0, eos; M_mag, tol, maxiter, verbose)
    @inline particle_inverse(e, m, V, eos; guess = nothing) = target_inverse(e, m, V, eos)
end

# Particle energy e_a = s_a ℰ(S_a, m_a/V_a) at the current state, through the forward map.
# (Same arithmetic as solve_eos! on a copy with M_smooth = m_a S_a/s_a, without the copy.)
function particle_energy(pa::Particle, eos::AbstractEOS)
    SSCHEME && return sscheme_from_label(pa.s, pa.m, pa.V, SSchemeConsts(eos), pa.St).H
    return pa.s * particle_forward(pa.s, pa.S_smooth, pa.m * (pa.S_smooth / pa.s), pa.e, eos; γ0 = pa.γ).HS
end
