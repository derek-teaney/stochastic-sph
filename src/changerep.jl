# changerep.jl — the viscous step in the particle representation (docs/changerep.tex).
#
# The Hamiltonian step works with smoothed fields: b_a = ẋ_a / Teff_a is the fluid
# velocity smoothed twice.  Here the viscous step is done instead with each
# particle's own (pointwise) thermodynamic force
#
#   β_a = v_a / T_a ,   (v_a, T_a) from the EOS at entropy density S_a and momentum
#                       density m_a / V_a ,   V_a = s_a / S_a ,
#
# where T_a is the code's density-frame temperature (e+p)/S = ∂h/∂S.  β_a is the
# derivative of the particle energy E_a = V_a h(s_a/V_a, m_a/V_a) at fixed V_a:
#   ∂E_a/∂m_a = v_a ,  ∂E_a/∂s_a = T_a ,  so  β_a = (∂E_a/∂m_a)/(∂E_a/∂s_a).
# The step Δm_a = Δt F_a, Δs_a = −β_a·Δm_a then keeps each E_a fixed to first order.
#
# Include after sph.jl and celllist.jl.

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

# One viscous step in the particle representation, at fixed positions.
# Requires params.dissipation = :bond (the bond force takes an arbitrary b-field).
# Returns (K, ΣΔs): K evaluated with the pointwise β, and the entropy produced.
function dissipative_step_changerep!(particles::Vector{Particle}, params::SimParams,
                                     cl::CellList2D, dt::Float64, eos::AbstractEOS)
    params.dissipation === :bond ||
        error("dissipative_step_changerep! uses the bond form (params.dissipation = :bond)")
    N    = length(particles)
    dotm = zeros(SVector{2,Float64}, N)
    β    = zeros(SVector{2,Float64}, N)

    smooth_to_particles!(particles, params, cl)     # S_a, hence V_a = s_a/S_a
    update_thermodynamics!(particles, eos)          # warm start for the pointwise solves
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

# ── Exact step: forward map to particle energies, momentum transfer, back map ──
#
# Particle a has volume V_a = s_a/S_a (held fixed), energy e_a = V_a h(S_a, m_a/V_a)
# (held fixed), and entropy S_a(e_a, m_a) = V_a σ(e_a/V_a, m_a/V_a).  The viscous step
# changes m_a; the back map sets s_a' = S_a(e_a, m_a').  docs/changerep.tex.

# State from the energy density ε = wγ² − p and the Landau momentum density
# M = wγ²v = (ε + p) v, as in solve_eos!.  The rest-frame energy density is
# e = ε − M·v = ε − |M|²/(ε + p(e)) (a fixed point; a contraction for |v| < 1).
# Returns (S, v, T) with S = w γ / T_rel the entropy density and T = (e+p)/S.
function eos_from_energy(ε::Float64, M::SVector{2,Float64}, eos::AbstractEOS;
                         tol=1e-15, maxiter=500)
    M2 = dot(M, M)
    e  = ε
    for _ in 1:maxiter
        e_new = ε - M2 / (ε + pressure(eos, e))
        done  = abs(e_new - e) <= tol * ε
        e = e_new
        done && break
    end
    w = e + pressure(eos, e)
    v = M / (ε + pressure(eos, e))
    γ = 1 / sqrt(1 - dot(v, v))
    S = w * γ / Trel(eos, e)
    return S, v, w / S
end

# Particle energy e_a = s_a ℰ(S_a, m_a/V_a) at the current state.
particle_energy(pa::Particle, eos::AbstractEOS) = begin
    q = deepcopy(pa)
    q.M_smooth = pa.m * (pa.S_smooth / pa.s)
    solve_eos!(q, eos)
    pa.s * q.H_over_S
end

# One viscous step with the exact back map.  Returns (K, ΣΔs, Σe) with ΣΔs the exact
# change of Σ_a S_a(e_a, m_a) and Σe the (conserved) total particle energy.
function dissipative_step_changerep_exact!(particles::Vector{Particle}, params::SimParams,
                                           cl::CellList2D, dt::Float64, eos::AbstractEOS)
    params.dissipation === :bond ||
        error("dissipative_step_changerep_exact! uses the bond form (params.dissipation = :bond)")
    N    = length(particles)
    dotm = zeros(SVector{2,Float64}, N)
    β    = zeros(SVector{2,Float64}, N)

    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
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

# ── Gradient-corrected conversion map (docs/changerep.tex, appendix) ──────────
#
# The excess Σe − H is the local variance of μ = m/s around each particle.  For a smooth
# flow part of it is the gradient across the star, g_a = ½ s_a tr(C_a G_a Q_a G_aᵀ),
# with G_a the least-squares gradient of μ on the star of a (kernel weights
# λ_ab = s_b W_ab / S_a), Q_a the second moment of the star, and C_a = S ∂v/∂M the
# inverse susceptibility.  The corrected step holds e_a' = e_a − g_a fixed, refits the
# gradient after the momentum transfer, and sets s_a' = S_a(e_a' + g_a', m_a').

# Star of a for the kernel weights: (b, r_ab, λ_ab), including b = a.
function kernel_star(a, particles, params, cl)
    pa = particles[a];  out = Tuple{Int,SVector{2,Float64},Float64}[]
    ix, iy = cell_index(pa.x, cl)
    for dx in -1:1, dy in neighbor_range(cl.ncy)
        jx = mod(ix + dx - 1, cl.ncx) + 1;  jy = mod(iy + dy - 1, cl.ncy) + 1
        for b in particles_in_cell(cl, jx, jy)
            r = minimum_image(particles[b].x - pa.x, params.L)
            W = wendland_c2(norm(r), params.h, params.kernel)
            W > 0 && push!(out, (b, r, particles[b].s * W / pa.S_smooth))
        end
    end
    return out
end

# Centred second moment Q_a and the fitted gradient G_a (G[i,j] = ∂_j μ_i) of a field
# μ_b on a star; returns (G, Q).  Singular stars (1D) use the pseudo-inverse.
function star_fit(star, μ)
    rbar = sum(λ * r for (b, r, λ) in star)
    μbar = sum(λ * μ[b] for (b, r, λ) in star)
    Q = zero(SMatrix{2,2,Float64});  Y = zero(SMatrix{2,2,Float64})
    for (b, r, λ) in star
        δr = r - rbar
        Q += λ * (δr * δr');  Y += λ * ((μ[b] - μbar) * δr')
    end
    return Y * pinv(Matrix(Q)) |> SMatrix{2,2,Float64}, Q
end

# Inverse susceptibility C = S ∂v/∂M at the smoothed state of particle a (finite differences).
function inverse_susceptibility(pa::Particle, eos::AbstractEOS)
    C = zero(MMatrix{2,2,Float64});  δ = 1e-6 * max(norm(pa.M_smooth), pa.S_smooth * 1e-3)
    for j in 1:2
        dM = SVector{2,Float64}(j == 1 ? (δ, 0.0) : (0.0, δ))
        vp = (q = deepcopy(pa); q.M_smooth = pa.M_smooth + dM; solve_eos!(q, eos); q.v_over_S * q.S_smooth)
        vm = (q = deepcopy(pa); q.M_smooth = pa.M_smooth - dM; solve_eos!(q, eos); q.v_over_S * q.S_smooth)
        C[:, j] = pa.S_smooth * (vp - vm) / (2δ)
    end
    return SMatrix(C)
end

gradient_energy(s, C, G, Q) = 0.5 * s * tr(C * G * Q * G')

# One viscous step with the gradient-corrected map.  Returns (K, ΣΔs, Σe′).
function dissipative_step_changerep_grad!(particles::Vector{Particle}, params::SimParams,
                                          cl::CellList2D, dt::Float64, eos::AbstractEOS)
    params.dissipation === :bond ||
        error("dissipative_step_changerep_grad! uses the bond form (params.dissipation = :bond)")
    N    = length(particles)
    dotm = zeros(SVector{2,Float64}, N)
    β    = zeros(SVector{2,Float64}, N)

    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
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
