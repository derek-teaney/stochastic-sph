# ideal.jl — the ideal (Hamiltonian) step: smoothing, velocities, the Hamiltonian force,
# the time step and the leapfrog; smooth_to_grid for pictures.

"""
    smooth_to_particles!(particles::Vector{Particle}, params::SimParams, cl::CellList2D)

    Compute the smoothed entropy density S_smooth and momentum density M_smooth at each particle position using the SPH kernel and the cell list for efficient neighbor search.
"""
function smooth_to_particles!(particles::Vector{Particle}, params::SimParams,
                               cl::CellList2D)
    params.hamiltonian === :energy && return smooth_entropy_energy!(particles, params, cl)
    nl = neighbor_list!(cl, particles, params.h)
    Threads.@threads for a in eachindex(particles)
        S_acc = 0.0
        M_acc = zero(SVector{2,Float64})
        begin                                            # neighbours in cell-scan order
            for b in neighbors(nl, a)
                r_mic = minimum_image(particles[a].x - particles[b].x, params.L)
                r_mag = sqrt(r_mic[1]^2 + r_mic[2]^2)
                w = kernel_W(r_mag, params.h, params.kernel)
                S_acc += particles[b].s * w
                M_acc += particles[b].m * w
            end
        end
        particles[a].S_smooth = S_acc
        particles[a].V = particles[a].s / S_acc
        particles[a].M_smooth = M_acc
    end
end

# hamiltonian = :energy (energystep.tex, eqs. Va, H): only the entropy density is
# smoothed, S_a = Σ_b s_b W_ab.  The momentum density is the particle's own,
#   M_smooth = m_a / V_a = m_a S_a / s_a ,
# so solve_eos! returns, unchanged, the particle's own T_a, v_a, p_a and e_a/s_a.
# volume = :number (volume_n.tex): S_a = s_a n_a, n_a = Σ_b W_ab, so V_a = 1/n_a.
function smooth_entropy_energy!(particles::Vector{Particle}, params::SimParams,
                                cl::CellList2D)
    nl = neighbor_list!(cl, particles, params.h)
    number = params.volume === :number
    Threads.@threads for a in eachindex(particles)
        S_acc = 0.0
        for b in neighbors(nl, a)
            r_mic = minimum_image(particles[a].x - particles[b].x, params.L)
            r_mag = sqrt(r_mic[1]^2 + r_mic[2]^2)
            S_acc += (number ? 1.0 : particles[b].s) * kernel_W(r_mag, params.h, params.kernel)
        end
        if SSCHEME                                # V_a = 1/n_a stored; S_smooth the target's entropy density (the label can be ≤ 0)
            pa = particles[a];  pa.V = 1 / S_acc;  pa.M_smooth = pa.m * S_acc;  pa.S_smooth = pa.St * S_acc
            continue
        end
        number && (S_acc *= particles[a].s)
        particles[a].S_smooth = S_acc
        particles[a].M_smooth = particles[a].m * (S_acc / particles[a].s)
        particles[a].V = particles[a].s / S_acc   # the expression the code used before V was stored (bit-identical)
    end
end

# Writes pa.dotx = Σ_b s_b (v/S)_b W  and  pa.Teff = H_over_S_a − Σ_b s_b (F/S)_b W
# for every particle in a single neighbor pass (step 3 of the update chain).
function update_velocities!(particles::Vector{Particle},
                             params::SimParams, cl::CellList2D)
    params.hamiltonian === :energy && return update_velocities_energy!(particles, params, cl)
    nl = neighbor_list!(cl, particles, params.h)
    Threads.@threads for a in eachindex(particles)
        dotx_acc = zero(SVector{2,Float64})
        FS_acc   = 0.0
        begin                                            # neighbours in cell-scan order
            for b in neighbors(nl, a)
                r_vec = minimum_image(particles[a].x - particles[b].x, params.L)
                r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
                w = kernel_W(r_mag, params.h, params.kernel)
                dotx_acc += particles[b].s * particles[b].v_over_S * w
                FS_acc   += particles[b].s * particles[b].F_over_S * w
            end
        end
        particles[a].dotx = dotx_acc
        particles[a].Teff = particles[a].H_over_S - FS_acc
    end
end

# hamiltonian = :energy (energystep.tex, eqs. xdot, Teff):
#   ẋ_a = ∂H/∂m_a = v_a                          (local, no neighbour sum)
#   Teff_a = ∂H/∂s_a = T_a − p_a/S_a + Σ_b s_b (p_b/S_b²) W_ab
# Needs S_smooth and the EOS fields (smooth_to_particles! + update_thermodynamics!).
# volume = :number (volume_n.tex): V_a does not depend on s, so Teff_a = T_a exactly.
function update_velocities_energy!(particles::Vector{Particle},
                                   params::SimParams, cl::CellList2D)
    if params.volume === :number
        Threads.@threads for a in eachindex(particles)
            pa = particles[a];  pa.dotx = pa.v;  pa.Teff = pa.T
        end
        return
    end
    nl = neighbor_list!(cl, particles, params.h)
    Threads.@threads for a in eachindex(particles)
        pS2_acc = 0.0
        for b in neighbors(nl, a)
            r_vec = minimum_image(particles[a].x - particles[b].x, params.L)
            r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
            pb = particles[b]
            pS2_acc += pb.s * (pb.p / pb.S_smooth^2) * kernel_W(r_mag, params.h, params.kernel)
        end
        pa = particles[a]
        pa.dotx = pa.v_over_S * pa.S_smooth
        pa.Teff = pa.T - pa.p / pa.S_smooth + pS2_acc
    end
end

# Each particle gathers its own force over its neighbours (threads over particles).
# The pair term is odd under a ↔ b (bracket symmetric, kernel gradient odd in r), so
# Σ_a ṁ_a = 0 up to roundoff; the previous pair loop (each pair once, ± to both) made
# it exact but could not be threaded.
function hamiltonian_force!(dotm::Vector{SVector{2,Float64}},
                             particles::Vector{Particle},
                             params::SimParams, cl::CellList2D)
    params.hamiltonian === :energy && return hamiltonian_force_energy!(dotm, particles, params, cl)
    nl = neighbor_list!(cl, particles, params.h)
    Threads.@threads for a in eachindex(particles)
        sa, ma = particles[a].s, particles[a].m
        vSa, FSa = particles[a].v_over_S, particles[a].F_over_S
        acc = zero(SVector{2,Float64})
        begin                                            # neighbours in cell-scan order
            for b in neighbors(nl, a)
                b == a && continue
                r_vec = minimum_image(particles[a].x - particles[b].x, params.L)
                r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
                r_mag >= 2params.h && continue
                grad = kernel_gradW(r_vec, r_mag, params.h, params.kernel)
                sb, mb = particles[b].s, particles[b].m
                vSb, FSb = particles[b].v_over_S, particles[b].F_over_S
                bracket = -(dot(vSa, mb)*sa + dot(vSb, ma)*sb) +
                           (FSa + FSb)*sa*sb
                acc += bracket * grad
            end
        end
        dotm[a] = acc
    end
end

# hamiltonian = :energy (energystep.tex, eq. mdot): the symmetric SPH pressure force
#   ṁ_a = −∂H/∂x_a = −Σ_b s_a s_b (p_a/S_a² + p_b/S_b²) ∂W_ab/∂x_a ,
# odd under a ↔ b, so Σ_a ṁ_a = 0 up to roundoff.
# volume = :number (volume_n.tex), V_a = 1/n_a:
#   ṁ_a = −Σ_b (p_a V_a² + p_b V_b²) ∂W_ab/∂x_a ,
# the same loop with s → 1 and S → n (p/n² = p V², V = s/S_smooth).
function hamiltonian_force_energy!(dotm::Vector{SVector{2,Float64}},
                                   particles::Vector{Particle},
                                   params::SimParams, cl::CellList2D)
    nl = neighbor_list!(cl, particles, params.h)
    number = params.volume === :number
    Threads.@threads for a in eachindex(particles)
        sa  = number ? 1.0 : particles[a].s
        pSa = number ? particles[a].p * particles[a].V^2 :
                       particles[a].p / particles[a].S_smooth^2
        acc = zero(SVector{2,Float64})
        for b in neighbors(nl, a)
            b == a && continue
            r_vec = minimum_image(particles[a].x - particles[b].x, params.L)
            r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
            r_mag >= 2params.h && continue
            grad = kernel_gradW(r_vec, r_mag, params.h, params.kernel)
            pb   = particles[b]
            acc -= number ? (pSa + pb.p * pb.V^2) * grad :
                            sa * pb.s * (pSa + pb.p / pb.S_smooth^2) * grad
        end
        dotm[a] = acc
    end
end

# ── Timestep ─────────────────────────────────────────────────────────────────
# mode=:ideal  → acoustic CFL only: dt = CFL * h / cs
# mode=:viscous → min(acoustic, diffusive) where diffusive uses minimum T_rel
#   T0=nothing  : scan particles (ignoring those with s < s_thresh * s_mean)
#   T0=<value>  : use a fixed rest-frame temperature T_rel (uniform background, tests)

function compute_dt(particles, params, eos;
                    mode     = :viscous,
                    CFL      = 0.2,
                    T0       = nothing,
                    s_thresh = 1e-4)
    cs          = sqrt(cs2(eos))
    dt_acoustic = CFL * params.h / cs

    mode == :ideal && return dt_acoustic

    η_s  = eta_by_s(eos)
    ζ_s  = zeta_by_s(eos)
    d    = params.phys_dim
    visc = 2 * (d - 1) / d * η_s + ζ_s          # longitudinal viscosity 2η(1 − 1/d) + ζ  (4/3 η + ζ at d = 3)
    visc == 0.0 && return dt_acoustic

    if T0 !== nothing
        Trel_min = T0
    else
        s_mean = sum(entropy_of(pa) for pa in particles) / length(particles)
        s_cut  = s_thresh * s_mean
        Trel_min = minimum(target_trel(pa, eos) for pa in particles if entropy_of(pa) > s_cut)   # T_rel = γ T
    end

    Γ_s          = visc / Trel_min
    dt_diffusive = CFL * params.h^2 / Γ_s
    dt = min(dt_acoustic, dt_diffusive)
    # The bond step damps the shortest modes at up to Γ_max (changerep.tex, "The time step").  generic_step! applies two
    # viscous half steps of Δt/2 each; require Γ_bound Δt/2 ≤ 1, with Γ_bound ≥ Γ_max.
    dt = min(dt, 2 / viscous_rate_bound(params, eos, Trel_min))
    return dt
end

# Upper bound on the damping rate of the particle-representation viscous step.
# Gershgorin on the bond coefficients: Γ_max ≤ 4 (κ_b/w) max_a Σ_b w_ab, and for the
# Wendland bond weight ψ ∝ (1 − ρ/2h)³ in two dimensions
#   Σ_b w_ab ≤ 2 ∫ψ / ∫ψ ρ² = c_2 / h²,   c_2 = 7/2
# (the continuum integral includes ρ → 0, which the bond sum omits, so it bounds the
# lattice sum).  κ_b is the largest viscosity carried by the bonds: 2η + ζ_eff.  κ_b/w = (κ_b/s)/T_rel.
function viscous_rate_bound(params::SimParams, eos::AbstractEOS, Trel::Float64)
    d  = params.phys_dim
    η_s, ζ_s = eta_by_s(eos), zeta_by_s(eos)
    κb_s = 2η_s + zeta_eff_by_s(eos, d)
    c_d  = 7 / 2
    return 4 * (κb_s / Trel) * c_d / params.h^2       # (the bond length h_ψ was h; bond_h removed, focus_plan.tex Q5)
end

# The ideal step: the generalized leapfrog (hmc.jl, generalized_step!; implicit, symplectic and reversible for the
# non-separable H; equilibrium_positions.tex).  It needs the n-scheme (hamiltonian = :energy, volume = :number).  The
# drift–kick–drift leapfrog (SPH_IDEAL=particle) and the grid step (SPH_IDEAL=grid) were removed on 2026-11-27
# (docs/focus_plan.tex, Q6; tag pre-prune).
function leapfrog_step!(particles::Vector{Particle}, params::SimParams,
                         cl::CellList2D, dt::Float64, eos::AbstractEOS,
                         dotm::Vector{SVector{2,Float64}})
    params.hamiltonian === :energy && params.volume === :number ||
        error("leapfrog_step!: the generalized leapfrog needs hamiltonian = :energy and volume = :number (the drift–kick–drift " *
              "leapfrog of the other schemes was removed on 2026-11-27, docs/focus_plan.tex Q6; tag pre-prune)")
    return generalized_step!(particles, params, cl, dt, eos, dotm)   # hmc.jl
end

# ── Visualization ─────────────────────────────────────────────────────────────

function smooth_to_grid(particles::Vector{Particle}, params::SimParams,
                        cl::CellList2D, field)
    ncx       = cl.ncx;  ncy = cl.ncy
    cell_size = cl.cell_size
    L         = params.L
    h         = params.h
    S  = zeros(ncx, ncy)
    xs = LinRange(0.5 * cell_size[1], L[1] - 0.5 * cell_size[1], ncx)
    ys = LinRange(0.5 * cell_size[2], L[2] - 0.5 * cell_size[2], ncy)
    for iy in 1:ncy, ix in 1:ncx
        xg_vec = SVector((ix - 0.5) * cell_size[1], (iy - 0.5) * cell_size[2])
        acc = 0.0
        for dy in neighbor_range(ncy), dx in -1:1
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                r_mic = minimum_image(xg_vec - particles[b].x, L)
                r_mag = sqrt(r_mic[1]^2 + r_mic[2]^2)
                acc  += field(particles[b]) * kernel_W(r_mag, h, params.kernel)
            end
        end
        S[ix, iy] = acc
    end
    return S, xs, ys
end
