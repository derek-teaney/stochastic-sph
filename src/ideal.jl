# ideal.jl — the ideal (Hamiltonian) step: smoothing, velocities, the Hamiltonian force,
# the time step and the leapfrog; smooth_to_grid for pictures.

# The n-scheme (volume_n.tex, energystep.tex): H = Σ_a e_a(s_a, m_a, V_a) with the number volume V_a = 1/n_a, n_a = Σ_b W_ab.
# The smoothed Hamiltonian and the entropy volume (SPH_HAMILTONIAN, SPH_VOLUME) were removed on 2026-11-27 (docs/focus_plan.tex, Q7;
# tag pre-prune).

# Only the number density is smoothed, n_a = Σ_b W_ab, V_a = 1/n_a.  Stored as S_smooth = s_a n_a (the round trip
# V_a = s_a/S_smooth is kept: removing it changes the last bits), and the momentum density is the particle's own,
#   M_smooth = m_a / V_a ,
# so solve_eos! returns, unchanged, the particle's own T_a, v_a, p_a and e_a/s_a.
function smooth_to_particles!(particles::Vector{Particle}, params::SimParams,
                              cl::CellList2D)
    nl = neighbor_list!(cl, particles, params.h)
    Threads.@threads for a in eachindex(particles)
        S_acc = 0.0
        for b in neighbors(nl, a)
            r_mic = minimum_image(particles[a].x - particles[b].x, params.L)
            r_mag = sqrt(r_mic[1]^2 + r_mic[2]^2)
            S_acc += 1.0 * kernel_W(r_mag, params.h, params.kernel)
        end
        if SSCHEME                                # V_a = 1/n_a stored; S_smooth the target's entropy density (the label can be ≤ 0)
            pa = particles[a];  pa.V = 1 / S_acc;  pa.M_smooth = pa.m * S_acc;  pa.S_smooth = pa.St * S_acc
            continue
        end
        S_acc *= particles[a].s
        particles[a].S_smooth = S_acc
        particles[a].M_smooth = particles[a].m * (S_acc / particles[a].s)
        particles[a].V = particles[a].s / S_acc   # the expression the code used before V was stored (bit-identical)
    end
end

# energystep.tex, eqs. xdot, Teff:  ẋ_a = ∂H/∂m_a = v_a (local, no neighbour sum), and Teff_a = ∂H/∂s_a = T_a exactly,
# since V_a does not depend on s (volume_n.tex).  Needs the EOS fields (smooth_to_particles! + update_thermodynamics!).
function update_velocities!(particles::Vector{Particle},
                            params::SimParams, cl::CellList2D)
    Threads.@threads for a in eachindex(particles)
        pa = particles[a];  pa.dotx = pa.v;  pa.Teff = pa.T
    end
    return
end

# energystep.tex, eq. mdot, with V_a = 1/n_a (volume_n.tex): the symmetric SPH pressure force
#   ṁ_a = −∂H/∂x_a = −Σ_b (p_a V_a² + p_b V_b²) ∂W_ab/∂x_a ,
# odd under a ↔ b, so Σ_a ṁ_a = 0 up to roundoff.  Each particle gathers its own force over its neighbours (threads over
# particles).
function hamiltonian_force!(dotm::Vector{SVector{2,Float64}},
                            particles::Vector{Particle},
                            params::SimParams, cl::CellList2D)
    nl = neighbor_list!(cl, particles, params.h)
    Threads.@threads for a in eachindex(particles)
        pSa = particles[a].p * particles[a].V^2
        acc = zero(SVector{2,Float64})
        for b in neighbors(nl, a)
            b == a && continue
            r_vec = minimum_image(particles[a].x - particles[b].x, params.L)
            r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
            r_mag >= 2params.h && continue
            grad = kernel_gradW(r_vec, r_mag, params.h, params.kernel)
            pb   = particles[b]
            acc -= (pSa + pb.p * pb.V^2) * grad
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
# non-separable H; equilibrium_positions.tex).  The
# drift–kick–drift leapfrog (SPH_IDEAL=particle) and the grid step (SPH_IDEAL=grid) were removed on 2026-11-27
# (docs/focus_plan.tex, Q6; tag pre-prune).
function leapfrog_step!(particles::Vector{Particle}, params::SimParams,
                         cl::CellList2D, dt::Float64, eos::AbstractEOS,
                         dotm::Vector{SVector{2,Float64}})
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
