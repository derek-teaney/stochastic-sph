# With ncy==1 (1D mode), mod(anything, 1)==0 maps all three dy offsets to the same
# cell, triple-counting every particle.  Clamp the stencil to a single offset.
neighbor_range(n::Int) = n == 1 ? (0:0) : (-1:1)

mutable struct CellList2D
    cell_start :: Matrix{Int}       # (ncx, ncy)  start index into flat array
    cell_count :: Matrix{Int}       # (ncx, ncy)  particles currently in cell
    indices    :: Vector{Int}       # (N,)        flat contiguous particle index array
    ncx        :: Int               # cells in x direction
    ncy        :: Int               # cells in y direction
    cell_size  :: Float64           # Lx/ncx ≈ Ly/ncy ≈ 2h
    L          :: SVector{2,Float64}
end

function CellList2D(L::SVector{2,Float64}, h::Float64, N::Int)
    ncx = floor(Int, L[1] / (2h))
    ncy = max(1, floor(Int, L[2] / (2h)))
    @assert ncx >= 3 "Box too small in x: need ncx ≥ 3, got $ncx"
    cell_size = L[1] / ncx
    CellList2D(zeros(Int,ncx,ncy), zeros(Int,ncx,ncy), zeros(Int,N), ncx, ncy, cell_size, L)
end

function cell_index(x::SVector{2,Float64}, cl::CellList2D)
    ix = mod(floor(Int, x[1] / cl.cell_size), cl.ncx) + 1
    iy = mod(floor(Int, x[2] / cl.cell_size), cl.ncy) + 1
    return ix, iy
end

function rebuild_cell_list!(cl::CellList2D, particles::Vector{Particle})
    N   = length(particles)
    ncx = cl.ncx
    ncy = cl.ncy

    # pass 1: count particles per cell
    cl.cell_count .= 0
    for a in 1:N
        ix, iy = cell_index(particles[a].x, cl)
        cl.cell_count[ix, iy] += 1
    end

    # prefix sum → start indices (column-major order: ix fast)
    idx = 1
    for iy in 1:ncy, ix in 1:ncx
        cl.cell_start[ix, iy] = idx
        idx += cl.cell_count[ix, iy]
    end

    # pass 2: fill flat index array (reuse cell_count as fill counter)
    cl.cell_count .= 0
    for a in 1:N
        ix, iy = cell_index(particles[a].x, cl)
        offset = cl.cell_count[ix, iy]
        cl.indices[cl.cell_start[ix, iy] + offset] = a
        cl.cell_count[ix, iy] += 1
    end
end

function particles_in_cell(cl::CellList2D, ix::Int, iy::Int)
    start = cl.cell_start[ix, iy]
    count = cl.cell_count[ix, iy]
    return @view cl.indices[start : start + count - 1]
end

function wrap_positions!(particles::Vector{Particle}, L::SVector{2,Float64})
    for a in eachindex(particles)
        particles[a].x = mod.(particles[a].x, L)
    end
end

"""
    smooth_to_particles!(particles::Vector{Particle}, params::SimParams, cl::CellList2D)

    Compute the smoothed entropy density S_smooth and momentum density M_smooth at each particle position using the SPH kernel and the cell list for efficient neighbor search.
"""
function smooth_to_particles!(particles::Vector{Particle}, params::SimParams,
                               cl::CellList2D)
    ncx = cl.ncx;  ncy = cl.ncy
    for a in eachindex(particles)
        S_acc = 0.0
        M_acc = zero(SVector{2,Float64})
        ix, iy = cell_index(particles[a].x, cl)
        for dx in -1:1, dy in neighbor_range(ncy)
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                r_mic = minimum_image(particles[a].x - particles[b].x, params.L)
                r_mag = sqrt(r_mic[1]^2 + r_mic[2]^2)
                w = wendland_c2(r_mag, params.h, params.kernel)
                S_acc += particles[b].s * w
                M_acc += particles[b].m * w
            end
        end
        particles[a].S_smooth = S_acc
        particles[a].M_smooth = M_acc
    end
end

# Writes pa.dotx = Σ_b s_b (v/S)_b W  and  pa.Teff = H_over_S_a − Σ_b s_b (F/S)_b W
# for every particle in a single neighbor pass (step 3 of the update chain).
function update_velocities!(particles::Vector{Particle},
                             params::SimParams, cl::CellList2D)
    ncx = cl.ncx;  ncy = cl.ncy
    for a in eachindex(particles)
        dotx_acc = zero(SVector{2,Float64})
        FS_acc   = 0.0
        ix, iy = cell_index(particles[a].x, cl)
        for dx in -1:1, dy in neighbor_range(ncy)
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                r_vec = minimum_image(particles[a].x - particles[b].x, params.L)
                r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
                w = wendland_c2(r_mag, params.h, params.kernel)
                dotx_acc += particles[b].s * particles[b].v_over_S * w
                FS_acc   += particles[b].s * particles[b].F_over_S * w
            end
        end
        particles[a].dotx = dotx_acc
        particles[a].Teff = particles[a].H_over_S - FS_acc
    end
end

function hamiltonian_force!(dotm::Vector{SVector{2,Float64}},
                             particles::Vector{Particle},
                             params::SimParams, cl::CellList2D)
    ncx = cl.ncx;  ncy = cl.ncy
    fill!(dotm, zero(SVector{2,Float64}))
    for a in eachindex(particles)
        ix, iy = cell_index(particles[a].x, cl)
        for dx in -1:1, dy in neighbor_range(ncy)
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                b <= a && continue # avoid double counting and self-interaction

                r_vec = minimum_image(particles[a].x - particles[b].x, params.L)
                r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
                r_mag >= 2params.h && continue
                grad = wendland_c2_grad(r_vec, r_mag, params.h, params.kernel)

                sa, sb   = particles[a].s,        particles[b].s
                ma, mb   = particles[a].m,        particles[b].m
                vSa, vSb = particles[a].v_over_S, particles[b].v_over_S
                FSa, FSb = particles[a].F_over_S, particles[b].F_over_S

                bracket = -(dot(vSa, mb)*sa + dot(vSb, ma)*sb) +
                           (FSa + FSb)*sa*sb
                force_a  = bracket * grad
                dotm[a] += force_a
                dotm[b] -= force_a
            end
        end
    end
end

# ── Timestep ─────────────────────────────────────────────────────────────────
# mode=:ideal  → acoustic CFL only: dt = CFL * h / cs
# mode=:viscous → min(acoustic, diffusive) where diffusive uses minimum T_rel
#   T0=nothing  : scan particles (ignoring those with s < s_thresh * s_mean)
#   T0=<value>  : use fixed temperature (uniform background, tests)

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
    visc = 4/3 * η_s + ζ_s
    visc == 0.0 && return dt_acoustic

    if T0 !== nothing
        T_eff = T0
    else
        s_mean = sum(pa.s for pa in particles) / length(particles)
        s_cut  = s_thresh * s_mean
        T_eff  = minimum(pa.T * pa.γ for pa in particles if pa.s > s_cut)
    end

    Γ_s          = visc / T_eff
    dt_diffusive = CFL * params.h^2 / Γ_s
    return min(dt_acoustic, dt_diffusive)
end

function leapfrog_step!(particles::Vector{Particle}, params::SimParams,
                         cl::CellList2D, dt::Float64, eos::AbstractEOS,
                         dotm::Vector{SVector{2,Float64}})
    # Störmer–Verlet / position Verlet (DKD) — 2nd-order symplectic:
    #
    #   x^{n+1/2} = x^n      + (dt/2) v(x^n,       m^n      )
    #   m^{n+1}   = m^n      +  dt    F(x^{n+1/2},  m^n      )
    #   x^{n+1}   = x^{n+1/2}+ (dt/2) v(x^{n+1/2},  m^{n+1}  )

    #Step 1: half-drift  x^n → x^{n+1/2}
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)
    for a in eachindex(particles)
        particles[a].x += (dt/2) * particles[a].dotx
    end
    wrap_positions!(particles, params.L)
    rebuild_cell_list!(cl, particles)

    # Step 2: full kick  m^n → m^{n+1}  evaluated at x^{n+1/2}
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    hamiltonian_force!(dotm, particles, params, cl)
    for a in eachindex(particles)
        particles[a].m += dt * dotm[a]
    end

    # Step 3: half-drift  x^{n+1/2} → x^{n+1}  with updated M^{n+1}
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)
    for a in eachindex(particles)
        particles[a].x += (dt/2) * particles[a].dotx
    end
    wrap_positions!(particles, params.L)
    rebuild_cell_list!(cl, particles)
end

function kdk_step!(particles::Vector{Particle}, params::SimParams,
                    cl::CellList2D, dt::Float64, eos::AbstractEOS,
                    dotm::Vector{SVector{2,Float64}})
    # Velocity Verlet (KDK) — 2nd-order symplectic:
    #
    #   m^{n+1/2} = m^n      + (dt/2) F(x^n,       m^n      )
    #   x^{n+1}   = x^n      +  dt    v(x^n,        m^{n+1/2})
    #   m^{n+1}   = m^{n+1/2}+ (dt/2) F(x^{n+1},   m^{n+1/2})

    # Step 1: half kick  m^n → m^{n+1/2}  at x^n
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    hamiltonian_force!(dotm, particles, params, cl)
    for a in eachindex(particles)
        particles[a].m += (dt/2) * dotm[a]
    end

    # Step 2: full drift  x^n → x^{n+1}  using v(x^n, m^{n+1/2})
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)
    for a in eachindex(particles)
        particles[a].x += dt * particles[a].dotx
    end
    wrap_positions!(particles, params.L)
    rebuild_cell_list!(cl, particles)

    # Step 3: half kick  m^{n+1/2} → m^{n+1}  at x^{n+1}
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    hamiltonian_force!(dotm, particles, params, cl)
    for a in eachindex(particles)
        particles[a].m += (dt/2) * dotm[a]
    end
end

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
function compute_strain(particles::Vector{Particle}, params::SimParams,
                        cl::CellList2D)
    N   = length(particles)
    ncx = cl.ncx;  ncy = cl.ncy
    h      = params.h
    L      = params.L
    kernel = params.kernel
    ε = Vector{SMatrix{2,2,Float64,4}}(undef, N)
    for a in 1:N
        ε_acc = zero(SMatrix{2,2,Float64})
        β_a   = particles[a].dotx / particles[a].Teff
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
                Δβ    = particles[b].dotx / particles[b].Teff - β_a
                # outer product: (grad * Δβ')[j,i] = D_j * Δβ_i  →  ε_{ji}
                ε_acc += particles[b].s * (grad * Δβ')
            end
        end
        ε[a] = ε_acc / S_a
    end
    return ε
end

# Requires up-to-date dotx, Teff, S_smooth (from smooth_to_particles! +
# update_thermodynamics! + update_velocities!).
function dissipative_force!(dotm::Vector{SVector{2,Float64}},
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

# One dissipative half/full step: smooth → EOS → viscous force → update (m, s).
# Positions are unchanged; entropy grows by Δs_a = −b_a · Δm_a ≥ 0.
function dissipative_step!(particles::Vector{Particle}, params::SimParams,
                            cl::CellList2D, dt::Float64, eos::AbstractEOS)
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

# Full GENERIC step: Strang splitting D(dt/2) · H(dt) · D(dt/2).
function generic_step!(particles::Vector{Particle}, params::SimParams,
                        cl::CellList2D, dt::Float64, eos::AbstractEOS,
                        dotm::Vector{SVector{2,Float64}})
    dissipative_step!(particles, params, cl, dt/2, eos)
    leapfrog_step!(particles, params, cl, dt, eos, dotm)
    dissipative_step!(particles, params, cl, dt/2, eos)
end

# ── Visualization ─────────────────────────────────────────────────────────────

function smooth_to_grid(particles::Vector{Particle}, params::SimParams,
                        cl::CellList2D, field)
    ncx       = cl.ncx;  ncy = cl.ncy
    cell_size = cl.cell_size
    L         = params.L
    h         = params.h
    S  = zeros(ncx, ncy)
    xs = LinRange(0.5 * cell_size, L[1] - 0.5 * cell_size, ncx)
    ys = LinRange(0.5 * cell_size, L[2] - 0.5 * cell_size, ncy)
    for iy in 1:ncy, ix in 1:ncx
        xg_vec = SVector((ix - 0.5) * cell_size, (iy - 0.5) * cell_size)
        acc = 0.0
        for dy in neighbor_range(ncy), dx in -1:1
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                r_mic = minimum_image(xg_vec - particles[b].x, L)
                r_mag = sqrt(r_mic[1]^2 + r_mic[2]^2)
                acc  += field(particles[b]) * wendland_c2(r_mag, h, params.kernel)
            end
        end
        S[ix, iy] = acc
    end
    return S, xs, ys
end