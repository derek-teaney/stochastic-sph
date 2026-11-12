# two_level.jl — superseded: the two-level SPH strain form of the viscous force (sph2.tex),
# SPH_DISSIPATION=two_level.  Kept so the older notes' figures can be made again.
# Loaded by src/legacy/load.jl.

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
