# With ncy==1 (1D mode), mod(anything, 1)==0 maps all three dy offsets to the same
# cell, triple-counting every particle.  Clamp the stencil to a single offset.
# With ncy==2 the offsets −1 and +1 both reach the other cell, double-counting it
# (thin strips, e.g. examples/shearwave_scan.jl): use the offsets 0 and 1.
neighbor_range(n::Int) = n == 1 ? (0:0) : n == 2 ? (0:1) : (-1:1)

mutable struct CellList2D
    cell_start :: Matrix{Int}       # (ncx, ncy)  start index into flat array
    cell_count :: Matrix{Int}       # (ncx, ncy)  particles currently in cell
    indices    :: Vector{Int}       # (N,)        flat contiguous particle index array
    ncx        :: Int               # cells in x direction
    ncy        :: Int               # cells in y direction
    cell_size  :: SVector{2,Float64} # (Lx/ncx, Ly/ncy), each ≥ 2h; cells tile the box exactly
    L          :: SVector{2,Float64}
    nl         :: Any                # NeighborList of the current positions, built on demand
end

function CellList2D(L::SVector{2,Float64}, h::Float64, N::Int)
    ncx = floor(Int, L[1] / (2h))
    ncy = max(1, floor(Int, L[2] / (2h)))
    @assert ncx >= 3 "Box too small in x: need ncx ≥ 3, got $ncx"
    # Separate widths in x and y.  A single width Lx/ncx reused in y does not tile
    # [0, Ly) when Ly/ncy ≠ Lx/ncx, and the periodic y-stencil then misses pairs
    # across y = 0 (test/testcelllist_box.jl).
    cell_size = SVector(L[1] / ncx, L[2] / ncy)
    CellList2D(zeros(Int,ncx,ncy), zeros(Int,ncx,ncy), zeros(Int,N), ncx, ncy, cell_size, L, nothing)
end

# Run body(range) over contiguous chunks of 1:N, one chunk per thread, so that each
# chunk can allocate its scratch buffers once.  Results do not depend on the chunking
# as long as body writes only the entries of its own range.
function threaded_chunks(body, N::Int)
    n = max(1, min(N, Threads.nthreads()))
    rs = [((c - 1) * N ÷ n + 1):(c * N ÷ n) for c in 1:n]
    Threads.@threads for c in 1:n
        body(rs[c])
    end
end

# ── Neighbour list ───────────────────────────────────────────────────────────
# For each particle a, the particles b (a itself included) with |x_b − x_a| < 2h(1 + 1e-9),
# in exactly the order of the cell scan (dx in −1:1, dy in neighbor_range(ncy), cell
# order), so that a pass over the list sums in the same order as the scan.  Each pass
# keeps its own exact distance test.  The list remembers the positions it was built from
# and is rebuilt whenever they differ (one comparison per particle per call).
mutable struct NeighborList
    offsets :: Vector{Int}                 # neighbours of a: index[offsets[a]:offsets[a+1]-1]
    index   :: Vector{Int}
    x       :: Vector{SVector{2,Float64}}  # positions at the build
    h       :: Float64
    valid   :: Bool
    parts   :: Vector{Vector{Int}}         # per-chunk buffers, reused between builds
    counts  :: Vector{Int}
end
NeighborList() = NeighborList(Int[], Int[], SVector{2,Float64}[], 0.0, false, Vector{Int}[], Int[])
neighbors(nl::NeighborList, a::Int) = @view nl.index[nl.offsets[a]:nl.offsets[a+1]-1]

function neighbor_list!(cl::CellList2D, particles::Vector{Particle}, h::Float64)
    cl.nl isa NeighborList || (cl.nl = NeighborList())
    nl = cl.nl::NeighborList;  N = length(particles)
    if nl.valid && nl.h == h && length(nl.x) == N && all(i -> nl.x[i] == particles[i].x, 1:N)
        return nl
    end
    cut2 = (2h * (1 + 1e-9))^2
    ncx = cl.ncx;  ncy = cl.ncy
    nch = max(1, min(N, Threads.nthreads()))
    length(nl.parts) == nch || (nl.parts = [Int[] for _ in 1:nch])
    resize!(nl.counts, N);  counts = nl.counts;  parts = nl.parts
    Threads.@threads for c in 1:nch
        idx = empty!(parts[c])
        for a in ((c - 1) * N ÷ nch + 1):(c * N ÷ nch)
            ix, iy = cell_index(particles[a].x, cl);  n0 = length(idx)
            for dx in -1:1, dy in neighbor_range(ncy)
                jx = mod(ix + dx - 1, ncx) + 1
                jy = mod(iy + dy - 1, ncy) + 1
                for b in particles_in_cell(cl, jx, jy)
                    r = minimum_image(particles[a].x - particles[b].x, cl.L)
                    r[1]^2 + r[2]^2 < cut2 && push!(idx, b)
                end
            end
            counts[a] = length(idx) - n0
        end
    end
    resize!(nl.offsets, N + 1);  nl.offsets[1] = 1
    for a in 1:N;  nl.offsets[a + 1] = nl.offsets[a] + counts[a];  end
    resize!(nl.index, nl.offsets[N + 1] - 1);  k = 0
    for c in 1:nch, b in parts[c];  k += 1;  nl.index[k] = b;  end
    resize!(nl.x, N)
    for a in 1:N;  nl.x[a] = particles[a].x;  end
    nl.h = h;  nl.valid = true
    return nl
end

function cell_index(x::SVector{2,Float64}, cl::CellList2D)
    ix = mod(floor(Int, x[1] / cl.cell_size[1]), cl.ncx) + 1
    iy = mod(floor(Int, x[2] / cl.cell_size[2]), cl.ncy) + 1
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
    Threads.@threads for a in eachindex(particles)
        particles[a].x = mod.(particles[a].x, L)
    end
end

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
                w = wendland_c2(r_mag, params.h, params.kernel)
                S_acc += particles[b].s * w
                M_acc += particles[b].m * w
            end
        end
        particles[a].S_smooth = S_acc
        particles[a].M_smooth = M_acc
    end
end

# hamiltonian = :energy (energystep.tex, eqs. Va, H): only the entropy density is
# smoothed, S_a = Σ_b s_b W_ab.  The momentum density is the particle's own,
#   M_smooth = m_a / V_a = m_a S_a / s_a ,
# so solve_eos! returns, unchanged, the particle's own T_a, v_a, p_a and e_a/s_a.
function smooth_entropy_energy!(particles::Vector{Particle}, params::SimParams,
                                cl::CellList2D)
    nl = neighbor_list!(cl, particles, params.h)
    Threads.@threads for a in eachindex(particles)
        S_acc = 0.0
        for b in neighbors(nl, a)
            r_mic = minimum_image(particles[a].x - particles[b].x, params.L)
            r_mag = sqrt(r_mic[1]^2 + r_mic[2]^2)
            S_acc += particles[b].s * wendland_c2(r_mag, params.h, params.kernel)
        end
        particles[a].S_smooth = S_acc
        particles[a].M_smooth = particles[a].m * (S_acc / particles[a].s)
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
                w = wendland_c2(r_mag, params.h, params.kernel)
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
function update_velocities_energy!(particles::Vector{Particle},
                                   params::SimParams, cl::CellList2D)
    nl = neighbor_list!(cl, particles, params.h)
    Threads.@threads for a in eachindex(particles)
        pS2_acc = 0.0
        for b in neighbors(nl, a)
            r_vec = minimum_image(particles[a].x - particles[b].x, params.L)
            r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
            pb = particles[b]
            pS2_acc += pb.s * (pb.p / pb.S_smooth^2) * wendland_c2(r_mag, params.h, params.kernel)
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
                grad = wendland_c2_grad(r_vec, r_mag, params.h, params.kernel)
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
function hamiltonian_force_energy!(dotm::Vector{SVector{2,Float64}},
                                   particles::Vector{Particle},
                                   params::SimParams, cl::CellList2D)
    nl = neighbor_list!(cl, particles, params.h)
    Threads.@threads for a in eachindex(particles)
        sa  = particles[a].s
        pSa = particles[a].p / particles[a].S_smooth^2
        acc = zero(SVector{2,Float64})
        for b in neighbors(nl, a)
            b == a && continue
            r_vec = minimum_image(particles[a].x - particles[b].x, params.L)
            r_mag = sqrt(r_vec[1]^2 + r_vec[2]^2)
            r_mag >= 2params.h && continue
            grad = wendland_c2_grad(r_vec, r_mag, params.h, params.kernel)
            pb   = particles[b]
            acc -= sa * pb.s * (pSa + pb.p / pb.S_smooth^2) * grad
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
    visc = 4/3 * η_s + ζ_s
    visc == 0.0 && return dt_acoustic

    if T0 !== nothing
        Trel_min = T0
    else
        s_mean = sum(pa.s for pa in particles) / length(particles)
        s_cut  = s_thresh * s_mean
        Trel_min = minimum(pa.T * pa.γ for pa in particles if pa.s > s_cut)   # T_rel = γ T
    end

    Γ_s          = visc / Trel_min
    dt_diffusive = CFL * params.h^2 / Γ_s
    dt = min(dt_acoustic, dt_diffusive)
    if params.dissipation === :changerep
        # The particle representation damps the shortest modes at up to Γ_max
        # (changerep.tex, "The time step").  generic_step! applies two viscous half
        # steps of Δt/2 each; require Γ_bound Δt/2 ≤ 1, with Γ_bound ≥ Γ_max.
        dt = min(dt, 2 / changerep_rate_bound(params, eos, Trel_min))
    end
    return dt
end

# Upper bound on the damping rate of the particle-representation viscous step.
# Gershgorin on the bond coefficients: Γ_max ≤ 4 (κ_b/w) max_a Σ_b w_ab, and for the
# Wendland bond weight ψ ∝ (1 − ρ/2h_ψ)³ in d_s dimensions
#   Σ_b w_ab ≤ d_s ∫ψ / ∫ψ ρ² = c_d / h_ψ²,   c_1 = 15/4,  c_2 = 7/2
# (the continuum integral includes ρ → 0, which the bond sum omits, so it bounds the
# lattice sum).  κ_b is the largest viscosity carried by the bonds: κ(v=0) = 2η(d−1)/d + ζ
# in 1D, 2η + ζ_eff in 2D.  κ_b/w = (κ_b/s)/T_rel.  In 1D at h/a = 4 the bound is about twice
# the measured rate of the zigzag mode.
function changerep_rate_bound(params::SimParams, eos::AbstractEOS, Trel::Float64)
    ds = params.kernel isa WendlandC2_1D ? 1 : 2
    d  = params.d
    η_s, ζ_s = eta_by_s(eos), zeta_by_s(eos)
    κb_s = ds == 1 ? 2η_s * (d - 1) / d + ζ_s : 2η_s + ζ_s + 2η_s * (1 / 2 - 1 / d)
    c_d  = ds == 1 ? 15 / 4 : 7 / 2
    return 4 * (κb_s / Trel) * c_d / params.bond_h^2
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
    Threads.@threads for a in eachindex(particles)
        particles[a].x += (dt/2) * particles[a].dotx
    end
    wrap_positions!(particles, params.L)
    rebuild_cell_list!(cl, particles)

    # Step 2: full kick  m^n → m^{n+1}  evaluated at x^{n+1/2}
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    hamiltonian_force!(dotm, particles, params, cl)
    Threads.@threads for a in eachindex(particles)
        particles[a].m += dt * dotm[a]
    end

    # Step 3: half-drift  x^{n+1/2} → x^{n+1}  with updated M^{n+1}
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)
    Threads.@threads for a in eachindex(particles)
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
#   :changerep  the bond form evaluated on the particle force β_a (changerep.tex);
#               the full viscous step is dissipative_step_changerep_grad!
# Requires up-to-date dotx, Teff, S_smooth (from smooth_to_particles! +
# update_thermodynamics! + update_velocities!).
function dissipative_force!(dotm::Vector{SVector{2,Float64}},
                             particles::Vector{Particle},
                             params::SimParams, cl::CellList2D,
                             eos::AbstractEOS)
    if params.dissipation === :bond
        bond_dissipative_force!(dotm, particles, params, cl, eos)
    elseif params.dissipation === :changerep
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
# For :changerep the shortest modes are damped at a rate of order (κ/w)/h²; compute_dt
# bounds this rate (changerep_rate_bound) so that each viscous half step of
# generic_step!, of length Δt/2, decays every mode monotonically.
function dissipative_step!(particles::Vector{Particle}, params::SimParams,
                            cl::CellList2D, dt::Float64, eos::AbstractEOS)
    if params.hamiltonian === :energy
        eta_by_s(eos) == 0.0 && zeta_by_s(eos) == 0.0 && return
        dissipative_step_energy!(particles, params, cl, dt, eos)
        return
    end
    if params.dissipation === :changerep
        # no viscosity: nothing to do (the conversion maps would only round-trip s)
        eta_by_s(eos) == 0.0 && zeta_by_s(eos) == 0.0 && return
        dissipative_step_changerep_grad!(particles, params, cl, dt, eos)
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
function dissipative_step_energy!(particles::Vector{Particle}, params::SimParams,
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
                acc  += field(particles[b]) * wendland_c2(r_mag, h, params.kernel)
            end
        end
        S[ix, iy] = acc
    end
    return S, xs, ys
end

# The viscous step in the particle representation (params.dissipation = :changerep).
include(joinpath(@__DIR__, "changerep.jl"))
