# Shared helpers for shear-wave test scripts.
# Include after sph.jl and celllist.jl.

function make_cl(params::SimParams)
    CellList2D(params.L, params.h, params.N)
end

# Regular NxN grid of particles with uniform s, zero M.
# s_density is the entropy density; per-particle entropy = s_density * params.vol.
# jitter ∈ [0,1]: each particle is displaced uniformly in [−jitter·a/2, jitter·a/2]² before
# wrapping into [0,L).  jitter=0 (default) gives the exact lattice.
function make_uniform_particles(params::SimParams, s_density; jitter=0.0)
    Lx = params.L[1];  Ly = params.L[2]
    Nx = round(Int, Lx / params.a)
    Ny = round(Int, Ly / params.a)
    vol    = params.vol
    particles = Particle[]
    jitter > 0.0 && Random.seed!(RNG, params.seed)
    for iy in 0:Ny-1, ix in 0:Nx-1
        x0 = SVector((ix + 0.5) * params.a, (iy + 0.5) * params.a)
        if jitter > 0.0
            δ  = jitter * params.a * (SVector(rand(RNG), rand(RNG)) .- 0.5)
            x0 = SVector(mod(x0[1] + δ[1], Lx), mod(x0[2] + δ[2], Ly))
        end
        push!(particles, Particle(x0, s_density * vol))
    end
    return particles
end

# Hexagonal close-packed lattice.  N_x columns (spacing a), N_y rows (spacing a√3/2).
# Odd rows are offset by a/2 in x.  Geometry (N, vol, a) is already set by SimParams
# constructor; this function only places the particles.
function make_hex_particles!(params::SimParams, s_density; jitter=0.0)
    Lx  = params.L[1];  Ly = params.L[2]
    a_x = params.a
    N_x = round(Int, Lx / a_x)
    N_y = round(Int, params.N / N_x)
    a_y = Ly / N_y
    jitter > 0.0 && Random.seed!(RNG, params.seed)
    particles = Particle[]
    for iy in 0:N_y-1
        x_off = (iy % 2 == 0) ? 0.0 : 0.5 * a_x
        for ix in 0:N_x-1
            x0 = SVector(mod(ix * a_x + x_off, Lx), (iy + 0.5) * a_y)
            if jitter > 0.0
                δ  = jitter * a_x * (SVector(rand(RNG), rand(RNG)) .- 0.5)
                x0 = SVector(mod(x0[1] + δ[1], Lx), mod(x0[2] + δ[2], Ly))
            end
            push!(particles, Particle(x0, s_density * params.vol))
        end
    end
    return particles
end

# Single row of N particles along x, all at y = a/2 (centre of the single cell row).
# vol = a (1D), so per-particle entropy = s_density * a.
function make_1d_particles!(params::SimParams, s_density; jitter=0.0)
    a = params.a
    N = params.N
    jitter > 0.0 && Random.seed!(RNG, params.seed)
    particles = Particle[]
    for ix in 0:N-1
        x0 = SVector((ix + 0.5) * a, 0.5 * a)
        if jitter > 0.0
            δx = jitter * a * (rand(RNG) - 0.5)
            x0 = SVector(mod(x0[1] + δx, params.L[1]), x0[2])
        end
        push!(particles, Particle(x0, s_density * params.vol))
    end
    return particles
end

# Set initial conditions from continuous density fields.
# S_func(x,y) → entropy density S(x); M_func(x,y) → momentum density (SVector{2}).
# Per-particle quantities are S_func(x,y)*vol and M_func(x,y)*vol respectively.
# Pass nothing to leave that field unchanged.
function set_ic!(particles, params::SimParams; S_func=nothing, M_func=nothing)
    vol = params.vol
    for pa in particles
        S_func !== nothing && (pa.s = S_func(pa.x[1], pa.x[2]) * vol)
        M_func !== nothing && (pa.m = M_func(pa.x[1], pa.x[2]) * vol)
    end
end

# Set a sinusoidal y-momentum density: M_y(x) = M_amp * sin(k * x[1]).
function set_shear_wave!(particles, params::SimParams, k, M_amp)
    set_ic!(particles, params;
            M_func = (x, y) -> SVector(0.0, M_amp * sin(k * x)))
end

# Full thermodynamic update: smooth → EOS → velocities.
function thermalize!(particles, params, cl, eos)
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)
end

# 1D line plot of field(p) vs p.x[1] for a 1D simulation.
function plot_line(particles, params;
                   field=p->p.s, ylabel="field", title="1D profile", ax=nothing)
    xs = [pa.x[1] for pa in particles]
    fs = [field(pa) for pa in particles]
    standalone = ax === nothing
    if standalone
        _, ax = subplots(1, 1, figsize=(8, 4))
    end
    ax.plot(xs, fs, "o", markersize=4)
    ax.set_xlim(0, params.L[1])
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    standalone && tight_layout()
    return ax
end

# Dispatch: 1D → plot_line, 2D → plot_particles.
plot_results(particles, params; kw...) =
    params.kernel isa WendlandC2_1D ? plot_line(particles, params; kw...) :
                                      plot_particles(particles, params; kw...)

# 2D scatter of particle positions, colored by field(p) (default: p.s).
# Three particles chosen with a local RNG get a circle of radius 2h.
# Pass an existing PyPlot axis via `ax`; omit it for a standalone figure.
function plot_particles(particles, params;
                        field=p->p.s, clabel="field", title="Particles", ax=nothing)
    L = params.L
    h = params.h

    xs = [pa.x[1]   for pa in particles]
    ys = [pa.x[2]   for pa in particles]
    cs = [field(pa) for pa in particles]

    standalone = ax === nothing
    if standalone
        _, ax = subplots(1, 1, figsize=(6, 6))
    end

    sc = ax.scatter(xs, ys, c=cs, s=6, cmap="viridis")
    colorbar(sc, ax=ax, label=clabel)

    # Pick 3 particles with a local RNG so we don't disturb global state.
    rng  = MersenneTwister(42)
    idxs = randperm(rng, length(particles))[1:3]
    for (color, idx) in zip(("C1", "C2", "C3"), idxs)
        cx, cy = particles[idx].x
        θ = range(0.0, 2π, length=200)
        ax.plot(cx .+ 2h .* cos.(θ), cy .+ 2h .* sin.(θ), color=color, linewidth=1.2)
        ax.plot([cx], [cy], "+", color=color, markersize=7, markeredgewidth=1.5)
    end

    ax.set_xlim(0, params.L[1])
    ax.set_ylim(0, params.L[2])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    standalone && tight_layout()
    return ax
end

# SPH kernel estimate of field(p) on a 1D slice through the box.
# Formula: f̄(r) = Σ_a field(pa)·pa.s·W(|r - rₐ|, h) / Σ_a pa.s·W(|r - rₐ|, h)
# dim=:x (default): vary x at fixed y = yslice (default L/2).
# dim=:y          : vary y at fixed x = xslice (default L/2).
# Returns (xs, fs) — caller is responsible for plotting, labels, and title.
function plot_slice(particles, params, cl;
                    field=p->p.s, dim=:x,
                    xslice=params.L[1]/2, yslice=params.L[2]/2, Ng=400)
    Lx = params.L[1];  Ly = params.L[2]
    h  = params.h
    ncx = cl.ncx;  ncy = cl.ncy

    xs = LinRange(0.0, dim == :x ? Lx : Ly, Ng)
    fs = zeros(Ng)

    for (i, t) in enumerate(xs)
        xp  = dim == :x ? t      : xslice
        yp  = dim == :x ? yslice : t
        ix, iy = cell_index(SVector(xp, yp), cl)
        num   = 0.0
        denom = 0.0
        for ddx in -1:1, ddy in neighbor_range(ncy)
            jx = mod(ix + ddx - 1, ncx) + 1
            jy = mod(iy + ddy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                pa   = particles[b]
                rvec = minimum_image(SVector(xp, yp) - pa.x, params.L)
                r    = norm(rvec)
                r >= 2h && continue
                w      = wendland_c2(r, h, params.kernel)
                num   += field(pa) * pa.s * w
                denom += pa.s * w
            end
        end
        fs[i] = denom > 0.0 ? num / denom : 0.0
    end

    return xs, fs
end

# General setup: build params, particles, cl from smooth density fields.
# eos must be constructed by the caller (allows S_func/M_func to close over eos quantities).
# S_func(x,y) → entropy density S(x); M_func(x,y) → momentum density (SVector{2}).
# Lattice kwargs: jitter, lattice.  Returns (params, particles, cl, eos).
function setup_ic(h, ha_ratio, eos, S_func, M_func;
                  L=nothing, Lx=1.0, Ly=1.0, jitter=0.0, lattice=:hex, sim_dim=2, d=3)
    if L !== nothing
        Lx = Float64(L); Ly = Float64(L)
    end
    params    = SimParams(Lx, Ly, h, ha_ratio; lattice, sim_dim, d)
    particles = if sim_dim == 1
        make_1d_particles!(params, 1.0; jitter)
    elseif lattice == :hex
        make_hex_particles!(params, 1.0; jitter)
    elseif lattice == :square
        make_uniform_particles(params, 1.0; jitter)
    else
        error("Unknown lattice type: $lattice")
    end
    set_ic!(particles, params; S_func, M_func)
    cl = CellList2D(params.L, params.h, params.N)
    rebuild_cell_list!(cl, particles)
    thermalize!(particles, params, cl, eos)
    return params, particles, cl, eos
end

# Gaussian energy density: e(x,y) = e0 + A * exp(-Δx²/2σx² - Δy²/2σy²), zero momentum.
# Entropy density is derived via srel(eos, e(x,y)).
# Center defaults to box center (L/2, L/2); distances use periodic minimum image.
# Pass σx=Inf or σy=Inf to suppress variation in that direction (factor becomes 1).
# Returns (params, particles, cl, eos).
function setup_gaussian(h, ha_ratio;
                        L=1.0, e0=1.0, A=1.0, σx=Inf, σy=Inf,
                        x0=L/2, y0=L/2,
                        cs2v=1.0/3.0, η_by_s=0.0, ζ_by_s=0.0, T_scale=T_scale_gluon,
                        jitter=0.0, lattice=:hex)
    eos    = ConformalEOS(cs2v, T_scale, η_by_s, ζ_by_s)
    S_func = (x, y) -> begin
        dx = mod(x - x0 + L/2, L) - L/2
        dy = mod(y - y0 + L/2, L) - L/2
        fx = isfinite(σx) ? exp(-dx^2 / (2*σx^2)) : 1.0
        fy = isfinite(σy) ? exp(-dy^2 / (2*σy^2)) : 1.0
        srel(eos, e0 + A * fx * fy)
    end
    params, particles, cl, eos = setup_ic(h, ha_ratio, eos, S_func,
                                          (x, y) -> SVector(0.0, 0.0); L, jitter, lattice)
    return params, particles, cl, eos
end

# Shear-wave IC: uniform entropy s0, sinusoidal y-momentum M_amp*sin(k*x), k=4π/L.
# Returns (params, particles, cl, eos, k, M_amp).
function setup_shear_wave(h, ha_ratio;
                          L=1.0, M_amp=0.01, s0=1.0,
                          cs2v=1.0/3.0, η_by_s=0.0, ζ_by_s=0.0, T_scale=T_scale_gluon,
                          jitter=0.0, lattice=:hex, d=3)
    k   = 4π / L
    eos = ConformalEOS(cs2v, T_scale, η_by_s, ζ_by_s)
    params, particles, cl, eos = setup_ic(h, ha_ratio, eos,
        (x, y) -> s0,
        (x, y) -> SVector(0.0, M_amp * sin(k * x));
        L, jitter, lattice, d)
    return params, particles, cl, eos, k, M_amp
end
