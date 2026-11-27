# Shared helpers for shear-wave test scripts.
# Include after sph.jl and celllist.jl.

# (The square lattice, make_uniform_particles, was removed on 2026-11-27: docs/focus_plan.tex, Q8; tag pre-prune.)

# Hexagonal close-packed lattice.  N_x columns (spacing a), N_y rows (spacing a√3/2).
# Odd rows are offset by a/2 in x.  Geometry (N, vol, a) is already set by SimParams
# constructor; this function only places the particles.
function make_hex_particles!(params::SimParams, s_density; jitter=0.0)
    Lx  = params.L[1];  Ly = params.L[2]
    a_x = params.a
    N_x = round(Int, Lx / a_x)
    N_y = round(Int, params.N / N_x)
    a_y = Ly / N_y
    rng = Xoshiro(params.seed)        # the jitter's own random numbers (docs/focus_plan.tex Q10; the global RNG until then)
    particles = Particle[]
    for iy in 0:N_y-1
        x_off = (iy % 2 == 0) ? 0.0 : 0.5 * a_x
        for ix in 0:N_x-1
            x0 = SVector(mod(ix * a_x + x_off, Lx), (iy + 0.5) * a_y)
            if jitter > 0.0
                δ  = jitter * a_x * (SVector(rand(rng), rand(rng)) .- 0.5)
                x0 = SVector(mod(x0[1] + δ[1], Lx), mod(x0[2] + δ[2], Ly))
            end
            push!(particles, Particle(x0, s_density * params.vol))
        end
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

# Full thermodynamic update: smooth → EOS → velocities.
function thermalize!(particles, params, cl, eos)
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)
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
                w      = kernel_W(r, h, params.kernel)
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
                  L=nothing, Lx=1.0, Ly=1.0, jitter=0.0, lattice=:hex, phys_dim=default_phys_dim(), seed=42)
    if L !== nothing
        Lx = Float64(L); Ly = Float64(L)
    end
    params    = SimParams(Lx, Ly, h, ha_ratio; lattice, phys_dim, seed)
    particles = make_hex_particles!(params, 1.0; jitter)
    set_ic!(particles, params; S_func, M_func)
    cl = CellList2D(params.L, params.h, params.N)
    rebuild_cell_list!(cl, particles)
    SSCHEME && sscheme_labels!(particles, params, cl, eos)     # pa.s = S_func V is the target entropy: the label at that state
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
    eos    = ConstantCs2EOS(cs2v, T_scale, η_by_s, ζ_by_s)
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
                          jitter=0.0, lattice=:hex, phys_dim=default_phys_dim())
    k   = 4π / L
    eos = ConstantCs2EOS(cs2v, T_scale, η_by_s, ζ_by_s)
    params, particles, cl, eos = setup_ic(h, ha_ratio, eos,
        (x, y) -> s0,
        (x, y) -> SVector(0.0, M_amp * sin(k * x));
        L, jitter, lattice, phys_dim)
    return params, particles, cl, eos, k, M_amp
end
