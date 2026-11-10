# measure.jl — equilibrium initial states, recording of Fourier modes, and correlators
# (docs/plan.tex, "Measurements in thermalized ensembles").  Include after stochrep.jl;
# needs FFTW, BinningAnalysis and HDF5.
using FFTW, BinningAnalysis, HDF5, Serialization

# ── Equilibrium initial state ────────────────────────────────────────────────
# Single-particle covariance of δm at fixed e_a: [−∂²S_a/∂m∂m]⁻¹ (numerical Hessian).
function momentum_covariance(e, m0, V, eos)
    f(x, y) = particle_entropy(e, m0 + SVector(x, y), V, eos)
    T = eos_from_energy(e / V, m0 / V, eos)[3]
    h = 0.05 * sqrt(e * T)                  # a twentieth of the thermal momentum, ≈ √(V w T)
    Hxx = (f(h, 0) - 2f(0, 0) + f(-h, 0)) / h^2
    Hyy = (f(0, h) - 2f(0, 0) + f(0, -h)) / h^2
    Hxy = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4h^2)
    return inv(-SMatrix{2,2,Float64}(Hxx, Hxy, Hxy, Hyy))
end

# Draw thermal momentum fluctuations at fixed particle energies e_a: δm_a from the
# Gaussian with the covariance above, the total momentum kept, and s_a = S_a(e_a, m_a).
# The positions and the mean flow are unchanged.  Returns the mean covariance.
function thermal_init!(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                       eos::AbstractEOS; rng::AbstractRNG = Random.default_rng())
    fz = stochrep_frozen(particles, params, cl, eos)
    N  = length(particles)
    Cs = [momentum_covariance(fz.e[a], particles[a].m, fz.V[a], eos) for a in 1:N]
    δ  = [cholesky(Symmetric(Matrix(Cs[a]))).L * SVector(randn(rng), randn(rng)) for a in 1:N]
    δ  = [SVector{2}(d) for d in δ]
    mean_δ = sum(δ) / N
    for (a, pa) in enumerate(particles)
        pa.m += δ[a] - mean_δ
        pa.s  = particle_entropy(fz.e[a], pa.m, fz.V[a], eos)
    end
    thermalize!(particles, params, cl, eos)
    return sum(Cs) / N
end

# ── Equilibrium start with energy fluctuations (docs/eqstart.tex) ────────────
# At fixed positions and particle volumes V_a, the entropy Σ_a S_a(e_a, m_a) is
# maximized at fixed totals E = Σ e_a, M = Σ m_a by a common β^μ:
#     ∂S_a/∂e_a = β⁰ = 1/T,   ∂S_a/∂m_a = −β = −v/T     (T = T_rel/γ, the bare temperature).
# Each particle is then distributed as  p_a(e, m) ∝ exp[S_a(e, m) − β⁰ e + β·m]; the
# draws are made exact in E and M afterwards.

# Hessian of S_a(e, m) in (e, m_x, m_y) by central differences with step δ.
function entropy_hessian(e, m, V, eos, δ)
    f(x) = particle_entropy(e + x[1], m + SVector(x[2], x[3]), V, eos)
    H = zeros(3, 3);  u(i) = SVector{3,Float64}(i == 1, i == 2, i == 3) * δ
    for i in 1:3, j in i:3
        H[i, j] = H[j, i] = i == j ?
            (f(u(i)) - 2f(zero(SVector{3,Float64})) + f(-u(i))) / δ^2 :
            (f(u(i) + u(j)) - f(u(i) - u(j)) - f(-u(i) + u(j)) + f(-u(i) - u(j))) / (4δ^2)
    end
    return H
end

# One exact draw (δe, δm) around the most probable state (ē, m̄) of a particle of volume V:
# an independence Metropolis chain of nmh steps whose proposal is the Gaussian with the
# covariance widen·C, C = (−∂²S_a)⁻¹.  Widened so that its tails cover the skewed target
# (exact as nmh → ∞).  L = cholesky(C).L, Hn = −∂²S_a.  Returns (x, accepted moves).
function draw_particle(ē, m̄, V, β0, β, eos, L, Hn, rng; nmh = 30, widen = 2.0)
    logp(x) = particle_entropy(ē + x[1], m̄ + SVector(x[2], x[3]), V, eos) - β0 * x[1] + dot(β, SVector(x[2], x[3]))
    logq(x) = -0.5 * dot(x, Hn * x) / widen
    Lw = sqrt(widen) * L
    x = Lw * SVector(randn(rng), randn(rng), randn(rng));  wx = logp(x) - logq(x);  nacc = 0
    for _ in 1:nmh
        y = Lw * SVector(randn(rng), randn(rng), randn(rng));  wy = logp(y) - logq(y)
        if log(rand(rng)) < wy - wx
            x, wx = y, wy;  nacc += 1
        end
    end
    return x, nacc
end

# Most probable state of a particle at given multipliers: T = 1/β⁰ (bare), v = β/β⁰;
# returns the energy and momentum densities (ε, M) of that state.
function state_from_beta(β0, β::SVector{2,Float64}, eos::AbstractEOS)
    T = 1 / β0;  v = β / β0;  γ = 1 / sqrt(1 - dot(v, v));  Tr = γ * T
    lo, hi = 0.0, 1.0
    while Trel(eos, hi) < Tr;  hi *= 2;  end
    for _ in 1:200                                          # T_rel(e) is monotonic
        mid = (lo + hi) / 2;  Trel(eos, mid) < Tr ? (lo = mid) : (hi = mid)
        hi - lo <= 1e-15 * hi && break
    end
    e = (lo + hi) / 2;  w = e + pressure(eos, e)
    return (ε = w * γ^2 - pressure(eos, e), M = w * γ^2 * v)
end

# Exact single-particle averages at (β⁰, β) for volume V by quadrature of
# exp[S_a(e, m) − β⁰ e + β·m] on a grid around the most probable state (skewed range in e):
# returns the most probable (ē, m̄), the mean ⟨(e, m)⟩, the covariance, and the Gaussian
# pieces (L, Hn) for the sampler.
function particle_moments(V, β0, β::SVector{2,Float64}, eos::AbstractEOS; nq::Int = 41)
    st = state_from_beta(β0, β, eos);  ē = V * st.ε;  m̄ = V * st.M
    H  = entropy_hessian(ē, m̄, V, eos, 0.02 * sqrt(ē / β0));  Hn = SMatrix{3,3,Float64}(-H)
    C  = inv(Matrix(Hn));  L = SMatrix{3,3,Float64}(cholesky(Symmetric(C)).L);  sd = sqrt.(diag(C))
    x1 = range(-6sd[1], 10sd[1], length = nq);  x2 = range(-9sd[2], 9sd[2], length = nq)
    x3 = range(-9sd[3], 9sd[3], length = nq)
    lw = [particle_entropy(ē + a, m̄ + SVector(b, c), V, eos) - β0 * a + β[1] * b + β[2] * c
          for a in x1, b in x2, c in x3]
    w  = exp.(lw .- maximum(filter(isfinite, lw)));  w[.!isfinite.(w)] .= 0;  Z = sum(w)
    X  = [SVector(a, b, c) for a in x1, b in x2, c in x3]
    μ  = sum(w .* X) / Z;  Cq = sum(w .* [(x - μ) * (x - μ)' for x in X]) / Z
    return (ē = ē, m̄ = m̄, mean = SVector(ē + μ[1], m̄[1] + μ[2], m̄[2] + μ[3]),
            cov = SMatrix{3,3,Float64}(Cq), L = L, Hn = Hn)
end

# Single-particle moments for every volume in V, as a function V_a -> moments.  Particles of
# equal volume share them (a lattice: one quadrature).  When there are many distinct
# volumes (jittered or random positions), the quadrature is done at nnode volumes spanning
# [min V, max V] and, per particle:
#   ē = V ε̄, m̄ = V M̄ (extensive, exact); the offsets ⟨x⟩ − x̄ and Cov(x), which are smooth
#   in V (≈ const and ∝ V), are interpolated linearly; the sampler's Gaussian is rescaled
#   from the nearest node (Hn ∝ 1/V, L ∝ √V).  The draws stay exact (the Metropolis target
#   is the particle's own S_a); the interpolation only enters the averages used for β.
function moments_table(V::Vector{Float64}, β0, β::SVector{2,Float64}, eos::AbstractEOS;
                       nq::Int = 41, nnode::Int = 9, maxgroups::Int = 32)
    vs = unique(V)
    if length(vs) <= maxgroups
        d = Dict(v => particle_moments(v, β0, β, eos; nq) for v in vs)
        return v -> d[v]
    end
    st = state_from_beta(β0, β, eos)
    nodes = collect(range(minimum(V), maximum(V), length = nnode))
    P = [particle_moments(v, β0, β, eos; nq) for v in nodes]
    off = [p.mean - SVector(p.ē, p.m̄[1], p.m̄[2]) for p in P]
    return function (v)
        j = clamp(searchsortedlast(nodes, v), 1, nnode - 1);  f = (v - nodes[j]) / (nodes[j+1] - nodes[j])
        n = f < 0.5 ? j : j + 1;  r = v / nodes[n]
        ē = v * st.ε;  m̄ = v * st.M
        (ē = ē, m̄ = m̄, mean = SVector(ē, m̄[1], m̄[2]) + (1 - f) * off[j] + f * off[j+1],
         cov = (1 - f) * P[j].cov + f * P[j+1].cov, L = sqrt(r) * P[n].L, Hn = P[n].Hn / r)
    end
end

# The global multipliers whose *average* totals are (E, M): Σ_a ⟨e_a⟩ = E, Σ_a ⟨m_a⟩ = M.
# Newton in θ = (−β⁰, β), conjugate to x = (e, m), with ∂⟨x⟩/∂θ = Cov(x) (exact for this
# exponential family).  Starts from the most probable state at the mean densities.
# Returns (β⁰, β, mom) with mom(V_a) the moments of a particle of volume V_a.
function beta_from_averages(V::Vector{Float64}, E::Float64, M::SVector{2,Float64},
                            eos::AbstractEOS; nq::Int = 41, iters::Int = 8)
    Vtot = sum(V);  _, v0, T0 = eos_from_energy(E / Vtot, M / Vtot, eos)
    θ = SVector(-1 / T0, v0[1] / T0, v0[2] / T0);  target = SVector(E, M[1], M[2])
    mom = nothing
    for it in 1:iters
        β0 = -θ[1];  β = SVector(θ[2], θ[3])
        mom = moments_table(V, β0, β, eos; nq)
        ms = mom.(V);  X = sum(p.mean for p in ms);  Ct = sum(p.cov for p in ms)
        r = target - X
        abs(r[1]) <= 1e-12 * abs(E) && norm(r[2:3]) <= 1e-12 * abs(E) && break
        θ = θ + Ct \ r
    end
    return -θ[1], SVector(θ[2], θ[3]), mom
end

# Draw (e_a, m_a) for all particles at the multipliers (β⁰, β) (moments mom(V_a) from
# beta_from_averages): independent draws from the canonical ensemble, so the totals
# fluctuate about their averages (E, M).  With exact_totals = true the difference between
# the drawn totals and (E, M) is then removed uniformly, i.e. in proportion to the
# volumes (an equal share per particle on an equal-volume lattice), so that a run starts
# at exactly the target energy and momentum.
function draw_configuration(V::Vector{Float64}, β0, β, mom, eos, rng; nmh::Int = 30,
                            exact_totals::Bool = false, E = nothing, M = nothing)
    N = length(V);  X = Vector{SVector{3,Float64}}(undef, N);  nacc = 0
    for a in 1:N
        p = mom(V[a])
        x, na = draw_particle(p.ē, p.m̄, V[a], β0, β, eos, p.L, p.Hn, rng; nmh)
        X[a] = SVector(p.ē + x[1], p.m̄[1] + x[2], p.m̄[2] + x[3]);  nacc += na
    end
    if exact_totals
        d = sum(X) - SVector(E, M[1], M[2]);  Vtot = sum(V)
        X .= [X[a] - (V[a] / Vtot) * d for a in 1:N]
    end
    return X, nacc / (N * nmh)
end

# Equilibrium start: draw the particle energies and momenta of a 2D state at the global
# multipliers β^μ whose averages are the totals (E, M) (default: those of the current
# state), keeping the positions and the volumes V_a = s_a/S_a.  Sets m_a and
# s_a = S_a(e_a, m_a), then re-smooths.  exact_totals as in draw_configuration: true
# starts an SPH run at exactly (E, M), with no thermalization step needed.
function equilibrium_init!(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                           eos::AbstractEOS; rng::AbstractRNG = Random.default_rng(),
                           E = nothing, M = nothing, nmh::Int = 30, exact_totals::Bool = false)
    thermalize!(particles, params, cl, eos)
    V = [pa.s / pa.S_smooth for pa in particles]
    Etot = E === nothing ? sum(particle_energy(pa, eos) for pa in particles) : Float64(E)
    Mtot = M === nothing ? sum(pa.m for pa in particles) : SVector{2,Float64}(M)
    β0, β, mom = beta_from_averages(V, Etot, Mtot, eos)
    X, acc = draw_configuration(V, β0, β, mom, eos, rng; nmh, exact_totals, E = Etot, M = Mtot)
    e = [x[1] for x in X]
    for (a, pa) in enumerate(particles)
        pa.m = SVector(X[a][2], X[a][3])
        pa.s = particle_entropy(e[a], pa.m, V[a], eos)
        isfinite(pa.s) || error("equilibrium_init!: unphysical draw for particle $a")
    end
    thermalize!(particles, params, cl, eos)
    return (β0 = β0, β = β, T = 1 / β0, v = β / β0, e = e, V = V, acc = acc, E = Etot, M = Mtot)
end

# Initial state of a stochastic run (drivers: SR_START):
#   :cold     the state as set up (uniform, at rest): no fluctuations; the run thermalizes
#   :eq       equilibrium_init! with exact_totals = true: energy and momentum fluctuations
#             at the global β^μ whose averages are the current totals, totals shifted to
#             them exactly, so no thermalization step is needed (docs/eqstart.tex)
#   :thermal  thermal_init!: momenta only, at fixed particle energies (campaign `first`)
function initial_state!(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                        eos::AbstractEOS, start::Symbol; rng::AbstractRNG = Random.default_rng())
    # returns a description of the start, for the run record (src/runinfo.jl)
    if start === :cold
        thermalize!(particles, params, cl, eos)
        return (type = "cold", what = "the state as set up (uniform, at rest), no fluctuations; the run thermalizes")
    elseif start === :eq
        r = equilibrium_init!(particles, params, cl, eos; rng, exact_totals = true)
        acc = r.acc isa AbstractArray ? sum(r.acc) / length(r.acc) : r.acc
        return (type = "eq", what = "equilibrium draw of (e_a, m_a) at the global beta^mu of the totals, totals shifted to them exactly",
                T0 = r.T, v0 = collect(r.v), E_total = r.E, M_total = collect(r.M), exact_totals = true, nmh = 30,
                mh_acceptance = acc)
    elseif start === :thermal
        thermal_init!(particles, params, cl, eos; rng)
        return (type = "thermal", what = "momenta drawn at fixed particle energies (momentum covariance)")
    else
        error("unknown start $start (use cold, eq or thermal)")
    end
end

# ── Recording ────────────────────────────────────────────────────────────────
# Wave vectors of the periodic box with |k| h ≤ khmax, one of each ±k pair.
function box_wavevectors(params::SimParams, khmax::Float64)
    Lx, Ly = params.L;  h = params.h
    nxm = floor(Int, khmax / h * Lx / 2π);  nym = floor(Int, khmax / h * Ly / 2π)
    ks = SVector{2,Float64}[]
    for nx in 0:nxm, ny in -nym:nym
        (nx == 0 && ny <= 0) && continue
        k = SVector(2π * nx / Lx, 2π * ny / Ly)
        norm(k) * h <= khmax && push!(ks, k)
    end
    return ks
end

# Time series of the Fourier sums
#   M_k = Σ_a m_a e^{−ik·x_a},  S_k = Σ_a s_a e^{−ik·x_a},  E_k = Σ_a e_a e^{−ik·x_a}
# (E_k optional, it needs the particle energies).
mutable struct Recorder
    ks   :: Vector{SVector{2,Float64}}
    t    :: Vector{Float64}
    ML   :: Vector{Vector{ComplexF64}}     # k̂·M_k
    MT   :: Vector{Vector{ComplexF64}}     # k̂ × M_k
    S    :: Vector{Vector{ComplexF64}}
    E    :: Vector{Vector{ComplexF64}}
    energies :: Bool
    # displacements from the initial sites x0 (minimum image in the box L), if x0 is given:
    # u_k = Σ_a (x_a − x0_a) e^{−i k·x0_a}, split as for M.  Then M_k ≈ wV u̇_k.
    UL   :: Vector{Vector{ComplexF64}}
    UT   :: Vector{Vector{ComplexF64}}
    x0   :: Vector{SVector{2,Float64}}
    L    :: SVector{2,Float64}
end
Recorder(ks; energies = false, x0 = SVector{2,Float64}[], L = SVector(0.0, 0.0)) =
    Recorder(ks, Float64[], [], [], [], [], energies, [], [], x0, L)
Recorder(ks, t, ML, MT, S, E, energies) = Recorder(ks, t, ML, MT, S, E, energies, [], [], SVector{2,Float64}[], SVector(0.0, 0.0))

function record!(rec::Recorder, particles::Vector{Particle}, t::Float64, eos::AbstractEOS;
                 u::Union{Nothing,Vector{SVector{2,Float64}}} = nothing)
    nk = length(rec.ks)
    ML = zeros(ComplexF64, nk);  MT = zeros(ComplexF64, nk);  S = zeros(ComplexF64, nk)
    E  = zeros(ComplexF64, rec.energies ? nk : 0)
    ea = rec.energies ? [particle_energy(pa, eos) for pa in particles] : Float64[]
    disp = !isempty(rec.x0)
    if disp
        UL = zeros(ComplexF64, nk);  UT = zeros(ComplexF64, nk)
        # displacements: unwrapped ones if given (ParticleRecorder), else by minimum image
        u === nothing && (u = [minimum_image(pa.x - rec.x0[a], rec.L) for (a, pa) in enumerate(particles)])
        Threads.@threads for i in 1:nk
            k = rec.ks[i];  kh = k / norm(k);  U = zero(SVector{2,ComplexF64})
            for a in eachindex(u);  U += u[a] * cis(-dot(k, rec.x0[a]));  end
            UL[i] = kh[1] * U[1] + kh[2] * U[2];  UT[i] = kh[1] * U[2] - kh[2] * U[1]
        end
        push!(rec.UL, UL);  push!(rec.UT, UT)
    end
    Threads.@threads for i in 1:nk
        k = rec.ks[i];  kh = k / norm(k)
        M = zero(SVector{2,ComplexF64});  s = 0.0im;  en = 0.0im
        for (a, pa) in enumerate(particles)
            ph = cis(-dot(k, pa.x))
            M += pa.m * ph;  s += pa.s * ph
            rec.energies && (en += ea[a] * ph)
        end
        ML[i] = kh[1] * M[1] + kh[2] * M[2];  MT[i] = kh[1] * M[2] - kh[2] * M[1];  S[i] = s
        rec.energies && (E[i] = en)
    end
    push!(rec.t, t);  push!(rec.ML, ML);  push!(rec.MT, MT);  push!(rec.S, S)
    rec.energies && push!(rec.E, E)
    return rec
end

# ── Particle motion: pair separation and the lattice's Fourier components ───────────
# Unwrapped displacements u_a(t) = x_a(t) − x_a(0), accumulated record by record by minimum
# image (a particle moves far less than L/2 between records), with the mean drift removed.
#   D2[c](t) = ⟨|u_i − u_j|²⟩ over the pairs (i, j) at distance rs[c]·a at t = 0
#              (saturates in a solid; grows as 4 D_rel t in a 2D fluid),
#   MSD(t)   = ⟨|u_a|²⟩,
#   B(t)     = ⟨|Σ_a e^{−i b·x_a}|²⟩_b / N²   (Bragg peak: ≈ 1 crystal, ≈ 1/N melted),
#   Fs(t)    = ⟨Re (1/N) Σ_a e^{−i b·u_a}⟩_b   (self part: plateau in a solid, → 0 in a fluid),
# over the shortest reciprocal-lattice vectors b of the starting arrangement (box wave vectors
# with |k| ≈ 4π/(√3 a) and |Σ_a e^{−ik·x_a}|/N > 0.5 at t = 0; none for a random start).
mutable struct ParticleRecorder
    L     :: SVector{2,Float64}
    x0    :: Vector{SVector{2,Float64}}
    xu    :: Vector{SVector{2,Float64}}       # unwrapped positions
    xprev :: Vector{SVector{2,Float64}}
    rs    :: Vector{Float64}                  # pair distances / a
    pairs :: Vector{Vector{Tuple{Int,Int}}}
    bs    :: Vector{SVector{2,Float64}}
    t     :: Vector{Float64}
    D2    :: Vector{Vector{Float64}}
    MSD   :: Vector{Float64}
    B     :: Vector{Float64}
    Fs    :: Vector{Float64}
end

function ParticleRecorder(particles::Vector{Particle}, params; rs = (1.0, 2.0, 4.0, 8.0))
    x0 = [pa.x for pa in particles];  N = length(x0);  a = params.a;  L = params.L
    pairs = [Tuple{Int,Int}[] for _ in rs]
    for i in 1:N, j in i+1:N
        d = norm(minimum_image(x0[j] - x0[i], L)) / a
        for (c, r) in enumerate(rs);  abs(d - r) < 0.05 && push!(pairs[c], (i, j));  end
    end
    bmag = 4π / (sqrt(3) * a);  bs = SVector{2,Float64}[]
    nxm = ceil(Int, 1.2bmag * L[1] / 2π);  nym = ceil(Int, 1.2bmag * L[2] / 2π)
    for nx in -nxm:nxm, ny in -nym:nym
        k = SVector(2π * nx / L[1], 2π * ny / L[2])
        abs(norm(k) / bmag - 1) < 0.1 || continue
        abs(sum(cis(-dot(k, x)) for x in x0)) / N > 0.5 && push!(bs, k)
    end
    ParticleRecorder(L, x0, copy(x0), copy(x0), collect(Float64, rs), pairs, bs,
                     Float64[], Vector{Float64}[], Float64[], Float64[], Float64[])
end

# current displacements (unwrapped, mean drift removed); call update! first
function displacements(pr::ParticleRecorder)
    u = pr.xu .- pr.x0;  ū = sum(u) / length(u)
    return [v - ū for v in u]
end

function record!(pr::ParticleRecorder, particles::Vector{Particle}, t::Float64)
    for (a, pa) in enumerate(particles)
        pr.xu[a] += minimum_image(pa.x - pr.xprev[a], pr.L);  pr.xprev[a] = pa.x
    end
    u = displacements(pr);  N = length(u)
    push!(pr.t, t)
    push!(pr.D2, [isempty(P) ? NaN : sum(sum(abs2, u[i] - u[j]) for (i, j) in P) / length(P) for P in pr.pairs])
    push!(pr.MSD, sum(sum(abs2, v) for v in u) / N)
    if isempty(pr.bs)
        push!(pr.B, NaN);  push!(pr.Fs, NaN)
    else
        push!(pr.B, sum(abs2(sum(cis(-dot(b, pa.x)) for pa in particles)) for b in pr.bs) / (length(pr.bs) * N^2))
        push!(pr.Fs, sum(real(sum(cis(-dot(b, v)) for v in u)) for b in pr.bs) / (length(pr.bs) * N))
    end
    return u
end

# ── Correlators ──────────────────────────────────────────────────────────────
# C(τ) = ⟨x(t+τ) x*(t)⟩_t for one complex time series, lags 0..maxlag (records), by FFT
# (zero-padded, so no wrap-around); real part (the imaginary part averages out).
function autocorr(x::Vector{ComplexF64}, maxlag::Int)
    n = length(x);  nf = nextpow(2, 2n)
    X = fft(vcat(x, zeros(ComplexF64, nf - n)))
    c = ifft(abs2.(X))                                   # c[τ+1] = Σ_t x(t+τ) x*(t)
    return [real(c[τ+1]) / (n - τ) for τ in 0:maxlag]
end

# Equal-time average of |x(t)|² with its error from logarithmic binning (the samples are
# correlated): returns (mean, standard error, integrated autocorrelation time in records).
function binned_mean(y::AbstractVector{Float64})
    B = LogBinner(y)
    return mean(B), std_error(B), tau(B)
end

# Correlators of one recorder, grouped in shells of equal |k| (all k of the same
# length averaged).  field ∈ (:MT, :ML, :S).  Returns (kshell, lags in time, C[shell, lag]).
function shell_correlators(rec::Recorder, field::Symbol; t0 = 0.0, maxlag_t = 5.0)
    i0 = findfirst(>=(t0), rec.t);  dtr = rec.t[2] - rec.t[1]
    maxlag = min(round(Int, maxlag_t / dtr), length(rec.t) - i0 - 1)
    data = getfield(rec, field)
    kn = norm.(rec.ks);  shells = sort(unique(round.(kn; digits=8)))
    C = zeros(length(shells), maxlag + 1)
    for (j, kv) in enumerate(shells)
        idx = findall(k -> abs(k - kv) < 1e-6, kn)
        for i in idx
            C[j, :] .+= autocorr([data[r][i] for r in i0:length(rec.t)], maxlag)
        end
        C[j, :] ./= length(idx)
    end
    return shells, (0:maxlag) .* dtr, C
end

# ── Transverse mode as a damped oscillator ───────────────────────────────────
# Viscous fluid with a shear modulus G (Kelvin–Voigt): ü = −2γ u̇ − ω0² u, with
# 2γ = νk², ω0 = c_t k.  Velocity (momentum) autocorrelation, normalized at τ = 0:
#   ω0 > γ:  e^{−γτ}[cos ωτ − (γ/ω) sin ωτ],     ω = (ω0² − γ²)^{1/2}
#   ω0 < γ:  e^{−γτ}[cosh κτ − (γ/κ) sinh κτ],   κ = (γ² − ω0²)^{1/2}
# ω0 = 0 gives e^{−2γτ} = e^{−νk²τ}, the pure fluid.
function oscillator_corr(τ, γ, ω0)
    d = ω0^2 - γ^2
    if d > 1e-14
        ω = sqrt(d);  return @. exp(-γ * τ) * (cos(ω * τ) - (γ / ω) * sin(ω * τ))
    elseif d < -1e-14
        κ = sqrt(-d);  return @. exp(-γ * τ) * (cosh(κ * τ) - (γ / κ) * sinh(κ * τ))
    else
        return @. exp(-γ * τ) * (1 - γ * τ)
    end
end

# Least-squares fit of A·oscillator_corr(τ, γ, ω0) to C(τ): coarse grid, then a fine grid
# around the best point; A is linear.  Returns (A, γ, ω0).
function fit_oscillator(τ, C; γmax = 0.4, ω0max = 0.6, n = 120)
    best = (Inf, 0.0, 0.0, 0.0)
    function scan(γs, ωs)
        for γ in γs, ω0 in ωs
            f = oscillator_corr(τ, γ, ω0);  ff = dot(f, f);  ff > 0 || continue
            A = dot(f, C) / ff;  r = sum(abs2, C .- A .* f)
            r < best[1] && (best = (r, A, γ, ω0))
        end
    end
    scan(range(0, γmax, length = n), range(0, ω0max, length = n))
    dγ, dω = γmax / n, ω0max / n
    scan(range(max(best[3] - 2dγ, 0), best[3] + 2dγ, length = 41), range(max(best[4] - 2dω, 0), best[4] + 2dω, length = 41))
    return best[2], best[3], best[4]
end

# ── Output: one HDF5 file per realization ────────────────────────────────────
# Datasets (Julia shape; h5py/numpy sees the axes reversed, e.g. MT as (nt, nk)):
#   ks (2, nk)  t (nt)  ML, MT, S (nk, nt) complex  [E (nk, nt) if recorded]  H, KE (nt)
# Complex numbers are stored as the compound {r, i}, which h5py reads as complex.
# The run parameters are attributes of the root group; NamedTuple entries are flattened
# as <name>_<field> (e.g. guards_rejected).
# The file is written to path.tmp and renamed, so a reader (or a run killed while
# writing) never sees a half-written file.
function write_equilibrium_h5(path::AbstractString, meta::NamedTuple, rec::Recorder,
                              H::Vector{Float64}, KE::Vector{Float64}; prec = nothing, extra = NamedTuple())
    nk = length(rec.ks)
    stack(v) = isempty(v) ? zeros(ComplexF64, nk, 0) : reduce(hcat, v)
    tmp = path * ".tmp"
    h5open(tmp, "w") do f
        f["ks"] = reduce(hcat, Vector.(rec.ks));  f["t"] = rec.t
        f["ML"] = stack(rec.ML);  f["MT"] = stack(rec.MT);  f["S"] = stack(rec.S)
        rec.energies && (f["E"] = stack(rec.E))
        isempty(rec.UT) || (f["UL"] = stack(rec.UL);  f["UT"] = stack(rec.UT))
        f["H"] = H;  f["KE"] = KE
        for (k, v) in pairs(extra);  f[string(k)] = v;  end            # further time series
        if prec !== nothing && !isempty(prec.t)
            f["part_t"] = prec.t;  f["part_rs"] = prec.rs;  f["part_D2"] = reduce(hcat, prec.D2)
            f["part_MSD"] = prec.MSD;  f["part_B"] = prec.B;  f["part_Fs"] = prec.Fs
            f["part_npairs"] = length.(prec.pairs);  f["part_nbragg"] = length(prec.bs)
        end
        at = attributes(f)
        for (k, v) in pairs(meta)
            if v isa NamedTuple
                for (k2, v2) in pairs(v);  at["$(k)_$(k2)"] = v2;  end
            else
                at[string(k)] = v isa AbstractVector ? collect(v) : v
            end
        end
    end
    mv(tmp, path; force = true)
    return path
end

# ── Checkpoints ──────────────────────────────────────────────────────────────
# The complete state of a run (any NamedTuple: particles with all their fields, the cell
# list, the random streams, counters, the time series so far), serialized to path,
# atomically (path.tmp, then renamed).  A restart from it continues bit-identically.
# Julia serialization: read back with the same code version only (a run resumes with
# the checkout it started from).
function save_checkpoint(path::AbstractString, state::NamedTuple)
    tmp = path * ".tmp"
    open(io -> serialize(io, state), tmp, "w")
    mv(tmp, path; force = true)
    return path
end
load_checkpoint(path::AbstractString) = open(deserialize, path)

# Read a file written by write_equilibrium_h5 (or an old .jls) as (meta, rec, H, KE),
# with meta a NamedTuple of the attributes and rec a Recorder.
function read_equilibrium(path::AbstractString)
    endswith(path, ".jls") && return deserialize(path)
    h5open(path, "r") do f
        at = attributes(f)
        meta = NamedTuple(Symbol(k) => read(at[k]) for k in keys(at))
        ks = [SVector{2,Float64}(c) for c in eachcol(read(f["ks"]))]
        cols(name) = haskey(f, name) ? [Vector(c) for c in eachcol(read(f[name]))] : Vector{ComplexF64}[]
        rec = Recorder(ks, read(f["t"]), cols("ML"), cols("MT"), cols("S"), cols("E"), haskey(f, "E"),
                       cols("UL"), cols("UT"), SVector{2,Float64}[], SVector(0.0, 0.0))
        part = haskey(f, "part_t") ? (t = read(f["part_t"]), rs = read(f["part_rs"]), D2 = read(f["part_D2"]),
                                      MSD = read(f["part_MSD"]), B = read(f["part_B"]), Fs = read(f["part_Fs"])) : nothing
        return (meta = meta, rec = rec, H = read(f["H"]), KE = read(f["KE"]), part = part)
    end
end
