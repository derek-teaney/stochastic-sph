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

# The global multipliers whose *average* totals are (E, M): Σ_a ⟨e_a⟩ = E, Σ_a ⟨m_a⟩ = M.
# Newton in θ = (−β⁰, β), conjugate to x = (e, m), with ∂⟨x⟩/∂θ = Cov(x) (exact for this
# exponential family).  Starts from the most probable state at the mean densities.
# Particles with equal volume share their moments.
function beta_from_averages(V::Vector{Float64}, E::Float64, M::SVector{2,Float64},
                            eos::AbstractEOS; nq::Int = 41, iters::Int = 8)
    Vtot = sum(V);  _, v0, T0 = eos_from_energy(E / Vtot, M / Vtot, eos)
    θ = SVector(-1 / T0, v0[1] / T0, v0[2] / T0);  target = SVector(E, M[1], M[2])
    groups = Dict{Float64,Int}();  for v in V;  groups[v] = get(groups, v, 0) + 1;  end
    mom = nothing
    for it in 1:iters
        β0 = -θ[1];  β = SVector(θ[2], θ[3])
        mom = Dict(v => particle_moments(v, β0, β, eos; nq) for v in keys(groups))
        X = sum(n * mom[v].mean for (v, n) in groups);  Ct = sum(n * mom[v].cov for (v, n) in groups)
        r = target - X
        abs(r[1]) <= 1e-12 * abs(E) && norm(r[2:3]) <= 1e-12 * abs(E) && break
        θ = θ + Ct \ r
    end
    return -θ[1], SVector(θ[2], θ[3]), mom
end

# Draw (e_a, m_a) for all particles at the multipliers (β⁰, β) (moments from
# beta_from_averages): independent draws from the canonical ensemble, so the totals
# fluctuate about their averages (E, M).  With exact_totals = true the difference between
# the drawn totals and (E, M) is then removed uniformly, i.e. in proportion to the
# volumes (an equal share per particle on an equal-volume lattice), so that a run starts
# at exactly the target energy and momentum.
function draw_configuration(V::Vector{Float64}, β0, β, mom, eos, rng; nmh::Int = 30,
                            exact_totals::Bool = false, E = nothing, M = nothing)
    N = length(V);  X = Vector{SVector{3,Float64}}(undef, N);  nacc = 0
    for a in 1:N
        p = mom[V[a]]
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
end
Recorder(ks; energies = false) = Recorder(ks, Float64[], [], [], [], [], energies)

function record!(rec::Recorder, particles::Vector{Particle}, t::Float64, eos::AbstractEOS)
    nk = length(rec.ks)
    ML = zeros(ComplexF64, nk);  MT = zeros(ComplexF64, nk);  S = zeros(ComplexF64, nk)
    E  = zeros(ComplexF64, rec.energies ? nk : 0)
    ea = rec.energies ? [particle_energy(pa, eos) for pa in particles] : Float64[]
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

# ── Output: one HDF5 file per realization ────────────────────────────────────
# Datasets (Julia shape; h5py/numpy sees the axes reversed, e.g. MT as (nt, nk)):
#   ks (2, nk)  t (nt)  ML, MT, S (nk, nt) complex  [E (nk, nt) if recorded]  H, KE (nt)
# Complex numbers are stored as the compound {r, i}, which h5py reads as complex.
# The run parameters are attributes of the root group; NamedTuple entries are flattened
# as <name>_<field> (e.g. guards_rejected).
function write_equilibrium_h5(path::AbstractString, meta::NamedTuple, rec::Recorder,
                              H::Vector{Float64}, KE::Vector{Float64})
    nk = length(rec.ks)
    stack(v) = isempty(v) ? zeros(ComplexF64, nk, 0) : reduce(hcat, v)
    h5open(path, "w") do f
        f["ks"] = reduce(hcat, Vector.(rec.ks));  f["t"] = rec.t
        f["ML"] = stack(rec.ML);  f["MT"] = stack(rec.MT);  f["S"] = stack(rec.S)
        rec.energies && (f["E"] = stack(rec.E))
        f["H"] = H;  f["KE"] = KE
        at = attributes(f)
        for (k, v) in pairs(meta)
            if v isa NamedTuple
                for (k2, v2) in pairs(v);  at["$(k)_$(k2)"] = v2;  end
            else
                at[string(k)] = v isa AbstractVector ? collect(v) : v
            end
        end
    end
    return path
end

# Read a file written by write_equilibrium_h5 (or an old .jls) as (meta, rec, H, KE),
# with meta a NamedTuple of the attributes and rec a Recorder.
function read_equilibrium(path::AbstractString)
    endswith(path, ".jls") && return deserialize(path)
    h5open(path, "r") do f
        at = attributes(f)
        meta = NamedTuple(Symbol(k) => read(at[k]) for k in keys(at))
        ks = [SVector{2,Float64}(c) for c in eachcol(read(f["ks"]))]
        cols(name) = haskey(f, name) ? [Vector(c) for c in eachcol(read(f[name]))] : Vector{ComplexF64}[]
        rec = Recorder(ks, read(f["t"]), cols("ML"), cols("MT"), cols("S"), cols("E"), haskey(f, "E"))
        return (meta = meta, rec = rec, H = read(f["H"]), KE = read(f["KE"]))
    end
end
