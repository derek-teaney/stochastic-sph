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
