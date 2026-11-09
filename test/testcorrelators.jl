# testcorrelators.jl — does the correlator analysis recover known dynamics?
# Synthetic transverse modes with a known shear modulus and viscosity,
#     du = v dt,   dv = (−2γ v − ω0² u) dt + √(4γ) dW,   2γ = νk²,  ω0 = c_t k,
# (stationary ⟨|v|²⟩ = 1), sampled like the runs (Δt_rec = 0.0173, t = 400, 6 vectors per
# shell, 8 runs), are put through shell_correlators and fit_oscillator exactly as the run data
# are.  Two cases: c_t = 0.022 (the measured rigidity) and c_t = 0 (a pure fluid, where the
# fit must return ω0 ≈ 0 and C = e^{−νk²τ}).
#
# Run:  julia --project test/testcorrelators.jl
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end

const ν = 0.0022;  const DTR = 0.01732;  const TEND = 400.0;  const NRUN = 8;  const NSUB = 4
const KS = [5.18, 7.36, 10.36]                     # kh = 0.26, 0.37, 0.52
const NV = 6                                       # vectors per shell

function synth(ct, rng)
    ks = SVector{2,Float64}[];  for k in KS, j in 1:NV
        φ = 2π * (j - 1) / NV;  push!(ks, SVector(k * cos(φ), k * sin(φ)))
    end
    nk = length(ks);  nt = round(Int, TEND / DTR);  dt = DTR / NSUB
    γ = [ν * norm(k)^2 / 2 for k in ks];  ω0 = [ct * norm(k) for k in ks]
    u = [(randn(rng) + im * randn(rng)) / sqrt(2) / max(ω0[i], 1e-12) * (ct > 0) for i in 1:nk]
    v = [(randn(rng) + im * randn(rng)) / sqrt(2) for i in 1:nk]
    MT = Vector{Vector{ComplexF64}}(undef, nt);  t = collect(1:nt) .* DTR
    for n in 1:nt
        for _ in 1:NSUB, i in 1:nk
            ξ = (randn(rng) + im * randn(rng)) / sqrt(2)
            v[i] += (-2γ[i] * v[i] - ω0[i]^2 * u[i]) * dt + sqrt(4γ[i] * dt) * ξ
            u[i] += v[i] * dt
        end
        MT[n] = copy(v)
    end
    return Recorder(ks, t, MT, MT, MT, Vector{ComplexF64}[], false)
end

ok = true
for ct in (0.022, 0.0)
    rng = Xoshiro(7)
    per = [shell_correlators(synth(ct, rng), :MT; t0 = 2.0, maxlag_t = 60.0) for _ in 1:NRUN]
    ks, τ = per[1][1], collect(per[1][2])
    C = mean(cat([p[3] for p in per]...; dims = 3); dims = 3)[:, :, 1]
    println("c_t = $ct (input), ν = $ν")
    for j in eachindex(ks)
        A, γ, ω0 = fit_oscillator(τ, C[j, :] ./ C[j, 1])
        ctf = ω0 / ks[j];  r = 2γ / (ν * ks[j]^2)
        good = (ct > 0 ? abs(ctf / ct - 1) < 0.1 : ctf < 0.006) && abs(r - 1) < 0.25
        global ok &= good
        @printf("  kh = %.2f:  C(0) = %.3f  A = %.3f  c_t fit = %.4f  2γ/νk² = %.3f  %s\n",
                ks[j] * 0.05, C[j, 1], A, ctf, r, good ? "✓" : "✗")
    end
end
println(ok ? "testcorrelators.jl: analysis recovers the input." : "testcorrelators.jl: FAILED")
