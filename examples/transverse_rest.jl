# transverse_rest.jl — the transverse correlator at rest, C_T(k,τ)/C_T(k,0) = ⟨M_T(t+τ) M_T*(t)⟩, for the particle
# step with C2 (boost.tex, b_v0) and C4 (kernel/k4_v0) and the grid (grid/g_v0), against hydrodynamics e^{−νk²τ}.
# A rigid particle arrangement shows as an oscillation (transverse sound, c_t).  Averaged over k along x and y.
#   julia --project examples/transverse_rest.jl   → docs/figs/transverse_rest.png and a table
using PyPlot, HDF5, Printf, StaticArrays, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const SC = joinpath(ENV["SCRATCH"], "sph")
const SETS = [("particles, C2", joinpath(SC, "boost", "b_v0"), "C1"), ("particles, C4", joinpath(SC, "kernel", "k4_v0"), "C2"),
              ("grid, C2", joinpath(SC, "grid", "g_v0"), "C0")]
const TBURN = 5.0;  const TAUMAX = 60.0;  const H = 0.05

function series(d)
    out = []
    for f in filter(f -> endswith(f, ".h5"), readdir(d; join = true))
        r = read_equilibrium(f);  t = r.rec.t;  it = findall(>(TBURN), t);  dt = t[2] - t[1]
        push!(out, (ks = r.rec.ks, MT = [[r.rec.MT[i][j] for i in it] for j in eachindex(r.rec.ks)], dt = dt,
                    ν = r.meta.etas / r.meta.Trel))
    end
    out
end
# for |k| = 2π n / L along x (n, 0) and along y (0, n): normalized C_T(τ), averaged over the runs and the two directions
function ct(runs, n)
    r1 = runs[1];  L = 2π ./ [minimum(abs(k[1]) for k in r1.ks if k[1] > 0), minimum(abs(k[2]) for k in r1.ks if k[2] > 0)]
    js = [findfirst(k -> abs(k[1] - 2π * n / L[1]) < 1e-6 && abs(k[2]) < 1e-9, r1.ks),
          findfirst(k -> abs(k[2] - 2π * n / L[2]) < 1e-6 && abs(k[1]) < 1e-9, r1.ks)]
    nl = round(Int, TAUMAX / r1.dt)
    C = zeros(nl + 1)
    for r in runs, j in js
        c = autocorr(r.MT[j], nl);  C .+= real.(c) ./ real(c[1])
    end
    C ./= length(runs) * length(js)
    return (0:nl) .* r1.dt, C, norm(r1.ks[js[1]]), r1.ν
end

data = [(lab, series(d), col) for (lab, d, col) in SETS if isdir(d)]
fig, axs = subplots(1, 3, figsize = (17, 4.6))
println("transverse correlator at rest: minimum of C_T(τ)/C_T(0) for τ ≤ $TAUMAX, and C_T at τ = 1/(νk²)")
for (ax, n) in zip(axs, (1, 2, 3))
    kk = 0.0;  ν = 0.0
    for (lab, runs, col) in data
        τ, C, k, ν = ct(runs, n);  kk = k
        ax.plot(τ, C, color = col, lw = 1.4, label = "$lab ($(length(runs)) runs)")
        i1 = argmin(abs.(τ .- 1 / (ν * k^2)))
        @printf("  kh = %.2f  %-14s  min %.3f at τ = %5.1f   C(1/νk²) = %.3f (hydro e^{-1} = 0.368)\n", k * H, lab, minimum(C), τ[argmin(C)], C[i1])
    end
    τs = range(0, TAUMAX, length = 400)
    ax.plot(τs, exp.(-ν * kk^2 .* τs), "k--", lw = 1, label = "e^{−νk²τ}")
    ax.axhline(0, color = "k", lw = 0.4);  ax.set_xlabel("τ");  ax.set_ylim(-0.4, 1.05)
    ax.set_title(@sprintf("at rest: C_T(k,τ)/C_T(k,0), kh = %.2f", kk * H), fontsize = 10)
    n == 1 && ax.legend(fontsize = 8)
end
tight_layout();  savefig(joinpath(@__DIR__, "..", "docs", "figs", "transverse_rest.png"), dpi = 100);  close()
println("wrote transverse_rest.png")
