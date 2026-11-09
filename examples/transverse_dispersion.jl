# transverse_dispersion.jl — the transverse correlator of the equilibrium ensembles as a
# damped oscillation (transverse sound of a weakly rigid particle arrangement).
# A viscous fluid with a small shear modulus G (Kelvin–Voigt): the transverse
# displacement obeys ü = −νk² u̇ − c_t²k² u, c_t² = G/w, and the momentum (velocity)
# correlator is that of a damped oscillator,
#     C(τ) = A e^{−γτ} [cos ωτ − (γ/ω) sin ωτ],   γ = νk²/2,   ω² = c_t²k² − γ²,
# whose initial slope −2γ = −νk² is the hydrodynamic one.  For each k shell the ensemble
# mean C_T(k,τ)/C_T(k,0) is fitted by a grid search in (γ, ω), A linear; then
# c_t = (ω² + γ²)^{1/2}/k and 2γ/(νk²) → 1 if the viscosity is the input one.
#
# Run:  julia --project examples/transverse_dispersion.jl [campaign:run ...]
#   default: first:eqL1.2 second:eqL1.2 first:eqL2.4 second:eqL2.4
#   → docs/figs/transverse_dispersion.png, docs/figs/transverse_fits.png
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, Statistics
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
const ROOT  = get(ENV, "SR_DATADIR", joinpath(get(ENV, "SCRATCH", "."), "sph"))
const SETS  = length(ARGS) > 0 ? ARGS : ["first:eqL1.2", "second:eqL1.2", "first:eqL2.4", "second:eqL2.4"]
const TBURN = 2.0;  const MAXLAG = 60.0;  const NSHELL = 8

model(τ, γ, ω) = exp.(-γ .* τ) .* (cos.(ω .* τ) .- (γ / ω) .* sin.(ω .* τ))
function fit(τ, C)
    best = (Inf, 0.0, 0.0, 0.0)
    for γ in range(0.0, 0.4, length = 201), ω in range(0.002, 0.6, length = 300)
        f = model(τ, γ, ω);  A = dot(f, C) / dot(f, f)
        r = sum(abs2, C .- A .* f)
        r < best[1] && (best = (r, A, γ, ω))
    end
    return best[2:4]
end

function analyse(camp, run)
    d = joinpath(ROOT, camp, run);  isdir(d) || return nothing
    fs = filter(f -> endswith(f, ".h5") && startswith(f, "equilibrium_"), readdir(d))
    runs = filter(r -> !haskey(r.meta, :complete) || r.meta.complete,
                  [read_equilibrium(joinpath(d, f)) for f in fs])
    isempty(runs) && return nothing
    m = runs[1].meta;  ν = m.etas / m.Trel;  cs = sqrt(m.cs2)
    per = [shell_correlators(r.rec, :MT; t0 = TBURN, maxlag_t = MAXLAG) for r in runs]
    ks, τ = per[1][1], collect(per[1][2])
    Cs = cat([p[3] for p in per]...; dims = 3)
    C  = mean(Cs; dims = 3)[:, :, 1];  E = std(Cs; dims = 3)[:, :, 1] ./ sqrt(length(runs))
    n = min(NSHELL, length(ks))
    fits = [fit(τ, C[j, :] ./ C[j, 1]) for j in 1:n]
    return (; camp, run, m, ν, cs, ks = ks[1:n], τ, C = C[1:n, :] ./ C[1:n, 1],
              E = E[1:n, :] ./ C[1:n, 1], fits, nruns = length(runs))
end

res = filter(!isnothing, [analyse(split(s, ":")...) for s in SETS])

# Figure 1: fits for the smallest shells of each set
fig, axs = subplots(length(res), 1, figsize = (11, 3.2 * length(res)), squeeze = false)
for (i, r) in enumerate(res)
    ax = axs[i, 1]
    for j in 1:min(4, length(r.ks))
        c = "C$(j-1)";  A, γ, ω = r.fits[j]
        ax.fill_between(r.τ, r.C[j, :] .- r.E[j, :], r.C[j, :] .+ r.E[j, :], color = c, alpha = 0.25)
        ax.plot(r.τ, A .* model(r.τ, γ, ω), color = c, lw = 1.0, ls = "--")
        ax.plot(r.τ, exp.(-r.ν * r.ks[j]^2 .* r.τ), color = c, lw = 0.6, ls = ":")
        ax.plot([], [], color = c, label = @sprintf("kh = %.2f: A = %.2f, c_t = %.3f, 2γ/νk² = %.2f",
                                                       r.ks[j] * r.m.h, A, sqrt(ω^2 + γ^2) / r.ks[j], 2γ / (r.ν * r.ks[j]^2)))
    end
    ax.axhline(0, color = "k", lw = 0.5)
    ax.set_title(@sprintf("%s / %s (%d runs, N = %d): band = measured ±1σ, dashed = damped-oscillator fit, dotted = e^{−νk²τ}",
                          r.camp, r.run, r.nruns, r.m.N), fontsize = 9)
    ax.set_ylabel("C_T(k,τ) / C_T(k,0)");  ax.legend(fontsize = 8, loc = "upper right")
end
axs[end, 1].set_xlabel("τ")
tight_layout()
savefig(joinpath(@__DIR__, "..", "docs", "figs", "transverse_fits.png"), dpi = 110)

# Figure 2: dispersion ω(k) and damping γ(k) over all sets
fig, (a1, a2) = subplots(1, 2, figsize = (13, 4.8))
mk = Dict("first" => "o", "second" => "s")
for (i, r) in enumerate(res)
    ω = [f[3] for f in r.fits];  γ = [f[2] for f in r.fits];  kh = r.ks .* r.m.h
    lab = "$(r.camp)/$(r.run)"
    c = occursin("L2.4", r.run) ? "C1" : "C0"
    fc = r.camp == "first" ? "none" : c
    a1.plot(kh, sqrt.(ω .^ 2 .+ γ .^ 2) ./ r.ks, get(mk, r.camp, "^"), color = c, mfc = fc, label = lab)
    a2.plot(kh, 2 .* γ ./ (r.ν .* r.ks .^ 2), get(mk, r.camp, "^"), color = c, mfc = fc, label = lab)
end
a1.set_xlabel("k h");  a1.set_ylabel("c_t = (ω² + γ²)^{1/2} / k");  a1.set_title("transverse sound speed (c_s = 0.577)")
a1.set_ylim(bottom = 0);  a1.legend(fontsize = 8)
a2.axhline(1, color = "k", lw = 0.5)
a2.set_xlabel("k h");  a2.set_ylabel("2γ / (ν k²)");  a2.set_title("damping rate / input viscous rate");  a2.set_ylim(bottom = 0)
a2.legend(fontsize = 8)
tight_layout()
savefig(joinpath(@__DIR__, "..", "docs", "figs", "transverse_dispersion.png"), dpi = 110)

for r in res
    println("== $(r.camp)/$(r.run)  $(r.nruns) runs  ν = $(round(r.ν, sigdigits=3))")
    for j in eachindex(r.ks)
        A, γ, ω = r.fits[j]
        @printf("  kh = %.3f  A = %.3f  ω/k = %.4f  2γ/νk² = %.3f  ω²+γ² = (%.4f k)²\n",
                r.ks[j] * r.m.h, A, ω / r.ks[j], 2γ / (r.ν * r.ks[j]^2), sqrt(ω^2 + γ^2) / r.ks[j])
    end
end
