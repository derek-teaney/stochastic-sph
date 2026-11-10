# energystep_correlators.jl — C_T(k,τ) and C_L(k,τ) for SPH_HAMILTONIAN = energy against
# smoothed, at the same settings (L = 1.2, h/a = 2, C2, eq start, t = 200, 16 realizations each):
#   smoothed: $SCRATCH/sph/rigidity/ha2,  energy: $SCRATCH/sph/energystep/ha2_energy.
# Normalized by NVwT (1 in equilibrium at τ = 0); band = ±1 standard error over realizations.
# Figure 1: C_T in the six smallest shells with the fluid e^{−νk²τ}.
# Figure 2: C_L in three shells with the sound mode cos(c_s kτ) e^{−Γ_s k²τ/2}, Γ_s = (4/3)ν.
#
# Run:  julia --project examples/energystep_correlators.jl [smoothed_dir energy_dir]
#   → docs/figs/energystep_CT.png, docs/figs/energystep_CL.png
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, HDF5, Serialization
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
S = ENV["SCRATCH"]
const SETS = [("smoothed", length(ARGS) > 0 ? ARGS[1] : joinpath(S, "sph", "rigidity", "ha2"), "C3"),
              ("energy",   length(ARGS) > 1 ? ARGS[2] : joinpath(S, "sph", "energystep", "ha2_energy"), "C0")]
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
const TBURN = 2.0;  const TMAX = 60.0
mean_(v) = sum(v) / length(v)

function load(d)
    fs = filter(f -> endswith(f, ".h5") && startswith(f, "equilibrium_"), readdir(d))
    filter(r -> get(r.meta, :complete, true), [read_equilibrium(joinpath(d, f)) for f in fs])
end

# per-realization shell correlators, normalized; mean and standard error over realizations
function corrs(rs, field)
    m = rs[1].meta;  n0 = m.N * m.V * m.w * m.T
    per = [shell_correlators(r.rec, field; t0 = TBURN, maxlag_t = TMAX) for r in rs]
    ks, τ = per[1][1], collect(per[1][2])
    Cs = [p[3] ./ n0 for p in per];  C = mean_(Cs)
    err = sqrt.(mean_([(c .- C) .^ 2 for c in Cs]) ./ (length(Cs) - 1))
    (; ks, τ, C, err, m, ν = m.etas / m.Trel, n = length(rs))
end

res = Dict(name => (rs = load(d); (T = corrs(rs, :MT), L = corrs(rs, :ML))) for (name, d, _) in SETS)

# Figure 1: C_T, six smallest shells
fig, axs = subplots(2, 3, figsize = (15, 8.5))
for j in 1:6
    ax = axs[(j - 1) ÷ 3 + 1, (j - 1) % 3 + 1]
    for (name, _, c) in SETS
        r = res[name].T
        ax.fill_between(r.τ, r.C[j, :] .- r.err[j, :], r.C[j, :] .+ r.err[j, :], color = c, alpha = 0.3, lw = 0)
        ax.plot(r.τ, r.C[j, :], color = c, lw = 1.5, label = @sprintf("%s (%d runs)", name, r.n))
    end
    r = res["energy"].T
    ax.plot(r.τ, exp.(-r.ν * r.ks[j]^2 .* r.τ), "k:", lw = 1.2, label = "fluid  e^{−νk²τ}")
    ax.axhline(0, color = "k", lw = 0.5)
    ax.set_title(@sprintf("C_T,  k h = %.3f", r.ks[j] * r.m.h));  ax.set_xlabel("τ")
    j == 1 && ax.legend(fontsize = 9)
end
suptitle("transverse momentum correlator C_T(k,τ)/(NVwT): energy (blue) vs smoothed (red); L = 1.2, h/a = 2; band ±1 s.e.")
tight_layout();  savefig(joinpath(FIGD, "energystep_CT.png"), dpi = 100)

# Figure 2: C_L, three shells
fig, axs = subplots(1, 3, figsize = (15, 4.5))
for (i, j) in enumerate((1, 3, 6))
    ax = axs[i]
    for (name, _, c) in SETS
        r = res[name].L
        ax.fill_between(r.τ, r.C[j, :] .- r.err[j, :], r.C[j, :] .+ r.err[j, :], color = c, alpha = 0.3, lw = 0)
        ax.plot(r.τ, r.C[j, :], color = c, lw = 1.2, label = @sprintf("%s (%d runs)", name, r.n))
    end
    r = res["energy"].L;  k = r.ks[j];  cs = sqrt(1/3)
    ax.plot(r.τ, cos.(cs * k .* r.τ) .* exp.(-(4/3) * r.ν * k^2 .* r.τ ./ 2), "k:", lw = 1.2, label = "sound  cos(c_s kτ) e^{−Γ_s k²τ/2}")
    ax.axhline(0, color = "k", lw = 0.5)
    ax.set_title(@sprintf("C_L,  k h = %.3f", k * r.m.h));  ax.set_xlabel("τ");  ax.set_xlim(0, 30)
    i == 1 && ax.legend(fontsize = 8)
end
tight_layout();  savefig(joinpath(FIGD, "energystep_CL.png"), dpi = 100)

for (name, _, _) in SETS
    r = res[name]
    @printf("%-9s C_T(k,0): %s\n", name, join([@sprintf("%.3f±%.3f", r.T.C[j, 1], r.T.err[j, 1]) for j in 1:6], "  "))
    @printf("%-9s C_L(k,0): %s\n", name, join([@sprintf("%.3f±%.3f", r.L.C[j, 1], r.L.err[j, 1]) for j in 1:6], "  "))
end
println("kh: ", join([@sprintf("%.3f", k * res["energy"].T.m.h) for k in res["energy"].T.ks[1:6]], "  "))
