# frozen_analysis.jl — campaign `frozen` (nersc/campaign_frozen.txt, docs/frozen.tex).
# Four sets at L = 1.2, h/a = 2: :energy at S0 = 1e6, 1e7, 1e8 and :smoothed at 1e6.
#   fig 1  frozen_D2t.png    pair separation D²(r = a, t)/a² against t, with the hydrodynamic
#                            estimate 4 D_rel(a) t (dashed)
#   fig 2  frozen_D2r.png    D²(r, t_end)/a² against r/a, with the hydrodynamic r dependence
#   fig 3  frozen_CT.png     C_T(k,τ)/C_T(k,0) in three shells, with the fluid e^{−νk²τ}
#   fig 4  frozen_summary.png  c_t and 2γ/νk² against δv; D_rel(a) and D_self against δv with
#                            the hydrodynamic estimates
#   fig 5  frozen_energy.png KE/eq and ΔH/H against t
# Hydrodynamic estimates (frozen.tex, eqs. Drel, Dself), T/w = δv² V, k_max = π/a:
#   D_rel(r) = (T/w)/(2πν) ln(k_max r),   D_self = (T/w)/(4πν) ln(k_max L/2π).
# Diffusion constants from a straight-line fit D²(t) = c + 4 D t over t ≥ TFIT.
#
# Run:  julia --project examples/frozen_analysis.jl [dir]   (default $SCRATCH/sph/frozen)
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, HDF5, Serialization
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "frozen")
const SETS = [("e_s1e6", "energy, S0 = 1e6", "C0"), ("e_s1e7", "energy, S0 = 1e7", "C2"),
              ("e_s1e8", "energy, S0 = 1e8", "C1"), ("s_s1e6", "smoothed, S0 = 1e6", "C3")]
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
const TBURN0 = 2.0;  const TFIT = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 50.0;  const NSH = 3
mean_(v) = sum(v) / length(v)
sem(v) = length(v) > 1 ? sqrt(sum(abs2, v .- mean_(v)) / (length(v) * (length(v) - 1))) : NaN

function load(run)
    d = joinpath(DIR, run);  isdir(d) || return []
    fs = filter(f -> endswith(f, ".h5") && startswith(f, "equilibrium_"), readdir(d))
    filter(r -> get(r.meta, :complete, true), [read_equilibrium(joinpath(d, f)) for f in fs])
end
slope(t, y) = (tm = mean_(t); sum((t .- tm) .* (y .- mean_(y))) / sum(abs2, t .- tm))

res = Dict{String,Any}()
for (run, lab, c) in SETS
    rs = load(run);  isempty(rs) && (println("$run: no complete runs");  continue)
    m = rs[1].meta;  a = m.a;  ν = m.etas / m.Trel;  Tw = m.T / m.w;  δv = sqrt(m.T / (m.w * m.V))
    kmax = π / a
    Drel_h(r) = Tw / (2π * ν) * log(kmax * r)
    Dself_h = Tw / (4π * ν) * log(kmax * m.L[1] / 2π)
    # pair separation and MSD
    p1 = rs[1].part;  tp = p1.t;  nt = minimum(length(r.part.t) for r in rs);  tp = tp[1:nt]
    D2 = [mean_([r.part.D2[c, n] for r in rs]) for c in 1:length(p1.rs), n in 1:nt]
    MSD = [mean_([r.part.MSD[n] for r in rs]) for n in 1:nt]
    it = findall(>=(TFIT), tp)
    Drel = [slope(tp[it], [r.part.D2[c, n] for n in it]) / 4 for r in rs, c in 1:length(p1.rs)]
    Dself = [slope(tp[it], r.part.MSD[it]) / 4 for r in rs]
    # transverse correlator
    tb = min(TBURN0, rs[1].rec.t[end] / 4)                       # burn-in (shorter for test runs)
    per = [shell_correlators(r.rec, :MT; t0 = tb, maxlag_t = 60.0) for r in rs]
    ks, τ = per[1][1], collect(per[1][2]);  C = mean_([p[3] for p in per])
    fits = [fit_oscillator(τ, C[j, :] ./ C[j, 1]) for j in 1:4]
    nk = minimum(length(r.KE) for r in rs)
    res[run] = (; lab, c, m, a, ν, Tw, δv, rs = p1.rs, tp, D2, MSD, Drel, Dself, Drel_h, Dself_h,
                ks, τ, C, fits, n = length(rs), tk = rs[1].rec.t[1:nk],
                KE = mean_([r.KE[1:nk] for r in rs]), H = mean_([r.H[1:nk] for r in rs]))
    @printf("== %-8s %2d runs  δv = %.4f  ν = %.2e  T/w = %.2e\n", run, length(rs), δv, ν, Tw)
    for (ci, r) in enumerate(p1.rs)
        @printf("   r = %2.0fa:  D_rel = %.2e ± %.1e   hydro %.2e   ratio %.2f\n", r, mean_(Drel[:, ci]),
                sem(Drel[:, ci]), Drel_h(r * a), mean_(Drel[:, ci]) / Drel_h(r * a))
    end
    @printf("   D_self = %.2e ± %.1e   hydro %.2e   ratio %.2f\n", mean_(Dself), sem(Dself), Dself_h, mean_(Dself) / Dself_h)
    for j in 1:4
        A, γ, ω0 = fits[j]
        @printf("   kh = %.3f  c_t = %.4f  2γ/νk² = %.2f\n", ks[j] * m.h, ω0 / ks[j], 2γ / (ν * ks[j]^2))
    end
    @printf("   KE/eq %.3f   ΔH/H(t_end) %+.2e\n", mean_(res[run].KE[end÷2:end]), res[run].H[end])
end
isempty(res) && exit()
runs = [s[1] for s in SETS if haskey(res, s[1])]

# fig 1: D²(a, t)
figure(figsize = (8, 6))
for run in runs
    r = res[run];  a2 = r.a^2
    loglog(r.tp, r.D2[1, :] ./ a2, color = r.c, lw = 1.5, label = r.lab)
    loglog(r.tp, 4 * r.Drel_h(r.a) .* r.tp ./ a2, color = r.c, lw = 1, ls = "--")
end
axhline(1, color = "k", lw = 0.5, ls = ":")
xlabel("t");  ylabel("D²(r = a, t) / a²");  ylim(1e-5, 1e3)
title("separation of nearest neighbours; dashed: hydrodynamic estimate 4 D_rel(a) t")
legend(fontsize = 9);  tight_layout();  savefig(joinpath(FIGD, "frozen_D2t.png"), dpi = 110)

# fig 2: D²(r, t_end) against r
figure(figsize = (8, 6))
for run in runs
    r = res[run];  a2 = r.a^2
    semilogy(r.rs, r.D2[:, end] ./ a2, "o-", color = r.c, lw = 1.5, label = r.lab)
    semilogy(r.rs, [4 * r.Drel_h(x * r.a) * r.tp[end] for x in r.rs] ./ a2, color = r.c, lw = 1, ls = "--")
end
xlabel("initial separation r / a");  ylabel(@sprintf("D²(r, t = %.0f) / a²", res[runs[1]].tp[end]))
title("pair separation against initial distance; dashed: hydrodynamics ∝ ln(πr/a)")
legend(fontsize = 9);  tight_layout();  savefig(joinpath(FIGD, "frozen_D2r.png"), dpi = 110)

# fig 3: C_T in three shells
fig, axs = subplots(1, NSH, figsize = (16, 4.8))
for (i, j) in enumerate((1, 3, 4))
    for run in runs
        r = res[run]
        axs[i].plot(r.τ, r.C[j, :] ./ r.C[j, 1], color = r.c, lw = 1.5, label = r.lab)
        axs[i].plot(r.τ, exp.(-r.ν * r.ks[j]^2 .* r.τ), color = r.c, lw = 0.8, ls = ":")
    end
    axs[i].axhline(0, color = "k", lw = 0.5);  axs[i].set_xlabel("τ")
    axs[i].set_title(@sprintf("C_T(k,τ)/C_T(k,0),  kh = %.3f   (dotted: fluid, each set's ν)", res[runs[1]].ks[j] * res[runs[1]].m.h))
    i == 1 && axs[i].legend(fontsize = 8)
end
tight_layout();  savefig(joinpath(FIGD, "frozen_CT.png"), dpi = 100)

# fig 4: summary against δv (energy sets as a line, smoothed as an open marker)
fig, axs = subplots(1, 3, figsize = (17, 5))
for run in runs
    r = res[run];  mk = startswith(run, "s_") ? "s" : "o";  ms = 9
    ct = [f[3] / r.ks[j] for (j, f) in enumerate(r.fits)];  dmp = [2f[2] / (r.ν * r.ks[j]^2) for (j, f) in enumerate(r.fits)]
    axs[1].semilogx(fill(r.δv, length(ct)), ct, mk, color = r.c, ms = ms, mfc = "none", label = r.lab)
    axs[2].semilogx(fill(r.δv, length(dmp)), dmp, mk, color = r.c, ms = ms, mfc = "none", label = r.lab)
    axs[3].loglog(r.δv, mean_(r.Drel[:, 1]), mk, color = r.c, ms = ms, label = r.lab * ": D_rel(a)")
    axs[3].loglog(r.δv, mean_(r.Dself), mk, color = r.c, ms = ms, mfc = "none", label = r.lab * ": D_self")
    axs[3].loglog(r.δv, r.Drel_h(r.a), "_", color = r.c, ms = 22, mew = 2)
end
axs[1].axhline(0.034, color = "k", lw = 0.8, ls = "--", label = "static c_t (T = 0 lattice)")
axs[1].set_xlabel("δv");  axs[1].set_ylabel("c_t (fit of C_T, kh = 0.26–0.52)");  axs[1].set_ylim(bottom = -0.002);  axs[1].legend(fontsize = 8)
axs[2].axhline(1, color = "k", lw = 0.5);  axs[2].set_xlabel("δv");  axs[2].set_ylabel("2γ/νk²  (1 = input viscosity)")
axs[3].set_xlabel("δv");  axs[3].set_ylabel("diffusion constant");  axs[3].legend(fontsize = 7)
axs[3].set_title("filled: D_rel(a), open: D_self, bar: hydrodynamic D_rel(a)")
tight_layout();  savefig(joinpath(FIGD, "frozen_summary.png"), dpi = 100)

# fig 5: KE and H
fig, axs = subplots(1, 2, figsize = (14, 4.8))
for run in runs
    r = res[run]
    axs[1].plot(r.tk, r.KE, color = r.c, lw = 0.6, label = r.lab);  axs[2].plot(r.tk, r.H, color = r.c, lw = 1.5, label = r.lab)
end
axs[1].axhline(1, color = "k", lw = 0.5);  axs[1].set_xlabel("t");  axs[1].set_ylabel("KE / equilibrium");  axs[1].legend(fontsize = 8)
axs[2].axhline(0, color = "k", lw = 0.5);  axs[2].set_xlabel("t");  axs[2].set_ylabel("ΔH/H");  axs[2].legend(fontsize = 8)
tight_layout();  savefig(joinpath(FIGD, "frozen_energy.png"), dpi = 100)
println("figures written to $FIGD")
