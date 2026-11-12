# energystep_compare.jl — the energy-representation Hamiltonian (SPH_HAMILTONIAN=energy) against
# the smoothed one at the same settings (L = 1.2, h/a = 2, C2, t = 200, eq start):
#   smoothed: $SCRATCH/sph/rigidity/ha2          (campaign_rigidity.txt)
#   energy:   $SCRATCH/sph/energystep/ha2_energy (campaign_energystep.txt)
# Panels: (a) KE/eq(t); (b) ΔH/H(t); (c) C_T(k,τ)/C_T(k,0), two smallest shells, with the
# damped-oscillator fit and the pure fluid e^{−νk²τ}; (d) c_t from the fit (o) and from the
# displacement statics (□); (e) 2γ/νk² (1 = the input viscosity); (f) equal-time
# C_T(k,0)/(NVwT) (1 in equilibrium) and C_L(k,0)/(NVwT); (g) ⟨|u_T|²⟩(t), smallest shell.
#
# Run:  julia --project examples/energystep_compare.jl [smoothed_dir energy_dir out.png]
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, HDF5, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))
S = ENV["SCRATCH"]
const SETS = [("smoothed", length(ARGS) > 0 ? ARGS[1] : joinpath(S, "sph", "rigidity", "ha2"), "C3"),
              ("energy",   length(ARGS) > 1 ? ARGS[2] : joinpath(S, "sph", "energystep", "ha2_energy"), "C0")]
const OUT = length(ARGS) > 2 ? ARGS[3] : joinpath(@__DIR__, "..", "docs", "figs", "energystep_ha2.png")
const NSH = 4;  const TBURN = 2.0
mean_(v) = sum(v) / length(v)

function load(d)
    isdir(d) || return []
    fs = filter(f -> endswith(f, ".h5") && startswith(f, "equilibrium_"), readdir(d))
    filter(r -> get(r.meta, :complete, true), [read_equilibrium(joinpath(d, f)) for f in fs])
end
shellidx(ks) = (kn = round.(norm.(ks); digits = 8); sh = sort(unique(kn)); (sh, [findall(==(s), kn) for s in sh]))

function analyse(rs)
    m = rs[1].meta;  ν = m.etas / m.Trel;  wV = m.w * m.V;  n0 = m.N * m.V * m.w * m.T
    corr(f) = (per = [shell_correlators(r.rec, f; t0 = TBURN, maxlag_t = 60.0) for r in rs];
               (per[1][1], collect(per[1][2]), mean_([p[3] for p in per])))
    ks, τ, CT = corr(:MT);  _, _, CL = corr(:ML)
    fits = [fit_oscillator(τ, CT[j, :] ./ CT[j, 1]) for j in 1:NSH]
    sh, idx = shellidx(rs[1].rec.ks);  t = rs[1].rec.t;  i0 = findfirst(>=(TBURN), t)
    nt = minimum(length(r.rec.t) for r in rs)
    u2t = [mean_([mean_(abs2.(r.rec.UT[n][idx[j]])) for r in rs]) for j in 1:NSH, n in 1:nt]
    m2 = [mean_([mean_(abs2.(r.rec.MT[n][idx[j]])) for r in rs for n in i0:nt]) for j in 1:NSH]
    ctstat = [sqrt(m2[j] / (wV^2 * sh[j]^2 * mean_(u2t[j, i0:end]))) for j in 1:NSH]
    nk = minimum(length(r.KE) for r in rs)
    (; m, ν, n0, ks, τ, CT, CL, fits, ctstat, u2t, t = t[1:nt], tk = t[1:nk], n = length(rs),
       KE = [r.KE[1:nk] for r in rs], H = [r.H[1:nk] for r in rs], wall = mean_([r.meta.wall for r in rs]))
end

res = Dict{String,Any}()
for (name, d, c) in SETS
    rs = load(d);  isempty(rs) && (println("$name: no complete runs in $d");  continue)
    r = analyse(rs);  res[name] = r
    @printf("== %-8s %2d runs  N = %d  wall %.0f s/run  KE/eq %.3f  ΔH/H(t=%.0f) %+.2e\n", name, r.n, r.m.N, r.wall,
            mean_([mean_(k[length(k)÷10:end]) for k in r.KE]), r.tk[end], mean_([h[end] for h in r.H]))
    for j in 1:NSH
        A, γ, ω0 = r.fits[j]
        @printf("   kh = %.3f  c_t fit %.4f  statics %.4f  2γ/νk² %.2f  C_T(0)/NVwT %.3f  C_L(0)/NVwT %.3f\n",
                r.ks[j] * r.m.h, ω0 / r.ks[j], r.ctstat[j], 2γ / (r.ν * r.ks[j]^2), r.CT[j, 1] / r.n0, r.CL[j, 1] / r.n0)
    end
end
isempty(res) && exit()

fig = figure(figsize = (18, 10))
ax = [subplot(2, 4, i) for i in 1:8]
for (name, _, c) in SETS
    haskey(res, name) || continue;  r = res[name];  kh = r.ks[1:NSH] .* r.m.h
    for k in r.KE;  ax[1].plot(r.tk, k, color = c, lw = 0.3, alpha = 0.3);  end
    ax[1].plot(r.tk, mean_(r.KE), color = c, lw = 1.2, label = @sprintf("%s (%d runs, %.0f s/run)", name, r.n, r.wall))
    for h in r.H;  ax[2].plot(r.tk, h, color = c, lw = 0.4, alpha = 0.4);  end
    ax[2].plot(r.tk, mean_(r.H), color = c, lw = 1.5, label = name)
    for (j, a) in ((1, ax[3]), (2, ax[4]))
        A, γ, ω0 = r.fits[j]
        a.plot(r.τ, r.CT[j, :] ./ r.CT[j, 1], color = c, lw = 1.5, label = @sprintf("%s: c_t = %.4f", name, ω0 / r.ks[j]))
        a.plot(r.τ, A .* oscillator_corr(r.τ, γ, ω0), color = c, lw = 0.8, ls = "--")
        a.plot(r.τ, exp.(-r.ν * r.ks[j]^2 .* r.τ), color = c, lw = 0.8, ls = ":")
        a.set_title(@sprintf("C_T(k,τ)/C_T(k,0), kh = %.3f\n(-- fit, ··· fluid e^{-νk²τ})", r.ks[j] * r.m.h))
    end
    ax[5].plot(kh, [f[3] for f in r.fits] ./ r.ks[1:NSH], "o-", color = c, label = name * " fit of C_T")
    ax[5].plot(kh, r.ctstat, "s--", color = c, mfc = "none", label = name * " displacements")
    ax[6].plot(kh, [2f[2] for f in r.fits] ./ (r.ν .* r.ks[1:NSH] .^ 2), "o-", color = c, label = name)
    ax[7].plot(kh, r.CT[1:NSH, 1] ./ r.n0, "o-", color = c, label = name * " C_T")
    ax[7].plot(kh, r.CL[1:NSH, 1] ./ r.n0, "^--", color = c, mfc = "none", label = name * " C_L")
    ax[8].plot(r.t, r.u2t[1, :], color = c, label = name)
end
ax[1].axhline(1, color = "k", lw = 0.5);  ax[1].set_xlabel("t");  ax[1].set_title("KE / equilibrium");  ax[1].legend(fontsize = 8)
ax[2].axhline(0, color = "k", lw = 0.5);  ax[2].set_xlabel("t");  ax[2].set_title("ΔH/H");  ax[2].legend(fontsize = 8)
for a in ax[3:4];  a.axhline(0, color = "k", lw = 0.5);  a.set_xlabel("τ");  a.legend(fontsize = 8);  end
ax[5].set_ylim(bottom = 0);  ax[5].set_xlabel("k h");  ax[5].set_title("transverse sound speed c_t");  ax[5].legend(fontsize = 7)
ax[6].axhline(1, color = "k", lw = 0.5);  ax[6].set_xlabel("k h");  ax[6].set_title("damping 2γ/νk² (1 = input η)");  ax[6].legend(fontsize = 8)
ax[7].axhline(1, color = "k", lw = 0.5);  ax[7].set_xlabel("k h");  ax[7].set_title("equal time C(k,0)/(NVwT) (1 = equilibrium)");  ax[7].legend(fontsize = 7)
ax[8].set_xlabel("t");  ax[8].set_title("⟨|u_T(k,t)|²⟩, smallest shell\n(saturates: solid; grows ∝ t: fluid)");  ax[8].legend(fontsize = 8)
suptitle("energy (blue) vs smoothed (red) Hamiltonian: L = 1.2, h/a = 2, C2, eq start, t = 200")
tight_layout();  savefig(OUT, dpi = 100)
println("wrote $OUT")
