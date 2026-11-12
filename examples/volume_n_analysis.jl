# volume_n_analysis.jl — campaign `volume_n` (SPH_VOLUME=number) against the entropy volume V_a = s_a/S_a
# (campaigns energy2 and frozen), all SPH_HAMILTONIAN=energy, h/a = 2.  docs/volume_n.tex.
#   volume_n_H.png        ΔH/H and KE/eq against t, L = 0.6 (entropy dt1; number dt1, dt/2, cold)
#   volume_n_spread.png   rms(s_a/s̄ − 1) and rms(V_a/V̄ − 1) against t
#   volume_n_stability.png  KE/eq and ΔH/H at t = 200 against δv, entropy and number
#   volume_n_CT.png       C_T(k,τ)/C_T(k,0) at L = 1.2, S0 = 1e6 and 1e7, entropy and number
#   volume_n_D2.png       pair separation D²(a,t)/a² at L = 1.2, entropy and number
# Run:  julia --project examples/volume_n_analysis.jl
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, HDF5, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const S = joinpath(ENV["SCRATCH"], "sph")
const VN = joinpath(S, "volume_n");  const E2 = joinpath(S, "energy2");  const FR = joinpath(S, "frozen")
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
mean_(v) = sum(v) / length(v)
sem(v) = length(v) > 1 ? sqrt(sum(abs2, v .- mean_(v)) / (length(v) * (length(v) - 1))) : NaN
slope(t, y) = (tm = mean_(t); sum((t .- tm) .* (y .- mean_(y))) / sum(abs2, t .- tm))
files(d) = isdir(d) ? [joinpath(d, f) for f in readdir(d) if endswith(f, ".h5") && startswith(f, "equilibrium_")] : String[]
load(d) = filter(r -> get(r.meta, :complete, true), [read_equilibrium(f) for f in files(d)])
series(d, name) = [h5open(f, "r") do h; haskey(h, name) ? read(h[name]) : Float64[]; end for f in files(d)]

# ── fig 1: H and KE at L = 0.6 ──────────────────────────────────────────────
SETS1 = [("entropy, Δt", joinpath(E2, "dt1"), "C3", "-"), ("number, Δt", joinpath(VN, "n_dt1"), "C0", "-"),
         ("number, Δt/2", joinpath(VN, "n_dt0.5"), "C2", "-"), ("number, cold start", joinpath(VN, "n_cold"), "C1", "-")]
fig, axs = subplots(1, 3, figsize = (19, 5))
for (lab, d, c, ls) in SETS1
    rs = load(d);  isempty(rs) && continue
    nk = minimum(length(r.H) for r in rs);  t = rs[1].rec.t[1:nk];  H = mean_([r.H[1:nk] for r in rs])
    rate = [slope(t, r.H[1:nk]) for r in rs]
    axs[1].plot(t, H, color = c, lw = 1.2, label = @sprintf("%s (%d)", lab, length(rs)))
    axs[2].plot(t, H, color = c, lw = 1.2, label = lab)
    axs[3].plot(t, mean_([r.KE[1:nk] for r in rs]), color = c, lw = 0.6, label = lab)
    @printf("%-20s n = %d  dH/H dt = %+.2e ± %.1e   ΔH/H(end) %+.2e   KE/eq(2nd half) %.3f\n", lab, length(rs),
            mean_(rate), sem(rate), H[end], mean_([mean_(r.KE[nk÷2:nk]) for r in rs]))
end
axs[1].axhline(0, color = "k", lw = 0.5);  axs[1].set_xlabel("t");  axs[1].set_ylabel("ΔH/H");  axs[1].legend(fontsize = 8)
axs[1].set_title("L = 0.6, h/a = 2, δv = 0.043 (mean of runs)")
axs[2].axhline(0, color = "k", lw = 0.5);  axs[2].set_ylim(-2e-4, 6e-4);  axs[2].set_xlabel("t");  axs[2].set_ylabel("ΔH/H")
axs[2].set_title("the number volume, enlarged")
axs[3].axhline(1, color = "k", lw = 0.5);  axs[3].set_xlabel("t");  axs[3].set_ylabel("KE / equilibrium");  axs[3].set_ylim(0, 1.3)
axs[3].legend(fontsize = 8);  axs[3].set_title("kinetic energy")
tight_layout();  savefig(joinpath(FIGD, "volume_n_H.png"), dpi = 100)

# ── fig 2: spreads ───────────────────────────────────────────────────────────
SETS2 = [("L = 0.6, S0 = 1e6", joinpath(VN, "n_dt1"), "C0"), ("L = 0.6, cold start", joinpath(VN, "n_cold"), "C1"),
         ("L = 1.2, S0 = 1e6", joinpath(VN, "n_e6"), "C2"), ("L = 1.2, S0 = 1e7", joinpath(VN, "n_e7"), "C4")]
fig, axs = subplots(1, 2, figsize = (14, 5))
for (lab, d, c) in SETS2
    rs = load(d);  isempty(rs) && continue
    ss = series(d, "sspr");  vs = series(d, "Vspr");  nk = minimum(length.(ss));  t = rs[1].rec.t[1:nk]
    s0 = 1 / sqrt(rs[1].meta.S0 * rs[1].meta.V / 3)                     # √(3/s_a), the spread at fixed volume
    axs[1].plot(t, mean_([x[1:nk] for x in ss]), color = c, lw = 1.2, label = lab)
    axs[1].axhline(s0, color = c, lw = 0.8, ls = ":")
    axs[2].plot(t, mean_([x[1:nk] for x in vs]), color = c, lw = 1.2, label = lab)
    @printf("%-22s spread s: start %.4f end %.4f (√(3/s) = %.4f)   V: start %.4f end %.4f\n", lab,
            mean_([x[1] for x in ss]), mean_([x[nk] for x in ss]), s0, mean_([x[1] for x in vs]), mean_([x[nk] for x in vs]))
end
axs[1].set_xlabel("t");  axs[1].set_ylabel("rms(s_a/s̄ − 1)");  axs[1].legend(fontsize = 8)
axs[1].set_title("entropy per particle; dotted: √(3/s_a), the spread at fixed volume")
axs[2].set_xlabel("t");  axs[2].set_ylabel("rms(V_a/V̄ − 1)");  axs[2].set_title("particle volumes 1/n_a")
tight_layout();  savefig(joinpath(FIGD, "volume_n_spread.png"), dpi = 100)

# ── fig 3: stability scan ────────────────────────────────────────────────────
function scan(dir, run)
    ke = Float64[];  dh = Float64[];  δv = NaN
    for (root, _, fs) in walkdir(dir), f in fs
        endswith(f, ".out") || continue
        txt = read(joinpath(root, f), String);  occursin("run $run seed", txt) || continue
        for l in split(txt, '\n')
            m = match(r"thermal \|δv\| per component=([0-9.]+)", l);  m === nothing || (δv = parse(Float64, m[1]))
            m = match(r"^full\s+seed=.*KE/eq\(end\)=([-0-9.e]+)\s+ΔH/H=([-0-9.e+]+)", l)
            m === nothing || (push!(ke, parse(Float64, m[1])); push!(dh, parse(Float64, m[2])))
        end
    end
    (; δv, ke, dh)
end
fig, axs = subplots(1, 2, figsize = (14, 5))
for (lab, dir, runs, c, mk) in (("entropy volume", E2, ("ref", "s3e5", "s1e5"), "C3", "s"),
                                ("number volume", VN, ("n_ref", "n_s3e5", "n_s1e5"), "C0", "o"))
    sc = [scan(dir, r) for r in runs];  sc = filter(s -> !isempty(s.ke), sc);  isempty(sc) && continue
    x = [s.δv for s in sc]
    axs[1].errorbar(x, [mean_(s.ke) for s in sc], yerr = [sem(s.ke) for s in sc], fmt = mk * "-", color = c, ms = 8, label = lab)
    axs[2].semilogx(x, [mean_(s.dh) for s in sc], mk * "-", color = c, ms = 8, label = lab)
    for (r, s) in zip(runs, sc)
        @printf("%-14s %-7s δv = %.3f  n = %2d  KE/eq = %.3f ± %.3f  ΔH/H = %+.2e ± %.1e\n", lab, r, s.δv, length(s.ke),
                mean_(s.ke), sem(s.ke), mean_(s.dh), sem(s.dh))
    end
end
axs[1].axhline(1, color = "k", lw = 0.5);  axs[1].set_xscale("log");  axs[1].set_xlabel("δv");  axs[1].set_ylabel("KE / equilibrium at t = 200")
axs[1].legend(fontsize = 9);  axs[1].set_title("stability scan, L = 0.6, h/a = 2, 16 runs each")
axs[2].axhline(0, color = "k", lw = 0.5);  axs[2].set_xlabel("δv");  axs[2].set_ylabel("ΔH/H at t = 200");  axs[2].legend(fontsize = 9)
axs[2].set_yscale("symlog", linthresh = 1e-4)
tight_layout();  savefig(joinpath(FIGD, "volume_n_stability.png"), dpi = 100)

# ── fig 4, 5: C_T and pair separation at L = 1.2 ─────────────────────────────
SETS4 = [("entropy, S0 = 1e6", joinpath(FR, "e_s1e6"), "C3", "-"), ("number, S0 = 1e6", joinpath(VN, "n_e6"), "C0", "-"),
         ("entropy, S0 = 1e7", joinpath(FR, "e_s1e7"), "C1", "--"), ("number, S0 = 1e7", joinpath(VN, "n_e7"), "C2", "--")]
res = []
for (lab, d, c, ls) in SETS4
    rs = load(d);  isempty(rs) && continue
    m = rs[1].meta;  ν = m.etas / m.Trel
    per = [shell_correlators(r.rec, :MT; t0 = 2.0, maxlag_t = 60.0) for r in rs]
    ks, τ = per[1][1], collect(per[1][2]);  C = mean_([p[3] for p in per]);  n0 = m.N * m.V * m.w * m.T
    fits = [fit_oscillator(τ, C[j, :] ./ C[j, 1]) for j in 1:6]
    nt = minimum(length(r.part.t) for r in rs);  tp = rs[1].part.t[1:nt]
    D2 = mean_([r.part.D2[1, 1:nt] for r in rs]);  it = findall(>=(50.0), tp)
    Drel = mean_([slope(tp[it], r.part.D2[1, it]) / 4 for r in rs])
    push!(res, (; lab, c, ls, m, ν, ks, τ, C, n0, fits, tp, D2, Drel, n = length(rs)))
    @printf("%-17s n = %2d  C_T(k,0)/NVwT: %s\n   c_t: %s  2γ/νk²: %s  D_rel(a) = %.2e  KE/eq %.3f  ΔH/H %+.2e\n", lab, length(rs),
            join([@sprintf("%.3f", C[j, 1] / n0) for j in 1:6], " "),
            join([@sprintf("%.4f", f[3] / ks[j]) for (j, f) in enumerate(fits)], " "),
            join([@sprintf("%.2f", 2f[2] / (ν * ks[j]^2)) for (j, f) in enumerate(fits)], " "), Drel,
            mean_([mean_(r.KE[end÷2:end]) for r in rs]), mean_([r.H[end] for r in rs]))
end
fig, axs = subplots(1, 3, figsize = (18, 5))
for (i, j) in enumerate((1, 3, 6)), r in res
    axs[i].plot(r.τ, r.C[j, :] ./ r.C[j, 1], color = r.c, ls = r.ls, lw = 1.4, label = r.lab)
    axs[i].plot(r.τ, exp.(-r.ν * r.ks[j]^2 .* r.τ), color = "k", lw = 0.6, ls = ":")
    axs[i].set_title(@sprintf("C_T(k,τ)/C_T(k,0),  kh = %.3f  (dotted: e^{−νk²τ})", r.ks[j] * r.m.h))
    axs[i].axhline(0, color = "k", lw = 0.5);  axs[i].set_xlabel("τ");  i == 1 && axs[i].legend(fontsize = 8)
end
tight_layout();  savefig(joinpath(FIGD, "volume_n_CT.png"), dpi = 100)
figure(figsize = (8, 6))
for r in res
    loglog(r.tp, r.D2 ./ r.m.a^2, color = r.c, ls = r.ls, lw = 1.4, label = @sprintf("%s: D_rel(a) = %.1e", r.lab, r.Drel))
end
xlabel("t");  ylabel("D²(r = a, t)/a²");  title("separation of nearest neighbours, L = 1.2");  legend(fontsize = 8)
tight_layout();  savefig(joinpath(FIGD, "volume_n_D2.png"), dpi = 110)
println("figures written to $FIGD")
