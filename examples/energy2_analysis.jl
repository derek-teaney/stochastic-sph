# energy2_analysis.jl — campaign `energy2` (nersc/campaign_energy2_{small,large}.txt), all with
# SPH_HAMILTONIAN = energy.  Figures for docs/campaigns.tex:
#   energy2_dt.png        ΔH/H(t) at L = 0.6 for Δt = 1, 1/2, 1/4 × default and with positions
#                         frozen (noham); right: loss rate against Δt
#   energy2_stability.png the stability scan (stochrep_hamiltonian.jl, L = 0.6, t = 200): KE/eq
#                         and −ΔH/H at the end against δv, full (λ = 1) and det (λ = ∞)
#   energy2_ha.png        C_T(k,τ)/C_T(k,0) at L = 1.2 for h/a = 2, 2.5, 3 in three shells, and the
#                         fitted c_t against kh
#   energy2_L24.png       C_T at L = 2.4 (h/a = 2) in the six smallest shells, with e^{−νk²τ}
# Run:  julia --project examples/energy2_analysis.jl [dir]   (default $SCRATCH/sph/energy2)
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, HDF5, Serialization
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
const S = ENV["SCRATCH"]
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(S, "sph", "energy2")
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
const TBURN = 2.0;  const TMAX = 60.0
mean_(v) = sum(v) / length(v)
sem(v) = length(v) > 1 ? sqrt(sum(abs2, v .- mean_(v)) / (length(v) * (length(v) - 1))) : NaN
slope(t, y) = (tm = mean_(t); sum((t .- tm) .* (y .- mean_(y))) / sum(abs2, t .- tm))

function load(d)
    isdir(d) || return []
    fs = filter(f -> endswith(f, ".h5") && startswith(f, "equilibrium_"), readdir(d))
    filter(r -> get(r.meta, :complete, true), [read_equilibrium(joinpath(d, f)) for f in fs])
end

# ---------------- 1. loss of H against Δt, and with positions frozen ----------------
DT = [("dt1", "Δt = default", "C0", 1.0), ("dt0.5", "Δt/2", "C2", 0.5), ("dt0.25", "Δt/4", "C1", 0.25),
      ("noham", "Δt = default, positions frozen (no ideal step)", "C3", 1.0)]
fig, axs = subplots(1, 2, figsize = (14, 5))
rates = Dict{String,Tuple{Float64,Float64}}()
for (run, lab, c, f) in DT
    rs = load(joinpath(DIR, run));  isempty(rs) && continue
    nk = minimum(length(r.H) for r in rs);  t = rs[1].rec.t[1:nk]
    H = [r.H[1:nk] for r in rs];  Hm = mean_(H)
    rate = [slope(t, h) for h in H];  rates[run] = (mean_(rate), sem(rate))
    axs[1].plot(t, Hm, color = c, lw = 1.5, label = @sprintf("%s (%d runs)", lab, length(rs)))
    @printf("%-7s %d runs  dH/H dt = %.3e ± %.1e   KE/eq %.3f   dt = %.3e\n", run, length(rs),
            rates[run]..., mean_([mean_(r.KE[nk÷2:nk]) for r in rs]), f)
end
axs[1].axhline(0, color = "k", lw = 0.5);  axs[1].set_xlabel("t");  axs[1].set_ylabel("ΔH/H")
axs[1].set_title("L = 0.6, h/a = 2, S0 = 1e6 (δv = 0.043)");  axs[1].legend(fontsize = 8)
fs = [f for (run, _, _, f) in DT[1:3] if haskey(rates, run)]
rr = [-rates[run][1] for (run, _, _, _) in DT[1:3] if haskey(rates, run)]
ee = [rates[run][2] for (run, _, _, _) in DT[1:3] if haskey(rates, run)]
axs[2].errorbar(fs, rr, yerr = ee, fmt = "o", color = "C0", ms = 8, label = "full step")
haskey(rates, "noham") && axs[2].errorbar([1.0], [-rates["noham"][1]], yerr = [rates["noham"][2]], fmt = "s",
                                          color = "C3", ms = 8, mfc = "none", label = "positions frozen")
axs[2].plot(fs, rr[1] .* fs, "k:", lw = 1, label = "∝ Δt");  axs[2].plot(fs, rr[1] .* fs .^ 2, "k--", lw = 1, label = "∝ Δt²")
axs[2].set_xscale("log");  axs[2].set_yscale("log");  axs[2].set_xlabel("Δt / default");  axs[2].set_ylabel("−(1/H) dH/dt")
axs[2].set_title("loss rate (slope of each run, mean ± s.e.)");  axs[2].legend(fontsize = 8)
tight_layout();  savefig(joinpath(FIGD, "energy2_dt.png"), dpi = 100)

# ---------------- 2. stability scan (text output of stochrep_hamiltonian.jl) ----------------
function scan(run)
    out = Dict("full" => (Float64[], Float64[]), "det" => (Float64[], Float64[]));  δv = NaN
    for f in filter(f -> startswith(f, "slurm-") && endswith(f, ".out"), readdir(DIR))
        txt = read(joinpath(DIR, f), String);  occursin("run $run seed", txt) || continue
        for l in split(txt, '\n')
            m = match(r"thermal \|δv\| per component=([0-9.]+)", l);  m === nothing || (δv = parse(Float64, m[1]))
            m = match(r"^(full|det)\s+seed=.*KE/eq\(end\)=([-0-9.e]+)\s+ΔH/H=([-0-9.e+]+)", l)
            m === nothing && continue
            push!(out[m[1]][1], parse(Float64, m[2]));  push!(out[m[1]][2], parse(Float64, m[3]))
        end
    end
    (; δv, out)
end
SC = [(r, scan(r)) for r in ("ref", "s3e5", "s1e5")]
fig, axs = subplots(1, 2, figsize = (14, 5))
for (mode, mk, c, lab) in (("full", "o", "C0", "full step"), ("det", "s", "C3", "det: λ = ∞ (deterministic residual, stress noise only)"))
    x = [s.δv for (_, s) in SC if !isempty(s.out[mode][1])]
    k = [mean_(s.out[mode][1]) for (_, s) in SC if !isempty(s.out[mode][1])]
    ke = [sem(s.out[mode][1]) for (_, s) in SC if !isempty(s.out[mode][1])]
    h = [-mean_(s.out[mode][2]) for (_, s) in SC if !isempty(s.out[mode][2])]
    axs[1].errorbar(x, k, yerr = ke, fmt = mk * "-", color = c, ms = 8, label = lab)
    axs[2].loglog(x, h, mk * "-", color = c, ms = 8, label = lab)
    for (r, s) in SC
        @printf("%-5s δv = %.3f  %-4s  n = %2d  KE/eq = %.3f ± %.3f  ΔH/H = %+.2e (min %+.2e)\n", r, s.δv, mode,
                length(s.out[mode][1]), mean_(s.out[mode][1]), sem(s.out[mode][1]), mean_(s.out[mode][2]), minimum(s.out[mode][2]))
    end
end
axs[1].axhline(1, color = "k", lw = 0.5);  axs[1].set_xscale("log");  axs[1].set_xlabel("δv");  axs[1].set_ylabel("KE / equilibrium at t = 200")
axs[1].set_title("stability scan, L = 0.6, h/a = 2, 16 seeds each");  axs[1].legend(fontsize = 8)
xs = [0.03, 0.15];  h0 = -mean_(SC[1][2].out["full"][2]);  x0 = SC[1][2].δv
axs[2].loglog(xs, h0 .* (xs ./ x0) .^ 4, "k:", lw = 1, label = "∝ δv⁴")
axs[2].set_xlabel("δv");  axs[2].set_ylabel("−ΔH/H at t = 200");  axs[2].legend(fontsize = 8)
tight_layout();  savefig(joinpath(FIGD, "energy2_stability.png"), dpi = 100)

# ---------------- 3. h/a dependence at L = 1.2 ----------------
function corrs(rs)
    m = rs[1].meta;  n0 = m.N * m.V * m.w * m.T
    per = [shell_correlators(r.rec, :MT; t0 = TBURN, maxlag_t = TMAX) for r in rs]
    ks, τ = per[1][1], collect(per[1][2])
    Cs = [p[3] ./ n0 for p in per];  C = mean_(Cs)
    err = sqrt.(mean_([(c .- C) .^ 2 for c in Cs]) ./ (length(Cs) - 1))
    fits = [fit_oscillator(τ, C[j, :] ./ C[j, 1]) for j in 1:min(6, length(ks))]
    (; ks, τ, C, err, m, ν = m.etas / m.Trel, n = length(rs), fits,
       KE = mean_([mean_(r.KE[end÷2:end]) for r in rs]), H = mean_([r.H[end] for r in rs]))
end
HA = [("h/a = 2", joinpath(S, "sph", "energystep", "ha2_energy"), "C0"),
      ("h/a = 2.5", joinpath(DIR, "ha2.5_energy"), "C2"), ("h/a = 3", joinpath(DIR, "ha3_energy"), "C1")]
hres = [(lab, corrs(rs), c) for (lab, d, c) in HA for rs in (load(d),) if !isempty(rs)]
fig, axs = subplots(1, 4, figsize = (20, 4.6))
for (i, j) in enumerate((1, 3, 6))
    for (lab, r, c) in hres
        y = r.C[j, :] ./ r.C[j, 1];  e = r.err[j, :] ./ r.C[j, 1]
        axs[i].fill_between(r.τ, y .- e, y .+ e, color = c, alpha = 0.25, lw = 0)
        axs[i].plot(r.τ, y, color = c, lw = 1.5, label = @sprintf("%s, kh = %.2f (%d runs)", lab, r.ks[j] * r.m.h, r.n))
        axs[i].plot(r.τ, exp.(-r.ν * r.ks[j]^2 .* r.τ), color = c, lw = 0.8, ls = ":")
    end
    axs[i].axhline(0, color = "k", lw = 0.5);  axs[i].set_xlabel("τ");  axs[i].legend(fontsize = 7)
    axs[i].set_title(@sprintf("C_T(k,τ)/C_T(k,0), shell %d (kL/2π fixed)", j))
end
for (lab, r, c) in hres
    ct = [f[3] / r.ks[j] for (j, f) in enumerate(r.fits)];  kh = r.ks[1:length(ct)] .* r.m.h
    axs[4].plot(kh, ct, "o", color = c, ms = 7, mfc = "none", label = lab)
    @printf("%-9s n = %2d  δv = %.4f  KE/eq %.3f  ΔH/H %+.2e  c_t: %s  2γ/νk²: %s\n", lab, r.n,
            sqrt(r.m.T / (r.m.w * r.m.V)), r.KE, r.H, join([@sprintf("%.3f", x) for x in ct], " "),
            join([@sprintf("%.2f", 2f[2] / (r.ν * r.ks[j]^2)) for (j, f) in enumerate(r.fits)], " "))
end
axs[4].axhline(0.034, color = "k", ls = "--", lw = 0.8, label = "static c_t, h/a = 2, T = 0")
axs[4].set_xlabel("kh");  axs[4].set_ylabel("fitted c_t");  axs[4].set_ylim(-0.002, 0.04);  axs[4].legend(fontsize = 7)
axs[4].set_title("c_t from the damped-oscillator fit")
tight_layout();  savefig(joinpath(FIGD, "energy2_ha.png"), dpi = 100)

# ---------------- 4. L = 2.4 ----------------
rs = load(joinpath(DIR, "L2.4_energy"))
if !isempty(rs)
    r = corrs(rs)
    fig, axs = subplots(2, 3, figsize = (15, 8.5))
    for j in 1:6
        ax = axs[(j - 1) ÷ 3 + 1, (j - 1) % 3 + 1]
        ax.fill_between(r.τ, r.C[j, :] .- r.err[j, :], r.C[j, :] .+ r.err[j, :], color = "C0", alpha = 0.3, lw = 0)
        ax.plot(r.τ, r.C[j, :], color = "C0", lw = 1.5, label = @sprintf("energy, L = 2.4 (%d runs)", r.n))
        ax.plot(r.τ, exp.(-r.ν * r.ks[j]^2 .* r.τ), "k:", lw = 1.2, label = "fluid  e^{−νk²τ}")
        ax.axhline(0, color = "k", lw = 0.5);  ax.set_xlabel("τ")
        ax.set_title(@sprintf("C_T/(NVwT),  kh = %.3f", r.ks[j] * r.m.h));  j == 1 && ax.legend(fontsize = 9)
    end
    tight_layout();  savefig(joinpath(FIGD, "energy2_L24.png"), dpi = 100)
    @printf("L2.4  n = %d  KE/eq %.3f  ΔH/H %+.2e\n", r.n, r.KE, r.H)
    for (j, f) in enumerate(r.fits)
        @printf("   kh = %.3f  C_T(0) = %.3f ± %.3f  c_t = %.4f  2γ/νk² = %.2f\n", r.ks[j] * r.m.h, r.C[j, 1], r.err[j, 1],
                f[3] / r.ks[j], 2f[2] / (r.ν * r.ks[j]^2))
    end
else
    println("L2.4: no complete runs yet")
end
println("figures written to $FIGD")
