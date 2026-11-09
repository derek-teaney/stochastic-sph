# rigidity_analysis.jl — the rigidity campaign (nersc/campaign_rigidity.txt).
# For each run, in the smallest k shells:
#   (1) dynamics: fit C_T(k,τ)/C_T(k,0) to the damped oscillator (fit_oscillator), giving
#       c_t = ω0/k and 2γ/νk²;
#   (2) statics, independently, from the recorded displacements: an elastic transverse
#       mode has ⟨|u̇_T|²⟩ = ω0² ⟨|u_T|²⟩, and M_k ≈ wV u̇_k, so
#           c_t² = ⟨|M_T(k)|²⟩ / ( (wV)² k² ⟨|u_T(k)|²⟩ ),
#       while in a fluid ⟨|u_T|²⟩ grows without bound (and this c_t → 0);
#   (3) ⟨|u_T(k,t)|²⟩ against t: saturating (solid) or growing ∝ t (fluid).
# Ideal-SPH runs (nv_*): KE/eq(t).
#
# Run:  julia --project examples/rigidity_analysis.jl [dir]   (default $SCRATCH/sph/rigidity)
#   → docs/figs/rigidity_*.png
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, Statistics
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "rigidity")
const RUNS = ["ha2", "ha2_wc4", "ha2.5", "ha3", "ha2_noham"]
const NSH = 4;  const TBURN = 2.0

function load(run)
    d = joinpath(DIR, run);  isdir(d) || return []
    fs = filter(f -> endswith(f, ".h5") && startswith(f, "equilibrium_"), readdir(d))
    filter(r -> !haskey(r.meta, :complete) || r.meta.complete, [read_equilibrium(joinpath(d, f)) for f in fs])
end

shellidx(ks) = (kn = round.(norm.(ks); digits = 8); sh = sort(unique(kn)); (sh, [findall(==(s), kn) for s in sh]))

res = Dict{String,Any}()
for run in RUNS
    rs = load(run);  isempty(rs) && (println("$run: no complete runs yet"); continue)
    m = rs[1].meta;  ν = m.etas / m.Trel;  wV = m.w * m.V;  n0 = m.N * m.V * m.w * m.T
    per = [shell_correlators(r.rec, :MT; t0 = TBURN, maxlag_t = 60.0) for r in rs]
    ks, τ = per[1][1], collect(per[1][2])
    C = mean(cat([p[3] for p in per]...; dims = 3); dims = 3)[:, :, 1]
    fits = [fit_oscillator(τ, C[j, :] ./ C[j, 1]) for j in 1:NSH]
    sh, idx = shellidx(rs[1].rec.ks)
    i0 = findfirst(>=(TBURN), rs[1].rec.t)
    hasu = !isempty(rs[1].rec.UT)
    u2t = hasu ? [mean(mean(abs2, r.rec.UT[n][idx[j]]) for r in rs) for j in 1:NSH, n in eachindex(rs[1].rec.t)] : nothing
    ctstat = hasu ? [sqrt(mean(mean(abs2, r.rec.MT[n][idx[j]]) for r in rs for n in i0:length(r.rec.t)) /
                          (wV^2 * sh[j]^2 * mean(u2t[j, i0:end]))) for j in 1:NSH] : fill(NaN, NSH)
    res[run] = (; m, ν, ks = ks[1:NSH], τ, C = C[1:NSH, :] ./ C[1:NSH, 1], fits, ctstat, u2t, t = rs[1].rec.t, n = length(rs))
    @printf("== %-10s %2d runs  h/a = %.1f  N = %d  kernel %s\n", run, length(rs), m.ha, m.N, get(m, :kernel, "wc2"))
    for j in 1:NSH
        A, γ, ω0 = fits[j]
        @printf("   kh = %.3f  c_t (dynamics) = %.4f  c_t (statics) = %.4f  2γ/νk² = %.3f  A = %.3f\n",
                ks[j] * m.h, ω0 / ks[j], ctstat[j], 2γ / (ν * ks[j]^2), A)
    end
end

isempty(res) && exit()
figd = joinpath(@__DIR__, "..", "docs", "figs")
cols = Dict("ha2" => "C0", "ha2_wc4" => "C1", "ha2.5" => "C2", "ha3" => "C3", "ha2_noham" => "k")
fig, axs = subplots(1, 3, figsize = (18, 5))
for run in RUNS
    haskey(res, run) || continue;  r = res[run];  c = cols[run]
    axs[1].plot(r.τ, r.C[1, :], color = c, label = @sprintf("%s (%d)", run, r.n))
    axs[1].plot(r.τ, exp.(-r.ν * r.ks[1]^2 .* r.τ), color = c, lw = 0.6, ls = ":")
    kh = r.ks .* r.m.h
    axs[2].plot(kh, [f[3] for f in r.fits] ./ r.ks, "o-", color = c, label = run * " dynamics")
    axs[2].plot(kh, r.ctstat, "s--", color = c, mfc = "none", label = run * " statics")
    r.u2t === nothing || axs[3].plot(r.t, r.u2t[1, :] ./ r.u2t[1, end], color = c, label = run)
end
axs[1].axhline(0, color = "k", lw = 0.5);  axs[1].set_xlabel("τ");  axs[1].set_ylabel("C_T(k,τ)/C_T(k,0)")
axs[1].set_title("smallest shell (kh = 0.26); dotted e^{−νk²τ}");  axs[1].legend(fontsize = 8)
axs[2].set_xlabel("k h");  axs[2].set_ylabel("c_t");  axs[2].set_title("transverse sound speed: fit of C_T (o) and displacement statics (□)")
axs[2].set_ylim(bottom = 0);  axs[2].legend(fontsize = 7)
axs[3].set_xlabel("t");  axs[3].set_ylabel("⟨|u_T|²⟩ / final");  axs[3].set_title("transverse displacement, smallest shell")
axs[3].legend(fontsize = 8)
tight_layout();  savefig(joinpath(figd, "rigidity_ct.png"), dpi = 110)

# ideal SPH: KE/eq against t
nv = [r for r in ("nv_ha2", "nv_ha2_wc4", "nv_ha2.5", "nv_ha3") if !isempty(load(r))]
if !isempty(nv)
    figure(figsize = (7, 5))
    for (i, run) in enumerate(nv)
        rs = load(run);  t = rs[1].rec.t
        plot(t, mean(reduce(hcat, [r.KE for r in rs]); dims = 2)[:], color = "C$(i-1)", label = run)
    end
    axhline(1, color = "k", lw = 0.5);  xlabel("t");  ylabel("KE / equilibrium");  legend()
    title("ideal SPH only (no viscous step), thermal start, box 0.6")
    tight_layout();  savefig(joinpath(figd, "rigidity_idealsph.png"), dpi = 110)
end
