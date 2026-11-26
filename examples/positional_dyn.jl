# positional_dyn.jl — campaign `hastings_positional` (nersc/campaign_hastings_positional.txt; docs/sscheme_plan.tex, level 2, item K and the
# approach to equilibrium): the dynamics with the positional counterterm (SPH_POSITIONAL=1), split+ω (pp) and the measure form+ω (pm), at
# c_s² = 1/2, s_a = 16, 54, at rest.
#   julia --project examples/positional_dyn.jl <campaign dir>     → table; docs/figs/positional_dyn_spectra.png, positional_dyn_approach.png
# 1. Dynamics against its own HMC: the equal-time spectra |M_L,k|²/N, |M_T,k|²/N, |E_k − ⟨E_k⟩|²/N in bands of kh (PD_DKH, default 0.26),
#    dynamics (hs, after the burn-in PD_BURN of each run's records, default 0.2) / HMC chains (hq, after PD_HBURN updates, default 20);
#    KE/eq and the drift of H over the run.
# 2. The approach to equilibrium from the lattice (hl, no HMC start): the mean over the runs of KE/eq(t), |ψ6|(t) and the spectra in windows
#    of t, against the HMC values (dashed).
using Printf, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "plotsetup.jl"))
const DIR = ARGS[1]
const DKH = parse(Float64, get(ENV, "PD_DKH", "0.26"))
const BURN = parse(Float64, get(ENV, "PD_BURN", "0.2"));  const HBURN = parse(Int, get(ENV, "PD_HBURN", "20"))
const FIGS = get(ENV, "SR_FIGDIR", joinpath(@__DIR__, "..", "docs", "figs"))
files(set) = (d = joinpath(DIR, set);  isdir(d) ? [joinpath(d, f) for f in readdir(d) if endswith(f, ".h5")] : String[])

# band-averaged equal-time spectra of the records i0:i1 of one run
function spectra(r, i0, i1)
    R = r.rec;  N = r.meta.N;  h = r.meta.h
    kh = [norm(k) * h for k in R.ks];  bands = round.(kh ./ DKH) .* DKH
    out = Dict{Float64,Vector{Float64}}()
    for b in unique(bands)
        idx = findall(==(b), bands)
        L = mean(mean(abs2.(R.ML[i][idx])) for i in i0:i1) / N
        T = mean(mean(abs2.(R.MT[i][idx])) for i in i0:i1) / N
        E = isempty(R.E) ? NaN : (Em = mean(R.E[i][idx] for i in i0:i1);  mean(mean(abs2.(R.E[i][idx] .- Em)) for i in i0:i1) / N)
        out[b] = [L, T, E]
    end
    return out
end
meanerr(xs) = (mean(xs), length(xs) > 1 ? std(xs) / sqrt(length(xs)) : NaN)
function set_spectra(set, hmc)
    fs = files(set);  isempty(fs) && return nothing
    rs = map(fs) do f
        r = read_equilibrium(f);  nt = length(r.rec.t)
        spectra(r, hmc ? min(HBURN, nt - 2) + 1 : max(1, round(Int, BURN * nt)) + 1, nt)
    end
    bands = sort(collect(keys(rs[1])))
    return (bands = bands, m = Dict(b => mean(r[b] for r in rs) for b in bands),
            e = Dict(b => std([r[b] for r in rs]) ./ sqrt(length(rs)) for b in bands), n = length(rs))
end
function set_scalars(set, hmc)
    fs = files(set);  isempty(fs) && return nothing
    ke = Float64[];  dh = Float64[]
    for f in fs
        r = read_equilibrium(f);  n = length(r.KE)
        push!(ke, mean(r.KE[(hmc ? min(HBURN, n - 2) : max(1, round(Int, BURN * n))) + 1:end]))
        hmc || push!(dh, r.H[end])
    end
    return (ke = meanerr(ke), dh = isempty(dh) ? (NaN, NaN) : meanerr(dh), dhmax = isempty(dh) ? NaN : maximum(abs, dh), n = length(fs))
end

const SCH = [("pp", "split+ω"), ("pm", "measure+ω")]
const SAS = (16, 54)
println("1. the dynamics against its own HMC (ratios dynamics/HMC per band of kh; KE/eq; ΔH/H at the end of the runs)")
fig, axs = subplots(length(SAS), 3, figsize = (14, 4.2 * length(SAS)), squeeze = false)
for (si, sa) in enumerate(SAS), (ci, (sch, lab)) in enumerate(SCH)
    dyn = set_spectra("hs$(sa)_$(sch)_v0", false);  ref = set_spectra("hq$(sa)_$(sch)_v0", true)
    (dyn === nothing || ref === nothing) && (println("  s_a = $sa $lab: missing");  continue)
    sd = set_scalars("hs$(sa)_$(sch)_v0", false);  sq = set_scalars("hq$(sa)_$(sch)_v0", true)
    @printf("  s_a = %d, %s: %d runs / %d chains.  KE/eq %.4f±%.4f (1 in equilibrium);  ΔH/H at t_end %+.1e±%.1e (largest |ΔH/H| %.1e)\n",
            sa, lab, dyn.n, ref.n, sd.ke..., sd.dh..., sd.dhmax)
    bands = [b for b in dyn.bands if b in ref.bands && b > 0]
    for (vi, var) in enumerate(("M_L", "M_T", "E"))
        rat = [dyn.m[b][vi] / ref.m[b][vi] for b in bands]
        err = [abs(rat[j]) * sqrt((dyn.e[b][vi] / dyn.m[b][vi])^2 + (ref.e[b][vi] / ref.m[b][vi])^2) for (j, b) in enumerate(bands)]
        @printf("    %-3s dyn/HMC: %s\n", var, join([@sprintf("kh=%.2f %.3f±%.3f", b, rat[j], err[j]) for (j, b) in enumerate(bands) if b <= 2.1], "  "))
        ax = axs[si, vi]
        ax.errorbar(bands .+ (ci - 1.5) * 0.03, rat, err, fmt = ci == 1 ? "o-" : "s--", ms = 4, label = lab)
        ax.axhline(1, color = "k", lw = 0.5);  ax.set_xlabel("kh");  ax.set_title("s_a = $sa: $var, dynamics / own HMC", fontsize = 10)
        ax.set_ylim(0.85, 1.15)
    end
end
axs[1, 1].legend(fontsize = 8)
fig.suptitle("hastings_positional (SPH_POSITIONAL=1, a₂ = 0), c_s² = 1/2, L = 0.6, h/a = 2, at rest: equal-time spectra of the dynamics (t = 100, " *
             "after 20%) over those of hybrid Monte Carlo of the same Hamiltonian", fontsize = 9)
fig.tight_layout();  fig.savefig(joinpath(FIGS, "positional_dyn_spectra.png"), dpi = 120);  plt.close(fig)

println("\n2. the approach to equilibrium from the lattice (hl): |ψ6|(t) and the spectra in windows of t over HMC")
println("   (KE/eq of the lattice-start runs is computed with the nominal volume, not V_a: not comparable, not shown)")
const WIN = [(0.0, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, 100.0)]
fig, axs = subplots(length(SAS), 3, figsize = (14, 4.2 * length(SAS)), squeeze = false)
for (si, sa) in enumerate(SAS), (ci, (sch, lab)) in enumerate(SCH)
    fs = files("hl$(sa)_$(sch)_v0");  isempty(fs) && (println("  s_a = $sa $lab: no hl runs");  continue)
    rs = [read_equilibrium(f) for f in fs]
    ref = set_spectra("hq$(sa)_$(sch)_v0", true);  ref === nothing && continue
    rq = [read_equilibrium(f) for f in files("hq$(sa)_$(sch)_v0")]
    psq = mean(mean(r.part.psi6[min(HBURN, end - 1):end]) for r in rq)
    np_ = minimum(length(r.part.psi6) for r in rs);  tp = rs[1].part.t[1:np_]
    ps = [mean(r.part.psi6[i] for r in rs) for i in 1:np_]
    ax = axs[si, 1];  ax.plot(vcat(1e-3, tp), vcat(1.0, ps), ci == 1 ? "-" : "--", label = lab);  ax.axhline(psq, color = "C$(ci-1)", ls = ":", lw = 0.8)
    ax.set_xscale("log");  ax.set_yscale("log");  ax.set_xlabel("t");  ax.set_title("s_a = $sa: global bond order |ψ6| (1 at t = 0; dotted: HMC)", fontsize = 9)
    bands = [b for b in ref.bands if 0 < b <= 1.6];  b0 = bands[1]
    tw = [sqrt(max(ta, 0.01) * tb) for (ta, tb) in WIN]
    rows = Dict{String,Vector{Float64}}()
    for (vi, var) in enumerate(("M_L", "M_T", "E"))
        low = Float64[];  avg = Float64[];  err = Float64[]
        for (ta, tb) in WIN
            sps = [spectra(r, first(idx), last(idx)) for r in rs for idx in (findall(x -> ta <= x < tb, r.rec.t),) if !isempty(idx)]
            push!(low, mean(sp[b0][vi] for sp in sps) / ref.m[b0][vi])
            per = [mean(sp[b][vi] / ref.m[b][vi] for b in bands) for sp in sps]        # each run's band average / HMC
            push!(avg, mean(per));  push!(err, std(per) / sqrt(length(per)))
        end
        rows[var * " kh=" * @sprintf("%.2f", b0)] = low;  rows[var * " kh≤1.6"] = avg
        ax = axs[si, vi == 3 ? 3 : 2]
        ax.errorbar(tw .* (1 + 0.04 * (ci - 1.5)), avg, err, fmt = ci == 1 ? "o-" : "s--", ms = 4, color = "C$(2(ci-1) + (vi == 2 ? 1 : 0))",
                    label = "$lab, $var")
    end
    for k in (2, 3);  axs[si, k].axhline(1, color = "k", lw = 0.5);  axs[si, k].set_xscale("log");  axs[si, k].set_xlabel("t (window centre)");  end
    axs[si, 2].set_title("s_a = $sa: momentum spectra / HMC, mean over kh ≤ 1.6", fontsize = 9)
    axs[si, 3].set_title("s_a = $sa: energy spectrum / HMC, mean over kh ≤ 1.6", fontsize = 9)
    @printf("  s_a = %d, %s: %d runs; |ψ6| at t = 0.03, 0.06, 0.2, 1, 100: %s (HMC %.3f)\n", sa, lab, length(rs),
            join([@sprintf("%.3f", ps[argmin(abs.(tp .- tt))]) for tt in (0.03, 0.06, 0.2, 1.0, 100.0)], " "), psq)
    println("     windows ", join(["[$(a),$(b)]" for (a, b) in WIN], " "))
    for k in sort(collect(keys(rows)));  @printf("     %-12s %s\n", k, join([@sprintf("%.3f", x) for x in rows[k]], " "));  end
end
for si in 1:length(SAS), k in 1:3;  axs[si, k].legend(fontsize = 7);  end
fig.suptitle("hastings_positional: the approach to equilibrium from the hexagonal lattice (hl, 8 runs, no HMC start), against hybrid Monte Carlo " *
             "of the same Hamiltonian", fontsize = 9)
fig.tight_layout();  fig.savefig(joinpath(FIGS, "positional_dyn_approach.png"), dpi = 120);  plt.close(fig)
println("\nwritten ", joinpath(FIGS, "positional_dyn_spectra.png"), ", ", joinpath(FIGS, "positional_dyn_approach.png"))
