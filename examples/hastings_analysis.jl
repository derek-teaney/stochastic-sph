# hastings_analysis.jl — campaign `hastings` (nersc/campaign_hastings.txt; docs/inverse_moment.tex, stage 2): the equal-time spectra of
# the dynamics, moment without (hs<sa>_nh_v<v>) and with (hs<sa>_h_v<v>) the Hastings correction, against hybrid Monte Carlo chains of
# the same model and measure (hq<sa>_v<v>), at the same s_a, box and fluid velocity.
#   julia --project examples/hastings_analysis.jl <campaign dir>        → table, hastings_spectra.png in SR_FIGDIR (default examples/)
# For each set: the mean over records and runs of |M_L,k|², |M_T,k|² and |E_k − ⟨E_k⟩|² divided by N, in bands of kh (HA_DKH, default 0.26),
# after a burn-in (HA_BURN, the fraction of each run's records dropped, default 0.2; for the HMC chains also HA_HBURN updates, default 20);
# the error from the spread between runs.  Reported: dynamics / HMC per band, and for the stochastic step alone (hs<sa>_{nh,h}_alone_v<v>)
# the same ratio (the alone runs keep the positions of their HMC start, so HMC is their reference too only up to the positions).
using Printf, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "plotsetup.jl"))
const DIR = ARGS[1]
const DKH = parse(Float64, get(ENV, "HA_DKH", "0.26"))
const BURN = parse(Float64, get(ENV, "HA_BURN", "0.2"));  const HBURN = parse(Int, get(ENV, "HA_HBURN", "20"))

files(set) = (d = joinpath(DIR, set);  isdir(d) ? [joinpath(d, f) for f in readdir(d) if endswith(f, ".h5")] : String[])

# per run: band-averaged ⟨|X_k|²⟩/N for X = M_L, M_T, E; returns Dict band => [L, T, E]
function run_spectra(file, hmc::Bool)
    r = read_equilibrium(file);  R = r.rec;  N = r.meta.N;  h = r.meta.h
    nt = length(R.t);  i0 = hmc ? min(HBURN, nt - 2) + 1 : max(1, round(Int, BURN * nt)) + 1
    kh = [norm(k) * h for k in R.ks];  bands = round.(kh ./ DKH) .* DKH
    out = Dict{Float64,Vector{Float64}}()
    for b in unique(bands)
        idx = findall(==(b), bands)
        L = mean(mean(abs2.(R.ML[i][idx])) for i in i0:nt) / N
        T = mean(mean(abs2.(R.MT[i][idx])) for i in i0:nt) / N
        E = isempty(R.E) ? NaN : (Em = mean(R.E[i][idx] for i in i0:nt);  mean(mean(abs2.(R.E[i][idx] .- Em)) for i in i0:nt) / N)
        out[b] = [L, T, E]
    end
    return out, r.meta
end

function set_spectra(set, hmc)
    fs = files(set);  isempty(fs) && return nothing
    rs = [run_spectra(f, hmc)[1] for f in fs]
    bands = sort(collect(keys(rs[1])))
    m = Dict(b => mean(r[b] for r in rs) for b in bands)
    e = Dict(b => std([r[b] for r in rs]) ./ sqrt(length(rs)) for b in bands)
    return (bands = bands, m = m, e = e, n = length(rs))
end

ratio(a, ea, b, eb) = (a / b, abs(a / b) * sqrt((ea / a)^2 + (eb / b)^2))

fig, axs = subplots(2, 4, figsize = (18, 8), squeeze = false)
COL = Dict("nh" => "#eb6834", "h" => "#2a78d6", "nh_alone" => "#eda100", "h_alone" => "#1baf7a")
for (ci, (sa, v)) in enumerate([(sa, v) for sa in (16, 54) for v in ("0", "0.8")])
    ref = set_spectra("hq$(sa)_v$(v)", true)
    ref === nothing && (println("no HMC reference hq$(sa)_v$(v)");  continue)
    @printf("\ns_a = %d, v0 = %s: HMC reference %d chains\n", sa, v, ref.n)
    for (vi, var) in enumerate(("M_L", "M_T", "E"))
        vi == 3 && all(isnan(ref.m[b][3]) for b in ref.bands) && continue
        for which in ("nh", "h", "nh_alone", "h_alone")
            set = occursin("alone", which) ? "hs$(sa)_$(split(which, '_')[1])_alone_v$(v)" : "hs$(sa)_$(which)_v$(v)"
            S = set_spectra(set, false);  S === nothing && continue
            bs = [b for b in S.bands if b in ref.bands]
            rr = [ratio(S.m[b][vi], S.e[b][vi], ref.m[b][vi], ref.e[b][vi]) for b in bs]
            @printf("  %-22s %-3s / HMC: %s   (%d runs)\n", set, var,
                    join([@sprintf("kh %.2f %.3f ± %.3f", b, r[1], r[2]) for (b, r) in zip(bs, rr)], "  "), S.n)
            vi <= 2 || continue
            ax = axs[vi, ci]
            ax.errorbar(bs, first.(rr), yerr = last.(rr), fmt = occursin("alone", which) ? "s--" : "o-", color = COL[which], lw = 2,
                        ms = 5, capsize = 2, label = which == "nh" ? "full, without" : which == "h" ? "full, with Hastings" :
                                                                  which == "nh_alone" ? "alone, without" : "alone, with Hastings")
        end
    end
    for vi in 1:2
        ax = axs[vi, ci];  ax.axhline(1, color = "#52514e", lw = 0.6);  ax.grid(true, color = "#e6e5e0", lw = 0.5)
        ax.set_title("s_a = $sa, v₀ = $v: ⟨|$(vi == 1 ? "M_L" : "M_T")|²⟩ / HMC", fontsize = 9);  ax.set_xlabel("kh")
        ax.legend(fontsize = 7, frameon = false)
    end
end
tight_layout();  out = figpath("hastings_spectra.png");  savefig(out, dpi = 110);  close();  println("\nwritten ", out)
