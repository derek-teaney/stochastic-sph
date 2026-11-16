# equaltime_over_hmc.jl — the equal-time correlators of the dynamics divided by hybrid Monte Carlo of the same model and measure
# (docs/inverse_moment.tex, Stage 2: the owner's request, "as Fig. 36, divided by HMC instead of χ").  Same layout as the equal-time figure
# of boost_analysis.jl: rows E, M_x, M_y; columns v = 0 and 0.8 with k along x and along y; the three smallest box modes on each axis.
# ⟨|A_k|²⟩ per run (records with t ≥ BURN_T) and per HMC chain (updates > BURN_U), mean and standard error over runs / chains; the ratio
# dynamics / HMC needs no χ (the normalisation N V cancels).
#   julia --project examples/equaltime_over_hmc.jl <campaign dir> <s_a> <out.png>      (sets hs<sa>_nh_v<v>, hs<sa>_h_v<v>, hq<sa>_v<v>)
include(joinpath(@__DIR__, "..", "src", "load.jl"));  include(joinpath(@__DIR__, "plotsetup.jl"))
using Statistics, Printf, LinearAlgebra
const BURN_T = 5.0;  const BURN_U = 20
D, SA, OUT = ARGS[1], ARGS[2], ARGS[3]
names = ("E", "M_x", "M_y")
# per file: for direction (:x, :y) and mode j = 1:3 the means of |E|², |M_x|², |M_y|²; and kh
function permode(f, sel)
    r = read_equilibrium(f);  rec = r.rec;  ks = rec.ks;  h = r.meta.h
    it = sel(rec.t);  out = Dict{Tuple{Symbol,Int},Vector{Float64}}();  khs = Dict{Tuple{Symbol,Int},Float64}()
    for (dir, pick) in ((:x, i -> abs(ks[i][2]) < 1e-9 && ks[i][1] > 0), (:y, i -> abs(ks[i][1]) < 1e-9 && ks[i][2] > 0))
        js = sort(filter(pick, eachindex(ks)), by = i -> norm(ks[i]))[1:3]
        for (j, i) in enumerate(js)
            E = mean(abs2(rec.E[n][i]) for n in it);  L = mean(abs2(rec.ML[n][i]) for n in it);  T = mean(abs2(rec.MT[n][i]) for n in it)
            out[(dir, j)] = dir === :x ? [E, L, T] : [E, T, L]          # k ∥ x: M_x = M_L; k ∥ y: M_x = M_T
            khs[(dir, j)] = norm(ks[i]) * h
        end
    end
    out, khs
end
files(set) = (d = joinpath(D, set);  isdir(d) ? sort([f for f in readdir(d; join = true) if endswith(f, ".h5")]) : String[])
fig, axg = subplots(3, 4, figsize = (16.8, 9.5), squeeze = false)
cols = [("0", :x), ("0", :y), ("0.8", :x), ("0.8", :y)]
for (ci, (v, dir)) in enumerate(cols)
    hq = [permode(f, t -> findall(n -> n > BURN_U, eachindex(t)))[1] for f in files("hq$(SA)_v$v")]
    isempty(hq) && continue
    hm = [mean(c[(dir, j)][a] for c in hq) for j in 1:3, a in 1:3];  he = [std([c[(dir, j)][a] for c in hq]) / sqrt(length(hq)) for j in 1:3, a in 1:3]
    for (m, (lab, set)) in enumerate((("no Hastings", "hs$(SA)_nh_v$v"), ("Hastings", "hs$(SA)_h_v$v")))
        fs = files(set);  isempty(fs) && continue
        P = [permode(f, t -> findall(>=(BURN_T), t)) for f in fs];  runs = first.(P);  kh = P[1][2]
        for a in 1:3
            dm = [mean(r[(dir, j)][a] for r in runs) for j in 1:3];  de = [std([r[(dir, j)][a] for r in runs]) / sqrt(length(runs)) for j in 1:3]
            ratio = dm ./ hm[:, a];  err = ratio .* sqrt.((de ./ dm) .^ 2 .+ (he[:, a] ./ hm[:, a]) .^ 2)
            x = [kh[(dir, j)] for j in 1:3] .+ 0.012 * (m - 1.5)
            axg[a, ci].errorbar(x, ratio, err, fmt = "o-", ms = 4, capsize = 2, lw = 1, color = "C$(m-1)", label = "$lab ($(length(runs)) runs)")
            @printf("s_a=%s v=%-3s k∥%s %-3s %-11s %s\n", SA, v, dir, names[a], lab, join([@sprintf("kh %.2f: %.3f(%d)", x[j], ratio[j], round(Int, 1000err[j])) for j in 1:3], "  "))
        end
    end
    for a in 1:3
        ax = axg[a, ci];  ax.axhline(1, color = "k", lw = 0.5);  ax.set_ylim(0.6, 1.4)
        ax.set_title(@sprintf("s_a = %s, v = %s, k ∥ %s:  ⟨|%s_k|²⟩ / HMC  (%d chains)", SA, v, dir, names[a], length(hq)), fontsize = 9)
        a == 3 && ax.set_xlabel("kh")
    end
end
h_, l_ = axg[1, 1].get_legend_handles_labels();  fig.legend(h_, l_, loc = "upper center", ncol = 2, fontsize = 10, frameon = false)
tight_layout(rect = (0, 0, 1, 0.96));  savefig(OUT, dpi = 100);  println("written ", OUT)
