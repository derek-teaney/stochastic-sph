# long_vs_hmc.jl — S(k)/(4/s_a) of the long runs of hmc_dyn (hd_long_s541, hd_long_s135; GLF, lattice start) in windows of 50,
# against the HMC equilibrium of hmc_eq (dashed; the values of hmc_analysis.jl for hq12_s541, hq12_s135, written in below).
#   julia --project examples/long_vs_hmc.jl <hd_long_s541 dir> <hd_long_s135 dir> <out.png>   (copy checkpoint .h5 files first)
include(joinpath(@__DIR__, "..", "src", "load.jl"));  include(joinpath(@__DIR__, "plotsetup.jl"))
using Statistics, Printf
const SC = ENV["SCRATCH"] * "/sph"
const BANDS = (0.3, 0.5, 1.0, 1.5)
hmcband(d) = (r = read_equilibrium(first(filter(f -> endswith(f, ".h5"), readdir(d; join = true)))); r.meta.s_a)
fig, ax = subplots(1, 2, figsize = (11, 4.2))
for (i, (s, hq)) in enumerate((("s541", "hq12_s541"), ("s135", "hq12_s135")))
    sa = hmcband("$SC/hmc_eq/$hq")
    d = ARGS[i];  rs = [read_equilibrium(f) for f in readdir(d; join = true) if endswith(f, ".h5")]
    nt = minimum(length(r.part.t) for r in rs);  P = rs[1].part;  t = P.t[1:nt];  h = rs[1].meta.h
    kh = [hypot(P.ks[1, j], P.ks[2, j]) * h for j in 1:size(P.ks, 2)]
    win = [(t0, t0 + 50) for t0 in 0:50:t[end]-25]
    @printf("hd_long_%s: %d runs to t = %.0f, s_a = %.1f (from %s)\n", s, length(rs), t[end], sa, hq)
    for (c, kc) in enumerate(BANDS)
        js = findall(x -> abs(x - kc) < 0.05, kh)
        m = Float64[];  e = Float64[];  tm = Float64[]
        for (t0, t1) in win
            it = findall(x -> t0 <= x < t1, t);  isempty(it) && continue
            v = [mean(abs2.(r.part.Nk[js, it])) / r.meta.N / (4 / sa) for r in rs]
            push!(tm, (t0 + t1) / 2);  push!(m, mean(v));  push!(e, std(v) / sqrt(length(v)))
        end
        ax[i].errorbar(tm, m, e, fmt = "o-", ms = 3, color = "C$(c-1)", label = "kh = $kc")
        @printf("   kh %.1f: last 200: %.2f\n", kc, mean(m[tm .> t[end] - 200]))
    end
    ax[i].set_title("hd_long_$s (GLF, lattice start, $(length(rs)) runs) — dashed: HMC $hq", fontsize = 9)
    ax[i].set_xlabel("t");  ax[i].set_ylabel("S(k)/(4/s_a), windows of 50");  ax[i].set_yscale("log");  ax[i].legend(fontsize = 7)
end
for (i, vals) in enumerate(((19.29, 10.45, 3.85, 2.52), (8.66, 6.62, 3.36, 2.32)))
    for (c, v) in enumerate(vals);  ax[i].axhline(v, color = "C$(c-1)", ls = "--", lw = 0.8);  end
end
tight_layout();  savefig(ARGS[3], dpi = 120);  println("written ", ARGS[3])
