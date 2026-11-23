# shear_overlay.jl — the shear correlator at rest, Re C_yy(k x̂, τ)/C_yy(k x̂, 0) at the smallest k, of several run sets on one axis
# (log scale), each with its own hydrodynamics e^{−ν0 k² τ}, ν0 = (η/s)/T_rel (dashed, same colour).  For the runs without the viscous
# step the dashed line is the η/s of the run record (0.2), for reference.
#   julia --project examples/shear_overlay.jl <tag> "<label>|<dir>" ...     (dir: a run set, absolute)
# Writes docs/figs/shear_overlay_<tag>.png.
using PyPlot, Printf, HDF5, Serialization, StaticArrays, LinearAlgebra, TOML
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const TAG = ARGS[1];  const SETS = [split(a, "|"; limit = 2) for a in ARGS[2:end]]
const DTS = 0.05;  const BURN = 5.0;  const TMAX = 12.0

function corr_yy(rec, i, NV)          # ⟨M_y(t+τ) M_y(t)*⟩ for k along x: M_y = M_T (k̂ = x̂)
    t = rec.t;  dtr = t[2] - t[1];  st = max(1, round(Int, DTS / dtr));  i0 = findfirst(>=(BURN), t)
    x = [rec.MT[s][i] for s in i0:st:length(t)];  n = length(x);  nl = min(round(Int, TMAX / (st * dtr)), n ÷ 2)
    ([sum(x[q+l] * conj(x[q]) for q in 1:n-l) / (n - l) / NV for l in 0:nl], collect(0:nl) .* st .* dtr)
end
fig, axs = subplots(1, 2, figsize = (13, 4.8))
for (ci, (lab, d)) in enumerate(SETS)
    fs = [joinpath(d, f) for f in readdir(d) if endswith(f, ".h5")]
    rs = filter(r -> get(r.meta, :complete, true), [read_equilibrium(f) for f in fs]);  m = rs[1].meta
    ks = rs[1].rec.ks;  i = argmin([abs(k[2]) < 1e-9 && k[1] > 0 ? k[1] : Inf for k in ks]);  k = ks[i][1]
    cs = [corr_yy(r.rec, i, m.N * m.V) for r in rs];  nl = minimum(length(c[1]) for c in cs);  τ = cs[1][2][1:nl]
    C = real.(sum(c[1][1:nl] for c in cs) / length(cs));  y = C ./ C[1]
    toml = replace(fs[1], r"\.h5$" => ".toml");  Tr = m.Trel
    if isfile(toml);  t = TOML.parsefile(toml)["start"];  haskey(t, "T0") && get(t, "bath", "") == "nominal" && (Tr = t["T0"]);  end
    ν0 = m.etas / Tr;  col = "C$(ci - 1)";  sa = m.S0 * m.a^2 * sqrt(3) / 2
    full = @sprintf("%s: s_a=%.0f, T_rel=%.1f, ν0=%.2e%s", lab, sa, Tr, ν0, get(m, :novisc, false) ? " (dashed: η/s=0.2)" : "")
    for (ax, x) in ((axs[1], τ), (axs[2], τ .* ν0 .* k^2))
        ax.semilogy(x, abs.(y), color = col, lw = 1.8, label = full)
        ax.semilogy(x, exp.(-ν0 * k^2 .* τ), color = col, lw = 1, ls = "--")
    end
    @printf("%-28s k h = %.2f  ν0 k² = %.4f  e-fold time: measured %.2f, hydro %.2f\n", lab, k * m.h, ν0 * k^2,
            τ[something(findfirst(<(exp(-1)), y), length(y))], 1 / (ν0 * k^2))
end
axs[1].set_xlabel("τ");  axs[2].set_xlabel("ν0 k² τ");  axs[2].set_xlim(0, 4)
for ax in axs;  ax.set_ylim(0.01, 1.1);  ax.set_ylabel("Re C_yy(k x̂, τ) / C_yy(k x̂, 0), kh = 0.52");  end
h, l = axs[1].get_legend_handles_labels();  fig.legend(h, l, loc = "lower center", ncol = 2, fontsize = 8, frameon = false);  axs[1].set_title("against τ (dashed: e^{−ν0 k² τ})", fontsize = 10)
axs[2].set_title("against ν0 k² τ: hydrodynamics is one line", fontsize = 10)
tight_layout(rect = (0, 0.12, 1, 1));  out = joinpath(@__DIR__, "..", "docs", "figs", "shear_overlay_$(TAG).png");  savefig(out, dpi = 100);  println("written ", out)
