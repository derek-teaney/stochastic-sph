# hbya_summary.jl — the shear damping Γ/Γ_hydro against h/a for each kernel (docs/hbya.tex), from the printed fits of boost_analysis.jl
# (groups HB2, HB3, HB4): at rest the weighted mean of k along x and y, at v = 0.8 along and across the flow.
#   julia --project examples/hbya_summary.jl <hb2.txt> <hb3.txt> <hb4.txt> <out.png>
include(joinpath(@__DIR__, "plotsetup.jl"));  using Printf
re = r"^\s+(C\d) v=(0|0\.8)\s+kh=([\d.]+) (par|perp)\s+Ω/k = \S+ \(hydro \S+\)\s+Γ/hydro = ([\d.]+)±([\d.]+)"
D = Dict()                                 # (ha, kernel, v, j, dir) → (kh, Γ, err), j = 1, 2 the two smallest k
for (ha, f) in zip((2, 3, 4), ARGS[1:3])
    seen = Dict()
    for l in eachline(f)
        m = match(re, l);  m === nothing && continue
        key = (m[1], m[2], m[4]);  seen[key] = get(seen, key, 0) + 1
        D[(ha, m[1], m[2], seen[key], m[4])] = (parse(Float64, m[3]), parse(Float64, m[5]), parse(Float64, m[6]))
    end
end
wm(a, b) = (w1 = 1 / a[3]^2;  w2 = 1 / b[3]^2;  ((a[2] * w1 + b[2] * w2) / (w1 + w2), 1 / sqrt(w1 + w2)))
fig, ax = subplots(1, 2, figsize = (12, 4.5))
for (c, k) in enumerate(("C2", "C4", "C6")), j in 1:2
    pts = [(ha, D[(ha, k, "0", j, "par")][1], wm(D[(ha, k, "0", j, "par")], D[(ha, k, "0", j, "perp")])...) for ha in (2, 3, 4) if haskey(D, (ha, k, "0", j, "par"))]
    ax[1].errorbar([p[1] for p in pts] .+ 0.04 * (c - 2) .+ 0.015 * (j - 1.5), [p[3] for p in pts], [p[4] for p in pts], fmt = j == 1 ? "o-" : "s--",
                   color = "C$(c-1)", mfc = j == 1 ? "C$(c-1)" : "none", capsize = 2, label = "$k, $(j == 1 ? "smallest" : "second") k")
    for p in pts;  @printf("rest %s ha=%d kh=%.2f  Γ/Γ_hyd %.3f(%d)\n", k, p[1], p[2], p[3], round(Int, 1000p[4]));  end
end
ax[1].plot([0, 5], [1, 1], "k-", lw = 0.5);  ax[1].set_xlim(1.7, 4.3);  ax[1].set_xticks([2, 3, 4])
ax[1].set_xlabel("h/a  (a fixed, s_a = 135)");  ax[1].set_ylabel("Γ/Γ_hydro at rest");  ax[1].legend(fontsize = 7);  ax[1].set_title("shear damping at rest", fontsize = 10)
for (c, k) in enumerate(("C2", "C4")), (dir, mk) in (("par", "o-"), ("perp", "s--")), j in 1:2
    pts = [(ha, D[(ha, k, "0.8", j, dir)]...) for ha in (2, 3, 4) if haskey(D, (ha, k, "0.8", j, dir))]
    j == 2 || continue
    ax[2].errorbar([p[1] for p in pts] .+ 0.05 * (c - 1.5) .+ (dir == "par" ? 0 : 0.02), [p[3] for p in pts], [p[4] for p in pts], fmt = mk, color = "C$(c-1)",
                   mfc = dir == "par" ? "C$(c-1)" : "none", capsize = 2, label = "$k, $(dir == "par" ? "along" : "across") the flow")
end
ax[2].plot([0, 5], [1, 1], "k-", lw = 0.5);  ax[2].set_xlim(1.7, 4.3);  ax[2].set_xticks([2, 3, 4])
ax[2].set_xlabel("h/a");  ax[2].set_ylabel("Γ/Γ_hydro at v0 = 0.8, second k (kh 0.52, 0.79, 1.05)");  ax[2].legend(fontsize = 7);  ax[2].set_title("shear damping at v0 = 0.8", fontsize = 10)
tight_layout();  savefig(ARGS[4], dpi = 110);  println("written ", ARGS[4])
