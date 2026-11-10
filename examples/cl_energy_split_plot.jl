# cl_energy_split_plot.jl — the energy of the particles against that of the coarse-grained
# fields, from the table of examples/cl_energy_split.jl (docs/continuum_limit.tex).
#   left:  excess over the uniform state, (E_p − E_0)/E_0 with E_p = Σ e_a, and (E_cg − E_0)/E_0
#          with E_cg = ∫ T⁰⁰(S̄, M̄), against h/a
#   right: the resolved fraction f_res = (E_cg − E_0)/(E_p − E_0) against h/a, with the
#          equipartition estimate f_ref = Σ_{k≠0} |W̃(k)|²/N
# open symbols: equilibrium draws (t = 0); filled: checkpoints of the cl runs (t > 0).
# Run:  julia --project examples/cl_energy_split_plot.jl [table.csv]  → docs/figs/cl_energy_split.png
include(joinpath(@__DIR__, "plotsetup.jl"))
using Printf
const CSV = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "cl_energy", "cl_energy_split.csv")
lines_ = readlines(CSV);  hdr = split(lines_[1], ',')
rows = [Dict(zip(hdr, split(l, ','))) for l in lines_[2:end]]
num(r, k) = parse(Float64, r[k])
isdraw(r) = startswith(r["label"], "draw")
# one row per state: keep the last entry of each label
byl = Dict(r["label"] => r for r in rows);  rows = collect(values(byl))
draws = sort(filter(isdraw, rows), by = r -> num(r, "ha"))
runs = filter(!isdraw, rows)

fig, axs = subplots(1, 2, figsize = (12, 4.6))
ax = axs[1]
ax.loglog([num(r, "ha") for r in draws], [num(r, "exc_p") for r in draws], "o-", mfc = "none", color = "C0",
          label = "particles, \$\\Sigma e_a\$ (draw, t = 0)")
ax.loglog([num(r, "ha") for r in draws], [num(r, "exc_cg") for r in draws], "s-", mfc = "none", color = "C1",
          label = "coarse grained, \$\\int T^{00}(\\bar S, \\bar M)\$ (draw)")
if !isempty(runs)
    ax.loglog([num(r, "ha") for r in runs], [num(r, "exc_p") for r in runs], "o", color = "C0", ms = 4, label = "particles (runs, t > 0)")
    ax.loglog([num(r, "ha") for r in runs], [num(r, "exc_cg") for r in runs], "s", color = "C1", ms = 4, label = "coarse grained (runs)")
end
ha = exp.(range(log(1.8), log(26), length = 50))
ax.loglog(ha, 2 .* ha .^ 2 ./ 2165, "k:", lw = 0.8, label = "\$2/s_a\$ (equipartition)")
ax.loglog([num(r, "ha") for r in draws], [2 * num(r, "f_ref") * num(r, "ha")^2 / 2165 for r in draws], "k--", lw = 0.8,
          label = "\$2 f_{ref}/s_a\$ (equipartition, \$k\\lesssim1/h\$)")
ax.set_xlabel("h/a");  ax.set_ylabel("\$(E - E_0)/E_0\$")
ax.set_title("thermal energy above the uniform state");  ax.legend(fontsize = 8)
ax = axs[2]
ax.loglog([num(r, "ha") for r in draws], [num(r, "f_res") for r in draws], "o-", mfc = "none", color = "C2",
          label = "\$f_{res}\$ (draw)")
isempty(runs) || ax.loglog([num(r, "ha") for r in runs], [num(r, "f_res") for r in runs], "o", color = "C2", ms = 4,
                           label = "\$f_{res}\$ (runs)")
ax.loglog([num(r, "ha") for r in draws], [num(r, "f_ref") for r in draws], "k--", lw = 1,
          label = "\$\\Sigma_{k\\neq0}|\\tilde W(k)|^2/N\$")
ax.set_xlabel("h/a");  ax.set_ylabel("\$(E_{cg} - E_0)/(\\Sigma e_a - E_0)\$")
ax.set_title("fraction of the thermal energy in the resolved modes");  ax.legend(fontsize = 8)
for a in axs
    a.set_xticks([2, 3, 4, 6, 8, 12, 16, 23]);  a.set_xticklabels(["2", "3", "4", "6", "8", "12", "16", "23"])
    a.minorticks_off()
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "cl_energy_split.png")
savefig(out, dpi = 130);  println("Saved ", out)
for r in sort(rows, by = r -> (num(r, "ha"), r["label"]))
    @printf("%-36s h/a=%4.1f t=%6.2f  exc_p=%.3e exc_cg=%.3e f=%.4f f_ref=%.4f\n", r["label"], num(r, "ha"), num(r, "t"),
            num(r, "exc_p"), num(r, "exc_cg"), num(r, "f_res"), num(r, "f_ref"))
end
