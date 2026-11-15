# moment_det_plot.jl — the figures of examples/moment_det.jl (docs/inverse_moment.tex, "Stage 1"): common | moment | moment/common.
#   julia --project examples/moment_det_plot.jl [data/moment_det/moment_det.toml]   → moment_det_*.png in SR_FIGDIR (default examples/)
# Rows: A1 the viscous step alone on the lattice at v0 = 0.8, K/K_target of the four modes; A2 its decay rates Γ/Γ_hyd for the two
# pure modes (shear along the flow, longitudinal across it; the other two are not pure modes of the viscous step alone, which holds
# each particle's energy); B the viscous step alone on disordered positions at rest, Γ/Γ_hyd; C the full deterministic step at rest,
# Γ/Γ_hyd of the shear and sound waves and ω/ω_hyd of the sound.
using TOML, Printf
include(joinpath(@__DIR__, "plotsetup.jl"))
const FILE = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "moment_det", "moment_det.toml")
res = TOML.parsefile(FILE)["case"]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # categorical, fixed order
INK = "#52514e"

# (row title, y label, [(label, selector, quantity)])
rows = [
    ("viscous step alone, lattice, v₀ = 0.8", "K / K_target",
     [("shear, k ∥ v", r -> r["set"] == "A" && r["kind"] == "shear" && r["dir"] == "x", "K_over_Ktarget"),
      ("long., k ∥ v", r -> r["set"] == "A" && r["kind"] == "long" && r["dir"] == "x", "K_over_Ktarget"),
      ("shear, k ⊥ v", r -> r["set"] == "A" && r["kind"] == "shear" && r["dir"] == "y", "K_over_Ktarget"),
      ("long., k ⊥ v", r -> r["set"] == "A" && r["kind"] == "long" && r["dir"] == "y", "K_over_Ktarget")]),
    ("viscous step alone, lattice, v₀ = 0.8: decay", "Γ / Γ_hydro",
     [("shear, k ∥ v", r -> r["set"] == "A" && r["kind"] == "shear" && r["dir"] == "x", "Gamma_over_hydro"),
      ("long., k ⊥ v", r -> r["set"] == "A" && r["kind"] == "long" && r["dir"] == "y", "Gamma_over_hydro")]),
    ("viscous step alone, disordered positions, at rest", "Γ / Γ_hydro",
     [("shear, jitter 0.3a", r -> r["set"] == "B" && r["kind"] == "shear" && r["jitter"] == 0.3, "Gamma_over_hydro"),
      ("long., jitter 0.3a", r -> r["set"] == "B" && r["kind"] == "long" && r["jitter"] == 0.3, "Gamma_over_hydro"),
      ("shear, random", r -> r["set"] == "B" && r["kind"] == "shear" && r["jitter"] == 3.0, "Gamma_over_hydro"),
      ("long., random", r -> r["set"] == "B" && r["kind"] == "long" && r["jitter"] == 3.0, "Gamma_over_hydro")]),
    ("full deterministic step, lattice, at rest, η/s = 0.02", "Γ/Γ_hydro, ω/ω_hydro",
     [("shear wave Γ", r -> r["set"] == "C" && r["kind"] == "shear", "Gamma_over_hydro"),
      ("sound wave Γ", r -> r["set"] == "C" && r["kind"] == "sound", "Gamma_over_hydro"),
      ("sound wave ω", r -> r["set"] == "C" && r["kind"] == "sound", "omega_over_hydro")])]

curve(sel, q, diss) = (pts = sort([r for r in res if sel(r) && r["scheme"] == diss], by = r -> r["kh"]);
                       ([r["kh"] for r in pts], [r[q] for r in pts]))

fig, axs = subplots(length(rows), 3, figsize = (14, 3.3 * length(rows)), squeeze = false)
for (i, (title, ylab, cs)) in enumerate(rows)
    for (ci, (lab, sel, q)) in enumerate(cs)
        xc, yc = curve(sel, q, "common");  xm, ym = curve(sel, q, "moment")
        axs[i, 1].plot(xc, yc, "o-", color = COL[ci], lw = 2, ms = 6, label = lab)
        axs[i, 2].plot(xm, ym, "o-", color = COL[ci], lw = 2, ms = 6, label = lab)
        axs[i, 3].plot(xm, ym ./ yc, "o-", color = COL[ci], lw = 2, ms = 6, label = lab)
    end
    lo = minimum(minimum(l.get_ydata()) for j in 1:2 for l in axs[i, j].get_lines());  hi = maximum(maximum(l.get_ydata()) for j in 1:2 for l in axs[i, j].get_lines())
    pad = 0.08 * (hi - lo + 1e-3)
    for j in 1:3
        ax = axs[i, j];  ax.axhline(1, color = INK, lw = 0.6);  ax.grid(true, color = "#e6e5e0", lw = 0.5)
        ax.set_xlabel("kh");  j < 3 && ax.set_ylim(min(lo, 1) - pad, max(hi, 1) + pad)
        if j == 3                                    # at least ±3%, wider if the data need it
            rl = [y for l in ax.get_lines() for y in l.get_ydata() if y != 1];  ext = max(0.03, 1.15 * maximum(abs.(rl .- 1)))
            ax.set_ylim(1 - ext, 1 + ext)
        end
        ax.legend(fontsize = 7, frameon = false)
    end
    axs[i, 1].set_ylabel(ylab);  axs[i, 3].set_ylabel("moment / common")
    axs[i, 1].set_title("common: $title", fontsize = 9);  axs[i, 2].set_title("moment: $title", fontsize = 9)
    axs[i, 3].set_title("ratio", fontsize = 9)
end
tight_layout()
out = figpath("moment_det.png");  savefig(out, dpi = 110);  close();  println("written ", out)
