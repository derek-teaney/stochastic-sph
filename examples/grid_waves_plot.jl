# grid_waves_plot.jl — figures of examples/grid_waves.jl (docs/gridstep.tex, "Results").
#   julia --project examples/grid_waves_plot.jl [dir]      (dir default $SCRATCH/sph/grid/waves)
# Reads grid_waves_ideal.jls / grid_waves.jls (ideal waves, Δt scan) and grid_waves_viscous.jls.
# Writes docs/figs/grid_dispersion.png, grid_damping.png, grid_viscous.png, grid_series.png.
using PyPlot, Serialization, Printf, LinearAlgebra
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "grid", "waves")
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
const CS = sqrt(1 / 3);  const H = 0.05
load(f) = (p = joinpath(DIR, f);  isfile(p) ? deserialize(p) : nothing)
ideal = something(load("grid_waves_all.jls"), load("grid_waves_ideal.jls"), load("grid_waves.jls"), Some(nothing))
visc = load("grid_waves_viscous.jls")

const MODES = (("shear", "shear"), ("soundp", "sound +"), ("soundm", "sound −"))
const DIRS = (("x", "k ∥ v"), ("y", "k ⊥ v"))
const VCOL = Dict(0.0 => "C0", 0.3 => "C2", 0.8 => "C3")
pick(rs, sch, mode, dir, v0) = sort([r for (r, _) in rs if r.scheme == sch && r.mode == mode && r.dir == dir && r.v0 == v0], by = r -> r.kh)

# one 3×2 figure: y(r) against kh, rows modes, columns directions; grid solid, particle dashed
function grid_fig(rs, y, ylab, fname; title = "", ylim = nothing, hline = 0.0)
    fig, axs = subplots(3, 2, figsize = (13, 10.5), squeeze = false)
    for (i, (mode, mlab)) in enumerate(MODES), (j, (dir, dlab)) in enumerate(DIRS)
        ax = axs[i, j]
        for v0 in (0.0, 0.3, 0.8), (sch, ls, mk) in (("grid", "-", "o"), ("particle", "--", "s"))
            rr = pick(rs, sch, mode, dir, v0);  isempty(rr) && continue
            ax.plot([r.kh for r in rr], [y(r) for r in rr], ls * mk, color = VCOL[v0], mfc = sch == "grid" ? VCOL[v0] : "none",
                    label = @sprintf("%s, v0 = %.1f", sch, v0))
        end
        hline === nothing || ax.axhline(hline, color = "k", lw = 0.5)
        ylim === nothing || ax.set_ylim(ylim...)
        ax.set_title("$mlab, $dlab", fontsize = 10);  ax.set_xlabel("kh");  j == 1 && ax.set_ylabel(ylab)
        i == 1 && j == 1 && ax.legend(fontsize = 7, ncol = 2)
    end
    isempty(title) || suptitle(title, fontsize = 11)
    tight_layout();  savefig(joinpath(FIGD, fname), dpi = 100);  close()
    println("wrote $fname")
end

if ideal !== nothing
    rs = ideal.res
    grid_fig(rs, r -> (r.omega - r.omega_hydro) / (CS * r.kh / H), "(ω − ω_hydro)/(c_s k)", "grid_dispersion.png";
             title = "Ideal step: frequency error of the mode (filled/solid: grid, open/dashed: particles)")
    grid_fig(rs, r -> r.Gamma / (CS * r.kh / H), "Γ/(c_s k)", "grid_damping.png";
             title = "Ideal step: numerical damping (Γ > 0) or growth (Γ < 0), ideal hydrodynamics Γ = 0")
    # time series: shear at rest and along the flow at v0 = 0.8, sound+ along the flow at v0 = 0.8, kh = 0.52
    fig, axs = subplots(1, 3, figsize = (16, 4.4))
    for (ax, (mode, dir, v0, ttl)) in zip(axs, (("shear", "x", 0.0, "shear at rest"), ("shear", "x", 0.8, "shear along the flow, v0 = 0.8"),
                                               ("soundp", "x", 0.8, "sound + along the flow, v0 = 0.8")))
        for (sch, col) in (("grid", "C0"), ("particle", "C1"))
            i = findfirst(((r, _),) -> r.scheme == sch && r.mode == mode && r.dir == dir && r.v0 == v0 && r.nk == 2, rs)
            i === nothing && continue
            r, sr = rs[i]
            ax.plot(sr.ts, real.(sr.cs) ./ abs(sr.cs[1]), color = col, lw = 1, label = sch * ", Re c")
            ax.plot(sr.ts, abs.(sr.cs) ./ abs(sr.cs[1]), color = col, lw = 2, ls = "--", label = sch * ", |c|")
        end
        ax.set_title(ttl * ", kh = 0.52", fontsize = 10);  ax.set_xlabel("t");  ax.axhline(0, color = "k", lw = 0.5)
        ax.set_ylim(-1.3, 1.3);  ax.legend(fontsize = 7)
    end
    tight_layout();  savefig(joinpath(FIGD, "grid_series.png"), dpi = 100);  close();  println("wrote grid_series.png")
end
if visc !== nothing && !isempty(visc.vres)
    grid_fig(visc.vres, r -> r.Gamma / r.Gamma_hydro, "Γ / Γ_hydro", "grid_viscous.png"; hline = 1.0, ylim = (0, 2),
             title = "Deterministic viscosity η/s = 0.2: damping against boosted hydrodynamics (filled/solid: grid, open/dashed: particles)")
end
