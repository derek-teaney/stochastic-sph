# grid_waves_series.jl — the waves themselves: Re c(t)/|c(0)| of the mode amplitude, grid and particles,
# against hydrodynamics c₀ e^{−iω_n t − Γ_hyd t}.
#   julia --project examples/grid_waves_series.jl [ideal_dir] [viscous_dir]
#   (defaults $SCRATCH/sph/grid/waves, $SCRATCH/sph/grid/waves_quick)
# docs/figs/grid_ideal_waves_kh<..>.png: ideal step, rows mode × direction, columns v0 = 0, 0.3, 0.8.
# docs/figs/grid_viscous_waves_kh<..>.png: η/s = 0.2, rows mode × direction, columns v0 = 0, 0.8,
#   grid and particles in separate columns.
using PyPlot, Serialization, Printf, StaticArrays
const IDIR = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "grid", "waves")
const VDIR = length(ARGS) > 1 ? ARGS[2] : joinpath(ENV["SCRATCH"], "sph", "grid", "waves_quick")
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
load(d, f) = (p = joinpath(d, f);  isfile(p) ? deserialize(p) : nothing)
ideal = something(load(IDIR, "grid_waves_all.jls"), load(IDIR, "grid_waves_ideal.jls"), load(IDIR, "grid_waves.jls"), Some(nothing))
visc = load(VDIR, "grid_waves_viscous.jls")
const ROWS = (("shear", "x", "shear, k ∥ v"), ("shear", "y", "shear, k ⊥ v"), ("soundp", "x", "sound +, k ∥ v"),
              ("soundp", "y", "sound +, k ⊥ v"), ("soundm", "x", "sound −, k ∥ v"), ("soundm", "y", "sound −, k ⊥ v"))
COL = Dict("grid" => "C0", "particle" => "C1")
find(rs, sch, mode, dir, v0, nk) = (i = findfirst(((r, _),) -> r.scheme == sch && r.mode == mode && r.dir == dir && r.v0 == v0 && r.nk == nk, rs);
                                    i === nothing ? nothing : rs[i])
function hydro!(ax, r, sr)
    Γh = hasproperty(r, :Gamma_hydro) ? r.Gamma_hydro : 0.0
    c0 = sr.cs[1];  hyd = [c0 * exp(-im * r.omega_hydro * t - Γh * t) for t in sr.ts]
    ax.plot(sr.ts, real.(hyd) ./ abs(c0), "k--", lw = 0.9, label = "hydrodynamics")
    Γh > 0 && (ax.plot(sr.ts, abs.(hyd) ./ abs(c0), "k:", lw = 0.7);  ax.plot(sr.ts, -abs.(hyd) ./ abs(c0), "k:", lw = 0.7))
end
wave!(ax, r, sr; lw = 1.2) = ax.plot(sr.ts, real.(sr.cs) ./ abs(sr.cs[1]), color = COL[r.scheme], lw = lw,
                                     label = @sprintf("%s (Γ/c_s k = %+.1e)", r.scheme, r.Gamma / (sqrt(1/3) * r.kh / 0.05)))

if ideal !== nothing
    for nk in (1, 2)
        rows = [rw for rw in ROWS if find(ideal.res, "grid", rw[1], rw[2], 0.0, nk) !== nothing]
        fig, axs = subplots(length(rows), 3, figsize = (18, 2.6 * length(rows)), squeeze = false)
        for (i, (mode, dir, ttl)) in enumerate(rows), (j, v0) in enumerate((0.0, 0.3, 0.8))
            ax = axs[i, j];  kh = 0.0
            for sch in ("particle", "grid")
                x = find(ideal.res, sch, mode, dir, v0, nk);  x === nothing && continue
                r, sr = x;  kh = r.kh;  wave!(ax, r, sr;  lw = sch == "grid" ? 1.6 : 1.0)
                sch == "grid" && hydro!(ax, r, sr)
            end
            ax.axhline(0, color = "k", lw = 0.4);  ax.set_ylim(-1.15, 1.15)
            ax.set_title(@sprintf("%s, v0 = %.1f, kh = %.2f, ideal", ttl, v0, kh), fontsize = 10)
            ax.legend(fontsize = 6.5, loc = "lower left");  i == length(rows) && ax.set_xlabel("t")
        end
        tight_layout()
        f = @sprintf("grid_ideal_waves_kh%.2f.png", nk * 0.262);  savefig(joinpath(FIGD, f), dpi = 90);  close();  println("wrote $f")
    end
end
if visc !== nothing && !isempty(visc.vres)
    for nk in (2, 4)
        rows = [rw for rw in ROWS if find(visc.vres, "grid", rw[1], rw[2], 0.0, nk) !== nothing]
        fig, axs = subplots(length(rows), 4, figsize = (22, 2.7 * length(rows)), squeeze = false)
        for (i, (mode, dir, ttl)) in enumerate(rows), (j, (v0, sch)) in enumerate(((0.0, "grid"), (0.0, "particle"), (0.8, "grid"), (0.8, "particle")))
            ax = axs[i, j];  x = find(visc.vres, sch, mode, dir, v0, nk);  x === nothing && continue
            r, sr = x;  wave!(ax, r, sr);  hydro!(ax, r, sr)
            ax.axhline(0, color = "k", lw = 0.4);  ax.set_ylim(-1.15, 1.15)
            ax.set_title(@sprintf("%s: %s, v0 = %.1f, kh = %.2f  (Γ/Γ_hyd = %.2f)", sch, ttl, v0, r.kh, r.Gamma / r.Gamma_hydro), fontsize = 9)
            ax.legend(fontsize = 6.5, loc = "lower left");  i == length(rows) && ax.set_xlabel("t")
        end
        tight_layout()
        f = @sprintf("grid_viscous_waves_kh%.2f.png", nk * 0.262);  savefig(joinpath(FIGD, f), dpi = 90);  close();  println("wrote $f")
    end
end
