# campaign_stability_plot.jl — the stability scan of the NERSC campaigns `first` and
# `first_det` (nersc/LOG.md): stochrep_hamiltonian.jl runs to t = 200, many seeds per set.
# Kinetic energy, particle energies, smoothed energy and local disorder versus time,
# each set as a band (min to max over seeds) around its median; a table of the values
# at the end, the fraction of seeds whose lattice melted (local disorder > 1 a) and when.
#
# Run:  SR_DATADIR=$SCRATCH/sph julia --project examples/campaign_stability_plot.jl [out.png]
#   reads $SR_DATADIR/first/<run>/*.jls and $SR_DATADIR/first_det/<run>/*.jls
using PyPlot, Printf, Serialization, StaticArrays, Statistics
const ROOT = get(ENV, "SR_DATADIR", joinpath(ENV["SCRATCH"], "sph"))
const OUT  = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "docs", "figs", "campaign_first_stability.png")

# (label, campaign/run, kind, colour, line style)
sets = [("S0=1e6 (δv=0.043), λ=1",       "first/ref",          :full, "C0", "-"),
        ("S0=1e6, λ=∞",                  "first/ref",          :det,  "C0", "--"),
        ("S0=3e5 (δv=0.078), λ=1",       "first/s3e5",         :full, "C1", "-"),
        ("S0=3e5, λ=∞",                  "first_det/s3e5_det", :det,  "C1", "--"),
        ("S0=1e5 (δv=0.14), λ=1",        "first/s1e5",         :full, "C3", "-"),
        ("S0=1e5, λ=∞",                  "first_det/s1e5_det", :det,  "C3", "--"),
        ("S0=1e5, λ=1, energy fix",      "first/s1e5_hfix",    :full, "C4", "-"),
        ("S0=1e5, λ=1, h/a=3",           "first/s1e5_ha3",     :full, "C2", "-")]

function load(dir, kind)
    d = joinpath(ROOT, dir);  isdir(d) || return []
    fs = filter(f -> endswith(f, ".jls") && !occursin("_real_", f), readdir(d))
    rs = []
    for f in fs
        r = deserialize(joinpath(d, f))
        haskey(r, kind) && push!(rs, r[kind])
    end
    return rs
end

# pad a series with NaN to length n (runs that stopped early, or differ in length)
pad(y, n) = length(y) >= n ? y[1:n] : vcat(y, fill(NaN, n - length(y)))
qband(Y, q) = [let v = filter(isfinite, Y[i, :]); isempty(v) ? NaN : quantile(v, q) end for i in 1:size(Y, 1)]

fig, axs = subplots(2, 2, figsize=(16, 11));  axs = vec(permutedims(axs))
fields = (:KE, :Es, :Hs, :loc)
titles = ("kinetic energy / equilibrium", "particle energies  Σe/Σe₀ − 1",
          "smoothed energy  H/H₀ − 1", "local disorder / a")
rows = String[]
heat = []          # (δv, kind, h/a, median H/H₀−1 at t = 100, 200, blown up?)
dvof = Dict("S0=1e6" => 0.043, "S0=3e5" => 0.078, "S0=1e5" => 0.14)
for (lab, dir, kind, c, ls) in sets
    rs = load(dir, kind)
    isempty(rs) && (push!(rows, @sprintf("%-28s  (no data yet)", lab)); continue)
    n = maximum(length(r.ts) for r in rs)
    t = first(r.ts for r in rs if length(r.ts) == n)
    for (ax, fld) in zip(axs, fields)
        Y = reduce(hcat, [pad(getfield(r, fld), n) for r in rs])
        fld in (:Es, :Hs) && (Y = abs.(Y))                    # log scale: magnitude of the drift
        ax.fill_between(t, qband(Y, 0.0), qband(Y, 1.0), color=c, alpha=ls == "-" ? 0.18 : 0.08)
        ax.plot(t, qband(Y, 0.5), color=c, ls=ls, lw=1.3, label=@sprintf("%s (%d)", lab, length(rs)))
    end
    tend = [r.ts[end] for r in rs]
    fin(f) = [getfield(r, f)[end] for r in rs]
    melt = [let i = findfirst(>(1.0), r.loc); i === nothing ? NaN : r.ts[i] end for r in rs]
    nmelt = count(isfinite, melt)
    blown = count(r -> !(isfinite(r.KE[end]) && r.KE[end] < 10), rs)
    if !occursin("energy fix", lab)
        hat(tq) = median([r.Hs[findfirst(>=(tq - 1e-9), r.ts)] for r in rs if r.ts[end] >= tq])
        push!(heat, (dvof[lab[1:6]], kind, occursin("h/a=3", lab) ? 3 : 2, hat(100.0), hat(200.0), blown > length(rs) ÷ 2))
    end
    push!(rows, @sprintf("%-28s %3d seeds  t_end %5.0f  KE/eq %8.3g  ΔΣe/Σe %+9.2e  ΔH/H %+9.2e  disorder %5.2f a  melted %2d/%d %s  KE/eq>10 %d",
                         lab, length(rs), median(tend), median(fin(:KE)), median(fin(:Es)), median(fin(:Hs)),
                         median(fin(:loc)), nmelt, length(rs),
                         nmelt > 0 ? @sprintf("(median t %.0f)", median(filter(isfinite, melt))) : "", blown))
end
for (ax, ti) in zip(axs, titles)
    ax.set_title(ti);  ax.set_xlabel("t");  ax.set_yscale("log")
end
axs[1].axhline(1, color="k", lw=0.5);  axs[4].axhline(1, color="k", lw=0.5, ls=":")
axs[2].set_title("particle energies  |Σe/Σe₀ − 1|");  axs[3].set_title("smoothed energy  |H/H₀ − 1|")
axs[1].legend(fontsize=8, loc="upper left")
suptitle("Stability scan (campaigns first, first_det): box 0.6 × 0.6, h = 0.05, η/s = 0.2 (N = 672 at h/a = 2); band = min–max over seeds, line = median")
tight_layout()
savefig(OUT, dpi=110);  println("Saved ", OUT)

# heating versus thermal velocity: median H/H₀ − 1 at t = 100 and 200, log-log
figure(figsize=(7, 5))
for (tcol, mkr) in ((4, "o"), (5, "s"))
    for (kind, c) in ((:full, "C3"), (:det, "C0"))
        pts = [(h[1], h[tcol]) for h in heat if h[2] == kind && h[3] == 2 && !h[6]]
        isempty(pts) && continue
        plot(first.(pts), last.(pts), mkr * "-", color=c, mfc=tcol == 4 ? "none" : c,
             label=(kind == :full ? "λ = 1" : "λ = ∞") * (tcol == 4 ? ", t = 100" : ", t = 200"))
    end
end
dv = [0.04, 0.15];  plot(dv, 4.6e-2 .* (dv ./ 0.043) .^ 4, "k:", lw=0.8, label="∝ δv⁴")
xscale("log");  yscale("log");  xlabel("thermal velocity per component δv");  ylabel("median H/H₀ − 1")
title("heating of the smoothed energy, box 0.6, h/a = 2\n(λ = 1 at δv = 0.14 blows up at t ≈ 45: not shown)", fontsize=10)
legend(fontsize=8);  tight_layout()
savefig(replace(OUT, ".png" => "_heating.png"), dpi=110)
for h in heat;  println(h);  end
foreach(println, rows)
