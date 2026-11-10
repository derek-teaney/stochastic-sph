# energystep_equilibration.jl — debug job 61055621 (L = 0.6, h/a = 2, C2, t = 200, 4 seeds per run):
# SPH_HAMILTONIAN = energy against smoothed, the full scheme from a cold start and the ideal step
# alone from a thermal start.  KE/eq(t) and ΔH/H(t).
# Run:  julia --project examples/energystep_equilibration.jl $SCRATCH/sph/energystep_debug2 docs/figs/energystep_equilibration.png
using PyPlot, StaticArrays, LinearAlgebra, Printf, HDF5, Serialization
R = joinpath(@__DIR__, "..")
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure");  include(joinpath(R, "src", "$f.jl"));  end
D = ARGS[1];  out = ARGS[2]
load(run) = (d = joinpath(D, run); isdir(d) ? [read_equilibrium(joinpath(d, f)) for f in readdir(d) if endswith(f, ".h5")] : [])
avg(v) = (n = minimum(length.(v)); sum(x[1:n] for x in v) / length(v))
fig, axs = subplots(2, 2, figsize = (13, 9))
for (col, runs, ttl) in ((1, ("energy_cold", "smoothed_cold"), "full scheme, cold start"),
                         (2, ("energy_nv", "smoothed_nv"), "ideal step only (no viscous step), thermal start"))
    for (run, c) in zip(runs, ("C0", "C3"))
        rs = load(run);  isempty(rs) && continue
        t = rs[1].rec.t;  n = minimum(length.([r.KE for r in rs]));  t = t[1:n]
        for r in rs;  axs[1, col].plot(t, r.KE[1:n], color = c, lw = 0.4, alpha = 0.4);  end
        ke = avg([r.KE for r in rs]);  H = avg([r.H for r in rs])
        lab = @sprintf("%s (%d seeds, wall %.0f s/seed)", run, length(rs), sum(r.meta.wall for r in rs) / length(rs))
        axs[1, col].plot(t, ke, color = c, lw = 1.5, label = lab)
        for r in rs;  axs[2, col].plot(t, r.H[1:n], color = c, lw = 0.4, alpha = 0.4);  end
        axs[2, col].plot(t, H, color = c, lw = 1.5, label = run)
        @printf("%-14s t=%.0f  KE/eq(last half) %.3f  ΔH/H end %+.2e  complete %s\n", run, t[end],
                sum(ke[end÷2:end]) / length(ke[end÷2:end]), H[end], all(get(r.meta, :complete, true) for r in rs))
    end
    axs[1, col].axhline(1, color = "k", lw = 0.5);  axs[1, col].set_ylabel("KE / equilibrium")
    axs[1, col].set_title(ttl);  axs[1, col].legend(fontsize = 8)
    axs[2, col].axhline(0, color = "k", lw = 0.5);  axs[2, col].set_ylabel("ΔH/H");  axs[2, col].set_xlabel("t")
    axs[2, col].legend(fontsize = 8)
end
suptitle("SPH_HAMILTONIAN = energy (blue) vs smoothed (red); L = 0.6, h/a = 2, C2; thin: seeds, thick: mean")
tight_layout();  savefig(out, dpi = 110)
