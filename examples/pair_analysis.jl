# pair_analysis.jl — the pair observables of equilibrium runs (docs/equilibrium_positions.tex, stage 1).
# Reads every equilibrium_*.h5 in the given directories (one set each), averages over the runs of a set, and plots
#   (1) g(r) at a few times (t = 0 from the starting positions), with g_par and g_perp at the last time;
#   (2) ψ6 (global and local) and the height of the first peak of g against time;
#   (3) S(k) = |N_k|²/N against kh in time windows.
#   julia --project examples/pair_analysis.jl <dir> [<dir> ...]      → figures in SR_FIGDIR (default examples/)
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "plotsetup.jl"))
using Printf, Statistics

files(d) = isdir(d) ? [joinpath(d, f) for f in readdir(d) if endswith(f, ".h5") && startswith(f, "equilibrium_")] : String[]

function load_set(d)
    rs = [read_equilibrium(f) for f in files(d)]
    rs = filter(r -> r.part !== nothing && r.part.g !== nothing, rs)
    isempty(rs) && return nothing
    nt = minimum(length(r.part.t) for r in rs)
    P = rs[1].part;  a = rs[1].meta.a;  h = rs[1].meta.h
    avg(f) = sum(f(r) for r in rs) / length(rs)
    return (name = basename(rstrip(d, '/')), n = length(rs), a = a, h = h, t = P.t[1:nt], rb = P.rbins,
            g = avg(r -> r.part.g[:, 1:nt]), gpar = avg(r -> r.part.gpar[:, 1:nt]), gperp = avg(r -> r.part.gperp[:, 1:nt]),
            g0 = avg(r -> r.part.g0), psi6 = avg(r -> r.part.psi6[1:nt]), psi6loc = avg(r -> r.part.psi6loc[1:nt]),
            ks = P.ks, Sk = avg(r -> abs2.(r.part.Nk[:, 1:nt]) ./ r.meta.N))
end

sets = filter(!isnothing, [load_set(d) for d in ARGS])
isempty(sets) && error("no runs with pair observables in $(ARGS)")

for S in sets
    r = (S.rb[1:end-1] .+ S.rb[2:end]) ./ 2 ./ S.a
    nt = length(S.t);  pick = unique(round.(Int, [1, nt / 50, nt / 10, nt / 3, nt]))
    fig, ax = subplots(1, 3, figsize = (15, 4.2))
    ax[1].plot(r, S.g0, "k:", lw = 1, label = "t = 0")
    for (c, i) in enumerate(pick)
        ax[1].plot(r, S.g[:, max(i, 1)], color = "C$(c-1)", label = @sprintf("t = %.3g", S.t[max(i, 1)]))
    end
    ax[1].plot(r, S.gpar[:, end], "--", color = "C$(length(pick)-1)", lw = 0.8, label = "last: along x")
    ax[1].plot(r, S.gperp[:, end], ":", color = "C$(length(pick)-1)", lw = 0.8, label = "last: across")
    ax[1].axhline(1, color = "gray", lw = 0.5);  ax[1].set_xlabel("r / a");  ax[1].set_ylabel("g(r)")
    ax[1].set_ylim(0, min(6, 1.2 * maximum(S.g[:, end]) + 1));  ax[1].legend(fontsize = 7)
    ax[1].set_title("$(S.name): $(S.n) runs, h/a = $(round(S.h / S.a, digits = 2))")
    k1 = argmax(S.g[:, end])
    ax[2].semilogx(S.t, S.psi6, label = "ψ6 global");  ax[2].semilogx(S.t, S.psi6loc, label = "ψ6 local")
    ax[2].semilogx(S.t, S.g[k1, :] ./ maximum(S.g[k1, :]), label = @sprintf("g at r = %.2f a (÷ max)", r[k1]))
    ax[2].set_xlabel("t");  ax[2].legend(fontsize = 8);  ax[2].set_ylim(0, 1.05)
    kh = [norm(k) * S.h for k in eachcol(S.ks)]
    for (c, (t1, t2)) in enumerate(((1, nt ÷ 10), (nt ÷ 10 + 1, nt ÷ 2), (nt ÷ 2 + 1, nt)))
        t2 >= t1 || continue
        ax[3].semilogy(kh, vec(mean(S.Sk[:, t1:t2], dims = 2)), "o", ms = 3, color = "C$(c-1)",
                       label = @sprintf("t = %.3g–%.3g", S.t[t1], S.t[t2]))
    end
    ax[3].axhline(1, color = "gray", lw = 0.5, label = "Poisson");  ax[3].set_xlabel("kh");  ax[3].set_ylabel("S(k) = |N_k|²/N")
    ax[3].legend(fontsize = 8)
    tight_layout();  savefig(figpath("g_$(S.name).png"), dpi = 100);  close()
    @printf("%s: %d runs, t = 0 … %.3g;  ψ6 %.3f → %.3f (local %.3f → %.3f);  first peak of g at %.2f a, height %.2f → %.2f\n",
            S.name, S.n, S.t[end], S.psi6[1], S.psi6[end], S.psi6loc[1], S.psi6loc[end], r[k1], S.g[k1, 1], S.g[k1, end])
end

# The combined figure of docs/equilibrium_positions.tex (Fig. g_vs_t): g(r) at a sequence of times for each set (one panel
# per set, up to four), and the global and local ψ6 of all sets against time.  → equilibrium_g_vs_t.png
let tsel = (0.035, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0), cm = get_cmap("viridis")
    ns = min(4, length(sets));  fig, ax = subplots(2, 3, figsize = (16, 9));  axs = vec(permutedims(ax))
    for (j, S) in enumerate(sets[1:ns])
        r = (S.rb[1:end-1] .+ S.rb[2:end]) ./ 2 ./ S.a;  A = axs[j]
        A.plot(r, S.g0, color = "0.7", lw = 0.8, label = "t = 0 (lattice)")
        ts = [t for t in tsel if t <= S.t[end] * 1.01]
        for (c, t) in enumerate(ts)
            i = argmin(abs.(S.t .- t))
            A.plot(r, S.g[:, i], color = cm((c - 1) / max(1, length(ts) - 1)), lw = 1.4, label = @sprintf("t = %.3g", S.t[i]))
        end
        A.axhline(1, color = "k", lw = 0.4);  A.set_xlim(0, 4);  A.set_ylim(0, 4);  A.set_xlabel("r / a");  A.set_ylabel("g(r)")
        A.set_title("$(S.name), $(S.n) runs", fontsize = 10);  A.legend(fontsize = 7, loc = "upper right")
    end
    for (j, S) in enumerate(sets)
        axs[5].semilogx(S.t, S.psi6, color = "C$(j-1)", label = S.name);  axs[6].semilogx(S.t, S.psi6loc, color = "C$(j-1)", label = S.name)
    end
    axs[5].set_xlabel("t");  axs[5].set_ylabel("global ψ6");  axs[5].set_ylim(0, 1.05);  axs[5].legend(fontsize = 7)
    axs[6].axhline(0.365, color = "k", lw = 0.5, ls = ":");  axs[6].set_xlabel("t");  axs[6].set_ylabel("local ψ6");  axs[6].set_ylim(0, 1.05)
    ns < 4 && for j in ns+1:4;  axs[j].axis("off");  end
    tight_layout();  savefig(figpath("equilibrium_g_vs_t.png"), dpi = 110);  close()
end
