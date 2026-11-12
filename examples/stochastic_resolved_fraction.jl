# stochastic_resolved_fraction.jl — the fraction R(k) = K_res(k)/K(k) of the bond dissipation
# that the fitted gradient sees, versus kh, for several h/a (docs/stochastic.tex).  With the
# stress move alone, a mode relaxes at R(k)Γ(k) instead of Γ(k).
#
# Run:  julia --project examples/stochastic_resolved_fraction.jl  → docs/figs/stochrep_resolved_fraction.png
using PyPlot, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "stochastic_thermalize_common.jl"))     # Kmode

eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
# compute first, plot afterwards (PyCall objects must not be finalized inside threaded code)
curves = []
for (j, (dim, has)) in enumerate(((1, (2.0, 2.5, 4.0)), (2, (2.0, 2.5, 3.0))))
    for (c, ha) in enumerate(has)
        params, ps, cl, _ = dim == 1 ?
            setup_ic(0.05, ha, eos, (x, y) -> 1e4, (x, y) -> SVector(0.0, 0.0); Lx=2.0, sim_dim=1) :
            setup_ic(0.05, ha, eos, (x, y) -> 1e4, (x, y) -> SVector(0.0, 0.0); L=1.0, lattice=:hex)
        st = stochastic_frozen(ps, params, cl, eos).stars[1]
        h = params.h;  a = params.a
        kh = collect(range(0.02, π * h / a, length=200))      # up to the zone edge k = π/a
        for pol in (:L, :T)
            dim == 1 && pol == :T && continue
            ê = pol == :L ? SVector(1.0, 0.0) : SVector(0.0, 1.0)
            R = [(K = Kmode(st, SVector(x / h, 0.0), ê, dim == 1); K[2] / K[1]) for x in kh]
            for target in (0.9, 0.5)
                i = findfirst(<(target), R)
                i === nothing || @printf("%dD h/a=%.1f %s (%d bonds): R < %.1f for kh > %.2f  (k a/π > %.3f)\n",
                                         dim, ha, pol, length(st.nb), target, kh[i], kh[i] * a / (π * h))
            end
            push!(curves, (j = j, c = c, ha = ha, pol = pol, dim = dim, nb = length(st.nb), kh = kh, R = R))
        end
    end
end
fig, axs = subplots(1, 2, figsize=(14, 5))
for cv in curves
    ax = axs[cv.j]
    ax.plot(cv.kh, cv.R, cv.pol == :L ? "-" : "--", color="C$(cv.c-1)",
            label=@sprintf("h/a = %.1f%s, %d bonds", cv.ha, cv.dim == 1 ? "" :
                           (cv.pol == :L ? " longitudinal" : " transverse"), cv.nb))
end
for (j, dim) in enumerate((1, 2))
    ax = axs[j]
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel("k h");  ax.set_ylabel("R(k) = K_res / K");  ax.set_ylim(0, 1.05)
    ax.set_title("$(dim)D: fraction of the bond dissipation seen by the fit")
    ax.legend(fontsize=8)
end
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_resolved_fraction.png")
savefig(out, dpi=120);  println("Saved ", out)
