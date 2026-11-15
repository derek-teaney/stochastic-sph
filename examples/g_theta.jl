# g_theta.jl — the pair correlation against the angle of the bond to the flow, g(r, θ) = g(r) + 2 Σ_{n=2,4,6} g_n(r) cos nθ, at fixed r,
# for sets at rest and moving along x (docs/equilibrium_positions.tex, campaign hmc_liou2).  Time average over t > TB per run, mean and error
# over runs.  The reference: a rest-frame isotropic g seen at equal lab time, g_lab(r, θ) = g_0(r √((γ² cos²θ + sin²θ)/γ)), g_0 the g(r) of
# the set at rest (lab spacing a; rest-frame spacing √γ a along with the contraction 1/γ along x).
#   julia --project examples/g_theta.jl <out.png> <dir v=0> <dir> [<dir> ...]      (the first directory is the set at rest)
include(joinpath(@__DIR__, "..", "src", "load.jl"));  include(joinpath(@__DIR__, "plotsetup.jl"))
using Statistics, Printf
const TB = 5.0;  const RS = (0.75, 1.15);  const θs = range(0, π / 2, length = 91)
function load(d)                       # per run: time-averaged g, g2, g4, g6 (one run in memory at a time)
    out = []
    for f in sort(readdir(d; join = true))
        endswith(f, ".h5") || continue
        r = read_equilibrium(f);  P = r.part;  it = findall(>(TB), P.t)
        push!(out, (g = vec(mean(P.g[:, it], dims = 2)), g2 = vec(mean(P.g2[:, it], dims = 2)), g4 = vec(mean(P.g4[:, it], dims = 2)),
                    g6 = vec(mean(P.g6[:, it], dims = 2)), rm = (P.rbins[1:end-1] .+ P.rbins[2:end]) ./ 2 ./ r.meta.a, v = r.meta.boost))
        r = nothing;  GC.gc()
    end
    out
end
interp(x, xs, ys) = (i = clamp(searchsortedlast(xs, x), 1, length(xs) - 1);  ys[i] + (ys[i+1] - ys[i]) * (x - xs[i]) / (xs[i+1] - xs[i]))
gθ(R, r0, θ) = (i = argmin(abs.(R.rm .- r0));  R.g[i] + 2 * (R.g2[i] * cos(2θ) + R.g4[i] * cos(4θ) + R.g6[i] * cos(6θ)))
out = ARGS[1];  dirs = ARGS[2:end]
sets = [(basename(rstrip(d, '/')), load(d)) for d in dirs]
fig, ax = subplots(1, length(RS), figsize = (6.5 * length(RS), 4.6))
g0 = mean(R.g for R in sets[1][2]);  rm = sets[1][2][1].rm
for (k, r0) in enumerate(RS)
    for (j, (name, rs)) in enumerate(sets)
        Y = [[gθ(R, r0, θ) for θ in θs] for R in rs];  m = mean(Y);  e = std(Y) ./ sqrt(length(Y))
        ax[k].plot(rad2deg.(θs), m, color = "C$(j-1)", label = "$name ($(length(rs)) runs)")
        ax[k].fill_between(rad2deg.(θs), m .- e, m .+ e, color = "C$(j-1)", alpha = 0.25, lw = 0)
        i = argmin(abs.(rs[1].rm .- r0))
        @printf("%-18s r = %.2f a (bin %.3f):  g %.3f  g2 %+.4f(%.0f)  g4 %+.4f(%.0f)  g6 %+.4f(%.0f)  [×1e4]\n", name, r0, rs[1].rm[i], mean(R.g[i] for R in rs),
                mean(R.g2[i] for R in rs), 1e4 * std([R.g2[i] for R in rs]) / sqrt(length(rs)), mean(R.g4[i] for R in rs), 1e4 * std([R.g4[i] for R in rs]) / sqrt(length(rs)),
                mean(R.g6[i] for R in rs), 1e4 * std([R.g6[i] for R in rs]) / sqrt(length(rs)))
    end
    v = maximum(R.v for (_, rs) in sets for R in rs);  γ = 1 / sqrt(1 - v^2)
    ax[k].plot(rad2deg.(θs), [interp(r0 * sqrt((γ^2 * cos(θ)^2 + sin(θ)^2) / γ), rm, g0) for θ in θs], "k--", lw = 1,
               label = @sprintf("rest-frame isotropic, contracted (v = %.1f)", v))
    ax[k].set_xlabel("θ, the angle of the bond to the flow (deg)");  ax[k].set_ylabel("g(r, θ)")
    ax[k].set_title(@sprintf("r = %.2f a:  g + 2Σ g_n cos nθ, n = 2, 4, 6", r0), fontsize = 10);  ax[k].set_xticks(0:15:90)
    k == 1 && ax[k].legend(fontsize = 7)
end
tight_layout();  savefig(out, dpi = 110);  println("written ", out)
