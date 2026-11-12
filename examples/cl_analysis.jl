# cl_analysis.jl — campaign `cl` (nersc/campaign_cl.txt, docs/continuum_limit.tex): the
# n-scheme at fixed h = 0.05, S0 = 1e6, η/s = 0.2, lowering s_a = 2165/(h/a)².
# Figures (docs/figs/cl_*.png), one colour per h/a, dashed = Δt/2:
#   cl_energy.png    KE/eq and ΔH/H against t
#   cl_step.png      acceptance of the stress and residual moves, guard rejections, max |v|
#                    against t; and their means against s_a (with the short runs, cl_short)
#   cl_static.png    equal-time C_T(k,0), C_L(k,0) per NVwT against kh
#   cl_CT.png        C_T(k,τ)/C_T(k,0) in three shells, with e^{−νk²τ}; fitted 2γ/νk² against h/a
#   cl_CL.png        C_L(k,τ)/C_L(k,0) in three shells, with the damped sound mode
#   cl_spread.png    spread of s_a and of V_a against t, with √(3/s_a)
#   cl_pairs.png     D²(a,t)/a² and the mean square displacement against t; D_self against s_a
#                    with the hydrodynamic estimate
# Run:  julia --project examples/cl_analysis.jl [dir]       (login node)
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, HDF5, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const S = joinpath(ENV["SCRATCH"], "sph")
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(S, "cl")
const SHORT = joinpath(S, "cl_short")
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
const HAS = [2, 3, 4, 6, 8, 12, 16, 23, 32]
mean_(v) = isempty(v) ? NaN : sum(v) / length(v)
sem(v) = length(v) > 1 ? sqrt(sum(abs2, v .- mean_(v)) / (length(v) * (length(v) - 1))) : NaN
slope(t, y) = (tm = mean_(t); sum((t .- tm) .* (y .- mean_(y))) / sum(abs2, t .- tm))
colour(ha) = "C$(findfirst(==(ha), HAS) - 1)"
sa(ha) = 2165 / ha^2

files(d) = isdir(d) ? [joinpath(d, f) for f in readdir(d) if endswith(f, ".h5") && startswith(f, "equilibrium_")] : String[]
function series(f, name)
    h5open(f, "r") do h;  haskey(h, name) ? read(h[name]) : Float64[];  end
end
# all runs of one set; partial (checkpointed) runs are included, cut to a common length
function loadset(d)
    fs = files(d);  isempty(fs) && return nothing
    rs = [read_equilibrium(f) for f in fs]
    ex = Dict(k => [series(f, k) for f in fs] for k in ("sspr", "Vspr", "acc_s", "acc_r", "grej_s", "grej_r", "vmax"))
    nk = minimum(length(r.KE) for r in rs);  nk < 4 && return nothing
    (; rs, ex, nk, t = rs[1].rec.t[1:nk], m = rs[1].meta, n = length(rs),
       complete = count(r -> get(r.meta, :complete, true), rs))
end
avgser(v, nk) = mean_([x[1:min(nk, length(x))] for x in v if length(x) >= nk])

sets = Dict{Tuple{Int,Bool},Any}()
for ha in HAS, dt in (false, true)
    r = loadset(joinpath(DIR, "cl_ha$(ha)" * (dt ? "_dt" : "")));  r === nothing || (sets[(ha, dt)] = r)
end
short = Dict(ha => loadset(joinpath(SHORT)) for ha in ())     # filled below
println("sets: ", sort([(k[1], k[2], v.n, v.complete, round(v.t[end], digits = 1)) for (k, v) in sets]))

# ── summary table ────────────────────────────────────────────────────────────
@printf("%5s %5s %6s %6s %4s %6s %7s %10s %7s %7s %9s %9s %6s\n", "h/a", "check", "s_a", "δv", "n", "t", "KE/eq", "ΔH/H", "acc_s", "acc_r", "|v|≥1 s", "|v|≥1 r", "vmax")
for ha in HAS, dt in (false, true)
    haskey(sets, (ha, dt)) || continue
    r = sets[(ha, dt)];  nk = r.nk;  h2 = nk ÷ 2 + 1
    KE = mean_([mean_(x.KE[h2:nk]) for x in r.rs]);  dH = mean_([x.H[nk] for x in r.rs])
    @printf("%5d %5s %6.1f %6.3f %4d %6.1f %7.3f %+10.2e %7.3f %7.3f %9.2e %9.2e %6.3f\n", ha, dt ? "Δt/2" : "", sa(ha),
            1 / sqrt(sa(ha)), r.n, r.t[end], KE, dH, mean_(avgser(r.ex["acc_s"], nk)), mean_(avgser(r.ex["acc_r"], nk)),
            mean_(avgser(r.ex["grej_s"], nk)), mean_(avgser(r.ex["grej_r"], nk)), maximum(avgser(r.ex["vmax"], nk)))
end

# ── fig 1: energy ────────────────────────────────────────────────────────────
fig, axs = subplots(1, 2, figsize = (15, 5))
for ((ha, dt), r) in sort(collect(sets), by = first)
    ls = dt ? "--" : "-";  nk = r.nk
    axs[1].plot(r.t, mean_([x.KE[1:nk] for x in r.rs]), ls, color = colour(ha), lw = 0.8,
                label = @sprintf("h/a=%d  s_a=%.0f%s", ha, sa(ha), dt ? "  Δt/2" : ""))
    axs[2].plot(r.t, mean_([x.H[1:nk] for x in r.rs]), ls, color = colour(ha), lw = 1.2)
end
axs[1].axhline(1, color = "k", lw = 0.5);  axs[1].set_xlabel("t");  axs[1].set_ylabel("KE / equilibrium");  axs[1].legend(fontsize = 7)
axs[2].axhline(0, color = "k", lw = 0.5);  axs[2].set_xlabel("t");  axs[2].set_ylabel("ΔH/H")
axs[1].set_title("kinetic energy (mean of runs)");  axs[2].set_title("energy")
tight_layout();  savefig(joinpath(FIGD, "cl_energy.png"), dpi = 100);  close()

# ── fig 2: the stochastic step ───────────────────────────────────────────────
# short runs: one file per h/a in cl_short (tag short_ha<h/a>)
shortres = Dict{Int,Any}()
for f in files(SHORT)
    m = match(r"short_ha(\d+)_r", f);  m === nothing && continue
    ha = parse(Int, m[1]);  shortres[ha] = Dict(k => series(f, k) for k in ("acc_s", "acc_r", "grej_s", "grej_r", "vmax"))
end
fig, axs = subplots(2, 2, figsize = (15, 9))
for ((ha, dt), r) in sort(collect(sets), by = first)
    dt && continue;  nk = r.nk;  c = colour(ha)
    for (ax, k) in zip((axs[1, 1], axs[1, 2], axs[2, 1]), ("acc_r", "grej_r", "vmax"))
        ax.plot(r.t, avgser(r.ex[k], nk), color = c, lw = 0.8, label = @sprintf("h/a=%d  s_a=%.0f", ha, sa(ha)))
    end
    axs[1, 1].plot(r.t, avgser(r.ex["acc_s"], nk), color = c, lw = 0.8, ls = ":")
    axs[1, 2].plot(r.t, avgser(r.ex["grej_s"], nk), color = c, lw = 0.8, ls = ":")
end
axs[1, 1].set_ylabel("acceptance: residual (solid), stress (dotted)");  axs[1, 2].set_ylabel("rejected by the guard, |v| ≥ 1: residual (solid), stress (dotted)")
axs[2, 1].set_ylabel("largest particle speed");  for ax in (axs[1, 1], axs[1, 2], axs[2, 1]);  ax.set_xlabel("t");  end
axs[1, 1].legend(fontsize = 7);  axs[1, 2].set_yscale("symlog", linthresh = 1e-6)
ax = axs[2, 2]
for (k, mk, lab) in (("acc_s", "o", "acceptance, stress moves"), ("acc_r", "s", "acceptance, residual moves"),
                     ("grej_s", "^", "|v| ≥ 1, stress proposals"), ("grej_r", "<", "|v| ≥ 1, residual proposals"), ("vmax", "v", "largest |v|"))
    pts = Dict{Float64,Float64}()
    for ((ha, dt), r) in sets;  dt || (pts[sa(ha)] = mean_(avgser(r.ex[k], r.nk)));  end
    for (ha, d) in shortres;  haskey(pts, sa(ha)) || isempty(d[k]) || (pts[sa(ha)] = mean_(d[k]));  end
    xs = sort(collect(keys(pts)));  isempty(xs) && continue
    ax.semilogx(xs, [pts[x] for x in xs], mk * "-", label = lab)
end
ax.set_xlabel("entropy per particle s_a");  ax.set_title("means against s_a (short runs where no long run)")
ax.axhline(0, color = "k", lw = 0.5);  ax.invert_xaxis();  ax.legend(fontsize = 8)
tight_layout();  savefig(joinpath(FIGD, "cl_step.png"), dpi = 100);  close()
for ha in sort(collect(keys(shortres)))
    d = shortres[ha]
    @printf("short h/a=%2d s_a=%5.1f  acc_s %.3f  acc_r %.3f  |v|≥1: stress %.2e resid %.2e  vmax %.3f\n", ha, sa(ha),
            mean_(d["acc_s"]), mean_(d["acc_r"]), mean_(d["grej_s"]), mean_(d["grej_r"]), isempty(d["vmax"]) ? NaN : maximum(d["vmax"]))
end

# ── correlators ──────────────────────────────────────────────────────────────
function corr(r, field; t0 = 2.0, tmax = 30.0)
    tb = min(t0, r.t[end] / 4);  m = r.m;  n0 = m.N * m.V * m.w * m.T
    per = [shell_correlators(x.rec, field; t0 = tb, maxlag_t = min(tmax, r.t[end] / 3)) for x in r.rs]
    ks, τ = per[1][1], collect(per[1][2]);  C = mean_([p[3] for p in per]) ./ n0
    (; ks, τ, C, ν = m.etas / m.Trel, h = m.h)
end
long = sort([k for k in keys(sets) if !k[2] && sets[k].t[end] >= 8], by = first)
cs = Dict(k => (T = corr(sets[k], :MT), L = corr(sets[k], :ML)) for k in long)

fig, axs = subplots(1, 2, figsize = (14, 5))
for k in long
    c = cs[k];  ha = k[1]
    axs[1].plot(c.T.ks .* c.T.h, c.T.C[:, 1], "o-", color = colour(ha), ms = 4, label = @sprintf("h/a=%d  s_a=%.0f", ha, sa(ha)))
    axs[2].plot(c.L.ks .* c.L.h, c.L.C[:, 1], "o-", color = colour(ha), ms = 4)
end
for (ax, l) in zip(axs, ("C_T(k,0)/NVwT", "C_L(k,0)/NVwT"))
    ax.axhline(1, color = "k", lw = 0.5);  ax.set_xlabel("kh");  ax.set_ylabel(l);  ax.set_ylim(0.5, 1.5)
end
axs[1].legend(fontsize = 7);  axs[1].set_title("equal-time transverse");  axs[2].set_title("equal-time longitudinal")
tight_layout();  savefig(joinpath(FIGD, "cl_static.png"), dpi = 100);  close()

fig, axs = subplots(1, 4, figsize = (21, 4.8))
for (i, j) in enumerate((1, 3, 6)), k in long
    c = cs[k].T;  j > size(c.C, 1) && continue
    axs[i].plot(c.τ, c.C[j, :] ./ c.C[j, 1], color = colour(k[1]), lw = 1.3, label = @sprintf("h/a=%d", k[1]))
    axs[i].plot(c.τ, exp.(-c.ν * c.ks[j]^2 .* c.τ), "k:", lw = 0.8)
    axs[i].set_title(@sprintf("C_T(k,τ)/C_T(k,0),  kh = %.2f", c.ks[j] * c.h));  axs[i].set_xlabel("τ");  axs[i].axhline(0, color = "k", lw = 0.5)
end
axs[1].legend(fontsize = 7)
damp = Dict{Int,Float64}()                                   # 2γ/νk² in the smallest shells (kh ≈ 0.52)
for k in long
    c = cs[k].T;  js = [j for j in 1:size(c.C, 1) if c.ks[j] * c.h < 0.8]   # beyond, the fit range runs out
    d = [2 * fit_oscillator(c.τ, c.C[j, :] ./ c.C[j, 1])[2] / (c.ν * c.ks[j]^2) for j in js]
    o = sortperm(c.ks[js])
    axs[4].plot(c.ks[js][o] .* c.h, d[o], "o-", color = colour(k[1]), ms = 4, label = @sprintf("h/a=%d", k[1]))
    j1 = [j for j in js if c.ks[j] * c.h < 0.6];  damp[k[1]] = mean_(d[[findfirst(==(j), js) for j in j1]])
end
axs[4].axhline(1, color = "k", lw = 0.5);  axs[4].legend(fontsize = 7);  axs[4].set_xlabel("kh");  axs[4].set_ylabel("2γ/νk²  (1 = input viscosity)");  axs[4].set_title("transverse damping")
tight_layout();  savefig(joinpath(FIGD, "cl_CT.png"), dpi = 100);  close()

fig, axs = subplots(1, 3, figsize = (17, 4.8))
for (i, j) in enumerate((1, 3, 6)), k in long
    c = cs[k].L;  j > size(c.C, 1) && continue;  kk = c.ks[j];  csd = sqrt(1 / 3)
    axs[i].plot(c.τ, c.C[j, :] ./ c.C[j, 1], color = colour(k[1]), lw = 1.3, label = @sprintf("h/a=%d", k[1]))
    axs[i].plot(c.τ, cos.(csd * kk .* c.τ) .* exp.(-(4 / 3) * c.ν * kk^2 .* c.τ ./ 2), "k:", lw = 0.8)
    axs[i].set_xlim(0, 15);  axs[i].set_title(@sprintf("C_L(k,τ)/C_L(k,0),  kh = %.2f", kk * c.h));  axs[i].set_xlabel("τ")
    axs[i].axhline(0, color = "k", lw = 0.5)
end
axs[1].legend(fontsize = 7)
tight_layout();  savefig(joinpath(FIGD, "cl_CL.png"), dpi = 100);  close()

# ── spreads ──────────────────────────────────────────────────────────────────
fig, axs = subplots(1, 2, figsize = (14, 5))
for ((ha, dt), r) in sort(collect(sets), by = first)
    dt && continue;  c = colour(ha)
    axs[1].plot(r.t, avgser(r.ex["sspr"], r.nk), color = c, lw = 1, label = @sprintf("h/a=%d", ha))
    axs[1].axhline(sqrt(3 / sa(ha)), color = c, lw = 0.7, ls = ":")
    axs[2].plot(r.t, avgser(r.ex["Vspr"], r.nk), color = c, lw = 1)
end
axs[1].set_xlabel("t");  axs[1].set_ylabel("rms(s_a/s̄ − 1)");  axs[1].set_yscale("log");  axs[1].legend(fontsize = 7)
axs[1].set_title("entropy per particle; dotted √(3/s_a)")
axs[2].set_xlabel("t");  axs[2].set_ylabel("rms(V_a/V̄ − 1)");  axs[2].set_yscale("log");  axs[2].set_title("particle volumes 1/n_a")
tight_layout();  savefig(joinpath(FIGD, "cl_spread.png"), dpi = 100);  close()

# ── pairs ────────────────────────────────────────────────────────────────────
fig, axs = subplots(1, 3, figsize = (19, 5))
Dself = Dict{Int,Tuple{Float64,Float64}}()
for ((ha, dt), r) in sort(collect(sets), by = first)
    dt && continue;  ps = [x.part for x in r.rs if x.part !== nothing];  isempty(ps) && continue
    nt = minimum(length(p.t) for p in ps);  tp = ps[1].t[1:nt];  c = colour(ha);  a = r.m.a
    D2 = mean_([p.D2[1, 1:nt] for p in ps]);  MSD = mean_([p.MSD[1:nt] for p in ps])
    axs[1].loglog(tp, D2 ./ a^2, color = c, lw = 1.2, label = @sprintf("h/a=%d", ha))
    axs[2].loglog(tp, MSD ./ r.m.h^2, color = c, lw = 1.2, label = @sprintf("h/a=%d", ha))
    it = findall(>=(tp[end] / 2), tp);  length(it) > 3 || continue
    ds = [slope(tp[it], p.MSD[it]) / 4 for p in ps]
    m = r.m;  ν = m.etas / m.Trel;  Tw = m.T / m.w
    Dself[ha] = (mean_(ds), Tw / (4π * ν) * log((π / a) * m.L[1] / 2π))
end
axs[1].set_xlabel("t");  axs[1].set_ylabel("D²(r = a, t)/a²");  axs[1].set_title("separation of nearest neighbours");  axs[1].legend(fontsize = 7)
axs[2].set_xlabel("t");  axs[2].set_ylabel("MSD / h²");  axs[2].set_title("mean square displacement, in units of h")
if !isempty(Dself)
    xs = sort(collect(keys(Dself)))
    axs[3].semilogx([sa(x) for x in xs], [Dself[x][1] for x in xs], "o-", label = "measured D_self")
    axs[3].semilogx([sa(x) for x in xs], [Dself[x][2] for x in xs], "k--", label = "hydrodynamics (T/w)/(4πν) ln(πL/2πa)")
    axs[3].invert_xaxis();  axs[3].set_xlabel("s_a");  axs[3].set_ylabel("self-diffusion D_self");  axs[3].legend(fontsize = 8)
    for x in xs;  @printf("D_self h/a=%2d: %.2e  hydro %.2e  ratio %.2f\n", x, Dself[x]..., Dself[x][1] / Dself[x][2]);  end
end
tight_layout();  savefig(joinpath(FIGD, "cl_pairs.png"), dpi = 100);  close()
# ── damping against the particle walk ────────────────────────────────────────
ν0 = sets[(2, false)].m.etas / sets[(2, false)].m.Trel
xs = sort([x for x in keys(damp) if haskey(Dself, x)])
if !isempty(xs)
    figure(figsize = (7.5, 5))
    plot([sa(x) for x in xs], [damp[x] for x in xs], "o-", label = "measured 2γ/νk², kh ≈ 0.52")
    plot([sa(x) for x in xs], [1 + Dself[x][1] / ν0 for x in xs], "s--", label = "1 + D_self/ν  (momentum carried by the particle walk)")
    xscale("log");  gca().invert_xaxis();  axhline(1, color = "k", lw = 0.5)
    xlabel("s_a");  ylabel("transverse damping / input");  legend(fontsize = 8)
    tight_layout();  savefig(joinpath(FIGD, "cl_walk.png"), dpi = 100);  close()
    for x in xs;  @printf("h/a=%2d  2γ/νk² = %.2f   1 + D_self/ν = %.2f\n", x, damp[x], 1 + Dself[x][1] / ν0);  end
end
println("figures written to $FIGD")
