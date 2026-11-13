# boost_analysis.jl — campaign `boost` (docs/boost.tex): a fluid moving with v along x.
# Direction-resolved complex correlators of U = (E, M_x, M_y) for k along the flow (k = k x̂)
# and across it (k = k ŷ), against
#   equal time: the susceptibilities χ_ij = ∂⟨U_i⟩/∂θ_j, θ = (−β⁰, β^x, β^y)  (boost.tex, eq. chi)
#   dynamics:   C(k,τ) = Σ_n P_n χ e^{−iω_n τ}, ω_n the eigenvalues of the ideal K(k), each damped
#               by its boosted rest-frame damping (boost.tex, eq. damp).
# Figures docs/figs/boost_*.png:
#   boost_static.png   C_AA(k,0)/χ_AA against kh, A = E, M_x, M_y, both directions, each v
#   boost_shear.png    k ∥ v: Re and Im of C_yy(k,τ)/C_yy(k,0) (advected shear) with the prediction
#   boost_par.png      k ∥ v: Re C_xx and Re C_EE normalized (Doppler-shifted sound)
#   boost_perp.png     k ⊥ v: Re C_yy (sound across the flow) and Re C_xx (shear + sound)
#   boost_fit.png      fitted advection speed Ω/k and damping Γ of the shear modes against v
# Run (login node):  julia --project examples/boost_analysis.jl [dir]
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, HDF5, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "boost")
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
# sets: (label, directories pooled, v, colour); BOOST_GROUP = L12 (default) or L24; figures get the suffix
const GROUP = get(ENV, "BOOST_GROUP", "L12")
# ETA: the follow-up at η/s = 0.1, 0.4 and without the stochastic step (Γ/hydro there is against η/s = 0.2)
const SETS = GROUP == "L24" ?
    [("v=0", ["b24_v0"], 0.0, "C0"), ("v=0.6", ["b24_v0.6"], 0.6, "C1"), ("v=0.8", ["b24_v0.8"], 0.8, "C3")] :
             GROUP == "GRID" ?                    # the grid step (gridstep.tex), run with: julia ... $SCRATCH/sph/grid
    [("v=0", ["g_v0"], 0.0, "C0"), ("v=0.3", ["g_v0.3"], 0.3, "C2"), ("v=0.6", ["g_v0.6"], 0.6, "C1"),
     ("v=0.8", ["g_v0.8", "g_v0.8_long"], 0.8, "C3")] :
             GROUP == "GSTATS" ?                  # high statistics, $SCRATCH/sph/grid_stats
    [("v=0", ["gs_v0"], 0.0, "C0"), ("v=0.3", ["gs_v0.3"], 0.3, "C2"), ("v=0.6", ["gs_v0.6"], 0.6, "C1"),
     ("v=0.8", ["gs_v0.8"], 0.8, "C3")] :
             GROUP == "GSTATS24" ?
    [("v=0", ["gs24_v0"], 0.0, "C0"), ("v=0.6", ["gs24_v0.6"], 0.6, "C1"), ("v=0.8", ["gs24_v0.8"], 0.8, "C3")] :
             GROUP == "K4" ?                      # particles with the C4 kernel, $SCRATCH/sph/kernel
    [("v=0", ["k4_v0"], 0.0, "C0"), ("v=0.8", ["k4_v0.8"], 0.8, "C3")] :
             GROUP == "GRID24" ?
    [("v=0", ["g24_v0"], 0.0, "C0"), ("v=0.6", ["g24_v0.6"], 0.6, "C1"), ("v=0.8", ["g24_v0.8"], 0.8, "C3")] :
             GROUP == "ETA" ?
    [("v=0 η/s=0.1", ["b_e01_v0E"], 0.0, "C0"), ("v=0.8 η/s=0.1", ["b_e01_v0.8"], 0.8, "C3"),
     ("v=0 η/s=0.4", ["b_e04_v0E"], 0.0, "C2"), ("v=0.8 η/s=0.4", ["b_e04_v0.8"], 0.8, "C1"),
     ("v=0 ideal", ["b_nv_v0E"], 0.0, "C4"), ("v=0.8 ideal", ["b_nv_v0.8"], 0.8, "C5")] :
    [("v=0", ["b_v0"], 0.0, "C0"), ("v=0.3", ["b_v0.3"], 0.3, "C2"), ("v=0.6", ["b_v0.6"], 0.6, "C1"),
     ("v=0.8", ["b_v0.8", "b_v0.8_long"], 0.8, "C3"), ("v=0.8, Δt/2", ["b_v0.8_dt2"], 0.8, "C4")]
const SUFFIX = GROUP == "L12" ? "" : "_" * GROUP
const T0BURN = 5.0;  const DTS = 0.05;  const TAUMAX = 100.0
const CS2 = 1 / 3
mean_(v) = sum(v) / length(v)

# ── thermodynamics of the boosted conformal fluid (e = T⁴ in code units) ─────────────────
function Uof(θ)                     # θ = (−β⁰, β^x, β^y) → (ε, M_x, M_y)
    β0 = -θ[1];  b = SVector(θ[2], θ[3]);  T = 1 / sqrt(β0^2 - dot(b, b));  v = b / β0
    γ2 = 1 / (1 - dot(v, v));  e = T^4;  p = CS2 * e;  w = e + p
    SVector(w * γ2 - p, w * γ2 * v[1], w * γ2 * v[2])
end
function chi(Trel, v)               # susceptibilities per unit volume
    γ = 1 / sqrt(1 - v^2);  θ = SVector(-γ / Trel, γ * v / Trel, 0.0);  J = zeros(3, 3)
    for j in 1:3
        d = 1e-6 * maximum(abs, θ);  e = SVector{3}(i == j ? d : 0.0 for i in 1:3)
        J[:, j] = (Uof(θ + e) - Uof(θ - e)) / 2d
    end
    (J + J') / 2
end
# ideal fluxes F_j(U), U = (ε, M_x, M_y): F_j = (M_j, M_x v_j + p δ_jx, M_y v_j + p δ_jy)
function vel(U)                     # conformal: M = (ε + p) v, p = c_s²(ε − M·v); solve |v|
    ε = U[1];  M = SVector(U[2], U[3]);  m = norm(M);  m == 0 && return (zero(M), CS2 * ε)
    f(x) = (ε + CS2 * (ε - m * x)) * x - m;  lo, hi = 0.0, 1.0
    for _ in 1:200;  mid = (lo + hi) / 2;  f(mid) > 0 ? (hi = mid) : (lo = mid);  end
    x = (lo + hi) / 2;  (M / m * x, CS2 * (ε - m * x))
end
flux(U, j) = (v = vel(U);  p = v[2];  vv = v[1];
              SVector(U[1 + j], U[2] * vv[j] + (j == 1 ? p : 0.0), U[3] * vv[j] + (j == 2 ? p : 0.0)))
function Kmat(U, k)
    K = zeros(3, 3)
    for i in 1:3
        d = 1e-7 * abs(U[1]);  e = SVector{3}(l == i ? d : 0.0 for l in 1:3)
        for j in 1:2
            k[j] == 0 && continue
            K[:, i] += k[j] * (flux(U + e, j) - flux(U - e, j)) / 2d
        end
    end
    K
end
# predicted C(k,τ) (3×3 per τ): ideal modes of K, each damped by its boosted rest-frame damping
function predict(Trel, v, ν, k, τs)
    γ = 1 / sqrt(1 - v^2);  U = Uof(SVector(-γ / Trel, γ * v / Trel, 0.0));  χ = chi(Trel, v)
    F = eigen(Kmat(U, k));  ω0 = real.(F.values);  R = F.vectors;  Rinv = inv(R)
    ωs = ComplexF64[]
    for ω in ω0
        Ω = γ * (ω - v * k[1]);  kx′ = γ * (k[1] - v * ω);  k2′ = kx′^2 + k[2]^2
        if abs(Ω) < 1e-3 * norm(k)                        # shear
            Γ′ = ν * k2′;  dΩ = 0.0
        else                                              # sound
            Γ′ = 0.5 * (4 / 3) * ν * k2′;  dΩ = CS2 * kx′ / Ω
        end
        push!(ωs, ω - im * Γ′ / (γ * (1 + v * dΩ)))
    end
    [R * Diagonal(exp.(-im .* ωs .* τ)) * Rinv * χ for τ in τs], χ, ωs
end
# left eigenvector ℓ of K for the shear mode: ℓ·U is the shear amplitude (no sound in it)
function shear_left(Trel, v, k)
    γ = 1 / sqrt(1 - v^2);  U = Uof(SVector(-γ / Trel, γ * v / Trel, 0.0))
    F = eigen(Kmat(U, k));  Rinv = inv(F.vectors)
    i = argmin([abs(γ * (real(ω) - v * k[1])) for ω in F.values])
    real.(Rinv[i, :])
end

# ── measured correlators ─────────────────────────────────────────────────────────────────
function files(d)
    isdir(d) || return String[]
    [joinpath(d, f) for f in readdir(d) if endswith(f, ".h5") && startswith(f, "equilibrium_")]
end
# U_k(t) for one run and one wave vector index: (E, M_x, M_y) from (E, ML, MT)
function useries(rec, i)
    k = rec.ks[i];  kh = k / norm(k);  n = length(rec.t)
    E = [rec.E[s][i] for s in 1:n];  ML = [rec.ML[s][i] for s in 1:n];  MT = [rec.MT[s][i] for s in 1:n]
    Mx = kh[1] .* ML .- kh[2] .* MT;  My = kh[2] .* ML .+ kh[1] .* MT
    (E, Mx, My)
end
function corr_run(rec, i, NV)
    t = rec.t;  dtr = t[2] - t[1];  stride = max(1, round(Int, DTS / dtr));  i0 = findfirst(>=(T0BURN), t)
    sel = i0:stride:length(t);  us = [x[sel] for x in useries(rec, i)];  n = length(sel)
    nl = min(round(Int, TAUMAX / (stride * dtr)), n ÷ 2)
    C = zeros(ComplexF64, 3, 3, nl + 1)
    for a in 1:3, b in 1:3, l in 0:nl
        s = 0.0im;  A = us[a];  B = us[b]
        @inbounds for q in 1:n-l;  s += A[q+l] * conj(B[q]);  end
        C[a, b, l+1] = s / (n - l) / NV
    end
    (C, collect(0:nl) .* stride .* dtr)
end

function unwrap(p)
    q = copy(p)
    for i in 2:length(q);  d = q[i] - q[i-1];  q[i] -= 2π * round(d / 2π);  end
    q
end

# jackknife over runs: f applied to the mean of all runs but one
function jack(f, Cs)
    n = length(Cs);  tot = sum(Cs);  vals = [f((tot - Cs[i]) / (n - 1)) for i in 1:n];  m = sum(vals) / n
    (f(tot / n), sqrt((n - 1) / n * sum(abs2, vals .- m)))
end

res = Dict{String,Any}()
for (run, dirs, v, col) in SETS
    fs = vcat([files(joinpath(DIR, d)) for d in dirs]...);  isempty(fs) && continue
    rs = [read_equilibrium(f) for f in fs];  rs = filter(r -> r.rec.energies && length(r.rec.t) > 1000, rs)
    isempty(rs) && continue
    m = rs[1].meta;  ks = rs[1].rec.ks;  NV = m.N * m.V
    ipar = sort([i for i in eachindex(ks) if abs(ks[i][2]) < 1e-9 && ks[i][1] > 0], by = i -> ks[i][1])
    iperp = sort([i for i in eachindex(ks) if abs(ks[i][1]) < 1e-9 && ks[i][2] > 0], by = i -> ks[i][2])
    C = Dict{Tuple{Symbol,Int},Any}()
    for (dir, idx) in ((:par, ipar), (:perp, iperp)), (j, i) in enumerate(idx[1:min(3, end)])
        per = [corr_run(r.rec, i, NV) for r in rs];  nl = minimum(size(p[1], 3) for p in per)
        Cs = [p[1][:, :, 1:nl] for p in per]
        C[(dir, j)] = (C = mean_(Cs), Cs = Cs, τ = per[1][2][1:nl], k = ks[i])
    end
    Trel = m.Trel;  ν = m.etas / Trel
    nk = minimum(length(r.KE) for r in rs)
    res[run] = (; v, col, C, lab = run, Trel, ν, h = m.h, n = length(rs), t = rs[1].rec.t[end],
                KE = mean_([mean_(r.KE[nk÷2:nk]) for r in rs]), dH = mean_([r.H[nk] for r in rs]))
    @printf("%-15s v=%.1f  %2d runs  t=%.0f  T_rel=%.3f  KE(fluct)/(2VwT)=%.3f  ΔH/H=%+.2e\n", run, v, length(rs),
            res[run].t, Trel, res[run].KE, res[run].dH)
end
runs = [s[1] for s in SETS if haskey(res, s[1])]
fitruns = filter(r -> !occursin("Δt", r), runs)      # the Δt/2 set: in the tables only
isempty(runs) && (println("no complete runs yet");  exit())
names = ("E", "M_x", "M_y")

# ── equal time ───────────────────────────────────────────────────────────────────────────
fig, axg = subplots(length(runs), 3, figsize = (15, 2.9 * length(runs)), squeeze = false)
println("equal time, C_AA(k,0)/χ_AA (smallest k):")
for (row, run) in enumerate(runs)
    r = res[run];  χ = chi(r.Trel, r.v)
    for a in 1:3
        ax = axg[row, a]
        for (dir, mk) in ((:par, "o"), (:perp, "s"))
            js = sort([j for (d, j) in keys(r.C) if d === dir])
            kh = [norm(r.C[(dir, j)].k) * r.h for j in js];  y = [real(r.C[(dir, j)].C[a, a, 1]) / χ[a, a] for j in js]
            ax.plot(kh, y, mk * "-", color = r.col, mfc = dir === :perp ? "none" : r.col,
                    label = dir === :par ? "k ∥ v" : "k ⊥ v")
        end
        ax.axhline(1, color = "k", lw = 0.5);  ax.set_ylim(0.7, 1.3)
        row == length(runs) && ax.set_xlabel("kh")
        ax.set_title("$(r.lab):  C_$(names[a])$(names[a])(k,0) / χ", fontsize = 10)
        a == 1 && ax.legend(fontsize = 8)
    end
    for j in 1:2
        eq(dir, a) = jack(C -> real(C[a, a, 1]) / χ[a, a], r.C[(dir, j)].Cs)
        @printf("  %-15s kh=%.2f  k∥: %s   k⊥: %s\n", r.lab, norm(r.C[(:par, j)].k) * r.h,
                join([@sprintf("%s %.3f±%.3f", names[a], eq(:par, a)...) for a in 1:3], "  "),
                join([@sprintf("%s %.3f±%.3f", names[a], eq(:perp, a)...) for a in 1:3], "  "))
    end
end
tight_layout();  savefig(joinpath(FIGD, "boost_static" * SUFFIX * ".png"), dpi = 100);  close()

# ── dynamics: one figure per (direction, component) family ───────────────────────────────
function panel!(ax, r, dir, a, b; part = real, j = 1, tmax = 60.0)
    c = r.C[(dir, j)];  it = findall(<=(tmax), c.τ);  τ = c.τ[it]
    pred, χ, _ = predict(r.Trel, r.v, r.ν, c.k, τ)
    y = part.(c.C[a, b, it]) ./ real(c.C[a, b, 1]);  yp = [part(p[a, b]) for p in pred] ./ χ[a, b]
    ax.plot(τ, y, color = r.col, lw = 1.3, label = "measured")
    ax.plot(τ, yp, color = "k", lw = 0.9, ls = "--", label = "hydrodynamics")
end
for (fname, specs) in (("boost_shear", [(:par, 3, 3, real, "Re C_yy, k ∥ v (advected shear)", 80.0),
                                        (:par, 3, 3, imag, "Im C_yy, k ∥ v", 80.0)]),
                       ("boost_par",   [(:par, 2, 2, real, "Re C_xx, k ∥ v (sound along the flow)", 15.0),
                                        (:par, 1, 1, real, "Re C_EE, k ∥ v", 15.0)]),
                       ("boost_perp",  [(:perp, 3, 3, real, "Re C_yy, k ⊥ v (sound across the flow)", 15.0),
                                        (:perp, 2, 2, real, "Re C_xx, k ⊥ v (shear + sound)", 80.0)]))
    fig, axg = subplots(length(fitruns), 2, figsize = (15, 3.0 * length(fitruns)), squeeze = false)
    for (row, run) in enumerate(fitruns), (col, (dir, a, b, part, title, tmax)) in enumerate(specs)
        ax = axg[row, col];  panel!(ax, res[run], dir, a, b; part, tmax)
        k = res[run].C[(dir, 1)].k
        ax.set_title(res[run].lab * ":  " * title * @sprintf(",  kh = %.2f", norm(k) * res[run].h), fontsize = 10)
        ax.axhline(0, color = "k", lw = 0.5);  ax.set_ylim(-1.1, 1.1)
        row == length(fitruns) && ax.set_xlabel("τ")
        row == 1 && col == 1 && ax.legend(fontsize = 8, loc = "lower left")
    end
    tight_layout();  savefig(joinpath(FIGD, fname * SUFFIX * ".png"), dpi = 100);  close()
end

# ── fits of the shear modes: C ≈ A e^{−iΩτ − Γτ} on the early part ────────────────────────
println("shear modes (two smallest k): fitted Ω/k and Γ against hydrodynamics")
fit = Dict{String,Any}()
for run in runs
    r = res[run];  out = []
    for j in 1:2, (dir, a) in ((:par, 3), (:perp, 2))
        haskey(r.C, (dir, j)) || continue
        c = r.C[(dir, j)];  k = norm(c.k);  γ = 1 / sqrt(1 - r.v^2)
        sl(x, z) = (xm = mean_(x); sum((x .- xm) .* (z .- mean_(z))) / sum(abs2, x .- xm))
        # project on the shear mode, ℓ C(τ) ℓᵀ with ℓ the left eigenvector of K (across the flow
        # C_xx alone also carries sound when v ≠ 0); fit window from the full mean (|C/C0| > 0.25),
        # then the same window for every jackknife sample
        ℓ = shear_left(r.Trel, r.v, c.k)
        proj(C) = [dot(ℓ, C[:, :, i] * ℓ) for i in axes(C, 3)]
        y0 = proj(c.C) ./ proj(c.C)[1];  iend = findfirst(i -> abs(y0[i]) < 0.25, eachindex(y0))
        it = 2:(iend === nothing ? length(y0) : iend - 1);  τ = c.τ[it]
        fΩ(C) = (P = proj(C);  y = P[it] ./ P[1];  -sl(τ, unwrap(angle.(y))))
        fΓ(C) = (P = proj(C);  y = P[it] ./ P[1];  -sl(τ, log.(abs.(y))))
        Ω, dΩ = jack(fΩ, c.Cs);  Γ, dΓ = jack(fΓ, c.Cs)
        Γth = dir === :par ? r.ν * k^2 / γ^3 : r.ν * k^2 / γ;  Ωth = dir === :par ? r.v * k : 0.0
        j == 1 && push!(out, (; dir, Ω, Γ, dΩ, dΓ, Ωth, Γth, k))
        @printf("  %-15s kh=%.2f %-4s  Ω/k = %+.3f±%.3f (hydro %+.3f)   Γ/hydro = %.2f±%.2f   window τ ≤ %.1f\n", r.lab, k * r.h, dir,
                Ω / k, dΩ / k, Ωth / k, Γ / Γth, dΓ / Γth, τ[end])
    end
    fit[run] = out
end
fig, axs = subplots(1, 2, figsize = (13, 4.8))
vs = [res[r].v for r in fitruns]
axs[1].errorbar(vs, [fit[r][1].Ω / fit[r][1].k for r in fitruns], yerr = [fit[r][1].dΩ / fit[r][1].k for r in fitruns], fmt = "o", ms = 8, label = "measured, k ∥ v");  axs[1].plot([0, 0.85], [0, 0.85], "k:", label = "Ω/k = v")
axs[1].set_xlabel("v");  axs[1].set_ylabel("advection speed Ω/k of the shear mode");  axs[1].legend(fontsize = 8)
axs[2].errorbar(vs, [fit[r][1].Γ / fit[r][1].Γth for r in fitruns], yerr = [fit[r][1].dΓ / fit[r][1].Γth for r in fitruns], fmt = "o", ms = 8, label = "k ∥ v, against νk²/γ³")
axs[2].errorbar(vs .+ 0.01, [fit[r][2].Γ / fit[r][2].Γth for r in fitruns], yerr = [fit[r][2].dΓ / fit[r][2].Γth for r in fitruns], fmt = "s", ms = 8, mfc = "none", label = "k ⊥ v, against νk²/γ")
axs[2].axhline(1, color = "k", lw = 0.5);  axs[2].set_xlabel("v");  axs[2].set_ylabel("Γ / hydrodynamics");  axs[2].legend(fontsize = 8)
tight_layout();  savefig(joinpath(FIGD, "boost_fit" * SUFFIX * ".png"), dpi = 100);  close()
println("figures written to $FIGD")
