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
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "boost")
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
const SETS = [("b_v0", 0.0, "C0"), ("b_v0.3", 0.3, "C2"), ("b_v0.6", 0.6, "C1"), ("b_v0.8", 0.8, "C3")]
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

res = Dict{String,Any}()
for (run, v, col) in SETS
    fs = files(joinpath(DIR, run));  isempty(fs) && continue
    rs = [read_equilibrium(f) for f in fs];  rs = filter(r -> r.rec.energies && length(r.rec.t) > 1000, rs)
    isempty(rs) && continue
    m = rs[1].meta;  ks = rs[1].rec.ks;  NV = m.N * m.V
    ipar = sort([i for i in eachindex(ks) if abs(ks[i][2]) < 1e-9 && ks[i][1] > 0], by = i -> ks[i][1])
    iperp = sort([i for i in eachindex(ks) if abs(ks[i][1]) < 1e-9 && ks[i][2] > 0], by = i -> ks[i][2])
    C = Dict{Tuple{Symbol,Int},Any}()
    for (dir, idx) in ((:par, ipar), (:perp, iperp)), (j, i) in enumerate(idx[1:min(3, end)])
        per = [corr_run(r.rec, i, NV) for r in rs];  nl = minimum(size(p[1], 3) for p in per)
        C[(dir, j)] = (C = mean_([p[1][:, :, 1:nl] for p in per]), τ = per[1][2][1:nl], k = ks[i])
    end
    Trel = m.Trel;  ν = m.etas / Trel
    nk = minimum(length(r.KE) for r in rs)
    res[run] = (; v, col, C, Trel, ν, h = m.h, n = length(rs), t = rs[1].rec.t[end],
                KE = mean_([mean_(r.KE[nk÷2:nk]) for r in rs]), dH = mean_([r.H[nk] for r in rs]))
    @printf("%-8s v=%.1f  %2d runs  t=%.0f  T_rel=%.3f  KE(fluct)/(2VwT)=%.3f  ΔH/H=%+.2e\n", run, v, length(rs),
            res[run].t, Trel, res[run].KE, res[run].dH)
end
runs = [s[1] for s in SETS if haskey(res, s[1])]
isempty(runs) && (println("no complete runs yet");  exit())
names = ("E", "M_x", "M_y")

# ── equal time ───────────────────────────────────────────────────────────────────────────
fig, axs = subplots(1, 3, figsize = (17, 4.8))
println("equal time, C_AA(k,0)/χ_AA (smallest k):")
for run in runs
    r = res[run];  χ = chi(r.Trel, r.v)
    for (a, ax) in enumerate(axs)
        for (dir, mk) in ((:par, "o"), (:perp, "s"))
            js = sort([j for (d, j) in keys(r.C) if d === dir])
            kh = [norm(r.C[(dir, j)].k) * r.h for j in js];  y = [real(r.C[(dir, j)].C[a, a, 1]) / χ[a, a] for j in js]
            ax.plot(kh, y, mk * "-", color = r.col, mfc = dir === :perp ? "none" : r.col,
                    label = @sprintf("v=%.1f, k %s v", r.v, dir === :par ? "∥" : "⊥"))
        end
    end
    @printf("  v=%.1f  k∥: %s   k⊥: %s\n", r.v,
            join([@sprintf("%s %.3f", names[a], real(r.C[(:par, 1)].C[a, a, 1]) / χ[a, a]) for a in 1:3], "  "),
            join([@sprintf("%s %.3f", names[a], real(r.C[(:perp, 1)].C[a, a, 1]) / χ[a, a]) for a in 1:3], "  "))
end
for (a, ax) in enumerate(axs)
    ax.axhline(1, color = "k", lw = 0.5);  ax.set_ylim(0.7, 1.3);  ax.set_xlabel("kh")
    ax.set_title("C_$(names[a])$(names[a])(k,0) / χ  (filled: k ∥ v, open: k ⊥ v)")
end
axs[1].legend(fontsize = 6, ncol = 2)
tight_layout();  savefig(joinpath(FIGD, "boost_static.png"), dpi = 100);  close()

# ── dynamics: one figure per (direction, component) family ───────────────────────────────
function panel!(ax, r, dir, a, b; part = real, j = 1, tmax = 60.0)
    c = r.C[(dir, j)];  it = findall(<=(tmax), c.τ);  τ = c.τ[it]
    pred, χ, _ = predict(r.Trel, r.v, r.ν, c.k, τ)
    y = part.(c.C[a, b, it]) ./ real(c.C[a, b, 1]);  yp = [part(p[a, b]) for p in pred] ./ χ[a, b]
    ax.plot(τ, y, color = r.col, lw = 1.3, label = @sprintf("v=%.1f", r.v))
    ax.plot(τ, yp, color = r.col, lw = 0.9, ls = ":")
end
for (fname, specs) in (("boost_shear", [(:par, 3, 3, real, "Re C_yy, k ∥ v (advected shear)", 80.0),
                                        (:par, 3, 3, imag, "Im C_yy, k ∥ v", 80.0)]),
                       ("boost_par",   [(:par, 2, 2, real, "Re C_xx, k ∥ v (sound along the flow)", 15.0),
                                        (:par, 1, 1, real, "Re C_EE, k ∥ v", 15.0)]),
                       ("boost_perp",  [(:perp, 3, 3, real, "Re C_yy, k ⊥ v (sound across the flow)", 15.0),
                                        (:perp, 2, 2, real, "Re C_xx, k ⊥ v (shear + sound)", 80.0)]))
    fig, axs = subplots(1, 2, figsize = (15, 5))
    for (ax, (dir, a, b, part, title, tmax)) in zip(axs, specs)
        for run in runs;  panel!(ax, res[run], dir, a, b; part, tmax);  end
        k = res[runs[1]].C[(dir, 1)].k
        ax.set_title(title * @sprintf(",  kh = %.2f;  dotted: hydrodynamics", norm(k) * res[runs[1]].h), fontsize = 10)
        ax.axhline(0, color = "k", lw = 0.5);  ax.set_xlabel("τ");  ax.legend(fontsize = 8)
    end
    tight_layout();  savefig(joinpath(FIGD, fname * ".png"), dpi = 100);  close()
end

# ── fits of the shear modes: C ≈ A e^{−iΩτ − Γτ} on the early part ────────────────────────
println("shear modes (smallest k): fitted Ω/k and Γ against hydrodynamics")
fit = Dict{String,Any}()
for run in runs
    r = res[run];  out = []
    for (dir, a) in ((:par, 3), (:perp, 2))
        c = r.C[(dir, 1)];  y = c.C[a, a, :] ./ c.C[a, a, 1]
        iend = findfirst(i -> abs(y[i]) < 0.25, eachindex(y));  it = 2:(iend === nothing ? length(y) : iend - 1)
        τ = c.τ[it];  ph = unwrap(angle.(y[it]));  la = log.(abs.(y[it]))
        sl(x, z) = (xm = mean_(x); sum((x .- xm) .* (z .- mean_(z))) / sum(abs2, x .- xm))
        Ω = -sl(τ, ph);  Γ = -sl(τ, la);  k = norm(c.k);  γ = 1 / sqrt(1 - r.v^2)
        Γth = dir === :par ? r.ν * k^2 / γ^3 : r.ν * k^2 / γ;  Ωth = dir === :par ? r.v * k : 0.0
        push!(out, (; dir, Ω, Γ, Ωth, Γth, k))
        @printf("  v=%.1f %-4s  Ω/k = %+.3f (hydro %+.3f)   Γ = %.4f (hydro %.4f, ratio %.2f)\n", r.v, dir, Ω / k, Ωth / k, Γ, Γth, Γ / Γth)
    end
    fit[run] = out
end
fig, axs = subplots(1, 2, figsize = (13, 4.8))
vs = [res[r].v for r in runs]
axs[1].plot(vs, [fit[r][1].Ω / fit[r][1].k for r in runs], "o", ms = 8, label = "measured, k ∥ v");  axs[1].plot([0, 0.85], [0, 0.85], "k:", label = "Ω/k = v")
axs[1].set_xlabel("v");  axs[1].set_ylabel("advection speed Ω/k of the shear mode");  axs[1].legend(fontsize = 8)
axs[2].plot(vs, [fit[r][1].Γ / fit[r][1].Γth for r in runs], "o", ms = 8, label = "k ∥ v, against νk²/γ³")
axs[2].plot(vs, [fit[r][2].Γ / fit[r][2].Γth for r in runs], "s", ms = 8, mfc = "none", label = "k ⊥ v, against νk²/γ")
axs[2].axhline(1, color = "k", lw = 0.5);  axs[2].set_xlabel("v");  axs[2].set_ylabel("Γ / hydrodynamics");  axs[2].legend(fontsize = 8)
tight_layout();  savefig(joinpath(FIGD, "boost_fit.png"), dpi = 100);  close()
println("figures written to $FIGD")
