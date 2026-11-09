# stochrep_correlators.jl — momentum correlators from stochrep_equilibrium.jl runs.
#
#   C_T(k, τ) = ⟨M_T(t+τ, k) M_T*(t, k)⟩ / (N V w T)   →  e^{−ν k² τ},            ν = η/w
#   C_L(k, τ) = ⟨M_L(t+τ, k) M_L*(t, k)⟩ / (N V w T)   →  cos(c_s k τ) e^{−Γ k² τ/2},  Γ = (4/3)η/w
# averaged over the start time t (after a burn-in), over the k of each shell |k|, and
# over realizations (errors: spread over realizations).  Also the fitted transverse
# decay rate, as an effective ν(k).
#
# Run:  julia --project examples/stochrep_correlators.jl tag [t_burn] [maxlag]
#   reads data/equilibrium_<tag>_r*.jls → docs/figs/correlators_<tag>.png
using PyPlot, StaticArrays, LinearAlgebra, Printf, Serialization, Statistics, LsqFit
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "rng.jl"))
include(joinpath(@__DIR__, "..", "src", "stochrep.jl"))
include(joinpath(@__DIR__, "..", "src", "measure.jl"))

const TAG    = ARGS[1]
const TBURN  = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 2.0
const MAXLAG = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 6.0
dir = joinpath(@__DIR__, "..", "data")
files = filter(f -> startswith(f, "equilibrium_$(TAG)_r") && endswith(f, ".jls"), readdir(dir))
runs = [deserialize(joinpath(dir, f)) for f in files]
m = runs[1].meta
norm0 = m.N * m.V * m.w * m.T
ν = m.etas / m.Trel;  Γs = 4 / 3 * ν;  cs = sqrt(m.cs2)
@printf("%s: %d realizations, N = %d, t_end = %.1f, ν = η/w = %.3e\n", TAG, length(runs), m.N, runs[1].rec.t[end], ν)

res = Dict{Symbol,Any}()
for f in (:MT, :ML)
    per = [shell_correlators(r.rec, f; t0 = TBURN, maxlag_t = MAXLAG) for r in runs]
    ks, τ = per[1][1], per[1][2]
    Cs = cat([p[3] for p in per]...; dims = 3) ./ norm0
    res[f] = (k = ks, τ = τ, C = dropdims(mean(Cs; dims = 3); dims = 3),
              E = dropdims(std(Cs; dims = 3); dims = 3) ./ sqrt(length(runs)))
end
ks = res[:MT].k;  τ = res[:MT].τ;  nsh = min(length(ks), 6)

# Fitted transverse rate: C_T = A e^{−γτ} over τ < 1/(ν k²), weighted by the errors
# (LsqFit); ν_fit = γ/k² with its error.
fits = map(1:length(ks)) do j
    C = res[:MT].C[j, :];  E = max.(res[:MT].E[j, :], 1e-6)
    sel = findall(i -> τ[i] < 1 / (ν * ks[j]^2), eachindex(τ))
    length(sel) < 4 && return (NaN, NaN)
    model(t, p) = p[1] .* exp.(-p[2] .* t)
    f = curve_fit(model, τ[sel], C[sel], 1 ./ E[sel] .^ 2, [C[1], ν * ks[j]^2])
    se = try stderror(f) catch; [NaN, NaN] end
    (f.param[2] / ks[j]^2, se[2] / ks[j]^2)
end
νfit = [f[1] for f in fits];  νerr = [f[2] for f in fits]

# Equal-time |M_T|² per realization: binning error and autocorrelation time
for j in 1:length(ks)
    kv = ks[j];  idx = findall(k -> abs(norm(k) - kv) < 1e-6, runs[1].rec.ks)
    i0 = findfirst(>=(TBURN), runs[1].rec.t)
    st = [binned_mean([abs2(r.rec.MT[t][i]) / norm0 for t in i0:length(r.rec.t)]) for r in runs for i in idx]
    τint = mean(s[3] for s in st) * (runs[1].rec.t[2] - runs[1].rec.t[1])
    @printf("  k h = %.3f  C_T(0) = %.3f ± %.3f  C_L(0) = %.3f ± %.3f  ν_fit/ν = %.2f ± %.2f  τ_int(|M_T|²) = %.2f  (1/(2νk²) = %.2f)\n",
            kv * m.h, res[:MT].C[j, 1], res[:MT].E[j, 1], res[:ML].C[j, 1], res[:ML].E[j, 1],
            νfit[j] / ν, νerr[j] / ν, τint, 1 / (2ν * kv^2))
end

fig, axs = subplots(1, 3, figsize=(21, 5.5))
for j in 1:nsh
    c = "C$(j-1)";  k = ks[j]
    axs[1].errorbar(τ, res[:MT].C[j, :], yerr=res[:MT].E[j, :], fmt=".", ms=2, color=c, alpha=0.6)
    axs[1].plot(τ, exp.(-ν * k^2 .* τ), "-", color=c, label=@sprintf("k h = %.2f", k * m.h))
    axs[2].errorbar(τ, res[:ML].C[j, :], yerr=res[:ML].E[j, :], fmt=".", ms=2, color=c, alpha=0.6)
    axs[2].plot(τ, cos.(cs * k .* τ) .* exp.(-Γs * k^2 .* τ ./ 2), "-", color=c, lw=1)
end
axs[1].set_xlabel("τ");  axs[1].set_ylabel("C_T(k, τ) / (N V w T)");  axs[1].legend(fontsize=8)
axs[1].set_title("transverse: points measured, lines e^{−νk²τ}")
axs[2].set_xlabel("τ");  axs[2].set_ylabel("C_L(k, τ) / (N V w T)")
axs[2].set_title("longitudinal: lines cos(c_s k τ) e^{−Γk²τ/2}")
axs[3].errorbar(ks .* m.h, νfit ./ ν, yerr=νerr ./ ν, fmt="o-", label="fitted transverse rate / ν k²")
axs[3].plot(ks .* m.h, res[:MT].C[:, 1], "s-", label="C_T(k, 0)")
axs[3].plot(ks .* m.h, res[:ML].C[:, 1], "^-", label="C_L(k, 0)")
axs[3].axhline(1, color="k", lw=0.5);  axs[3].set_xlabel("k h");  axs[3].legend(fontsize=8)
axs[3].set_title("statics and the effective shear viscosity")
suptitle(@sprintf("%s: N = %d, h/a = %.1f, box %.2f × %.2f, %d realizations, burn-in %.1f", TAG, m.N, m.ha,
                  m.L[1], m.L[2], length(runs), TBURN))
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "correlators_$(TAG).png")
savefig(out, dpi=110);  println("Saved ", out)
