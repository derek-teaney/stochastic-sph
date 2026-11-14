# hmc_analysis.jl — the equilibrium structure factor of the positions from many hybrid Monte Carlo chains
# (examples/hmc_positions.jl; docs/equilibrium_positions.tex, Fig. Skk).  For each set (a directory of chains):
#   S(k)/(4/s_a) in bands of kh (all box modes in each band), the mean over chains after a burn-in, with the error from the
#   spread between chains; the integrated autocorrelation time of |N_k|²/N in updates; g(r) and local/global ψ6.
#   julia --project examples/hmc_analysis.jl <dir> [<dir> ...]     → hmc_Sk_vs_k.png in SR_FIGDIR (default examples/)
# Environment: HMC_BURN (updates dropped at the start of each chain, default 20), HMC_DKH (band width in kh, default 0.1).
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "plotsetup.jl"))
using Printf, Statistics

const BURN = parse(Int, get(ENV, "HMC_BURN", "20"))
const DKH  = parse(Float64, get(ENV, "HMC_DKH", "0.1"))
chains(d) = isdir(d) ? [joinpath(d, f) for f in readdir(d) if endswith(f, ".h5") && startswith(f, "equilibrium_")] : String[]

function tau_int(x)
    n = length(x);  μ = mean(x);  v = var(x);  t = 0.5
    for l in 1:min(n ÷ 4, 200)
        c = sum((x[1:n-l] .- μ) .* (x[1+l:n] .- μ)) / ((n - l) * v);  c < 0 && break;  t += c
    end
    return t
end

function analyse(d)
    fs = chains(d);  isempty(fs) && return nothing
    rs = [read_equilibrium(f) for f in fs];  r1 = rs[1];  sa = r1.meta.s_a;  h = r1.meta.h;  N = r1.meta.N
    kh = [norm(k) * h for k in eachcol(r1.part.ks)]
    bands = sort(unique(round.(kh ./ DKH) .* DKH))
    per = Dict(b => Float64[] for b in bands);  taus = Dict(b => Float64[] for b in bands)
    for r in rs
        P = r.part;  nt = size(P.Nk, 2);  nt > BURN + 10 || continue
        for b in bands
            idx = findall(x -> abs(x - b) <= DKH / 2 + 1e-9, kh)
            x = vec(mean(abs2.(P.Nk[idx, BURN+1:end]) ./ N, dims = 1)) ./ (4 / sa)
            push!(per[b], mean(x));  push!(taus[b], tau_int(x))
        end
    end
    nc = length(per[bands[1]])
    S = [mean(per[b]) for b in bands];  E = [std(per[b]) / sqrt(max(nc, 1)) for b in bands]
    g = mean(r.part.g[:, BURN+1:end] for r in rs if size(r.part.g, 2) > BURN);  gm = vec(mean(g, dims = 2))
    ψl = mean(mean(r.part.psi6loc[BURN+1:end]) for r in rs);  ψg = mean(mean(r.part.psi6[BURN+1:end]) for r in rs)
    acc = mean(r.meta.acceptance for r in rs)
    return (name = basename(rstrip(d, '/')), sa = sa, L = r1.meta.L[1], tau = r1.meta.tau, nc = nc, acc = acc, kh = bands, S = S, E = E,
            taus = [mean(taus[b]) for b in bands], r = (r1.part.rbins[1:end-1] .+ r1.part.rbins[2:end]) ./ 2 ./ r1.meta.a, g = gm, ψl = ψl, ψg = ψg)
end

sets = filter(!isnothing, [analyse(d) for d in ARGS])
isempty(sets) && error("no chains in $(ARGS)")
fig, ax = subplots(1, 2, figsize = (13, 4.8))
for (j, A) in enumerate(sets)
    @printf("%s: s_a = %.4g, L = %.2g, τ = %.3g, %d chains, acceptance %.2f, ψ6 local %.3f global %.3f\n", A.name, A.sa, A.L, A.tau, A.nc, A.acc, A.ψl, A.ψg)
    for i in eachindex(A.kh)
        @printf("   kh %.2f  S/(4/s_a) = %6.2f ± %.2f   τ_int %.1f updates\n", A.kh[i], A.S[i], A.E[i], A.taus[i])
    end
    ax[1].errorbar(A.kh, A.S, yerr = A.E, fmt = "o-", ms = 4, capsize = 2, color = "C$(j-1)",
                   label = @sprintf("%s: s_a = %.3g, L = %.2g (%d chains)", A.name, A.sa, A.L, A.nc))
    ax[2].plot(A.r, A.g, color = "C$(j-1)", label = A.name)
end
ax[1].set_xlabel("kh");  ax[1].set_ylabel("S(k)/(4/s_a)");  ax[1].set_ylim(0, nothing);  ax[1].legend(fontsize = 7)
ax[2].axhline(1, color = "k", lw = 0.4);  ax[2].set_xlim(0, 4);  ax[2].set_xlabel("r / a");  ax[2].set_ylabel("g(r)");  ax[2].legend(fontsize = 7)
tight_layout();  savefig(figpath("hmc_Sk_vs_k.png"), dpi = 110);  close()
