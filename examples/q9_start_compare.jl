# q9_start_compare.jl — check (iii) of docs/focus_plan.tex Q9: the hmc start of focus against that of pre-prune (campaign q9_start,
# nersc/campaign_q9_start.txt), state by state (s_a = 16, 54; v0 = 0, 0.8), over the second half of each run (t > t_end/2):
#   KE/eq, ⟨H⟩/H0 − 1, the static spectra ⟨|M_L,k|²⟩, ⟨|M_T,k|²⟩, ⟨|S_k|²⟩ at the five smallest |k|, and the normalized transverse and
#   longitudinal correlators C(τ)/C(0) at the smallest |k| at τ = 0.5, 1, 2.
# Each quantity is a mean over the seeds; the error is the standard error over the seeds; the table gives old, new and
# (new − old)/σ.
#   julia --project examples/q9_start_compare.jl $SCRATCH/sph/q9_old $SCRATCH/sph/q9_new
using Printf, Statistics, LinearAlgebra, StaticArrays
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const OLD, NEW = ARGS[1], ARGS[2]
const STATES = ("q9_16_v0", "q9_16_v0.8", "q9_54_v0", "q9_54_v0.8")

# per run: a NamedTuple of scalars
function run_stats(path)
    d = read_equilibrium(path);  r = d.rec;  t = r.t;  nt = length(t)
    keep = findall(>(t[end] / 2), t);  dtr = t[2] - t[1]
    kn = norm.(r.ks);  ord = sortperm(kn)[1:5]
    spec(F) = [mean(abs2(F[i][k]) for i in keep) for k in ord]
    function corr(F, k, τ)
        lag = round(Int, τ / dtr);  idx = [i for i in keep if i + lag <= nt]
        c = mean(real(F[i + lag][k] * conj(F[i][k])) for i in idx);  c0 = mean(abs2(F[i][k]) for i in idx)
        return c / c0
    end
    k1 = ord[1]
    return (KE = mean(d.KE[keep]), H = mean(d.H[keep]),
            SL = spec(r.ML), ST = spec(r.MT), SS = spec(r.S),
            CT = [corr(r.MT, k1, τ) for τ in (0.5, 1.0, 2.0)], CL = [corr(r.ML, k1, τ) for τ in (0.5, 1.0, 2.0)],
            kh = kn[ord] .* d.meta.h)
end

function side(dir, state)
    files = filter(f -> endswith(f, ".h5"), readdir(joinpath(dir, state); join = true))
    return [run_stats(f) for f in files]
end
ms(x) = (mean(x), std(x) / sqrt(length(x)))
function line(name, a, b)
    (ma, ea), (mb, eb) = ms(a), ms(b);  σ = sqrt(ea^2 + eb^2)
    @printf("    %-22s old %12.5g ± %-9.2g new %12.5g ± %-9.2g  (new−old)/σ %+5.2f\n", name, ma, ea, mb, eb, (mb - ma) / σ)
    return abs(mb - ma) / σ
end

worst = 0.0;  nq = 0
for st in STATES
    A = side(OLD, st);  B = side(NEW, st)
    @printf("%s: %d old runs, %d new runs\n", st, length(A), length(B))
    (isempty(A) || isempty(B)) && continue
    zs = Float64[]
    push!(zs, line("KE/eq", [x.KE for x in A], [x.KE for x in B]))
    push!(zs, line("<H>/H0 - 1", [x.H for x in A], [x.H for x in B]))
    for (name, f) in (("|M_L,k|^2", :SL), ("|M_T,k|^2", :ST), ("|S_k|^2", :SS)), j in 1:5
        push!(zs, line(@sprintf("%s kh=%.3f", name, A[1].kh[j]), [getfield(x, f)[j] for x in A], [getfield(x, f)[j] for x in B]))
    end
    for (name, f) in (("C_T(τ)/C_T(0)", :CT), ("C_L(τ)/C_L(0)", :CL)), (j, τ) in enumerate((0.5, 1.0, 2.0))
        push!(zs, line(@sprintf("%s τ=%.1f", name, τ), [getfield(x, f)[j] for x in A], [getfield(x, f)[j] for x in B]))
    end
    global worst = max(worst, maximum(zs));  global nq += length(zs)
    @printf("  largest |new − old|/σ in this state: %.2f\n", maximum(zs))
end
@printf("%d comparisons; the largest |new − old|/σ: %.2f\n", nq, worst)
