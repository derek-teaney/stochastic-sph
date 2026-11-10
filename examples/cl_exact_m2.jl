# cl_exact_m2.jl — the exact momentum variance of one particle at fixed volume, against the
# large-entropy value used to normalize KE/eq (docs/continuum_limit.tex).
#
# A particle of volume V in contact with the rest of the fluid at temperature T (at rest) has
# energy e and momentum m distributed as
#     P(e, m) ∝ exp(S(e, m; V) − e/T),     |m| < e,
# with S the particle entropy of the code (particle_entropy, conformal EOS).  For large s_a this
# is Gaussian in m with ⟨m²⟩ = 2 V w T (the KE/eq = 1 normalization); for a few units of entropy
# the bound |m| < e cuts it off.  Integrated on a grid in e and r = |m|/e:
#     d²m de = 2π e² r dr de.
# Run (login node, seconds):  julia --project examples/cl_exact_m2.jl
using StaticArrays, LinearAlgebra, Printf
for f in ("sph", "celllist", "setup", "rng", "stochrep")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
const S0 = 1e6;  const H = 0.05
eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
# uniform state at S0: temperature, enthalpy density (as the runs use them)
_, ps, _, _ = setup_ic(H, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex)
const T = ps[1].T;  const w = ps[1].e + ps[1].p;  const ebar_density = ps[1].e
@printf("uniform state: T = %.5g  w = %.5g  e = %.5g\n", T, w, ebar_density)
# The run's total energy is fixed at N ē, so the temperature of the distribution is the one
# whose mean energy is ē (as beta_from_averages does for the equilibrium start): solve for it.
@printf("%5s %8s %10s %8s %12s %12s %10s %10s\n", "h/a", "s_a", "V", "T*/T", "<m²>/2VwT", "<m²>/2VwT*", "P(|v|>0.9)", "P(|v|>0.99)")
for ha in (2, 3, 4, 6, 8, 12, 16, 23, 32, 46)
    a = H / ha;  V = sqrt(3) / 2 * a^2;  ebar = ebar_density * V
    es = range(1e-4, 8.0, length = 1600) .* ebar
    rs = range(0.0, 1.0, length = 2002)[1:end-1] .+ 0.5 / 2001
    S = [particle_entropy(e, SVector(e * r, 0.0), V, eos) + log(2π * e^2 * r) for e in es, r in rs]
    weights(Tx) = (l = [S[i, j] - es[i] / Tx for i in eachindex(es), j in eachindex(rs)];
                   lm = maximum(x for x in l if isfinite(x));  [isfinite(x) ? exp(x - lm) : 0.0 for x in l])
    emean(Tx) = (W = weights(Tx);  sum(W[i, j] * es[i] for i in eachindex(es), j in eachindex(rs)) / sum(W))
    lo, hi = 0.05T, 2T                                   # bisection: <e>(T*) = ē
    for _ in 1:50;  mid = sqrt(lo * hi);  emean(mid) > ebar ? (hi = mid) : (lo = mid);  end
    Ts = sqrt(lo * hi);  W = weights(Ts);  Z = sum(W)
    m2 = sum(W[i, j] * (es[i] * rs[j])^2 for i in eachindex(es), j in eachindex(rs)) / Z
    p09 = 0.0;  p099 = 0.0
    for i in eachindex(es), j in eachindex(rs)
        W[i, j] < 1e-300 && continue
        _, v, _ = eos_from_energy(es[i] / V, SVector(es[i] * rs[j] / V, 0.0), eos)
        nv = norm(v);  nv > 0.9 && (p09 += W[i, j]);  nv > 0.99 && (p099 += W[i, j])
    end
    @printf("%5d %8.1f %10.3e %8.4f %12.4f %12.4f %10.2e %10.2e\n", ha, S0 * V, V, Ts / T, m2 / (2 * V * w * T),
            m2 / (2 * V * w * Ts), p09 / Z, p099 / Z)
end
