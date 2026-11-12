# multistep_damping.jl — realized damping rate of a lattice-scale momentum mode (k along x,
# transverse polarization, kh ≈ 4 where the residual move does almost all the damping) on a
# frozen lattice, single vote vs bond vote, versus h/a at fixed S0 and Δt.  Prediction:
# Γ_eff ≈ acc_star Γ_res(k) + acc_resid Γ_resid(k), with Γ from Kmode on a code star.
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "stochrep_thermalize_common.jl"))          # Kmode
const h = 0.05;  const S0 = 1e6;  const DT = 8.66e-3;  const NST = 24;  const NSEED = 12
eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
println(" h/a  kh   Γ_res  Γ_resid (1/time) | vote    acc_s  acc_r  Γ_eff measured   predicted   Γ_eff/Γ")
for ha in (2.0, 3.0, 4.0)
    params, ps0, cl, _ = setup_ic(h, ha, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 6.2h, lattice = :hex)
    N = params.N;  Lx = params.L[1];  nk = round(Int, 4 * Lx / (2π * h));  k = SVector(2π * nk / Lx, 0.0);  ê = SVector(0.0, 1.0)
    x = [p.x for p in ps0];  c = [cos(dot(k, xi)) for xi in x]
    for vote in (:star, :bonds)
        Ps = zeros(NST + 1);  as = Float64[];  ar = Float64[];  Γres = 0.0;  Γdif = 0.0
        for seed in 1:NSEED
            ps = deepcopy(ps0)
            initial_state!(ps, params, cl, eos, :eq; rng = Xoshiro(100seed))
            fz = stochrep_frozen(ps, params, cl, eos);  st = fz.stars[1];  w = ps[1].e + ps[1].p
            M0 = 0.7 * sqrt(st.V * w * st.T)                   # 0.7 × thermal momentum per particle
            for i in 1:N; ps[i].m += M0 * c[i] * ê; end
            K, Kr = Kmode(st, k, ê, false);  Γ = 2K / (st.V * w * st.T);  Γres = 2Kr / (st.V * w * st.T);  Γdif = Γ - Γres
            proj() = sum(dot(ps[i].m, ê) * c[i] for i in 1:N) / sum(abs2, c) / M0
            rngs = make_phys_rng(N, 7seed);  rng = Xoshiro(8seed)
            Ps[1] += proj()
            for n in 1:NST
                acc = stochrep_step!(ps, params, cl, DT, eos; rng, rngs, frozen = fz, vote, flip = isodd(n))
                push!(as, acc.stress);  push!(ar, acc.resid);  Ps[n+1] += proj()
            end
        end
        Ps ./= NSEED;  t = DT .* (0:NST)
        sel = findall(>(0.15), Ps);  Γm = -(t[sel] \ log.(Ps[sel]))     # fit log P = −Γ t through 0
        pred = mean(as) * Γres + mean(ar) * Γdif
        @printf("%4.0f %4.2f  %6.2f  %6.2f          | %-6s  %.3f  %.3f    %6.2f        %6.2f      %.3f\n",
                ha, norm(k) * h, Γres, Γdif, vote, mean(as), mean(ar), Γm, pred, Γm / (Γres + Γdif))
    end
end
