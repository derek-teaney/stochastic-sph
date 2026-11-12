# stochastic_smallk.jl — the long-wavelength end of the equilibrium spectrum, 1D frozen lattice
# (docs/stochastic.tex).  Does the stochastic step, with a cold or deterministic residual,
# give the right equilibrium at small k?  Prediction:
#     ⟨|m_k|²⟩/(N V w T) = T_eff(k)/T = 1 − (1 − 1/λ)(1 − R(k)),   1 − R(k) = O((kh)²).
# The run starts from a thermal draw of the momenta (variance V w T, zero total momentum),
# so the small-k modes need not warm up from zero; it is long enough that every mode shown
# has relaxed many times (Γ_k t_run ≥ 10), so a wrong stationary value would show.
#
# Run:  julia -t auto --project examples/stochastic_smallk.jl  → docs/figs/stochrep_smallk.png
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "stochastic_thermalize_common.jl"))     # Kmode

const LAMS = (1.0, Inf)
const NENS = 64

eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
params, ps0, cl, eos = setup_ic(0.05, 4.0, eos, (x, y) -> 4e4, (x, y) -> SVector(0.0, 0.0);
                                Lx=2.0, sim_dim=1)
N = params.N;  fz = stochastic_frozen(ps0, params, cl, eos)
st = fz.stars[1];  p = ps0[1];  w = p.e + p.p;  V = fz.V[1];  T = p.T;  a = params.a
eq = N * V * w * T
dt = 0.25 * 0.5 * compute_dt(ps0, params, eos; mode=:viscous)
nk = 24                                            # modes n = 1..nk  (k a/π up to nk·2a/L)
ks = 2π .* (1:nk) ./ params.L[1]
KK = [Kmode(st, SVector(k, 0.0), SVector(1.0, 0.0), true) for k in ks]
Γ  = [2K[1] / (V * w * T) for K in KK];  R = [K[2] / K[1] for K in KK]
nsteps = ceil(Int, 10 / (Γ[2] * dt));  burn = nsteps ÷ 5
@printf("N=%d  dt=%.2e  nsteps=%d  Γ_2 t_run=%.1f  Γ_1 t_run=%.1f  %d threads\n",
        N, dt, nsteps, Γ[2] * nsteps * dt, Γ[1] * nsteps * dt, Threads.nthreads())

res = Dict{Float64,Matrix{Float64}}()
for λ in LAMS
    t0 = time()
    A = zeros(NENS, nk)
    Threads.@threads for r in 1:NENS
        ps = deepcopy(ps0);  rngs = make_phys_rng(N, 7000r + 3);  rng = Xoshiro(r)
        g = randn(Xoshiro(10^6 + r), N);  g .-= sum(g) / N
        for (i, q) in enumerate(ps); q.m = SVector(sqrt(V * w * T) * g[i], 0.0); end
        acc = zeros(nk);  nav = 0
        for n in 1:nsteps
            stochastic_step!(ps, params, cl, dt, eos; λ, frozen=fz, rng, rngs, flip=isodd(n), vote=:bonds)
            if n > burn && n % 5 == 0
                for (j, k) in enumerate(ks)
                    acc[j] += abs2(sum(q.m[1] * cis(-k * q.x[1]) for q in ps))
                end
                nav += 1
            end
        end
        A[r, :] = acc ./ (nav * eq)
    end
    res[λ] = A
    @printf("  λ=%-4s %.0f s\n", string(λ), time() - t0)
end
serialize(joinpath(@__DIR__, "..", "docs", "figs", "stochastic_smallk.jls"),
          Dict(:ks => ks, :R => R, :Γ => Γ, :res => res, :a => a, :h => params.h, :dt => dt, :nsteps => nsteps))

x = ks .* a ./ π
fig, axs = subplots(1, 2, figsize=(14, 5.5))
col = Dict(1.0 => "C0", Inf => "C3")
lab(λ) = λ == 1 ? "λ = 1" : "λ = ∞ (deterministic residual)"
for λ in LAMS
    A = res[λ];  μ = vec(sum(A, dims=1)) ./ NENS;  σ = vec(sqrt.(sum((A .- μ') .^ 2, dims=1) ./ (NENS - 1))) ./ sqrt(NENS)
    Te = R .+ (1 .- R) ./ λ
    axs[1].errorbar(x, μ, yerr=σ, fmt="o", ms=4, color=col[λ], label=lab(λ))
    axs[1].plot(x, Te, "-", color=col[λ], lw=1)
    if isinf(λ)
        axs[2].errorbar(x, 1 .- μ, yerr=σ, fmt="o", ms=4, color=col[λ], label="1 − measured")
        axs[2].plot(x, 1 .- Te, "-", color=col[λ], lw=1, label="1 − T_eff/T  (prediction)")
        c = (1 - Te[3]) / x[3]^2
        axs[2].plot(x, c .* x .^ 2, "k:", lw=1, label="∝ k²")
    end
    for j in (1, 2, 4, 8)
        @printf("λ=%-4s k a/π=%.4f  measured %.4f ± %.4f   predicted %.4f   Γ t_run=%.0f\n",
                string(λ), x[j], μ[j], σ[j], Te[j], Γ[j] * nsteps * dt)
    end
end
axs[1].axhline(1, color="k", lw=0.5)
axs[1].set_xlabel("k a / π");  axs[1].set_ylabel("⟨|m_k|²⟩ / (N V w T)")
axs[1].set_title("small k: equilibrium spectrum (points, ±1σ) vs T_eff(k)/T (lines)")
axs[1].legend(fontsize=9)
axs[2].set_xscale("log");  axs[2].set_yscale("log")
axs[2].set_xlabel("k a / π");  axs[2].set_ylabel("1 − ⟨|m_k|²⟩ / (N V w T)")
axs[2].set_title("λ = ∞: deficit at small k")
axs[2].legend(fontsize=9)
suptitle(@sprintf("1D frozen lattice, N = %d, h/a = %.0f, thermal start, %d runs, Γ_k t ≥ %.0f for k a/π ≥ %.3f",
                  N, params.h / a, NENS, Γ[2] * nsteps * dt, x[2]))
tight_layout()
out = joinpath(@__DIR__, "..", "docs", "figs", "stochrep_smallk.png")
savefig(out, dpi=120);  println("Saved ", out)
