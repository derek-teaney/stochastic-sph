# sscheme_fold_hmc.jl — why hybrid Monte Carlo is rejected with SPH_RENORM=sscheme at s_a = 16 (docs/sscheme_plan.tex, item E); a diagnostic.
# For each δt factor: 40 updates of the equilibrium.jl start (L = 0.6, h/a = 2, s_rel,0 = 2.956e4, τ ≈ L/c_s); counts the trajectories that
# leave the domain (dH = NaN) and the Metropolis rejections, and the smallest 𝒟 = T/T_a over the particles before each update.
ENV["SPH_RENORM"] = get(ENV, "SPH_RENORM", "sscheme")
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
eos = ConstantCs2EOS(cs2_val = 0.5, η_by_s = 0.2)
S0 = 2.956e4
h = 0.6 / sqrt(672 / (2 / sqrt(3))) * 2      # a from N = 672 in L = 0.6 (hex), h = 2a
params, ps, cl, _ = setup_ic(h, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex, seed = 5)
@printf("N = %d, s_a(set-up) = %.1f\n", params.N, S0 * params.vol)
Trel0 = trel_of_srel(eos, S0)
r = equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1), bath = (1 / Trel0, SVector(0.0, 0.0)))
tg = HMCTarget(r.β0, r.β, params.vol, eos)
c = SSCHEME ? SSchemeConsts(eos) : nothing
function dstats()
    SSCHEME || return (NaN, NaN, NaN)
    Ds = [sscheme_from_label(q.s, q.m, q.V, c, q.St).D for q in ps]
    return (minimum(Ds), count(<(0.3), Ds) / length(Ds), minimum(q.St for q in ps))
end
for fac in parse.(Float64, split(get(ENV, "FACS", "0.35,0.1"), ","))
    dt = fac * compute_dt(ps, params, eos; mode = :ideal);  ns = round(Int, (0.6 / sqrt(0.5)) / dt)
    nexit = 0;  nrej = 0;  nacc = 0;  dHs = Float64[]
    for i in 1:40
        d = dstats()
        u = hmc_update!(ps, params, cl, eos, tg, Xoshiro(100i); nsteps = ns, dt = dt)
        if !isfinite(u.dH);  nexit += 1;  elseif u.accepted;  nacc += 1;  else;  nrej += 1;  end
        isfinite(u.dH) && push!(dHs, u.dH)
        i <= 10 && @printf("  fac %.2f update %2d: min 𝒟 %.3f, fraction 𝒟 < 0.3 %.3f, min S_t %.2f -> %s dH/H %.2e\n", fac, i, d..., u.accepted ? "accepted" : (isfinite(u.dH) ? "rejected" : "LEFT THE DOMAIN"), u.dH)
    end
    @printf("fac %.2f (δt = %.2e, %d steps): %d accepted, %d rejected by Metropolis, %d left the domain; median |dH/H| %.1e\n",
            fac, dt, ns, nacc, nrej, nexit, isempty(dHs) ? NaN : sort(abs.(dHs))[cld(length(dHs), 2)])
end
