# volume_n_pressure.jl — the particle pressures through the coupled step, kernel volume
# against number volume (docs/volume_n.tex), SPH_HAMILTONIAN=energy, L = 0.6, h/a = 2,
# S0 = 1e6, η/s = 0.2, equilibrium start.  Each step is: ideal (leapfrog Δt) then stochastic
# (Δt).  At every step, three snapshots of the rest-frame pressures p_a:
#   A  after the ideal step  (= before the stochastic step)
#   B  after the stochastic step (positions unchanged)
# and the next A.  Recorded: rms(p_a/p̄ − 1) at A and B; rms of the change of p_a in the
# stochastic step (A → B) and in the ideal step (B → next A); the same for the pressure
# smoothed over the kernel, p̃_a = Σ_b V_b p_b W_ab (what the long wavelengths see); and
# the correlation of Δp_a (stochastic) with Δs_a.
# Thermodynamic estimate for a particle at fixed volume: δp/p = δe/e = 4/√(3 s_a).
#
# Run (compute node):  julia -t 2 --project examples/volume_n_pressure.jl [tend=20] [nseed=2]
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
ENV["SPH_HAMILTONIAN"] = "energy"
const TEND = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 20.0
const NSEED = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 2
const S0 = 1e6
rel(v) = (m = sum(v) / length(v); sqrt(sum(abs2, v ./ m .- 1) / length(v)))

function state!(ps, params, cl, eos)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    p = [q.p for q in ps];  V = [q.s / q.S_smooth for q in ps]
    nl = neighbor_list!(cl, ps, params.h)
    pt = [sum(V[b] * p[b] * wendland_c2(norm(minimum_image(ps[a].x - ps[b].x, params.L)), params.h, params.kernel)
              for b in neighbors(nl, a)) for a in eachindex(ps)]
    (; p, pt, s = [q.s for q in ps])
end

function run(vol::Symbol, seed::Int)
    ENV["SPH_VOLUME"] = string(vol)
    eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    params, ps, cl, eos = setup_ic(0.05, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex)
    N = params.N;  rngs = make_phys_rng(N, 1000seed + 1);  rng = Xoshiro(1000seed + 2)
    initial_state!(ps, params, cl, eos, :eq; rng = Xoshiro(1000seed + 3))
    dt = 0.5 * compute_dt(ps, params, eos; mode = :viscous);  nst = ceil(Int, TEND / dt)
    dotm = zeros(SVector{2,Float64}, N);  cache = StochCache()
    out = Dict(k => Float64[] for k in (:t, :pA, :pB, :dps, :dpi, :ptA, :dpts, :dpti, :corr, :sspr))
    stochrep_step!(ps, params, cl, dt / 2, eos; rng, rngs, cache)
    B = state!(ps, params, cl, eos)
    for n in 1:nst
        leapfrog_step!(ps, params, cl, dt, eos, dotm)
        A = state!(ps, params, cl, eos)
        stochrep_step!(ps, params, cl, dt, eos; rng, rngs, flip = isodd(n), cache)
        Bn = state!(ps, params, cl, eos)
        pm = sum(A.p) / N
        dps = (Bn.p .- A.p) ./ pm;  dpi = (A.p .- B.p) ./ pm;  ds = Bn.s .- A.s
        push!(out[:t], n * dt);  push!(out[:pA], rel(A.p));  push!(out[:pB], rel(Bn.p))
        push!(out[:dps], sqrt(sum(abs2, dps) / N));  push!(out[:dpi], sqrt(sum(abs2, dpi) / N))
        push!(out[:ptA], rel(A.pt));  push!(out[:dpts], sqrt(sum(abs2, (Bn.pt .- A.pt) ./ pm) / N))
        push!(out[:dpti], sqrt(sum(abs2, (A.pt .- B.pt) ./ pm) / N))
        push!(out[:corr], dot(dps, ds) / sqrt(dot(dps, dps) * dot(ds, ds) + 1e-300));  push!(out[:sspr], rel(Bn.s))
        B = Bn
    end
    sa = ps[1].s;  merge(out, Dict(:est => [4 / sqrt(3 * S0 * (ps[1].s / ps[1].S_smooth))], :dt => [dt]))
end

avg(rs, k) = sum(r[k] for r in rs) / length(rs)
fig, axs = subplots(1, 3, figsize = (19, 5))
for (vol, c) in ((:kernel, "C3"), (:number, "C0"))
    rs = [run(vol, s) for s in 1:NSEED];  t = rs[1][:t]
    late = length(t) ÷ 2:length(t)
    m(k) = sum(avg(rs, k)[late]) / length(late)
    @printf("%-7s δp/p particle: %.4f (A) %.4f (B)   change per stochastic step %.2e, per ideal step %.2e | kernel-smoothed δp/p %.4f, changes %.2e / %.2e | corr(Δp, Δs) %.2f | s spread %.4f→%.4f | estimate 4/√(3s) %.4f  dt %.2e\n",
            vol, m(:pA), m(:pB), m(:dps), m(:dpi), m(:ptA), m(:dpts), m(:dpti), m(:corr), avg(rs, :sspr)[1], avg(rs, :sspr)[end], rs[1][:est][1], rs[1][:dt][1])
    axs[1].plot(t, avg(rs, :pA), color = c, lw = 1, label = "$vol: particle p_a")
    axs[1].plot(t, avg(rs, :ptA), color = c, lw = 1, ls = "--", label = "$vol: kernel-smoothed p̃_a")
    axs[2].semilogy(t, avg(rs, :dps), color = c, lw = 1, label = "$vol: stochastic step")
    axs[2].semilogy(t, avg(rs, :dpi), color = c, lw = 1, ls = ":", label = "$vol: ideal step")
    axs[3].plot(t, avg(rs, :sspr), color = c, lw = 1, label = "$vol")
    vol === :number && axs[1].axhline(rs[1][:est][1], color = "k", lw = 0.8, ls = ":", label = "4/√(3s): fixed-volume estimate")
end
axs[1].set_xlabel("t");  axs[1].set_ylabel("rms(p/p̄ − 1)");  axs[1].legend(fontsize = 8);  axs[1].set_title("spread of the pressures (after the ideal step)")
axs[2].set_xlabel("t");  axs[2].set_ylabel("rms Δp_a / p̄ per step");  axs[2].legend(fontsize = 8);  axs[2].set_title("change of the particle pressures in each half of the step")
axs[3].set_xlabel("t");  axs[3].set_ylabel("rms(s_a/s̄ − 1)");  axs[3].legend(fontsize = 8);  axs[3].set_title("spread of the weights")
tight_layout();  savefig(joinpath(@__DIR__, "..", "docs", "figs", "volume_n_pressure.png"), dpi = 100)
