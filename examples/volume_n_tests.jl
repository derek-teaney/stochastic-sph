# volume_n_tests.jl — tests of the number-density volume V_a = 1/n_a (SPH_VOLUME=number,
# docs/volume_n.tex) against the kernel volume V_a = s_a/S_a, both with SPH_HAMILTONIAN=energy.
#
#   test 1  Hamilton's equations: ṁ_a = −∂H/∂x_a and Teff_a = ∂H/∂s_a against central
#           differences of H = Σ_a e_a, on a jittered lattice with unequal s_a and moving
#           particles.
#   test 2  the stochastic step at fixed positions: ΔH per half step (L = 0.6, h/a = 2,
#           S0 = 1e6, η/s = 0.2, equilibrium start), for :kernel with the estimate
#           ΔH ≈ −½ c_s² Σ_a T_a s_a (ε_a² − ε̄_a²), ε_a = Δs_a/s_a, ε̄ its kernel average.
#   test 3  the ideal step alone after test 2 has made the s_a unequal: ΔH/H after t = 1 at
#           Δt and Δt/2 (a second-order integrator: ratio ≈ 4).
#
# Run (on a compute node):  julia -t 4 --project examples/volume_n_tests.jl [nstep=400] [nseed=4]
using StaticArrays, LinearAlgebra, Printf, Random
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
ENV["SPH_HAMILTONIAN"] = "energy"
const NSTEP = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 400
const NSEED = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4
const S0 = 1e6;  const ETAS = 0.2;  const HA = 2.0

function make(vol::Symbol; L = 0.6, jitter = 0.0)
    ENV["SPH_VOLUME"] = string(vol)
    eos = ConformalEOS(1/3, 1.0, ETAS, 0.0)
    setup_ic(0.05, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L, lattice = :hex, jitter)
end
Hof(ps, params, cl, eos) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos);
                            sum(p.s * p.H_over_S for p in ps))

# ── test 1 ───────────────────────────────────────────────────────────────────
println("== test 1: force and Teff against central differences of H")
for vol in (:kernel, :number)
    params, ps, cl, eos = make(vol; jitter = 0.15)
    rng = Xoshiro(7)
    for p in ps                                   # unequal weights, moving particles (|v| ~ 0.1)
        e = particle_energy(p, eos)
        p.s *= 1 + 0.2 * (rand(rng) - 0.5)
        p.m += 0.1 * e * SVector(randn(rng), randn(rng))
    end
    rebuild_cell_list!(cl, ps)
    H0 = Hof(ps, params, cl, eos);  update_velocities!(ps, params, cl)
    dotm = zeros(SVector{2,Float64}, length(ps));  hamiltonian_force!(dotm, ps, params, cl)
    Teff = [p.Teff for p in ps];  T = [p.T for p in ps]
    ex = 0.0;  et = 0.0;  fs = 0.0;  ts = 0.0
    for a in 1:37:length(ps)
        δ = 1e-6 * params.a;  g = zeros(2)
        for i in 1:2
            e = SVector(i == 1 ? δ : 0.0, i == 2 ? δ : 0.0)
            ps[a].x += e;  Hp = Hof(ps, params, cl, eos)
            ps[a].x -= 2e; Hm = Hof(ps, params, cl, eos);  ps[a].x += e
            g[i] = -(Hp - Hm) / 2δ
        end
        ex = max(ex, norm(g - dotm[a]));  fs = max(fs, norm(dotm[a]))
        ds = 1e-6 * ps[a].s
        ps[a].s += ds;  Hp = Hof(ps, params, cl, eos);  ps[a].s -= 2ds;  Hm = Hof(ps, params, cl, eos);  ps[a].s += ds
        et = max(et, abs((Hp - Hm) / 2ds - Teff[a]));  ts = max(ts, abs(Teff[a]))
    end
    @printf("%-7s max |ṁ − (−∂H/∂x)| / max|ṁ| = %.1e   max |Teff − ∂H/∂s| / T = %.1e   max |Teff − T|/T = %.1e\n",
            vol, ex / fs, et / ts, maximum(abs.(Teff .- T) ./ T))
end
flush(stdout)

# ── test 2 and 3 ─────────────────────────────────────────────────────────────
function onestep(vol::Symbol, seed::Int)
    params, ps, cl, eos = make(vol)
    N = params.N;  rngs = make_phys_rng(N, 1000seed + 1);  rng = Xoshiro(1000seed + 2)
    initial_state!(ps, params, cl, eos, :eq; rng = Xoshiro(1000seed + 3))
    dt = 0.5 * compute_dt(ps, params, eos; mode = :viscous)
    cs2v = cs2(eos);  cache = StochCache()
    H0 = Hof(ps, params, cl, eos)
    dH = Float64[];  est = Float64[]
    for n in 1:NSTEP
        Hb = Hof(ps, params, cl, eos)
        sb = [p.s for p in ps];  Ts = [p.T * p.s for p in ps]
        kst = [kernel_star(a, ps, params, cl) for a in 1:N]              # weights s_b W_ab / S_a
        stochrep_step!(ps, params, cl, dt / 2, eos; rng, rngs, flip = isodd(n), cache)
        Ha = Hof(ps, params, cl, eos)
        ε = [(ps[a].s - sb[a]) / sb[a] for a in 1:N]
        εbar = [sum(w * ε[b] for (b, _, w) in kst[a]) for a in 1:N]
        push!(dH, (Ha - Hb) / H0)
        push!(est, -0.5 * cs2v * sum(Ts[a] * (ε[a]^2 - εbar[a]^2) for a in 1:N) / H0)
    end
    s = [p.s for p in ps];  spread = sqrt(sum(abs2, s .- sum(s) / N) / N) / (sum(s) / N)
    # test 3: ideal step alone from this state, to t = 1, at Δt and Δt/2
    ideal = map((1.0, 0.5)) do f
        qs = deepcopy(ps);  cq = CellList2D(params.L, params.h, params.N);  rebuild_cell_list!(cq, qs)
        dotm = zeros(SVector{2,Float64}, N);  H1 = Hof(qs, params, cq, eos);  n = round(Int, 1.0 / (f * dt))
        for _ in 1:n;  leapfrog_step!(qs, params, cq, f * dt, eos, dotm);  end
        (Hof(qs, params, cq, eos) - H1) / H1
    end
    return (; dH, est, dt, N, spread, ideal)
end

println("== test 2: the stochastic step at fixed positions, $NSTEP half steps;  test 3: ideal step alone")
for vol in (:kernel, :number), seed in 1:NSEED
    r = onestep(vol, seed)
    c = sum(r.dH .* r.est) / sqrt(sum(abs2, r.dH) * sum(abs2, r.est) + 1e-300)
    @printf("%-7s seed %d  N=%d dt/2=%.3g | ΔH/H per half step: mean %+.2e  rms %.2e | estimate: mean %+.2e  corr %.2f | s spread %.2e | ideal ΔH/H(t=1): Δt %+.2e  Δt/2 %+.2e\n",
            vol, seed, r.N, r.dt / 2, sum(r.dH) / NSTEP, sqrt(sum(abs2, r.dH) / NSTEP), sum(r.est) / NSTEP, c,
            r.spread, r.ideal...)
    flush(stdout)
end
