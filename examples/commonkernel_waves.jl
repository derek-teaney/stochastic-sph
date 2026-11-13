# commonkernel_waves.jl — the viscous step alone (positions fixed, no ideal step), deterministic
# (docs/commonkernel.tex, test 3): one mode δm_a = ε e sin(k·x_a) on a uniform fluid at rest or moving with v0
# along x, evolved with viscous_step_energy! for dissipation = :particle (the present step), :common_lab and
# :common.  The entropy deficit D = Σ_a [S_a(e_a, m̄) − S_a(e_a, m_a)] obeys dD/dt = −2K, so the mode decays
# at Γ = K/D.  Reported: the fitted Γ of the mode amplitude against K/D of the same scheme at t = 0 (the step
# itself) and against K_target/D, K_target the continuum V T [η tr(ℓYℓY) + ½ ζ_eff θ²] (hydrodynamics).
#   julia -t 64 --project=nersc examples/commonkernel_waves.jl [out]   → <out>/commonkernel_waves.toml, table
using StaticArrays, LinearAlgebra, Printf, Random, TOML
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const OUT = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "commonkernel_waves");  mkpath(OUT)
const SMOKE = setting("SR_SMOKE", false)
quiet(f) = redirect_stdout(f, devnull)

# SPH_DISSIPATION is set in the environment by the caller (one scheme at a time: ENV is shared by the threads)
function run_case(diss, v0, kind, dir, nk)
    begin
        @assert ENV["SPH_DISSIPATION"] == string(diss)
        eos = ConformalEOS(1/3, 1.0, 0.2, 0.0);  γ = 1 / sqrt(1 - v0^2)
        _, q, _, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L = 1.2, lattice = :hex))
        w0 = q[1].e + q[1].p
        params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> γ * 1e6, (x, y) -> SVector(w0 * γ^2 * v0, 0.0);
                                                  L = 1.2, lattice = :hex))
        k = dir === :x ? SVector(2π * nk / params.L[1], 0.0) : SVector(0.0, 2π * nk / params.L[2])
        e = kind === :long ? k / norm(k) : SVector(-k[2], k[1]) / norm(k)
        mbar = ps[1].m;  ε = 1e-4 * w0 * params.vol
        for p in ps;  p.m = mbar + ε * e * sin(dot(k, p.x));  end
        smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos);  update_velocities!(ps, params, cl)
        ea = [particle_energy(p, eos) for p in ps];  Va = [p.s / p.S_smooth for p in ps]
        Sa(a, m) = particle_entropy(ea[a], m, Va[a], eos)
        D0 = sum(Sa(a, mbar) - Sa(a, ps[a].m) for a in eachindex(ps))
        K0 = bond_dissipation(ps, params, cl, eos)
        # continuum target from the β amplitude of the mode
        βs = [p.dotx / p.Teff for p in ps];  βb = sum(βs) / length(βs)
        B = 2 * sum(dot(βs[a] - βb, e) * sin(dot(k, ps[a].x)) for a in eachindex(ps)) / length(ps)
        p = ps[1];  ℓ, _, P, Pℓ = comoving_structures(p.dotx, 1/3, 2)
        Ta = p.T * p.γ;  sr = srel(eos, p.e);  η = 0.2 * sr;  ζeff = 2η * (1 / 2 - 1 / 3)
        G = B * e * k';  θ = sum(ℓ .* G) / Pℓ;  Y = (G + G') / 2 - θ * P
        Ktar = 0.5 * params.L[1] * params.L[2] * Ta * (η * tr(ℓ * Y * ℓ * Y) + 0.5 * ζeff * θ^2)
        Γs = K0 / D0;  Γt = Ktar / D0
        # evolve the viscous step alone
        dt = 0.5 * compute_dt(ps, params, eos; mode = :viscous)
        tend = SMOKE ? 20dt : min(0.7 / Γs, 15.0);  n = ceil(Int, tend / dt);  dt = tend / n
        amp() = sum(dot(pp.m - mbar, e) * sin(dot(k, pp.x)) for pp in ps)
        ts = [0.0];  cs = [amp()];  S0 = sum(pp.s for pp in ps);  H0 = sum(particle_energy(pp, eos) for pp in ps);  sumK = 0.0
        for i in 1:n
            Kstep, _ = viscous_step_energy!(ps, params, cl, dt, eos);  sumK += Kstep * dt
            if i % max(1, n ÷ 100) == 0 || i == n;  push!(ts, i * dt);  push!(cs, amp());  end
        end
        smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
        X = hcat(ones(length(ts)), ts);  Γm = -(X \ log.(abs.(cs)))[2]
        dS = sum(pp.s for pp in ps) - S0;  dH = sum(particle_energy(pp, eos) for pp in ps) / H0 - 1
        return (diss = string(diss), v0 = v0, kind = string(kind), dir = string(dir), kh = norm(k) * params.h,
                Gamma = Γm, Gamma_scheme = Γs, Gamma_target = Γt, K_over_Ktarget = K0 / Ktar,
                dS_over_2Kdt = dS / (2 * sumK), dH_over_H = dH, steps = n, t_end = tend)
    end
end

cases = [(diss, v0, kind, dir, nk) for diss in (:particle, :common_lab, :common), v0 in (0.0, 0.8),
         (kind, dir) in ((:shear, :x), (:long, :x), (:shear, :y), (:long, :y)), nk in (1, 2, 4)]
SMOKE && (cases = cases[[1, 30, 70]])
res = Vector{Any}(undef, length(cases))
for diss in unique(c[1] for c in cases)               # one scheme at a time; its cases in parallel
    ENV["SPH_DISSIPATION"] = string(diss)
    idx = [i for i in eachindex(cases) if cases[i][1] === diss]
    Threads.@threads :dynamic for i in idx
        res[i] = run_case(cases[i]...)
    end
end
println("the viscous step alone: Γ fitted / Γ of the scheme (K/D at t = 0) / Γ of hydrodynamics (K_target/D)")
@printf("  %-10s %4s %-5s %-2s %5s  %9s %9s  %8s %8s  %10s %9s\n", "scheme", "v0", "mode", "k", "kh", "Γ/Γ_sch", "Γ/Γ_hyd", "K/K_tar",
        "ΔS/2∫K", "ΔH/H", "steps")
for r in sort(res, by = r -> (r.v0, r.kind, r.dir, r.kh, r.diss))
    @printf("  %-10s %4.1f %-5s %-2s %5.2f  %9.4f %9.4f  %8.4f %8.4f  %+10.1e %9d\n", r.diss, r.v0, r.kind, r.dir, r.kh,
            r.Gamma / r.Gamma_scheme, r.Gamma / r.Gamma_target, r.K_over_Ktarget, r.dS_over_2Kdt, r.dH_over_H, r.steps)
end
open(joinpath(OUT, "commonkernel_waves.toml"), "w") do io
    TOML.print(io, Dict("case" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in res]))
end
println("written to $OUT")
