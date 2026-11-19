# moment_det.jl — deterministic tests of the viscous step with inverse-moment faces (SPH_DISSIPATION=moment) against the common
# scheme (docs/inverse_moment.tex, "Stage 1"), side by side.  Three sets, each for both schemes:
#   A. the viscous step alone (positions fixed, no ideal step; as examples/commonkernel_waves.jl) on the lattice moving with
#      v0 = 0.8: shear and longitudinal modes along and across the flow, kh = 0.26, 0.52, 1.05;
#   B. the viscous step alone on disordered positions at rest (jittered lattice, 0.3a, and random positions, jitter 3a, where the
#      correction Λ_a of common and the inverse moment differ): shear and longitudinal modes along x;
#   C. the full deterministic step at rest (test/physics/lib/waves.jl): the shear wave and the damped sound wave, η/s = 0.02,
#      kh = 1.4, 0.7, 0.35, h/a = 3.
# A and B: the fitted decay rate Γ of the mode against Γ_hyd = K_target/D (the continuum dissipation of the mode over its entropy
# deficit) and against the scheme's own K/D; ΔS/(2∫K dt) and ΔH/H.  C: Γ/Γ_hyd and ω/ω_hyd.
#   julia -t 16 --project examples/moment_det.jl            → <out>/moment_det.toml and docs/figs/moment_det.png (SR_FIGDIR)
# Environment: MD_OUT (default data/moment_det), MD_SETS (default "A,B,C"), SR_SMOKE=1 (a few short cases).
using StaticArrays, LinearAlgebra, Printf, Random, TOML
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "..", "test", "physics", "lib", "waves.jl"))
const OUT  = get(ENV, "MD_OUT", joinpath(@__DIR__, "..", "data", "moment_det"));  mkpath(OUT)
const SETS = split(get(ENV, "MD_SETS", "A,B,C"), ",")
const SMOKE = get(ENV, "SR_SMOKE", "0") == "1"
const SCHEMES = (:common, :moment)
quiet(f) = redirect_stdout(f, devnull)

# One mode on a fixed configuration, the viscous step alone (cf. commonkernel_waves.jl).  jitter: setup_ic jitter (seed fixed,
# so both schemes see the same positions); the background momentum is uniform (v0 along x) and zero at rest.
function alone_case(diss, v0, jitter, kind, dir, nk)
    @assert ENV["SPH_DISSIPATION"] == string(diss)
    eos = conformal_eos(default_phys_dim(); η_by_s = 0.2);  γ = 1 / sqrt(1 - v0^2)
    _, q, _, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L = 1.2, lattice = :hex))
    w0 = q[1].e + q[1].p
    params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> γ * 1e6, (x, y) -> SVector(w0 * γ^2 * v0, 0.0);
                                              L = 1.2, lattice = :hex, jitter, seed = 17))
    k = dir === :x ? SVector(2π * nk / params.L[1], 0.0) : SVector(0.0, 2π * nk / params.L[2])
    e = kind === :long ? k / norm(k) : SVector(-k[2], k[1]) / norm(k)
    m0 = [p.m for p in ps];  ε = 1e-4 * w0 * params.vol
    for (a, p) in enumerate(ps);  p.m = m0[a] + ε * e * sin(dot(k, p.x));  end
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos);  update_velocities!(ps, params, cl)
    ea = [particle_energy(p, eos) for p in ps];  Va = [p.s / p.S_smooth for p in ps]
    D0 = sum(particle_entropy(ea[a], m0[a], Va[a], eos) - particle_entropy(ea[a], ps[a].m, Va[a], eos) for a in eachindex(ps))
    K0 = bond_dissipation(ps, params, cl, eos)
    βs = [p.dotx / p.Teff for p in ps];  βb = sum(βs) / length(βs)
    B = 2 * sum(dot(βs[a] - βb, e) * sin(dot(k, ps[a].x)) for a in eachindex(ps)) / length(ps)
    p = ps[1];  ℓ, _, P, Pℓ = comoving_structures(sum(q.dotx for q in ps) / length(ps), 1/3, 2)
    Ta = sum(q.T * q.γ for q in ps) / length(ps);  sr = srel(eos, sum(q.e for q in ps) / length(ps));  η = 0.2 * sr;  ζeff = 2η * (1 / 2 - 1 / 3)
    G = B * e * k';  θ = sum(ℓ .* G) / Pℓ;  Y = (G + G') / 2 - θ * P
    Ktar = 0.5 * params.L[1] * params.L[2] * Ta * (η * tr(ℓ * Y * ℓ * Y) + 0.5 * ζeff * θ^2)
    # D0 < 0 for a momentum perturbation along a strong flow (v0 = 0.8: long k∥x, shear k∥y): the entropy at fixed lab energy is not
    # concave in m along v, so K/D is not a rate there; |K/D| sets the run length and the fitted Γ is compared between the schemes.
    Γs = abs(K0 / D0);  Γt = abs(Ktar / D0)
    dt = 0.5 * compute_dt(ps, params, eos; mode = :viscous)
    tend = SMOKE ? 20dt : min(0.7 / Γs, 15.0);  n = ceil(Int, tend / dt);  dt = tend / n
    amp() = sum(dot(ps[a].m - m0[a], e) * sin(dot(k, ps[a].x)) for a in eachindex(ps))
    ts = [0.0];  cs = [amp()];  S0 = sum(q.s for q in ps);  H0 = sum(particle_energy(q, eos) for q in ps);  sumK = 0.0
    for i in 1:n
        Kstep, _ = viscous_step_energy!(ps, params, cl, dt, eos);  sumK += Kstep * dt
        if i % max(1, n ÷ 100) == 0 || i == n;  push!(ts, i * dt);  push!(cs, amp());  end
    end
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    X = hcat(ones(length(ts)), ts);  Γm = -(X \ log.(abs.(cs)))[2]
    dS = sum(q.s for q in ps) - S0;  dH = sum(particle_energy(q, eos) for q in ps) / H0 - 1
    return Dict("set" => v0 == 0 ? "B" : "A", "scheme" => string(diss), "v0" => v0, "jitter" => jitter, "kind" => string(kind),
                "dir" => string(dir), "kh" => norm(k) * params.h, "Gamma_over_scheme" => Γm / Γs, "Gamma_over_hydro" => Γm / Γt,
                "K_over_Ktarget" => K0 / Ktar, "dS_over_2Kdt" => dS / (2 * sumK), "dH_over_H" => dH, "steps" => n, "D0_negative" => D0 < 0, "Gamma" => Γm)
end

res = Dict{String,Any}[]
alone = Tuple[]
"A" in SETS && append!(alone, [(0.8, 0.0, kind, dir, nk) for (kind, dir) in ((:shear, :x), (:long, :x), (:shear, :y), (:long, :y)) for nk in (1, 2, 4)])
"B" in SETS && append!(alone, [(0.0, j, kind, :x, nk) for j in (0.3, 3.0) for kind in (:shear, :long) for nk in (1, 2, 4)])
SMOKE && (alone = alone[1:min(3, end)])
for diss in SCHEMES
    isempty(alone) && break
    ENV["SPH_DISSIPATION"] = string(diss)                  # one scheme at a time (ENV is shared by the threads)
    out = Vector{Any}(undef, length(alone));  t0 = time()
    Threads.@threads :dynamic for i in eachindex(alone)
        out[i] = alone_case(diss, alone[i]...)
    end
    append!(res, out);  @printf("viscous step alone, %s: %d cases, %.0f s\n", diss, length(alone), time() - t0)
end
if "C" in SETS
    khs = SMOKE ? (1.4,) : (1.4, 0.7, 0.35)
    for diss in SCHEMES
        ENV["SPH_DISSIPATION"] = string(diss);  t0 = time()
        jobs = [(kind, kh) for kind in (:shear, :sound) for kh in khs]
        out = Vector{Any}(undef, length(jobs))
        Threads.@threads :dynamic for i in eachindex(jobs)
            kind, kh = jobs[i]
            r = wave_run(kind; kh, etas = 0.02, periods = SMOKE ? 0.2 : 2.0)
            out[i] = Dict("set" => "C", "scheme" => string(diss), "kind" => string(kind), "kh" => r.kh,
                          "Gamma_over_hydro" => r.Gamma / r.Gamma_hydro,
                          "omega_over_hydro" => r.omega_hydro == 0 ? 0.0 : r.omega / r.omega_hydro, "steps" => r.steps)
        end
        append!(res, out);  @printf("full deterministic step, %s: %d cases, %.0f s\n", diss, length(jobs), time() - t0)
    end
end
open(joinpath(OUT, "moment_det.toml"), "w") do io;  TOML.print(io, Dict("case" => res));  end

# ── table ──
key(r) = (r["set"], get(r, "jitter", 0.0), r["kind"], get(r, "dir", ""), round(r["kh"], digits = 2))
for k in sort(unique(key.(res)))
    c = [r for r in res if key(r) == k && r["scheme"] == "common"];  m = [r for r in res if key(r) == k && r["scheme"] == "moment"]
    (isempty(c) || isempty(m)) && continue
    c = c[1];  m = m[1]
    if k[1] == "C"
        @printf("C  %-6s kh %.2f   Γ/Γ_hyd common %.4f moment %.4f   ω/ω_hyd common %.4f moment %.4f\n", k[3], k[5],
                c["Gamma_over_hydro"], m["Gamma_over_hydro"], c["omega_over_hydro"], m["omega_over_hydro"])
    else
        @printf("%s  jitter %.1f %-5s k∥%s kh %.2f   Γ/Γ_hyd common %.4f moment %.4f   Γ/Γ_scheme %.4f %.4f   ΔS/2∫K %.4f %.4f   ΔH/H %+.0e %+.0e\n",
                k[1], k[2], k[3], k[4], k[5], c["Gamma_over_hydro"], m["Gamma_over_hydro"], c["Gamma_over_scheme"], m["Gamma_over_scheme"],
                c["dS_over_2Kdt"], m["dS_over_2Kdt"], c["dH_over_H"], m["dH_over_H"])
    end
end

# ── figure: one row per set, common | moment side by side, shared y in a row ──
include(joinpath(@__DIR__, "plotsetup.jl"))
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # categorical, fixed order (dataviz reference palette)
rows = [s for s in ("A", "B", "C") if any(r["set"] == s for r in res)]
fig, axs = subplots(length(rows), 2, figsize = (11, 3.4 * length(rows)), sharey = "row", squeeze = false)
titles = Dict("A" => "viscous step alone, lattice, v₀ = 0.8", "B" => "viscous step alone, disordered positions, at rest",
              "C" => "full deterministic step, lattice, at rest (η/s = 0.02)")
for (i, s) in enumerate(rows), (j, diss) in enumerate(("common", "moment"))
    ax = axs[i, j];  rr = [r for r in res if r["set"] == s && r["scheme"] == diss]
    curves = s == "A" ? [("shear", "x", "shear, k ∥ v"), ("long", "x", "long., k ∥ v"), ("shear", "y", "shear, k ⊥ v"), ("long", "y", "long., k ⊥ v")] :
             s == "B" ? [("shear", 0.3, "shear, jitter 0.3a"), ("long", 0.3, "long., jitter 0.3a"), ("shear", 3.0, "shear, random"), ("long", 3.0, "long., random")] :
                        [("shear", nothing, "shear wave"), ("sound", nothing, "sound wave")]
    for (ci, (kind, sel, lab)) in enumerate(curves)
        pts = [r for r in rr if r["kind"] == kind && (s == "A" ? r["dir"] == sel : s == "B" ? r["jitter"] == sel : true)]
        sort!(pts, by = r -> r["kh"])
        ax.plot([r["kh"] for r in pts], [r["Gamma_over_hydro"] for r in pts], "o-", color = COL[ci], lw = 2, ms = 6, label = lab)
    end
    ax.axhline(1, color = "#52514e", lw = 0.6)
    ax.set_title("$diss: $(titles[s])", fontsize = 9);  ax.set_xlabel("kh");  j == 1 && ax.set_ylabel("Γ / Γ_hydro")
    ax.grid(true, color = "#e6e5e0", lw = 0.5);  ax.legend(fontsize = 7, frameon = false)
end
tight_layout()
fig_out = figpath("moment_det.png");  savefig(fig_out, dpi = 120);  close()
println("written ", joinpath(OUT, "moment_det.toml"), " and ", fig_out)
