# testmoment.jl — the viscous step with inverse-moment faces (docs/inverse_moment.tex, stage 1), the only viscous step:
#   1. on the lattice at rest the conditioning vectors are the faces over V_a (𝓜_a = 𝟙, f_ab = 1);
#   2. disordered stars (jittered lattice, Poisson positions): the closure Σ_b A_ab r_abᵀ = V_a 𝟙 and the zero virial of the
#      conditioned residual tractions to roundoff, positive weights, Σ_a F_a = 0, momentum conserved by a stochastic step;
#   3. one mode in a moving fluid (v0 = 0.8): K/K_target (test 2 of commonkernel.tex).
# (The comparisons with :common, its tests 1 and 4, went with the other dissipations: docs/focus_plan.tex, Q5.)
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
eos = ConstantCs2EOS(1/3, 1.0, 0.2, 0.0)

function setup(v0; L = 0.6, jitter = 0.0, ha = 2.0, poisson = false)
    let
        γ = 1 / sqrt(1 - v0^2)
        _, q, _, _ = quiet(() -> setup_ic(0.05, ha, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L, lattice = :hex))
        w0 = q[1].e + q[1].p
        params, ps, cl, _ = quiet(() -> setup_ic(0.05, ha, eos, (x, y) -> γ * 1e6, (x, y) -> SVector(w0 * γ^2 * v0, 0.0);
                                                   L, lattice = :hex, jitter, seed = 3))
        if poisson
            rng = Xoshiro(9)
            for p in ps;  p.x = SVector(rand(rng) * params.L[1], rand(rng) * params.L[2]);  end
            rebuild_cell_list!(cl, ps)
        end
        smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos);  update_velocities!(ps, params, cl)
        return params, ps, cl
    end
end
KF(params, ps, cl, β) = (F = zeros(SVector{2,Float64}, length(ps));
                         K = bond_dissipation(ps, params, cl, eos, β);  bond_dissipative_force!(F, ps, params, cl, eos, β);  (K, F))

println("1. at rest on the lattice: the conditioning vectors are the faces over V_a")
let
    Pm, psm, clm = setup(0.0)
    sm = build_stars(psm, Pm, clm, eos, [p.s / p.S_smooth for p in psm])
    dU = maximum(maximum(norm(sm[a].U[k] - sm[a].A[k] / sm[a].V) for k in eachindex(sm[a].U)) / maximum(norm, sm[a].U) for a in eachindex(sm))
    @printf("  stars: conditioning vectors − faces/V rel %.1e\n", dU)
    check(dU < 1e-12, "stars: conditioning = subtraction through the faces")
end

println("2. disordered stars: closure, zero virial, momentum")
for (name, kw) in (("jittered 0.3a, v0 = 0.8", (jitter = 0.3,)), ("Poisson, v0 = 0", (poisson = true,)))
    v0 = occursin("0.8", name) ? 0.8 : 0.0
    P, ps, cl = setup(v0; kw...)
    V = [p.s / p.S_smooth for p in ps];  st = build_stars(ps, P, cl, eos, V)
    clos = maximum(norm(sum(s.A[k] * s.r[k]' for k in eachindex(s.A)) / s.V - I) for s in st if !isempty(s.A))
    rng = Xoshiro(2);  δ = SVector{2,Float64}[];  vir = 0.0
    for s in st[1:50]
        resize!(δ, length(s.nb));  propose_resid!(δ, s, 1e-3, rng)
        vir = max(vir, norm(sum(δ[k] * s.r[k]' for k in eachindex(δ))) / maximum(norm, δ))
    end
    wmin = minimum(minimum(s.w) for s in st if !isempty(s.w))
    rngβ = Xoshiro(5);  β = [SVector(randn(rngβ), randn(rngβ)) * 1e-4 for _ in ps]
    K, F = KF(P, ps, cl, β)
    params = P;  m0 = sum(p.m for p in ps)
    rngs = make_phys_rng(P.N, 7);  dt = 0.5 * compute_dt(ps, P, eos; mode = :viscous)
    a = quiet(() -> stochastic_step!(ps, P, cl, dt, eos, StochOptions(P); rng = Xoshiro(8), rngs))
    dP = norm(sum(p.m for p in ps) - m0) / sum(norm(p.m) for p in ps)
    @printf("  %s: closure %.1e, conditioned virial %.1e, min w %.2e, K = %.3e, |ΣF|/max|F| %.1e, |ΔP| %.1e, acc %.3f %.3f\n",
            name, clos, vir, wmin, K, norm(sum(F)) / maximum(norm, F), dP, a.stress, a.resid)
    check(clos < 1e-12 && vir < 1e-12 && wmin > 0 && K > 0, "$name: closure and zero virial to roundoff, positive weights, K > 0")
    check(norm(sum(F)) / maximum(norm, F) < 1e-10 && dP < 1e-12, "$name: momentum conserved (force and stochastic step)")
end

println("3. one mode in a moving fluid, v0 = 0.8: K/K_target at kh = 0.26, 0.52, 1.05")
for (kind, dir) in ((:shear, :x), (:long, :x), (:shear, :y), (:long, :y))
    out = Dict{Symbol,Vector{Float64}}()
    for diss in (:moment,)
        rs = Float64[]
        for nk in (1, 2, 4)
            params, ps, cl = setup(0.8; L = 1.2)
            k = dir === :x ? SVector(2π * nk / params.L[1], 0.0) : SVector(0.0, 2π * nk / params.L[2])
            e = kind === :long ? k / norm(k) : SVector(-k[2], k[1]) / norm(k)
            A = 1e-6;  β = [A * e * sin(dot(k, p.x)) for p in ps]
            K = bond_dissipation(ps, params, cl, eos, β)
            p = ps[1];  ℓ, ℓi, Pp, Pℓ = comoving_structures(p.dotx, 1/3, 2)
            Ta = p.T * p.γ;  sr = srel(eos, p.e);  η = 0.2 * sr;  ζeff = 2η * (1 / 2 - 1 / params.phys_dim)
            G = A * e * k';  θ = sum(ℓ .* G) / Pℓ;  Y = (G + G') / 2 - θ * Pp
            push!(rs, K / (0.5 * params.L[1] * params.L[2] * Ta * (η * tr(ℓ * Y * ℓ * Y) + 0.5 * ζeff * θ^2)))
        end
        out[diss] = rs
    end
    @printf("  %-5s k ∥ %s   moment %s\n", kind, dir, join([@sprintf("%.4f", r) for r in out[:moment]], " "))
    check(abs(out[:moment][1] - 1) < 0.03, "$kind k∥$dir moment: K → target at small kh")
end

println(ok ? "testmoment: PASS" : "testmoment: FAIL");  ok || exit(1)
