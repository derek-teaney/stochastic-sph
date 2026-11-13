# testcommonkernel.jl — the one-kernel viscous step (dissipation = :common, docs/commonkernel.tex):
#   1. at rest, C2: K and the force equal those of the present step (:particle), on a perfect and a jittered lattice;
#   2. a moving fluid: K of a long-wavelength shear and longitudinal mode against the continuum target
#      V T [η tr(ℓYℓY) + ½ ζ_eff θ²] of its gradient, both schemes, as kh → 0;
#   3. Σ_a F_a = 0 (momentum) for the new form.
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)

function setup(diss, v0; L = 0.6, jitter = 0.0, ha = 2.0)
    withenv("SPH_DISSIPATION" => string(diss)) do
        eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
        γ = 1 / sqrt(1 - v0^2)
        _, q, _, _ = quiet(() -> setup_ic(0.05, ha, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L, lattice = :hex))
        w0 = q[1].e + q[1].p
        params, ps, cl, _ = quiet(() -> setup_ic(0.05, ha, eos, (x, y) -> γ * 1e6, (x, y) -> SVector(w0 * γ^2 * v0, 0.0);
                                                   L, lattice = :hex, jitter, seed = 3))
        smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos);  update_velocities!(ps, params, cl)
        return params, ps, cl, eos
    end
end

println("1. at rest, C2: :common against :particle")
for jit in (0.0, 0.3)
    P1, ps1, cl1, eos = setup(:particle, 0.0; jitter = jit);  P2, ps2, cl2, _ = setup(:common, 0.0; jitter = jit)
    rng = Xoshiro(5);  β = [SVector(randn(rng), randn(rng)) * 1e-4 for _ in ps1]
    F1 = zeros(SVector{2,Float64}, length(ps1));  F2 = similar(F1)
    K1 = bond_dissipation(ps1, P1, cl1, eos, β);  K2 = bond_dissipation(ps2, P2, cl2, eos, β)
    bond_dissipative_force!(F1, ps1, P1, cl1, eos, β);  bond_dissipative_force!(F2, ps2, P2, cl2, eos, β)
    dF = maximum(norm(F1[a] - F2[a]) for a in eachindex(F1)) / maximum(norm, F1)
    @printf("  jitter %.1f: K %.6e vs %.6e (rel %.1e), force rel %.1e, Σ F (common) %.1e\n", jit, K1, K2, abs(K2 / K1 - 1), dF,
            norm(sum(F2)) / maximum(norm, F2))
    check(abs(K2 / K1 - 1) < 1e-10 && dF < 1e-10, "jitter $jit: identical to the present step at rest")
    check(norm(sum(F2)) / maximum(norm, F2) < 1e-10, "jitter $jit: momentum conserved")
end

println("2. moving fluid, v0 = 0.8: K of one mode / continuum target")
for (kind, dir) in ((:shear, :x), (:long, :x), (:shear, :y), (:long, :y)), diss in (:particle, :common_lab, :common)
    rs = Float64[]
    for (L, nk) in ((1.2, 1), (1.2, 2), (1.2, 4))
        params, ps, cl, eos = setup(diss, 0.8; L)
        k = dir === :x ? SVector(2π * nk / params.L[1], 0.0) : SVector(0.0, 2π * nk / params.L[2])
        e = kind === :long ? k / norm(k) : SVector(-k[2], k[1]) / norm(k)
        A = 1e-6
        β = [A * e * sin(dot(k, p.x)) for p in ps]
        K = bond_dissipation(ps, params, cl, eos, β)
        # continuum: G = A e kᵀ cos(k·x), ⟨cos²⟩ = ½; target per volume, summed over the box
        p = ps[1];  va = p.dotx;  ℓ, ℓi, P, Pℓ = comoving_structures(va, 1/3, 2)
        Ta = p.T * p.γ;  sr = srel(eos, p.e);  η = 0.2 * sr;  ζeff = 2η * (1 / 2 - 1 / 3)
        G = A * e * k';  θ = sum(ℓ .* G) / Pℓ;  Y = (G + G') / 2 - θ * P
        Ktarget = 0.5 * params.L[1] * params.L[2] * Ta * (η * tr(ℓ * Y * ℓ * Y) + 0.5 * ζeff * θ^2)
        push!(rs, K / Ktarget)
    end
    @printf("  %-5s k ∥ %s  %-8s  K/K_target at kh = 0.26, 0.52, 1.05:  %s\n", kind, dir, diss, join([@sprintf("%.4f", r) for r in rs], "  "))
    diss === :common_lab || check(abs(rs[1] - 1) < 0.03, "$kind k∥$dir $diss: K → target at small kh ($(round(rs[1], digits = 4)))")
end
println(ok ? "testcommonkernel: PASS" : "testcommonkernel: FAIL");  ok || exit(1)
