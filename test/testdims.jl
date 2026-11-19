# testdims.jl — the physical dimension of a run and the constant-cs² equation of state (docs/dimension.tex):
#   (1) thermodynamic consistency of ConstantCs2EOS at cs² = 1/2 and 1/3: ds/de = 1/T, dp/dT = s, p ∝ T^(1+1/cs²);
#   (2) conformal_eos(d) has cs² = 1/d; the default physical dimension is the simulation dimension;
#   (3) the guard: the stochastic step and hybrid Monte Carlo refuse phys_dim ≠ sim_dim and sim_dim = 1;
#   (4) ζ_eff = ζ + 2η(1/2 − 1/d) is ζ at d = 2, floored at ε η (SPH_ZETA_FLOOR) in the stochastic step.
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))

println("testdims.jl — physical dimension and the constant-cs² EOS")

# (1) consistency, by centered differences in e
for c in (0.5, 1 / 3), Ts in (1.0, T_scale_2d_lq1)
    eos = ConstantCs2EOS(c, Ts, 0.0, 0.0)
    T(e) = Trel(eos, e);  s(e) = srel(eos, e);  p(e) = pressure(eos, e)
    for e in (0.01, 1.0, 37.0)
        δ = 1e-5 * e
        dsde = (s(e + δ) - s(e - δ)) / (2δ)
        dpdT = (p(e + δ) - p(e - δ)) / (T(e + δ) - T(e - δ))
        @assert abs(dsde * T(e) - 1) < 1e-8 "ds/de ≠ 1/T at cs² = $c, e = $e: $(dsde * T(e))"
        @assert abs(dpdT / s(e) - 1) < 1e-8 "dp/dT ≠ s at cs² = $c, e = $e: $(dpdT / s(e))"
        # power law p ∝ T^(1+1/cs²): d ln p / d ln T
        n = (log(p(e + δ)) - log(p(e - δ))) / (log(T(e + δ)) - log(T(e - δ)))
        @assert abs(n - (1 + 1 / c)) < 1e-6 "d ln p/d ln T = $n ≠ $(1 + 1/c) at cs² = $c"
    end
    # the fast paths (sqrt∘sqrt at 1/3, cbrt at 1/2) agree with the power
    for e in (0.3, 5.0)
        @assert abs(Trel(eos, e) / (Ts * e^(c / (1 + c))) - 1) < 1e-14
    end
end
println("  (1) ds/de = 1/T, dp/dT = s, p ∝ T^(1+1/cs²) at cs² = 1/2 and 1/3  ✓")

# (2) conformal cs² = 1/d; default phys_dim = sim_dim
@assert cs2(conformal_eos(2)) == 0.5 && cs2(conformal_eos(3)) == 1 / 3
@assert default_phys_dim(2) == 2 && default_phys_dim(1) == 1
quiet(f) = redirect_stdout(f, devnull)
p2 = quiet(() -> SimParams(1.0, 0.1, 2.0; lattice = :hex))
@assert p2.phys_dim == 2 && sim_dim(p2) == 2
println("  (2) conformal_eos(d): cs² = 1/d;  default phys_dim = sim_dim = 2  ✓")

# (3) the guard
eos = conformal_eos(2; η_by_s = 0.2)
mk(; kw...) = quiet(() -> setup_ic(0.1, 2.0, eos, (x, y) -> 1e3, (x, y) -> SVector(0.0, 0.0); L = 0.6, kw...))
refuses(f) = try f(); false catch err; err isa ErrorException && occursin("dimension.tex", err.msg) end
params, ps, cl, _ = mk()
@assert check_stochastic_dims(params) === nothing
params3, ps3, cl3, _ = mk(phys_dim = 3)
seed_phys!(length(ps3), 1)
dt = compute_dt(ps3, params3, eos; mode = :viscous)
@assert refuses(() -> stochastic_step!(ps3, params3, cl3, dt, eos; rng = Xoshiro(1)))
@assert refuses(() -> hmc_update!(ps3, params3, cl3, eos, HMCTarget(1.0, SVector(0.0, 0.0), params3.vol, eos), Xoshiro(1); dt = 1e-3))
params1, ps1, cl1, _ = quiet(() -> setup_ic(0.05, 4.0, eos, (x, y) -> 1e3, (x, y) -> SVector(0.0, 0.0); Lx = 1.0, sim_dim = 1))
@assert refuses(() -> check_stochastic_dims(params1))
println("  (3) the stochastic step and hybrid Monte Carlo refuse phys_dim ≠ sim_dim and sim_dim = 1  ✓")

# (4) ζ_eff at d = 2 and d = 3
ζeff(η, ζ, d) = ζ + 2η * (1 / 2 - 1 / d)
@assert ζeff(1.0, 0.0, 2) == 0.0 && abs(ζeff(1.0, 0.0, 3) - 1 / 3) < 1e-15
ε = ZETA_FLOOR;  @assert ε > 0
fl(η, ζ, d) = zeta_eff_by_s(ConstantCs2EOS(1 / d, 1.0, η, ζ), d)
@assert fl(0.2, 0.0, 2) == ε * 0.2                    # ζ = 0: the floor
@assert fl(0.2, 0.1, 2) == 0.1                        # ζ above the floor: unchanged
@assert abs(fl(0.2, 0.0, 3) - 0.2 / 3) < 1e-15       # d = 3: η/3 is above the floor
println("  (4) ζ_eff = max(ζ + 2η(1/2 − 1/d), ε η): ε η at d = 2, ζ = 0 (ε = $ε); η/3 at d = 3  ✓")
println("testdims.jl passed")
