# ideal_lattice_modes.jl — phonon spectrum of the ideal SPH Hamiltonian on the hexagonal
# lattice, from the energy alone (no dynamics, no noise).
# At fixed s_a and zero momenta, displace the lattice by one plane wave,
#     u_a = ε ê cos(q·x_a),   ê ∥ q (longitudinal) or ê ⊥ q (transverse),
# and take the second difference of U = H({x}):
#     ΔU = [U(ε) + U(−ε) − 2U(0)]/2 = ½ ε² (N/2) K(q),      ω²(q) = K(q)/(wV)
# (inertia wV per particle for small velocities).  A fluid has ω_L = c_s q, ω_T = 0;
# ω² < 0 means the lattice is unstable to that mode.
#
# Run:  [SPH_KERNEL=wc2|wc4|wc6] julia --project examples/ideal_lattice_modes.jl [h/a ...]
using StaticArrays, LinearAlgebra, Printf, Random
for f in ("sph", "celllist", "setup", "rng", "stochrep", "measure")
    include(joinpath(@__DIR__, "..", "src", "$f.jl"))
end
const HAS = length(ARGS) > 0 ? parse.(Float64, ARGS) : [2.0]
const LBOX = 1.2;  const S0 = 1e6;  const EPS = 0.01;  const QHMAX = parse(Float64, get(ENV, "QHMAX", "3.0"))
# amplitude: ε = EPS·a, or, if STRAIN is set, ε = STRAIN/|q| (fixed strain amplitude q ε; needed at
# large h/a, where the transverse energy change is a tiny fraction of H)
const STRAIN = parse(Float64, get(ENV, "STRAIN", "0"))

function energy!(ps, params, cl, eos, x0, ê, q, ε)
    for (a, p) in enumerate(ps)
        x = x0[a] + ε * ê * cos(dot(q, x0[a]))
        p.x = SVector(mod(x[1], params.L[1]), mod(x[2], params.L[2]))
    end
    rebuild_cell_list!(cl, ps)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    return sum(p.s * p.H_over_S for p in ps)
end

for ha in HAS
    eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    params, ps, cl, eos = setup_ic(0.05, ha, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex)
    x0 = [p.x for p in ps];  N = params.N;  a = params.a
    p1 = ps[1];  V = p1.s / p1.S_smooth;  w = p1.e + p1.p;  cs = sqrt(1/3)
    U0 = energy!(ps, params, cl, eos, x0, SVector(1.0, 0.0), SVector(0.0, 0.0), 0.0)
    qs = box_wavevectors(params, QHMAX)
    # one q per shell, in two directions
    byshell = Dict{Float64,SVector{2,Float64}}()
    for q in qs;  kn = round(norm(q); digits = 6);  haskey(byshell, kn) || (byshell[kn] = q);  end
    println(@sprintf("== %s, h/a = %.1f, N = %d, a = %.4f   (c_s = %.3f)", string(params.kernel), ha, N, a, cs))
    println("   q h     ω_L/(c_s q)   ω_T/q (= c_t)   ω_T²·sign")
    for kn in sort(collect(keys(byshell)))
        q = byshell[kn];  q̂ = q / norm(q);  t̂ = SVector(-q̂[2], q̂[1])
        ε = STRAIN > 0 ? STRAIN / kn : EPS * a
        K = map((q̂, t̂)) do ê
            Up = energy!(ps, params, cl, eos, x0, ê, q, ε)
            Um = energy!(ps, params, cl, eos, x0, ê, q, -ε)
            ΔU = (Up + Um - 2U0) / 2
            ΔU / (0.5 * ε^2 * N / 2)
        end
        ωL2, ωT2 = K ./ (w * V)
        @printf("   %.3f   %9.4f      %s        %+.3e\n", kn * params.h, sqrt(max(ωL2, 0)) / (cs * kn),
                ωT2 >= 0 ? @sprintf("%.5f", sqrt(ωT2) / kn) : "unstable", ωT2)
    end
end
