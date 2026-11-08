# testeos_consistency.jl — is solve_eos! thermodynamically consistent for a moving fluid?
#
# testeos.jl checks the Newton residual with the solver's own relation between M and u,
# so it cannot see an error in that relation.  (Such an error, u = M/(e+p) instead of
# M = (e+p)γu, went unnoticed: it vanishes at rest.)  Here everything is built from a
# known rest-frame state (e, v) with textbook relativistic hydro:
#     S = s γ,  T^{00} = (e+p)γ² − p,  T^{0i} = (e+p)γ² v.
# Checks:
#   1  round trip: feed (S, M) to solve_eos! for two candidate momentum densities,
#        density frame  M = T^{0i} = (e+p)γ² v = w γ u   and   M = (e+p) u = w u,
#      and see which one returns the input e and v, and ε = H_over_S·S = T^{00}.
#   2  conjugacy: with h(S, M) = S·H_over_S, compare ∂h/∂M|_S with v and ∂h/∂S|_M with
#      (e+p)/S, by finite differences on one branch (warm start).
#   3  Hamiltonian velocity: on a moving particle configuration, compare ∂H/∂m_a with
#      ẋ_a (update_velocities!) and ∂H/∂s_a with Teff_a, by finite differences.
# Fails (exit 1) if the density-frame round trip of 1 is off by more than 1e-10, or 2 or 3
# disagree beyond 1e-5 relative.
#
# Run:  julia --project test/testeos_consistency.jl
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

eos = ConformalEOS(1/3, 1.0, 0.0, 0.0)
function solved(S, M::SVector{2,Float64}, e_guess)
    pa = Particle(SVector(0.0, 0.0), 1.0)
    pa.S_smooth = S;  pa.M_smooth = M;  pa.e = e_guess
    solve_eos!(pa, eos)
    return pa
end
ok = true

# ── 1  round trip from a known state ──────────────────────────────────────────
println("1  Round trip from a known rest-frame state (e, v)")
@printf("   %-6s %-6s | %-34s | %-34s\n", "e", "|v|", "input M = (e+p)γ²v  (T^{0i})", "input M = (e+p)u")
for (e, vx, vy) in ((1.0, 0.1, 0.0), (1.0, 0.3, 0.2), (2.0, 0.6, 0.0), (0.5, 0.0, 0.8))
    v = SVector(vx, vy);  γ = 1 / sqrt(1 - dot(v, v));  w = e + pressure(eos, e)
    S = srel(eos, e) * γ;  ε = w * γ^2 - pressure(eos, e)
    out = String[]
    for (k, M) in enumerate((w * γ^2 * v, w * γ * v))
        pa = solved(S, M, e)
        vout = pa.v_over_S * S
        de, dv, dε = abs(pa.e - e) / e, norm(vout - v), abs(pa.H_over_S * S - ε) / ε
        push!(out, @sprintf("Δe/e=%.1e Δv=%.1e Δε/ε=%.1e", de, dv, dε))
        k == 1 && max(de, dv, dε) > 1e-10 && (global ok = false)
    end
    @printf("   %-6.2f %-6.2f | %-34s | %-34s\n", e, norm(v), out[1], out[2])
end

# ── 2  conjugacy of h(S, M) ───────────────────────────────────────────────────
println("\n2  Conjugacy:  ∂h/∂M|_S = v  and  ∂h/∂S|_M = (e+p)/S,   h = S·H_over_S")
S0 = 1.0
e_rest = solved(S0, SVector(0.0, 0.0), 1.0).e
for Mx in (0.0, 0.05, 0.1, 0.2, 0.3)
    M  = SVector(Mx, 0.0)
    pa = solved(S0, M, e_rest)
    e0 = pa.e;  d = 1e-6
    hof(S, M) = (q = solved(S, M, e0); q.H_over_S * S)
    dhdM = (hof(S0, M + SVector(d, 0.0)) - hof(S0, M - SVector(d, 0.0))) / (2d)
    dhdS = (hof(S0 + d, M) - hof(S0 - d, M)) / (2d)
    v = pa.v_over_S[1] * S0
    rM = abs(dhdM - v) / max(abs(v), 1e-12);  rS = abs(dhdS - pa.T) / pa.T
    @printf("   M=%.2f  u=%.3f:  ∂h/∂M=%.6f  v=%.6f  (rel %.1e)   ∂h/∂S=%.6f  T=%.6f  (rel %.1e)\n",
            Mx, v * pa.γ, dhdM, v, Mx == 0 ? 0.0 : rM, dhdS, pa.T, rS)
    Mx > 0 && (rM > 1e-5 || rS > 1e-5) && (global ok = false)
end

# ── 3  Hamiltonian velocity on particles ──────────────────────────────────────
println("\n3  Hamiltonian velocity:  ∂H/∂m_a = ẋ_a  and  ∂H/∂s_a = Teff_a,   H = Σ s_a H_over_S_a")
params, ps, cl, eos3 = setup_ic(0.1, 2.5, ConformalEOS(1/3, 1.0, 0.0, 0.0), (x, y) -> 1.0,
                                (x, y) -> SVector(0.4 * sin(2π * x), 0.2 * cos(2π * y)); L=1.0, lattice=:hex)
thermalize!(ps, params, cl, eos3)
Hof(ps) = (smooth_to_particles!(ps, params, cl); update_thermodynamics!(ps, eos3); sum(p.s * p.H_over_S for p in ps))
for a in (1, length(ps) ÷ 4, length(ps) ÷ 2)
    d = 1e-6
    q = deepcopy(ps); q[a].m += SVector(d, 0.0); Hp = Hof(q)
    q = deepcopy(ps); q[a].m -= SVector(d, 0.0); Hm = Hof(q)
    dHdm = (Hp - Hm) / (2d)
    q = deepcopy(ps); q[a].s += d; Hp = Hof(q)
    q = deepcopy(ps); q[a].s -= d; Hm = Hof(q)
    dHds = (Hp - Hm) / (2d)
    rm = abs(dHdm - ps[a].dotx[1]) / max(abs(ps[a].dotx[1]), 1e-3)
    rs = abs(dHds - ps[a].Teff) / ps[a].Teff
    @printf("   a=%4d |v|=%.3f:  ∂H/∂m_x=%.6f  ẋ_x=%.6f  (rel %.1e)   ∂H/∂s=%.6f  Teff=%.6f  (rel %.1e)\n",
            a, norm(ps[a].v_over_S * ps[a].S_smooth), dHdm, ps[a].dotx[1], rm, dHds, ps[a].Teff, rs)
    (rm > 1e-5 || rs > 1e-5) && (global ok = false)
end

println()
println(ok ? "testeos_consistency.jl: consistent." :
             "testeos_consistency.jl: INCONSISTENT — H_over_S, v and T are not conjugate for moving fluid.")
ok || exit(1)
