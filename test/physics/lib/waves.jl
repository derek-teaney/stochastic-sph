# waves.jl — the plane-wave harness of the physics tests (test/physics/*.jl).
#
# A conformal fluid at rest (entropy density s0, hexagonal lattice, periodic box Lx = L, Ly
# from the lattice) with one plane wave of wave vector k = 2π (nx/Lx, ny/Ly) (periodic in both
# directions).  The full deterministic step (generic_step!: viscous half steps around the
# leapfrog) evolves it, and the Fourier mode of the particle momenta along e,
#     c(t) = Σ_a (m_a·e) e^{−i k·x_a},   e = k̂ (sound) or k̂⊥ (shear),
# is recorded every step and fitted to c(t) = c₀ e^{−iωt − Γt}: ω from the slope of its phase,
# Γ from the slope of log|c|.  These are compared with linearized hydrodynamics
# (d = 3 transport coefficients, as in the library):
#     sound   ω = √(c_s²k² − (Γ_s k²/2)²),   Γ = Γ_s k²/2,   Γ_s = (4/3 η/s + ζ/s)/T_rel
#     shear   ω = 0,                          Γ = (η/s) k²/T_rel
# The sound wave starts as the exact damped eigenmode (testsound_viscous.jl), so c(t) is one
# exponential; h is set from the requested kh, with h/a fixed.
using Printf, LinearAlgebra, StaticArrays

function wave_run(kind::Symbol; kh, ha = 3.0, etas = 0.0, L = 1.0, nx = 1, ny = 1, s0 = 2.0,
                  A = 0.01, cs2 = 1/3, periods = 2.0, CFL = 0.2)
    eos = ConformalEOS(cs2, 1.0, etas, 0.0)
    quiet(f) = redirect_stdout(f, devnull)
    # box and background at a provisional h (Ly/L depends only on h/a)
    p0, bg, _, _ = quiet(() -> setup_ic(0.05, ha, eos, (x, y) -> s0, (x, y) -> SVector(0.0, 0.0); L))
    Trel = bg[1].T * bg[1].γ;  cs = sqrt(cs2)
    Ly = L * p0.L[2] / p0.L[1]
    kv = SVector(2π * nx / L, 2π * ny / Ly);  k = norm(kv);  h = kh / k
    kh_ = kv / k;  kp = SVector(-kh_[2], kh_[1])
    if kind === :sound
        Γs = (4/3 * etas) / Trel;  Γ = Γs * k^2 / 2;  ω = sqrt(max(cs2 * k^2 - Γ^2, 0.0))
        S = (x, y) -> s0 + (A / Trel) * cos(dot(kv, SVector(x, y)))
        M = (x, y) -> A * ((ω / k) * cos(dot(kv, SVector(x, y))) + (Γ / k) * sin(dot(kv, SVector(x, y)))) * kh_
        e = kh_;  tend = periods * 2π / ω
    else
        Γ = etas * k^2 / Trel;  ω = 0.0
        S = (x, y) -> s0
        M = (x, y) -> A * cos(dot(kv, SVector(x, y))) * kp
        e = kp;  tend = 1 / Γ                        # one e-folding
    end
    params, ps, cl, eos = quiet(() -> setup_ic(h, ha, eos, S, M; L))
    dt = compute_dt(ps, params, eos; mode = :viscous, CFL);  n = ceil(Int, tend / dt);  dt = tend / n
    mode() = sum(dot(p.m, e) * cis(-dot(kv, p.x)) for p in ps)
    t = zeros(n + 1);  c = zeros(ComplexF64, n + 1);  c[1] = mode()
    dotm = zeros(SVector{2,Float64}, params.N)
    for i in 1:n
        generic_step!(ps, params, cl, dt, eos, dotm)
        t[i+1] = i * dt;  c[i+1] = mode()
    end
    # least-squares slopes of log|c| and of the unwrapped phase
    φ = angle.(c);  for i in 2:length(φ);  φ[i] = φ[i-1] + rem2pi(φ[i] - φ[i-1], RoundNearest);  end
    X = hcat(ones(length(t)), t)
    Γfit = -(X \ log.(abs.(c)))[2];  ωfit = -(X \ φ)[2]
    return (kind = kind, kh = k * h, h = h, ha = ha, N = params.N, etas = etas, k = k, dt = dt, steps = n,
            omega = ωfit, omega_hydro = ω, Gamma = Γfit, Gamma_hydro = Γ, t_end = tend)
end

# print a table, write it to $SR_TESTOUT/<name>.toml if set, return the results
function wave_report(name, rs)
    println(name)
    @printf("  %6s %7s %6s %9s %9s %10s %10s\n", "kh", "N", "steps", "ω/ω_hyd", "Γ/Γ_hyd", "ω", "Γ")
    for r in rs
        @printf("  %6.3f %7d %6d %9.5f %9.5f %10.5g %10.5g\n", r.kh, r.N, r.steps,
                r.omega_hydro > 0 ? r.omega / r.omega_hydro : NaN, r.Gamma_hydro > 0 ? r.Gamma / r.Gamma_hydro : NaN,
                r.omega, r.Gamma)
    end
    out = get(ENV, "SR_TESTOUT", "")
    if !isempty(out)
        mkpath(out)
        open(joinpath(out, name * ".toml"), "w") do io
            TOML.print(io, Dict("run" => [Dict(string(k) => (v isa Symbol ? string(v) : v) for (k, v) in pairs(r)) for r in rs]))
        end
    end
    return rs
end

# checks: print each, count failures; exit 1 at the end if any failed
const NFAIL = Ref(0)
check(ok, msg) = (println(ok ? "  ✓ " : "  ✗ FAIL: ", msg); ok || (NFAIL[] += 1); ok)
finish(name) = (println(NFAIL[] == 0 ? "$name: PASS" : "$name: $(NFAIL[]) FAILED"); NFAIL[] == 0 || exit(1))
