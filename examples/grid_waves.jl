# grid_waves.jl — deterministic tests of the grid step (docs/gridstep.tex, "Tests" 1–4),
# against the present particle step (leapfrog, n-scheme), ideal step only (η = 0).
#
# A uniform conformal fluid (rest-frame entropy density S0, hexagonal lattice, h/a = 2) moving
# with v0 along x, plus one plane wave δU = A Re(r_n e^{ik·x}), U = (ε, M_x, M_y), r_n the right
# eigenvector of the ideal K = k_j ∂F_j/∂U for the mode n (shear, sound+, sound−).  The mode
# amplitude c(t) = (1/N) Σ_a ℓ_n·(U_a − U_0) e^{−ik·x_a} (ℓ_n the left eigenvector) is fitted to
# c₀ e^{−iωt − Γt}; ideal hydrodynamics has ω = ω_n, Γ = 0, so Γ is the numerical damping
# (Γ < 0: growth).  For the grid also ΣΔH/H and the entropy made by the return ΣΔS_T/S.
# Also: the uniform state (test 1) and a Δt scan (ΔH/H against Δt).
#
# Run (compute node):  julia -t 64 --project=nersc examples/grid_waves.jl [out_dir]
#   → <out>/grid_waves.toml (the table), <out>/grid_waves.jls (the series, for plots)
using StaticArrays, LinearAlgebra, Printf, Random, Serialization, TOML
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const OUT  = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "grid_waves");  mkpath(OUT)
const S0   = 1e6;  const H = 0.05;  const HA = setting("SR_HA", 2.0);  const LBOX = 1.2
const AMP  = 1e-4                                   # δU amplitude relative to ε0
const PERIODS = setting("SR_PERIODS", setting("SR_SMOKE", false) ? 0.3 : 10.0)        # run length in sound periods 2π/(c_s k)
const EOS0 = ConformalEOS(1/3, 1.0, 0.0, 0.0);  const CS = sqrt(1/3)
quiet(f) = redirect_stdout(f, devnull)

# ── ideal hydrodynamics of U = (ε, M_x, M_y) ─────────────────────────────────
function state(U)
    ε = U[1];  M = SVector(U[2], U[3])
    er = rest_energy(ε, dot(M, M), EOS0);  p = pressure(EOS0, er)
    return p, M / (ε + p)
end
flux(U, j) = ((p, v) = state(U);  SVector(U[1 + j], U[2] * v[j] + (j == 1) * p, U[3] * v[j] + (j == 2) * p))
function Kmat(U, k)
    J = zeros(3, 3)
    for i in 1:3
        dU = zeros(3);  δ = 1e-6 * U[1];  dU[i] = δ
        Up = U + SVector{3}(dU);  Um = U - SVector{3}(dU)
        J[:, i] = k[1] * (flux(Up, 1) - flux(Um, 1)) / 2δ + k[2] * (flux(Up, 2) - flux(Um, 2)) / 2δ
    end
    return J
end
# modes: (ω, r, ℓ) for :shear (ω closest to k·v0), :soundp (largest ω), :soundm (smallest)
function modes(U0, k, v0)
    F = eigen(Kmat(U0, k));  ω = real.(F.values);  R = real.(F.vectors);  L = inv(R)
    i_sh = argmin(abs.(ω .- k[1] * v0));  rest = setdiff(1:3, i_sh)
    i_p = rest[argmax(ω[rest])];  i_m = rest[argmin(ω[rest])]
    return Dict(:shear => (ω[i_sh], R[:, i_sh], L[i_sh, :]), :soundp => (ω[i_p], R[:, i_p], L[i_p, :]),
                :soundm => (ω[i_m], R[:, i_m], L[i_m, :]))
end

# ── one run ──────────────────────────────────────────────────────────────────
function background(v0)
    _, q, _, _ = quiet(() -> setup_ic(H, HA, EOS0, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex))
    w0 = q[1].e + q[1].p;  p0 = q[1].p;  γ = 1 / sqrt(1 - v0^2)
    return SVector(w0 * γ^2 - p0, w0 * γ^2 * v0, 0.0)
end
const TREL0 = (q = quiet(() -> setup_ic(H, HA, EOS0, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex))[2][1];  q.T * q.γ)

# damping of mode ω (ideal frequency) in the moving fluid (boost.tex): in the rest frame
# Ω = γ(ω − v k_x), k′_x = γ(k_x − v ω); shear Γ′ = ν k′², sound Γ′ = ½(4/3) ν k′²;
# lab Γ = Γ′ / (γ (1 + v ∂Ω/∂k′_x)), ∂Ω/∂k′_x = c_s² k′_x/Ω for sound.
function gamma_hydro(mode, ω, k, v0, etas)
    etas == 0 && return 0.0
    ν = etas / TREL0;  γ = 1 / sqrt(1 - v0^2)
    Ω = γ * (ω - v0 * k[1]);  kx = γ * (k[1] - v0 * ω);  k2 = kx^2 + k[2]^2
    mode === :shear && return ν * k2 / γ
    return 0.5 * (4 / 3) * ν * k2 / (γ * (1 + v0 * (1 / 3) * kx / Ω))
end

# lattice, stencil and the state (s_a, m_a) with V_a = 1/n_a, for δU(x) on the background U0
function make_state(U0, δU)
    params, ps, cl, _ = quiet(() -> setup_ic(H, HA, EOS0, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex))
    st = GridStencil(ps, params, cl)
    for pa in ps
        U = U0 + δU(pa.x)
        s, _, _, _, ok = grid_inverse(st.V * U[1], st.V * SVector(U[2], U[3]), st.V, EOS0)
        ok || error("no state at the start")
        pa.s = s;  pa.m = st.V * SVector(U[2], U[3])
    end
    thermalize!(ps, params, cl, EOS0)
    return params, ps, cl, st
end

dtfac(v0) = v0 == 0 ? 1.0 : CS * (1 + v0 * CS) / (v0 + CS)      # c_s / v_signal, as in boost.tex

# U_a densities of every particle (smoothed state current): ε = s ℰ / V_a, M = m / V_a, V_a = s/S
densities(ps) = [SVector(p.s * p.H_over_S, p.m[1], p.m[2]) * (p.S_smooth / p.s) for p in ps]

function wave(scheme::Symbol, mode::Symbol, dir::Symbol, v0::Float64, nk::Int; dtmul = 1.0, periods = PERIODS, etas = 0.0)
    eos = ConformalEOS(1/3, 1.0, etas, 0.0)
    U0 = background(v0)
    params0, _, _, _ = quiet(() -> setup_ic(H, HA, EOS0, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex))
    k = dir === :x ? SVector(2π * nk / params0.L[1], 0.0) : SVector(0.0, 2π * nk / params0.L[2])
    ωn, r, ℓ = modes(U0, k, v0)[mode]
    δU = x -> AMP * U0[1] * (r / norm(r)) * cos(dot(k, x))
    params, ps, cl, st = make_state(U0, δU)
    dt0 = (etas == 0 ? 0.2 * H / CS : 0.5 * compute_dt(ps, params, eos; mode = :viscous)) * dtfac(v0);  tend = periods * 2π / (CS * norm(k))
    dt = dt0 * dtmul;  n = ceil(Int, tend / dt);  dt = tend / n
    sc = GridScratch(st.N);  g = GridStats();  dotm = zeros(SVector{2,Float64}, st.N)
    for a in eachindex(ps);  sc.er[a] = ps[a].e;  end
    amp() = (U = densities(ps);  sum(dot(ℓ, U[a] - U0) * cis(-dot(k, ps[a].x)) for a in eachindex(ps)) / length(ps))
    Hof() = sum(p.s * p.H_over_S for p in ps)
    every = max(1, n ÷ 400)
    ts = [0.0];  cs = [amp()];  H0 = Hof();  S00 = sum(p.s for p in ps);  fails = 0
    t0 = time()
    for i in 1:n
        if scheme === :grid
            etas > 0 && viscous_step!(ps, params, cl, dt / 2, eos)
            fails += grid_step!(ps, st, dt, eos, sc, g)
            fails > 0 && break
            etas > 0 && viscous_step!(ps, params, cl, dt / 2, eos)
        else
            etas > 0 ? generic_step!(ps, params, cl, dt, eos, dotm) : leapfrog_step!(ps, params, cl, dt, eos, dotm)
        end
        if i % every == 0 || i == n
            smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
            push!(ts, i * dt);  push!(cs, amp())
        end
    end
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    φ = angle.(cs);  for i in 2:length(φ);  φ[i] = φ[i-1] + rem2pi(φ[i] - φ[i-1], RoundNearest);  end
    X = hcat(ones(length(ts)), ts)
    Γ = -(X \ log.(abs.(cs)))[2];  ω = -(X \ φ)[2]
    return (scheme = string(scheme), mode = string(mode), dir = string(dir), v0 = v0, nk = nk, kh = norm(k) * H,
            etas = etas, omega = ω, omega_hydro = ωn, Gamma = Γ, Gamma_hydro = gamma_hydro(mode, ωn, k, v0, etas), Gamma_over_k2 = Γ / norm(k)^2, ratio_end = abs(cs[end] / cs[1]),
            dt = dt, dtmul = dtmul, steps = n, t_end = tend, dH_over_H = Hof() / H0 - 1,
            dS_over_S = sum(p.s for p in ps) / S00 - 1, dS_T_sum = g.sum_dS_T, fails = fails,
            disp_max = g.disp_max, wall = time() - t0), (ts = ts, cs = cs)
end

# ── test 1: the uniform state ────────────────────────────────────────────────
println("test 1: uniform state, 200 steps of the grid step")
uni = []
for v0 in (0.0, 0.3, 0.8)
    U0 = background(v0);  params, ps, cl, st = make_state(U0, x -> zero(SVector{3,Float64}))
    sc = GridScratch(st.N);  g = GridStats();  s0 = [p.s for p in ps];  m0 = [p.m for p in ps]
    dt = 0.2 * H / CS * dtfac(v0)
    c = grid_checks(st, params, EOS0, dt);  v0 == 0 && print_grid_checks(c)
    for _ in 1:(setting("SR_SMOKE", false) ? 5 : 200);  grid_step!(ps, st, dt, EOS0, sc, g);  end
    ds = maximum(abs(ps[a].s / s0[a] - 1) for a in eachindex(ps))
    dm = maximum(norm(ps[a].m - m0[a]) for a in eachindex(ps)) / (st.V * U0[1])
    @printf("  v0 = %.1f: max|Δs/s| = %.1e  max|Δm|/(Vε) = %.1e  ΣΔH/H = %+.1e  fails %d\n", v0, ds, dm, g.sum_dH, g.eos_fail)
    push!(uni, (v0 = v0, ds = ds, dm = dm, dH = g.sum_dH, checks = c))
end

# ── tests 2–4: waves ─────────────────────────────────────────────────────────
cases = [(sch, mode, dir, v0, nk) for sch in (:grid, :particle), mode in (:shear, :soundp, :soundm),
         dir in (:x, :y), v0 in (0.0, 0.3, 0.8), nk in (1, 2, 4)]
# Δt scan (grid): sound+ and shear along the flow at v0 = 0.8, kh = 0.52, to a fixed time
dtcases = [(:grid, mode, :x, 0.8, 2, m) for mode in (:soundp, :shear) for m in (1.0, 0.5, 0.25)]
# the same waves with deterministic viscosity, η/s = ETAS_V (the viscous step of the particle code
# around the ideal step: generic_step! for the particles, viscous ½ · grid · viscous ½ for the grid)
const ETAS_V = setting("SR_ETAS", 0.2)
vcases = cases
if setting("SR_SMOKE", false)                      # seconds: two short runs, code paths only
    cases = [(:grid, :soundp, :x, 0.8, 4), (:particle, :shear, :y, 0.3, 4)];  dtcases = dtcases[1:1];  vcases = cases
end
if setting("SR_QUICK", false)                      # a reduced set: v0 = 0, 0.8; kh = 0.52, 1.05; shear, sound+
    vcases = [(sch, mode, dir, v0, nk) for sch in (:grid, :particle), mode in (:shear, :soundp),
              dir in (:x, :y), v0 in (0.0, 0.8), nk in (2, 4)]
    cases = vcases;  dtcases = []                  # the same set for the ideal step, no Δt scan
end
const ONLY = setting("SR_ONLY", "all")             # all | ideal | viscous
ONLY == "viscous" && (cases = [];  dtcases = [])
ONLY == "ideal" && (vcases = [])
println("tests 2–4: $(length(cases)) wave runs + $(length(dtcases)) Δt-scan runs on $(Threads.nthreads()) threads")
res = Vector{Any}(undef, length(cases));  dres = Vector{Any}(undef, length(dtcases))
Threads.@threads :dynamic for i in eachindex(cases)
    res[i] = wave(cases[i]...)
end
Threads.@threads :dynamic for i in eachindex(dtcases)
    c = dtcases[i];  dres[i] = wave(c[1:5]...; dtmul = c[6], periods = min(5.0, PERIODS))
end
println("viscous: $(length(vcases)) wave runs at η/s = $ETAS_V")
vres = Vector{Any}(undef, length(vcases))
Threads.@threads :dynamic for i in eachindex(vcases)
    vres[i] = wave(vcases[i]...; etas = ETAS_V)
end

function table(rs, title)
    println(title)
    @printf("  %-8s %-6s %-2s %4s %5s %10s %10s %11s %9s %10s %9s %10s %10s %5s\n", "scheme", "mode", "k", "v0", "kh",
            "ω/k", "ω_hyd/k", "Γ/(c_s k)", "Γ/Γ_hyd", "|c(t)/c0|", "ΔH/H", "ΔS/S", "ΣΔS_T/S", "fail")
    for (r, _) in rs
        k = r.kh / H
        @printf("  %-8s %-6s %-2s %4.1f %5.2f %10.5f %10.5f %+11.2e %9.3f %10.4f %+9.1e %+10.1e %+10.1e %5d\n", r.scheme, r.mode,
                r.dir, r.v0, r.kh, r.omega / k, r.omega_hydro / k, r.Gamma / (CS * k),
                r.Gamma_hydro > 0 ? r.Gamma / r.Gamma_hydro : NaN, r.ratio_end, r.dH_over_H,
                r.dS_over_S, r.dS_T_sum, r.fails)
    end
end
table(res, "waves (Γ: numerical damping, ideal hydrodynamics has 0; run length $(PERIODS) sound periods)")
isempty(vres) || table(vres, "viscous waves, η/s = $ETAS_V (Γ_hyd: boosted hydrodynamics)")
table(dres, "Δt scan (grid, v0 = 0.8, k along the flow, kh = 0.52, 5 periods); Δt × $(join([c[6] for c in dtcases], ", "))")

open(joinpath(OUT, "grid_waves_$(ONLY).toml"), "w") do io
    TOML.print(io, Dict("uniform" => [Dict("v0" => u.v0, "max_ds" => u.ds, "max_dm" => u.dm, "dH" => u.dH) for u in uni],
                        "checks" => Dict(string(k) => v for (k, v) in pairs(uni[1].checks)),
                        "wave" => [Dict(string(k) => v for (k, v) in pairs(r)) for (r, _) in res],
                        "dtscan" => [Dict(string(k) => v for (k, v) in pairs(r)) for (r, _) in dres],
                        "viscous" => [Dict(string(k) => v for (k, v) in pairs(r)) for (r, _) in vres]))
end
serialize(joinpath(OUT, "grid_waves_$(ONLY).jls"), (res = res, dres = dres, vres = vres))
println("written to $OUT")
