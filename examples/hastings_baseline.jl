# hastings_baseline.jl — is the stationary distribution of the stochastic step the exact one?
# (docs/commonkernel.tex, "Detailed balance with state-dependent coefficients").
#
# Frozen 2D hexagonal lattice, equal e_a, no Hamiltonian step; the stochastic step alone, with its coefficients
# rebuilt every step from the current state (as in production, no Hastings correction).  The exact target is
# e^{𝕊} at fixed e_a and total momentum: per particle e^{S_a(e_a, m0 + δm) + μ·δm}, μ fixed by ⟨δm⟩ = 0.
# Compared: ⟨δm_x²⟩/C_xx, ⟨δm_y²⟩/C_yy (exact C by integration) and ⟨ln T_a⟩, T_a the bare particle temperature.
# A deviation that does not change when Δt is changed is the bias of the missing Hastings factor.
#
#   julia -t auto --project examples/hastings_baseline.jl
# Environment: HB_SA (comma list of s_a, default "16,54"), HB_V0 ("0,0.8"), HB_DTF (fractions of the viscous dt,
# "0.5,0.125"), HB_T (run time in units of the viscous dt, default 400), HB_NRUN (8).
using StaticArrays, LinearAlgebra, Printf, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))

envlist(k, d) = parse.(Float64, split(get(ENV, k, d), ","))
const SAS  = envlist("HB_SA", "16,54")
const V0S  = envlist("HB_V0", "0,0.8")
const DTFS = envlist("HB_DTF", "0.5,0.125")
const TRUN = parse(Float64, get(ENV, "HB_T", "400"))
const FROZEN = get(ENV, "HB_FROZEN", "0") == "1"     # 1: coefficients frozen for the whole run (exactly reversible control)
const NRUN = parse(Int, get(ENV, "HB_NRUN", "8"))
const H, HA, L = 0.05, 2.0, 0.6

function setup_moving(sa, vt)
    eos = conformal_eos(default_phys_dim(); η_by_s = 0.2)
    vol = (H / HA)^2 * sqrt(3) / 2
    params, ps, cl, eos = redirect_stdout(devnull) do
        setup_ic(H, HA, eos, (x, y) -> sa / vol, (x, y) -> SVector(0.0, 0.0); L, lattice = :hex)
    end
    vt == 0 && return params, ps, cl, eos
    # bisection on |M| along x at fixed s_a: v rises monotonically with |M|
    setM(M) = (for p in ps;  p.m = (p.s / p.S_smooth) * SVector(M, 0.0);  end;  thermalize!(ps, params, cl, eos);
               norm(ps[1].v_over_S * ps[1].S_smooth))
    lo, hi = 0.0, ps[1].e
    while setM(hi) < vt;  hi *= 2;  end
    for _ in 1:60
        mid = (lo + hi) / 2;  setM(mid) < vt ? (lo = mid) : (hi = mid)
    end
    setM((lo + hi) / 2)
    return params, ps, cl, eos
end

lnT(e, m, V, eos) = (r = try eos_from_energy(e / V, m / V, eos) catch; nothing end;
                     r === nothing || !isfinite(r[3]) || r[3] <= 0 ? NaN : log(r[3]))

# The tilted single-particle distribution on a grid: weights, mean, covariance, ⟨ln T⟩.
# The exact target of the stochastic step: e^{S_a − ln T_a} (the Liouville measure, the only one since 2026-11-27).
function tilted(e, m0, V, eos, σ, μ; n = 301)
    xs = range(-8σ, 8σ, length = n);  S0 = vote_entropy(e, m0, V, eos)
    W = zeros(n, n);  LT = fill(NaN, n, n)
    for (i, x) in enumerate(xs), (j, y) in enumerate(xs)
        Sm = vote_entropy(e, m0 + SVector(x, y), V, eos)      # the weight of the run's measure
        isfinite(Sm) || continue
        W[i, j] = exp(Sm - S0 + μ[1] * x + μ[2] * y);  LT[i, j] = lnT(e, m0 + SVector(x, y), V, eos)
    end
    Z = sum(W)
    mx = sum(W[i, j] * xs[i] for i in 1:n, j in 1:n) / Z;  my = sum(W[i, j] * xs[j] for i in 1:n, j in 1:n) / Z
    cxx = sum(W[i, j] * (xs[i] - mx)^2 for i in 1:n, j in 1:n) / Z
    cyy = sum(W[i, j] * (xs[j] - my)^2 for i in 1:n, j in 1:n) / Z
    cxy = sum(W[i, j] * (xs[i] - mx) * (xs[j] - my) for i in 1:n, j in 1:n) / Z
    lt = sum(W[i, j] * LT[i, j] for i in 1:n, j in 1:n if W[i, j] > 0) / Z
    return (xs = xs, W = W ./ Z, mean = SVector(mx, my), C = [cxx cxy; cxy cyy], lnT = lt)
end
function exact(e, m0, V, eos, σ)
    _, v0, T0 = eos_from_energy(e / V, m0 / V, eos)
    μ = collect(v0 / T0);  t = tilted(e, m0, V, eos, σ, μ)
    for _ in 1:8
        μ .-= t.C \ collect(t.mean);  t = tilted(e, m0, V, eos, σ, μ)
    end
    return t
end

# Exact start: independent draws from the tilted distribution, then the mean removed (total momentum N m0).
# Draws that reach a state no fluid can have (|v| ≥ 1: the Liouville weight grows towards that edge, and the jitter within a grid
# cell can cross it) are drawn again; the whole set again if the removal of the mean pushes one across.
function exact_draw!(ps, t, m0, e, V, eos, rng)
    cw = cumsum(vec(t.W));  n = length(t.xs);  dx = step(t.xs)
    ok(m) = isfinite(particle_entropy(e, m, V, eos))
    while true
        dm = map(ps) do _
            while true
                k = min(searchsortedfirst(cw, rand(rng) * cw[end]), length(cw));  i = (k - 1) % n + 1;  j = (k - 1) ÷ n + 1
                d = SVector(t.xs[i] + (rand(rng) - 0.5) * dx, t.xs[j] + (rand(rng) - 0.5) * dx)
                ok(m0 + d) && return d
            end
        end
        dbar = sum(dm) / length(dm)
        all(ok(m0 + d - dbar) for d in dm) || continue
        for (p, d) in zip(ps, dm);  p.m = m0 + d - dbar;  p.s = particle_entropy(e, p.m, V, eos);  end   # at the fixed e_a
        return
    end
end

function one_run(r, ps0, params, cl, eos, dt, nsteps, m0, t, e, V)
    ps = deepcopy(ps0);  N = length(ps);  rng = Xoshiro(100r + 7);  rngs = make_phys_rng(N, 7000r + 13)
    exact_draw!(ps, t, m0, e, V, eos, rng)
    x0 = sum((q.m - m0)[1]^2 for q in ps) / (N - 1) / t.C[1, 1]
    opts = StochOptions(params; frozen = FROZEN);  burn = nsteps ÷ 5
    fz = FROZEN ? stochastic_frozen(ps, params, cl, eos) : nothing
    sx = 0.0;  sy = 0.0;  slt = 0.0;  ns = 0;  acc = zeros(2)
    for n in 1:nsteps
        a = stochastic_step!(ps, params, cl, dt, eos, opts; rng, rngs, flip = isodd(n), frozen = fz)
        acc .+= (a.stress, a.resid)
        if n > burn
            dm = [q.m - m0 for q in ps]
            sx += sum(d[1]^2 for d in dm) / (N - 1);  sy += sum(d[2]^2 for d in dm) / (N - 1)
            slt += mean(lnT(e, q.m, V, eos) for q in ps);  ns += 1
        end
    end
    return (x0 = x0, xx = sx / ns / t.C[1, 1], yy = sy / ns / t.C[2, 2], lnT = slt / ns - t.lnT, acc = acc ./ nsteps)
end

let P = setup_moving(54, 0.0)[1]
    println("dissipation = moment, measure = liouville, frozen coefficients = ", FROZEN,
            ", Hastings = ", setting("SPH_HASTINGS", false))
end
for sa in SAS, v0 in V0S
    params, ps0, cl, eos = setup_moving(sa, v0)
    p = ps0[1];  N = params.N;  V = p.s / p.S_smooth;  e = particle_energy(p, eos);  m0 = p.m
    σ = sqrt(V * (p.e + p.p) * p.T)
    t = exact(e, m0, V, eos, σ)
    dtv = compute_dt(ps0, params, eos; mode = :viscous)
    @printf("\ns_a = %.4g  |v0| = %.3f  N = %d  exact: C_xx/VwT = %.4f  C_yy/VwT = %.4f  ⟨ln T⟩ = %.5f\n",
            p.s, norm(p.v_over_S * p.S_smooth), N, t.C[1, 1] / σ^2, t.C[2, 2] / σ^2, t.lnT)
    for f in DTFS
        dt = f * dtv;  nsteps = round(Int, TRUN / f);  t0 = time()
        runs = Vector{Any}(undef, NRUN)
        Threads.@threads for r in 1:NRUN
            runs[r] = one_run(r, ps0, params, cl, eos, dt, nsteps, m0, t, e, V)
        end
        st(x) = (mean(x), std(x) / sqrt(length(x)))
        xx = st([r.xx for r in runs]);  yy = st([r.yy for r in runs]);  lt = st([r.lnT for r in runs])
        acc = mean(r.acc for r in runs);  x0 = mean(r.x0 for r in runs)
        @printf("  Δt = %.3g dt_visc (%d steps, %.0f s): ⟨δm_x²⟩/C_xx = %.4f ± %.4f   ⟨δm_y²⟩/C_yy = %.4f ± %.4f   ⟨ln T⟩ − exact = %+.5f ± %.5f   acc %.3f %.3f  (start: %.4f)\n",
                f, nsteps, time() - t0, xx..., yy..., lt..., acc..., x0)
    end
end
