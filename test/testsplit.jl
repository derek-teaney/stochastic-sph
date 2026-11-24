# testsplit.jl — the split renormalization, Step 5 (src/renorm_split.jl, SPH_RENORM=split; docs/step5_plan.tex, D):
#   1. the split adds up to the counterterm of legendre_plan.tex, Δs_lab(u) + Δs_dyn = δln Z_1, to 1e-13;
#   2. reference values of H(u, m, V) from the model of examples/renorm_quadrature_split.py (V̄ = 1), to 1e-10 relative;
#   3. the identities of the forward map by central differences of H: ∂H/∂u = pa.T, ∂H/∂m = ẋ, −∂H/∂V = pa.p;
#   4b. the inverse from the particle's label (the votes' warm start) = the cold inverse;
#   4. the inverse of the forward map is the identity, u(H(u, m, V), m, V) = u, to 3e-12 (the target's solve, eos_newton, to 1e-12);
#   5. at c_s² = 1/2 (𝒩 = 0) with the sound term off, the vote weight equals that of SPH_RENORM=1 (stochastic.jl, renorm_vote_entropy) to 1e-12;
#   6. hybrid Monte Carlo: the energy error of a trajectory falls as δt², a few updates are accepted, no failed solves.
ENV["SPH_RENORM"] = "split"
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
SPLIT || error("SPH_RENORM=split not set at load")
SPLIT_VBAR[] = 1.0                                   # as the quadrature script (Δs_dyn = 𝒩 ln(γV))
rel(a, b) = b == 0 ? abs(a) : abs(a / b - 1)

# the forward map of the particle (u, m, V), cold; H = u HS
fwd(u, m, V, eos; kw...) = split_forward(u, u / V, m / V, 1.0, eos; kw...)
Hof(u, m, V, eos; kw...) = u * fwd(u, m, V, eos; kw...).HS
# the target state (T, |v|, V) → (u, |m|) (step5_plan.tex, "The target state")
function target_um(T, v, V, eos)
    c = SplitConsts(eos);  γ = 1 / sqrt(1 - v^2)
    return c.n * c.A * V * γ^c.n * T^(c.n - 1), c.n * c.A * V * γ^(c.n + 2) * T^c.n * v
end

println("1. the split adds up to the counterterm")
for c2 in (0.5, 0.25), (T, v, V) in ((3.0, 0.0, 4.0), (3.0, 0.8, 4.0), (1.7, 0.95, 0.5), (0.9, 0.3, 2.0))
    eos = ConstantCs2EOS(cs2_val = c2);  lc = LegendreConsts(eos);  c = SplitConsts(eos);  γ = 1 / sqrt(1 - v^2);  q = 1 - c2 * v^2
    u, _ = target_um(T, v, V, eos)
    Δleg = 2.5 * log(V) + 2.5 * log(γ) + lc.k * log(γ * T) + 0.5 * log(q) + lc.C           # renorm.jl, Δ
    Δlab = c.b * log(u) + lc.C - lc.k * c2 * log(c.n * c.A)
    Δdyn = c.N * log(γ * V) + 0.5 * log(q)
    err = abs(Δlab + Δdyn - Δleg) / abs(Δleg)
    check(err < 1e-13, @sprintf("c_s² = %.2f, T = %.1f, v = %.2f, V = %.1f: |Δs_lab + Δs_dyn − δln Z_1| / |δln Z_1| = %.1e", c2, T, v, V, err))
end

println("2. reference values (examples/renorm_quadrature_split.py, without ∂Δs_dyn/∂u)")
ref = [(0.5, 3.0, 0.0, 4.0, 5.400000000000e+01, 0.000000000000e+00, 1.080000000000e+02),
       (0.5, 3.0, 0.8, 4.0, 2.500000000000e+02, 1.666666666667e+03, 1.832754839612e+03),
       (0.5, 1.7, 0.95, 0.5, 7.119544069180e+01, 1.179288581715e+03, 1.200502223280e+03),
       (0.5, 0.9, 0.3, 2.0, 2.799266102455e+00, 8.305514809483e-01, 1.908005333432e+00),
       (0.25, 3.0, 0.0, 4.0, 4.050000000000e+02, 0.000000000000e+00, 9.740794415417e+02),
       (0.25, 3.0, 0.8, 4.0, 5.208333333333e+03, 3.472222222222e+04, 4.028036192767e+04),
       (0.25, 1.7, 0.95, 0.5, 1.758588235891e+03, 2.912943590733e+04, 3.006482695223e+04),
       (0.25, 0.9, 0.3, 2.0, 2.076378702371e+00, 6.160684061979e-01, 2.002708677494e+00)]
for (c2, T, v, V, u, m, H) in ref
    eos = ConstantCs2EOS(cs2_val = c2)
    err = rel(Hof(u, SVector(m, 0.0), V, eos), H)
    check(err < 1e-10, @sprintf("c_s² = %.2f, T = %.1f, v = %.2f, V = %.1f: H to %.1e", c2, T, v, V, err))
end

println("3. the fields of the forward map (central differences of H)")
for c2 in (0.5, 0.25), (T, v, V) in ((3.0, 0.0, 4.0), (3.0, 0.8, 4.0), (1.7, 0.95, 0.5), (2.0, 0.5, 1.3))
    eos = ConstantCs2EOS(cs2_val = c2)
    u, mm = target_um(T, v, V, eos);  m = SVector(mm * 0.6, mm * 0.8);  st = fwd(u, m, V, eos)
    hu = 1e-5 * u;  hm = 1e-5 * (norm(m) + u * T * 1e-2);  hV = 1e-5 * V
    dHu = (Hof(u + hu, m, V, eos) - Hof(u - hu, m, V, eos)) / (2hu)
    dHm = SVector((Hof(u, m + SVector(hm, 0.0), V, eos) - Hof(u, m - SVector(hm, 0.0), V, eos)) / (2hm),
                  (Hof(u, m + SVector(0.0, hm), V, eos) - Hof(u, m - SVector(0.0, hm), V, eos)) / (2hm))
    dHV = (Hof(u, m, V + hV, eos) - Hof(u, m, V - hV, eos)) / (2hV)
    eT = rel(st.T, dHu);  ex = norm(st.v - dHm) / max(norm(dHm), 1e-3);  eP = abs(st.p + dHV) / abs(dHV)
    check(max(eT, ex, eP) < 1e-8, @sprintf("c_s² = %.2f, T = %.1f, v = %.2f: ∂H/∂u = T to %.1e, ∂H/∂m = ẋ to %.1e (|ẋ| = %.4f, |v| = %.2f), −∂H/∂V = P to %.1e",
                                           c2, T, v, eT, ex, norm(st.v), v, eP))
end

println("4. the inverse of the forward map")
for c2 in (0.5, 0.25)
    eos = ConstantCs2EOS(cs2_val = c2);  worst = 0.0;  nst = 0;  rng = Xoshiro(3)
    for sa in (8.0, 16.0, 54.0, 541.0, 1e4), γ in (1.0, 1.5, 4.0, 20.0)
        v = sqrt(1 - 1 / γ^2);  T = 1.0
        c = SplitConsts(eos);  V = sa / (c.n * c.A * γ^c.n * T^(c.n - 1))      # the target's label sa at (T, v)
        u, mm = target_um(T, v, V, eos);  ϕ = 2π * rand(rng);  m = mm * SVector(cos(ϕ), sin(ϕ))
        st = fwd(u, m, V, eos);  isfinite(st.T) || continue
        e = u * st.HS
        ui, xd, TH = split_inverse(e, m, V, eos)
        worst = max(worst, abs(ui - u) / u, rel(TH, st.T));  nst += 1
    end
    check(worst < 3e-12, @sprintf("c_s² = %.2f: %d states (s_a 8–1e4, γ 1–20), max |u(H)/u − 1|, |∂H/∂u| rel. = %.1e", c2, nst, worst))
end

println("4b. warm and cold starts of the inverse (the votes start from the particle's label)")
for c2 in (0.5, 0.25)
    eos = ConstantCs2EOS(cs2_val = c2);  c = SplitConsts(eos);  worst = 0.0;  rng = Xoshiro(11);  nst = 0
    itw = 0;  itc = 0
    for sa in (8.0, 16.0, 54.0, 541.0), γ in (1.0, 1.5, 4.0)
        v = sqrt(1 - 1 / γ^2);  V = sa / (c.n * c.A * γ^c.n)
        u, mm = target_um(1.0, v, V, eos);  ϕ = 2π * rand(rng);  m = mm * SVector(cos(ϕ), sin(ϕ))
        st = fwd(u, m, V, eos);  isfinite(st.T) || continue
        # a vote: the energy and momentum moved a little from the particle's state (u, m, V)
        δm = 0.05 * sqrt(u) * SVector(randn(rng), randn(rng));  e = u * st.HS + 0.05 * st.T * sqrt(u) * randn(rng)
        reset_split_counts!();  uc, _, Tc = split_inverse(e, m + δm, V, eos);  itc += split_counts().iters
        reset_split_counts!();  uw, _, Tw = split_inverse(e, m + δm, V, eos; guess = (u, SVector(st.e, 0.0)));  itw += split_counts().iters
        isfinite(uc) || continue
        worst = max(worst, abs(uw / uc - 1), rel(Tw, Tc));  nst += 1
    end
    check(worst < 3e-12, @sprintf("c_s² = %.2f: %d votes, warm = cold to %.1e; Newton iterations per solve: cold %.2f, warm %.2f", c2, nst, worst, itc / nst, itw / nst))
end

println("5. c_s² = 1/2, sound term off: the votes of SPH_RENORM=1")
let eos = conformal_eos(2), worst = 0.0, rng = Xoshiro(5), nst = 0
    for sa in (8.0, 16.0, 54.0, 541.0), γ in (1.0, 1.3, 3.0)
        v = sqrt(1 - 1 / γ^2);  c = SplitConsts(eos);  V = sa / (c.n * c.A * γ^c.n)
        u, mm = target_um(1.0, v, V, eos);  ϕ = 2π * rand(rng);  m = mm * SVector(cos(ϕ), sin(ϕ))
        e = Hof(u, m, V, eos; sound = false)
        for lv in (true, false)
            s5 = split_vote_entropy(e, m, V, eos, lv; sound = false)
            # renorm_vote_entropy of stochastic.jl with the target's inverse (α/2 = 5/2)
            ut, _, Tt = target_inverse(e, m, V, eos)
            s1 = ut - 2.5 * log(ut);  lv && (s1 -= log(Tt / (1 - 2.5 / ut)))
            worst = max(worst, abs(s5 - s1) / abs(s1));  nst += 1
        end
    end
    check(worst < 1e-12, @sprintf("%d weights, max relative difference %.1e", nst, worst))
end

println("6. hybrid Monte Carlo with SPH_RENORM=split")
for c2 in (0.5, 0.25)
    global eos = ConstantCs2EOS(cs2_val = c2, η_by_s = 0.2)
    params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 54.0 / (sqrt(3) / 2 * (0.05 / 2)^2), (x, y) -> SVector(0.0, 0.0);
                                             L = 0.3, lattice = :hex, seed = 4))
    SPLIT_VBAR[] = NaN;  check_split(params)                       # V̄ = L²/N, as a run
    r = quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1)))
    Hnow() = hmc_measure!(ps, params, cl, eos)[1]
    dt0 = 0.5 * compute_dt(ps, params, eos; mode = :ideal)
    x0 = [q.x for q in ps];  m0 = [q.m for q in ps]
    errs = Float64[]
    for f in (1, 2, 4)
        setx!(ps, params, cl, x0);  setm!(ps, m0);  H0 = Hnow()
        hmc_trajectory!(ps, params, cl, eos, dt0 / f, 8f)
        push!(errs, abs(Hnow() - H0) / abs(H0))
    end
    setx!(ps, params, cl, x0);  setm!(ps, m0);  Hnow()
    ratio = errs[1] / errs[3]
    check(10 < ratio < 25, @sprintf("c_s² = %.2f: |ΔH/H| at δt, δt/2, δt/4: %.2e %.2e %.2e (ratio %.1f, δt² gives 16)", c2, errs..., ratio))
    tg = HMCTarget(r.β0, r.β, params.vol, eos)
    reset_split_counts!()
    acc = [hmc_update!(ps, params, cl, eos, tg, Xoshiro(7 + i); nsteps = 8, dt = dt0) for i in 1:5]
    na = count(x -> x.accepted, acc);  sc = split_counts()
    check(na >= 3, @sprintf("c_s² = %.2f: %d of 5 updates accepted, heat-bath acceptance %.2f", c2, na, sum(x.heatbath for x in acc) / 5))
    check(sc.fails == 0, @sprintf("c_s² = %.2f: %d inverse solves, %.2f iterations per solve, %d failures, %d with no state",
                                  c2, sc.solves, sc.iters / max(1, sc.solves), sc.fails, sc.nostate))
end

ok || error("testsplit: failures")
println("testsplit: all passed")
