# testsscheme_hmc.jl — the S-scheme in the steps (docs/sscheme_plan.tex, item D, test 7; SPH_RENORM=sscheme at load):
#   7. hybrid Monte Carlo: the energy error of a trajectory falls as δt², a few updates are accepted, no state without a Hamiltonian;
#   and the set-up: the labels of setup_ic are those of the target entropy S_func V (pa.St), and pa.V = 1/n_a.
ENV["SPH_RENORM"] = "sscheme"
haskey(ENV, "SPH_POSITIONAL") || (ENV["SPH_POSITIONAL"] = "0")   # the bare scheme (ω: testpositional*.jl); the maps are called without SimParams
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const CS2S = (SSCHEME && SS_FORM === :exact) ? (0.5,) : (0.5, 0.25)   # the exact form is tabulated for c_s² = 1/2 only
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)
SSCHEME || error("SPH_RENORM=sscheme not set at load")

println("7. hybrid Monte Carlo with SPH_RENORM=sscheme")
for c2 in CS2S
    global eos = ConstantCs2EOS(cs2_val = c2, η_by_s = 0.2)
    Sd = 54.0 / (sqrt(3) / 2 * (0.05 / 2)^2)                       # s_a = 54 at h/a = 2
    params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> Sd, (x, y) -> SVector(0.0, 0.0); L = 0.3, lattice = :hex, seed = 4))
    eSt = maximum(abs(q.St / (Sd * params.vol) - 1) for q in ps);  eV = maximum(abs(q.V / params.vol - 1) for q in ps)
    # the label: below S_t in the full form; within a few 1e-3 of S_t in the measure form (Δ^H = 0 at the reference state, V within 2e-3 of it)
    lab_ok = SS_MEASURE ? all(abs(q.s - q.St) < 0.02 for q in ps) : all(q.s < q.St for q in ps)
    check(eSt < 1e-12 && eV < 2e-3 && lab_ok,
          @sprintf("c_s² = %.2f (form %s): set-up S_t = S V to %.0e, V = 1/n_a within %.0e of the nominal volume (the lattice sum of W), label s = %.4f, S_t = %.4f", c2, SS_FORM, eSt, eV, ps[1].s, ps[1].St))
    r = quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1), bath = target_state(Sd, 0.0, eos).bath))
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
    reset_sscheme_counts!()
    acc = [hmc_update!(ps, params, cl, eos, tg, Xoshiro(7 + i); nsteps = 8, dt = dt0) for i in 1:5]
    na = count(x -> x.accepted, acc);  sc = sscheme_counts()
    check(na >= 3, @sprintf("c_s² = %.2f: %d of 5 updates accepted, heat-bath acceptance %.2f", c2, na, sum(x.heatbath for x in acc) / 5))
    check(sc.nostate == 0, @sprintf("c_s² = %.2f: %d solves from the label, %.2f Newton steps per solve, %d fallbacks, %d with no state",
                                    c2, sc.solves, sc.iters / max(1, sc.solves), sc.fallbacks, sc.nostate))
end

if get(ENV, "SR_TEST_QUICK", "0") != "1"
    println("8. (N3 iii) hybrid Monte Carlo at s_a = 16, 20 updates of length 0.6/c_s (SR_TEST_QUICK=1 skips it)")
    for c2 in CS2S
        global eos = ConstantCs2EOS(cs2_val = c2, η_by_s = 0.2)
        h = 0.6 / sqrt(672 / (2 / sqrt(3))) * 2;  S0 = 16 / (sqrt(3) / 2 * (h / 2)^2)
        params, ps, cl, _ = quiet(() -> setup_ic(h, 2.0, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex, seed = 5))
        Tr0 = trel_of_srel(eos, S0)
        r = quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1), bath = (1 / Tr0, SVector(0.0, 0.0))))
        tg = HMCTarget(r.β0, r.β, params.vol, eos)
        dt = 0.35 * compute_dt(ps, params, eos; mode = :ideal);  ns = round(Int, 0.6 / sqrt(c2) / dt)
        nacc = 0;  nexit = 0;  hb = 0.0;  reset_sscheme_counts!()
        for i in 1:20
            u = hmc_update!(ps, params, cl, eos, tg, Xoshiro(100i); nsteps = ns, dt = dt)
            nacc += u.accepted;  nexit += !isfinite(u.dH);  hb += u.heatbath / 20
        end
        smin = minimum(q.s for q in ps);  sc = sscheme_counts()
        check(nexit == 0 && nacc >= 16, @sprintf("c_s² = %.2f: %d of 20 accepted, %d left the domain, heat-bath acceptance %.3f; smallest label %.3f (b_ρ = %.2f); %d solves without a state",
                                                c2, nacc, nexit, hb, smin, SS_MEASURE ? SSchemeConsts(eos).bᵨ : NaN, sc.nostate))
    end
end

ok || error("testsscheme_hmc: failures")
println("testsscheme_hmc: all passed")
