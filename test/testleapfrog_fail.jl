# testleapfrog_fail.jl — a failed step of the generalized leapfrog (docs/focus_plan.tex Q11; src/hmc.jl generalized_step!).
# A step 1000 times the acoustic one fails at every depth (the implicit iterations diverge or reach a state without a Hamiltonian):
#   1. the step is retried as half steps down to Δt/16 (four retries, counted in guards().leapfrog_retries), then a
#      LeapfrogFailure is thrown, with the reason, and the particles are back at the start of the step;
#   2. the driver (examples/equilibrium.jl) stops with a nonzero exit, writes <base>_leapfrog_fail.jls (the state at the start of
#      the step and the failing particles) and marks the record failed with the reason.
# (That a step that does not fail is unchanged is the regression's: bit-identical runs.)
haskey(ENV, "SPH_RENORM") || (ENV["SPH_RENORM"] = "0")
using StaticArrays, LinearAlgebra, Printf, Random, Serialization, TOML
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)

println("1. the retries and the failure")
let eos = conformal_eos(2; η_by_s = 0.2)
    params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex))
    quiet(() -> equilibrium_init!(ps, params, cl, eos; rng = Xoshiro(1), bath = target_state(1e6, 0.0, eos).bath))
    x0 = [q.x for q in ps];  m0 = [q.m for q in ps];  reset_guards!()
    dt = 1000 * compute_dt(ps, params, eos; mode = :ideal);  dotm = zeros(SVector{2,Float64}, params.N)
    err = try
        leapfrog_step!(ps, params, cl, dt, eos, dotm);  nothing
    catch e
        e
    end
    check(err isa LeapfrogFailure, "a LeapfrogFailure is thrown" * (err isa LeapfrogFailure ? ": " * sprint(showerror, err) : ""))
    check(guards().leapfrog_retries == 4, "four retries (Δt/2, /4, /8, /16) counted: $(guards().leapfrog_retries)")
    check(all(ps[a].x == x0[a] && ps[a].m == m0[a] for a in eachindex(ps)), "the particles are back at the start of the step")
    check(err isa LeapfrogFailure && err.x == x0 && err.m == m0 && err.dt == dt && !isempty(err.reason),
          "the failure holds the state at the start of the step, Δt and the reason")
end

println("2. the driver stops, saves the state and marks the record failed")
let out = mktempdir(get(ENV, "SR_TMP", tempdir()))
    drv = joinpath(@__DIR__, "..", "examples", "equilibrium.jl")
    cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t 1 $drv 0.6 2.0 0.2 1 2 lf`,
                 "SR_OUTDIR" => out, "SR_START" => "cold", "SR_DTFAC" => "1000", "SPH_RENORM" => "0")
    p = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = devnull))
    check(p.exitcode != 0, "nonzero exit ($(p.exitcode))")
    f = joinpath(out, "equilibrium_lf_r1_leapfrog_fail.jls")
    check(isfile(f) && (d = deserialize(f); d.step == 1 && !isempty(d.x) && d.dt > 0), "the state at the start of the step saved")
    rec = joinpath(out, "equilibrium_lf_r1.toml")
    o = isfile(rec) ? get(TOML.parsefile(rec), "outcome", Dict()) : Dict()
    check(get(o, "failed", false) == true && occursin("step 1", get(o, "reason", "")) && get(o, "complete", true) == false,
          "the record is marked failed with the reason: $(get(o, "reason", "?"))")
    rm(out; recursive = true)
end
println(ok ? "testleapfrog_fail: PASS" : "testleapfrog_fail: FAIL");  ok || exit(1)
