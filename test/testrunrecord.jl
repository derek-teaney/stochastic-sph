# testrunrecord.jl — the run record (src/runinfo.jl) is complete: short runs of the two
# production drivers write <base>.toml, which must hold every field of SimParams, of the
# equation of state and of StochOptions, and every SR_*/SPH_* setting the library and the
# driver read; and no such setting may be read from ENV directly (src/settings.jl).
using TOML
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const ROOT = normpath(joinpath(@__DIR__, ".."))
nfail = 0
check(ok, msg) = (println(ok ? "  ✓ " : "  ✗ FAIL: ", msg); ok || (global nfail += 1))

src(f) = read(joinpath(ROOT, f), String)
libfiles = [joinpath("src", f) for f in readdir(joinpath(ROOT, "src")) if endswith(f, ".jl")]
names_read(text) = Set(m.captures[1] for m in eachmatch(r"setting\(\"((?:SR|SPH)_[A-Z0-9_]+)\"", text))
lib_names = union((names_read(src(f)) for f in libfiles)...)

function runrecord(driver, args, env)
    dir = mktempdir()
    cmd = setenv(`$(Base.julia_cmd()) -t 2 --project=$(Base.active_project()) $(joinpath(ROOT, "examples", driver)) $args`,
                 merge(copy(ENV), Dict("SR_OUTDIR" => dir), env))
    run(pipeline(cmd; stdout = devnull, stderr = devnull))
    f = only(filter(endswith(".toml"), readdir(dir)))
    return TOML.parsefile(joinpath(dir, f))
end

for (driver, args, env) in (("equilibrium.jl", ["0.6", "2.0", "0.05", "1", "2", "rr"], Dict{String,String}()),
                            ("stability.jl", ["2.0", "0.05", "1", "full", "star"], Dict{String,String}()))
    println(driver)
    rec = runrecord(driver, args, env)
    for (sec, T) in (("sim", SimParams), ("eos", ConformalEOS), ("stochastic", StochOptions))
        missing = setdiff(string.(fieldnames(T)), keys(get(rec, sec, Dict())))
        check(isempty(missing), "[$sec] has every field of $T" * (isempty(missing) ? "" : ", missing $(collect(missing))"))
    end
    wanted = union(lib_names, names_read(src(joinpath("examples", driver))))
    missing = setdiff(wanted, keys(rec["settings"]))
    check(isempty(missing), "[settings] has every setting read ($(length(wanted)))" * (isempty(missing) ? "" : ", missing $(collect(missing))"))
    check(get(rec["outcome"], "complete", false) == true, "[outcome] complete")
    check(!isempty(rec["provenance"]["commit"]), "[provenance] commit $(rec["provenance"]["commit"])")
end

println("no SR_/SPH_ setting read from ENV directly:")
for f in vcat(libfiles, ["examples/equilibrium.jl", "examples/stability.jl"])
    direct = [m.match for m in eachmatch(r"ENV\s*,\s*\"(?:SR|SPH)_[A-Z0-9_]+\"|ENV\[\"(?:SR|SPH)_[A-Z0-9_]+\"\]", src(f))]
    check(isempty(direct), f * (isempty(direct) ? "" : ": $direct"))
end
println(nfail == 0 ? "testrunrecord.jl: PASS" : "testrunrecord.jl: $nfail FAILED")
nfail == 0 || exit(1)
