# testcheckpoint.jl — restart of examples/stochrep_equilibrium.jl from a checkpoint.
# Runs the driver three times (subprocesses): once straight through, once stopped after
# step 41 (SR_STOP_AFTER), and once more to resume.  The resumed output must be
# bit-identical to the uninterrupted one; the partial output must be marked incomplete.
#
# Run:  julia --project test/testcheckpoint.jl
using HDF5, Printf
check(ok, msg) = (println(ok ? "  ✓ " : "  ✗ FAIL: ", msg); ok || error(msg))

drv  = joinpath(@__DIR__, "..", "examples", "stochrep_equilibrium.jl")
jl   = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t 2`
args = ["0.6", "2.0", "1.0", "1", "2", "ck"]                     # L h/a t_end nreal rec_every tag
root = mktempdir(get(ENV, "SR_TMP", tempdir()));  A = joinpath(root, "A");  B = joinpath(root, "B")
drive(out; env...) = run(addenv(`$jl $drv $args`, "SR_OUTDIR" => out, "SR_START" => "eq",
                                [string(k) => string(v) for (k, v) in env]...))

println("uninterrupted run");  drive(A)
println("run stopped after step 41");  drive(B; SR_STOP_AFTER = 41)
fB = joinpath(B, "equilibrium_ck_r1.h5");  cB = joinpath(B, "equilibrium_ck_r1.chk")
check(isfile(cB), "checkpoint written")
h5open(fB) do f
    check(read(attributes(f)["complete"]) == false && read(attributes(f)["nsteps_done"]) == 41,
          "partial output marked complete = false, nsteps_done = 41")
end
println("resumed run");  drive(B)
check(!isfile(cB), "checkpoint removed after completion")

fA = joinpath(A, "equilibrium_ck_r1.h5")
h5open(fA) do a
    h5open(fB) do b
        for k in ("ks", "t", "ML", "MT", "S", "H", "KE")
            check(read(a[k]) == read(b[k]), "dataset $k identical")
        end
        for k in ("complete", "nsteps_done", "acc", "guards_rejected", "nsteps", "dt")
            check(read(attributes(a)[k]) == read(attributes(b)[k]), "attribute $k identical")
        end
    end
end
rm(root; recursive = true)
println("All testcheckpoint.jl checks passed.")
