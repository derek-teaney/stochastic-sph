#!/bin/bash
# run.sh — the test suite (docs/codeplan.tex, "The test suite").
#
#   test/run.sh [unit|physics|regress|load|all] [-r <ref>] [-o <outdir>] [-j <jobs>]
#
#   unit     every test/*.jl, in parallel, one thread each.  A test fails if it exits nonzero;
#            a test whose first line is "# test: visual" only makes plots and is reported as
#            "visual".
#   regress  no change to the numerics against the commit <ref> (default main), built in a git
#            worktree: test/regress/step.jl bit-identical for the n-scheme and the smoothed
#            Hamiltonian (at 1 and 4 threads), short fixed-seed runs of the equilibrium driver
#            (at rest and boosted) and of the stability driver (smoothed and energy) with
#            bit-identical data, and every unit test with the same output (timing masked).
#   physics  test/physics/*.jl, four threads each: plane waves (ideal and viscous sound, shear)
#            at kh = 1.4, 0.7, 0.35 against linearized hydrodynamics, with tolerances; the
#            measured ω and Γ go to <outdir>/physics/<test>.toml (minutes).
#   load     every examples/*.jl and test/*.jl still loads (its using/include lines run).
#   all      unit, physics, load, regress.
#
# Output in <outdir> (default data/tests/<date>): one log per test, results.toml with one
# [[test]] table per test (name, kind, status, seconds), and a summary on stdout.  Exits 1 if
# anything fails.  Environment: SR_TESTPROJECT, the Julia project (default: the repository;
# it needs PyPlot for the visual tests), MPLBACKEND (default Agg).
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:-unit};  [ $# -gt 0 ] && shift
REF=main;  OUT=$R/data/tests/$(date +%Y%m%d-%H%M%S);  JOBS=${JOBS:-8}
while getopts "r:o:j:" o; do case $o in r) REF=$OPTARG;; o) OUT=$OPTARG;; j) JOBS=$OPTARG;; *) exit 2;; esac; done
PROJ=${SR_TESTPROJECT:-$R};  export MPLBACKEND=${MPLBACKEND:-Agg}
mkdir -p "$OUT";  OUT=$(cd "$OUT" && pwd);  RES=$OUT/results.toml;  : > "$RES";  FAILED=0
cd "$R"

record() {   # name kind status seconds
    printf '[[test]]\nname = "%s"\nkind = "%s"\nstatus = "%s"\nseconds = %s\n\n' "$1" "$2" "$3" "$4" >> "$RES"
    printf '  %-8s %-34s %-7s %5s s\n' "$2" "$1" "$3" "$4"
    [ "$3" = fail ] && FAILED=1
}

# one unit test: <repo> <outdir> <file>; prints "name status seconds"
cat > "$OUT/one_unit.sh" <<'EOS'
#!/bin/bash
r=$1; o=$2; f=$3; n=$(basename "$f" .jl); t0=$(date +%s)
cd "$r" && julia -t 1 --startup-file=no --project="$PROJ" "$f" > "$o/$n.log" 2>&1;  e=$?
s=pass; [ $e -ne 0 ] && s=fail; head -1 "$f" | grep -q '^# test: visual' && [ $e -eq 0 ] && s=visual
echo "$n $s $(( $(date +%s) - t0 ))"
EOS
chmod +x "$OUT/one_unit.sh";  export PROJ

run_unit() {   # <repo> <outdir>: every <repo>/test/*.jl → <outdir>/<name>.log and <outdir>/unit.txt
    mkdir -p "$2"
    ls "$1"/test/*.jl | grep -v '/testregress_step.jl$' | xargs -P "$JOBS" -n 1 "$OUT/one_unit.sh" "$1" "$2" | sort > "$2/unit.txt"
}

# log comparison, timing, temporary paths and commit hashes masked; test/regress/logmask.sed maps renames
mask() { grep -v -E 'seconds|elapsed|threads, ' "$1" | sed -E 's/wall= *[0-9.]+ *s/wall/g; s/jl_[A-Za-z0-9]+/jl_TMP/g; s/[0-9]+\.[0-9]+ s\b/T s/g; s/(steps=[0-9]+) [0-9]+ s\b/\1 T s/g; s/commit [0-9a-f]{7,}(\+dirty)?/commit HASH/g' | sed -E -f "$R/test/regress/logmask.sed"; }

if [ "$MODE" = unit ] || [ "$MODE" = all ]; then
    echo "unit tests ($(git rev-parse --short HEAD), $JOBS at a time):"
    run_unit "$R" "$OUT/unit"
    while read -r n s t; do record "$n" unit "$s" "$t"; done < "$OUT/unit/unit.txt"
fi

if [ "$MODE" = physics ] || [ "$MODE" = all ]; then
    echo "physics tests:"
    P=$OUT/physics;  mkdir -p "$P"
    for f in test/physics/*.jl; do
        n=$(basename "$f" .jl)
        ( t0=$(date +%s); SR_TESTOUT=$P julia -t 4 --startup-file=no --project="$PROJ" "$f" > "$P/$n.log" 2>&1
          echo "$n $([ $? = 0 ] && echo pass || echo fail) $(( $(date +%s) - t0 ))" > "$P/$n.status" ) &
    done; wait
    for st in "$P"/*.status; do read -r n s t < "$st"; record "$n" physics "$s" "$t"; done
fi

if [ "$MODE" = load ] || [ "$MODE" = all ]; then
    echo "load check:"
    L=$OUT/load;  mkdir -p "$L"
    for f in examples/*.jl test/*.jl test/regress/*.jl test/physics/*.jl; do
        h=$L/$(echo "$f" | tr / _).jl
        { echo "cd(\"$R/$(dirname "$f")\")"; grep -E '^(using|import|include)' "$f" | sed "s#@__DIR__#\"$R/$(dirname "$f")\"#g"; } > "$h"
        echo "$h"
    done | xargs -P "$JOBS" -n 1 sh -c 'julia --startup-file=no --project="$PROJ" "$0" > "$0.log" 2>&1; echo "$(basename "$0" .jl) $?"' | sort > "$L/status.txt"
    nf=$(awk '$2 != 0' "$L/status.txt" | wc -l | tr -d ' ')
    record "load ($(wc -l < "$L/status.txt" | tr -d ' ') scripts)" load "$([ "$nf" = 0 ] && echo pass || echo fail)" 0
    awk '$2 != 0 {print "    does not load: " $1}' "$L/status.txt"
fi

if [ "$MODE" = regress ] || [ "$MODE" = all ]; then
    W=$OUT/ref;  git worktree add -q --detach "$W" "$REF" || exit 1
    echo "regression against $REF ($(git -C "$W" rev-parse --short HEAD)):"
    refstep=$W/test/regress/step.jl;  [ -f "$refstep" ] || refstep=$W/test/testregress_step.jl
    drv() { [ -f "$1/examples/equilibrium.jl" ] && echo "$1/examples/equilibrium.jl" || echo "$1/examples/stochrep_equilibrium.jl"; }
    for H in energy smoothed; do
        t0=$(date +%s)
        ( cd "$W" && SPH_HAMILTONIAN=$H julia -t 1 --project="$PROJ" "$refstep" save "$OUT/ref_$H.jls" > "$OUT/ref_step_$H.log" 2>&1 )
        ok=pass
        for th in 1 4; do
            SPH_HAMILTONIAN=$H julia -t $th --project="$PROJ" test/regress/step.jl compare "$OUT/ref_$H.jls" > "$OUT/step_${H}_t$th.log" 2>&1 || ok=fail
        done
        record "step.jl, $H" regress $ok $(( $(date +%s) - t0 ))
    done
    for c in rest boost; do
        t0=$(date +%s);  b=""; [ $c = boost ] && b=0.6
        for side in ref here; do
            d=$R; [ $side = ref ] && d=$W
            ( cd "$d" && SR_BOOST=${b:-0} SR_OUTDIR=$OUT/drv_$side/$c julia -t 2 --project="$PROJ" "$(drv "$d")" 0.6 2.0 0.5 1 2 $c > "$OUT/drv_${side}_$c.log" 2>&1 ) &
        done; wait
        ok=fail
        julia --project="$PROJ" test/regress/h5cmp.jl "$OUT/drv_ref/$c/equilibrium_${c}_r1.h5" "$OUT/drv_here/$c/equilibrium_${c}_r1.h5" > "$OUT/drv_cmp_$c.log" 2>&1 && ok=pass
        record "equilibrium driver, $c" regress $ok $(( $(date +%s) - t0 ))
    done
    sdrv() { [ -f "$1/examples/stability.jl" ] && echo "$1/examples/stability.jl" || echo "$1/examples/stochrep_hamiltonian.jl"; }
    for H in smoothed energy; do
        t0=$(date +%s)
        for side in ref here; do
            d=$R; [ $side = ref ] && d=$W
            ( cd "$d" && SPH_HAMILTONIAN=$H SR_OUTDIR=$OUT/stab_$side/$H julia -t 2 --project="$PROJ" "$(sdrv "$d")" 2.0 0.3 1 full star > "$OUT/stab_${side}_$H.log" 2>&1 ) &
        done; wait
        ok=fail;  f=$(ls "$OUT/stab_ref/$H" 2>/dev/null | head -1)
        [ -n "$f" ] && julia --project="$PROJ" test/regress/jlscmp.jl "$OUT/stab_ref/$H/$f" "$OUT/stab_here/$H/$f" > "$OUT/stab_cmp_$H.log" 2>&1 && ok=pass
        record "stability driver, $H" regress $ok $(( $(date +%s) - t0 ))
    done
    t0=$(date +%s)
    [ -f "$OUT/unit/unit.txt" ] || run_unit "$R" "$OUT/unit"
    run_unit "$W" "$OUT/unit_ref"
    ok=pass;  ls "$OUT"/unit_ref/*.log > /dev/null 2>&1 || ok=fail
    for f in "$OUT"/unit_ref/*.log; do
        [ -f "$f" ] || continue
        n=$(basename "$f" | sed -E -f test/regress/logmask.sed)
        if [ -f "$OUT/unit/$n" ]; then
            diff <(mask "$f") <(mask "$OUT/unit/$n") > "$OUT/unit_diff_${n%.log}.txt" || { ok=fail; echo "    output differs: ${n%.log} (unit_diff_${n%.log}.txt)"; }
        else echo "    only in $REF: ${n%.log}"; fi
    done
    record "unit-test output" regress $ok $(( $(date +%s) - t0 ))
    git worktree remove --force "$W"
fi

echo "results: $RES"
[ $FAILED = 0 ] && echo "PASS" || { echo "FAIL"; exit 1; }
