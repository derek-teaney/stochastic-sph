# astcmp.jl <ref> [files...]: the code of each file is unchanged against git <ref> — the parsed expressions, line numbers removed,
# are equal (comments are not in them).  The gate of a stage that changes only comments (docs/focus_plan.tex, Q13).  Without files:
# every .jl file changed since <ref> (git diff --name-only).
strip_lines(x) = x
strip_lines(x::LineNumberNode) = nothing
function strip_lines(x::Expr)
    args = [strip_lines(a) for a in x.args if !(a isa LineNumberNode)]
    return Expr(x.head, args...)
end
ref = ARGS[1]
files = length(ARGS) > 1 ? ARGS[2:end] : filter(f -> endswith(f, ".jl") && isfile(f), split(readchomp(`git diff --name-only $ref`), '\n'))
bad = String[]
for f in files
    old = try read(`git show $ref:$f`, String) catch; println("new file (not compared): $f"); continue end
    a = strip_lines(Meta.parseall(old));  b = strip_lines(Meta.parseall(read(f, String)))
    same = a == b
    println(same ? "  same code  " : "  CHANGED    ", f)
    same || push!(bad, f)
end
isempty(bad) ? println("astcmp: the code is unchanged in $(length(files)) files") : (println("astcmp: code changed in ", join(bad, ", ")); exit(1))
