# h5cmp.jl a.h5 b.h5: every dataset bit-identical (attributes and metadata strings ignored)
using HDF5
function walk(g, p = "", out = Dict{String,Any}())
    for k in keys(g)
        o = g[k];  q = p * "/" * k
        o isa HDF5.Group ? walk(o, q, out) : (out[q] = read(o))
    end
    out
end
a = h5open(walk, ARGS[1]);  b = h5open(walk, ARGS[2])
skip(k) = occursin(r"wall|commit|date|host|meta"i, k)
bad = [k for k in union(keys(a), keys(b)) if !skip(k) && (!haskey(a, k) || !haskey(b, k) || !isequal(a[k], b[k]))]
println(length(keys(a)), " datasets; ", isempty(bad) ? "IDENTICAL" : "DIFFER: " * join(bad, ", "))
isempty(bad) || exit(1)
