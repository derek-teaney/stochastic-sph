# jlscmp.jl a.jls b.jls: the serialized results identical (wall time, timing and commit ignored)
using Serialization
skipkey(k) = occursin(r"wall|time|commit"i, string(k))
strip(x::NamedTuple) = NamedTuple(k => strip(v) for (k, v) in pairs(x) if !skipkey(k))
strip(x::AbstractDict) = Dict(k => strip(v) for (k, v) in x if !skipkey(k))
strip(x::Tuple) = map(strip, x)
strip(x::AbstractVector) = eltype(x) <: Number ? x : map(strip, x)
strip(x) = x
same = isequal(strip(deserialize(ARGS[1])), strip(deserialize(ARGS[2])))
println(same ? "IDENTICAL" : "DIFFER");  same || exit(1)
