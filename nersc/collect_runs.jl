# collect_runs.jl — gather the per-run rows (<run>.csv, written by src/runinfo.jl) of a
# campaign into one spreadsheet, <campaign dir>/runs.csv, with the union of the columns.
#   julia nersc/collect_runs.jl $SCRATCH/sph/<campaign> [more dirs…]      (no packages needed)
function readrow(f)
    ls = readlines(f);  length(ls) < 2 && return nothing
    split_csv(l) = [m.captures[1] !== nothing ? replace(m.captures[1], "\"\"" => "\"") : m.captures[2]
                    for m in eachmatch(r"(?:^|,)(?:\"((?:[^\"]|\"\")*)\"|([^,]*))", l)]
    Dict(zip(split_csv(ls[1]), split_csv(ls[2]))), split_csv(ls[1])
end
for dir in ARGS
    rows = Dict{String,String}[];  cols = String[]
    for (root, _, fs) in walkdir(dir), f in sort(fs)
        (endswith(f, ".csv") && f != "runs.csv") || continue
        r = readrow(joinpath(root, f));  r === nothing && continue
        d, hdr = r;  d["file"] = relpath(joinpath(root, f), dir);  push!(rows, d)
        for h in hdr;  h in cols || push!(cols, h);  end
    end
    pushfirst!(cols, "file")
    q(v) = occursin(r"[,\"]", v) ? "\"" * replace(v, "\"" => "\"\"") * "\"" : v
    open(joinpath(dir, "runs.csv"), "w") do io
        println(io, join(q.(cols), ","))
        for d in rows;  println(io, join([q(get(d, c, "")) for c in cols], ","));  end
    end
    println("$(length(rows)) runs, $(length(cols)) columns → $(joinpath(dir, "runs.csv"))")
end
