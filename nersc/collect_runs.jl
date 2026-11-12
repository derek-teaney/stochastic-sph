# collect_runs.jl — gather the run records (<run>.toml, src/runinfo.jl) of a campaign into one
# spreadsheet, <campaign dir>/runs.csv: one row per run, one column per entry ("sim.h",
# "outcome.KE_over_eq", ...; a setting appears as its value, "settings.SR_ETAS").
#   julia --project=nersc nersc/collect_runs.jl $SCRATCH/sph/<campaign> [more dirs…]
using TOML
function flatten!(row, d, prefix = "")
    for (k, v) in d
        key = prefix * k
        if v isa AbstractDict
            startswith(key, "settings.") && haskey(v, "value") ? (row[key] = string(v["value"])) : flatten!(row, v, key * ".")
        else
            row[key] = v isa AbstractVector ? join(string.(v), " ") : string(v)
        end
    end
    return row
end
q(v) = occursin(r"[,\"\n]", v) ? "\"" * replace(v, "\"" => "\"\"") * "\"" : v
for dir in ARGS
    rows = Dict{String,String}[]
    for (root, _, fs) in walkdir(dir), f in sort(fs)
        endswith(f, ".toml") || continue
        row = flatten!(Dict{String,String}(), TOML.parsefile(joinpath(root, f)))
        row["file"] = relpath(joinpath(root, f), dir);  push!(rows, row)
    end
    cols = vcat("file", sort(collect(setdiff(union(keys.(rows)...), ["file"]))))
    open(joinpath(dir, "runs.csv"), "w") do io
        println(io, join(q.(cols), ","))
        for r in rows;  println(io, join([q(get(r, c, "")) for c in cols], ","));  end
    end
    println("$(length(rows)) runs, $(length(cols)) columns → $(joinpath(dir, "runs.csv"))")
end
