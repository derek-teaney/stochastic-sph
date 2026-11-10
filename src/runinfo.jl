# runinfo.jl — what a run actually ran, written next to its data.
#
# run_settings(params, eos; kw...) collects every setting that defines a run, the defaults
# included: the SimParams fields (Hamiltonian, volume, dissipation, kernel, h, a, N, box),
# the EOS, any driver settings passed as keywords, the environment variables SPH_* and SR_*,
# and the provenance (commit, Julia version, threads, host, Slurm job and task, date).
# write_runinfo(base, settings) writes
#   <base>.json   one JSON object, keys in order
#   <base>.csv    a header line and one row (the same keys), for spreadsheets
# Call it at the start (outcomes empty) and again at the end with the outcomes merged in.
# nersc/collect_runs.jl gathers the .csv rows of a campaign into one runs.csv.
# No packages needed.  Include after sph.jl.

function run_settings(params::SimParams, eos; kw...)
    p = Pair{String,Any}[]
    push!(p, "hamiltonian" => string(params.hamiltonian), "volume" => string(params.volume),
          "dissipation" => string(params.dissipation), "kernel" => string(nameof(typeof(params.kernel))),
          "lattice" => string(params.lattice), "Lx" => params.L[1], "Ly" => params.L[2], "h" => params.h,
          "a" => params.a, "h_over_a" => params.h / params.a, "N" => params.N, "vol" => params.vol,
          "bond_h" => params.bond_h, "d" => params.d)
    if eos isa ConformalEOS
        push!(p, "eos" => "conformal", "cs2" => eos.cs2_val, "T_scale" => eos.T_scale,
              "eta_over_s" => eos.η_by_s, "zeta_over_s" => eos.ζ_by_s)
    else
        push!(p, "eos" => string(nameof(typeof(eos))))
    end
    for (k, v) in kw;  push!(p, string(k) => v);  end
    for k in sort(collect(keys(ENV)))
        (startswith(k, "SPH_") || startswith(k, "SR_")) && push!(p, "env_" * k => ENV[k])
    end
    commit = try
        c = readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
        success(`git -C $(@__DIR__) diff --quiet HEAD`) ? c : c * "+dirty"
    catch
        "unknown"
    end
    push!(p, "commit" => commit, "julia" => string(VERSION), "threads" => Threads.nthreads(),
          "host" => gethostname(), "slurm_job" => get(ENV, "SLURM_ARRAY_JOB_ID", get(ENV, "SLURM_JOB_ID", "")),
          "slurm_task" => get(ENV, "SLURM_ARRAY_TASK_ID", ""), "date" => Libc.strftime("%Y-%m-%dT%H:%M:%S", time()))
    return p
end

# flatten: a NamedTuple value becomes key_field entries
function flatten_settings(p)
    out = Pair{String,Any}[]
    for (k, v) in p
        if v isa NamedTuple
            for (k2, v2) in pairs(v);  push!(out, "$(k)_$(k2)" => v2);  end
        else
            push!(out, k => v)
        end
    end
    out
end

jsonval(v::AbstractString) = "\"" * escape_string(v) * "\""
jsonval(v::Symbol) = jsonval(string(v))
jsonval(v::Bool) = v ? "true" : "false"
jsonval(v::Integer) = string(v)
jsonval(v::Real) = isfinite(v) ? repr(Float64(v)) : "null"
jsonval(v::Nothing) = "null"
jsonval(v::AbstractVector) = "[" * join(jsonval.(v), ", ") * "]"
jsonval(v) = jsonval(string(v))

csvval(v::AbstractString) = occursin(r"[,\"\n]", v) ? "\"" * replace(v, "\"" => "\"\"") * "\"" : v
csvval(v::AbstractVector) = csvval(join(string.(v), " "))
csvval(v::Real) = v isa Bool ? string(v) : (isfinite(v) ? string(v) : "")
csvval(v::Nothing) = ""
csvval(v) = csvval(string(v))

function write_runinfo(base::AbstractString, settings)
    p = flatten_settings(settings)
    open(base * ".json.tmp", "w") do io
        println(io, "{")
        for (i, (k, v)) in enumerate(p)
            println(io, "  ", jsonval(k), ": ", jsonval(v), i < length(p) ? "," : "")
        end
        println(io, "}")
    end
    mv(base * ".json.tmp", base * ".json"; force = true)
    open(base * ".csv.tmp", "w") do io
        println(io, join(csvval.(first.(p)), ","));  println(io, join(csvval.(last.(p)), ","))
    end
    mv(base * ".csv.tmp", base * ".csv"; force = true)
    return base
end
