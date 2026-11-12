# runinfo.jl — the run record: what a run ran and how it went, <base>.toml next to its data
# (docs/codeplan.tex, "The run record").  Generated from the values the code used, never
# typed in:
#
#   [run]         driver, tag, seed, command-line arguments
#   [sim]         every field of SimParams
#   [eos]         every field of the equation of state
#   [stochastic]  every field of StochOptions (absent if the run has no stochastic step)
#   [settings]    every setting read through `setting` (settings.jl): value, default, set
#   [driver]      quantities the driver derives (time step, steps, entropy per particle, ...)
#   [start]       how the initial state was made
#   [provenance]  commit (+dirty and a hash of the diff), Julia, threads, host, Slurm, date
#   [outcome]     progress and results; rewritten at each checkpoint and at the end
#
# write_record(base; run, params, eos, opts, driver, start, outcome) writes it (atomically).
# Python reads it with tomllib; nersc/collect_runs.jl gathers the records of a campaign into
# one table.

using TOML

# a value TOML can hold: numbers, strings, booleans, arrays and tables
tomlval(v::Union{Bool,Integer,AbstractString}) = v
tomlval(v::AbstractFloat) = Float64(v)
tomlval(v::Symbol) = string(v)
tomlval(v::Nothing) = "none"
tomlval(v::Union{AbstractVector,Tuple}) = [tomlval(x) for x in v]
tomlval(v::NamedTuple) = Dict{String,Any}(string(k) => tomlval(x) for (k, x) in pairs(v))
tomlval(v::AbstractDict) = Dict{String,Any}(string(k) => tomlval(x) for (k, x) in v)
tomlval(v) = isstructtype(typeof(v)) && fieldcount(typeof(v)) > 0 ? fields_table(v) : string(nameof(typeof(v)))

# every field of a struct, by name
fields_table(x) = Dict{String,Any}(string(k) => tomlval(getfield(x, k)) for k in fieldnames(typeof(x)))
fields_table(::Nothing) = Dict{String,Any}()

settings_table() = Dict{String,Any}(n => Dict{String,Any}("value" => tomlval(s.value), "default" => tomlval(s.default),
                                                         "set" => s.set) for (n, s) in settings_read())

function provenance()
    dir = @__DIR__
    commit, dirty, diffhash = "unknown", false, ""
    try
        commit = readchomp(`git -C $dir rev-parse --short HEAD`)
        dirty = !success(`git -C $dir diff --quiet HEAD`)
        dirty && (diffhash = string(hash(read(`git -C $dir diff HEAD`)); base = 16))
    catch
    end
    Dict{String,Any}("commit" => commit * (dirty ? "+dirty" : ""), "diff_hash" => diffhash,
                     "julia" => string(VERSION), "threads" => Threads.nthreads(), "host" => gethostname(),
                     "slurm_job" => get(ENV, "SLURM_ARRAY_JOB_ID", get(ENV, "SLURM_JOB_ID", "")),
                     "slurm_task" => get(ENV, "SLURM_ARRAY_TASK_ID", ""),
                     "date" => Libc.strftime("%Y-%m-%dT%H:%M:%S", time()))
end

function write_record(base::AbstractString; run = NamedTuple(), params = nothing, eos = nothing,
                      opts = nothing, driver = NamedTuple(), start = NamedTuple(), outcome = NamedTuple())
    rec = Dict{String,Any}("run" => tomlval(run), "sim" => fields_table(params), "eos" => fields_table(eos),
                           "settings" => settings_table(), "driver" => tomlval(driver),
                           "start" => tomlval(start), "provenance" => provenance(), "outcome" => tomlval(outcome))
    opts === nothing || (rec["stochastic"] = fields_table(opts))
    open(io -> TOML.print(io, rec; sorted = true), base * ".toml.tmp", "w")
    mv(base * ".toml.tmp", base * ".toml"; force = true)
    return base * ".toml"
end
