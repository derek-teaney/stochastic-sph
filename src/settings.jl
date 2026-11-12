# settings.jl — every setting read from the environment goes through `setting`.
#
#   setting(name, default)  the value of the environment variable `name`, parsed to the type
#                           of `default` (Float64, Int, Bool, Symbol, String), else `default`
#
# Each call notes the name, the value used, the default and whether it was set; the run record
# (runinfo.jl) writes them all.  Bool: "1", "true" or "yes" is true.  Loaded first by load.jl.

const SETTINGS = Dict{String,NamedTuple{(:value, :default, :set),Tuple{Any,Any,Bool}}}()
const SETTING_ORDER = String[]

parse_setting(::Type{Float64}, s) = parse(Float64, s)
parse_setting(::Type{Int}, s)     = parse(Int, s)
parse_setting(::Type{Bool}, s)    = lowercase(strip(s)) in ("1", "true", "yes")
parse_setting(::Type{Symbol}, s)  = Symbol(s)
parse_setting(::Type{<:AbstractString}, s) = String(s)

const SETTINGS_LOCK = ReentrantLock()     # setting may be called from threads (SimParams in a run)

function setting(name::AbstractString, default)
    set = haskey(ENV, name)
    value = set ? parse_setting(typeof(default), ENV[name]) : default
    lock(SETTINGS_LOCK) do
        haskey(SETTINGS, name) || push!(SETTING_ORDER, name)
        SETTINGS[name] = (value = value, default = default, set = set)
    end
    return value
end

# every setting read so far, in the order first read: name => (value, default, set)
settings_read() = lock(() -> [n => SETTINGS[n] for n in SETTING_ORDER], SETTINGS_LOCK)
