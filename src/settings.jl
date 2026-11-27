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

# Retired settings: a hard stop at load time, so that a script or campaign file that still sets one fails at once instead of being
# silently ignored.  The campaign files that used them are kept as the record of what was run (nersc/README.md, "Retired settings").
const RETIRED_SETTINGS = Dict(
    "SR_S0" => "renamed SR_SREL0 on 2026-11-23: the rest-frame entropy density s_rel,0 of the run (docs/inputs.tex)",
    "SPH_SSCHEME_BETA" => "renamed SPH_SSCHEME_BRHO on 2026-11-26: b_ρ, the exponent of the measure of the labels (β is u^μ/T in the notes; docs/sscheme_plan.tex, P4b)",
    "SPH_MEASURE" => "removed on 2026-11-27 (docs/focus_plan.tex, Q2): the Liouville measure is the only one; SPH_MEASURE=einstein is in the tag pre-prune",
    "SPH_DISSIPATION" => "removed on 2026-11-27 (docs/focus_plan.tex, Q5): the inverse-moment step (moment) is the only viscous step; the others are in the tag pre-prune",
    "SR_LAMBDA" => "removed on 2026-11-27 (docs/focus_plan.tex, Q1): the residual move always has λ = 1; λ ≠ 1 is in the tag pre-prune",
)
for (name, why) in RETIRED_SETTINGS
    haskey(ENV, name) && error("setting $name is retired: $why.  Unset it (to rerun an old campaign line, replace $name= by the new name).")
end

# The raw string of a setting, or nothing, not recorded: only to choose how a setting is then read with `setting`
# (sph.jl: SPH_RENORM=split or sscheme is read as a Symbol, 0 as a Bool, as before).
setting_peek(name::AbstractString) = get(ENV, name, nothing)

# every setting read so far, in the order first read: name => (value, default, set)
settings_read() = lock(() -> [n => SETTINGS[n] for n in SETTING_ORDER], SETTINGS_LOCK)
