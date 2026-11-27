# settings.jl — the inputs of a run: the table of every setting, and `setting`, through which every one is read.
#
#   setting(name)           the value of the environment variable `name`, parsed to the type of its default in the table
#   setting(name, default)  the same, for a setting whose default depends on the driver (varies in the table); for any other the
#                           default given must be the table's
#
# Strict (docs/focus_plan.tex, Q12): at load, any SR_* or SPH_* in the environment that is not in the table stops the run, with the
# nearest valid name; a retired name stops with its own message.  A Bool must be 1/true/yes or 0/false/no (any case), a Symbol one of
# its allowed values, a number a number: anything else stops.  Each call notes the name, the value used, the default and whether it was
# set; the run record (runinfo.jl) writes them all, and [unused] lists the names set in the environment but never read (record only).
# `julia src/settings.jl --list` prints the table.  Loaded first by load.jl.

const SETTINGS = Dict{String,NamedTuple{(:value, :default, :set),Tuple{Any,Any,Bool}}}()
const SETTING_ORDER = String[]

# The table.  type: the Julia type the value is parsed to; default: the value when unset (nothing: it depends on the driver or on
# other settings, said in `meaning`, and the call gives it); allowed: the values of a Symbol; infra: set by the batch scripts or the
# tests, read directly from ENV (never through `setting`), not listed as unused.
spec(type, default, meaning; allowed = nothing, infra = false) = (; type, default, meaning, allowed, infra)
const SETTING_SPECS = Dict{String,Any}(
    # the scheme (src/)
    "SPH_RENORM"        => spec(Symbol, nothing, "the renormalization: split (default), sscheme, or 0 (the tree level); read by sph.jl"),
    "SPH_SSCHEME_FORM"  => spec(Symbol, :measure, "the form of the S-scheme: measure (default) or exact"; allowed = (:measure, :exact)),
    "SPH_SSCHEME_BRHO"  => spec(Float64, nothing, "b_ρ, the exponent of the measure of the labels of the S-scheme (default D/2 + 1)"),
    "SPH_RENORM_VBAR"   => spec(Float64, 0.0, "V̄, the reference volume of split (0: the mean volume L²/N of the run)"),
    "SPH_POSITIONAL"    => spec(Int, 1, "the positional counterterm ω: 1 on (default; Wendland C2 only), 0 off"),
    "SPH_POSITIONAL_A2" => spec(Float64, 0.0, "the coefficient a_2 of the positional counterterm"),
    "SPH_KERNEL"        => spec(String, "wc2", "the kernel: wc2 (Wendland C2, default), wc4, wc6"; allowed = ("wc2", "wc4", "wc6")),
    "SPH_HASTINGS"      => spec(Bool, nothing, "the Hastings correction of the stochastic step (default on; off in hastings_baseline.jl)"),
    "SPH_GAMMA_MAX"     => spec(Float64, 100.0, "the largest Lorentz factor a particle may have"),
    "SPH_GLF_TOL"       => spec(Float64, 1e-10, "the tolerance of the implicit half steps of the generalized leapfrog"),
    "SPH_ZETA_FLOOR"    => spec(Float64, 0.01, "the floor of ζ_eff in units of η: ζ_eff = max(ζ + 2η(1/2 − 1/d), floor·η)"),
    # the run (examples/: equilibrium.jl and the other drivers)
    "SR_SREL0"          => spec(Float64, 1e6, "s_rel,0, the rest-frame entropy density of the run"),
    "SR_BOOST"          => spec(Float64, 0.0, "the speed v of the fluid along x"),
    "SR_ETAS"           => spec(Float64, 0.2, "η/s"),
    "SR_ZETAS"          => spec(Float64, 0.0, "ζ/s"),
    "SR_CS2"            => spec(Float64, nothing, "c_s² of the constant-c_s² EOS (equilibrium.jl: 0, the conformal fluid of d = 2; eos_measure.jl: 0.5)"),
    "SR_H"              => spec(Float64, 0.05, "the kernel scale h"),
    "SR_START"          => spec(Symbol, nothing, "the start: cold, hot, hmc (equilibrium.jl: default hmc; stability.jl: hot)"; allowed = (:cold, :hot, :hmc)),
    "SR_HMC_N"          => spec(Int, 200, "the number of hybrid Monte Carlo updates of the hmc start"),
    "SR_HMC_TAU"        => spec(Float64, 0.0, "the length of an HMC trajectory (0: L/c_s)"),
    "SR_HMC_DTFAC"      => spec(Float64, nothing, "the HMC step / the acoustic step (0.35; hmc_positions.jl 0.5)"),
    "SR_HMC_TAUJIT"     => spec(Float64, nothing, "the random fraction of an HMC trajectory length (0.5; hmc_positions.jl 0)"),
    "SR_HMC_SHIFT_TOTALS" => spec(Bool, false, "shift the hmc start to the totals of the cold start (debugging)"),
    "SR_JITTER"         => spec(Float64, 0.0, "the jitter of the lattice sites, in units of a"),
    "SR_DTFAC"          => spec(Float64, 1.0, "Δt / the default time step"),
    "SR_KHMAX"          => spec(Float64, nothing, "the largest kh recorded (equilibrium.jl, hmc_positions.jl 1.5; eos_measure.jl, volume_fluctuations.jl 3)"),
    "SR_RECORD_E"       => spec(Bool, false, "record the energy modes E_k (always in motion)"),
    "SR_NOVISC"         => spec(Bool, false, "test: no stochastic step (ideal SPH only)"),
    "SR_NOHAM"          => spec(Bool, false, "test: no ideal step (positions frozen)"),
    "SR_HFIX"           => spec(Bool, false, "stability.jl: rescale the s_a after each stochastic step to restore H"),
    "SR_SEED0"          => spec(Int, nothing, "the seed offset of the realizations (equilibrium.jl, stability.jl 0; the HMC drivers 1); set by nersc/array.sh"),
    "SR_SEEDSTRIDE"     => spec(Int, 1, "seeds SEED0·STRIDE + 1 …: NREAL when an array task runs NREAL seeds"),
    "SR_SEEDOFFSET"     => spec(Int, 0, "added to every seed (new realizations of a set)"),
    "SR_V"              => spec(Float64, 0.0, "eos_measure.jl: the speed of the bath"),
    "SR_BURN"           => spec(Float64, 1 / 3, "eos_measure.jl: the fraction of the updates discarded"),
    "SR_OUTDIR"         => spec(String, nothing, "the output directory (each driver its own default)"),
    "SR_CHECKPOINT_MIN" => spec(Float64, 20.0, "minutes of wall time between checkpoints"),
    "SR_STOP_AFTER"     => spec(Int, 0, "test: stop after this step, after a checkpoint"),
    "SR_SMOKE"          => spec(Bool, false, "a few short cases (moment_det.jl)"),
    # infrastructure: the batch scripts and the tests (read from ENV directly)
    "SR_REPO"           => spec(String, nothing, "nersc/: the checkout a campaign runs"; infra = true),
    "SR_THREADS"        => spec(String, nothing, "nersc/node.sh: Julia threads per task"; infra = true),
    "SR_OUTROOT"        => spec(String, nothing, "nersc/node.sh: the output root"; infra = true),
    "SR_JULIA_MODULE"   => spec(String, nothing, "nersc/: the Julia module"; infra = true),
    "SR_PLOTPROJECT"    => spec(String, nothing, "nersc/: the Julia project of the plots"; infra = true),
    "SR_TESTPROJECT"    => spec(String, nothing, "test/run.sh: the Julia project of the tests"; infra = true),
    "SR_TESTOUT"        => spec(String, nothing, "test/physics: the output directory"; infra = true),
    "SR_TEST_QUICK"     => spec(String, nothing, "tests: shorter runs"; infra = true),
    "SR_TMP"            => spec(String, nothing, "tests: the temporary directory"; infra = true),
    "SR_NOCACHE"        => spec(String, nothing, "test/regress/step.jl: no StochCache"; infra = true),
    "SR_DATADIR"        => spec(String, nothing, "analysis scripts: the data directory"; infra = true),
    "SR_FIGDIR"         => spec(String, nothing, "analysis scripts: the figure directory"; infra = true),
)

parse_bool(name, s) = (t = lowercase(strip(s));  t in ("1", "true", "yes") ? true : t in ("0", "false", "no") ? false :
    error("setting $name=$s: a Bool must be 1, true, yes or 0, false, no"))
parse_setting(::Type{Float64}, s) = parse(Float64, s)
parse_setting(::Type{Int}, s)     = parse(Int, s)
parse_setting(::Type{Symbol}, s)  = Symbol(s)
parse_setting(::Type{<:AbstractString}, s) = String(s)

const SETTINGS_LOCK = ReentrantLock()     # setting may be called from threads (SimParams in a run)

function setting(name::AbstractString, default)
    sp = get(SETTING_SPECS, name, nothing)
    sp === nothing && error("setting $name is not in the table SETTING_SPECS (src/settings.jl)")
    sp.default === nothing || isequal(sp.default, default) ||
        error("setting $name: the default $default given differs from the table's $(sp.default) (src/settings.jl)")
    set = haskey(ENV, name)
    value = !set ? default : default isa Bool ? parse_bool(name, ENV[name]) :
            try parse_setting(typeof(default), ENV[name]) catch err
                err isa ArgumentError ? error("setting $name=$(ENV[name]): not a $(typeof(default))") : rethrow() end
    sp.allowed === nothing || value in sp.allowed ||
        error("setting $name=$value: the values are $(join(sp.allowed, ", "))")
    lock(SETTINGS_LOCK) do
        haskey(SETTINGS, name) || push!(SETTING_ORDER, name)
        SETTINGS[name] = (value = value, default = default, set = set)
    end
    return value
end
function setting(name::AbstractString)
    sp = get(SETTING_SPECS, name, nothing)
    sp === nothing && error("setting $name is not in the table SETTING_SPECS (src/settings.jl)")
    sp.default === nothing && error("setting $name: its default depends on the driver; give it, setting(name, default)")
    return setting(name, sp.default)
end

# Retired settings: a hard stop at load time, so that a script or campaign file that still sets one fails at once instead of being
# silently ignored.  The campaign files that used them are kept as the record of what was run (nersc/README.md, "Retired settings").
const RETIRED_SETTINGS = Dict(
    "SR_S0" => "renamed SR_SREL0 on 2026-11-23: the rest-frame entropy density s_rel,0 of the run (docs/inputs.tex)",
    "SPH_SSCHEME_BETA" => "renamed SPH_SSCHEME_BRHO on 2026-11-26: b_ρ, the exponent of the measure of the labels (β is u^μ/T in the notes; docs/sscheme_plan.tex, P4b)",
    "SPH_MEASURE" => "removed on 2026-11-27 (docs/focus_plan.tex, Q2): the Liouville measure is the only one; SPH_MEASURE=einstein is in the tag pre-prune",
    "SPH_DISSIPATION" => "removed on 2026-11-27 (docs/focus_plan.tex, Q5): the inverse-moment step (moment) is the only viscous step; the others are in the tag pre-prune",
    "SPH_IDEAL" => "removed on 2026-11-27 (docs/focus_plan.tex, Q6): the generalized leapfrog is the only ideal step; SPH_IDEAL=particle and grid are in the tag pre-prune",
    "SR_GRID" => "the grid step was removed on 2026-11-27 (docs/focus_plan.tex, Q6; tag pre-prune)",
    "SPH_HAMILTONIAN" => "removed on 2026-11-27 (docs/focus_plan.tex, Q7): the n-scheme (energy Hamiltonian, number volume) is the only one; SPH_HAMILTONIAN=smoothed is in the tag pre-prune",
    "SPH_VOLUME" => "removed on 2026-11-27 (docs/focus_plan.tex, Q7): the number volume V_a = 1/n_a is the only one; SPH_VOLUME=entropy is in the tag pre-prune",
    "SR_POSITIONS" => "removed on 2026-11-27 (docs/focus_plan.tex, Q9): SR_START = cold | hot | hmc says where and how a run starts (lattice = cold or hot); poisson is in the tag pre-prune",
    "SR_BATH" => "removed on 2026-11-27 (docs/focus_plan.tex, Q9): the start is always at the requested state (the former nominal); totals is in the tag pre-prune",
    "SR_LAMBDA" => "removed on 2026-11-27 (docs/focus_plan.tex, Q1): the residual move always has λ = 1; λ ≠ 1 is in the tag pre-prune",
)
for (name, why) in RETIRED_SETTINGS
    haskey(ENV, name) && error("setting $name is retired: $why.  Unset it (to rerun an old campaign line, replace $name= by the new name).")
end

# Unknown names stop (the nearest valid name in the message): a misspelled setting must not be silently ignored.
function levenshtein(a::AbstractString, b::AbstractString)
    a, b = collect(a), collect(b);  d = collect(0:length(b))
    for i in 1:length(a)
        prev = d[1];  d[1] = i
        for j in 1:length(b)
            cur = d[j + 1];  d[j + 1] = min(d[j + 1] + 1, d[j] + 1, prev + (a[i] == b[j] ? 0 : 1));  prev = cur
        end
    end
    return d[end]
end
nearest_setting(name) = argmin(n -> levenshtein(name, n), collect(keys(SETTING_SPECS)))
for name in sort(collect(keys(ENV)))
    (startswith(name, "SR_") || startswith(name, "SPH_")) || continue
    haskey(SETTING_SPECS, name) || error("setting $name is not a setting of this code (src/settings.jl, SETTING_SPECS): did you mean " *
                                         "$(nearest_setting(name))?  `julia src/settings.jl --list` lists them.")
end

# The raw string of a setting, or nothing, not recorded: only to choose how a setting is then read with `setting`
# (sph.jl: SPH_RENORM=split or sscheme is read as a Symbol, 0 as a Bool, as before).
setting_peek(name::AbstractString) = get(ENV, name, nothing)

# every setting read so far, in the order first read: name => (value, default, set)
settings_read() = lock(() -> [n => SETTINGS[n] for n in SETTING_ORDER], SETTINGS_LOCK)

# the settings set in the environment but never read (not the infrastructure): name => value.  The run record's [unused].
unused_settings() = lock(SETTINGS_LOCK) do
    Dict{String,String}(n => ENV[n] for n in keys(ENV) if haskey(SETTING_SPECS, n) && !SETTING_SPECS[n].infra && !haskey(SETTINGS, n))
end

# julia src/settings.jl --list: the table
if abspath(PROGRAM_FILE) == (@__FILE__) && "--list" in ARGS
    for n in sort(collect(keys(SETTING_SPECS)))
        sp = SETTING_SPECS[n]
        d = sp.default === nothing ? "(see meaning)" : repr(sp.default)
        println(rpad(n, 21), rpad(string(sp.type), 8), rpad(d, 16), sp.meaning, sp.allowed === nothing ? "" : "  [$(join(sp.allowed, " | "))]")
    end
end
