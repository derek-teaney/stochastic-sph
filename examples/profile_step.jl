# profile_step.jl — where the time and the allocations go in one time step of the
# equilibrium driver (merged stochastic steps: leapfrog L(Δt), then stochastic_step!(Δt)).
#
#   julia -t <n> --project=nersc examples/profile_step.jl L mode [nsteps]
#     mode = time     wall, GC time and bytes allocated per step for the two parts
#     mode = profile  sampling profile (flat, by line in src/) and allocation profile
#                     (by allocating line in src/), single thread recommended
# Output on stdout; the full flat profile goes to $SR_OUTDIR/profile_flat_L<L>.txt.
using StaticArrays, LinearAlgebra, Printf, Random, Profile
const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "src", "load.jl"))
const LBOX  = parse(Float64, ARGS[1])
const MODE  = ARGS[2]
const NST   = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 20
const OUT   = get(ENV, "SR_OUTDIR", ".");  mkpath(OUT)

eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
params, ps, cl, eos = setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L=LBOX, lattice=:hex)
N = params.N;  rngs = make_phys_rng(N, 1);  rng = Xoshiro(2);  dotm = zeros(SVector{2,Float64}, N)
thermal_init!(ps, params, cl, eos; rng = Xoshiro(3))
dt = 0.5 * compute_dt(ps, params, eos; mode=:viscous)
const CACHE = StochCache()
step!(n) = (leapfrog_step!(ps, params, cl, dt, eos, dotm);
            stochastic_step!(ps, params, cl, dt, eos; rng, rngs, vote=:star, flip=isodd(n), cache=CACHE))
for n in 1:5; step!(n); end                                  # compile and settle

let                                                          # hard scope for the loops
if MODE == "time"
    tl = tg = 0.0;  bl = bs = 0;  gl = gs = 0.0;  ts = 0.0
    for n in 1:NST
        a = @timed leapfrog_step!(ps, params, cl, dt, eos, dotm)
        b = @timed stochastic_step!(ps, params, cl, dt, eos; rng, rngs, vote=:star, flip=isodd(n), cache=CACHE)
        tl += a.time;  bl += a.bytes;  gl += a.gctime
        ts += b.time;  bs += b.bytes;  gs += b.gctime
    end
    @printf("L=%.1f N=%d threads=%d  per step (avg of %d):\n", LBOX, N, Threads.nthreads(), NST)
    @printf("  leapfrog        %8.1f ms  GC %5.1f%%  %8.1f MB\n", 1e3tl/NST, 100gl/tl, bl/NST/1e6)
    @printf("  stochastic_step!  %8.1f ms  GC %5.1f%%  %8.1f MB\n", 1e3ts/NST, 100gs/ts, bs/NST/1e6)
    @printf("  total           %8.1f ms  = %.1f μs per particle per step\n", 1e3(tl+ts)/NST, 1e6(tl+ts)/NST/N)
elseif MODE == "profile"
    # sampling profile
    Profile.clear();  Profile.init(n = 10^7, delay = 0.001)
    @profile for n in 1:NST; step!(n); end
    io = IOBuffer();  Profile.print(IOContext(io, :displaysize => (100000, 300)); format = :flat, sortedby = :count, mincount = 5)
    flat = String(take!(io));  write(joinpath(OUT, "profile_flat_L$(LBOX).txt"), flat)
    # self time per line in src/ (leaf frames only)
    data, ldict = Profile.retrieve();  self = Dict{String,Int}();  total = 0
    i = 1
    while i <= length(data)                                  # samples are separated by 0
        j = findnext(==(0), data, i);  j === nothing && break
        if j > i
            total += 1
            for ip in data[i:j-1]                            # innermost frame first
                fr = get(ldict, ip, nothing);  fr === nothing && continue
                f = fr[1];  file = string(f.file)
                if occursin("stochastic-sph", file) || occursin("/src/", file) && !occursin("julia", file)
                    k = @sprintf("%s:%d %s", basename(file), f.line, f.func);  self[k] = get(self, k, 0) + 1;  break
                end
            end
        end
        i = j + 1
        while i <= length(data) && data[i] == 0; i += 1; end
    end
    @printf("L=%.1f N=%d threads=%d: %d samples over %d steps\n", LBOX, N, Threads.nthreads(), total, NST)
    println("innermost repository line on the stack (≈ where the time is spent, incl. calls into Base):")
    for (k, c) in first(sort(collect(self), by = x -> -x[2]), 30)
        @printf("  %5.1f%%  %s\n", 100c / total, k)
    end
    # allocation profile
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = 0.01 for n in 1:min(NST, 5); step!(n); end
    res = Profile.Allocs.fetch();  site = Dict{String,Tuple{Int,Int}}()
    for al in res.allocs
        k = "?"
        for fr in al.stacktrace
            file = string(fr.file)
            if occursin("stochastic-sph", file)
                k = @sprintf("%s:%d %s", basename(file), fr.line, fr.func);  break
            end
        end
        c, b = get(site, k, (0, 0));  site[k] = (c + 1, b + al.size)
    end
    tb = sum(last.(values(site)));  tc = sum(first.(values(site)))
    println("\nallocations by repository line (1% sample of ", min(NST, 5), " steps; ", tc, " sampled, ",
            @sprintf("%.1f", tb / 1e6), " MB sampled):")
    for (k, (c, b)) in first(sort(collect(site), by = x -> -x[2][2]), 25)
        @printf("  %5.1f%% of bytes  %6.1f%% of count  %s\n", 100b / tb, 100c / tc, k)
    end
end
end
