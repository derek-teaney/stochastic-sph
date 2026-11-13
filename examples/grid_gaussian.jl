# grid_gaussian.jl — the Gaussian pulse (examples/testgaussian.jl: A = 1 on a floor 0.1,
# σx = 0.1, σy = 0.14, h = 0.015, h/a = 2, t = 2.2) with the grid step (docs/gridstep.tex, test 5),
# against the particle step.  Three runs: grid + deterministic viscosity η/s = 0.002 (viscous ½ ·
# grid · viscous ½), particles (generic_step!, the same viscosity), and the grid without viscosity.
# Saves slices of the rest-frame energy density through the centre along x and y, the site
# fields, and ΔH/H, ΔS/S, to <out>/grid_gaussian.jls.  Plots: examples/grid_gaussian_plot.jl.
#   julia -t 32 --project=nersc examples/grid_gaussian.jl [out]
using StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const OUT = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "grid_gaussian");  mkpath(OUT)
const SMOKE = setting("SR_SMOKE", false)
const Hs = SMOKE ? 0.05 : 0.015;  const L = 1.0;  const TEND = SMOKE ? 0.1 : 2.2;  const SNAP = (0.0, 0.55, 1.1, 2.2)

function init(etas)
    eos = ConformalEOS(1/3, T_scale_gluon, etas, 0.0)
    e_ic(x, y) = 0.1 + sum(exp(-(x - L/2 - mx * L)^2 / (2 * 0.1^2) - (y - L/2 - my * L)^2 / (2 * 0.14^2)) for mx in -1:1, my in -1:1)
    params, ps, cl, eos = setup_ic(Hs, 2.0, eos, (x, y) -> srel(eos, e_ic(x, y)), (x, y) -> SVector(0.0, 0.0); L)
    return params, ps, cl, eos
end

function slices(ps, params, cl)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos_of[])
    xs, ex = plot_slice(ps, params, cl; field = p -> p.e, dim = :x)
    ys, ey = plot_slice(ps, params, cl; field = p -> p.e, dim = :y)
    return (xs = xs, ex = ex, ys = ys, ey = ey, x = [p.x for p in ps], e = [p.e for p in ps],
            v = [p.v_over_S * p.S_smooth for p in ps])
end
const eos_of = Ref{Any}(nothing)

function run(scheme, etas)
    params, ps, cl, eos = init(etas);  eos_of[] = eos
    st = GridStencil(ps, params, cl);  sc = GridScratch(st.N);  g = GridStats()
    for a in eachindex(ps);  sc.er[a] = ps[a].e;  end
    dotm = zeros(SVector{2,Float64}, params.N)
    dt = compute_dt(ps, params, eos; mode = :viscous);  n = ceil(Int, TEND / dt);  dt = TEND / n
    scheme === :grid && print_grid_checks(grid_checks(st, params, eos, dt))
    H0 = sum(p.s * p.H_over_S for p in ps);  S0 = sum(p.s for p in ps)
    snaps = Dict(0.0 => slices(ps, params, cl));  Ht = [0.0];  St = [0.0];  ts = [0.0];  fails = 0
    for i in 1:n
        if scheme === :grid
            etas > 0 && viscous_step!(ps, params, cl, dt / 2, eos)
            fails += grid_step!(ps, st, dt, eos, sc, g);  fails > 0 && (println("  grid step failed: ", g.first_fail);  break)
            etas > 0 && viscous_step!(ps, params, cl, dt / 2, eos)
        else
            generic_step!(ps, params, cl, dt, eos, dotm)
        end
        t = i * dt
        if i % 10 == 0 || i == n
            smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
            push!(ts, t);  push!(Ht, sum(p.s * p.H_over_S for p in ps) / H0 - 1);  push!(St, sum(p.s for p in ps) / S0 - 1)
        end
        for ts_ in SNAP;  (ts_ > 0 && i == round(Int, ts_ / dt)) && (snaps[ts_] = slices(ps, params, cl));  end   # nearest step
    end
    @printf("%-8s η/s = %.3f: %d steps, ΔH/H = %+.2e  ΔS/S = %+.2e  fails %d  (grid: ΣΔS_T/S %+.2e, max|v|Δt/a %.3f)\n",
            scheme, etas, n, Ht[end], St[end], fails, g.sum_dS_T, g.disp_max)
    return (scheme = string(scheme), etas = etas, dt = dt, steps = n, ts = ts, Ht = Ht, St = St, snaps = snaps,
            grid = grid_summary(g), h = Hs, a = params.a, L = params.L)
end

res = [run(:grid, 0.002), run(:particle, 0.002), run(:grid, 0.0)]
serialize(joinpath(OUT, "grid_gaussian.jls"), res)
println("written to $OUT")
