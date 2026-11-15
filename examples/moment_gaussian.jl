# moment_gaussian.jl — the Gaussian energy blob (examples/testgaussian.jl) in the full deterministic step, for the viscous schemes
# common and moment side by side (docs/inverse_moment.tex, "Stage 1").  A strongly nonlinear expansion: e = 0.1 + exp(−x²/2σx² −
# y²/2σy²) at rest, σx = 0.1, σy = 0.14, L = 1 with periodic images; the particles move and rearrange, so the stars are not lattice
# stars.  Recorded at several times: slices of the smoothed energy density along x (y = L/2) and along y (x = L/2), the total
# entropy Σ s_a and the energy H.
#   julia -t 16 --project examples/moment_gaussian.jl   → <out>/moment_gaussian.jls, docs/figs/moment_gaussian.png (SR_FIGDIR)
# Environment: MG_H (0.02), MG_ETAS ("0.02,0.2"), MG_TEND (2.2), MG_OUT (data/moment_gaussian)
using StaticArrays, LinearAlgebra, Printf, Serialization
include(joinpath(@__DIR__, "plotsetup.jl"))
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const H    = parse(Float64, get(ENV, "MG_H", "0.02"))
const ETAS = parse.(Float64, split(get(ENV, "MG_ETAS", "0.02,0.2"), ","))
const TEND = parse(Float64, get(ENV, "MG_TEND", "2.2"))
const OUT  = get(ENV, "MG_OUT", joinpath(@__DIR__, "..", "data", "moment_gaussian"));  mkpath(OUT)
const TSNAP = [0.0, 0.5, 1.0, 2.2]
const L = 1.0;  const SX = 0.1;  const SY = 0.14;  const FLOOR = 0.1

function blob(etas)
    eos = ConformalEOS(1/3, T_scale_gluon, etas, 0.0)
    e_ic(x, y) = FLOOR + sum(exp(-((x - L/2 - mx * L)^2) / (2SX^2) - ((y - L/2 - my * L)^2) / (2SY^2)) for mx in -1:1, my in -1:1)
    params, ps, cl, eos = redirect_stdout(devnull) do
        setup_ic(H, 2.0, eos, (x, y) -> srel(eos, e_ic(x, y)), (x, y) -> SVector(0.0, 0.0); L)
    end
    return params, ps, cl, eos
end

slices(ps, params, cl) = (plot_slice(ps, params, cl; field = p -> p.e, dim = :x), plot_slice(ps, params, cl; field = p -> p.e, dim = :y))

function run(diss, etas)
    ENV["SPH_DISSIPATION"] = string(diss)
    params, ps, cl, eos = blob(etas)
    dotm = zeros(SVector{2,Float64}, params.N)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    Hof() = sum(particle_energy(p, eos) for p in ps);  H0 = Hof();  S0 = sum(p.s for p in ps)
    snaps = Dict{Float64,Any}(0.0 => slices(ps, params, cl))
    ts = [0.0];  Ss = [0.0];  Hs = [0.0];  t = 0.0;  nstep = 0;  t0 = time()
    for tt in TSNAP[2:end]
        while t < tt - 1e-12
            dt = min(compute_dt(ps, params, eos; mode = :viscous, CFL = 0.2), tt - t)
            generic_step!(ps, params, cl, dt, eos, dotm);  t += dt;  nstep += 1
            if nstep % 10 == 0
                smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
                push!(ts, t);  push!(Ss, sum(p.s for p in ps) / S0 - 1);  push!(Hs, Hof() / H0 - 1)
            end
        end
        smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
        snaps[tt] = slices(ps, params, cl)
    end
    @printf("%-7s η/s = %.3g: N = %d, %d steps, %.0f s, ΔS/S = %.4e, ΔH/H = %+.2e, max |v| = %.3f\n", diss, etas, params.N, nstep,
            time() - t0, Ss[end], Hs[end], maximum(norm(p.v_over_S * p.S_smooth) for p in ps))
    return (snaps = snaps, t = ts, dS = Ss, dH = Hs, N = params.N)
end

res = Dict((d, e) => run(d, e) for e in ETAS for d in (:common, :moment))
serialize(joinpath(OUT, "moment_gaussian.jls"), res)

# ── figure: per η/s, the x-slice and the y-slice at the snapshot times, common | moment | moment − common ──
COL = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a"]           # t = 0 in ink, then the categorical order
fig, axs = subplots(2 * length(ETAS), 3, figsize = (14, 3.2 * 2 * length(ETAS)), squeeze = false)
for (i, e) in enumerate(ETAS), (k, dim) in enumerate(("x (y = L/2)", "y (x = L/2)"))
    row = 2(i - 1) + k
    for (ti, tt) in enumerate(TSNAP)
        xc, fc = res[(:common, e)].snaps[tt][k];  xm, fm = res[(:moment, e)].snaps[tt][k]
        axs[row, 1].plot(xc, fc, color = COL[ti], lw = 2, label = "t = $tt")
        axs[row, 2].plot(xm, fm, color = COL[ti], lw = 2, label = "t = $tt")
        tt > 0 && axs[row, 3].plot(xm, fm .- fc, color = COL[ti], lw = 2, label = "t = $tt")
    end
    for j in 1:3
        axs[row, j].grid(true, color = "#e6e5e0", lw = 0.5);  axs[row, j].set_xlabel(dim[1:1])
        axs[row, j].legend(fontsize = 7, frameon = false)
    end
    lo = min(axs[row, 1].get_ylim()[1], axs[row, 2].get_ylim()[1]);  hi = max(axs[row, 1].get_ylim()[2], axs[row, 2].get_ylim()[2])
    axs[row, 1].set_ylim(lo, hi);  axs[row, 2].set_ylim(lo, hi)
    axs[row, 1].set_ylabel("energy density e")
    axs[row, 1].set_title("common, η/s = $e: slice along $dim", fontsize = 9)
    axs[row, 2].set_title("moment, η/s = $e: slice along $dim", fontsize = 9)
    axs[row, 3].set_title("moment − common", fontsize = 9)
end
tight_layout();  out = figpath("moment_gaussian.png");  savefig(out, dpi = 110);  close();  println("written ", out)

fig, axs = subplots(1, 2, figsize = (11, 3.6))
for (i, e) in enumerate(ETAS), (j, d) in enumerate((:common, :moment))
    r = res[(d, e)];  st = j == 1 ? "-" : "--"
    axs[1].plot(r.t, r.dS, st, color = COL[i+1], lw = 2, label = "$d, η/s = $e")
    axs[2].plot(r.t, r.dH, st, color = COL[i+1], lw = 2, label = "$d, η/s = $e")
end
axs[1].set_ylabel("Σ s_a / Σ s_a(0) − 1");  axs[2].set_ylabel("H / H(0) − 1")
for ax in axs;  ax.set_xlabel("t");  ax.grid(true, color = "#e6e5e0", lw = 0.5);  ax.legend(fontsize = 7, frameon = false);  end
tight_layout();  out = figpath("moment_gaussian_SH.png");  savefig(out, dpi = 110);  close();  println("written ", out)
