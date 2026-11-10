# shearwave_scan.jl — the transverse rigidity of the ideal step from a shear wave
# (docs/rigidity.tex, docs/energystep.tex).
#
# Ideal step only (η = 0, leapfrog_step!).  Initial state: hex lattice, uniform entropy
# density, M_y = A sin(kx), k = 2π/Lx, in a thin periodic strip Ly ≈ 6h (the wave depends
# on x only).  A fluid keeps A(t) = A(0); an elastic solid with transverse speed c_t gives
#     A(t)/A(0) = cos(c_t k t)        (c_t² < 0: cosh, unstable to shear).
# Measured for each (kernel, h/a, Hamiltonian):
#   c_t,fit from a least-squares fit of cos(c_t k t), or cosh for c_t < 0 (ω_T² < 0), over
#   the first quarter period,
#   t_0, the first zero of A, and c_t,0 = π / (2 k t_0) when reached,
#   the leak into M_x, the energy drift, and the transverse displacement
#   U(t) = Σ_a (y_a(t) − y_a(0)) sin(k x_a(0)) / Σ_a sin²(k x_a(0)), in units of a
#   (U_max/a ≳ 1 means the particles have slipped by a lattice spacing: not linear).
# SC_AMPS ("0.02") scans the amplitude.
#
# Run:  julia -t 4 --project examples/shearwave_scan.jl   → docs/figs/shearwave_scan.png
# Env:  SC_KERNELS ("C2,C4,C6")  SC_HA ("2,2.5,3,4")  SC_HAMS ("smoothed,energy"; energy_n = energy with SPH_VOLUME=number)
#       SC_H (0.05)  SC_L (1.0)  SC_LY (Ly/h, 6)  SC_AMP (0.002, linear: U_max ≲ 0.2a)  SC_TEND (30)  SC_CFL (0.2)
#       SC_OUT (figure name in docs/figs; the data go to examples/data/<name>.jls)
#       SC_REPLOT=<file.jls>: no runs, only redo the plot from saved data
include(joinpath(@__DIR__, "plotsetup.jl"))      # PyPlot, Agg in batch jobs
using StaticArrays, LinearAlgebra, Printf, Serialization
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

envf(k, d) = parse(Float64, get(ENV, k, string(d)))
envl(k, d) = split(get(ENV, k, d), ",")
KERNELS = Symbol.(envl("SC_KERNELS", "C2,C4,C6"))
HAS     = parse.(Float64, envl("SC_HA", "2,2.5,3,4"))
HAMS    = Symbol.(envl("SC_HAMS", "smoothed,energy"))
const H_   = envf("SC_H", 0.05)
const L    = envf("SC_L", 1.0)
const LY   = envf("SC_LY", 6.0) * H_
AMPS = parse.(Float64, envl("SC_AMPS", get(ENV, "SC_AMP", "0.002")))
const TEND = envf("SC_TEND", 30.0)
const CFL  = envf("SC_CFL", 0.2)
const S0   = 2.0

function run_ideal(kern::Symbol, ha::Float64, ham::Symbol, AMP::Float64)
    ENV["SPH_HAMILTONIAN"] = ham === :energy_n ? "energy" : string(ham)     # energy_n: SPH_VOLUME=number
    ENV["SPH_VOLUME"] = ham === :energy_n ? "number" : "kernel";  ENV["SPH_KERNEL"] = lowercase("w" * string(kern))
    eos = ConformalEOS(1/3, 1.0, 0.0, 0.0)
    params, ps, cl, eos = setup_ic(H_, ha, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0);
                                   Lx = L, Ly = LY)
    k = 2π / params.L[1]
    for pa in ps
        pa.m = SVector(0.0, AMP * sin(k * pa.x[1]) * params.vol)
    end
    thermalize!(ps, params, cl, eos)
    w = sum(p.T * p.S_smooth for p in ps) / length(ps)      # w = T S (bare T, at rest)
    dt = compute_dt(ps, params, eos; mode = :ideal, CFL)
    nstep = ceil(Int, TEND / dt);  dt = TEND / nstep
    dotm = zeros(SVector{2,Float64}, length(ps))
    proj(ps) = sum(p.m[2] * sin(k * p.x[1]) for p in ps)
    energy(ps) = sum(p.s * p.H_over_S for p in ps)
    a0 = proj(ps);  H0 = energy(ps)
    x0 = [p.x for p in ps];  sk = [sin(k * p.x[1]) for p in ps];  sk2 = sum(abs2, sk)
    disp(ps) = sum(minimum_image(ps[i].x - x0[i], params.L)[2] * sk[i] for i in eachindex(ps)) / sk2 / params.a
    t = [0.0];  A = [1.0];  Mx = [0.0];  dH = [0.0];  U = [0.0]
    every = max(1, nstep ÷ 400)
    for n in 1:nstep
        leapfrog_step!(ps, params, cl, dt, eos, dotm)
        if n % every == 0 || n == nstep
            smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
            Hn = energy(ps)
            push!(t, n * dt);  push!(A, proj(ps) / a0)
            push!(Mx, maximum(abs(p.m[1]) for p in ps) / (AMP * params.vol))
            push!(dH, (Hn - H0) / H0);  push!(U, disp(ps))
            (isfinite(Hn) && abs(A[end]) < 50) || (println("  blew up at t = $(n*dt)"); break)
        end
    end
    # c_t: least-squares fit of A(t) to cos(c k t) (c > 0) or cosh(|c| k t) (c < 0: ω_T² < 0)
    # over the first quarter period (up to the first zero of A), or the whole run if A
    # does not reach zero.
    i0 = findfirst(<(0), A)
    t0 = i0 === nothing ? NaN : t[i0 - 1] + (t[i0] - t[i0-1]) * A[i0-1] / (A[i0-1] - A[i0])
    seg = i0 === nothing ? (1:length(t)) : (1:i0-1)
    model(c, τ) = c >= 0 ? cos(c * k * τ) : cosh(-c * k * τ)
    cost(c) = sum((A[i] - model(c, t[i]))^2 for i in seg)
    cs = range(-0.1, 0.1, length = 4001);  ct = cs[argmin(cost.(cs))]
    for δ in (5e-5, 2.5e-6)                     # refine
        cs = range(ct - 20δ, ct + 20δ, length = 81);  ct = cs[argmin(cost.(cs))]
    end
    rms = sqrt(cost(ct) / length(seg))
    ct0 = π / (2k * t0)
    @printf("%-3s h/a=%-4g %-9s amp=%-6g N=%-6d kh=%.3f  c_t,fit=%+.5f (rms %.0e)  t0=%6.2f  c_t,0=%.5f  A(end)=%+.3f  Umax/a=%.3f  Mx/A=%.2e  ΔH/H=%+.1e\n",
            kern, ha, ham, AMP, params.N, k * H_, ct, rms, t0, ct0, A[end], maximum(abs, U), maximum(Mx), dH[end])
    flush(stdout)
    return (; t, A, Mx, dH, U, ct, t0, ct0, N = params.N)
end

const OUTNAME = get(ENV, "SC_OUT", "shearwave_scan.png")
if haskey(ENV, "SC_REPLOT")
    saved = deserialize(ENV["SC_REPLOT"])
    res = saved.res;  KERNELS = saved.KERNELS;  HAS = saved.HAS;  HAMS = saved.HAMS;  AMPS = saved.AMPS
else
    println("shear-wave scan: h=$H_ L=$L Ly=$LY amps=$AMPS tend=$TEND kernels=$KERNELS h/a=$HAS hams=$HAMS")
    res = Dict{Tuple{Symbol,Float64,Symbol,Float64},Any}()
    for kern in KERNELS, ha in HAS, ham in HAMS, amp in AMPS
        res[(kern, ha, ham, amp)] = run_ideal(kern, ha, ham, amp)
    end
    datadir = get(ENV, "SR_FIGDIR", joinpath(@__DIR__, "data"));  mkpath(datadir)
    datafile = joinpath(datadir, splitext(OUTNAME)[1] * ".jls")
    serialize(datafile, (; res, KERNELS, HAS, HAMS, AMPS, H_, L, LY, TEND, CFL, S0))
    println("Saved data ", datafile)
end

fig, axs = subplots(1, length(KERNELS), figsize = (5.2 * length(KERNELS), 4.3), squeeze = false)
for (j, kern) in enumerate(KERNELS)
    ax = axs[1, j]
    for (i, (ha, amp)) in enumerate(Iterators.product(HAS, AMPS)), ham in HAMS
        r = res[(kern, ha, ham, amp)]
        ax.plot(r.t, r.A, ham === :energy ? "--" : ham === :energy_n ? ":" : "-", color = "C$(mod(i-1, 10))",
                label = @sprintf("h/a=%g A=%g %s (c_t=%+.4f)", ha, amp, ham, r.ct))
    end
    ax.axhline(0, color = "k", lw = 0.5)
    ax.set_ylim(-1.1, 1.3);  ax.set_xlabel("t");  ax.set_ylabel("A(t)/A(0)")
    ax.set_title("Wendland $kern, ideal step");  ax.legend(fontsize = 6)
end
tight_layout()
out = haskey(ENV, "SR_FIGDIR") ? figpath(OUTNAME) : joinpath(@__DIR__, "..", "docs", "figs", OUTNAME)
savefig(out, dpi = 130)
println("Saved ", out)
