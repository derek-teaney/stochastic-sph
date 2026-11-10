# shearwave_decay.jl — a transverse (shear) wave in full deterministic hydro,
# generic_step! = D(Δt/2) H(Δt) D(Δt/2): ideal Hamiltonian step plus the viscous step.
#
# Initial state: uniform entropy density s0, momentum density M_y = A sin(kx), k = 2π n/L.
# Linear theory: the wave does not propagate; it diffuses,
#     A(t) = A(0) e^{−ν k² t},   ν = η/w = (η/s)/T_rel ,
# and with η = 0 it stays put.  Measured: the projection
#     A(t) ∝ Σ_a m_{y,a} sin(k x_a),
# the leak into the other quadrature (Σ m_y cos kx) and into M_x, the energy
# H = Σ_a s_a ℰ_a, and the total entropy Σ s_a.
#
# Both Hamiltonians are run (params.hamiltonian = :smoothed, sph2.tex; :energy,
# energystep.tex), each with η/s = SW_ETA and with η = 0 (ideal step only).
#
# Run:  julia -t 4 --project examples/shearwave_decay.jl   → docs/figs/shearwave_decay.png
# Env:  SW_H (0.05)  SW_HA (2.0)  SW_L (1.0)  SW_N (1, wave number in units of 2π/L)
#       SW_ETA (0.02)  SW_AMP (0.02)  SW_TEND (decay times, 2.0)  SW_CFL (0.2)
#       SW_OUT (figure name in docs/figs, or in SR_FIGDIR if set)
#       SW_HAMS (smoothed,energy,energy_n; energy_n = energy with SPH_VOLUME=number, volume_n.tex)
include(joinpath(@__DIR__, "plotsetup.jl"))      # PyPlot, Agg in batch jobs
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

envf(k, d) = parse(Float64, get(ENV, k, string(d)))
const H_   = envf("SW_H", 0.05)
const HA   = envf("SW_HA", 2.0)
const L    = envf("SW_L", 1.0)
const NW   = round(Int, envf("SW_N", 1))
const ETA  = envf("SW_ETA", 0.02)
const AMP  = envf("SW_AMP", 0.02)
const TEND = envf("SW_TEND", 2.0)
const CFL  = envf("SW_CFL", 0.2)
const S0   = 2.0

function run_shear(ham::Symbol, η_by_s::Float64)
    ENV["SPH_HAMILTONIAN"] = ham === :energy_n ? "energy" : string(ham)
    ENV["SPH_VOLUME"] = ham === :energy_n ? "number" : "entropy"
    eos = ConformalEOS(1/3, 1.0, η_by_s, 0.0)
    params, ps, cl, eos = setup_ic(H_, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L)
    k = 2π * NW / params.L[1]
    for pa in ps                                   # m_a = M(x_a) vol, as set_ic! does
        pa.m = SVector(0.0, AMP * sin(k * pa.x[1]) * params.vol)
    end
    thermalize!(ps, params, cl, eos)
    Trel0 = sum(p.T * p.γ for p in ps) / length(ps)
    ν     = η_by_s / Trel0
    ν_ref = ETA / Trel0                            # sets the run length for both η
    tend  = TEND / (ν_ref * k^2)
    dt    = compute_dt(ps, params, eos; mode = η_by_s > 0 ? :viscous : :ideal, CFL)
    nstep = ceil(Int, tend / dt);  dt = tend / nstep
    dotm  = zeros(SVector{2,Float64}, length(ps))

    proj(ps) = (sum(p.m[2] * sin(k * p.x[1]) for p in ps),
                sum(p.m[2] * cos(k * p.x[1]) for p in ps),
                maximum(abs(p.m[1]) for p in ps))
    energy(ps) = sum(p.s * p.H_over_S for p in ps)   # needs smooth + EOS (done by the steps)

    a0, _, _ = proj(ps);  H0 = energy(ps);  S0tot = sum(p.s for p in ps)
    t = [0.0];  A = [1.0];  B = [0.0];  Mx = [0.0];  dH = [0.0];  dS = [0.0]
    every = max(1, nstep ÷ 200)
    for n in 1:nstep
        generic_step!(ps, params, cl, dt, eos, dotm)
        if n % every == 0 || n == nstep
            smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
            a, b, mx = proj(ps);  Hn = energy(ps)
            push!(t, n * dt);  push!(A, a / a0);  push!(B, b / a0);  push!(Mx, mx / (AMP * params.vol))
            push!(dH, (Hn - H0) / H0);  push!(dS, (sum(p.s for p in ps) - S0tot) / S0tot)
            isfinite(Hn) || (println("  non-finite energy at t = $(n*dt): stopped"); break)
        end
    end
    @printf("%-9s η/s=%-5.3g  N=%d  kh=%.3f  ν=%.4g  dt=%.3g  steps=%d  A(end)=%.5f  analytic=%.5f  |B|max=%.2e  Mx/A=%.2e  ΔH/H=%.2e  ΔS/S=%.2e\n",
            ham, η_by_s, params.N, k * H_, ν, dt, nstep, A[end], exp(-ν * k^2 * t[end]),
            maximum(abs, B), maximum(Mx), dH[end], dS[end])
    return (; t, A, B, Mx, dH, dS, ν, k)
end

println("Shear wave: h=$H_ h/a=$HA L=$L n=$NW η/s=$ETA amp=$AMP tend=$TEND decay times")
runs = Dict{Tuple{Symbol,Float64},Any}()
const HAMS = Symbol.(split(get(ENV, "SW_HAMS", "smoothed,energy,energy_n"), ","))
for ham in HAMS, η in (ETA, 0.0)
    runs[(ham, η)] = run_shear(ham, η)
end

fig, ax = subplots(1, 3, figsize = (15, 4.4))
cols = Dict(:smoothed => "C0", :energy => "C3", :energy_n => "C2")
for ham in HAMS, η in (ETA, 0.0)
    r = runs[(ham, η)];  ls = η > 0 ? "-" : ":"
    lab = @sprintf("%s, η/s=%g", ham, η)
    # identical curves must stay visible: wide lines first, thin ones on top, markers for energy_n
    kw = (color = cols[ham], label = lab, lw = Dict(:smoothed => 3.5, :energy => 2.2, :energy_n => 1.0)[ham],
          marker = ham === :energy_n ? "o" : "", markevery = max(1, length(r.t) ÷ 15), ms = 5, mfc = "none")
    ax[1].plot(r.t, r.A, ls; kw...)
    ax[2].plot(r.t, r.dH, ls; kw...)
    ax[3].plot(r.t, r.dS, ls; kw...)
end
r = runs[(HAMS[1], ETA)]
ax[1].plot(r.t, exp.(-r.ν * r.k^2 .* r.t), "k--", lw = 1, label = "e^{−νk²t}")
ax[1].set_xlabel("t");  ax[1].set_ylabel("A(t)/A(0)");  ax[1].set_title("shear-wave amplitude")
ax[2].set_xlabel("t");  ax[2].set_ylabel("ΔH/H");      ax[2].set_title("energy")
ax[3].set_xlabel("t");  ax[3].set_ylabel("ΔS/S");      ax[3].set_title("total entropy Σ s_a")
for a in ax; a.legend(fontsize = 7); end
fig.suptitle(@sprintf("shear wave, h=%g, h/a=%g, kh=%.3f, A=%g", H_, HA, r.k * H_, AMP))
tight_layout()
out = haskey(ENV, "SR_FIGDIR") ? figpath(get(ENV, "SW_OUT", "shearwave_decay.png")) :
      joinpath(@__DIR__, "..", "docs", "figs", get(ENV, "SW_OUT", "shearwave_decay.png"))
savefig(out, dpi = 130)
println("Saved ", out)
