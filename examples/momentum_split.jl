# momentum_split.jl — where does the stochastic momentum go?
#
# Particles frozen on the hex lattice, cold start, stochastic sweep only.
# Split each particle's momentum into the part the smoothed field sees and the rest:
#
#     resolved    m_res,a = V_a M(x_a),  V_a = s_a/S_a,  M(x_a) = Σ_b m_b W_ab
#     unresolved  m_a − m_res,a
#
# and take the Fourier spectrum  P(k) = |Σ_a m_a e^{-ik·x_a}|² / N  vs |k|,
# next to the smoothing response  R(k) = V Σ_b W(r_ab) cos(k·r_ab)
# (R = 1 at k = 0; R ≈ 0 means the smoothed field — hence H, ẋ, β, the
#  strain and the Metropolis test — cannot see that wavelength).
#
# Run:  julia --project examples/momentum_split.jl   → writes momentum_split.png

using PyPlot
using StaticArrays
using Random
using LinearAlgebra
using Statistics
using Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "..", "src", "legacy", "load.jl"))   # two_level, particle_sweep (superseded)

# The stochastic sweep implements the noise of the two-level dissipative function
# only, so this script uses :two_level (the default is :bond).
ENV["SPH_DISSIPATION"] = "two_level"
n_steps    = 800
spec_steps = (25, 100, 400, 800)          # when to take a spectrum

eos = ConformalEOS(0.5, T_scale_2d_lq1, 10.0 / sqrt(2.0), 0.0)
params, particles, cl, eos = setup_ic(15.0, 2.55, eos, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                      L=150.0, lattice=:hex, d=2)
dt   = compute_dt(particles, params, eos; mode=:ideal, CFL=0.2)
rngs = make_phys_rng(params.N, 1)
L, h, N = params.L, params.h, params.N

# ── the split (needs smoothed fields current) ───────────────────────────────
resolved(p) = (p.s / p.S_smooth) * p.M_smooth
rms(v) = sqrt(mean(norm(x)^2 for x in v))

# ── Fourier modes of the periodic box, up to |k| = kmax ─────────────────────
kmax  = 1.3 * 2π / params.a
nxmax = ceil(Int, kmax * L[1] / 2π);  nymax = ceil(Int, kmax * L[2] / 2π)
ks    = [SVector(2π * nx / L[1], 2π * ny / L[2]) for nx in -nxmax:nxmax, ny in -nymax:nymax
         if 0 < hypot(2π * nx / L[1], 2π * ny / L[2]) <= kmax]
kmag  = norm.(ks)

function spectrum(particles)
    P = zeros(length(ks))
    for (i, k) in enumerate(ks)
        c = sum(p.m * cis(-dot(k, p.x)) for p in particles)
        P[i] = (abs2(c[1]) + abs2(c[2])) / N
    end
    return P
end

# smoothing response R(k) on this lattice, measured around particle 1
function response(k)
    p1 = particles[1];  acc = 0.0
    for pb in particles
        r = minimum_image(pb.x - p1.x, L);  rr = norm(r)
        rr < 2h && (acc += wendland_c2(rr, h, params.kernel) * cos(dot(k, r)))
    end
    return params.vol * acc
end
R = response.(ks)

# bin by |k|
edges = range(0, kmax, length=41);  kc = (edges[1:end-1] .+ edges[2:end]) ./ 2
binavg(y) = [(s = (kmag .>= edges[i]) .& (kmag .< edges[i+1]); any(s) ? mean(y[s]) : NaN)
             for i in 1:length(kc)]

# ── run ─────────────────────────────────────────────────────────────────────
steps = Int[];  rms_tot = Float64[];  rms_res = Float64[];  rms_unres = Float64[]
spectra = Dict{Int,Vector{Float64}}()
for step in 1:n_steps
    thermalize!(particles, params, cl, eos)
    particle_sweep!(particles, params, cl, eos, dt, rngs)
    apply_sweep!(particles)
    thermalize!(particles, params, cl, eos)              # refresh M_smooth for the split

    mres = [resolved(p) for p in particles]
    push!(steps, step)
    push!(rms_tot,   rms([p.m for p in particles]))
    push!(rms_res,   rms(mres))
    push!(rms_unres, rms([p.m - r for (p, r) in zip(particles, mres)]))
    step in spec_steps && (spectra[step] = spectrum(particles))
    step % 200 == 0 && @printf("step %4d  rms|m| = %6.2f   resolved %6.2f   unresolved %6.2f\n",
                                step, rms_tot[end], rms_res[end], rms_unres[end])
end

# ── plot ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = subplots(1, 2, figsize=(13, 5))

t = steps .* dt
ax1.plot(t, rms_tot,   "k",  lw=1.5, label="total  m")
ax1.plot(t, rms_res,   "C0", lw=1.5, label="resolved  V M(x)  (seen by H, β, strain)")
ax1.plot(t, rms_unres, "C3", lw=1.5, label="unresolved  m − V M(x)")
ax1.set_xlabel("t");  ax1.set_ylabel("rms per particle")
ax1.set_title("split of the particle momentum");  ax1.legend(fontsize=9)

for (j, s) in enumerate(sort(collect(keys(spectra))))
    ax2.semilogy(kc, binavg(spectra[s]), "-", color=get_cmap("viridis")((j - 1) / (length(spectra) - 0.5)),
                 lw=1.5, label="step $s")
end
ax2.set_xlabel("|k|");  ax2.set_ylabel("P(k) = |m_k|² / N")
ax2.set_title("momentum spectrum (frozen lattice, sweep only)")
ax2.axvline(1 / h, color="0.5", ls=":", lw=1);  ax2.text(1 / h, ax2.get_ylim()[2], " 1/h", va="top", color="0.4")
ax2.axvline(π / params.a, color="0.5", ls="--", lw=1)
ax2.text(π / params.a, ax2.get_ylim()[2], " π/a", va="top", color="0.4")
ax2.legend(fontsize=9, loc="lower left")
axR = ax2.twinx()
axR.plot(kc, binavg(R), "C3--", lw=1.2);  axR.set_ylim(-0.1, 1.05)
axR.set_ylabel("smoothing response R(k)", color="C3")
axR.axhline(0, color="C3", lw=0.5)

tight_layout()
out = joinpath(@__DIR__, "momentum_split.png")
savefig(out, dpi=120)
println("Saved ", out)
