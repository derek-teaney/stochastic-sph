# smoothing_response.jl — what the SPH smoothing sees of a momentum pattern.
#
# Put a plane-wave pattern on the particles,  m_b = m0 cos(k·x_b).
# The code only ever uses the smoothed field M(x_a) = Σ_b m_b W(x_a − x_b).
# Converted back to a per-particle momentum, V M(x_a) (V = particle volume),
# the pattern comes back as
#
#     V M(x_a) = R(k) · m0 cos(k·x_a),     R(k) = V Σ_b W(x_a − x_b) cos(k·(x_b − x_a))
#
# so R(k) is the fraction of the pattern the smoothed field keeps.
# R = 1 at k = 0 (uniform flow fully seen); R ≈ 0 means H, ẋ, β, the strain and
# the Metropolis test cannot see the pattern at all.
#
# For comparison: the continuum kernel transform  W̃(k) = ∫ d²r W(r) e^{ik·r}
# (what R would be with infinitely many particles per h).
#
# Run:  julia --project examples/smoothing_response.jl   → smoothing_response.png

using PyPlot
using StaticArrays
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

eos = ConformalEOS(0.5, T_scale_2d_lq1, 0.0, 0.0)
params, particles, cl, eos = setup_ic(15.0, 2.55, eos, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                      L=150.0, lattice=:hex, d=2)
h, a, V, L = params.h, params.a, params.vol, params.L

# neighbour offsets of one particle (all particles are equivalent on the lattice)
p1 = particles[1]
offsets = SVector{2,Float64}[];  weights = Float64[]
for pb in particles
    r = minimum_image(pb.x - p1.x, L)
    norm(r) < 2h && (push!(offsets, r); push!(weights, wendland_c2(norm(r), h, params.kernel)))
end
R(k) = V * sum(w * cos(dot(k, r)) for (w, r) in zip(weights, offsets))

# continuum transform of the 2D kernel: W̃(k) = 2π ∫ r W(r) J0(kr) dr,
# J0(x) = (1/π) ∫_0^π cos(x sin θ) dθ   (simple quadrature, no extra packages)
J0(x) = (θ = range(0, π, length=201); sum(cos.(x .* sin.(θ))) / length(θ))
rr = range(0, 2h, length=400);  dr = step(rr)
Wt(k) = 2π * sum(r * wendland_c2(r, h, params.kernel) * J0(k * r) for r in rr) * dr

@printf("lattice: a = %.2f, h = %.1f (h/a = %.2f), %d neighbours within 2h\n",
        a, h, h / a, length(offsets))
@printf("R(0) = %.6f  (= V·Σ W, the uniform-flow check)\n", R(SVector(0.0, 0.0)))

# reciprocal lattice of the hex lattice (rows along x, spacing a): |G| = 4π/(√3 a)
G = 4π / (sqrt(3) * a)

# ── plots ───────────────────────────────────────────────────────────────────
kmax = 1.4
fig, (ax1, ax2) = subplots(1, 2, figsize=(13, 5.5))

kg = range(-kmax, kmax, length=241)
Rmap = [R(SVector(kx, ky)) for ky in kg, kx in kg]
im = ax1.imshow(Rmap, extent=(-kmax, kmax, -kmax, kmax), origin="lower",
                cmap="RdBu_r", vmin=-1, vmax=1)
colorbar(im, ax=ax1, label="R(k)")
ax1.contour(kg, kg, Rmap, levels=[0.1], colors="k", linewidths=0.8)
θ = range(0, 2π, length=200)
ax1.plot(cos.(θ) ./ h, sin.(θ) ./ h, "k:", lw=1)
ax1.text(0.02, 1 / h + 0.03, "|k| = 1/h", fontsize=8)
for n in 0:5   # the 6 shortest reciprocal lattice vectors
    φ = π / 6 + n * π / 3     # hex rows along x ⇒ G points at 30°, 90°, ...
    ax1.plot(G * cos(φ), G * sin(φ), "k^", ms=6)
end
ax1.set_xlabel("kₓ"); ax1.set_ylabel("k_y"); ax1.set_aspect("equal")
ax1.set_title("R(k): fraction of a pattern cos(k·x) the smoothing keeps\n(black line: R = 0.1;  ▲: reciprocal lattice vectors, R = 1 again)",
              fontsize=10)

k1 = range(0, kmax, length=400)
ax2.plot(k1, [R(SVector(k, 0.0)) for k in k1], "C0", lw=1.5, label="lattice R(k), k along x")
ax2.plot(k1, [R(SVector(0.0, k)) for k in k1], "C2", lw=1.5, label="lattice R(k), k along y")
ax2.plot(k1, [Wt(k) for k in k1], "k--", lw=1.2, label="continuum kernel W̃(k)")
ax2.axhline(0, color="0.5", lw=0.5)
ax2.axvline(1 / h, color="0.5", ls=":", lw=1);  ax2.text(1 / h, 1.02, " 1/h", color="0.4")
ax2.axvline(π / a, color="0.5", ls="--", lw=1); ax2.text(π / a, 1.02, " π/a", color="0.4")
ax2.axvline(G, color="0.5", ls="-.", lw=1);     ax2.text(G, 1.02, " |G|", color="0.4")
ax2.set_xlabel("|k|"); ax2.set_ylabel("R(k)"); ax2.set_ylim(-0.1, 1.1)
ax2.set_title("cuts along kₓ and k_y", fontsize=10)
ax2.legend(fontsize=9)

tight_layout()
out = joinpath(@__DIR__, "smoothing_response.png")
savefig(out, dpi=120)
println("Saved ", out)
