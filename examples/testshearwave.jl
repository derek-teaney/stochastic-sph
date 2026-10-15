using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────
h  = 0.05
ha = 2.5       # h/a ≈ 2.5 → a ≈ 0.02, N ≈ 2900 (hex)
L  = 1.0
cs2v   = 1.0/3.0
η_by_s = 0.0

# ── Test 1: 2D Gaussian entropy blob ─────────────────────────────────────────
σ  = 0.12   # σ/h ≈ 2.4: well above one smoothing length
e0 = 1.0
A  = 0.8

params_g, particles_g, cl_g, eos_g =
    setup_gaussian(h, ha; L, e0, A, σx=σ, σy=σ, cs2v, η_by_s)

analytic_g = (x, y) -> srel(eos_g, e0 + A * exp(-((x - L/2)^2 + (y - L/2)^2) / (2*σ^2)))

xs_fine = range(0.0, L, length=500)

fig1, axes1 = subplots(1, 3, figsize=(14, 4.5))
fig1.suptitle("Test 1: 2D Gaussian blob  (h=$(h), h/a=$(ha), σ=$(σ))")

plot_particles(particles_g, params_g;
               field=p->p.S_smooth, clabel="s density", title="scatter", ax=axes1[1])

xs, fs = plot_slice(particles_g, params_g, cl_g; field=p->p.S_smooth, dim=:x)
axes1[2].plot(xs, fs, color="C0", label="S(x) sph")
axes1[2].plot(xs_fine, analytic_g.(xs_fine, L/2), "--", label="input s_func", linewidth=1.5)
axes1[2].set_xlabel("x"); axes1[2].set_ylabel("S(x)")
axes1[2].set_title("x-slice at y=$(round(L/2,digits=2))")
axes1[2].legend(fontsize=8)

ys, fs = plot_slice(particles_g, params_g, cl_g; field=p->p.S_smooth, dim=:y)
axes1[3].plot(ys, fs, color="C0", label="S(y) sph")
axes1[3].plot(xs_fine, analytic_g.(L/2, xs_fine), "--", label="input s_func", linewidth=1.5)
axes1[3].set_xlabel("y"); axes1[3].set_ylabel("S(y)")
axes1[3].set_title("y-slice at x=$(round(L/2,digits=2))")
axes1[3].legend(fontsize=8)

tight_layout()
savefig(joinpath(@__DIR__, "testshearwave_gaussian2d.png"), dpi=150)
println("Saved testshearwave_gaussian2d.png")

# ── Test 2: 1D Gaussian stripe  (σx finite, σy = Inf) ────────────────────────
params_s, particles_s, cl_s, eos_s =
    setup_gaussian(h, ha; L, e0, A, σx=σ, σy=Inf, cs2v, η_by_s)

analytic_stripe = x -> srel(eos_s, e0 + A * exp(-(x - L/2)^2 / (2*σ^2)))

fig2, axes2 = subplots(1, 3, figsize=(14, 4.5))
fig2.suptitle("Test 2: 1D Gaussian stripe  σx=$(σ), σy=Inf")

plot_particles(particles_s, params_s;
               field=p->p.S_smooth, clabel="s density", title="scatter", ax=axes2[1])

xs, fs = plot_slice(particles_s, params_s, cl_s; field=p->p.S_smooth, dim=:x)
axes2[2].plot(xs, fs, color="C0", label="S(x) sph")
axes2[2].plot(xs_fine, analytic_stripe.(xs_fine), "--", label="input s_func", linewidth=1.5)
axes2[2].set_xlabel("x"); axes2[2].set_ylabel("S(x)"); axes2[2].set_title("x-slice  (varies)")
axes2[2].legend(fontsize=8)

ys, fs = plot_slice(particles_s, params_s, cl_s; field=p->p.S_smooth, dim=:y)
axes2[3].plot(ys, fs, color="C0", label="S(y) sph")
axes2[3].axhline(s0 + A, linestyle="--", label="peak s0+A", linewidth=1.5)
axes2[3].set_xlabel("y"); axes2[3].set_ylabel("S(y)"); axes2[3].set_title("y-slice  (should be flat)")
axes2[3].legend(fontsize=8)

tight_layout()
savefig(joinpath(@__DIR__, "testshearwave_gaussian1d.png"), dpi=150)
println("Saved testshearwave_gaussian1d.png")

# ── Test 3: Shear wave ────────────────────────────────────────────────────────
M_amp = 0.02
s0_sw = 1.5

params_w, particles_w, cl_w, eos_w, k, _ =
    setup_shear_wave(h, ha; L, M_amp, s0=s0_sw, cs2v, η_by_s)

My_density_analytic = x -> M_amp * sin(k * x)

fig3, axes3 = subplots(1, 3, figsize=(14, 4.5))
fig3.suptitle("Test 3: Shear wave  M_amp=$(M_amp), k=4π/L")

plot_particles(particles_w, params_w;
               field=p->p.M_smooth[2], clabel="M_y density", title="scatter", ax=axes3[1])

xs, fs = plot_slice(particles_w, params_w, cl_w; field=p->p.M_smooth[2], dim=:x)
axes3[2].plot(xs, fs, color="C0", label="M_y(x) sph")
axes3[2].plot(xs_fine, My_density_analytic.(xs_fine), "--", label="M_amp sin(kx)", linewidth=1.5)
axes3[2].set_xlabel("x"); 
axes3[2].set_ylabel("M_y density");
axes3[2].set_title("M_y(x) slice")
axes3[2].legend(fontsize=8)

xs, fs = plot_slice(particles_w, params_w, cl_w; field=p->p.S_smooth, dim=:x)
axes3[3].plot(xs, fs, color="C0", label="S(x) sph")
axes3[3].axhline(s0_sw, linestyle="--", label="s0", linewidth=1.5)
axes3[3].set_xlabel("x"); 
axes3[3].set_ylabel("S(x)"); 
axes3[3].set_title("S(x) slice  (should be flat)")
axes3[3].legend(fontsize=8)

tight_layout()
savefig(joinpath(@__DIR__, "testshearwave_shearwave.png"), dpi=150)
println("Saved testshearwave_shearwave.png")

# ── Test 4: overlay p.s/vol vs p.S_smooth on the same axis ───────────────────
# p.s/vol  — raw per-particle entropy density evaluated at each particle site,
#             then entropy-weighted by the kernel (biased toward high-s regions)
# p.S_smooth — the SPH kernel sum Σ_b s_b W(r_ab,h), already a smoothed estimate
#              of the entropy density; we entropy-weight it again here via plot_slice
# Both should approximate s(x) but through different moments of the kernel.

fig4, ax4 = subplots(1, 1, figsize=(7, 4.5))

# Check that the particle labels are correct 
xs, fs = plot_slice(particles_g, params_g, cl_g; field=p->p.s/params_g.vol, dim=:x)
ax4.plot(xs, fs, color="C0", label="p.s / vol")

# Check that the smooth labels are correct
xs, fs = plot_slice(particles_g, params_g, cl_g; field=p->p.S_smooth, dim=:x)
ax4.plot(xs, fs, color="C1", linestyle="--", label="S(x)")
ax4.plot(xs_fine, analytic_g.(xs_fine, L/2), ":", color="k", linewidth=2, label="analytic")
ax4.set_xlabel("x"); ax4.set_ylabel("entropy density")
ax4.set_title("Test 4: p.s/vol vs p.S_smooth  (2D Gaussian x-slice)")
ax4.legend(fontsize=9)

tight_layout()
savefig(joinpath(@__DIR__, "testshearwave_compare.png"), dpi=150)
println("Saved testshearwave_compare.png")

# ── Test 5: overlay p.M[2]/vol vs p.M_smooth[2] on the shear wave ────────────
fig5, ax5 = subplots(1, 1, figsize=(7, 4.5))

# Check that the particle labels are correct
xs, fs = plot_slice(particles_w, params_w, cl_w; field=p->p.m[2]/params_w.vol, dim=:x)
ax5.plot(xs, fs, color="C0", label="p.m[2] / vol")

# Check that the smooth density is correct
xs, fs = plot_slice(particles_w, params_w, cl_w; field=p->p.M_smooth[2], dim=:x)
ax5.plot(xs, fs, color="C1", linestyle="--", label="M_smooth[2]")
ax5.plot(xs_fine, My_density_analytic.(xs_fine), ":", color="k", linewidth=2,
         label="M_amp sin(kx)")
ax5.set_xlabel("x"); ax5.set_ylabel("M_y density")
ax5.set_title("Test 5: p.m[2]/vol vs p.M_smooth[2]  (shear wave x-slice)")
ax5.legend(fontsize=9)

tight_layout()
savefig(joinpath(@__DIR__, "testshearwave_Mcompare.png"), dpi=150)
println("Saved testshearwave_Mcompare.png")

show()
