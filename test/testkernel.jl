using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))

h = 1.0
r = LinRange(0.0, 2.5h, 500)
rv = collect(r)

w = [wendland_c2(ri, h) for ri in r]

# gradient along three directions
gx_xaxis = [wendland_c2_grad(SVector(ri, 0.0),         ri, h)[1] for ri in rv]
gy_yaxis = [wendland_c2_grad(SVector(0.0, ri),         ri, h)[2] for ri in rv]
gx_diag  = [wendland_c2_grad(SVector(ri/√2, ri/√2),   ri, h)[1] for ri in rv]
gy_diag  = [wendland_c2_grad(SVector(ri/√2, ri/√2),   ri, h)[2] for ri in rv]

fig, (ax1, ax2) = subplots(1, 2, figsize=(11, 4))

# ── left: kernel ──────────────────────────────────────────────────────────────
ax1.plot(rv, w)
ax1.axvline(2h, color="gray", linestyle="--", label="support 2h")
ax1.set_xlabel("r");  ax1.set_ylabel("W(r, h)")
ax1.set_title("Wendland C2 kernel  (h = $h)")
ax1.legend()

# ── right: gradient components ────────────────────────────────────────────────
ax2.plot(rv, gx_xaxis, label=L"\partial_x W,\; \mathbf{r}=(r,0)")
ax2.plot(rv, gy_yaxis, label=L"\partial_y W,\; \mathbf{r}=(0,r)", linestyle="--")
ax2.plot(rv, gx_diag,  label=L"\partial_x W,\; \mathbf{r}=r(\hat{x}+\hat{y})/\sqrt{2}", linestyle="-.")
ax2.plot(rv, gy_diag,  label=L"\partial_y W,\; \mathbf{r}=r(\hat{x}+\hat{y})/\sqrt{2}", linestyle=":")
ax2.axhline(0.0, color="gray", linewidth=0.8)
ax2.axvline(2h,  color="gray", linestyle="--")
ax2.set_xlabel("r");  ax2.set_ylabel("gradient component")
ax2.set_title("∇W components")
ax2.legend(fontsize=8)

tight_layout()
savefig(joinpath(@__DIR__, "kernel.png"), dpi=150)
show()
