"""wendland_ft.py — the Fourier transforms of the 2D Wendland kernels (src/sph.jl), support 2h, W~(0) = 1.

    python3 examples/wendland_ft.py        # writes docs/figs/wendland_ft.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import j0
from scipy.integrate import simpson

q = np.linspace(0, 2, 40001)                                   # q = r/h, h = 1
K = {"C2": 7 / (4 * np.pi) * (1 - q / 2) ** 4 * (1 + 2 * q),
     "C4": 9 / (4 * np.pi) * (1 - q / 2) ** 6 * (1 + 3 * q + 35 * q ** 2 / 12),
     "C6": 78 / (28 * np.pi) * (1 - q / 2) ** 8 * (1 + 4 * q + 25 * q ** 2 / 4 + 4 * q ** 3)}
col = {"C2": "#1f5fa8", "C4": "#c0462a", "C6": "#2b8a3e"}
kh = np.geomspace(1e-2, 60, 1500)
FT = {n: simpson(2 * np.pi * q * W * j0(np.outer(kh, q)), x=q, axis=1) for n, W in K.items()}
for n in K: print(f"{n}: norm {np.trapezoid(2*np.pi*q*K[n], q):.6f}  min W~ {FT[n].min():+.2e}  W~=1/2 at kh = {np.interp(-0.5, -FT[n], kh):.3f}")

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for n in K:
    ax[0].plot(kh, FT[n], color=col[n], lw=2.5, label=f"Wendland {n}")
    ax[0].plot(kh, FT[n] * (1 - FT[n]), color=col[n], lw=1.2, ls="--")
    ax[1].loglog(kh, np.abs(FT[n]), color=col[n], lw=2.5, label=f"Wendland {n}")
ax[0].set(xlim=(0, 8), ylim=(0, 1.02), xlabel="$kh$", ylabel=r"$\widetilde W(k)$")
ax[0].text(5.0, 0.30, r"$\widetilde W(1-\widetilde W)$ (dashed)", fontsize=13, color="0.3")
ax[0].legend(frameon=False, fontsize=13); ax[0].grid(color="0.92", lw=0.6)
ax[1].set(xlim=(1e-1, 60), ylim=(1e-8, 2), xlabel="$kh$", ylabel=r"$|\widetilde W(k)|$")
ax[1].legend(frameon=False, fontsize=13, loc="lower left"); ax[1].grid(color="0.92", lw=0.6, which="major")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("docs/figs/wendland_ft.png", dpi=160)
print("wrote docs/figs/wendland_ft.png")
