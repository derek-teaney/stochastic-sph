"""gr_regimes.py — the pair correlation of the Gaussian theory (docs/equilibrium.tex, Wendland C2) for σ = pV/T from 4 to 10^4:
F = n̄h² [g(r) − 1], which depends on r/h and σ only.  Left: the correlation hole (r ≲ 2h).  Right: the halo, σF against r/ξ,
which collapses onto (9/5π) K0(r/ξ) for r >> h  (ξ² = σμ2/5, μ2 = 5h²/9).  Writes docs/figs/gr_regimes.png.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import j0, k0
from scipy.integrate import simpson

q = np.linspace(0, 2, 20001); W = 7 / (4 * np.pi) * (1 - q / 2) ** 4 * (1 + 2 * q)
K = np.geomspace(1e-5, 300, 80000)
Kc = np.geomspace(1e-5, 300, 3000); Wc = simpson(2 * np.pi * q * W * j0(np.outer(Kc, q)), x=q, axis=1)
WT = np.interp(np.log(K), np.log(Kc), Wc)
Sk = lambda s: 1 / (1 + 3 * WT * (1 - WT / 2) + 2 * s * WT * (1 - WT))
F = lambda r, s: np.array([simpson(K * K * j0(K * x) * (Sk(s) - 1), x=np.log(K)) / (2 * np.pi) for x in r])   # n̄h²(g-1)
# the halo (r > 2h): a k-grid fine enough for j0(kr) out to r ~ 200h, with a smooth taper at kh ~ 15
K2 = np.concatenate([np.geomspace(1e-5, 0.1, 4000)[:-1], np.linspace(0.1, 60, 90000)])
W2 = np.interp(np.log(K2), np.log(Kc), Wc)
def Fh(r, s):
    f = (1 / (1 + 3 * W2 * (1 - W2 / 2) + 2 * s * W2 * (1 - W2)) - 1) * np.exp(-(K2 / 15) ** 2)   # smooth taper: alters r < 0.1h only
    return np.array([simpson(K2 * j0(K2 * x) * f, x=K2) / (2 * np.pi) for x in r])

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
sig = {4: "#1f5fa8", 13.5: "#c0462a", 135: "#2b8a3e", 1350: "#7a4fa0", 13500: "#e08a1e"}
r1 = np.linspace(0.005, 2.5, 400)
for s, c in sig.items():
    ax[0].plot(r1, F(r1, s), color=c, lw=2.5, label=rf"$\sigma={s:g}$")
    xi = np.sqrt(s / 9); r2 = np.geomspace(max(2.0, 0.02 * xi), 6 * xi, 200); y = s * Fh(r2, s)
    ax[1].loglog(r2 / xi, np.where(y > 0, y, np.nan), color=c, lw=2.5)
x = np.geomspace(0.02, 6, 200); ax[1].loglog(x, 9 / (5 * np.pi) * k0(x), "k--", lw=1.5, label=r"$\frac{9}{5\pi}K_0(r/\xi)$")
ax[0].axhline(0, color="0.7", lw=0.8)
ax[0].set(xlabel="$r/h$", ylabel=r"$\bar n h^2\,[g(r)-1]$", xlim=(0, 2.5), title="the correlation hole")
ax[0].legend(frameon=False, loc="lower right")
ax[1].set(xlabel=r"$r/\xi$", ylabel=r"$\sigma\,\bar n h^2\,[g(r)-1]$", xlim=(0.02, 6), ylim=(1e-3, 3),
          title=r"the halo, $r>2h$: collapse onto the healing profile")
ax[1].legend(frameon=False, loc="lower left", fontsize=15)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("docs/figs/gr_regimes.png", dpi=160); print("wrote docs/figs/gr_regimes.png")
