"""sk_regimes.py — the Gaussian structure factor of the positions, eq. (Sk) of docs/equilibrium.tex, for sigma = pV/T >> 1
(Wendland C2, C4 or C6, 2D), with the asymptotic forms of its regimes.

    python3 examples/sk_regimes.py [C2|C4|C6]     # writes docs/figs/sk_regimes.png (C2) or sk_regimes_C4.png, ...
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import j0
from scipy.integrate import simpson

KER = sys.argv[1] if len(sys.argv) > 1 else "C2"
q = np.linspace(0, 2, 40001)
W = {"C2": 7 / (4 * np.pi) * (1 - q / 2) ** 4 * (1 + 2 * q),
     "C4": 9 / (4 * np.pi) * (1 - q / 2) ** 6 * (1 + 3 * q + 35 * q ** 2 / 12),
     "C6": 78 / (28 * np.pi) * (1 - q / 2) ** 8 * (1 + 4 * q + 25 * q ** 2 / 4 + 4 * q ** 3)}[KER]
kh = np.geomspace(1e-3, 60, 2500)
WT = simpson(2 * np.pi * q * W * j0(np.outer(kh, q)), x=q, axis=1)
mu2 = simpson(2 * np.pi * q ** 3 * W, x=q)                     # 5/9 for C2
khalf = np.interp(-0.5, -WT, kh)                                # W~ = 1/2: the minimum of S
Sk = lambda s: 1 / (1 + 3 * WT * (1 - WT / 2) + 2 * s * WT * (1 - WT))

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
fig, ax = plt.subplots(figsize=(11, 7.5))
sig = [1, 10, 100, 1000, 10000]
cols = ["#9aa7b8", "#1f5fa8", "#7a4fa0", "#c0462a", "#2b8a3e"]
for s, c in zip(sig, cols):
    ax.loglog(kh, Sk(s), color=c, lw=2.5, label=rf"$\sigma=10^{{{int(np.log10(s))}}}$")
# asymptotes for sigma = 1000
s = 1000; c = "0.15"
ax.loglog(kh, np.full_like(kh, 0.4), ":", color=c, lw=1.3)
m = (kh > 0.03) & (kh < 1.5); ax.loglog(kh[m], 2 / (s * mu2 * kh[m] ** 2), "--", color=c, lw=1.3)
m = (kh > 2.5) & (1 / (2 * s * WT) < 0.6); ax.loglog(kh[m], 1 / (2 * s * WT[m]), "--", color=c, lw=1.3)
ax.axhline(1, color="0.75", lw=0.8)
# k_*: where S climbs back through 1/2 above the minimum, 2 sigma W~(k_*) = 1 (open circles)
m = kh > 1; ax.loglog(kh[m], np.full(m.sum(), 0.5), "-", color="0.75", lw=0.8)
for s_, c_ in zip(sig, cols):
    S_ = Sk(s_); ks = kh[(kh > khalf) & (S_ > 0.5)][0]
    ax.plot([ks], [0.5], "o", ms=9, mfc="white", mec=c_, mew=2, zorder=5)
ax.text(20, 0.13, r"$\circ$ $k_*$: $S=1/2$" + "\n" + r"$2\sigma\widetilde W(k_*)=1$", fontsize=12, color="0.25", ha="left")
xi = np.sqrt(s * mu2 / 5); ax.axvline(1 / xi, color="0.6", lw=0.8, ls="-.")
ax.text(1 / xi * 1.08, 1.4, r"$k\xi=1$" + "\n" + r"($\sigma=10^3$)", fontsize=12, color="0.35")
ax.text(1.3e-3, 0.46, r"$S=2/5$", fontsize=13, color=c)
ax.text(0.09, 4e-3, r"$\dfrac{2}{\sigma\mu_2k^2}$", fontsize=15, color=c)
ka = kh[m][np.argmin(np.abs(np.log(1 / (2 * s * WT[m])) - np.log(0.03)))]       # where the tail asymptote is 0.03
ax.annotate(r"$\dfrac{1}{2\sigma\widetilde W}$", xy=(ka, 0.03), xytext=(22, 4e-3), fontsize=15, color=c, ha="center",
            arrowprops=dict(arrowstyle="->", color=c, lw=1))
ax.text(0.2, 1.25e-4, rf"minimum $\approx 2/\sigma$ at $kh={khalf:.2f}$", fontsize=12, color=c)
ax.set(xlabel="$kh$", ylabel="$S(k)$", xlim=(1e-3, 60), ylim=(1e-4, 3))
ax.legend(frameon=False, loc="lower left", title=f"Wendland {KER}", title_fontsize=14)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
out = "docs/figs/sk_regimes.png" if KER == "C2" else f"docs/figs/sk_regimes_{KER}.png"
fig.savefig(out, dpi=160); print(f"wrote {out}; mu2/h^2 = {mu2:.4f}")
for s in sig: print(f"sigma {s:6g}: xi/h {np.sqrt(s*mu2/5):6.2f}  min S {Sk(s).min():.3e} (2/sigma {2/s:.1e}) at kh {kh[Sk(s).argmin()]:.2f}  S=1/2 again at kh {kh[(kh>3)&(Sk(s)>0.5)][0]:.1f}")
