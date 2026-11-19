"""running_integral.py (runs labelled by sigma(T0); dashed: the exact single particle at sigma(T0), examples/statics_theory.py) — where the scales of the positions appear in real space: the running integral of the pair correlation,

    N(R) = n̄ ∫_{r<R} d²r [g(r) - 1] = R ∫_0^∞ dk J1(kR) [S(k) - 1],

the excess number of particles within R of a particle (Gaussian theory of docs/equilibrium.tex, Wendland C2; it depends on σ only,
not on n̄h²), against the HMC runs (g recorded to r = 2h).  N → -1 + S(0) = -3/5 for R >> ξ.  Writes docs/figs/running_integral.png.

    python3 examples/running_integral.py data/volfluct/vf_*.h5
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from statics_theory import S_exact, exact, run_constants
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import j0, j1
from scipy.integrate import simpson

q = np.linspace(0, 2, 20001); W = 7 / (4 * np.pi) * (1 - q / 2) ** 4 * (1 + 2 * q)
K = np.geomspace(1e-4, 300, 60000)
Kc = np.geomspace(1e-4, 300, 3000); Wc = simpson(2 * np.pi * q * W * j0(np.outer(Kc, q)), x=q, axis=1)
WT = np.interp(np.log(K), np.log(Kc), Wc)
Sk = lambda s: 1 / (1 + 3 * WT * (1 - WT / 2) + 2 * s * WT * (1 - WT))
def Sx(s): e = exact(s); return S_exact(WT, e["alpha"], e["sig"])       # improved: exact single particle at sigma(T0)
def NRx(R, s):
    f = Sx(s) - 1
    return np.array([x * simpson(j1(K * x) * f * K, x=np.log(K)) for x in R])
def NR(R, s):
    f = Sk(s) - 1
    return np.array([x * simpson(j1(K * x) * f * K, x=np.log(K)) for x in R])

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
fig, ax = plt.subplots(figsize=(11, 7))
R = np.geomspace(0.05, 300, 400)
# theory at sigma(T0) of the h/a = 2 runs (solid: Gaussian, dashed: exact single particle), and two larger sigma
runs = []
for f in sys.argv[1:]:
    F = h5py.File(f); runs.append((F, run_constants(F.attrs)["sigma"]))
sig2 = sorted(sg for F, sg in runs if int(round(float(F.attrs["ha"]))) == 2)
pal = ["#1f5fa8", "#c0462a", "#2b8a3e", "#7a4fa0", "#e08a1e"]
sig = dict(zip(sig2 + [10 * sig2[-1], 100 * sig2[-1]] if sig2 else [4, 13.5, 135, 1350, 13500], pal))
for s, c in sig.items():
    ax.semilogx(R, NR(R, s), "-", color=c, lw=2.5, label=rf"$\sigma={s:.3g}$")
    xi = np.sqrt(s * 5 / 9 / 5); ax.plot([xi], [np.interp(xi, R, NR(R, s))], "o", color=c, ms=9, mec="white", mew=1.5)
for F, s in runs:
    A = {k: F.attrs[k] for k in F.attrs}; h = float(A["h"]); nbar = float(A["N"]) / float(np.prod(A["L"]))
    b = len(F["t"]) // 3; g = F["part_g"][b:].mean(0); rb = F["part_rbins"][:]
    rc = 0.5 * (rb[1:] + rb[:-1]); dA = np.pi * (rb[1:] ** 2 - rb[:-1] ** 2); NRd = np.cumsum(nbar * (g - 1) * dA)
    c = sig[min(sig, key=lambda x: abs(x - s))]; ha = int(round(float(A["ha"])))
    ax.plot(rb[1:] / h, NRd, {2: "o", 3: "^", 4: "s"}[ha], color=c, ms=4, mfc=c if ha == 2 else "none", alpha=0.8)
ax.axhline(-1, color="0.6", lw=0.8, ls=":"); ax.axhline(-0.6, color="0.6", lw=0.8, ls=":")
ax.text(200, -0.97, "$-1$: one particle", ha="right", fontsize=12, color="0.4")
ax.text(200, -0.57, r"$-1+S(0)=-3/5$", ha="right", fontsize=12, color="0.4")
for s in sig2:                               # exact single particle at sigma(T0) of the h/a = 2 runs, dashed
    ax.semilogx(R, NRx(R, s), "--", color=sig[s], lw=2)
ax.plot([], [], "k--", lw=2, label=r"exact single particle")
ax.plot([], [], "o", color="0.3", ms=9, mec="white", label=r"$R=\xi$")
ax.plot([], [], "ko", ms=4, label="HMC $h/a=2$"); ax.plot([], [], "k^", ms=4, mfc="none", label="HMC $h/a=3$")
ax.plot([], [], "ks", ms=4, mfc="none", label="HMC $h/a=4$")
ax.set(xlabel="$R/h$", ylabel=r"$N(R)=\bar n\int_{r<R}d^2r\,[g(r)-1]$", xlim=(0.05, 300), ylim=(-1.4, 0.02))
ax.legend(frameon=False, fontsize=12, loc="upper right", ncol=2)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
fig.savefig("docs/figs/running_integral.png", dpi=160); print("wrote docs/figs/running_integral.png")
