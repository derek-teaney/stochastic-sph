"""momentum_fluct.py (at the bath temperature T0, w = T0 s(T0); the exact single particle gives r_m <V_a>nbar, examples/statics_theory.py) — the momentum density of the HMC runs (examples/volume_fluctuations.jl) against the Gaussian theory of
docs/equilibrium.tex.  At rest the mean momentum of each particle is zero, and at fixed positions the particles are independent
with <m_a,i m_a,j> = delta_ij T (e+p) V_a, so for the point sums M_k = sum_a m_a e^{-ik.x_a} (projected on k or across it)

    <|M_k|^2> / (L^2 T (e+p)) = sum_a V_a / L^2 = <V_a> nbar      (flat in k, the same for both projections)

no leak of the positions, only the total volume (a few per mille below 1).  Plots the transverse and longitudinal projections,
the counterpart of entropy_leak.py.  Writes docs/figs/momentum_fluct.png.

    python3 examples/momentum_fluct.py data/volfluct/vf_*.h5
"""
import sys, warnings; warnings.filterwarnings("ignore")
import os, h5py, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from statics_theory import exact, run_constants, from_label
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
cols = {16: "#1f5fa8", 54: "#c0462a", 541: "#2b8a3e"}; mk = {2: "o", 3: "^", 4: "s"}
print(f"{'h/a':>4} {'s_a':>5} {'<V>nbar':>8}  mean over k of <|M_k|^2>/(L^2 T w): T, L")
for f in sys.argv[1:]:
    F = h5py.File(f); A = {k: F.attrs[k] for k in F.attrs}; sa = float(A["s_a"]); ha = float(A["ha"]); h = float(A["h"])
    L2 = float(np.prod(A["L"])); N = float(A["N"]); rc = run_constants(F.attrs); T = rc["T0"]; w = T * float(A["S0"]); b = len(F["t"]) // 3
    Vn = F["vol_mean"][b:].mean()                                             # <V_a> nbar = sum_a V_a / L^2, measured
    k = np.hypot(*F["ks"][:].T) * h; ku = np.unique(np.round(k, 2)); nb = 8
    c = cols[int(round(sa))]; m = mk[int(ha)]; out = []
    for ax, key in zip(axs, ["MT", "ML"]):
        X = np.abs(F[key][b:]) ** 2 / (L2 * T * w); X = X[: len(X) // nb * nb]
        blk = X.reshape(nb, -1, X.shape[1]).mean(1)
        y = np.array([X[:, np.round(k, 2) == u].mean() for u in ku])
        e = np.array([blk[:, np.round(k, 2) == u].mean(1).std() / np.sqrt(nb) for u in ku])
        ax.errorbar(ku * (1 + 0.012 * (ha - 3)), y, e, fmt=m, color=c, ms=7, mfc=c if ha == 2 else "none", mew=1.5,
                    elinewidth=1, capsize=0)
        out.append(y.mean())
    print(f"{ha:4.0f} {sa:5.0f} {Vn:8.4f}  {out[0]:.3f}  {out[1]:.3f}   improved r_m <V>n (T0/T(S0))^3 = {rc['r_m'] * Vn * rc['w'] / w:.3f}")
for ax, name in zip(axs, ["transverse", "longitudinal"]):
    ax.axhline(1, color="0.45", lw=2, label=r"theory: $\langle V_a\rangle\bar n\simeq1$, every $\sigma$")
    for sa, c in cols.items():                    # improved: bath T0, exact single particle, in the units of the plot (<V_a>nbar = 0.985)
        lb = from_label(sa); ax.axhline(exact(lb["sigma"])["r_m"] * 0.985 * lb["ratio3"], color=c, lw=2, ls="--")
    ax.plot([], [], "k--", lw=2, label=r"improved: bath $T_0$, exact single particle")
    ax.set(xlabel="$kh$", xlim=(0.55, 3.2)); ax.set_title(f"{name} momentum", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
for sa, c in cols.items(): axs[0].plot([], [], "o", color=c, ms=8, label=rf"$\sigma={sa/4:.3g}$ ($s_a={sa}$)")
axs[0].plot([], [], "ko", ms=8, label="$h/a=2$"); axs[0].plot([], [], "k^", ms=8, mfc="none", label="$h/a=3$")
axs[0].plot([], [], "ks", ms=8, mfc="none", label="$h/a=4$")
axs[0].set_ylabel(r"$\langle|M_k|^2\rangle/(L^2\,T(e+p))$"); fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=12, ncol=5, loc="lower center")
fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig("docs/figs/momentum_fluct.png", dpi=160); print("wrote docs/figs/momentum_fluct.png")
