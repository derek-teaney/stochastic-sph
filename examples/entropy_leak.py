"""entropy_leak.py (at the bath temperature T0 with the exact single particle, examples/statics_theory.py: intrinsic r_s <V_a>nbar plus the leak through <s|V> = s_off + b V/Vbar) — do the fluid fields see the positions?  The entropy density S_k = Σ_a s_a e^{-ik·x_a} of the HMC runs
(examples/volume_fluctuations.jl) against the Gaussian theory of docs/equilibrium.tex:

    <|S_k|^2> / (L^2 C_V) = 1 + (s_a c_s^2 / ... ) -> for c_s^2 = 1/3:  1 + (s_a/3) (1 - W~)^2 S(k)

the 1 from the particles' own entropy fluctuations (variance C_V V_a, summing to C_V L^2 whatever the positions), the rest
the leak of the positions through (1 - W~), which vanishes as (kh)^2.  Writes docs/figs/entropy_leak.png.

    python3 examples/entropy_leak.py data/volfluct/vf_*.h5
"""
import sys, warnings; warnings.filterwarnings("ignore")
import os, h5py, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from statics_theory import S_exact, exact, run_constants, from_label
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import j0
from scipy.integrate import simpson

q = np.linspace(0, 2, 20001); W = 7 / (4 * np.pi) * (1 - q / 2) ** 4 * (1 + 2 * q)
KH = np.linspace(1e-3, 6, 1500); WT = simpson(2 * np.pi * q * W * j0(np.outer(KH, q)), x=q, axis=1)
Wt = lambda k: np.interp(k, KH, WT)
Sg = lambda k, sa: 1 / (1 + 3 * Wt(k) * (1 - Wt(k) / 2) + 2 * (sa / 4) * Wt(k) * (1 - Wt(k)))

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
fig, ax = plt.subplots(figsize=(10, 7))
cols = {16: "#1f5fa8", 54: "#c0462a", 541: "#2b8a3e"}; mk = {2: "o", 3: "^", 4: "s"}
print(f"{'h/a':>4} {'s_a':>5}  kh: data/theory of <|S_k|^2>/(L^2 C_V)")
for f in sys.argv[1:]:
    F = h5py.File(f); A = {k: F.attrs[k] for k in F.attrs}; sa = float(A["s_a"]); ha = float(A["ha"]); h = float(A["h"])
    L2 = float(np.prod(A["L"])); rc = run_constants(F.attrs); CV = 3 * float(A["S0"]); b = len(F["t"]) // 3
    k = np.hypot(*F["ks"][:].T) * h
    Sser = np.abs(F["S"][b:]) ** 2 / (L2 * CV); nb = 8; Sser = Sser[: len(Sser) // nb * nb]
    Ss = Sser.mean(0); Sblk = Sser.reshape(nb, -1, Sser.shape[1]).mean(1)
    Sp = (np.abs(F["part_Nk"][b:]) ** 2).mean(0) / float(A["N"])          # measured S(k) of the positions
    ku = np.unique(np.round(k, 2)); grp = lambda x: np.array([x[np.round(k, 2) == u].mean() for u in ku])
    Ss, Sp = grp(Ss), grp(Sp); Se = np.array([Sblk[:, np.round(k, 2) == u].mean(1).std() / np.sqrt(nb) for u in ku])
    vn = F["vol_mean"][b:].mean(); lk = lambda W: (rc["s_off"] + rc["b"] * (1 - W)) ** 2 * rc["saT0"] / 3   # (s_off + b(1-W))^2 (sV)^2/(C_V V)
    r3 = rc["CV"] / CV                                        # (T0/T(S0))^3: C_V(T0) over the C_V of the start
    pred_g = r3 * (rc["r_s"] * vn + lk(Wt(ku)) * S_exact(Wt(ku), rc["alpha"], rc["sig"]))
    c = cols[int(round(sa))]; m = mk[int(ha)]
    ax.errorbar(ku * (1 + 0.012 * (ha - 3)), Ss, Se, fmt=m, color=c, ms=7, mfc=c if ha == 2 else "none", mew=1.5, elinewidth=1, capsize=0)
    print(f"{ha:4.0f} {sa:5.0f}  " + "  ".join(f"{x:.2f}:{y:.3f}/{z:.3f}" for x, y, z in zip(ku[:6], Ss[:6], pred_g[:6])))
kk = np.linspace(0.3, 3.2, 300); W = Wt(kk)
for sa, c in cols.items():
    ax.plot(kk, 1 + sa / 3 * (1 - W) ** 2 * Sg(kk, sa), "-", color=c, lw=2, label=rf"theory, $\sigma={sa/4:.3g}$")
    lb = from_label(sa); sg = lb["sigma"]; e = exact(sg); saT = 4 * sg   # improved: at T0, exact single particle (<V_a>nbar = 0.985)
    y = lb["ratio3"] * (e["r_s"] * 0.985 + (e["s_off"] + e["b"] * (1 - W)) ** 2 * saT / 3 * S_exact(W, e["alpha"], e["sig"]))
    ax.plot(kk, y, "--", color=c, lw=2)
ax.plot([], [], "ko", ms=8, label="HMC $h/a=2$"); ax.plot([], [], "k^", ms=8, mfc="none", label="HMC $h/a=3$")
ax.plot([], [], "ks", ms=8, mfc="none", label="HMC $h/a=4$")
ax.axhline(1, color="0.6", lw=1, ls=":"); ax.text(2.25, 0.9, "fluid alone (intrinsic)", color="0.4", fontsize=12)
ax.plot([], [], "k--", lw=2, label=r"improved: bath $T_0$, exact single particle")
ax.set(xlabel="$kh$", ylabel=r"$\langle|S_k|^2\rangle/(L^2C_V)$", xlim=(0.55, 3.2))
ax.legend(frameon=False, fontsize=12, ncol=2, loc="upper left")
ax.set_title("the leak of the positions into the entropy density", fontsize=14)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
fig.savefig("docs/figs/entropy_leak.png", dpi=160); print("wrote docs/figs/entropy_leak.png")
