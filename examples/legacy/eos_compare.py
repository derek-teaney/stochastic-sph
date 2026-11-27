"""eos_compare.py — the side-by-side of the campaigns eos_legendre and eos_split (docs/step5_plan.tex, "For the NERSC session", 7):
the tree level, SPH_RENORM=1 (the label part alone), the Legendre Hamiltonian and the split one (Step 5) in the same panels.
Reads docs/figs/legendre_table.csv (Legendre) and docs/figs/split_table.csv (tree, SPH_RENORM=1, split; the tree level and
SPH_RENORM=1 at c_s^2 = 1/2, v = 0.8 are the eos_legendre runs), as written by examples/eos_legendre.py and examples/eos_split.py.
  python3 examples/eos_compare.py [figdir]
Writes split_compare_rest.png (E/e-1, P/p-1, chi_MM at rest), split_compare_moving.png (E, T^yy, M, R at v = 0.8),
split_compare_chi.png (the four susceptibilities at v = 0.8) and split_compare_table.csv, against 1/s (s = (1 + c2) sigma / c2, measured).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figs")


def load(path):
    a = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return [dict(zip(a.dtype.names, row)) for row in np.atleast_1d(a)]

rows = [r for r in load(os.path.join(FIG, "legendre_table.csv")) if r["scheme"] == "leg"]
rows += load(os.path.join(FIG, "split_table.csv"))
SCH = ("tree", "ren1", "leg", "split")
COL = {"tree": "C0", "ren1": "C2", "leg": "C3", "split": "C1"}
LAB = {"tree": "tree", "ren1": "SPH_RENORM=1 (label only)", "leg": "Legendre", "split": "split (Step 5)"}
MK = {2: "o", 3: "s"}


def sel(sc, c2, v, ha):
    r = [x for x in rows if x["scheme"] == sc and abs(x["c2"] - c2) < 1e-9 and abs(x["v"] - v) < 1e-9 and x["ha"] == ha]
    return sorted(r, key=lambda x: x["s_t"])


def panel(ax, key, c2, v, ylabel, schemes=SCH, ylim=None):
    for sc in schemes:
        for ha in (2, 3):
            r = sel(sc, c2, v, ha)
            if not r:
                continue
            x = [1 / q["s_t"] for q in r];  y = [q[key] for q in r];  e = [q[key + "_sd"] for q in r]
            ax.errorbar(x, y, e, fmt=MK[ha] + "-", lw=0.7, color=COL[sc], mfc=COL[sc] if ha == 2 else "none", ms=5,
                        label=f"{LAB[sc]}, h/a={ha}")
    ax.axhline(0, color="k", lw=0.6);  ax.set_xlabel("1/s");  ax.set_ylabel(ylabel)
    ax.set_title(f"c_s² = {c2:g}, v = {v:g}", fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)


# 1. at rest; the tree level is far off the scale of the others at small s: a second row without it
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for i, c2 in enumerate((0.5, 0.25)):
    panel(axs[i, 0], "E_over_e_minus1", c2, 0.0, "E/e − 1", schemes=SCH[1:])
    panel(axs[i, 1], "P_over_p_minus1", c2, 0.0, "P/p − 1", schemes=SCH[1:])
    panel(axs[i, 2], "chi_dMdv_over_target_minus1", c2, 0.0, "χ_MM/(T w) − 1", schemes=SCH[1:])
axs[0, 0].legend(fontsize=7)
fig.suptitle("At rest: SPH_RENORM=1 (label only), Legendre, split (the tree level is in legendre_EP.png)", fontsize=10)
fig.tight_layout();  fig.savefig(os.path.join(FIG, "split_compare_rest.png"), dpi=140)

# 2. in motion
fig, axs = plt.subplots(2, 4, figsize=(19, 8))
for i, c2 in enumerate((0.5, 0.25)):
    panel(axs[i, 0], "E_over_e_minus1", c2, 0.8, "E/e − 1 (lab)", schemes=SCH[1:])
    panel(axs[i, 1], "Tyy_over_p_minus1", c2, 0.8, "T^yy/p − 1", schemes=SCH[1:])
    panel(axs[i, 2], "M_over_target_minus1", c2, 0.8, "M/(wγ²v) − 1", schemes=SCH[1:])
    panel(axs[i, 3], "R", c2, 0.8, "R = M/(v(E + T^yy)) − 1")
axs[0, 0].legend(fontsize=7)
fig.suptitle("v = 0.8: SPH_RENORM=1 (label only), Legendre, split; R also for the tree level", fontsize=10)
fig.tight_layout();  fig.savefig(os.path.join(FIG, "split_compare_moving.png"), dpi=140)

# 3. the susceptibilities at v = 0.8
fig, axs = plt.subplots(2, 4, figsize=(19, 8))
for i, c2 in enumerate((0.5, 0.25)):
    panel(axs[i, 0], "chi_dEdT_over_target_minus1", c2, 0.8, "(∂E/∂T)_v / target − 1", schemes=SCH[1:])
    panel(axs[i, 1], "chi_dMdv_over_target_minus1", c2, 0.8, "(∂M/∂v)_T / target − 1", schemes=SCH[1:])
    panel(axs[i, 2], "chi_dEdv_over_target_minus1", c2, 0.8, "(∂E/∂v)_T / target − 1", schemes=SCH[1:])
    panel(axs[i, 3], "chi_dMdT_over_target_minus1", c2, 0.8, "(∂M/∂T)_v / target − 1", schemes=SCH[1:])
axs[0, 0].legend(fontsize=7)
fig.suptitle("The susceptibility matrix at v = 0.8: SPH_RENORM=1, Legendre, split", fontsize=10)
fig.tight_layout();  fig.savefig(os.path.join(FIG, "split_compare_chi.png"), dpi=140)

# 4. one table: per (c2, v, s_a, h/a) the main entries of the four Hamiltonians side by side
keys = ["s_t", "acceptance", "E_over_e_minus1", "P_over_p_minus1", "Tyy_over_p_minus1", "M_over_target_minus1", "R",
        "chi_dEdT_over_target_minus1", "chi_dMdv_over_target_minus1", "legendre_fails"]
with open(os.path.join(FIG, "split_compare_table.csv"), "w") as f:
    f.write("c2,v,sa,ha,scheme," + ",".join(k if k in ("acceptance", "legendre_fails", "s_t") else f"{k},{k}_sd" for k in keys) + "\n")
    for c2 in (0.5, 0.25):
        for v in (0.0, 0.8):
            for sa in (16, 54, 541):
                for ha in (2, 3):
                    for sc in SCH:
                        r = [x for x in sel(sc, c2, v, ha) if x["sa"] == sa]
                        if not r:
                            continue
                        q = r[0]
                        vals = [f"{q[k]:.6g}" if k in ("acceptance", "legendre_fails", "s_t") else f"{q[k]:.6g},{q[k + '_sd']:.3g}" for k in keys]
                        f.write(f"{c2},{v},{sa},{ha},{sc}," + ",".join(vals) + "\n")
print("wrote", FIG)
