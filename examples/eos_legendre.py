"""eos_legendre.py — campaign eos_legendre (docs/legendre_plan.tex, "The campaign"): combine the seeds of examples/eos_measure.jl and plot
the equation of state, the momentum and the susceptibility matrix of the model for the tree-level Hamiltonian, the entropy shift
(SPH_RENORM=1) and the Legendre Hamiltonian (SPH_RENORM=legendre), c_s^2 = 1/2 and 1/4, at rest and at v = 0.8.

    python3 examples/eos_legendre.py <root> [figdir]

<root> holds one directory per run, eosL_{tree,leg,ren1}_c{05,025}_s<sa>_ha<ha>_v{0,08}, each with <tag>_seed<k>.txt (key = value).
The abscissa is 1/s, s = (1 + c_s^2) sigma / c_s^2 the target's entropy per particle at the bath's T_rel (sigma = p V / T_rel).
One-loop expectations (legendre_plan.tex, table "Expectations"; the correction of the positions, O(1/(n h^2)), comes on top):
  tree:     P/p-1 = 2.5/sigma,  E/e-1 = k (1 + c2)/s,  chi_MM/(T w)-1 = (5/2 + k - c2)/s,  R = -c2 (1-v^2)^2 / ((1 - c2 v^2) s)
  renorm=1: 0, 0, -c2/s, R as tree;   Legendre: O(1/s^2) for all;   k = 2 + 3/(2 c2).
Writes to figdir (default docs/figs): legendre_EP.png, legendre_moving.png, legendre_chi.png, legendre_table.csv.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1]
FIG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "docs", "figs")
os.makedirs(FIG, exist_ok=True)
pat = re.compile(r"eosL_(tree|leg|ren1)_c(05|025)_s(\d+)_ha(\d)_v(0|08)$")
KEYS = ["sigma", "T0", "acceptance", "E_over_e_minus1", "P_over_p_minus1", "Tyy_over_p_minus1", "M_over_target_minus1", "R",
        "trace_over_nT", "Pkin_over_nT", "cs2_fixed_n", "dEdT_over_target_minus1", "dPdT_over_target_minus1",
        "chi_dEdT_over_target_minus1", "chi_dMdv_over_target_minus1", "chi_dEdv", "chi_dEdv_target", "chi_dMdT", "chi_dMdT_target",
        "chi_yy_over_target_minus1", "nbar_h2", "legendre_solves", "legendre_fails", "legendre_nostate", "seconds"]


def read(path):
    d = {}
    for line in open(path):
        if " = " in line:
            k, v = line.rstrip("\n").split(" = ", 1)
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


runs = {}
for d in sorted(glob.glob(os.path.join(ROOT, "eosL_*"))):
    name = os.path.basename(d);  m = pat.match(name)
    files = sorted(glob.glob(os.path.join(d, name + "_seed*.txt")))           # not the provenance files run_seed*.txt
    if not m or not files:
        continue
    seeds = [read(f) for f in files]
    out = dict(scheme=m.group(1), c2=0.5 if m.group(2) == "05" else 0.25, sa=int(m.group(3)), ha=int(m.group(4)),
               v=0.0 if m.group(5) == "0" else 0.8, nseeds=len(seeds))
    for k in KEYS:
        x = np.array([s.get(k, np.nan) for s in seeds], float)
        x = x[np.isfinite(x)]
        out[k] = x.mean() if len(x) else np.nan
        out[k + "_sd"] = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    # the off-diagonal entries relative to the target's
    for k in ("chi_dEdv", "chi_dMdT"):
        out[k + "_over_target_minus1"] = out[k] / out[k + "_target"] - 1 if out["v"] > 0 else np.nan
        out[k + "_over_target_minus1_sd"] = out[k + "_sd"] / abs(out[k + "_target"]) if out["v"] > 0 else np.nan
    c2 = out["c2"];  out["s_t"] = (1 + c2) * out["sigma"] / c2
    # M and R are 0/0 at rest; at v = 0.8 the isotropic P is not compared with the rest-frame p (T^yy is)
    for k in (("M_over_target_minus1", "R") if out["v"] == 0 else ("P_over_p_minus1",)):
        out[k] = out[k + "_sd"] = np.nan
    runs[name] = out
print(f"{len(runs)} runs")

kk = lambda c2: 2 + 1.5 / c2
def expect(q, scheme, c2, s, v=0.8):
    k = kk(c2);  sig = c2 * s / (1 + c2)
    if q == "P":   return 2.5 / sig if scheme == "tree" else 0 * s
    if q == "E":   return k * (1 + c2) / s if scheme == "tree" else 0 * s
    if q == "chiMM": return (2.5 + k - c2) / s if scheme == "tree" else (-c2 / s if scheme == "ren1" else 0 * s)
    if q == "R":   return -c2 * (1 - v * v)**2 / ((1 - c2 * v * v) * s) if scheme in ("tree", "ren1") else 0 * s
    return 0 * s

COL = {"tree": "C0", "ren1": "C2", "leg": "C3"};  LAB = {"tree": "tree", "ren1": "SPH_RENORM=1", "leg": "Legendre"}
MK = {2: "o", 3: "s"}

def sel(scheme, c2, v, ha=None):
    r = [x for x in runs.values() if x["scheme"] == scheme and x["c2"] == c2 and x["v"] == v and (ha is None or x["ha"] == ha)]
    return sorted(r, key=lambda x: x["s_t"])

def panel(ax, key, c2, v, qexp, ylabel, schemes=("tree", "ren1", "leg")):
    xs = np.linspace(0, 0.2, 100)
    for sc in schemes:
        for ha in (2, 3):
            r = sel(sc, c2, v, ha)
            if not r:
                continue
            x = [1 / q["s_t"] for q in r];  y = [q[key] for q in r];  e = [q[key + "_sd"] for q in r]
            ax.errorbar(x, y, e, fmt=MK[ha], color=COL[sc], mfc=COL[sc] if ha == 2 else "none", ms=5,
                        label=f"{LAB[sc]}, h/a={ha}")
        if qexp and sel(sc, c2, v):
            ax.plot(xs, expect(qexp, sc, c2, 1 / np.maximum(xs, 1e-9), v if v > 0 else 0.8), "--", color=COL[sc], lw=1)
    ax.axhline(0, color="k", lw=0.6);  ax.set_xlabel("1/s");  ax.set_ylabel(ylabel)
    ax.set_title(f"c_s² = {c2:g}, v = {v:g}", fontsize=9)

# 1. at rest: E, P, chi_MM
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for i, c2 in enumerate((0.5, 0.25)):
    panel(axs[i, 0], "E_over_e_minus1", c2, 0.0, "E", "E/e − 1")
    panel(axs[i, 1], "P_over_p_minus1", c2, 0.0, "P", "P/p − 1")
    panel(axs[i, 2], "chi_dMdv_over_target_minus1", c2, 0.0, "chiMM", "χ_MM/(T w) − 1")
axs[0, 0].legend(fontsize=7)
fig.suptitle("At rest (dashed: one loop, without the positions)", fontsize=10)
fig.tight_layout();  fig.savefig(os.path.join(FIG, "legendre_EP.png"), dpi=140)

# 2. in motion: E, T^yy, M, R
fig, axs = plt.subplots(2, 4, figsize=(19, 8))
for i, c2 in enumerate((0.5, 0.25)):
    panel(axs[i, 0], "E_over_e_minus1", c2, 0.8, None, "E/e − 1 (lab)")
    panel(axs[i, 1], "Tyy_over_p_minus1", c2, 0.8, None, "T^yy/p − 1")
    panel(axs[i, 2], "M_over_target_minus1", c2, 0.8, None, "M/(wγ²v) − 1")
    panel(axs[i, 3], "R", c2, 0.8, "R", "R = M/(v(E + T^yy)) − 1")
axs[0, 0].legend(fontsize=7)
fig.suptitle("v = 0.8 (dashed: one-loop R)", fontsize=10)
fig.tight_layout();  fig.savefig(os.path.join(FIG, "legendre_moving.png"), dpi=140)

# 3. the susceptibility matrix against the target's entries
fig, axs = plt.subplots(2, 5, figsize=(23, 8))
for i, c2 in enumerate((0.5, 0.25)):
    panel(axs[i, 0], "chi_dEdT_over_target_minus1", c2, 0.8, None, "(∂E/∂T)_v / target − 1")
    panel(axs[i, 1], "chi_dMdv_over_target_minus1", c2, 0.8, None, "(∂M/∂v)_T / target − 1")
    panel(axs[i, 2], "chi_dEdv_over_target_minus1", c2, 0.8, None, "(∂E/∂v)_T / target − 1")
    panel(axs[i, 3], "chi_dMdT_over_target_minus1", c2, 0.8, None, "(∂M/∂T)_v / target − 1")
    panel(axs[i, 4], "chi_dEdT_over_target_minus1", c2, 0.0, None, "(∂E/∂T) / target − 1, v = 0")
axs[0, 0].legend(fontsize=7)
fig.suptitle("The susceptibility matrix from the canonical covariances, v = 0.8 (last column: v = 0)", fontsize=10)
fig.tight_layout();  fig.savefig(os.path.join(FIG, "legendre_chi.png"), dpi=140)

# 4. the table
cols = ["scheme", "c2", "sa", "ha", "v", "nseeds", "sigma", "s_t", "acceptance", "E_over_e_minus1", "P_over_p_minus1", "Tyy_over_p_minus1",
        "M_over_target_minus1", "R", "chi_dMdv_over_target_minus1", "chi_dEdT_over_target_minus1", "chi_dEdv_over_target_minus1",
        "chi_dMdT_over_target_minus1", "chi_yy_over_target_minus1", "cs2_fixed_n", "trace_over_nT", "legendre_fails", "legendre_solves",
        "seconds"]
with open(os.path.join(FIG, "legendre_table.csv"), "w") as f:
    f.write("run," + ",".join(c if c in ("scheme", "c2", "sa", "ha", "v", "nseeds") else f"{c},{c}_sd" for c in cols) + "\n")
    for name, q in sorted(runs.items()):
        vals = []
        for c in cols:
            vals.append(str(q[c]) if c in ("scheme", "c2", "sa", "ha", "v", "nseeds") else f"{q[c]:.6g},{q.get(c + '_sd', np.nan):.3g}")
        f.write(name + "," + ",".join(vals) + "\n")
print("wrote", FIG)
