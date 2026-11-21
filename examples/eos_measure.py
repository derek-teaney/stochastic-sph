"""eos_measure.py — campaign eos_renorm (docs/gravity.tex, "Measurement plan: flat space first"): combine the seeds of
examples/eos_measure.jl and plot the equation of state of the model with the tree-level and the renormalized Hamiltonian.

    python3 examples/eos_measure.py <root> [figdir]

<root> holds one directory per run (e.g. $SCRATCH/sph/eos_renorm), each with <tag>_seed<k>.h5.  Run names follow the campaign:
eos_{tree,ren}_s<sa>_ha<ha>[_L0.9].  Writes to figdir (default docs/figs):
  eos_EP.png      E/e(T0) - 1 and P/p(T0) - 1 against sigma, both Hamiltonians, h/a = 2, 3, 4; the one-loop tree value 2.5/sigma
  eos_trace.png   (P - E/2)/(n T) times n h^2 against sigma: the correction of the positions, which scales as 1/(n h^2)
  eos_derivs.png  dE/dT and dP/dT at fixed density over the target (canonical fluctuations; finite differences from the
                  s_a x 0.9, 1.1 runs), and c_s^2 at fixed density
  eos_Sk.png      S(k) of the positions in the L = 0.9 boxes, tree against renormalized
  eos_table.csv   the combined numbers, one row per run
"""
import glob, os, re, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1]
FIG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "docs", "figs")
os.makedirs(FIG, exist_ok=True)
BURN = 1 / 3


def per_seed(path):
    with h5py.File(path, "r") as f:
        A = dict(f.attrs)
        E = f["E"][:]; P = f["P"][:]; Pk = f["Pkin"][:]; Pv = f["Pvir"][:]
        Nk = f["part_Nk"][:]; ks = f["part_ks"][:]
    b = int(BURN * len(E))
    E, P, Pk, Pv, Nk = E[b:], P[b:], Pk[b:], Pv[b:], Nk[b:]
    T0, et, pt, N = A["T0"], A["e_target"], A["p_target"], A["N"]
    L = A["L"]; area = L[0] * L[1]; n = N / area
    vE = np.var(E, ddof=1); cPE = np.cov(P, E)[0, 1]
    return dict(
        sigma=A["sigma"], T0=T0, ha=A["ha"], nh2=n * A["h"] ** 2, renorm=bool(A["renorm"]), h=A["h"], e_t=et, p_t=pt,
        E=E.mean(), P=P.mean(),
        Eerr=E.mean() / et - 1, Perr=P.mean() / pt - 1,
        trace=(P - E / 2).mean() / (n * T0), Pkin=Pk.mean() / (n * T0), Pvir=(Pv.mean() - pt) / (n * T0),
        dEdT=area * vE / T0**2 / (3 * et / T0) - 1, dPdT=area * cPE / T0**2 / (3 * pt / T0) - 1, c2=cPE / vE,
        Sk=(np.abs(Nk) ** 2 / N).mean(0), kh=np.hypot(ks[:, 0], ks[:, 1]) * A["h"],
    )


def combine(seeds):
    out = {k: seeds[0][k] for k in ("sigma", "T0", "ha", "nh2", "renorm", "h", "e_t", "p_t")}
    out["sigma"] = np.mean([s["sigma"] for s in seeds]); out["T0"] = np.mean([s["T0"] for s in seeds])
    for k in ("E", "P", "Eerr", "Perr", "trace", "Pkin", "Pvir", "dEdT", "dPdT", "c2"):
        v = np.array([s[k] for s in seeds])
        out[k] = v.mean(); out[k + "_sd"] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    out["Sk"] = np.mean([s["Sk"] for s in seeds], axis=0); out["kh"] = seeds[0]["kh"]; out["nseeds"] = len(seeds)
    return out


runs = {}
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    files = sorted(glob.glob(os.path.join(d, "*_seed*.h5")))
    if files:
        runs[os.path.basename(d)] = combine([per_seed(p) for p in files])
print(f"{len(runs)} runs")
pat = re.compile(r"eos_(tree|ren)_s([\d.]+)_ha(\d)(_L0\.9)?$")


def sel(kind, ha=None, big=False, sa=None):
    r = []
    for name, v in runs.items():
        m = pat.match(name)
        if not m or m.group(1) != kind or bool(m.group(4)) != big:
            continue
        if ha is not None and int(m.group(3)) != ha:
            continue
        if sa is not None and float(m.group(2)) not in sa:
            continue
        r.append((name, v))
    return sorted(r, key=lambda x: x[1]["sigma"])


MAIN = (16.0, 54.0, 541.0)
cols = {2: "C0", 3: "C1", 4: "C2"}
sg = np.geomspace(3, 250, 100)

# 1. E and P against the target
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for j, key in enumerate(("Eerr", "Perr")):
    for ha in (2, 3, 4):
        for kind, mk in (("tree", "o"), ("ren", "s")):
            r = sel(kind, ha, sa=MAIN)
            if r:
                x = [v["sigma"] for _, v in r]; y = [v[key] for _, v in r]; e = [v[key + "_sd"] for _, v in r]
                ax[j].errorbar(x, y, e, fmt=mk, color=cols[ha], mfc=cols[ha] if kind == "tree" else "none",
                               label=f"{'tree' if kind == 'tree' else 'renorm.'}, h/a={ha}")
    ax[j].plot(sg, 2.5 / sg, "k--", lw=1, label="tree, one loop: 2.5/σ")
    ax[j].axhline(0, color="k", lw=0.6)
    ax[j].set_xscale("log"); ax[j].set_xlabel("σ = p(T0)/(T0 n̄)")
    ax[j].set_ylabel(["E/e(T0) − 1", "P/p(T0) − 1"][j])
ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "eos_EP.png"), dpi=150)

# 2. the trace, scaled by the number of neighbours
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for ha in (2, 3, 4):
    for kind, mk in (("tree", "o"), ("ren", "s")):
        r = sel(kind, ha, sa=MAIN)
        if r:
            ax.errorbar([v["sigma"] for _, v in r], [v["trace"] * v["nh2"] for _, v in r],
                        [v["trace_sd"] * v["nh2"] for _, v in r], fmt=mk, color=cols[ha],
                        mfc=cols[ha] if kind == "tree" else "none", label=f"{kind}, h/a={ha}")
ax.set_xscale("log"); ax.set_xlabel("σ"); ax.set_ylabel("n̄h² (P − E/2)/(n̄T)")
ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "eos_trace.png"), dpi=150)

# 3. temperature derivatives and c_s^2 at fixed density
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for j, key in enumerate(("dEdT", "dPdT", "c2")):
    for ha in (2, 3, 4):
        for kind, mk in (("tree", "o"), ("ren", "s")):
            r = sel(kind, ha, sa=MAIN)
            if r:
                ax[j].errorbar([v["sigma"] for _, v in r], [v[key] for _, v in r], [v[key + "_sd"] for _, v in r],
                               fmt=mk, color=cols[ha], mfc=cols[ha] if kind == "tree" else "none", label=f"{kind}, h/a={ha}")
    if key != "c2":
        ax[j].plot(sg, 5 / (6 * sg), "k--", lw=1, label="tree, one loop: 5/(6σ)")
        ax[j].axhline(0, color="k", lw=0.6)
    else:
        ax[j].axhline(0.5, color="k", lw=0.6)
    ax[j].set_xscale("log"); ax[j].set_xlabel("σ")
    ax[j].set_ylabel(["(∂E/∂T)/(de/dT) − 1", "(∂P/∂T)/(dp/dT) − 1", "c_s² at fixed n̄"][j])
# finite differences from the s_a x 0.9, 1.1 runs at h/a = 2 (fixed geometry, so fixed density)
for kind, mk in (("tree", "D"), ("ren", "d")):
    for lo, mid, hi in ((48.6, 54.0, 59.4), (486.9, 541.0, 595.1)):
        rl = sel(kind, 2, sa=(lo,)); rm = sel(kind, 2, sa=(mid,)); rh = sel(kind, 2, sa=(hi,))
        if rl and rm and rh:
            a, m_, c = rl[0][1], rm[0][1], rh[0][1]
            dT = c["T0"] - a["T0"]
            fdE = (c["E"] - a["E"]) / dT / (3 * m_["e_t"] / m_["T0"]) - 1
            fdP = (c["P"] - a["P"]) / dT / (3 * m_["p_t"] / m_["T0"]) - 1
            ax[0].plot(m_["sigma"], fdE, mk, color="k", mfc="none" if kind == "ren" else "k", ms=8, label=f"finite diff., {kind}")
            ax[1].plot(m_["sigma"], fdP, mk, color="k", mfc="none" if kind == "ren" else "k", ms=8)
            ax[2].plot(m_["sigma"], (c["P"] - a["P"]) / (c["E"] - a["E"]), mk, color="k", mfc="none" if kind == "ren" else "k", ms=8)
ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "eos_derivs.png"), dpi=150)

# 4. S(k) in the large boxes
fig, ax = plt.subplots(figsize=(6, 4.2))
for kind, ls in (("tree", "-"), ("ren", "--")):
    for name, v in sel(kind, big=True):
        o = np.argsort(v["kh"])
        ax.plot(v["kh"][o], v["Sk"][o], ls, marker=".", label=f"{name.replace('eos_', '')}")
ax.set_yscale("log"); ax.set_xlabel("kh"); ax.set_ylabel("S(k)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "eos_Sk.png"), dpi=150)

# 5. the table
keys = ["sigma", "T0", "ha", "nh2", "nseeds", "Eerr", "Eerr_sd", "Perr", "Perr_sd", "trace", "trace_sd", "Pkin", "Pkin_sd",
        "Pvir", "Pvir_sd", "dEdT", "dEdT_sd", "dPdT", "dPdT_sd", "c2", "c2_sd"]
with open(os.path.join(FIG, "eos_table.csv"), "w") as f:
    f.write("run," + ",".join(keys) + "\n")
    for name, v in sorted(runs.items()):
        f.write(name + "," + ",".join(f"{v[k]:.6g}" for k in keys) + "\n")
print("wrote", FIG)
