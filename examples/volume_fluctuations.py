"""volume_fluctuations.py — the Gaussian theory of the positions (docs/equilibrium.tex) against hybrid Monte Carlo.

    python3 examples/volume_fluctuations.py data/volfluct/vf_*.h5

Prints the volume statistics (mean and variance, after a burn-in of a third) against the theory summed over the box
modes, and writes docs/figs/equilibrium_{Sk,gr,volvar}.png: S(k), g(r) and the volume variance against the theory.
Runs are labelled by sigma(T0) = p V/T at the bath temperature T0 of each run (c_s^2 from the run attributes; docs/dimension.tex).
Solid: the Gaussian theory at its saddle point (Pi = sigma + 3/2, K = 3/2) at sigma(T0).  Dashed: the exact single particle at sigma(T0) (examples/statics_theory.py).
"""
import sys, warnings
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import j0
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__("os").path.dirname(__file__))
from statics_theory import S_gauss, S_exact, exact, run_constants

# Wendland C2 in 2D, h = 1, and its Fourier transform
r = np.linspace(0, 2, 4001); Wr = 7 / (4 * np.pi) * (1 - r / 2) ** 4 * (1 + 2 * r)
KH = np.concatenate([np.linspace(1e-5, 1, 2001)[:-1], np.linspace(1, 400, 80001)])
WT = np.trapezoid(2 * np.pi * r * Wr * j0(np.outer(KH, r)), r, axis=1)

def Sk(w, sig):                     # eq. (Sk), the Gaussian theory at σ(T0)
    return S_gauss(w, sig)
def Sx(w, sig):                     # exact single particle at σ(T0)
    e = exact(sig); return S_exact(w, e["Pi"], e["K"])

def box_modes(Lh):
    n = np.arange(-int(400 * Lh / 2 / np.pi) - 1, int(400 * Lh / 2 / np.pi) + 2)
    kx, ky = np.meshgrid(n, n); k = 2 * np.pi * np.hypot(kx, ky) / Lh
    return k[(k > 0) & (k < 400)]

def volume_theory(ha, sig, Lh, exact_sp=False):   # eq. (Vfluct) summed over the box modes at σ(T0); Gaussian or exact single particle
    nh2 = ha ** 2 * 2 / np.sqrt(3); k = box_modes(Lh); w = np.interp(k, KH, WT); s = Sx(w, sig) if exact_sp else Sk(w, sig)
    dn = (w * s).sum() / Lh ** 2 / nh2; var = (w * w * s).sum() / Lh ** 2 / nh2
    return (1 + var) / (1 + dn), var

def g_theory(rh, ha, sig, exact_sp=False):
    nh2 = ha ** 2 * 2 / np.sqrt(3); s = Sx(WT, sig) if exact_sp else Sk(WT, sig)
    return 1 + np.array([np.trapezoid(KH * j0(KH * x) * (s - 1), KH) / (2 * np.pi) for x in rh]) / nh2

def attrs(F):
    return {k: (float(np.ravel(v)[0]) if np.asarray(v).dtype.kind in "fiu" else v) for k, v in F.attrs.items()}

runs = []
for f in sys.argv[1:]:
    F = h5py.File(f); A = attrs(F); n = len(F["vol_var"]); b = n // 3
    vm = F["vol_mean"][b:]; vv = F["vol_var"][b:] * vm ** 2       # var(V_a) n̄², i.e. normalized by V = 1/n̄ (the file has var/⟨V_a⟩²)
    nb = 10; blk = lambda x: x[: len(x) // nb * nb].reshape(nb, -1).mean(1)       # block errors
    ks = F["part_ks"][:]; k = np.hypot(ks[:, 0], ks[:, 1]) * A["h"]
    P = (np.abs(F["part_Nk"][:]) ** 2 / A["N"])[b:].mean(0)
    rb = F["part_rbins"][:]; g = F["part_g"][b:].mean(0)
    runs.append(dict(ha=A["ha"], sa=A["s_a"], sig=run_constants(F.attrs)["sigma"], Lh=A["L"] / A["h"] if np.ndim(A["L"]) == 0 else A["L"] / A["h"],
                     vm=vm.mean(), vme=blk(vm).std() / np.sqrt(nb), vv=vv.mean(), vve=blk(vv).std() / np.sqrt(nb),
                     k=k, S=P, r=0.5 * (rb[1:] + rb[:-1]) / A["h"], g=g, acc=F["hmc_acc"][:].mean()))
runs.sort(key=lambda d: (d["ha"], d["sa"]))

print(f"{'h/a':>4} {'s_a':>6} {'sig_T0':>6} {'L/h':>5} {'acc':>5}  {'<V>n HMC':>16} {'gauss':>7} {'exact':>7}   {'1e3 var HMC':>14} {'gauss':>7} {'exact':>7} {'ratio':>6}")
for d in runs:
    vt, vart = volume_theory(d["ha"], d["sig"], d["Lh"]); vx, varx = volume_theory(d["ha"], d["sig"], d["Lh"], True)
    print(f"{d['ha']:4.0f} {d['sa']:6.0f} {d['sig']:6.2f} {d['Lh']:5.1f} {d['acc']:5.2f}  {d['vm']:.4f} ± {d['vme']:.4f}  {vt:7.4f} {vx:7.4f}   "
          f"{1e3*d['vv']:6.2f} ± {1e3*d['vve']:4.2f}  {1e3*vart:7.2f} {1e3*varx:7.2f} {d['vv']/varx:6.2f}")

# figures, labelled by sigma(T0) = p V/T: S(k), g(r) at h/a = 2 and 4; the volume variance against sigma(T0)
plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
cols = {16: "#1f5fa8", 54: "#c0462a", 541: "#2b8a3e"}
hcol = {2: "#1f5fa8", 3: "#7a4fa0", 4: "#2b8a3e"}
def finish(fig, ax, name):
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(f"docs/figs/{name}.png", dpi=160); print(f"wrote docs/figs/{name}.png")

# structure factor
fig, ax = plt.subplots(figsize=(9, 6.5))
for d in runs:
    if d["ha"] not in (2, 4): continue
    c = cols.get(int(round(d["sa"])), "k")
    kk = np.unique(np.round(d["k"], 2)); Sm = [d["S"][np.round(d["k"], 2) == u].mean() for u in kk]
    ax.plot(kk, Sm, "o" if d["ha"] == 2 else "s", color=c, ms=7, mfc="none" if d["ha"] == 4 else c)
sigT = {int(round(d["sa"])): d["sig"] for d in runs if d["ha"] == 2}
for sa, c in cols.items():
    x = np.linspace(0.05, 3, 300); w = np.interp(x, KH, WT)
    ax.plot(x, Sk(w, sigT[sa]), "-", color=c, lw=2, label=rf"$\sigma(T_0)={sigT[sa]:.3g}$")
    ax.plot(x, Sx(w, sigT[sa]), "--", color=c, lw=2)
ax.plot([], [], "k--", lw=2, label=r"exact single particle")
ax.set(xlabel="$kh$", ylabel="$S(k)$", yscale="log", xlim=(0, 3))
ax.legend(frameon=False, title=r"filled $h/a=2$, open $h/a=4$", title_fontsize=12)
finish(fig, ax, "equilibrium_Sk")

# pair correlation
fig, ax = plt.subplots(figsize=(9, 6.5))
for d in runs:
    if d["ha"] not in (2, 4): continue
    ls = "-" if d["ha"] == 2 else "--"; c = cols.get(int(round(d["sa"])), "k")
    ax.plot(d["r"], d["g"], "o" if d["ha"] == 2 else "s", color=c, ms=4, mfc="none" if d["ha"] == 4 else c)
    rr = np.linspace(0.01, 2, 200); lw = 2 if d["ha"] == 2 else 1
    ax.plot(rr, g_theory(rr, d["ha"], d["sig"]), "-", color=c, lw=lw)
    ax.plot(rr, g_theory(rr, d["ha"], d["sig"], True), "--", color=c, lw=lw)
for sa, c in cols.items(): ax.plot([], [], "-", color=c, lw=2, label=rf"$\sigma(T_0)={sigT[sa]:.3g}$")
ax.plot([], [], "k-", lw=2, label="theory, $h/a=2$ (thin: $4$)"); ax.plot([], [], "k--", lw=2, label="exact single particle")
ax.set(xlabel="$r/h$", ylabel="$g(r)$", xlim=(0, 2)); ax.axhline(1, color="0.7", lw=0.8)
ax.legend(frameon=False, loc="lower right", title=r"filled $h/a=2$, open $h/a=4$", title_fontsize=12)
finish(fig, ax, "equilibrium_gr")

# volume variance times the number of neighbours
fig, ax = plt.subplots(figsize=(9, 6.5))
for ha, m in [(2, "o"), (3, "^"), (4, "s")]:
    sel = [d for d in runs if d["ha"] == ha]
    if not sel: continue
    nh2 = ha ** 2 * 2 / np.sqrt(3)
    ax.errorbar([d["sig"] for d in sel], [d["vv"] * nh2 for d in sel], [d["vve"] * nh2 for d in sel], fmt=m, ms=9,
                label=rf"$h/a={ha}$, $\bar nh^2={nh2:.3g}$", color=hcol[ha])
    sg = np.geomspace(2.5, 400, 30); Lh = sel[0]["Lh"]                     # sg: sigma(T0)
    ax.plot(sg, [volume_theory(ha, s, Lh)[1] * nh2 for s in sg], "-", lw=1.5, color=hcol[ha])
    ax.plot(sg, [volume_theory(ha, s, Lh, True)[1] * nh2 for s in sg], "--", lw=1.5, color=hcol[ha])
ax.plot([], [], "k-", lw=1.5, label=r"saddle point, $\Pi=\sigma+3/2$, $\mathcal{K}=3/2$")
ax.plot([], [], "k--", lw=1.5, label=r"exact single particle")
ax.set(xscale="log", yscale="log", xlabel=r"$\sigma(T_0)=p/(T_0\bar n)$", ylabel=r"$\bar n h^2\,\bar n^2\langle\delta V_a^2\rangle$")
ax.legend(frameon=False)
finish(fig, ax, "equilibrium_volvar")
