"""single_particle_compare.py — Step 0 of docs/statics.tex: the nine HMC runs (examples/volume_fluctuations.jl) against the exact
single-particle quadrature (examples/single_particle.jl, data/volfluct/single_particle.csv), everything at the bath temperature T0 of
each run (not at the temperature of the start's entropy density S0).  Continuum at T0 for the constant-c_s^2 equation of state, c_s^2
and T_scale from the run attributes (docs/dimension.tex): s = (1+c_s^2) e/T, w = T s, C_V = s/c_s^2, sigma = c_s^2/(1+c_s^2) s V.

  momentum:   <|M_k|^2>/(L^2 T0 w)  =  r_m <V_a>nbar
  entropy:    <|S_k|^2>/(L^2 C_V)   =  r_s <V_a>nbar + (s_off + b (1 - W~))^2 S(k) / (C_V V),   <s|V'> = s_off + b V'/V near V
  positions:  S(k) from the Gaussian theory at sigma(label) = c_s^2/(1+c_s^2) s_a, at sigma(T0), and with the exact alpha, sigma (statics.tex, eq. Sexact)
  volumes:    var(V_a) nbar^2 from the same S(k), summed over the modes of the box

    python3 examples/single_particle_compare.py data/volfluct/vf_*.h5
Writes docs/figs/step0_momentum_entropy.png and docs/figs/step0_positions.png.
"""
import sys, csv, warnings; warnings.filterwarnings("ignore")
import os, h5py, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from statics_theory import thermo
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import j0
from scipy.integrate import simpson

rows = {round(float(r["sa_label"])): r for r in csv.DictReader(open("data/volfluct/single_particle.csv")) if r["measure"] == "liouville"}
q = np.linspace(0, 2, 20001); Wq = 7 / (4 * np.pi) * (1 - q / 2) ** 4 * (1 + 2 * q)
KH = np.linspace(0, 80, 8001); WT = simpson(2 * np.pi * q * Wq * j0(np.outer(KH, q)), x=q, axis=1)
Wt = lambda k: np.interp(k, KH, WT)
S_g = lambda W, sig: 1 / (1 + 3 * W * (1 - W / 2) + 2 * sig * W * (1 - W))
S_x = lambda W, f1, f2: 1 / (1 + 2 * (-f2) * W * (1 - W / 2) + 2 * (f1 + f2) * W * (1 - W))   # alpha = -f2, sigma = f1 + f2

def boxvar(Lx, Ly, h, nbar, Sfun):                      # (1/(nbar L^2)) sum_{k != 0} W~^2 S over the box modes, kh < 60
    nx = int(60 / h * Lx / (2 * np.pi)); ny = int(60 / h * Ly / (2 * np.pi))
    kx, ky = np.meshgrid(2 * np.pi * np.arange(-nx, nx + 1) / Lx, 2 * np.pi * np.arange(-ny, ny + 1) / Ly)
    k = np.hypot(kx, ky).ravel() * h; k = k[(k > 0) & (k < 60)]; W = Wt(k)
    return (W ** 2 * Sfun(W)).sum() / (nbar * Lx * Ly)

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15})
cols = {16: "#1f5fa8", 54: "#c0462a", 541: "#2b8a3e"}; mk = {2: "o", 3: "^", 4: "s"}
f1g, ax1 = plt.subplots(1, 2, figsize=(15, 6))
f2g, ax2 = plt.subplots(1, 2, figsize=(15, 6))
print(f"{'h/a':>3} {'s_a':>4} {'sig_T0':>6} | {'M data':>7} {'pred':>6} | {'S_k data/pred (k-avg)':>22} | "
      f"{'S(k) data/theory: label, T0, exact':>36} | {'10^3 var V: data':>16} {'label':>6} {'T0':>6} {'exact':>6}")
for f in sys.argv[1:]:
    F = h5py.File(f); A = {k: F.attrs[k] for k in F.attrs}; sa = round(float(A["s_a"])); ha = round(float(A["ha"]))
    h = float(A["h"]); Lx, Ly = A["L"]; L2 = Lx * Ly; N = float(A["N"]); nbar = N / L2; V = 1 / nbar; T0 = float(A["T0"])
    r = {k: float(v) for k, v in rows[sa].items() if k not in ("file", "measure")}
    cs2 = float(A.get("cs2", 1 / 3)); th = thermo(T0, cs2, float(A.get("T_scale", 1.0))); fs = cs2 / (1 + cs2)
    sT = th["s"]; w = th["w"]; CV = th["CV"]; sigT = fs * sT * V; sig_l = fs * sa
    rm = r["mx2"] / r["Tw_V"]; rs = r["var_s"] / r["CV_V"]
    # rescale the per-particle offsets to this run's V (they are dimensionless in units of s(T0)V)
    soff = r["s_minus"] / r["sa_T0"] * sT * V; bsl = r["dsdV_V"] / r["sa_T0"] * sT * V
    b = len(F["t"]) // 3; vn = F["vol_mean"][b:].mean()
    k = np.hypot(*F["ks"][:].T) * h; ku = np.unique(np.round(k, 2)); grp = lambda x: np.array([x[np.round(k, 2) == u].mean() for u in ku])
    Mk = grp(((np.abs(F["MT"][b:]) ** 2 + np.abs(F["ML"][b:]) ** 2) / 2).mean(0)) / (L2 * T0 * w)
    Sk = grp((np.abs(F["S"][b:]) ** 2).mean(0)) / (L2 * CV)
    Sp = grp((np.abs(F["part_Nk"][b:]) ** 2).mean(0) / N)
    W = Wt(ku); Mpred = rm * vn
    Spred = rs * vn + (soff + bsl * (1 - W)) ** 2 * Sp / (CV * V)
    th_l, th_t, th_x = S_g(W, sig_l), S_g(W, sigT), S_x(W, r["f1"], r["f2"])
    vv = F["vol_var"][b:].mean() * vn ** 2 * 1e3
    vl, vt, vx = [1e3 * boxvar(Lx, Ly, h, nbar, fn) for fn in
                  (lambda W: S_g(W, sig_l), lambda W: S_g(W, sigT), lambda W: S_x(W, r["f1"], r["f2"]))]
    print(f"{ha:3d} {sa:4d} {sigT:6.2f} | {Mk.mean():7.3f} {Mpred:6.3f} | {np.mean(Sk / Spred):22.3f} | "
          f"{np.mean(Sp / th_l):12.3f} {np.mean(Sp / th_t):11.3f} {np.mean(Sp / th_x):11.3f} | {vv:16.2f} {vl:6.2f} {vt:6.2f} {vx:6.2f}")
    c = cols[sa]; m = mk[ha]; fill = c if ha == 2 else "none"; dx = 1 + 0.012 * (ha - 3)
    ax1[0].plot(ku * dx, Mk / Mpred, m, color=c, mfc=fill, ms=7, mew=1.5)
    ax1[1].plot(ku * dx, Sk / Spred, m, color=c, mfc=fill, ms=7, mew=1.5)
    ax2[0].plot(ku * dx, Sp / th_l, m, color=c, mfc=fill, ms=7, mew=1.5)
    ax2[1].plot(ku * dx, Sp / th_x, m, color=c, mfc=fill, ms=7, mew=1.5)
for a, t in zip(ax1, [r"momentum: HMC / ($r_m\langle V_a\rangle\bar n$)", r"entropy: HMC / (intrinsic + leak, exact)"]):
    a.axhline(1, color="0.4", lw=1.5); a.set(xlabel="$kh$", ylim=(0.6, 1.4)); a.set_title(t, fontsize=13)
for a, t in zip(ax2, [r"$S(k)$: HMC / Gaussian theory at the label's $\sigma$, $c_s^2 s_a/(1+c_s^2)$",
                      r"$S(k)$: HMC / theory with $\alpha_{\rm eff},\sigma_{\rm eff}$ at $T_0$"]):
    a.axhline(1, color="0.4", lw=1.5); a.set(xlabel="$kh$", ylim=(0.6, 1.6)); a.set_title(t, fontsize=13)
for fig, ax in ((f1g, ax1), (f2g, ax2)):
    for sa, c in cols.items(): ax[0].plot([], [], "o", color=c, label=rf"$s_a={sa}$ (label)")
    for ha, m in mk.items(): ax[0].plot([], [], m, color="k", mfc="k" if ha == 2 else "none", label=f"$h/a={ha}$")
    ax[0].legend(frameon=False, fontsize=11, ncol=2)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
f1g.savefig("docs/figs/step0_momentum_entropy.png", dpi=150); f2g.savefig("docs/figs/step0_positions.png", dpi=150)
print("wrote docs/figs/step0_momentum_entropy.png, docs/figs/step0_positions.png")
