# config_offset.py — the correction of the positions (the configurational offset) in the equilibrium of hybrid Monte Carlo, against the
# one-loop prediction of docs/sscheme_plan.tex (eq. loopPE) and the parametrization of what remains by ν_*² (item L, eq. a2).
# For each line at rest (c_s², s_a, h/a) and scheme:
#   offset = (X_HMC/x − 1) − (X_quadrature/x − 1),   X = P, E,
# the HMC value from examples/eos_measure.jl (mean over the seeds), the quadrature value the single particle of the same scheme
# (examples/renorm_quadrature_sscheme.py; docs/figs/sscheme_quadrature_measure.csv).  With σ = p V̄/T = S_t/n and N_h = n̄ h² (the record),
#   one loop:   σ (P/p − 1) = (I_0 + I_0')/(2 N_h),   q σ (E/e − 1) = q I_0'/(2 N_h)          (I_0, I_0' at σ: docs/figs/i0_table_wc2.csv)
#   remainder:  r_P = σ offset_P − (I_0 + I_0')/(2 N_h),   r_E = q σ offset_E − q I_0'/(2 N_h),
#   item L:     r_P/(1 + I_0'/I_0) = r_E/(q I_0'/I_0) = −2 a_2 ν_*²,   ν_* = −I_0/(2 a_1 N_h),   a_1 = π/(2 sin(2π/5)) (Wendland C2).
# Runs: split $SCRATCH/sph/eos_split/eosS_split_*_v0; the full one-loop S-scheme $SCRATCH/sph/eos_sscheme/eosS_ss_*_v0 (s_a = 54, 541);
# SR_SETS adds sets as name:dir:prefix:quadmodel,...
#   python3 examples/config_offset.py   → docs/figs/config_offset.csv, docs/figs/config_offset_nu2.png
import os, re, glob, math, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "docs", "figs")
SCR = os.environ.get("SCRATCH", "")
SETS = [("split", os.path.join(SCR, "sph", "eos_split"), "eosS_split_", "step5-du0"),
        ("sscheme-full", os.path.join(SCR, "sph", "eos_sscheme"), "eosS_ss_", "sscheme"),
        ("measure", os.path.join(SCR, "sph", "eos_measure"), "eosS_ssm_", "sscheme-m"),            # β = D/2 + 1
        ("measure-b", os.path.join(SCR, "sph", "eos_measure"), "eosS_ssmb_", "sscheme-m@2.0"),      # β = b = 2 (c_s² = 1/4)
        # with the positional counterterm in the Hamiltonian (SPH_POSITIONAL=1, a_2 = 0; campaign eos_positional): the offset is what one loop leaves
        ("split+omega", os.path.join(SCR, "sph", "eos_positional"), "eosP_split_", "step5-du0"),
        ("measure+omega", os.path.join(SCR, "sph", "eos_positional"), "eosP_ssm_c05_", "sscheme-m"),
        ("measure+omega", os.path.join(SCR, "sph", "eos_positional"), "eosP_ssm_c025_", "sscheme-m@2.0"),
        ("exact+omega", os.path.join(SCR, "sph", "eos_positional"), "eosP_ssx_c05_", "exact")]          # the exact form (X5)
for x in filter(None, os.environ.get("SR_SETS", "").split(",")):
    SETS.append(tuple(x.split(":")))

tab = np.loadtxt(os.path.join(FIGS, "i0_table_wc2.csv"), delimiter=",", skiprows=1)
lsig = np.log(tab[:, 0])
def I0(s):  return np.interp(np.log(s), lsig, tab[:, 1])
def I0p(s): return np.interp(np.log(s), lsig, tab[:, 2])
a1 = math.pi / (2 * math.sin(2 * math.pi / 5))

quad = {}
for r in csv.DictReader(open(os.path.join(FIGS, "sscheme_quadrature_measure.csv"))):
    quad[(float(r["c2"]), int(float(r["s"])), float(r["v"]), r["model"])] = (float(r["E"]), float(r["p"]))

def read(f):
    d = {}
    for line in open(f):
        m = re.match(r"\s*(\w+)\s*=\s*(.+)", line)
        if m:
            try: d[m.group(1)] = float(m.group(2))
            except ValueError: pass
    return d

def ms(xs):
    xs = np.array([x for x in xs if math.isfinite(x)])
    return xs.mean(), (xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan"))

rows = []
for scheme, root, pre, qm in SETS:
    for d in sorted(glob.glob(os.path.join(root, pre + "*_v0"))):
        m = re.search(r"_c(\d+)_s(\d+)_ha([\d.]+)_v0$", d)
        if not m: continue
        c2 = float("0." + m.group(1)[1:]);  sa = int(m.group(2));  ha = float(m.group(3))
        ds = [read(f) for f in glob.glob(os.path.join(d, "*seed*.txt"))];  ds = [x for x in ds if "P_over_p_minus1" in x]
        if not ds or (c2, sa, 0.0, qm) not in quad: continue
        sig = np.mean([x["sigma"] for x in ds]);  Nh = np.mean([x["nbar_h2"] for x in ds]);  q = 1 / c2
        P, dP = ms([x["P_over_p_minus1"] for x in ds]);  E, dE = ms([x["E_over_e_minus1"] for x in ds])
        qE, qP = quad[(c2, sa, 0.0, qm)]
        oP, oE = P - qP, E - qE
        i0, i0p = I0(sig), I0p(sig)
        predP, predE = (i0 + i0p) / (2 * Nh * sig), i0p / (2 * Nh * sig)
        nu = -i0 / (2 * a1 * Nh)
        rP = sig * oP - (i0 + i0p) / (2 * Nh);  rE = q * sig * oE - q * i0p / (2 * Nh)
        rows.append(dict(c2=c2, s_a=sa, ha=ha, scheme=scheme, n=len(ds), sigma=sig, Nh=Nh, I0=i0, I0p=i0p, nu=nu,
                         P_hmc=P, dP=dP, P_quad=qP, offP=oP, predP=predP, E_hmc=E, dE=dE, E_quad=qE, offE=oE, predE=predE,
                         rP=rP, drP=sig * dP, rE=rE, drE=q * sig * dE,
                         yP=rP / (1 + i0p / i0), dyP=sig * dP / abs(1 + i0p / i0), yE=rE / (q * i0p / i0), dyE=q * sig * dE / abs(q * i0p / i0)))

print("percent; offset = HMC − quadrature (the single particle of the same scheme); one loop = the prediction of the positions")
print(f"{'c_s²':>5} {'s_a':>4} {'h/a':>4} {'scheme':>13} {'σ':>7} {'ν_*':>6} | {'P: HMC':>12} {'quad':>7} {'offset':>7} {'1-loop':>7} | "
      f"{'E: HMC':>12} {'quad':>7} {'offset':>7} {'1-loop':>7} | {'r_P/(1+I′/I)':>14} {'r_E/(qI′/I)':>14}")
for r in sorted(rows, key=lambda r: (-r["c2"], r["s_a"], r["ha"], r["scheme"])):
    print(f"{r['c2']:5.2f} {r['s_a']:4d} {r['ha']:4.1f} {r['scheme']:>13} {r['sigma']:7.2f} {r['nu']:6.3f} | "
          f"{100*r['P_hmc']:+6.2f}±{100*r['dP']:4.2f} {100*r['P_quad']:+7.3f} {100*r['offP']:+7.2f} {100*r['predP']:+7.2f} | "
          f"{100*r['E_hmc']:+6.2f}±{100*r['dE']:4.2f} {100*r['E_quad']:+7.3f} {100*r['offE']:+7.2f} {100*r['predE']:+7.2f} | "
          f"{r['yP']:+7.3f}±{r['dyP']:5.3f} {r['yE']:+7.3f}±{r['dyE']:5.3f}")

A2 = float(os.environ.get("SR_A2", "0.091"))
print("\nthe EOS error after each correction, percent: (0) HMC, (1) − single particle, (2) − positions at one loop, (3) − a_2 term (a_2 = %g)" % A2)
for sch in ("split", "measure", "measure-b", "sscheme-full", "split+omega", "measure+omega", "exact+omega"):
    rr = sorted([r for r in rows if r["scheme"] == sch], key=lambda r: (-r["c2"], r["s_a"], r["ha"]))
    if not rr: continue
    print(f"  {sch}:")
    for r in rr:
        sig, nu, i0, i0p = r["sigma"], r["nu"], r["I0"], r["I0p"]
        P2, E2 = r["offP"] - r["predP"], r["offE"] - r["predE"]
        P3 = P2 + 2 * A2 * nu**2 * (1 + i0p / i0) / sig;  E3 = E2 + 2 * A2 * nu**2 * (i0p / i0) / sig
        r.update(P2=P2, P3=P3, E2=E2, E3=E3)
        print(f"    c_s² {r['c2']:.2f} s_a {r['s_a']:4d} h/a {r['ha']:.0f} ν_* {nu:.3f} | P {100*r['P_hmc']:+6.2f} {100*r['offP']:+6.2f} {100*P2:+6.3f} {100*P3:+6.3f} ±{100*r['dP']:.2f}"
              f" | E {100*r['E_hmc']:+6.2f} {100*r['offE']:+6.2f} {100*E2:+6.3f} {100*E3:+6.3f} ±{100*r['dE']:.2f}")

out = os.path.join(FIGS, "config_offset.csv")
with open(out, "w") as f:
    keys = sorted(set().union(*[r.keys() for r in rows]), key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 99)
    f.write(",".join(keys) + "\n")
    for r in rows: f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
print("written", out)

try:
    import matplotlib;  matplotlib.use("Agg");  import matplotlib.pyplot as plt
    a2 = float(os.environ.get("SR_A2", "0.091"))            # the fit of the s_a ≥ 54 points (P and E; docs/sscheme_plan.tex)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    panels = (("yP", "dyP", "pressure",
               r"$y_P=[\sigma\,\delta P-\frac{I_0+I_0'}{2N_h}]/(1+I_0'/I_0)$"),
              ("yE", "dyE", "energy",
               r"$y_E=[q\sigma\,\delta E-\frac{q\,I_0'}{2N_h}]/(q\,I_0'/I_0)$"))
    for ax, (y, dy, name, lab) in zip(axs, panels):
        for c2, mk in ((0.5, "o"), (0.25, "s")):
            for ha, col in ((2.0, "C0"), (3.0, "C1")):
                for sc, fill in (("split", True), ("sscheme-full", False)):          # the figure: split and the full form
                    rr = [r for r in rows if r["c2"] == c2 and r["ha"] == ha and r["scheme"] == sc and r["s_a"] >= 54]
                    if rr:
                        ax.errorbar([r["nu"]**2 for r in rr], [r[y] for r in rr], [r[dy] for r in rr], fmt=mk, color=col,
                                    mfc=(col if fill else "none"), ms=6,
                                    label=f"{'split' if fill else 'S-scheme, full one loop'}: $c_s^2$={c2}, $h/a$={ha:g}")
                rr = [r for r in rows if r["c2"] == c2 and r["ha"] == ha and r["scheme"] == "split" and r["s_a"] < 54]
                if rr:
                    ax.errorbar([r["nu"]**2 for r in rr], [r[y] for r in rr], [r[dy] for r in rr], fmt=mk, color="0.55", mfc="0.55", ms=5,
                                label=("$s_a=16$ (not in the fit)" if (c2, ha) == (0.5, 2.0) else None))
        xx = np.linspace(0, 0.8, 50);  ax.plot(xx, -2 * a2 * xx, "k--", lw=0.9, label=f"$-2a_2\\nu_*^2$, fitted $a_2$ = {a2:g}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$\nu_*^2$, $\nu_*=-I_0(\sigma)/(2a_1N_h)$ (stiff modes per particle)")
        ax.set_ylabel(lab);  ax.yaxis.set_tick_params(labelleft=True)
        ax.set_title(f"{name}: offset of the positions beyond one loop", fontsize=10)
    axs[0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("eos_split and eos_sscheme at rest, 8 seeds (examples/config_offset.py): "
                 r"$\delta X = X_{\rm HMC}/x-1-(X_{\rm quad}/x-1)$; if beyond one loop is $a_2\nu_*^2$, $y_P=y_E=-2a_2\nu_*^2$", fontsize=8.5)
    fig.tight_layout();  fig.savefig(os.path.join(FIGS, "config_offset_nu2.png"), dpi=130)
    print("written", os.path.join(FIGS, "config_offset_nu2.png"))
except Exception as ex:
    print("no figure:", ex)
