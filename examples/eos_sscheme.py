# eos_sscheme.py — item F of docs/sscheme_plan.tex: the equilibrium of the S-scheme against split, by hybrid Monte Carlo (examples/eos_measure.jl).
# Reads the per-seed summaries <run>/<tag>_seed<k>.txt ("key = value") of the run sets
#   split:    $SCRATCH/sph/eos_split/eosS_split_<c>_s<sa>_ha<ha>_v<v>
#   S-scheme: $SCRATCH/sph/eos_sscheme/eosS_ss_...   (one loop at s_a = 54, 541; SPH_SSCHEME_REG unset there, i.e. the code of 17ec558)
#             $SCRATCH/sph/eos_sscheme16/eosS_ssr_... (regularized, s_a = 16)
# and prints, for each (c_s^2, s_a, h/a, v), the mean over the seeds and its error (std/sqrt(n)) of E/e − 1, P/p − 1 (rest) or T^yy/p − 1 (moving),
# χ_MM/target − 1, M/target − 1 and R (moving), and the acceptance.  Writes docs/figs/sscheme_eos.csv.
#   python3 examples/eos_sscheme.py
import os, glob, re, math
SCR = os.environ.get("SCRATCH", "")
SETS = [("split", os.path.join(SCR, "sph", "eos_split"), "eosS_split_"),
        ("sscheme", os.path.join(SCR, "sph", "eos_sscheme"), "eosS_ss_"),
        ("sscheme-reg", os.path.join(SCR, "sph", os.environ.get("SR_SS16", "eos_sscheme16b")), "eosS_ssr_")]   # eos_sscheme16b: with the S_t > b fix (3c1e0ad)
KEYS = ["E_over_e_minus1", "P_over_p_minus1", "Tyy_over_p_minus1", "chi_dMdv_over_target_minus1", "M_over_target_minus1", "R", "acceptance"]

def read(f):
    d = {}
    for line in open(f):
        m = re.match(r"\s*(\w+)\s*=\s*(.+)", line)
        if m:
            try: d[m.group(1)] = float(m.group(2))
            except ValueError: pass
    return d

def stats(xs):
    xs = [x for x in xs if math.isfinite(x)]
    if not xs: return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    if len(xs) < 2: return (m, float("nan"))
    return (m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)))

rows = {}
for scheme, root, pre in SETS:
    for d in sorted(glob.glob(os.path.join(root, pre + "*"))):
        m = re.search(r"_c(\d+)_s(\d+)_ha(\d)_v(\d+)$", d)
        if not m: continue
        c2 = float("0." + m.group(1)[1:]) if m.group(1).startswith("0") else float(m.group(1))
        sa, ha, v = int(m.group(2)), int(m.group(3)), (0.8 if m.group(4) == "08" else 0.0)
        fs = [f for f in glob.glob(os.path.join(d, "*seed*.txt"))]
        ds = [read(f) for f in fs];  ds = [x for x in ds if "E_over_e_minus1" in x]
        if not ds: continue
        rows[(c2, sa, ha, v, scheme)] = {k: stats([x.get(k, float("nan")) for x in ds]) for k in KEYS} | {"n": len(ds)}

def fmt(t, scale=100):
    m, e = t
    return "      --      " if not math.isfinite(m) else (f"{m*scale:+7.2f}±{e*scale:4.2f}" if math.isfinite(e) else f"{m*scale:+7.2f}     ")

out = []
print("percent; mean ± error over the seeds.  split | S-scheme (one loop at s_a ≥ 54, regularized at s_a = 16)")
for c2 in (0.5, 0.25):
    for v in (0.0, 0.8):
        keys = ["E_over_e_minus1", "P_over_p_minus1", "chi_dMdv_over_target_minus1"] if v == 0 else \
               ["E_over_e_minus1", "Tyy_over_p_minus1", "M_over_target_minus1", "R"]
        print(f"\nc_s^2 = {c2}, v = {v}:   " + "   ".join(f"{k[:16]:>31s}" for k in keys) + "   acceptance")
        for sa in (16, 54, 541):
            for ha in (2, 3):
                sp = rows.get((c2, sa, ha, v, "split"))
                ss = rows.get((c2, sa, ha, v, "sscheme")) or rows.get((c2, sa, ha, v, "sscheme-reg"))
                if not sp and not ss: continue
                cells = []
                for k in keys:
                    cells.append((fmt(sp[k]) if sp else " " * 14) + " | " + (fmt(ss[k]) if ss else " " * 14))
                acc = f"{sp['acceptance'][0]:.3f} | {ss['acceptance'][0]:.3f}" if sp and ss else ""
                print(f"  s_a = {sa:3d}, h/a = {ha}:  " + "   ".join(cells) + "   " + acc)
                for sch, r in (("split", sp), ("sscheme", ss)):
                    if r: out.append([c2, sa, ha, v, sch] + [r[k][i] for k in KEYS for i in (0, 1)])
fig = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figs", "sscheme_eos.csv")
with open(fig, "w") as f:
    f.write("c2,s_a,ha,v,scheme," + ",".join(f"{k},d_{k}" for k in KEYS) + "\n")
    for r in out: f.write(",".join(str(x) for x in r) + "\n")
print("\nwritten", fig)
