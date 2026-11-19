"""statics_theory.py — the corrected theory of the positions and the particle variables used by the figures of docs/equilibrium.tex
(Appendix B, "The misses at sigma = 4"): everything at the bath temperature T0 of a run, with the exact single particle
(examples/single_particle.jl grid -> data/volfluct/single_particle_grid.csv).  The equation of state is the constant-c_s^2 one
(docs/dimension.tex), c_s^2 and T_scale from the run attributes (c_s^2 = 1/3, T_scale = 1 for files written before branch dimfix):

    e = (T/T_scale)^(1+1/c_s^2),  s = (1+c_s^2) e/T,  w = T s,  C_V = de/dT = s/c_s^2,  sigma(T0) = p V/T = c_s^2/(1+c_s^2) s(T0) V
    S(k) = 1 / (1 + 2 alpha_eff W~ (1 - W~/2) + 2 sigma_eff W~ (1 - W~))      (statics.tex, eq. Sexact)
    alpha_eff = -V^2 d^2lnZ/dV^2 (-> alpha = (d+1)/2), sigma_eff = V dlnZ/dV + V^2 d^2lnZ/dV^2 (-> sigma = p/(T nbar)); keys "alpha", "sig"
    r_m = <m_x^2|V> / (T w V),  r_s = Var(s|V) / (C_V V),  <s|V'> = s_off + b V'/V  (s_off, b in units of s(T0) V)

Import: from statics_theory import Wt, S_gauss, S_exact, exact, run_constants
"""
import csv, os
import numpy as np
from scipy.special import j0
from scipy.integrate import simpson

_here = os.path.dirname(os.path.abspath(__file__))
_rows = list(csv.DictReader(open(os.path.join(_here, "..", "data", "volfluct", "single_particle_grid.csv"))))
GRID_CS2 = float(_rows[0].get("cs2", 1 / 3))                    # the grid is for one c_s^2 (column absent before dimfix: 1/3)
_sig = np.array([float(r["sigma_T0"]) for r in _rows])
_tab = {key: np.array([float(r[key]) for r in _rows]) for key in
        ("f1", "f2", "var_s", "CV_V", "mx2", "Tw_V", "s_minus", "dsdV_V", "sa_T0", "tau2", "T2")}

_q = np.linspace(0, 2, 20001); _Wq = 7 / (4 * np.pi) * (1 - _q / 2) ** 4 * (1 + 2 * _q)
_KH = np.concatenate([np.linspace(0, 1, 1001)[:-1], np.linspace(1, 400, 40001)])
_WT = simpson(2 * np.pi * _q * _Wq * j0(np.outer(_KH, _q)), x=_q, axis=1)
KH, WT = _KH, _WT
def Wt(kh): return np.interp(kh, _KH, _WT)                         # Wendland C2, 2D, kh in units of h

def S_gauss(W, sigma): return 1 / (1 + 3 * W * (1 - W / 2) + 2 * sigma * W * (1 - W))
def S_exact(W, alpha, sig): return 1 / (1 + 2 * alpha * W * (1 - W / 2) + 2 * sig * W * (1 - W))

def exact(sigma):
    """The exact single-particle quantities at sigma(T0), interpolated in log sigma."""
    x = np.log(sigma); lx = np.log(_sig); ip = lambda y: float(np.interp(x, lx, y))
    f1 = sigma + ip(_tab["f1"] - _sig); f2 = ip(_tab["f2"])
    return dict(alpha=-f2, sig=f1 + f2, r_s=ip(_tab["var_s"] / _tab["CV_V"]),
                r_m=ip(_tab["mx2"] / _tab["Tw_V"]), r_tau=ip(_tab["tau2"] / _tab["T2"]),
                s_off=ip(_tab["s_minus"] / _tab["sa_T0"]), b=ip(_tab["dsdV_V"] / _tab["sa_T0"]))

def thermo(T, cs2=1 / 3, T_scale=1.0):
    """Continuum thermodynamics of the constant-c_s^2 equation of state at temperature T."""
    e = (T / T_scale) ** (1 + 1 / cs2); s = (1 + cs2) * e / T
    return dict(e=e, p=cs2 * e, s=s, w=T * s, CV=s / cs2)

def run_constants(A):
    """From the attributes of an HMC run: T0, V, sigma(T0), s(T0), w(T0), C_V(T0) and the exact quantities."""
    cs2 = float(A.get("cs2", 1 / 3)); Ts = float(A.get("T_scale", 1.0))
    if abs(cs2 - GRID_CS2) > 1e-12:
        raise ValueError(f"run has c_s^2 = {cs2}, the single-particle grid {GRID_CS2}: rerun examples/single_particle.jl grid")
    T0 = float(A["T0"]); V = float(np.prod(A["L"])) / float(A["N"]); th = thermo(T0, cs2, Ts)
    sig = cs2 / (1 + cs2) * th["s"] * V; d = exact(sig)
    d.update(T0=T0, V=V, sigma=sig, s=th["s"], w=th["w"], CV=th["CV"], saT0=th["s"] * V, cs2=cs2)
    return d

# The runs are labelled by the entropy per particle of the lattice start, s_a = s(T(S0)) V; the bath temperature T0 has the same mean
# energy, <e_a>(T0) = V e(T(S0)).  Along the grid (T = 1), with T ∝ e^(c_s^2/(1+c_s^2)) and s ∝ T^(1/c_s^2):
# T0/T(S0) = (eV/<e>)^(c_s^2/(1+c_s^2)), label s_a = s(T0)V (T(S0)/T0)^(1/c_s^2).  "ratio3" = (T0/T(S0))^(1/c_s^2) = s(T0)/S0
# (the cube at c_s^2 = 1/3; the name is kept for the plotting scripts).
_e = np.array([float(r["e_mean"]) for r in _rows]); _eV = np.array([float(r["eV"]) for r in _rows])
_rat = (_eV / _e) ** (GRID_CS2 / (1 + GRID_CS2))             # T0 / T(S0)
_label = _tab["sa_T0"] / _rat ** (1 / GRID_CS2)
def from_label(sa_label):
    """sigma(T0) and (T0/T(S0))^3 for a run labelled by its start's entropy per particle."""
    x = np.log(sa_label); lx = np.log(_label)
    r3 = float(np.interp(x, lx, _rat ** (1 / GRID_CS2))); return dict(sigma=float(np.interp(x, lx, _sig)), ratio3=r3)
