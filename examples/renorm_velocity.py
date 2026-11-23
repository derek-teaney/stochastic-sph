# renorm_velocity.py — the velocity of a particle, xdot = ∂H/∂m, near the light cone for the renormalized Hamiltonians
# (docs/partition.tex, "The velocity near the light cone"; docs/partition_v2.tex, "Implementation: the entropy representation").
# Conformal fluid in d = 2 (p = T^3/3, s = T^2, c_s^2 = 1/2, alpha = 5), V = 1, one particle with the code entropy s and momentum m.
# Prints (1 - xdot)/(1 - v), v the target velocity at the particle's target state (u, m), u = s + (alpha/2) ln s:
#   u only (SPH_RENORM), everything in Δs (the sound term too), the density-frame split (the sound term in Δm),
#   and the split with (1+K0) ln γ moved into Δm (a drag at large γ).
#   python3 examples/renorm_velocity.py
import numpy as np
from scipy.optimize import brentq

c2, al, K0 = 0.5, 5.0, 1.5

def state(S, m):
    """target state of a particle of entropy S and momentum m (V = 1): v, γ, T_rel, T = T_rel/γ, e"""
    def f(v):
        g = 1 / np.sqrt(1 - v * v);  return np.sqrt(S / g)**3 * g * g * v - m
    v = 0.0 if m == 0 else brentq(f, 0, 1 - 1e-15, xtol=1e-16)
    g = 1 / np.sqrt(1 - v * v);  Tr = np.sqrt(S / g)
    return v, g, Tr, Tr / g, Tr**3 * g * g - Tr**3 / 3

def e_t(S, m): return state(S, m)[4]
def u_of(s): return s + al / 2 * np.log(s)

def H_uonly(s, m): return e_t(u_of(s), m)
def H_alls(s, m):
    v = state(s, m)[0];  return e_t(u_of(s) + 0.5 * np.log(1 - c2 * v * v), m)
def H_split(s, m):
    v, g, Tr, T, e = state(s, m);  dm = 0.0 if v == 0 else T / (2 * v) * np.log(1 - c2 * v * v)
    return e_t(u_of(s), m + dm)
def H_drag(s, m):
    v, g, Tr, T, e = state(s, m)
    dm = 0.0 if v == 0 else T / v * ((1 + K0) * np.log(g) + 0.5 * np.log(1 - c2 * v * v))
    return e_t(u_of(s) - (1 + K0) * np.log(g), m + dm)

def xdot(H, s, m):
    h = m * 1e-6
    try:
        return (H(s, m + h) - H(s, m - h)) / (2 * h)
    except ValueError:          # the shifted state leaves the domain (entropy or momentum out of range)
        return np.nan

schemes = [("u only", H_uonly), ("all in Δs", H_alls), ("split (Δm)", H_split), ("ln γ in Δm", H_drag)]
print("(1 - xdot)/(1 - v)")
for s in [5, 16, 54, 541]:
    print(f"s = {s}  (u = {u_of(s):.1f})")
    for v in [0.5, 0.9, 0.99, 0.999, 0.9999]:
        g = 1 / np.sqrt(1 - v * v);  m = np.sqrt(u_of(s) / g)**3 * g * g * v
        print(f"  v = {v:<7}" + "".join(f"  {n}: {(1 - xdot(H, s, m)) / (1 - v):7.4f}" for n, H in schemes))
