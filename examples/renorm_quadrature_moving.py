# renorm_quadrature_moving.py — exact single-particle quadrature of Z_1(T, v, V) for a moving equilibrium
# (docs/partition.tex, "The label s_eff, the measure and the Hamiltonian"; docs/partition_v2.tex, Sects. "Equilibrium in motion"
# and "The counterterms: formal structure").  Conformal fluid in d = 2: p = T^3/3, s = T^2 (a = 1), c_s^2 = 1/2, K0 = 3/2, alpha = 5.
# Per particle, with the label gas (ln Z_N/N = ln V + 1 + ln Z_1):
#   E = <H>,  P = <m_x>,  pV = T + T V d(ln Z_1)/dV,  mu/T = -(ln V + ln Z_1 - V d ln Z_1/dV),
#   R = P/(v (E + pV)) - 1  (the density-frame / Lorentz identity, 0 for the target),
# against the target E_t = V[(e+p)γ^2 - p], P_t = V (e+p)γ^2 v, pV = p V, mu = 0.  Models, all with the Liouville measure
# flat in their label:
#   tree     label s, H = e_t(s, m, V)
#   renorm   SPH_RENORM (the measure route): label u, H = e_t(u, m, V), weight e^{u - Δs_lab(u) + ln(1 - α/2u)}
#   Heff     Step 5': label s, H = e_t(s, m, V) + T_a [Δs - ln J], Δs = (α/2) ln s + K + (1/2) ln(1 - c_s^2 v_a^2),
#            ln J = -∂Δs/∂s|_{m,V} (lowest order), T_a, v_a the target's at (s, m, V)
#   split    Step 5: the label part in the measure (as renorm), the sound term (1/2) ln(1 - c_s^2 v_a^2) - ln J in H, times T_a
#   python3 examples/renorm_quadrature_moving.py
import numpy as np
from scipy.optimize import minimize

c2, d, K0 = 0.5, 2, 1.5
al = d + 3.0
K = 1.0 + 0.5 * np.log(16 * np.pi**3)             # the constant of Δs: 1 + c_χ, a = 1 (mu = 0)

def target_state(S, mx, my, V):
    """vectorized target state of a particle (entropy S, momentum m, volume V): E, T, vx, vy, |v|"""
    S = np.asarray(S, float);  mx = np.asarray(mx, float);  my = np.asarray(my, float)
    sd = S / V;  mm = np.hypot(mx, my) / V
    q = np.where(sd > 0, mm / np.where(sd > 0, sd, 1.0)**1.5, np.inf)
    lo = np.zeros_like(q);  hi = np.ones_like(q)       # solve y (1 - y^2)^(-1/4) = q
    for _ in range(70):
        mid = 0.5 * (lo + hi);  f = mid * (1 - mid * mid)**-0.25 - q
        lo = np.where(f < 0, mid, lo);  hi = np.where(f < 0, hi, mid)
    v = 0.5 * (lo + hi);  g = 1 / np.sqrt(1 - v * v)
    Tr = np.sqrt(np.where(sd > 0, sd, 1.0) / g)
    E = V * Tr**3 * (g * g - 1.0 / 3.0);  T = Tr / g
    mn = np.hypot(mx, my);  ux = np.where(mn > 0, mx / np.where(mn > 0, mn, 1), 1.0);  uy = np.where(mn > 0, my / np.where(mn > 0, mn, 1), 0.0)
    return E, T, v * ux, v * uy, v

def ds_full(S, mx, my, V, sound=True):
    _, _, _, _, v = target_state(S, mx, my, V)
    return al / 2 * np.log(S) + K + (0.5 * np.log(1 - c2 * v * v) if sound else 0.0)

def model_HW(name, S, mx, my, V):
    """energy H and the extra log weight (beyond the label S) of a model; -inf where invalid"""
    E, T, vx, vy, v = target_state(S, mx, my, V)
    bad = S <= al / 2 + 1e-9
    if name == "tree":
        return np.where(S > 0, E, 0.0), np.where(S > 0, 0.0, -np.inf)
    if name == "renorm":
        w = -(al / 2 * np.log(np.where(bad, 1, S)) + K) + np.log(np.where(bad, 1, 1 - al / (2 * S)))
        return np.where(bad, 0.0, E), np.where(bad, -np.inf, w)
    if name == "split":          # Step 5: label part in the measure (as renorm), the sound term in H with the particle's T
        w = -(al / 2 * np.log(np.where(bad, 1, S)) + K) + np.log(np.where(bad, 1, 1 - al / (2 * S)))
        h = 1e-4 * S
        snd = lambda SS: 0.5 * np.log(1 - c2 * target_state(SS, mx, my, V)[4]**2)
        H = E + T * (0.5 * np.log(1 - c2 * v * v) + (snd(S + h) - snd(S - h)) / (2 * h))
        return np.where(bad, 0.0, H), np.where(bad, -np.inf, w)
    sound = (name == "Heff")
    h = 1e-4 * S
    dds = (ds_full(S + h, mx, my, V, sound) - ds_full(S - h, mx, my, V, sound)) / (2 * h)
    H = E + T * (ds_full(np.where(bad, al, S), mx, my, V, sound) + dds)          # -ln J = +∂Δs/∂s
    return np.where(bad, 0.0, H), np.where(bad, -np.inf, 0.0)

def exponent(name, x, T, vb, V):
    S, mx, my = x
    H, w = model_HW(name, S, mx, my, V)
    return S + w - (H - vb * mx) / T

def grid_setup(name, T, vb, V):
    Tr = T / np.sqrt(1 - vb * vb);  g = 1 / np.sqrt(1 - vb * vb)
    x0 = np.array([g * Tr**2 * V, 0.0, 0.0]);  x0[1] = Tr**3 * g * g * vb * V
    f = lambda x: -float(exponent(name, (np.array([x[0]]), np.array([x[1]]), np.array([x[2]])), T, vb, V)[0])
    r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-13, maxiter=20000))
    xs = r.x;  hs = np.array([1e-3 * xs[0], 1e-3 * max(abs(xs[1]), xs[0]**1.5), 1e-3 * xs[0]**1.5])
    Hm = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei = np.eye(3)[i] * hs[i];  ej = np.eye(3)[j] * hs[j]
            Hm[i, j] = (f(xs + ei + ej) - f(xs + ei - ej) - f(xs - ei + ej) + f(xs - ei - ej)) / (4 * hs[i] * hs[j])
    lam, U = np.linalg.eigh(Hm)
    return xs, U, 1 / np.sqrt(lam)

def lnZ_and_avgs(name, T, vb, V, grid, n=61, L=9.0):
    xs, U, sig = grid
    xi = np.linspace(-L, L, n);  dxi = xi[1] - xi[0]
    A, B, C = np.meshgrid(xi, xi, xi, indexing="ij")
    P = xs[:, None] + U @ (np.vstack([A.ravel(), B.ravel(), C.ravel()]) * sig[:, None])
    S, mx, my = P
    H, w = model_HW(name, S, mx, my, V)
    ex = S + w - (H - vb * mx) / T
    ex = np.where(np.isfinite(ex), ex, -np.inf);  emax = ex.max()
    wt = np.exp(ex - emax);  Zr = wt.sum() * dxi**3 * np.prod(sig)
    return np.log(Zr) + emax, (wt * H).sum() / wt.sum(), (wt * mx).sum() / wt.sum()

def observables(name, s_a, vb, V=1.0):
    g = 1 / np.sqrt(1 - vb * vb);  Tr = np.sqrt(s_a / (g * V));  T = Tr / g
    grid = grid_setup(name, T, vb, V)
    lz, E, P = lnZ_and_avgs(name, T, vb, V, grid)
    hV = 1e-4 * V
    lzp = lnZ_and_avgs(name, T, vb, V + hV, grid)[0];  lzm = lnZ_and_avgs(name, T, vb, V - hV, grid)[0]
    dlz = (lzp - lzm) / (2 * hV)
    pV = T + T * V * dlz;  mu = -(np.log(V) + lz - V * dlz)
    p = Tr**3 / 3;  w = Tr**3
    Et = V * (w * g * g - p);  Pt = V * w * g * g * vb;  pVt = p * V
    R = P / (vb * (E + pV)) - 1 if vb > 0 else np.nan
    return dict(E=E / Et - 1, P=(P / Pt - 1) if vb > 0 else np.nan, p=pV / pVt - 1, R=R, mu=mu, T=T)

if __name__ == "__main__":
    print("relative deviations from the target (per particle, with the label gas); R the identity violation, mu/T")
    for s_a in [16, 54, 541]:
        for vb in [0.0, 0.5, 0.8]:
            for name in ["tree", "renorm", "split", "Heff"]:
                o = observables(name, s_a, vb)
                print(f"s_a={s_a:4d} v={vb:.1f} {name:8s}  E {o['E']:+.2e}  M {o['P']:+.2e}  p {o['p']:+.2e}  R {o['R']:+.2e}  mu/T {o['mu']:+.2e}")
            print()
