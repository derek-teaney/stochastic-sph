# renorm_quadrature_legendre.py — the effective Hamiltonian as the Legendre transform of the wanted free energy,
#   F_want(T, v, V) = F_t(T, v, V) + T δlnZ_1(T, v, V),   H_eff(s, m, V) = F_want + T s + v·m  at  s = -∂F_want/∂T, m = -∂F_want/∂v,
# checked by the exact single-particle quadrature in motion (docs/partition.tex, "The procedure" and "Constant speed of sound").
# Constant c_s^2 (any value), d = 2, the code's ConstantCs2EOS: p = A T_rel^n, n = 1 + 1/c_s^2, A = c_s^2 T_scale^(-n), w = n p.
#   F_want = -A V (γT)^n + T [ (5/2) ln V + (5/2) ln γ + k ln(γT) + ½ ln(1 - c_s^2 v^2) + C ],  k = 2 + 3/(2 c_s^2),
#   C = (3/2) ln(2π n A) - ½ ln c_s^2 + 1   (B = 0, mu = 0).
# Per particle, with the label gas: E = <H>, P = <m_x>, pV = T + T V ∂_V ln Z_1, mu/T, R = P/(v(E + pV)) - 1, against the target.
# The quadrature runs over the Legendre variables (T, v_x, v_y): ds d^2m = |∂(s,M)/∂(T,v)| (M/v) dT d^2v.
#   python3 examples/renorm_quadrature_legendre.py [c_s^2 ...]      (default 0.5)
import sys
import numpy as np
from scipy.optimize import minimize

class Model:
    def __init__(self, c2, T_scale=1.0):
        self.c2 = c2;  self.n = 1 + 1 / c2;  self.A = c2 * T_scale**(-self.n)
        self.k = 2 + 1.5 / c2
        self.C = 1.5 * np.log(2 * np.pi * self.n * self.A) - 0.5 * np.log(c2) + 1.0

    def legendre(self, T, v, V):
        """s, M/v, H, |∂(s,M)/∂(T,v)| at the Legendre variables (T, |v|), analytic"""
        c2, n, A, k = self.c2, self.n, self.A, self.k
        g = 1 / np.sqrt(1 - v * v);  q = 1 - c2 * v * v;  K = 2.5 + k
        dl = 2.5 * np.log(V) + 2.5 * np.log(g) + k * np.log(g * T) + 0.5 * np.log(q) + self.C
        F = -A * V * (g * T)**n + T * dl
        S = n * A * V * g**n * T**(n - 1) - dl - k
        Mv = n * A * V * T**n * g**(n + 2) - T * (K * g * g - c2 / q)
        H = F + T * S + v * (Mv * v)
        dSdT = n * (n - 1) * A * V * g**n * T**(n - 2) - k / T
        dSdv = n * n * A * V * T**(n - 1) * g**(n + 2) * v - K * g * g * v + c2 * v / q        # = dM/dT (Maxwell)
        dMdv = n * A * V * T**n * g**(n + 2) * (1 + (n + 2) * g * g * v * v) - T * (K * (g * g + 2 * g**4 * v * v) - c2 * (1 + c2 * v * v) / q**2)
        return S, Mv, H, np.abs(dSdT * dMdv - dSdv * dSdv)

    def logweight(self, T, vx, vy, Tb, vb, V):
        v = np.hypot(vx, vy)
        ok = (T > 0) & (v < 1 - 1e-9)
        T = np.where(ok, T, 1.0);  v = np.where(ok, v, 0.0)
        S, Mv, H, J = self.legendre(T, v, V)
        mx = Mv * vx
        ok = ok & (J > 0) & (Mv > 0)
        lw = S - (H - vb * mx) / Tb + np.log(np.where(ok, J * Mv, 1.0))
        return np.where(ok, lw, -np.inf), np.where(ok, H, 0.0), np.where(ok, mx, 0.0)

    def observables(self, s_a, vb, V=1.0, n=61, L=9.0):
        c2, nn, A = self.c2, self.n, self.A
        g = 1 / np.sqrt(1 - vb * vb)
        Tr = (s_a / (g * nn * A * V))**(1 / (nn - 1));  Tb = Tr / g          # target entropy of the particle = s_a
        f = lambda x: -float(self.logweight(np.array([x[0]]), np.array([x[1]]), np.array([x[2]]), Tb, vb, V)[0][0])
        r = minimize(f, np.array([Tb, vb, 0.0]), method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-13, maxiter=40000))
        xs = r.x;  hs = np.array([1e-3 * xs[0], 1e-4, 1e-4])
        Hm = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                ei = np.eye(3)[i] * hs[i];  ej = np.eye(3)[j] * hs[j]
                Hm[i, j] = (f(xs + ei + ej) - f(xs + ei - ej) - f(xs - ei + ej) + f(xs - ei - ej)) / (4 * hs[i] * hs[j])
        lam, U = np.linalg.eigh(Hm);  sig = 1 / np.sqrt(lam)
        xi = np.linspace(-L, L, n);  dxi = xi[1] - xi[0]
        Ag, Bg, Cg = np.meshgrid(xi, xi, xi, indexing="ij")
        P = xs[:, None] + U @ (np.vstack([Ag.ravel(), Bg.ravel(), Cg.ravel()]) * sig[:, None])
        def lnZ(VV, avg=False):
            lw, H, mx = self.logweight(P[0], P[1], P[2], Tb, vb, VV)
            m = lw.max();  wt = np.exp(lw - m)
            lz = np.log(wt.sum() * dxi**3 * np.prod(sig)) + m
            return (lz, (wt * H).sum() / wt.sum(), (wt * mx).sum() / wt.sum()) if avg else lz
        lz, E, Px = lnZ(V, True)
        hV = 1e-4 * V;  dlz = (lnZ(V + hV) - lnZ(V - hV)) / (2 * hV)
        pV = Tb + Tb * V * dlz;  mu = -(np.log(V) + lz - V * dlz)
        p = A * Tr**nn;  w = nn * p
        Et = V * (w * g * g - p);  Pt = V * w * g * g * vb
        R = Px / (vb * (E + pV)) - 1 if vb > 0 else np.nan
        return dict(E=E / Et - 1, P=(Px / Pt - 1) if vb > 0 else np.nan, p=pV / (p * V) - 1, R=R, mu=mu)

if __name__ == "__main__":
    c2s = [float(x) for x in sys.argv[1:]] or [0.5]
    for c2 in c2s:
        mdl = Model(c2)
        print(f"c_s^2 = {c2}  (N = 1 - 2 c_s^2 = {1 - 2 * c2:+.3f}):  relative deviations from the target, R, mu/T")
        for s_a in [54, 541]:
            for vb in [0.0, 0.5, 0.8]:
                try:
                    o = mdl.observables(s_a, vb)
                except Exception as ex:
                    print(f"  s_a={s_a:4d} v={vb:.1f}  failed: {ex}");  continue
                print(f"  s_a={s_a:4d} v={vb:.1f}  E {o['E']:+.2e}  M {o['P']:+.2e}  p {o['p']:+.2e}  R {o['R']:+.2e}  mu/T {o['mu']:+.2e}")
