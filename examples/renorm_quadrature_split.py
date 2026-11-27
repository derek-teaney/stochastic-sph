# renorm_quadrature_split.py — Step 5 for any constant c_s^2 against the Legendre Hamiltonian, by the exact single-particle quadrature
# (docs/partition.tex, "Step 5 for any c_s^2: a plan for the comparison with the Legendre transform").
# Constant c_s^2, d = 2, ConstantCs2EOS: p = A T_rel^n, n = 1 + 1/c_s^2, A = c_s^2 T_scale^(-n).  N = 1 - 2 c_s^2.
# The counterterm of the Legendre script, written at the target state of the label u and the momentum m,
#   δlnZ_1 = Δs_lab(u) + Δs_dyn,   Δs_lab = (5/2 - N) ln u + C - k c_s^2 ln(nA),   Δs_dyn = N ln(γV) + ½ ln(1 - c_s^2 v^2).
# Models (all with the label gas):
#   tree     label u, measure e^u du d^2m, H = e_t(u, m, V)
#   label    the label part only: measure e^{u - Δs_lab(u)} J_lab du d^2m, J_lab = 1 - (5/2 - N)/u, H = e_t  (= SPH_RENORM at c_s^2 = 1/2)
#   step5    the label part in the measure, H = e_t + T_a [Δs_dyn + ∂Δs_dyn/∂u|_{m,V}]  (the last term is -ln J_dyn at lowest order)
#   step5-du0  the same without the term ∂Δs_dyn/∂u (step5_plan.tex, A: keep it only if it changes the residual at s_a >= 54)
#   legendre the Legendre Hamiltonian (examples/legacy/renorm_quadrature_legendre.py, imported unchanged)
# The quadrature for tree/label/step5 runs over the target's (T, v_x, v_y): du d^2m = |∂(u,M)/∂(T,v)| (M/v) dT d^2v.
# Also reported for step5: the weight within 3 of the label boundary u = 5/2 - N, and the largest |dx/dt| = |∂H/∂m| on the grid
# (and the weight beyond the light cone).
#   python3 examples/renorm_quadrature_split.py [c_s^2 ...]     (default 0.5 0.25; SR_SA, SR_V as in the Legendre script)
# SR_MODELS: the models to run (default tree,label,step5,legendre; also step5-du0).  SR_CSV=<file>: write the rows of the first model
# as c2,s,v,E,M,p,R (s = s_a), the single-particle reference of examples/eos_split.py (docs/figs/split_quadrature.csv, step5-du0).
import os, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))   # the Legendre model (examples/legacy/)
from renorm_quadrature_legendre import Model as LegendreModel


class Split:
    def __init__(self, c2, T_scale=1.0):
        self.leg = LegendreModel(c2, T_scale)
        self.c2 = c2;  self.n = 1 + 1 / c2;  self.A = c2 * T_scale**(-self.n);  self.k = 2 + 1.5 / c2
        self.N = 1 - 2 * c2
        self.blab = 2.5 - self.N                                   # coefficient of ln u in Δs_lab; the label boundary
        self.Clab = self.leg.C - self.k * c2 * np.log(self.n * self.A)

    def target(self, T, v, V):
        """the target's u, M/v, e_t and the entries of ∂(u,M)/∂(T,v) at (T, |v|)"""
        n, A = self.n, self.A
        g = 1 / np.sqrt(1 - v * v)
        u = n * A * V * g**n * T**(n - 1)
        Mv = n * A * V * T**n * g**(n + 2)
        e = A * V * (g * T)**n * (n * g * g - 1)
        uT = n * (n - 1) * A * V * g**n * T**(n - 2)
        uv = n * n * A * V * T**(n - 1) * g**(n + 2) * v            # = ∂M/∂T
        Mvv = n * A * V * T**n * g**(n + 2) * (1 + (n + 2) * g * g * v * v)
        return u, Mv, e, uT, uv, Mvv

    def ds_lab(self, u):
        return self.blab * np.log(u) + self.Clab

    def ds_dyn(self, v, V):
        g = 1 / np.sqrt(1 - v * v)
        return self.N * np.log(g * V) + 0.5 * np.log(1 - self.c2 * v * v)

    def dyn_energy(self, T, v, V, du=True):
        """T_a [Δs_dyn + ∂Δs_dyn/∂u|_{m,V}] at the target state (T, v)"""
        u, Mv, e, uT, uv, Mvv = self.target(T, v, V)
        det = uT * Mvv - uv * uv
        dvdu = -uv / det                                           # ∂v/∂u at fixed m (Cramer)
        g2 = 1 / (1 - v * v)
        dDdv = self.N * g2 * v - self.c2 * v / (1 - self.c2 * v * v)
        return T * (self.ds_dyn(v, V) + (dDdv * dvdu if du else 0.0))

    def logweight(self, name, T, vx, vy, Tb, vb, V):
        if name == "legendre":
            lw, H, mx = self.leg.logweight(T, vx, vy, Tb, vb, V)
            return lw, H, mx
        v = np.hypot(vx, vy)
        ok = (T > 0) & (v < 1 - 1e-9)
        T = np.where(ok, T, 1.0);  v = np.where(ok, v, 0.0)
        u, Mv, e, uT, uv, Mvv = self.target(T, v, V)
        det = np.abs(uT * Mvv - uv * uv)
        mx = Mv * vx
        if name == "tree":
            lw = u;  H = e
        else:
            ok = ok & (u > self.blab)
            us = np.where(ok, u, 2 * self.blab + 1)
            lw = u - self.ds_lab(us) + np.log(1 - self.blab / us)
            H = e + (self.dyn_energy(T, v, V, name == "step5") if name.startswith("step5") else 0.0)
        lw = lw - (H - vb * mx) / Tb + np.log(np.where(ok, det * Mv, 1.0))
        return np.where(ok, lw, -np.inf), np.where(ok, H, 0.0), np.where(ok, mx, 0.0)

    def xdot(self, T, vx, vy, V, h=1e-6, du=True):
        """|∂H/∂m| at fixed (u, V) for step5: v + ∂(T Δ)/∂m, by the chain rule through (T, |v|) (along m)"""
        v = np.hypot(vx, vy)
        u, Mv, e, uT, uv, Mvv = self.target(T, v, V)
        det = uT * Mvv - uv * uv
        f = lambda TT, vv: self.dyn_energy(TT, vv, V, du)
        fT = (f(T * (1 + h), v) - f(T * (1 - h), v)) / (2 * h * T)
        fv = (f(T, v + h * (1 - v)) - f(T, v - h * (1 - v))) / (2 * h * (1 - v))
        return v + (-uv * fT + uT * fv) / det                      # ∂f/∂|m| at fixed u

    def check_split(self, V=0.7):
        """Δs_lab(u) + Δs_dyn must equal the Legendre script's δlnZ_1 at the target state"""
        T = np.array([0.8, 2.0, 3.0]);  v = np.array([0.0, 0.6, 0.95])
        g = 1 / np.sqrt(1 - v * v)
        dl = 2.5 * np.log(V) + 2.5 * np.log(g) + self.k * np.log(g * T) + 0.5 * np.log(1 - self.c2 * v * v) + self.leg.C
        u = self.target(T, v, V)[0]
        return np.max(np.abs(self.ds_lab(u) + self.ds_dyn(v, V) - dl))

    def observables(self, name, s_a, vb, V=1.0, n=91, L=12.0):
        nn, A = self.n, self.A
        g = 1 / np.sqrt(1 - vb * vb)
        Tr = (s_a / (g * nn * A * V))**(1 / (nn - 1));  Tb = Tr / g
        f = lambda x: -float(self.logweight(name, np.array([x[0]]), np.array([x[1]]), np.array([x[2]]), Tb, vb, V)[0][0])
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
        out = {}
        def lnZ(VV, avg=False):
            lw, H, mx = self.logweight(name, P[0], P[1], P[2], Tb, vb, VV)
            m = lw.max();  wt = np.exp(lw - m)
            lz = np.log(wt.sum() * dxi**3 * np.prod(sig)) + m
            if avg:
                W = wt.sum();  out["E"] = (wt * H).sum() / W;  out["M"] = (wt * mx).sum() / W
                if name in ("label", "step5", "step5-du0"):
                    ok = np.isfinite(lw) & (P[0] > 0)
                    vv = np.hypot(P[1], P[2]);  u = np.where(ok, self.target(np.where(ok, P[0], 1.0), np.where(ok, vv, 0.0), VV)[0], np.inf)
                    out["edge"] = wt[u < self.blab + 3].sum() / W
                    # the edge of the grid: the weight on the outer shell, to see that the quadrature holds the particle
                if name.startswith("step5"):
                    sig_w = wt > 1e-12 * wt.max()
                    xd = np.where(sig_w, self.xdot(np.where(sig_w, P[0], 1.0), np.where(sig_w, P[1], 0.0), np.where(sig_w, P[2], 0.0), VV, du=(name == "step5")), 0.0)
                    out["xdot"] = np.abs(xd).max();  out["fast"] = wt[np.abs(xd) >= 1].sum() / W
            return lz
        lz = lnZ(V, True)
        hV = 1e-4 * V;  dlz = (lnZ(V + hV) - lnZ(V - hV)) / (2 * hV)
        pV = Tb + Tb * V * dlz;  mu = -(np.log(V) + lz - V * dlz)
        p = A * Tr**nn;  w = nn * p
        Et = V * (w * g * g - p);  Pt = V * w * g * g * vb
        R = out["M"] / (vb * (out["E"] + pV)) - 1 if vb > 0 else np.nan
        o = dict(E=out["E"] / Et - 1, M=(out["M"] / Pt - 1) if vb > 0 else np.nan, p=pV / (p * V) - 1, R=R, mu=mu)
        o.update({k: out[k] for k in ("edge", "xdot", "fast") if k in out})
        return o


if __name__ == "__main__":
    c2s = [float(x) for x in sys.argv[1:]] or [0.5, 0.25]
    sas = [float(x) for x in os.environ.get("SR_SA", "16,54,541").split(",")]
    vbs = [float(x) for x in os.environ.get("SR_V", "0,0.8").split(",")]
    models = os.environ.get("SR_MODELS", "tree,label,step5,legendre").split(",")   # also step5-du0
    csv = os.environ.get("SR_CSV")
    if csv:
        with open(csv, "w") as f:
            f.write("c2,s,v,E,M,p,R\n")
    for c2 in c2s:
        mdl = Split(c2)
        print(f"c_s^2 = {c2}  (N = {mdl.N:+.3f}, label boundary u = {mdl.blab:.2f}; check of the split: {mdl.check_split():.1e})")
        for s_a in sas:
            for vb in vbs:
                for name in models:
                    try:
                        o = mdl.observables(name, s_a, vb)
                    except Exception as ex:
                        print(f"  s_a={s_a:6.1f} v={vb:.1f} {name:8s} failed: {ex}");  continue
                    if csv and name == models[0]:
                        with open(csv, "a") as f:
                            f.write(f"{c2},{s_a},{vb},{o['E']},{o['M']},{o['p']},{o['R']}\n")
                    extra = ""
                    if "edge" in o:
                        extra += f"  w(u<b+3) {o['edge']:.1e}"
                    if "xdot" in o:
                        extra += f"  max|xdot| {o['xdot']:.4f}  w(|xdot|>=1) {o['fast']:.1e}"
                    print(f"  s_a={s_a:6.1f} v={vb:.1f} {name:9s}  E {o['E']:+.2e}  M {o['M']:+.2e}  p {o['p']:+.2e}"
                          f"  R {o['R']:+.2e}  mu/T {o['mu']:+.2e}{extra}", flush=True)
            print()
