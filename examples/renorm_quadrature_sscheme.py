# renorm_quadrature_sscheme.py — item A of docs/sscheme_plan.tex: the single particle of the S-scheme (state counting at one loop,
# docs/renorm3.tex) by the exact quadrature of Z_1(T, v, V) with the label gas, against split (SPH_RENORM=split = the model step5-du0 of
# examples/renorm_quadrature_split.py) and the tree level, at rest and in a moving bath.
# Constant c_s^2, d = 2, p = A T_rel^n, n = 1 + 1/c_s^2, A = c_s^2 T_scale^(-n).
# The S-scheme (sscheme_plan.tex, eqs. label, state1, weight): at the target state (T_rel, v) of (e, m, V),
#   s = S_t - Δ_χ - Δ_ℓ,   Δ_χ = (D/2) ln(2π V γ w T_rel) + ½ ln a,   Δ_ℓ = 1 + ln V + ln γ - ln T_rel,   a = 1/c_s^2 - v^2,
#   T_a = T/𝒟,   v_a = [v(1 - 2ε_p) + ε_∥]/𝒟,   𝒟 = 1 - (1 + v^2) ε_p + v ε_∥,   ε_p = Δ_p/(S_t a),  ε_∥ = Δ_θ/S_t;
# the domain 𝒟 > 0, |v_a| < 1.  The measure e^s ds d^2m with H = e is, in the variables (e, m), e^{s - ln T_a} de d^2m, and the
# quadrature runs over the target's (T, v_x, v_y): de d^2m = |∂(e,M)/∂(T,v)| (M/v) dT d^2v (as the split script does for u).
# The quadrature runs in (ln T, the rapidity vector) (observables_r): in (T, v) the renormalized weights rise toward the light cone
# and the Gaussian grid of the split script misses a tail worth a few 1e-4 in E at s_a = 54, v = 0.8.
# Reported for each model as in renorm_quadrature_split.py: the relative errors of E, M, p, the violation R of the density-frame
# relation M = v (E + pV), and μ/T of the labels (all zero for the exact model).  For the S-scheme also: the weight of the tree-level
# particle that lies outside the S-scheme's domain (plan, item A: "Ask" if more than a few percent at s_a = 16), the weight of the
# S-scheme within 3 of its boundary in S_t, and its largest |v_a|.
#   python3 examples/renorm_quadrature_sscheme.py [c_s^2 ...]   (default 0.5 0.25; SR_SA=16,54,541  SR_V=0,0.8  SR_MODELS=tree,step5-du0,sscheme)
# SR_CSV=<file>: one row per model and state, c2,s,v,model,E,M,p,R,mu (docs/figs/sscheme_quadrature.csv).
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from renorm_quadrature_split import Split

D = 3   # d + 1


class SScheme(Split):
    def sstate(self, T, v, V, reshape=False):
        """the S-scheme at the target state (bare T, |v|): label s, T_a, v_a, 𝒟, S_t, and the target energy and M/v.
        reshape = True: the momentum part X = ½ ln(1 − c_s² v²) of the counterterm multiplied by g(S_t) = 1 − h, h = 3u² − 2u³, u = b/S_t
        (c_s² = 1/2; the minimum label at S_t = b for every v; a change of order 1/S_t²)"""
        n, A, q = self.n, self.A, 1 / self.c2
        g = 1 / np.sqrt(1 - v * v);  Tr = g * T;  a = q - v * v
        p = A * Tr**n;  w = n * p;  St = g * (w / Tr) * V
        Dchi = D / 2 * np.log(2 * np.pi * V * g * w * Tr) + 0.5 * np.log(a)
        Dell = 1 + np.log(V) + np.log(g) - np.log(Tr)
        s = St - Dchi - Dell
        Dp = D / 2 * (2 + q - v * v) + v * v * (1 - v * v) / a - (1 + v * v)
        Dth = D / 2 * v - v * (1 - v * v) / a + v
        if reshape:
            b = D / 2 + 1 - (1 - 2 * self.c2);  u = b / St
            h = 3 * u**2 - 2 * u**3;  dh = -6 * u**2 * (1 - u)
            X = 0.5 * np.log(a / q);  Xth = -v * (1 - v * v) / a
            s = s + h * X
            Dp = Dp - dh * X * a + v * h * Xth
            Dth = Dth - dh * v * X - h * Xth
        ep, epar = Dp / (St * a), Dth / St
        den = 1 - (1 + v * v) * ep + v * epar
        Ta = T / den;  va = (v * (1 - 2 * ep) + epar) / den
        e = V * (w * g * g - p);  Mv = V * w * g * g
        return s, Ta, va, den, St, e, Mv

    def label_limit(self):
        """the hard limit on the label (sscheme_plan.tex, E, option 1; c_s^2 = 1/2): s_b = s_min(rest) + ½ ln(q/(q−1)), s_min(rest)
        the smallest label of a particle at rest (at c_s^2 = 1/2 it does not depend on V)"""
        if not hasattr(self, "_sb"):
            assert abs(self.c2 - 0.5) < 1e-12, "the hard limit is defined for c_s^2 = 1/2 only"
            q = 1 / self.c2;  n = self.n
            St = np.exp(np.linspace(np.log(0.5), np.log(20.0), 200001))
            T = (St / (n * self.A))**(1 / (n - 1))                   # at rest, V = 1: S_t = n A T^(n-1)
            s = self.sstate(T, np.zeros_like(T), 1.0)[0]
            self._sb = s[np.isfinite(s)].min() + 0.5 * np.log(q / (q - 1))
        return self._sb

    def jac_e(self, T, v, V):
        """|∂(e, M)/∂(T, v)| at fixed V, M = |m|"""
        n, A = self.n, self.A
        g = 1 / np.sqrt(1 - v * v)
        e = A * V * (g * T)**n * (n * g * g - 1)
        Mv = n * A * V * T**n * g**(n + 2);  M = Mv * v
        eT, MT = n * e / T, n * M / T
        ev = n * A * V * T**n * g**(n + 2) * v * ((n + 2) * g * g - 1)
        Mvv = n * A * V * T**n * g**(n + 2) * (1 + (n + 2) * g * g * v * v)
        return np.abs(eT * Mvv - ev * MT)

    def logweight(self, name, T, vx, vy, Tb, vb, V):
        if name not in ("sscheme", "sscheme-g", "sscheme-cut"):
            return super().logweight(name, T, vx, vy, Tb, vb, V)
        v = np.hypot(vx, vy)
        ok = (T > 0) & (v < 1 - 1e-9)
        T = np.where(ok, T, 1.0);  v = np.where(ok, v, 0.0)
        s, Ta, va, den, St, e, Mv = self.sstate(T, v, V, reshape=(name == "sscheme-g"))
        ok = ok & (den > 0) & (np.abs(va) < 1)
        if name == "sscheme-cut":
            ok = ok & (s > self.label_limit())
        Ta = np.where(ok, Ta, 1.0)
        mx = Mv * vx
        lw = s - np.log(Ta) - (e - vb * mx) / Tb + np.log(np.where(ok, self.jac_e(T, v, V) * Mv, 1.0))
        return np.where(ok, lw, -np.inf), np.where(ok, e, 0.0), np.where(ok, mx, 0.0)

    def logweight_r(self, name, lT, ex, ey, Tb, vb, V):
        """the weight per d ln T d^2η, η the rapidity vector (v = tanh|η| η̂): d T d^2v = T tanh θ/(θ cosh^2 θ) d ln T d^2η, θ = |η|.
        In (T, v) the weight density of the renormalized models rises toward v -> 1 (the volume element), so there is no peak to centre
        the grid on; in (ln T, η) there is."""
        th = np.hypot(ex, ey);  ths = np.where(th > 1e-12, th, 1.0)
        v = np.tanh(th);  c = np.where(th > 1e-12, ex / ths, 1.0);  sn = np.where(th > 1e-12, ey / ths, 0.0)
        T = np.exp(lT)
        lw, H, mx = self.logweight(name, T, v * c, v * sn, Tb, vb, V)
        jac = np.where(th > 1e-12, np.tanh(ths) / (ths * np.cosh(ths)**2), 1.0)
        return lw + lT + np.log(jac), H, mx

    def observables_r(self, name, s_a, vb, V=1.0, n=91, L=12.0):
        """as Split.observables, with the Gaussian grid in (ln T, η) around the maximum of the weight there"""
        from scipy.optimize import minimize
        nn, A = self.n, self.A
        g = 1 / np.sqrt(1 - vb * vb)
        Tr = (s_a / (g * nn * A * V))**(1 / (nn - 1));  Tb = Tr / g
        f = lambda x: -float(self.logweight_r(name, np.array([x[0]]), np.array([x[1]]), np.array([x[2]]), Tb, vb, V)[0][0])
        r = minimize(f, np.array([np.log(Tb), np.arctanh(vb), 0.0]), method="Nelder-Mead",
                     options=dict(xatol=1e-12, fatol=1e-13, maxiter=40000))
        xs = r.x;  hs = np.array([1e-3, 1e-3, 1e-3])
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
            lw, H, mx = self.logweight_r(name, P[0], P[1], P[2], Tb, vb, VV)
            m = lw.max();  wt = np.exp(lw - m)
            if avg:
                W = wt.sum();  out["E"] = (wt * H).sum() / W;  out["M"] = (wt * mx).sum() / W
                out["P"], out["wt"] = P, wt / W
            return np.log(wt.sum() * dxi**3 * np.prod(sig)) + m
        lz = lnZ(V, True)
        hV = 1e-4 * V;  dlz = (lnZ(V + hV) - lnZ(V - hV)) / (2 * hV)
        pV = Tb + Tb * V * dlz;  mu = -(np.log(V) + lz - V * dlz)
        p = A * Tr**nn;  w = nn * p
        Et = V * (w * g * g - p);  Pt = V * w * g * g * vb
        R = out["M"] / (vb * (out["E"] + pV)) - 1 if vb > 0 else np.nan
        return dict(E=out["E"] / Et - 1, M=(out["M"] / Pt - 1) if vb > 0 else np.nan, p=pV / (p * V) - 1, R=R, mu=mu,
                    grid=(out["P"], out["wt"]))



def extras(mdl, tree_grid, ss_grid, V=1.0):
    """the tree-level weight outside the S-scheme's domain; the S-scheme's weight within 3 of the boundary in S_t; its largest |v_a|"""
    out = {}
    for key, (P, wt) in (("tree", tree_grid), ("sscheme", ss_grid)):
        th = np.hypot(P[1], P[2]);  T = np.exp(P[0]);  v = np.tanh(th)
        okg = v < 1 - 1e-12
        s, Ta, va, den, St, e, Mv = mdl.sstate(np.where(okg, T, 1.0), np.where(okg, v, 0.0), V)
        inside = okg & (den > 0) & (np.abs(va) < 1)
        if key == "tree":
            out["outside"] = wt[~inside].sum()
        else:
            Sb = (1 - den) * St                       # 𝒟 = 1 - S_b/S_t: the boundary in S_t at this v
            out["near"] = wt[inside & (St < Sb + 3)].sum()
            out["vamax"] = np.abs(va[(wt > 1e-12 * wt.max()) & inside]).max()
    return out


if __name__ == "__main__":
    c2s = [float(x) for x in sys.argv[1:]] or [0.5, 0.25]
    sas = [float(x) for x in os.environ.get("SR_SA", "16,54,541").split(",")]
    vbs = [float(x) for x in os.environ.get("SR_V", "0,0.8").split(",")]
    models = os.environ.get("SR_MODELS", "tree,step5-du0,sscheme").split(",")
    NQ, LQ = int(os.environ.get("SR_NQ", "201")), float(os.environ.get("SR_LQ", "26"))   # converged to ~1e-6 (c_s^2 = 1/4, s_a = 54, v = 0.8)
    csv = os.environ.get("SR_CSV")
    if csv:
        with open(csv, "w") as f:
            f.write("c2,s,v,model,E,M,p,R,mu\n")
    for c2 in c2s:
        mdl = SScheme(c2)
        print(f"c_s^2 = {c2}  (split = step5-du0 = SPH_RENORM=split; quadrature in (ln T, rapidity), {NQ}^3 points, ±{LQ:g} sigma)")
        for s_a in sas:
            for vb in vbs:
                grids = {}
                for name in models:
                    try:
                        o = mdl.observables_r(name, s_a, vb, n=NQ, L=LQ)
                    except Exception as ex:
                        print(f"  s_a={s_a:6.1f} v={vb:.1f} {name:9s} failed: {ex}");  continue
                    grids[name] = o["grid"]
                    if csv:
                        with open(csv, "a") as f:
                            f.write(f"{c2},{s_a},{vb},{name},{o['E']},{o['M']},{o['p']},{o['R']},{o['mu']}\n")
                    extra = ""
                    if name == "sscheme" and "tree" in grids:
                        x = extras(mdl, grids["tree"], grids["sscheme"])
                        extra = f"  tree weight outside domain {x['outside']:.1e}  w(S_t<S_b+3) {x['near']:.1e}  max|v_a| {x['vamax']:.4f}"
                    print(f"  s_a={s_a:6.1f} v={vb:.1f} {name:9s}  E {o['E']:+.2e}  M {o['M']:+.2e}  p {o['p']:+.2e}"
                          f"  R {o['R']:+.2e}  mu/T {o['mu']:+.2e}{extra}", flush=True)
            print()
