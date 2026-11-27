# i0_table.py — item H of docs/sscheme_plan.tex: the loop integral of the positions,
#   I_0(σ) = h² ∫ d²k/(2π)² ln S(k),   1/S(k) = (1 − W̃)[1 + (2σ − 1) W̃],   I_0' = dI_0/d ln σ,
# with W̃(k) the Fourier (Hankel) transform of the kernel of src/sph.jl (support 2h, W̃(0) = 1).  In x = kh,
#   I_0 = (1/2π) ∫_0^∞ x ln S dx,   I_0' = −(σ/π) ∫_0^∞ x W̃/[1 + (2σ − 1) W̃] dx.
# W̃ by Gauss–Legendre in q = r/h on [0, 2] (W is a polynomial there, so the sum is exact up to the oscillation of J_0; 4000 nodes keep it
# to ~1e-15 absolute to x = 400).  The integrals in x: composite Gauss–Legendre to X, and the tail beyond X from the power law
# W̃ ≈ c_W x^(−n_W) (its mean; c_W measured on [X/2, X]).  Checks: the reference values of the plan (Wendland C2: I_0 = −0.979, −3.645,
# −3.874, −7.743, −13.288 at σ = 3.89, 16.4, 18, 60, 179) and, with 1/S = 1 + 2ΠW̃ − (2Π − 𝒦)W̃², Π = σ + 3/2, 𝒦 = 3/2, the values of
# partition.tex (−2.03, −4.14, −13.43 at σ = 3.89, 16.4, 179).  Also ν_* = −I_0/(2 a_1 N_h) per unit V/h² (N_h = h²/V̄), a_1 = π/(2 sin(2π/n_W)).
#   python3 examples/i0_table.py [wc2|wc4|wc6]   → docs/figs/i0_table_<kernel>.csv: sigma, I0, I0p (σ = 1e-8 to 1e5, 1201 points in ln σ)
import os, sys
import numpy as np
from scipy.special import j0

KERNELS = {   # W(q) h² for q = r/h < 2, and n_W (the power of the fall of W̃)
    "wc2": (lambda q: 7 / (4 * np.pi) * (1 - q / 2)**4 * (1 + 2 * q), 5),
    "wc4": (lambda q: 9 / (4 * np.pi) * (1 - q / 2)**6 * (1 + 3 * q + 35 * q**2 / 12), 7),
    "wc6": (lambda q: 78 / (28 * np.pi) * (1 - q / 2)**8 * (1 + 4 * q + 25 * q**2 / 4 + 4 * q**3), 9),
}

def make_Wt(name, nq=1200):
    W, nW = KERNELS[name]
    t, wt = np.polynomial.legendre.leggauss(nq)
    q = 1 + t;  wq = wt                                   # [0, 2]
    norm = 2 * np.pi * np.sum(wq * W(q) * q)
    def Wt(x):
        x = np.atleast_1d(x)
        out = np.empty_like(x, dtype=float)
        for i in range(0, len(x), 2000):
            xx = x[i:i + 2000]
            out[i:i + 2000] = 2 * np.pi * (j0(np.outer(xx, q)) * (wq * W(q) * q)).sum(axis=1)
        return out / norm
    return Wt, nW, norm

def xgrid(X, panel=0.5, n=16):
    t, w = np.polynomial.legendre.leggauss(n)
    edges = np.arange(0, X + 1e-12, panel)
    xs = (edges[:-1, None] + panel / 2 * (1 + t[None, :])).ravel()
    ws = np.tile(w * panel / 2, len(edges) - 1)
    return xs, ws

class I0Table:
    def __init__(self, name="wc2", X=400.0):
        self.Wt, self.nW, self.norm = make_Wt(name)
        self.x, self.w = xgrid(X)
        self.W = self.Wt(self.x)
        m = self.x > X / 2                                  # the mean amplitude of the power law
        self.cW = np.mean(self.W[m] * self.x[m]**self.nW)
        self.X = X
        self.a1 = np.pi / (2 * np.sin(2 * np.pi / self.nW))

    def I0(self, sigma, form="gauss"):
        W, x, w, X, n, c = self.W, self.x, self.w, self.X, self.nW, self.cW
        if form == "gauss":
            lnS = -np.log1p(-W) - np.log1p((2 * sigma - 1) * W)
            # tail: ln S ≈ −(2σ − 2) W̃ = −(2σ − 2) c x^(−n);  (1/2π) ∫_X^∞ x · that = −(2σ − 2) c X^(2−n)/(2π (n − 2))
            tail = -(2 * sigma - 2) * c * X**(2 - n) / (2 * np.pi * (n - 2))
        else:                                               # partition.tex: 1/S = 1 + 2ΠW̃ − (2Π − 𝒦)W̃²
            P, K = sigma + 1.5, 1.5
            lnS = -np.log(1 + 2 * P * W - (2 * P - K) * W * W)
            tail = -2 * P * c * X**(2 - n) / (2 * np.pi * (n - 2))
        return np.sum(w * x * lnS) / (2 * np.pi) + tail

    def I0p(self, sigma):
        W, x, w, X, n, c = self.W, self.x, self.w, self.X, self.nW, self.cW
        f = -2 * W / (1 + (2 * sigma - 1) * W)
        tail = -2 * c * X**(2 - n) / (2 * np.pi * (n - 2))
        return sigma * (np.sum(w * x * f) / (2 * np.pi) + tail)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "wc2"
    t = I0Table(name)
    print(f"kernel {name}: ∫W = {t.norm:.12f} (should be 1), W̃ x^{t.nW} → {t.cW:.2f} (mean on [{t.X/2:g}, {t.X:g}]), a_1 = {t.a1:.6f}")
    t2 = I0Table(name, X=200.0)
    if name == "wc2":
        ref = [(3.89, -0.979), (16.4, -3.645), (18, -3.874), (60, -7.743), (179, -13.288)]
        print("reference values (plan, item H):")
        for s, r in ref:
            print(f"  σ = {s:6.2f}: I_0 = {t.I0(s):+.4f}  (reference {r:+.3f}; X = 200: {t2.I0(s):+.4f})")
        print("partition.tex form, 1/S = 1 + 2ΠW̃ − (2Π − 𝒦)W̃²:")
        for s, r in [(3.89, -2.03), (16.4, -4.14), (179, -13.43)]:
            print(f"  σ = {s:6.2f}: {t.I0(s, 'partition'):+.3f}  (reference {r:+.2f})")
    # I_0' against a central difference
    for s in (3.89, 54 / 3, 541 / 3):
        h = 1e-4
        fd = (t.I0(s * np.exp(h)) - t.I0(s * np.exp(-h))) / (2 * h)
        print(f"  σ = {s:7.2f}: I_0' = {t.I0p(s):+.8f}, central difference {fd:+.8f}")
    sig = np.exp(np.linspace(np.log(1e-8), np.log(1e5), 1201))    # 1/S > 0 for every σ > 0 (W̃ ≤ 1); I₀ → const, I₀′ ~ σ ln σ as σ → 0
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figs", f"i0_table_{name}.csv")
    with open(out, "w") as f:
        f.write("sigma,I0,I0p\n")
        for s in sig:
            f.write(f"{s:.10g},{t.I0(s):.12g},{t.I0p(s):.12g}\n")
    print("written", out)
