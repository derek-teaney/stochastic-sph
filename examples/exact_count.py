"""exact_count.py — the exact number of states of one particle (docs/renorm2.tex, "With momentum"), for a constant speed of sound in d = 2.

The exact particle is a Poisson cloud of quanta in the volume V (renorm2: the massless gas for c_s^2 = 1/2, the mass spectrum rho(M) ~ M^(n-4)
for c_s^2 < 1/2), with the Laplace transform
    ln Z_t = alpha beta^0 (beta.beta)^(-nu),   nu = n/2,   n = 1 + 1/c_s^2,   alpha = A V   (p = A T_rel^n: at rest ln Z_t = pV/T).
Each term (ln Z_t)^N / N! inverts in closed form: (beta.beta)^(-lam) is the transform of the Riesz kernel on the forward cone of (e, m) in 2+1
dimensions, K(lam) (e^2 - m^2)^(lam - 3/2), K(lam) = 1/[2^(2 lam - 1) sqrt(pi) Gamma(lam) Gamma(lam - 1/2)], and each factor beta^0 is d/de.  The number
of states below the energy e at the momentum m (the N = 0 atom dropped, renorm2) is then
    G(e, m) = sum_{N>=1} alpha^N/N! K(N nu) d^(N-1)/de^(N-1) (e^2 - m^2)^(N nu - 3/2),
every term positive (Leibniz: the derivative of (e - m)^p (e + m)^p is a sum of positive falling factorials, since N nu - 3/2 >= N - 1).
The exact label is s = ln G - ln V - 1 (the label gas, renorm3.tex, "The accuracy of one loop"), and the correction to one loop is
    delta(S_t, v) = s_exact - S_1loop,   S_1loop = S_t - Delta_chi - Delta_ell (renorm3.tex, eq. scheme),
a function of the target entropy S_t and velocity v alone (scaling).  At large S_t, delta S_t -> c(a) (renorm3.tex, appendix, eq. c).

  python3 examples/exact_count.py check     the Laplace transform of G at one point; delta S_t against c(a) at S_t = 54, 541, 2000
  python3 examples/exact_count.py table     delta on a grid of (ln S_t, rapidity), c_s^2 = 1/2 and 1/4 -> docs/figs/exact_delta_c<..>.csv (SR_NPROC)
Needs mpmath."""
import sys, os
from mpmath import mp, mpf, gamma, loggamma, log, exp, pi, sqrt, cosh, tanh, binomial, ff, quad, inf, fsum
mp.dps = 40
D = 3;  d = 2

class EOS:
    def __init__(self, cs2, A=1):
        self.cs2 = mpf(cs2);  self.q = 1 / self.cs2;  self.n = 1 + self.q;  self.A = mpf(A);  self.nu = self.n / 2

def lnK(lam):
    return -((2 * lam - 1) * log(2) + log(pi) / 2 + loggamma(lam) + loggamma(lam - mpf(1) / 2))     # pi^((D-2)/2), D = 3

def ln_dpow(N, p, e, m):
    """ln of d^(N-1)/de^(N-1) [(e - m)^p (e + m)^p] (Leibniz; every term positive for p >= N - 1)"""
    k = N - 1;  terms = []
    for j in range(k + 1):
        t = log(binomial(k, j)) + log(ff(p, j)) + (p - j) * log(e - m) + log(ff(p, k - j)) + (p - k + j) * log(e + m) if m > 0 else None
        if m > 0:
            terms.append(t)
    if m == 0:                                     # (e^2)^p: d^k/de^k e^(2p) = ff(2p, k) e^(2p - k)
        return log(ff(2 * p, k)) + (2 * p - k) * log(e)
    mx = max(terms)
    return mx + log(fsum(exp(t - mx) for t in terms))

def lnG(e, m, V, c, tol=mpf(10)**-30):
    """ln G(e, |m|) of the exact particle, by the series in N (summed in logs until the terms fall below tol of the sum)"""
    alpha = c.A * V;  out = [];  N = 1;  mx = -inf;  peaked = False
    while True:
        lam = N * c.nu;  p = lam - mpf(3) / 2
        t = N * log(alpha) - loggamma(N + 1) + lnK(lam) + ln_dpow(N, p, e, m)
        out.append(t)
        if t > mx: mx = t
        elif t < mx + log(tol) and N > 5:
            break
        N += 1
        if N > 5000: raise RuntimeError("lnG: series did not converge")
    return mx + log(fsum(exp(t - mx) for t in out)), N

def target(St, v, V, c):
    """the target state of entropy S_t, velocity v, volume V: (e, |m|, T_rel)"""
    g = 1 / sqrt(1 - v * v)
    Tr = (St / (g * c.n * c.A * V))**(1 / (c.n - 1))          # S_t = gamma n A T_rel^(n-1) V
    p = c.A * Tr**c.n;  w = c.n * p
    return V * (w * g * g - p), V * w * g * g * v, Tr

def S1loop(St, v, V, c):
    g = 1 / sqrt(1 - v * v);  e, m, Tr = target(St, v, V, c)
    w = c.n * c.A * Tr**c.n;  a = c.q - v * v
    Dchi = mpf(D) / 2 * log(2 * pi * V * g * w * Tr) + log(a) / 2
    Dell = 1 + log(V) + log(g) - log(Tr)
    return St - Dchi - Dell

def delta(St, v, c, V=mpf(1)):
    e, m, Tr = target(mpf(St), mpf(v), V, c)
    lg, N = lnG(e, m, V, c)
    return lg - log(V) - 1 - S1loop(mpf(St), mpf(v), V, c), N

def c_of_a(c, v):
    q = c.q;  a = q - v * v;  l3 = 2 * q * (q - 1);  l4 = 2 * q * (q - 1) * (3 * q + 4 * d - 7)
    return -(d * d - 1) / mpf(8) - (d - 2) * (q + d - 1) / (2 * a) + l4 / (8 * a**2) - 5 * l3**2 / (24 * a**3)

def check():
    # 1. the Laplace transform of dG/de at one point: int de d^2m e^(-beta^0 e + b.m) g = Z_t - 1 (rest frame of beta, m integrated in angle)
    for cs2 in ('0.5', '0.25'):
        c = EOS(cs2);  V = mpf(1) / 4;  b0 = mpf('1.3')            # ln Z_t = alpha b0^(1 - n) at rest
        lnZt = c.A * V * b0**(1 - c.n)
        # int_0^inf de e^(-b0 e) g(e, m) d^2m = b0 int de e^(-b0 e) G(e, m) d^2m;  d^2m = 2 pi m dm, 0 <= m <= e
        f = lambda e: exp(-b0 * e) * quad(lambda m: 2 * pi * m * exp(lnG(e, m, V, c)[0]), [0, e]) if e > 0 else mpf(0)
        mp.dps = 20
        I = b0 * quad(f, [0, 1, 5, 20, inf])
        mp.dps = 40
        print(f"c_s^2 = {cs2}: Laplace transform of G: {float(I):.10f}, Z_t - 1 = {float(exp(lnZt) - 1):.10f}")
    # 2. delta S_t -> c(a)
    for cs2 in ('0.5', '0.25'):
        c = EOS(cs2)
        for v in ('0', '0.5', '0.8'):
            for St in (16, 54, 541, 2000):
                dl, N = delta(St, mpf(v), c)
                print(f"c_s^2 = {cs2} v = {v:>3} S_t = {St:5d}: delta = {float(dl):+.6e}  delta*S_t = {float(dl * St):+.5f}  c(a) = {float(c_of_a(c, mpf(v))):+.5f}  (N <= {N})")

def _pt(args):
    cs2, x, t = args
    mp.dps = 40
    dl, N = delta(mpf(x).__class__(exp(mpf(x))), tanh(mpf(t)), EOS(cs2))
    return float(dl)

def table():
    """delta on a grid of x = ln S_t (0.5 to 200, 141 points) and the rapidity theta (0 to 4, 41 points), both c_s^2; SR_NPROC workers"""
    import numpy as np
    from multiprocessing import Pool
    npr = int(os.environ.get("SR_NPROC", "8"))
    for cs2 in ('0.5', '0.25'):
        lS = np.linspace(np.log(0.5), np.log(200.0), 141);  th = np.linspace(0.0, 4.0, 41)
        pts = [(cs2, float(x), float(t)) for x in lS for t in th]
        with Pool(npr) as pool:
            vals = pool.map(_pt, pts, chunksize=8)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figs", f"exact_delta_c{cs2.replace('.', '')}.csv")
        with open(out, "w") as f:
            f.write("lnSt,theta,delta\n")
            for (c_, x, t), v in zip(pts, vals):
                f.write(f"{x:.12g},{t:.12g},{v:.15g}\n")
        print("written", out, flush=True)

if __name__ == "__main__":
    {"check": check, "table": table}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
