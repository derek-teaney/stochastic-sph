"""Exact single-particle renormalization without momentum (docs/renorm2.tex, "A particle without momentum").

The label is s = ln G(e), with G the cumulative density of states, the inverse Laplace transform of
Z(beta)/beta, Z = exp(alpha beta^-k), k = n - 1 = 1/c_s^2:
    G(e) = sum_{N>=0} alpha^N e^{kN} / (N! Gamma(kN+1)),   H(s) = G^{-1}(e^s) (s > 0),  H = 0 (s <= 0).
1. ln G against one loop, s_t - ln G = 1/2 ln(2 pi k s_t) + c_k/s_t.
2. The exact H reproduces ln Z = alpha beta^-k by quadrature.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.special import gammaln, logsumexp
from scipy.integrate import quad
from scipy.optimize import brentq


def lnG(e, a, k, Nmax=3000):
    if e <= 0:
        return 0.0
    N = np.arange(1, Nmax)
    return logsumexp(np.append(N*np.log(a) + k*N*np.log(e) - gammaln(N+1) - gammaln(k*N+1), 0.0))


def H(s, a, k):
    return 0.0 if s <= 0 else brentq(lambda e: lnG(e, a, k) - s, 1e-12, 1e4)


a = 1.0
print("1. the label against one loop")
for k in (2, 4):
    for st in (4, 16, 54, 541):
        b = ((k+1)*a/st)**(1.0/k)          # target beta at entropy s_t
        e = k*a*b**(-k-1)                  # target energy
        d = st - lnG(e, a, k)
        one = 0.5*np.log(2*np.pi*k*st)
        print(f"  k={k} s_t={st:4d}  s_t-lnG={d:8.5f}  one loop={one:8.5f}  remainder={d-one:9.2e}  x s_t={(d-one)*st:7.4f}")
print("2. ln Z of the exact H")
for k in (2, 4):
    for beta in (0.3, 0.7):
        smax = 3*(k+1)*a*beta**(-k) + 40
        I = quad(lambda s: np.exp(s - beta*H(s, a, k)), 0, smax, limit=400)[0]
        print(f"  k={k} beta={beta}: ln Z_model={np.log(1+I):.8f}  target={a*beta**(-k):.8f}")
