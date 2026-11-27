#!/usr/bin/env python3
"""Check of the saddle-point expansion of renorm2.tex, Sect. "The saddle point
for a general equation of state", for the constant-cs2 fluid in d=2.

  ln Z = V Phi(beta),  Phi = p beta^0 = A beta^0 (beta.beta)^(-n/2),  n = 1 + 1/cs2

Exact:   G(e,m) = sum_{N>=1} alpha^N/(N! 2 pi Gamma(nN-1)) d_e^(N-1) (e^2-m^2)^((nN-3)/2),
         alpha = A V (the atom N=0 dropped); every term is positive (Leibniz).
Saddle:  ln G = S_t - (1/2) ln det(2 pi V chi) - ln beta^0 + c(v)/S_t + ...
         with c = (C_g + C_h) S_t / V from the cubic and quartic vertices.

Prints (ln G_exact - one loop) * S_t against c(v), from the tensor formula and
from the closed form.  Then, for a two-component p = T^3 + 0.7 T^5 (cs2 from 1/2 to
1/4), the tensor contraction against the collapsed form in a = 1/cs2 - v^2.
Needs mpmath and sympy.  Usage: renorm_saddle_check.py [n]
"""
import sys, itertools
import sympy as sp
from mpmath import mp, mpf, gamma, factorial, binomial, ff, log, pi, sqrt

mp.dps = 40
n = int(sys.argv[1]) if len(sys.argv) > 1 else 5      # cs2 = 1/(n-1); n = 5 is cs2 = 1/4
D = 3                                                 # d + 1
alpha = mpf(1)                                        # A V; drops out of c(v) by scaling

# ---------------------------------------------------------------- exact series
def dpow(e, m, p, j):
    """d^j/de^j (e^2 - m^2)^p, as a sum of positive terms."""
    return sum(binomial(j, i) * ff(p, i) * (e - m)**(p - i) * ff(p, j - i) * (e + m)**(p - j + i)
               for i in range(j + 1))

def G_exact(e, m):
    tot, N = mpf(0), 1
    while True:
        t = alpha**N / (factorial(N) * 2 * pi * gamma(n * N - 1)) * dpow(e, m, mpf(n * N - 3) / 2, N - 1)
        tot += t
        if N > 5 and t < tot * mpf(10)**(-30):
            return tot
        N += 1

# ------------------------------------------------- derivatives of Phi (sympy)
l0, l1, l2 = sp.symbols('l0 l1 l2', real=True)        # beta_mu = (-beta^0, b_x, b_y)
lam = [l0, l1, l2]
Phi = float(alpha) * (-l0) * (l0**2 - l1**2 - l2**2)**sp.Rational(-n, 2)
h = -sp.log(-l0)                                      # the 1/beta^0 of G = int^e g

def tensor(f, rank):
    return {idx: sp.lambdify(lam, sp.diff(f, *[lam[i] for i in idx]), 'mpmath')
            for idx in itertools.product(range(D), repeat=rank)}

P1, P2, P3, P4 = (tensor(Phi, r) for r in (1, 2, 3, 4))
H1, H2 = tensor(h, 1), tensor(h, 2)
R = range(D)

def saddle(st, v):
    """Target point with entropy st and velocity v: returns e, m, one loop, c."""
    v = mpf(v); g = 1 / sqrt(1 - v * v)
    T = (mpf(st) / n / alpha / g**n)**(mpf(1) / (n - 1))          # ln Z = alpha g^n T^(n-1) = st/n
    b = (-1 / T, v / T, mpf(0))
    e, m = P1[(0,)](*b), P1[(1,)](*b)                             # P^mu = V Phi^mu
    chi = mp.matrix(D, D)
    for i in R:
        for j in R:
            chi[i, j] = P2[(i, j)](*b)
    Dl = chi**-1
    k3 = {i: f(*b) for i, f in P3.items()}
    W = [sum(k3[(a, x, y)] * Dl[x, y] for x in R for y in R) for a in R]
    A = sum(P4[(a, x, y, z)](*b) * Dl[a, x] * Dl[y, z] for a in R for x in R for y in R for z in R)
    B1 = sum(W[a] * Dl[a, x] * W[x] for a in R for x in R)
    B2 = sum(k3[(a, x, y)] * k3[(c, z, w)] * Dl[a, c] * Dl[x, z] * Dl[y, w]
             for a in R for x in R for y in R for c in R for z in R for w in R)
    Cg = A / 8 - B1 / 8 - B2 / 12
    h1 = [H1[(a,)](*b) for a in R]
    Ch = (-sum(H2[(a, x)](*b) * Dl[a, x] for a in R for x in R) / 2
          - sum(h1[a] * h1[x] * Dl[a, x] for a in R for x in R) / 2
          + sum(h1[a] * Dl[a, x] * W[x] for a in R for x in R) / 2)
    one_loop = st - log((2 * pi)**D * mp.det(chi)) / 2 + log(T)   # S_t - (1/2) ln det(2 pi chi) - ln beta^0
    return e, m, one_loop, Cg * st, (Cg + Ch) * st

def c_collapsed(q, q1, q2, v, d=2):
    """c = c_g + c_h for any equation of state, eq. (ccollapsed): q = 1/cs2, q1, q2 its d/dlnT derivatives."""
    a = q - mpf(v)**2
    l3 = 2 * q * (q - 1) - q1
    l4 = 2 * q * (q - 1) * (3 * q + 4 * d - 7) - (7 * q + 4 * d - 9) * q1 + q2
    return (-mpf(d * d - 1) / 8 - (d - 2) * (q + d - 1) / (2 * a) + l4 / (8 * a**2) - 5 * l3**2 / (24 * a**3))

def general_eos_check():
    """Tensor contraction from direct derivatives of p beta^0 against c_collapsed, for p = T^3 + 0.7 T^5."""
    Ts = sp.Symbol('T', positive=True)
    pT = Ts**3 + sp.Rational(7, 10) * Ts**5
    Phi2 = pT.subs(Ts, (l0**2 - l1**2 - l2**2)**sp.Rational(-1, 2)) * (-l0)
    Q2, Q3, Q4 = (tensor(Phi2, r) for r in (2, 3, 4))
    s = sp.diff(pT, Ts); q = Ts * sp.diff(s, Ts) / s; q1 = Ts * sp.diff(q, Ts); q2 = Ts * sp.diff(q1, Ts)
    print("\np = T^3 + 0.7 T^5, d = 2:   T_rel    v    cs2      c tensor     c collapsed")
    for Tr in ['0.5', '1.2', '3']:
        for v in ['0', '0.6', '0.9']:
            T, vv = mpf(Tr), mpf(v); g = 1 / sqrt(1 - vv * vv)
            b = (-g / T, g * vv / T, mpf(0))
            chi = mp.matrix(D, D)
            for i in R:
                for j in R:
                    chi[i, j] = Q2[(i, j)](*b)
            Dl = chi**-1
            k3 = {i: f(*b) for i, f in Q3.items()}
            W = [sum(k3[(x, y, z)] * Dl[y, z] for y in R for z in R) for x in R]
            A = sum(Q4[(x, y, z, w)](*b) * Dl[x, y] * Dl[z, w] for x in R for y in R for z in R for w in R)
            B1 = sum(W[x] * Dl[x, y] * W[y] for x in R for y in R)
            B2 = sum(k3[(x, y, z)] * k3[(c, e, f)] * Dl[x, c] * Dl[y, e] * Dl[z, f]
                     for x in R for y in R for z in R for c in R for e in R for f in R)
            Tl = T / g
            C = A / 8 - B1 / 8 - B2 / 12 - Tl**2 * Dl[0, 0] + Tl / 2 * sum(Dl[0, x] * W[x] for x in R)
            sV = g * mpf(sp.N(s.subs(Ts, sp.Rational(Tr)), 30))
            qq, qq1, qq2 = (mpf(sp.N(x.subs(Ts, sp.Rational(Tr)), 30)) for x in (q, q1, q2))
            print(f"                           {Tr:>5s} {v:>5s}  {float(1/qq):.4f}   {float(C*sV):+.8f}  {float(c_collapsed(qq, qq1, qq2, v)):+.8f}")

def c_closed(v):
    """c(v) = (C_g + C_h) S_t / V in closed form, d = 2, any n."""
    v2 = mpf(v)**2
    num = (-2 * n**4 + 27 * n**3 - 95 * n**2 + 117 * n - 47
           - (18 * n**3 - 93 * n**2 + 126 * n - 51) * v2 - 27 * (n - 1) * v2**2 + 9 * v2**3)
    return num / (24 * (n - 1 - v2)**3)

if __name__ == "__main__":
    sts = [10, 54, 541]
    print(f"n = {n}  (cs2 = 1/{n-1}),  d = 2")
    print("   v   " + "".join(f"  exact st={s:<4d}" for s in sts) + "   c tensor   c closed   c_g (ln g)")
    for v in ['0', '0.5', '0.8', '0.95']:
        row = []
        for st in sts:
            e, m, ol, cg, c = saddle(st, v)
            row.append((log(G_exact(e, m)) - ol) * st)
        print(f"{float(mpf(v)):5.2f}  " + "".join(f"  {float(r):+13.4f}" for r in row)
              + f"   {float(c):+8.4f}   {float(c_closed(v)):+8.4f}   {float(cg):+8.4f}")
        assert abs(c_closed(v) - c_collapsed(mpf(n - 1), 0, 0, v)) < mpf(10)**-25
    general_eos_check()
