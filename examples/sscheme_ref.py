#!/usr/bin/env python3
"""Reference implementation of the S-scheme maps for a constant speed of sound in d = 2 (docs/sscheme_plan.tex).

The particle carries the label s (flat Liouville measure, conserved by the ideal step) and the target entropy S_t.
Level 1:  s = S_t - Delta_chi - Delta_ell.        Level 2:  s = S_t - Delta_chi - Delta_ell + omega(sigma, V).
The script checks the closed forms for T_a, v_a, P_a against central differences of H(s, m, V), that the map from the
energy inverts the map from the label, and the identity that relates the scheme to the split scheme.
The functions state_m and from_label_m are the form with the logarithm of the entropy in the measure
(docs/sscheme_plan.tex, "Level 1, amended").
Needs mpmath.  At level 2 the loop integral I0 is replaced by a smooth stand-in: the checks test the chain rule only."""
from mpmath import mp, mpf, log, cosh, sinh, tanh, atanh, findroot, pi, sin, diff
mp.dps = 30
D = 3
class Consts:
    def __init__(c, cs2, A=1, h=mpf('0.8'), a2=0, level=1, nW=5):
        c.cs2 = mpf(cs2); c.q = 1/c.cs2; c.n = 1 + c.q; c.A = mpf(A)
        c.h = mpf(h); c.a2 = mpf(a2); c.level = level; c.a1 = pi/(2*sin(2*pi/nW))
def I0(sig):   return -mpf('1.9')*sig**mpf('0.42') + mpf('0.7')        # stand-in for the tabulated loop integral
def dI0(sig):  return diff(I0, sig)*sig                                  # dI0/dln sigma

def target_from_entropy(St, m, V, c):
    """(S_t, |m|, V) -> T_rel and rapidity of the target:  S_t = gamma s V,  m = V w gamma^2 v."""
    n, A = c.n, c.A
    if m == 0:
        return (St/(n*A*V))**(1/(n-1)), mpf(0)
    F = lambda lt, th: (cosh(th)*n*A*mp.e**((n-1)*lt)*V - St, V*n*A*mp.e**(n*lt)*cosh(th)*sinh(th) - m)
    lt0 = log(St/(n*A*V))/(n-1); T0 = mp.e**lt0
    lt, th = findroot(F, (lt0, atanh(min(mpf('0.9'), m/(V*n*A*T0**n + m)))))
    return mp.e**lt, th

def target_from_energy(e, m, V, c):
    n, A = c.n, c.A
    if m == 0:
        return (e/(V*(n-1)*A))**(1/n), mpf(0)
    F = lambda lt, th: (V*A*mp.e**(n*lt)*(n*cosh(th)**2 - 1) - e, V*n*A*mp.e**(n*lt)*cosh(th)*sinh(th) - m)
    lt, th = findroot(F, (log(e/(V*(n-1)*A))/n, atanh(m/e*mpf('0.8'))))
    return mp.e**lt, th

def state(Trel, th, V, c):
    """Everything at a target state: S_t, the label s, the energy, and T_a, v_a, P_a of the particle."""
    q, n, A = c.q, c.n, c.A
    g, v = cosh(th), tanh(th); a = q - v*v; T = Trel/g
    p = A*Trel**n; w = n*p; St = g*(w/Trel)*V
    D_chi = mpf(D)/2*log(2*pi*V*g*w*Trel) + log(a)/2                     # determinant term
    D_ell = 1 + log(V) + log(g) - log(Trel)                              # label term, ln(e beta^0 V)
    s = St - D_chi - D_ell
    Dp_chi, Dp_ell   = mpf(D)/2*(2 + q - v*v) + v*v*(1 - v*v)/a, -(1 + v*v)      # slopes along the pressure mode
    Dth_chi, Dth_ell = mpf(D)/2*v - v*(1 - v*v)/a, v                             # slopes along the comoving momentum
    Dp, Dth = Dp_chi + Dp_ell, Dth_chi + Dth_ell
    dtau, dlnV = mpf(D)/2*(2 + q) - 1, mpf(D)/2 + 1                      # dDelta/dtau = Dp + v Dth; explicit dDelta/dlnV
    if c.level == 2:
        sig = St/n; nu = -I0(sig)*V/(2*c.a1*c.h**2)
        om = c.a1*nu + c.a2*nu*nu; omp = c.a1 + 2*c.a2*nu
        om_sig, om_V = omp*nu*dI0(sig)/I0(sig), omp*nu                   # d omega/dln sigma |_V,  d omega/dln V |_sigma
        s += om
        Dp -= a*om_sig; Dth -= v*om_sig; dtau -= q*om_sig; dlnV -= om_sig + om_V
    ep, epar = Dp/(St*a), Dth/St
    den = 1 - (1 + v*v)*ep + v*epar                                      # T/T_a, and ds/dS_t at fixed m, V
    Ta = T/den; va = (v*(1 - 2*ep) + epar)/den
    Pa = (St/n - dlnV + dtau/n)*Ta/V                                     # constant cs2: p V gamma/T_rel = S_t/n
    return dict(St=St, s=s, e=V*(w*g*g - p), Ta=Ta, va=va, Pa=Pa, den=den, T=T, v=v, weight=s - log(Ta),
                D_chi=D_chi, D_ell=D_ell, g=g)

def from_label(s, m, V, c, St0):
    """(s, m, V) -> the state, by Newton's iteration on S_t; ds/dS_t at fixed m, V is den."""
    St = mpf(St0)
    for it in range(80):
        st = state(*target_from_entropy(St, m, V, c), V, c)
        d = (st['s'] - s)/st['den']; St -= d
        if abs(d) < mpf(10)**-25*St: break
    st = state(*target_from_entropy(St, m, V, c), V, c); st['iters'] = it + 1
    return st

def from_energy(e, m, V, c):
    """(e, m, V) -> the state.  No iteration beyond the target's inversion."""
    return state(*target_from_energy(e, m, V, c), V, c)

# ---- the logarithm of the entropy in the measure (docs/renorm3.tex, "The bounds, ..."; docs/sscheme_plan.tex, "Level 1, amended") ----
BETA = mpf(D)/2 + 1
def state_m(Trel, th, V, c, T0, factor=False):
    """s = S_t - F Delta_H with Delta_H = -(q-d) ln(T_rel/T0) + (1/2) ln(1 - cs2 v^2);  measure rho(s) = e^s s^-beta (1 - beta/s), s > beta.
    factor=False: F = 1 (plain form).  factor=True: F = (S_t + 2 beta)/(S_t + beta)."""
    q, n, A = c.q, c.n, c.A; d = D - 1
    g, v = cosh(th), tanh(th); a = q - v*v; T = Trel/g
    p = A*Trel**n; w = n*p; St = g*(w/Trel)*V
    DH = -(q - d)*log(Trel/T0) + log(a/q)/2
    F   = (St + 2*BETA)/(St + BETA) if factor else mpf(1)
    kap = 1 + BETA*DH/(St + BETA)**2 if factor else mpf(1)               # 1 - F' Delta_H
    s = St - F*DH
    Dp, Dth = -(q - d) + v*v*(1 - v*v)/a, -v*(1 - v*v)/a                 # slopes of Delta_H along the two modes
    ep, epar = F*Dp/(St*a), F*Dth/St
    den = kap - (1 + v*v)*ep + v*epar                                    # T/T_a, and ds/dS_t at fixed m, V
    Ta = T/den; va = (v*(kap - 2*ep) + epar)/den
    Pa = (kap*St - F*(q - d))/n*Ta/V
    lnrho = s - BETA*log(s) + log(1 - BETA/s) if s > BETA else -mp.inf
    return dict(St=St, s=s, e=V*(w*g*g - p), Ta=Ta, va=va, Pa=Pa, den=den, T=T, v=v, weight=lnrho - log(Ta), lnrho=lnrho, DH=DH, g=g)

def from_label_m(s, m, V, c, T0, St0, factor=False):
    St = mpf(St0)
    for it in range(80):
        st = state_m(*target_from_entropy(St, m, V, c), V, c, T0, factor)
        dd = (st['s'] - s)/st['den']; St -= dd
        if abs(dd) < mpf(10)**-25*St: break
    st = state_m(*target_from_entropy(St, m, V, c), V, c, T0, factor); st['iters'] = it + 1
    return st


if __name__ == "__main__":
    for level in (1, 2):
        for cs2 in ('0.5', '0.25'):
            c = Consts(cs2, level=level, a2=mpf('0.3') if level == 2 else 0)
            for (St0, m, V) in [(16, 0, 1), (16, 9, 1.3), (54, 40, 0.8), (541, 900, 1.1)]:
                m, V = mpf(m), mpf(V)
                ref = state(*target_from_entropy(mpf(St0), m, V, c), V, c)       # a state with target entropy St0
                s = ref['s']
                H = lambda ss, mm, VV: from_label(ss, mm, VV, c, St0)['e']
                hh = mpf(10)**-9; ds = hh*St0
                dHs = (H(s + ds, m, V) - H(s - ds, m, V))/(2*ds)
                dHm = (H(s, m + hh*(1+m), V) - H(s, m - hh*(1+m), V))/(2*hh*(1+m))
                dHV = (H(s, m, V*(1+hh)) - H(s, m, V*(1-hh)))/(2*V*hh)
                back = from_energy(ref['e'], m, V, c)['s']
                fwd = from_label(s, m, V, c, mpf(St0)*mpf('1.3'))
                errs = (abs(dHs/ref['Ta'] - 1), abs(dHm - ref['va']), abs(-dHV/ref['Pa'] - 1), abs(back - s), abs(fwd['St']/St0 - 1))
                print(f"level {level} cs2={cs2} S_t={St0:4d} m={float(m):6.1f} V={float(V):.1f}  s={float(s):9.4f} v={float(ref['v']):.3f} "
                      f"T_a/T={float(ref['Ta']/ref['T']):.5f} v_a={float(ref['va']):.5f} P_aV/T_a={float(ref['Pa']*V/ref['Ta']):9.4f} "
                      f"Newton steps={fwd['iters']}  errors: dH/ds {float(errs[0]):.0e} dH/dm {float(errs[1]):.0e} dH/dV {float(errs[2]):.0e} "
                      f"inverse {float(errs[3]):.0e} forward {float(errs[4]):.0e}")
    # the relation to the split scheme (level 1): Delta_chi + Delta_ell = b ln S_t + N ln(gamma V) + (1/2) ln(1 - cs2 v^2) + const
    for cs2 in ('0.5', '0.25'):
        c = Consts(cs2); N = 1 - 2*c.cs2; b = mpf(D)/2 + 1 - N; vals = []
        for (St0, m, V) in [(16, 9, 1.3), (54, 40, 0.8), (541, 900, 1.1)]:
            r = state(*target_from_entropy(mpf(St0), mpf(m), mpf(V), c), mpf(V), c)
            vals.append(r['D_chi'] + r['D_ell'] - b*log(r['St']) - N*log(r['g']*mpf(V)) - log(1 - c.cs2*r['v']**2)/2)
        print(f"cs2={cs2}: Delta - [b ln S_t + N ln(gamma V) + (1/2)ln(1 - cs2 v^2)] = {float(vals[0]):.12f}, spread {float(max(vals)-min(vals)):.0e}")

    # the logarithm of the entropy in the measure: derivatives of H, the inverse map, and the distance from the one-loop label
    for factor in (False, True):
        for cs2 in ('0.5', '0.25'):
            c = Consts(cs2)
            for (St0, m, V) in [(4, 2, 1), (16, 0, 1), (16, 9, 1.3), (54, 40, 0.8), (541, 900, 1.1)]:
                m, V = mpf(m), mpf(V); T0 = (mpf(St0)/(c.n*c.A))**(1/c.q)        # nominal state: S_t = St0 at rest, V = 1
                ref = state_m(*target_from_entropy(mpf(St0), m, V, c), V, c, T0, factor); s = ref['s']
                H = lambda ss, mm, VV: from_label_m(ss, mm, VV, c, T0, St0, factor)['e']
                hh = mpf(10)**-9; ds = hh*St0
                dHs = (H(s + ds, m, V) - H(s - ds, m, V))/(2*ds)
                dHm = (H(s, m + hh*(1+m), V) - H(s, m - hh*(1+m), V))/(2*hh*(1+m))
                dHV = (H(s, m, V*(1+hh)) - H(s, m, V*(1-hh)))/(2*V*hh)
                back = state_m(*target_from_energy(ref['e'], m, V, c), V, c, T0, factor)['s']
                one = state(*target_from_entropy(mpf(St0), m, V, c), V, c)['s']    # the one-loop label S_t - Delta
                rest = state(*target_from_entropy(mpf(St0), mpf(0), mpf(1), c), mpf(1), c)['s'] - (mpf(St0) - BETA*log(mpf(St0)))
                dist = (s - BETA*log(s) - one + rest)*St0                          # S_t [ (s - beta ln s) - (S_t - Delta) ], zero at the nominal state
                errs = (abs(dHs/ref['Ta'] - 1), abs(dHm - ref['va']), abs(-dHV/ref['Pa'] - 1), abs(back - s))
                print(f"measure{'-f' if factor else '  '} cs2={cs2} S_t={St0:4d} m={float(m):6.1f} V={float(V):.1f}  s={float(s):9.4f} v={float(ref['v']):.3f} "
                      f"T_a/T={float(ref['Ta']/ref['T']):.5f} v_a={float(ref['va']):.5f} P_aV/T_a={float(ref['Pa']*V/ref['Ta']):9.4f} weight={float(ref['weight']):10.4f} "
                      f"errors: dH/ds {float(errs[0]):.0e} dH/dm {float(errs[1]):.0e} dH/dV {float(errs[2]):.0e} inverse {float(errs[3]):.0e}  "
                      f"S_t x (distance from one loop) {float(dist):+.3f}")
    # Delta_chi + Delta_ell = (D/2 + 1) ln S_t + ln(T_rel^d/s) + (1/2) ln a + (D/2) ln 2 pi + 1, for any state
    for cs2 in ('0.5', '0.25'):
        c = Consts(cs2); vals = []
        for (St0, m, V) in [(16, 9, 1.3), (54, 40, 0.8), (541, 900, 1.1)]:
            Trel, th = target_from_entropy(mpf(St0), mpf(m), mpf(V), c); r = state(Trel, th, mpf(V), c)
            sden = c.n*c.A*Trel**(c.n - 1)
            vals.append(r['D_chi'] + r['D_ell'] - (BETA*log(r['St']) + log(Trel**(D-1)/sden) + log(c.q - r['v']**2)/2 + mpf(D)/2*log(2*pi) + 1))
        print(f"cs2={cs2}: Delta - [(D/2+1) ln S_t + ln(T^d/s) + (1/2) ln a + (D/2) ln 2pi + 1] = {float(max(abs(x) for x in vals)):.0e}")
