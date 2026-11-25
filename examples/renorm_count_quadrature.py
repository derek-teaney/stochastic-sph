#!/usr/bin/env python3
"""State counting at one loop (docs/renorm3.tex, "The accuracy of one loop"): the single-particle
partition function at rest, d = 2, constant cs2 (p = A T^n).  Label S = S_t - (1/2) ln det(2 pi V chi) + ln T
at the target state of (e, m) (the label-gas factor 1/(eV) does not depend on beta and is left out).
Z = beta * int d^2m de e^S e^(-beta e), against ln Z_t = alpha beta^(1-n); the label boundary is ignored,
so small S_t (16) is not reliable.  Prints the residual of ln Z times S_t (prediction -c) and the
relative error of p and E, against n c/S_t^2 and the Legendre scheme's n kappa/S_t^2.  Needs numpy."""
import numpy as np
from numpy.polynomial.legendre import leggauss
def lnZ_S(n, St, beta=1.0, N=400, W=14.0):
    cs2=1.0/(n-1)
    alpha=St/n*beta**(n-1)            # ln Z_t = alpha beta^(1-n) = St/n
    u0=-np.log(beta); du=W/np.sqrt(St); vmax=min(0.9995, W/np.sqrt(St)*1.2)
    xu,wu=leggauss(N); xv,wv=leggauss(N)
    u=u0+du*xu; wu=wu*du; v=0.5*vmax*(xv+1); wv=wv*0.5*vmax
    U,Vv=np.meshgrid(u,v,indexing='ij'); tau=np.exp(U); g2=1/(1-Vv**2); g=np.sqrt(g2)
    e=alpha*tau**n*(n*g2-1); m=alpha*tau**n*n*g2*Vv
    # jacobian d(e,m)/d(u,v)
    de_du=n*e; dm_du=n*m
    dg2=2*Vv*g2**2
    de_dv=alpha*tau**n*n*dg2; dm_dv=alpha*tau**n*n*(dg2*Vv+g2)
    J=np.abs(de_du*dm_dv-de_dv*dm_du)
    S=n*alpha*g*tau**(n-1)
    w_tau=n*tau**n*alpha            # w V
    lndet=3*np.log(2*np.pi*g*w_tau*tau)+np.log((1-cs2*Vv**2)/cs2)
    sS=S-0.5*lndet+np.log(tau/g)
    ex=sS-beta*e
    mx=ex.max()
    I=np.einsum('i,j,ij->',wu,wv,np.exp(ex-mx)*2*np.pi*m*J)
    return np.log(beta)+mx+np.log(I), alpha*beta**(1-n)
for n,c0 in [(3,1/12),(5,3/16)]:
    kap=(3*n+1)**2/(8*(n-1))+c0
    for St in [16,54,541,5000]:
        # residual and its beta-derivative (E), V-derivative (p) by scaling: r depends on St only
        r=lambda s: (lambda a: a[0]-a[1])(lnZ_S(n,s))
        r0=r(St); h=1e-3
        dr=(r(St*(1+h))-r(St*(1-h)))/(2*h)      # d r/d ln St
        # dE/E = -(n-1) dr/dlnSt /( (n-1) St/n ) ; dp/p = dr/dlnSt/(St/n)
        print(f"n={n} St={St:5d}  r*St={r0*St:+.4f} (pred {-c0:+.4f})   dp/p=dE/E={dr*n/St:+.3e}  S-scheme pred {n*c0/St**2:+.3e}   Legendre pred {n*kap/St**2:+.3e}")
