#!/usr/bin/env python3
"""State counting at one loop (docs/renorm3.tex, "The state of a particle"): the closed forms for the
particle's temperature, velocity and pressure, eqs. (Tava) and (Pa), against finite differences of
S(e, m, V) = S_t - Delta.  d = 2, two-component p = T^3 + 0.7 T^5 (cs2 from 1/2 to 1/4).  Needs mpmath.
Each pair of columns is (closed form, finite difference); the last two are the target's T and p."""
from mpmath import mp, mpf, sqrt, log, pi, findroot, diff, atanh, tanh, cosh, sinh
mp.dps=30
d=2; D=3
p  = lambda T: T**3+mpf('0.7')*T**5
s_ = lambda T: 3*T**2+mpf('3.5')*T**4
q_ = lambda T: T*diff(s_,T)/s_(T)
q1_= lambda T: T*diff(q_,T)
def target(e,m,V):
    # solve e = V(w g^2 - p), m = V w g^2 v for (lnT, theta)
    def f(lt,th):
        T=mp.e**lt; g=cosh(th); v=tanh(th); w=T*s_(T)
        return (V*(w*g*g-p(T))-e, V*w*g*g*v-m)
    lt,th=findroot(f,(log((e/V)**(mpf(1)/4)),atanh(m/e*mpf('0.8'))))
    return mp.e**lt, th
def state(e,m,V):
    T,th=target(e,m,V); g=cosh(th); v=tanh(th); w=T*s_(T); q=q_(T); q1=q1_(T); a=q-v*v
    St=g*s_(T)*V
    Delta=(mpf(D)/2+1)*log(g*V)+mpf(D)/2*log(2*pi*w*T)-log(T)+log(a)/2+1
    Dt=mpf(D)/2*(2+q)-1+q1/(2*a); Dth=(mpf(D)/2+1)*v-v*(1-v*v)/a
    eT=(Dt-v*Dth)/(St*a); eth=(q*Dth-v*Dt)/(St*a)
    invTa=(g*(1-eT)+g*v*eth)/T
    va=(v*(1-eT)+eth)/((1-eT)+v*eth)
    eps=w-p(T)
    PVoverT=p(T)*V*g/T-(mpf(D)/2+1)+(Dt*(eps-p(T)*v*v)+Dth*v*(q*p(T)-eps))/(w*a)
    return dict(S=St-Delta,Ta=1/invTa,va=va,Pa=PVoverT/V/invTa,T=T,v=v,St=St,eT=eT,eth=eth)
for (e,m,V) in [(mpf(30),mpf(0),mpf(2)),(mpf(30),mpf(12),mpf(2)),(mpf(8),mpf(6),mpf(1)),(mpf(300),mpf(250),mpf(3))]:
    st=state(e,m,V)
    Sf=lambda E,M,VV: state(E,M,VV)['S']
    h=mpf(10)**-8
    dSe=(Sf(e*(1+h),m,V)-Sf(e*(1-h),m,V))/(2*e*h)
    dSm=(Sf(e,m+e*h,V)-Sf(e,m-e*h,V))/(2*e*h)
    dSV=(Sf(e,m,V*(1+h))-Sf(e,m,V*(1-h)))/(2*V*h)
    Ta=1/dSe; va=-Ta*dSm; Pa=Ta*dSV
    print(f"St={float(st['St']):7.2f} v={float(st['v']):.3f}  T_a {float(st['Ta']):.9f} {float(Ta):.9f} | v_a {float(st['va']):.9f} {float(va):.9f} | P_a {float(st['Pa']):.9f} {float(Pa):.9f} | target T/g {float(st['T']/cosh(atanh(st['v']))):.6f} p {float(p(st['T'])):.6f}")
