#!/usr/bin/env python3
"""check_correlators.py — an independent check of the correlator analysis.

Reads the raw Fourier modes M_T(k, t) of equilibrium_*.h5 files with h5py and computes
    C_T(k, τ) = < Re M_T(k, t+τ) M_T*(k, t) >_{t > t0, k in shell, runs} / (N V w T)
by a direct sum over t (no FFT), sharing no code with src/measure.jl.  Prints C_T at a
few lags in the smallest shells, to be compared with shell_correlators (Julia).

Run:  python3 examples/check_correlators.py <dir with equilibrium_*.h5> [t0]
"""
import sys, glob
import numpy as np
import h5py

d = sys.argv[1]
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
lags_t = [0.0, 5.0, 10.0, 20.0, 30.0]
files = sorted(glob.glob(d + "/equilibrium_*.h5"))
acc, nruns = None, 0
for fn in files:
    with h5py.File(fn, "r") as f:
        if "complete" in f.attrs and not f.attrs["complete"]:
            continue
        t = f["t"][:]
        MT = f["MT"][:]                       # (nt, nk), complex
        ks = f["ks"][:]                       # (nk, 2)
        norm0 = f.attrs["N"] * f.attrs["V"] * f.attrs["w"] * f.attrs["T"]
    kn = np.round(np.hypot(ks[:, 0], ks[:, 1]), 8)
    shells = np.unique(kn)[:4]
    dt = t[1] - t[0]
    i0 = np.searchsorted(t, t0)
    x = MT[i0:, :]
    C = np.zeros((len(shells), len(lags_t)))
    for j, kv in enumerate(shells):
        idx = np.where(np.abs(kn - kv) < 1e-6)[0]
        for il, lt in enumerate(lags_t):
            L = int(round(lt / dt))
            prod = x[L:, idx] * np.conj(x[:x.shape[0] - L, idx])
            C[j, il] = prod.real.mean()       # mean over t and over the modes of the shell
    C /= norm0
    acc = C if acc is None else acc + C
    nruns += 1
acc /= nruns
h = 0.05
print(f"{d}: {nruns} runs, t0 = {t0}")
print("  kh      " + "  ".join(f"τ={lt:<5g}" for lt in lags_t))
for j, kv in enumerate(shells):
    print(f"  {kv*h:.3f}  " + "  ".join(f"{c:+.4f}" for c in acc[j]))
