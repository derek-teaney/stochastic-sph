# stencil_metric.py -- the bond moment M_a = sum_b V_b psi(rho_ab) r r^T of one star,
# at rest and in a moving fluid, and the same moving star in the coordinates
# s = h^{1/2} r in which the comoving stencil is round.
#
# psi = -W'/r of the 2D Wendland C2 kernel, psi(r) = 35/(4 pi h^4) (1 - q/2)^3,
# normalized so that n0 = (1/2) int d^2s psi(|s|) |s|^2 = 1.
# Jittered hexagonal lab lattice (shifts uniform in +-j a/2), V_b = sqrt(3)/2 a^2.
#
#   python3 examples/stencil_metric.py      -> docs/figs/stencil_metric.png
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

a, ha, j, vx = 1.0, 2.55, 0.3, 0.8
h = ha * a
g = 1 / np.sqrt(1 - vx**2)
Vb = np.sqrt(3) / 2 * a**2
rng = np.random.default_rng(7)


def psi(rho):
    q = rho / h
    return np.where(q < 2, 35 / (4 * np.pi * h**4) * (1 - q / 2) ** 3, 0.0)


R = int(np.ceil(4 * ha))
I, J = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1))
P = np.stack([a * (I + J / 2), a * J * np.sqrt(3) / 2], -1).reshape(-1, 2)
P = P + j * a * (rng.random(P.shape) - 0.5)
c = np.argmin(np.sum(P**2, 1))
r = P - P[c]
r = np.delete(r, c, 0)


def star(hmat):
    rho = np.sqrt(np.einsum("ni,ij,nj->n", r, hmat, r))
    w = psi(rho)
    M = Vb * np.einsum("n,ni,nj->ij", w, r, r)
    return rho, w, M


cases = []
# (a) at rest
h0 = np.eye(2)
rho, w, M = star(h0)
cases.append(("(a) at rest, lab coordinates", r, rho, w, M, h0, np.eye(2)))
# (b) moving, lab coordinates
hv = np.diag([g**2, 1.0])                     # h_ij = (ell^-1)_ij for v = vx xhat
rho, w, M = star(hv)
cases.append((f"(b) moving, $v={vx}\\,\\hat x$ ($\\gamma={g:.2f}$), lab coordinates",
              r, rho, w, M, hv, np.linalg.inv(hv)))
# (c) the same star in s = h^{1/2} r
S = r @ np.diag([g, 1.0])
Ms = np.diag([g, 1.0]) @ M @ np.diag([g, 1.0])
cases.append(("(c) the same star in $\\mathbf{s}=h^{1/2}\\mathbf{r}$", S, rho, w, Ms,
              np.eye(2), np.eye(2)))

fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.9))
vmax = psi(0.0)
for ax, (title, X, rho, w, M, hmat, target) in zip(axs, cases):
    inside = rho < 2 * h
    ax.scatter(X[~inside, 0], X[~inside, 1], s=6, c="0.8")
    sc = ax.scatter(X[inside, 0], X[inside, 1], s=16, c=w[inside], cmap="viridis",
                    vmin=0, vmax=vmax)
    ax.plot([0], [0], "r*", ms=13)
    # stencil boundary rho = 2h
    if ax is axs[1]:
        ax.add_patch(Ellipse((0, 0), 4 * h / g, 4 * h, fill=False, ls=":", color="k"))
        ax.annotate("", xy=(1.6 * h, -2.45 * h), xytext=(0.4 * h, -2.45 * h),
                    arrowprops=dict(arrowstyle="->", color="C3", lw=1.5))
        ax.text(1.0 * h, -2.35 * h, "$\\mathbf{v}$", color="C3", ha="center")
    else:
        ax.add_patch(Ellipse((0, 0), 4 * h, 4 * h, fill=False, ls=":", color="k"))
    # moment ellipse: r^T M^{-1} r = const, scaled so the target n0 h^{ij} has
    # semi-axes 1.6 h sqrt(eigenvalue)
    n = 0.5 * np.trace(np.linalg.inv(target) @ M) if ax is not axs[2] else 0.5 * np.trace(M)
    ev, U = np.linalg.eigh(M / n)
    ang = np.degrees(np.arctan2(U[1, 1], U[0, 1]))
    ax.add_patch(Ellipse((0, 0), 2 * 1.6 * h * np.sqrt(ev[1]), 2 * 1.6 * h * np.sqrt(ev[0]),
                         angle=ang, fill=False, color="C3", lw=1.8))
    dets = np.sqrt(np.linalg.det(hmat)) if ax is not axs[2] else g
    Mh = np.diag([g, 1.0]) @ M @ np.diag([g, 1.0]) if ax is axs[1] else M
    t = 0.5 * np.trace(Mh)
    A = np.linalg.norm(Mh - t * np.eye(2)) / t
    lab = (f"$n_a\\sqrt{{\\det h}}/n_0={n * dets:.3f}$\n$A_a={A:.3f}$")
    if ax is axs[2]:
        lab = f"$n_a\\gamma/n_0={n * g:.3f}$\n$A_a={A:.3f}$"
    ax.text(0.03, 0.97, lab, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(fc="w", ec="0.7"))
    L = 2.7 * h
    ax.set_xlim(-L, L)
    ax.set_ylim(-2.7 * h, 2.7 * h)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$s^x$" if ax is axs[2] else "$x$")
    ax.set_ylabel("$s^y$" if ax is axs[2] else "$y$")
cb = fig.colorbar(sc, ax=axs, shrink=0.8, pad=0.01)
cb.set_label("bond weight $\\psi(\\rho)$")
out = os.path.join(os.path.dirname(__file__), "..", "docs", "figs", "stencil_metric.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
for title, X, rho, w, M, hmat, target in cases:
    print(title, "\n  M =", np.round(M, 4).tolist())
print("Saved", out)
