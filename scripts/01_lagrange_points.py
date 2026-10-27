"""Phase 2: locate the Sun-Earth Lagrange points and classify their stability.

Prints a table (position, distance from Earth, eigenvalues, e-folding time)
and saves figures/lagrange_points.png.

Run from the repository root:  python scripts/01_lagrange_points.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cr3bp import (AU_KM, MU_SUN_EARTH, effective_potential, jacobi_constant,
                   lagrange_points, stability)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
mu = MU_SUN_EARTH
earth_x = 1.0 - mu
points = lagrange_points(mu)

# ---------------------------------------------------------------- table
print(f"Sun-Earth CR3BP, mu = {mu:.6e}\n")
print(f"{'pt':<4}{'x':>14}{'y':>10}{'from Earth [km]':>18}{'C_J':>12}"
      f"{'growth rate':>13}{'e-fold [days]':>15}  eigenvalues")
for name, p in points.items():
    st = stability(p, mu)
    d_km = np.linalg.norm(p - np.array([earth_x, 0, 0])) * AU_KM
    c = jacobi_constant(np.r_[p, 0, 0, 0], mu)
    eig = ", ".join(f"{e.real:+.4f}{e.imag:+.4f}i" for e in st["eigenvalues"])
    efold = "stable" if st["growth_rate"] == 0 else f"{st['efold_days']:.1f}"
    print(f"{name:<4}{p[0]:>14.8f}{p[1]:>10.5f}{d_km:>18,.0f}{c:>12.8f}"
          f"{st['growth_rate']:>13.4f}{efold:>15}  {eig}")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

# Left: the whole system in the rotating frame
theta = np.linspace(0, 2 * np.pi, 400)
ax1.plot(np.cos(theta), np.sin(theta), color="0.85", lw=0.8, zorder=0)
ax1.plot(-mu, 0, "o", color="#e8a33d", ms=14, label="Sun")
ax1.plot(earth_x, 0, "o", color="#3b7dd8", ms=6, label="Earth")
for name, p in points.items():
    unstable = stability(p, mu)["growth_rate"] > 1e-3
    ax1.plot(p[0], p[1], "x" if unstable else "+", color="#c0392b" if unstable else "#27ae60",
             ms=9, mew=2)
    offset = {"L1": (-0.05, 0.08), "L2": (0.06, 0.08), "L3": (-0.18, 0.08),
              "L4": (0.05, 0.05), "L5": (0.05, -0.12)}[name]
    ax1.annotate(name, (p[0], p[1]), xytext=(p[0] + offset[0], p[1] + offset[1]), fontsize=11)
ax1.set_aspect("equal")
ax1.set_xlim(-1.45, 1.45)
ax1.set_ylim(-1.25, 1.25)
ax1.set_xlabel("x [AU, rotating frame]")
ax1.set_ylabel("y [AU]")
ax1.set_title("Lagrange points (red x = unstable, green + = stable)")
ax1.legend(loc="lower left", frameon=False)

# Right: zoom on Earth with the zero-velocity curves through L1 and L2
half = 0.02
xs = np.linspace(earth_x - half, earth_x + half, 700)
ys = np.linspace(-half, half, 700)
X, Y = np.meshgrid(xs, ys)
two_omega = 2 * effective_potential(X, Y, 0.0, mu)
c_l1 = jacobi_constant(np.r_[points["L1"], 0, 0, 0], mu)
c_l2 = jacobi_constant(np.r_[points["L2"], 0, 0, 0], mu)
Xk, Yk = (X - earth_x) * AU_KM / 1e6, Y * AU_KM / 1e6   # million km from Earth
ax2.contourf(Xk, Yk, two_omega, levels=[two_omega.min(), c_l1], colors=["#d9dde6"])
ax2.contour(Xk, Yk, two_omega, levels=[c_l1], colors="#c0392b", linewidths=1.2)
ax2.contour(Xk, Yk, two_omega, levels=[c_l2], colors="#7f8c8d", linewidths=1.0, linestyles="--")
ax2.plot(0, 0, "o", color="#3b7dd8", ms=7)
for name in ("L1", "L2"):
    p = points[name]
    px = (p[0] - earth_x) * AU_KM / 1e6
    ax2.plot(px, 0, "x", color="#c0392b", ms=9, mew=2)
    ax2.annotate(name, (px, 0), xytext=(px - 0.12, 0.18), fontsize=11)
ax2.set_aspect("equal")
ax2.set_xlabel("x from Earth [million km]  (Sun is to the left)")
ax2.set_ylabel("y [million km]")
ax2.set_title("Zero-velocity curves near Earth\n(shaded: forbidden region at C = C_L1)")

fig.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
out = FIG_DIR / "lagrange_points.png"
fig.savefig(out, dpi=160)
print(f"\nSaved {out.relative_to(FIG_DIR.parent)}")
