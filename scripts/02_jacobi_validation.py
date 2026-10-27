"""Phase 1: validate the integrators.

1. Jacobi constant drift over 10 years for RK4 and DOP853.
2. RK4 convergence order: final-state error vs step size should scale as h^4.

The test orbit starts near L4 (a bounded tadpole orbit), so it stays far
from both primaries and is a clean test of integrator accuracy.

Run from the repository root:  python scripts/02_jacobi_validation.py
"""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cr3bp import MU_SUN_EARTH, dop853, jacobi_constant, lagrange_points, rhs, rk4, to_days

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
mu = MU_SUN_EARTH
f = partial(rhs, mu=mu)

s0 = np.r_[lagrange_points(mu)["L4"] + [0.01, 0.0, 0.0], 0.0, 0.0, 0.0]
T = 10 * 2 * np.pi                      # 10 years
c0 = jacobi_constant(s0, mu)

# Reference solution
t_ref, S_ref = dop853(f, s0, T, rtol=1e-13, atol=1e-13, t_eval=np.linspace(0, T, 2000))
drift_ref = np.abs(jacobi_constant(S_ref, mu) - c0) / abs(c0)

# RK4 at several step sizes
steps = [0.04, 0.02, 0.01, 0.005]
errors, drifts = [], {}
for h in steps:
    t, S = rk4(f, s0, T, h)
    errors.append(np.linalg.norm(S[-1, :3] - S_ref[-1, :3]))
    drifts[h] = (t, np.abs(jacobi_constant(S, mu) - c0) / abs(c0))
order = np.polyfit(np.log(steps), np.log(errors), 1)[0]

print(f"DOP853 max relative Jacobi drift over 10 years: {drift_ref.max():.2e}")
for h, e in zip(steps, errors):
    print(f"RK4 h={h:<6} final position error = {e:.3e}   max Jacobi drift = {drifts[h][1].max():.2e}")
print(f"Measured RK4 convergence order: {order:.2f}  (expected 4)")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

years = lambda t: to_days(t) / 365.25
for h, color in zip(steps, ["#f0b27a", "#e67e22", "#ba4a00", "#6e2c00"]):
    t, d = drifts[h]
    ax1.semilogy(years(t), np.maximum.accumulate(np.maximum(d, 1e-17)), color=color, lw=1.4,
                 label=f"RK4, h = {h}")
ax1.semilogy(years(t_ref), np.maximum.accumulate(np.maximum(drift_ref, 1e-17)), color="#2471a3",
             lw=1.6, label="DOP853")
ax1.set_ylim(1e-17, 1e-8)
ax1.set_xlabel("time [years]")
ax1.set_ylabel("max |C - C(0)| / |C(0)| so far")
ax1.set_title("Jacobi constant drift")
ax1.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")

ax2.loglog(steps, errors, "o-", color="#ba4a00", label=f"RK4 (slope {order:.2f})")
ref = errors[0] * (np.array(steps) / steps[0]) ** 4
ax2.loglog(steps, ref, "--", color="0.6", label="h^4 reference")
ax2.set_xlabel("step size h")
ax2.set_ylabel("final position error")
ax2.set_title("RK4 convergence after 10 years")
ax2.legend(frameon=False)

fig.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
out = FIG_DIR / "jacobi_validation.png"
fig.savefig(out, dpi=160)
print(f"Saved {out.relative_to(FIG_DIR.parent)}")
