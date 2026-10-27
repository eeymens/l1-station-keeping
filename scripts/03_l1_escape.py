"""Phase 2: why L1 needs station-keeping.

A spacecraft placed at L1 with a 150 m position error and no thrust drifts
away exponentially. The measured growth rate is compared with the unstable
eigenvalue from the linear analysis.

Run from the repository root:  python scripts/03_l1_escape.py
"""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cr3bp import AU_KM, MU_SUN_EARTH, dop853, lagrange_points, rhs, stability, to_days

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
mu = MU_SUN_EARTH
l1 = lagrange_points(mu)["L1"]
lam = stability(l1, mu)["growth_rate"]

offset = 1e-9                              # ~150 m
s0 = np.r_[l1 + [offset, 0.0, 0.0], 0.0, 0.0, 0.0]
T = 7.5                                    # ~436 days
t, S = dop853(partial(rhs, mu=mu), s0, T, t_eval=np.linspace(0, T, 3000))
dist = np.linalg.norm(S[:, :3] - l1, axis=1)

# Fit the exponential growth once the unstable mode dominates
window = (dist > 1e-7) & (dist < 1e-4)
measured = np.polyfit(t[window], np.log(dist[window]), 1)[0]
days = to_days(t)

print(f"Initial offset from L1:          {offset * AU_KM * 1000:.0f} m")
print(f"Predicted growth rate (eigen):   {lam:.4f}  -> e-fold every {to_days(1 / lam):.1f} days")
print(f"Measured growth rate (fit):      {measured:.4f}  ({100 * (measured / lam - 1):+.2f}%)")
for km in (1_000, 100_000, 1_000_000):
    idx = np.argmax(dist * AU_KM > km)
    if dist[idx] * AU_KM > km:
        print(f"Distance passes {km:>9,} km after {days[idx]:6.0f} days")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

ax1.semilogy(days, dist * AU_KM, color="#c0392b", lw=1.5, label="simulation")
t0 = t[window][0]
pred = dist[window][0] * np.exp(lam * (t - t0)) * AU_KM
ax1.semilogy(days, pred, "--", color="0.45", lw=1, label=f"exp({lam:.3f} t), linear theory")
ax1.set_ylim(1e-2, 5e6)
ax1.set_xlabel("time [days]")
ax1.set_ylabel("distance from L1 [km]")
ax1.set_title("Uncontrolled drift away from L1")
ax1.legend(frameon=False)

earth_x = 1 - mu
ax2.plot((S[:, 0] - earth_x) * AU_KM / 1e6, S[:, 1] * AU_KM / 1e6, color="#c0392b", lw=1.2)
ax2.plot((l1[0] - earth_x) * AU_KM / 1e6, 0, "x", color="#c0392b", ms=9, mew=2)
ax2.annotate("L1", ((l1[0] - earth_x) * AU_KM / 1e6, 0), xytext=(-1.45, 0.12), fontsize=11)
ax2.plot(0, 0, "o", color="#3b7dd8", ms=7)
ax2.annotate("Earth", (0, 0), xytext=(0.06, 0.1))
ax2.set_aspect("equal", adjustable="datalim")
ax2.set_xlabel("x from Earth [million km]  (Sun is to the left)")
ax2.set_ylabel("y [million km]")
ax2.set_title("Escape trajectory, rotating frame")

fig.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
out = FIG_DIR / "l1_escape.png"
fig.savefig(out, dpi=160)
print(f"Saved {out.relative_to(FIG_DIR.parent)}")
