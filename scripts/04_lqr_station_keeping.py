"""Phase 3: LQR keeps the spacecraft at L1.

The spacecraft starts 1,000 km from L1 at rest. Three one-year runs:
uncontrolled, LQR with perfect navigation, LQR with navigation and thrust
errors. Thrust is updated once per day.

Run from the repository root:  python scripts/04_lqr_station_keeping.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cr3bp.control import LQRController, NoiseModel, StationKeepingSim, run_episode

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
RHO = 1.0

runs = {
    "uncontrolled": run_episode(StationKeepingSim(noise=NoiseModel.none()), None),
    "LQR, perfect navigation": run_episode(StationKeepingSim(noise=NoiseModel.none()),
                                           LQRController(rho=RHO)),
    "LQR, with errors": run_episode(StationKeepingSim(seed=0), LQRController(rho=RHO)),
}

n = NoiseModel()
print(f"Errors (1 sigma): position {n.pos_km} km, velocity {n.vel_mps * 1000:.0f} mm/s, "
      f"thrust magnitude {n.thrust_mag_frac:.0%}, pointing {n.thrust_angle_deg} deg\n")
for name, r in runs.items():
    status = "held 1 year" if r["survived"] else f"lost after {r['days_survived']:.0f} days"
    print(f"{name:<26} {status:<22} delta-v {r['dv_total_mps']:6.3f} m/s   "
          f"final distance {r['final_dist_km']:9.2f} km")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
colors = {"uncontrolled": "#c0392b", "LQR, perfect navigation": "#2471a3",
          "LQR, with errors": "#17a589"}

ax = axes[0]
for name, r in runs.items():
    ax.semilogy(r["t_days"], np.maximum(r["dist_km"], 1e-3), color=colors[name], lw=1.4, label=name)
ax.axhline(10_000, color="0.6", ls=":", lw=1)
ax.text(365, 12_000, "lost (10,000 km)", ha="right", va="bottom", fontsize=9, color="0.4")
ax.set_ylim(1e-2, 3e4)
ax.set_xlabel("time [days]")
ax.set_ylabel("distance from L1 [km]")
ax.set_title("Distance from L1")
ax.legend(frameon=False, fontsize=9, loc="lower left")

ax = axes[1]
for name in ("LQR, perfect navigation", "LQR, with errors"):
    r = runs[name]
    ax.plot(r["t_days"], r["dv_mps"], color=colors[name], lw=1.4, label=name)
ax.set_xlabel("time [days]")
ax.set_ylabel("cumulative delta-v [m/s]")
ax.set_title("Fuel spent")
ax.legend(frameon=False, fontsize=9, loc="lower right")

ax = axes[2]
r = runs["LQR, with errors"]
ax.plot(r["pos_km"][:, 0], r["pos_km"][:, 1], color=colors["LQR, with errors"], lw=0.9)
ax.plot(*r["pos_km"][0], "o", color="0.3", ms=5)
ax.annotate("start", r["pos_km"][0], xytext=(8, 6), textcoords="offset points", fontsize=9)
ax.plot(0, 0, "x", color="#c0392b", ms=9, mew=2)
ax.annotate("L1", (0, 0), xytext=(8, -14), textcoords="offset points", fontsize=10)
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("x from L1 [km]  (Earth is to the right)")
ax.set_ylabel("y from L1 [km]")
ax.set_title("Path with errors, rotating frame")

fig.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
out = FIG_DIR / "lqr_station_keeping.png"
fig.savefig(out, dpi=160)
print(f"\nSaved {out.relative_to(FIG_DIR.parent)}")
