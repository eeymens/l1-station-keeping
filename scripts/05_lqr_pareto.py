"""Phase 3: the fuel / accuracy trade-off of LQR.

Sweeps the fuel weight rho over nine orders of magnitude, 5 random seeds each,
with navigation and thrust errors on. Two scenarios, one year each:

* holding:     start exactly at L1; fuel is spent only fighting errors
* acquisition: start 1,000 km away at rest; includes the initial correction

Writes experiments/lqr_pareto.csv and figures/lqr_pareto.png.

Run from the repository root:  python scripts/05_lqr_pareto.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cr3bp.control import LQRController, StationKeepingSim, run_episode

ROOT = Path(__file__).resolve().parents[1]
RHOS = np.logspace(-4, 5, 19)
SEEDS = range(5)
SCENARIOS = {"holding": (0.0, 0.0), "acquisition": (1000.0, 0.0)}

rows = []
for scenario, offset in SCENARIOS.items():
    for rho in RHOS:
        ctrl = LQRController(rho=rho)
        for seed in SEEDS:
            r = run_episode(StationKeepingSim(seed=seed), ctrl, offset_km=offset)
            rows.append({"scenario": scenario, "rho": rho, "seed": seed,
                         "dv_mps": r["dv_total_mps"], "mean_dist_km": r["mean_dist_km"],
                         "final_dist_km": r["final_dist_km"], "survived": r["survived"]})

(ROOT / "experiments").mkdir(exist_ok=True)
with open(ROOT / "experiments" / "lqr_pareto.csv", "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def summary(scenario, key):
    vals = np.array([[r[key] for r in rows if r["scenario"] == scenario and r["rho"] == rho]
                     for rho in RHOS])
    return vals.mean(axis=1), vals.std(axis=1)


hold_dv, hold_dv_sd = summary("holding", "dv_mps")
hold_err, hold_err_sd = summary("holding", "mean_dist_km")
acq_dv, acq_dv_sd = summary("acquisition", "dv_mps")
acq_final, _ = summary("acquisition", "final_dist_km")
recovered = acq_final < 10.0          # back within 10 km of L1 after one year
print(f"All runs survived one year: {all(r['survived'] for r in rows)}")
print(f"{'rho':>10}{'hold dv [m/s]':>16}{'hold error [km]':>18}{'acq. dv [m/s]':>16}{'acq. final [km]':>18}")
for i, rho in enumerate(RHOS):
    print(f"{rho:>10.0e}{hold_dv[i]:>16.4f}{hold_err[i]:>18.3f}{acq_dv[i]:>16.3f}{acq_final[i]:>18.2f}")
best = int(np.argmin(np.where(recovered, acq_dv, np.inf)))
print(f"\nCheapest acquisition: rho = {RHOS[best]:.0e}, {acq_dv[best]:.3f} m/s")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

ax1.errorbar(hold_err, hold_dv, yerr=hold_dv_sd, fmt="o-", color="#17a589",
             ms=4, lw=1.2, capsize=2)
for i in (0, 6, 10, 18):
    ax1.annotate(f"rho = {RHOS[i]:.0e}", (hold_err[i], hold_dv[i]), xytext=(8, 4),
                 textcoords="offset points", fontsize=8.5, color="0.35")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("mean distance from L1 [km]")
ax1.set_ylabel("delta-v per year [m/s]")
ax1.set_title("Holding at L1: tighter costs more fuel")

ax2.errorbar(RHOS, acq_dv, yerr=acq_dv_sd, fmt="-", color="#2471a3", lw=1.2, capsize=2)
ax2.plot(RHOS[recovered], acq_dv[recovered], "o", color="#2471a3", ms=5, label="back within 10 km")
ax2.plot(RHOS[~recovered], acq_dv[~recovered], "o", mfc="white", mec="#2471a3", ms=5,
         label="still > 10 km away after 1 year")
ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
ax2.plot(RHOS[best], acq_dv[best], "o", ms=10, mfc="none", mec="#c0392b", mew=1.5)
ax2.annotate("cheapest", (RHOS[best], acq_dv[best]), xytext=(0, 12), textcoords="offset points",
             ha="center", fontsize=9, color="#c0392b")
ax2.set_xscale("log")
ax2.set_xlabel("fuel weight rho")
ax2.set_ylabel("delta-v over first year [m/s]")
ax2.set_title("Recovering from 1,000 km: waiting costs fuel")

fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "lqr_pareto.png", dpi=160)
print("Saved figures/lqr_pareto.png and experiments/lqr_pareto.csv")
