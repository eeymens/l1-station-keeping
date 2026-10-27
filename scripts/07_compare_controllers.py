"""Phase 4: compare the trained DQN agent with the LQR baseline.

Test seeds 100-119 are never used during training or model selection
(training evaluates on seeds 0-4), so the comparison is not biased toward
the checkpoint that happened to score best.

For each scenario, LQR is shown as a curve over its fuel weight rho; DQN is
a single point. A controller below/left of the LQR curve beats it.

Writes experiments/comparison_<model>.csv and figures/lqr_vs_<model>.png.

Run from the repository root:  python scripts/07_compare_controllers.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

from cr3bp.control import LQRController, StationKeepingSim, run_episode
from cr3bp.envs import L1StationKeepingEnv, PolicyController
from cr3bp.evaluation import SCENARIOS, evaluate, summarize

ROOT = Path(__file__).resolve().parents[1]
TEST_SEEDS = range(100, 120)
RHOS = np.logspace(-3, 4, 15)
LQR_REF_RHO = 3.0                       # cheapest acquisition in the phase-3 sweep

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="dqn_l1", help="name of the model in models/")
args = ap.parse_args()

env = L1StationKeepingEnv()
model = DQN.load(ROOT / "models" / f"{args.model}.zip", device="cpu")
dqn = PolicyController(env, lambda f: model.predict(f, deterministic=True)[0])

rows = []
for name, ctrl in [("coast", None), ("DQN", dqn), (f"LQR rho={LQR_REF_RHO:g}",
                                                     LQRController(rho=LQR_REF_RHO))]:
    for r in evaluate(ctrl, seeds=TEST_SEEDS):
        rows.append({"controller": name, **r})
lqr_curve = {rho: summarize(evaluate(LQRController(rho=rho), seeds=TEST_SEEDS)) for rho in RHOS}

(ROOT / "experiments").mkdir(exist_ok=True)
with open(ROOT / "experiments" / f"comparison_{args.model}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

summ = {c: summarize([r for r in rows if r["controller"] == c])
        for c in dict.fromkeys(r["controller"] for r in rows)}
print(f"Test seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1}, one year per episode\n")
print(f"{'controller':<14}{'scenario':<13}{'kept':>6}{'days':>7}{'delta-v [m/s]':>17}"
      f"{'mean dist [km]':>18}")
for c, s in summ.items():
    for scen, m in s.items():
        print(f"{c:<14}{scen:<13}{m['survival']:>6.0%}{m['days']:>7.0f}"
              f"{m['dv_mps']:>10.3f} ± {m['dv_sd']:.3f}{m['mean_dist_km']:>11.1f} ± {m['dist_sd']:.1f}")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, scen in zip(axes[:2], SCENARIOS):
    xs = [lqr_curve[r][scen]["mean_dist_km"] for r in RHOS]
    ys = [lqr_curve[r][scen]["dv_mps"] for r in RHOS]
    ax.plot(xs, ys, "o-", color="#2471a3", ms=3.5, lw=1.2, label="LQR (each dot = one rho)")
    m = summ["DQN"][scen]
    lbl = f"DQN (kept {m['survival']:.0%} of runs)"
    ax.errorbar(m["mean_dist_km"], m["dv_mps"], xerr=m["dist_sd"], yerr=m["dv_sd"], fmt="s",
                color="#c0392b", ms=8, capsize=3, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean distance from L1 [km]")
    ax.set_ylabel("delta-v over one year [m/s]")
    ax.set_title({"holding": "Holding at L1", "acquisition": "Recovering from 1,000 km"}[scen])
    ax.legend(frameon=False, fontsize=9)

ax = axes[2]
for name, ctrl, color in (("LQR", LQRController(rho=LQR_REF_RHO), "#2471a3"),
                          ("DQN", dqn, "#c0392b")):
    r = run_episode(StationKeepingSim(seed=TEST_SEEDS.start), ctrl, offset_km=(1000.0, 0.0))
    ax.semilogy(r["t_days"], np.maximum(r["dist_km"], 1e-2), color=color, lw=1.3,
                label=f"{name}: {r['dv_total_mps']:.2f} m/s")
ax.set_xlabel("time [days]")
ax.set_ylabel("distance from L1 [km]")
ax.set_title(f"One recovery run (seed {TEST_SEEDS.start})")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(ROOT / "figures" / f"lqr_vs_{args.model}.png", dpi=160)
print(f"\nSaved figures/lqr_vs_{args.model}.png and experiments/comparison_{args.model}.csv")
