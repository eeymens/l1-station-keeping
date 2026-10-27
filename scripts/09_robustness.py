"""Phase 5: robustness of LQR, DQN and PPO outside the training conditions.

All agents were trained (and LQR was tuned) with the nominal error model and
start offsets up to 1,000 km. Three sweeps push them beyond that, on the
test seeds 100-119:

1. noise:       all errors (navigation, thrust, unmodelled forces) x k
2. disturbance: only the unmodelled force x k (e.g. a solar-pressure model
                that is badly off), everything else nominal
3. offset:      start 250 - 5,000 km from L1, in four directions

The RL agents can thrust at most 1e-6 m/s^2. Plain LQR has no limit, so the
sweeps also fly LQR saturated at the same 1e-6 m/s^2, for a fair comparison
at large offsets.

Writes experiments/robustness.csv and figures/robustness.png.

Run from the repository root:
    python scripts/09_robustness.py --model dqn_l1_run2 ppo_l1
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cr3bp.control import LQRController, NoiseModel, StationKeepingSim, run_episode
from cr3bp.evaluation import SCENARIOS, TEST_SEEDS
from cr3bp.training import load_policy

ROOT = Path(__file__).resolve().parents[1]
LQR_RHO = 3.0
U_MAX = 1e-6
NOISE_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
DIST_SCALES = (1.0, 3.0, 10.0, 30.0, 100.0)
OFFSETS_KM = (250.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0)
ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)
COLORS = {"LQR": "#2471a3", "LQR (sat.)": "#85c1e9", "DQN": "#c0392b", "PPO": "#d68910"}


def episode_row(ctrl, seed, noise, offset):
    r = run_episode(StationKeepingSim(seed=seed, noise=noise), ctrl, offset_km=offset)
    d = r["dist_km"]
    late = d[r["t_days"] >= 182.5]                    # second half of the year
    return {"survived": r["survived"], "days_survived": r["days_survived"],
            "dv_mps": r["dv_total_mps"], "mean_dist_km": r["mean_dist_km"],
            "rms_dist_km": float(np.sqrt(np.mean(d**2))), "max_dist_km": float(d.max()),
            "rms_late_km": float(np.sqrt(np.mean(late**2))) if r["survived"] else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["dqn_l1_run2", "ppo_l1"])
    ap.add_argument("--seeds", type=int, default=len(TEST_SEEDS),
                    help="number of test seeds (from 100)")
    args = ap.parse_args()
    seeds = range(TEST_SEEDS.start, TEST_SEEDS.start + args.seeds)

    ctrls = {"LQR": LQRController(rho=LQR_RHO),
             "LQR (sat.)": LQRController(rho=LQR_RHO, u_max_mps2=U_MAX)}
    for m in args.model:
        ctrls[m.split("_")[0].upper()] = load_policy(ROOT / "models", m)[1]

    rows, t0 = [], time.time()

    def run(sweep, level, scenario, noise, offsets):
        for name, ctrl in ctrls.items():
            for i, seed in enumerate(seeds):
                off = offsets[i % len(offsets)]
                rows.append({"sweep": sweep, "level": level, "scenario": scenario,
                             "controller": name, "seed": seed,
                             **episode_row(ctrl, seed, noise, off)})

    nominal = NoiseModel()
    for k in NOISE_SCALES:
        for scen, off in SCENARIOS.items():
            run("noise", k, scen, nominal.scaled(k), [off])
        print(f"noise x{k:g} done ({time.time() - t0:.0f} s)", flush=True)
    for k in DIST_SCALES:
        noise = NoiseModel(disturbance_mps2=nominal.disturbance_mps2 * k)
        for scen, off in SCENARIOS.items():
            run("disturbance", k, scen, noise, [off])
        print(f"disturbance x{k:g} done ({time.time() - t0:.0f} s)", flush=True)
    for r_km in OFFSETS_KM:
        offs = [(r_km * np.cos(np.radians(a)), r_km * np.sin(np.radians(a))) for a in ANGLES_DEG]
        run("offset", r_km, "acquisition", nominal, offs)
        print(f"offset {r_km:g} km done ({time.time() - t0:.0f} s)", flush=True)

    with open(ROOT / "experiments" / "robustness.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------ summary
    def agg(sweep, scen, name, level, key):
        """Mean of `key`; fuel and tracking error over the runs kept only."""
        v = [r[key] for r in rows if r["sweep"] == sweep and r["scenario"] == scen
             and r["controller"] == name and r["level"] == level
             and (key == "survived" or r["survived"])]
        return float(np.mean(v)) if v else float("nan")

    panels = [("noise", "holding", NOISE_SCALES, "all errors x k (holding)"),
              ("noise", "acquisition", NOISE_SCALES, "all errors x k (recovery from 1,000 km)"),
              ("disturbance", "holding", DIST_SCALES, "unmodelled force x k (holding)"),
              ("offset", "acquisition", OFFSETS_KM, "start offset [km]")]
    for sweep, scen, levels, title in panels:
        print(f"\n{title}   [kept, delta-v of kept runs, RMS distance in days 183-365]")
        print(f"{'controller':<12}" + "".join(f"{lv:>22g}" for lv in levels))
        for name in ctrls:
            cells = []
            for lv in levels:
                s = agg(sweep, scen, name, lv, "survived")
                cells.append(f"{s:4.0%} {agg(sweep, scen, name, lv, 'dv_mps'):6.2f}m/s "
                             f"{agg(sweep, scen, name, lv, 'rms_late_km'):6.1f}km")
            print(f"{name:<12}" + "".join(f"{c:>22}" for c in cells))

    # ------------------------------------------------------------ figure
    # k = 0 (no errors at all) is in the CSV and the table but not in the
    # figure: LQR then needs ~1e-10 m/s, which would squash the log axes.
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex="col")
    for j, (sweep, scen, levels, title) in enumerate(panels):
        levels = [lv for lv in levels if lv > 0]
        for name in ctrls:
            c = COLORS.get(name, "#555555")
            ls = "--" if "sat" in name else "-"
            for i, key in enumerate(("survived", "dv_mps", "rms_late_km")):
                ys = [agg(sweep, scen, name, lv, key) for lv in levels]
                if key == "survived":
                    ys = [100 * y for y in ys]
                axes[i, j].plot(levels, ys, "o" + ls, color=c, ms=3.5, lw=1.3, label=name)
        axes[0, j].set_title(title)
        axes[0, j].set_ylim(-5, 105)
        axes[2, j].set_xlabel("k" if sweep != "offset" else "start offset [km]")
        for i in range(3):
            axes[i, j].set_xscale("log")
            axes[i, j].set_xticks(levels)
            axes[i, j].set_xticklabels([f"{lv:g}" for lv in levels])
            axes[i, j].minorticks_off()
            if i > 0:
                axes[i, j].set_yscale("log")
    axes[0, 0].set_ylabel("runs kept 1 year [%]")
    axes[1, 0].set_ylabel("delta-v over one year [m/s]\n(runs kept)")
    axes[2, 0].set_ylabel("RMS distance, days 183-365 [km]\n(runs kept)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "robustness.png", dpi=150)
    print(f"\nSaved figures/robustness.png and experiments/robustness.csv "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
