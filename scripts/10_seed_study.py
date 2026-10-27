"""Phase 5 follow-up: are the PPO results a lucky training run?

Each PPO configuration (cubic and linear thrust map) was trained with three
training seeds (0, 1, 2) and otherwise identical hyperparameters. Every model
flies the standard test protocol (holding + acquisition, test seeds 100-119,
never used in training or model selection). Results are reported per model and
as mean ± sample standard deviation (ddof = 1) across the training seeds.

"Same-accuracy LQR ratio": LQR is swept over rho = logspace(-3, 4, 57) on the
same test seeds. Along the sweep, tighter tracking costs more fuel until the
delta-v minimum; past it, lazier gains cost more again. Only the left branch
(rho up to the delta-v minimum) is a sensible comparison, so the LQR delta-v at
the model's mean distance is interpolated log-log on that branch, and

    ratio = model delta-v / LQR delta-v at the same mean distance.

A model whose mean distance lies outside the branch gets ratio = NaN. If it is
*less* accurate than the end of the branch (the delta-v minimum), that LQR point
is both cheaper and more accurate, i.e. it dominates the model; the column
lqr_ratio_vs_cheapest = model delta-v / minimum LQR delta-v is then a lower bound
on the same-accuracy ratio. Mean ± std of lqr_ratio is NaN unless all seeds are
inside the branch.

Writes experiments/seed_study.csv (per-model rows and mean/std rows),
experiments/seed_study_lqr.csv (the LQR sweep) and figures/seed_study.png.

Run from the repository root (~10 min on one core):
    python scripts/10_seed_study.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from cr3bp.control import LQRController
from cr3bp.evaluation import SCENARIOS, TEST_SEEDS, evaluate, summarize
from cr3bp.training import load_policy

ROOT = Path(__file__).resolve().parents[1]
RHOS = np.logspace(-3, 4, 57)
CONFIGS = {                               # config -> model name per training seed
    "cubic": {0: "ppo_l1", 1: "ppo_l1_s1", 2: "ppo_l1_s2"},
    "linear": {0: "ppo_l1_linear", 1: "ppo_l1_linear_s1", 2: "ppo_l1_linear_s2"},
}
METRICS = ("survival", "dv_mps", "mean_dist_km", "rms_dist_km")
COLORS = {"LQR": "#2471a3", "cubic": "#d68910", "linear": "#7d3c98"}
MARKERS = {0: "o", 1: "s", 2: "^"}


def left_branch(curve):
    """LQR points from the smallest rho up to (and including) the delta-v minimum,
    as (log10 mean distance, log10 delta-v), sorted by distance."""
    dist = np.array([c["mean_dist_km"] for c in curve])
    dv = np.array([c["dv_mps"] for c in curve])
    kept = np.array([c["survival"] == 1.0 for c in curve])
    if not kept.all():                    # a lost run would fake a low delta-v
        dv = np.where(np.arange(len(dv)) < np.argmin(kept), dv, np.inf)
    k = int(np.argmin(dv))
    x, y = np.log10(dist[:k + 1]), np.log10(dv[:k + 1])
    order = np.argsort(x)
    if not np.all(np.diff(x[order]) > 0):
        raise RuntimeError("LQR left branch is not monotone in distance")
    return x[order], y[order], k


def lqr_dv_at(dist_km, branch):
    x, y, _ = branch
    lx = np.log10(dist_km)
    if lx < x[0] or lx > x[-1]:
        return float("nan")
    return float(10 ** np.interp(lx, x, y))


def best_checkpoint_step(name):
    path = ROOT / "experiments" / f"{name}_training.csv"
    if not path.exists():
        return float("nan")
    with open(path) as fh:
        log = list(csv.DictReader(fh))
    return int(max(log, key=lambda r: float(r["score"]))["step"])


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    # ------------------------------------------------------------ LQR sweep
    print(f"LQR sweep: {len(RHOS)} values of rho on test seeds "
          f"{TEST_SEEDS.start}-{TEST_SEEDS.stop - 1}", flush=True)
    lqr = [summarize(evaluate(LQRController(rho=rho), seeds=TEST_SEEDS)) for rho in RHOS]
    branches = {scen: left_branch([s[scen] for s in lqr]) for scen in SCENARIOS}
    with open(ROOT / "experiments" / "seed_study_lqr.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rho", "scenario", "survival", "dv_mps", "mean_dist_km", "rms_dist_km",
                     "left_branch"])
        for i, (rho, s) in enumerate(zip(RHOS, lqr)):
            for scen in SCENARIOS:
                m = s[scen]
                w.writerow([rho, scen, m["survival"], m["dv_mps"], m["mean_dist_km"],
                            m["rms_dist_km"], i <= branches[scen][2]])
    for scen, (_, _, k) in branches.items():
        m = lqr[k][scen]
        print(f"  {scen}: delta-v minimum at rho = {RHOS[k]:.3g} "
              f"({m['dv_mps']:.3f} m/s, {m['mean_dist_km']:.2f} km)")

    # ------------------------------------------------------------ PPO models
    rows = []
    for config, models in CONFIGS.items():
        for seed, name in models.items():
            summ = summarize(evaluate(load_policy(ROOT / "models", name)[1], seeds=TEST_SEEDS))
            step = best_checkpoint_step(name)
            for scen, m in summ.items():
                ref = lqr_dv_at(m["mean_dist_km"], branches[scen])
                x, y, _ = branches[scen]
                dv_min = 10 ** y.min()                     # end of the left branch
                where = ("inside" if np.isfinite(ref) else
                         "less accurate" if np.log10(m["mean_dist_km"]) > x[-1]
                         else "more accurate")
                rows.append({"config": config, "train_seed": seed, "model": name,
                             "scenario": scen, **{k: m[k] for k in METRICS},
                             "lqr_dv_same_acc_mps": ref, "lqr_ratio": m["dv_mps"] / ref,
                             "vs_lqr_branch": where,
                             "lqr_ratio_vs_cheapest": m["dv_mps"] / dv_min,
                             "best_ckpt_step": step})
            print(f"  evaluated {name} (best checkpoint at {step} steps)", flush=True)

    stat_keys = (*METRICS, "lqr_dv_same_acc_mps", "lqr_ratio", "lqr_ratio_vs_cheapest",
                 "best_ckpt_step")
    agg = []
    for config in CONFIGS:
        for scen in SCENARIOS:
            sel = [r for r in rows if r["config"] == config and r["scenario"] == scen]
            for stat, fn in (("mean", np.mean), ("std", lambda v: np.std(v, ddof=1))):
                agg.append({"config": config, "train_seed": stat, "model": "",
                            "scenario": scen,
                            **{k: float(fn([r[k] for r in sel])) for k in stat_keys},
                            "vs_lqr_branch": ""})

    with open(ROOT / "experiments" / "seed_study.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows + [{k: a[k] for k in rows[0]} for a in agg])

    # ------------------------------------------------------------ table
    print(f"\nTest seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1}; "
          f"mean ± std over training seeds {sorted(CONFIGS['cubic'])}\n")
    print(f"{'config':<8}{'seed':<6}{'scenario':<13}{'kept':>6}{'delta-v [m/s]':>17}"
          f"{'mean [km]':>17}{'RMS [km]':>17}{'x LQR':>14}{'x cheapest':>14}{'best ckpt':>11}"
          f"  vs LQR branch")
    for config in CONFIGS:
        for scen in SCENARIOS:
            for r in rows:
                if r["config"] == config and r["scenario"] == scen:
                    print(f"{config:<8}{r['train_seed']:<6}{scen:<13}{r['survival']:>6.0%}"
                          f"{r['dv_mps']:>17.3f}{r['mean_dist_km']:>17.2f}"
                          f"{r['rms_dist_km']:>17.2f}{r['lqr_ratio']:>14.2f}"
                          f"{r['lqr_ratio_vs_cheapest']:>14.2f}"
                          f"{r['best_ckpt_step']:>11}  {r['vs_lqr_branch']}")
            mu = next(a for a in agg if a["config"] == config and a["scenario"] == scen
                      and a["train_seed"] == "mean")
            sd = next(a for a in agg if a["config"] == config and a["scenario"] == scen
                      and a["train_seed"] == "std")
            cell = lambda k, f: f"{mu[k]:{f}} ± {sd[k]:{f}}"  # noqa: E731
            print(f"{config:<8}{'all':<6}{scen:<13}{mu['survival']:>6.0%}"
                  f"{cell('dv_mps', '.3f'):>17}{cell('mean_dist_km', '.2f'):>17}"
                  f"{cell('rms_dist_km', '.2f'):>17}{cell('lqr_ratio', '.2f'):>14}"
                  f"{cell('lqr_ratio_vs_cheapest', '.2f'):>14}\n")

    # ------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    titles = {"holding": "Holding at L1", "acquisition": "Recovering from 1,000 km"}
    for ax, scen in zip(axes, SCENARIOS):
        xs = [s[scen]["mean_dist_km"] for s in lqr]
        ys = [s[scen]["dv_mps"] for s in lqr]
        k = branches[scen][2]
        ax.plot(xs[:k + 1], ys[:k + 1], "-", color=COLORS["LQR"], lw=2,
                label="LQR, rho up to the delta-v minimum")
        ax.plot(xs[k:], ys[k:], ":", color=COLORS["LQR"], lw=1.3,
                label="LQR, lazier gains")
        ax.plot(xs, ys, "o", color=COLORS["LQR"], ms=2.5)
        for config in CONFIGS:
            c = COLORS[config]
            for r in rows:
                if r["config"] == config and r["scenario"] == scen:
                    ax.plot(r["mean_dist_km"], r["dv_mps"], MARKERS[r["train_seed"]],
                            color=c, ms=8, mec="white", mew=1.0,
                            label=f"PPO {config}, training seed {r['train_seed']}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(which="minor", labelsize=7)
        ax.set_xlabel("mean distance from L1 [km]")
        ax.set_ylabel("delta-v over one year [m/s]")
        ax.set_title(titles[scen])
        ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"PPO across training seeds vs the LQR trade-off (test seeds "
                 f"{TEST_SEEDS.start}-{TEST_SEEDS.stop - 1})", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "seed_study.png", dpi=160)
    print("Saved experiments/seed_study.csv, experiments/seed_study_lqr.csv, "
          "figures/seed_study.png")


if __name__ == "__main__":
    main()
