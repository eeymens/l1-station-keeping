"""Phase 9: does a recurrent (LSTM) policy succeed where the stacked-history one failed?

Phase 8's MEM-PPO got the last 10 observations and actions as one long input
vector and did worse than plain DR-PPO in all 10 k/scenario cases. Two possible
reasons: (a) a feed-forward network cannot pull a slowly varying force out of a
noisy window with this training budget (architecture), or (b) the information
is too weak in 10 days of noisy navigation for any network (information).

LSTM-PPO (sb3-contrib RecurrentPPO, scripts/08 --lstm 128) tests (a). It sees
only the current observation, like DR-PPO, and keeps its own memory over the
whole episode, so it is not limited to a 10-day window. Everything else is as
for DR-PPO and MEM-PPO: same hyperparameters, k ~ U(0, 4), 5 training seeds,
model selection on seeds 0-4, 3M steps (trained on CI, see
.github/workflows/train-lstm.yml).

Evaluation is exactly that of scripts/13 (test seeds 100-119, k = 0, 0.5, 1, 2,
4, the same 345 classical designs, same columns); the phase-8 families are read
from experiments/phase8.csv rather than flown again. Extra column:

    vs_dr_ppo   delta-v ratio to DR-PPO with the same training seed, k and
                scenario (< 1: memory helped)

Writes experiments/phase9.csv and figures/phase9_pareto.png.

Run from the repository root (~20 min; needs scripts/12 and 13 to have run):
    python scripts/16_lstm.py
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from cr3bp.evaluation import SCENARIOS, TEST_SEEDS, evaluate, summarize
from cr3bp.training import load_policy

ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mm = _load("mismatch", "12_model_mismatch.py")
p8 = _load("phase8", "13_memory_residual.py")

KS = mm.KS
FAM = "LSTM-PPO"
REF = ("DR-PPO", "MEM-PPO", "RES-PPO")
COLORS = {**p8.COLORS, FAM: "#d35400"}
KEYS = (*mm.METRICS, "best_classical_ratio", "envelope_ratio", "excess_dv_mps", "dominated",
        "vs_dr_ppo")


def num(v):
    try:
        return float(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(f"ppo_lstm_s{s}" for s in range(5)),
                    help="comma-separated model names, one per training seed 0, 1, ...")
    ap.add_argument("--out", default="phase9")
    args = ap.parse_args()
    names = args.models.split(",")

    classical = mm.read_csv(ROOT / "experiments" / "model_mismatch_classical.csv")
    ref = [{k: num(v) for k, v in r.items()}
           for r in csv.DictReader(open(ROOT / "experiments" / "phase8.csv", newline=""))]
    dr = {(r["train_seed"], r["k"], r["scenario"]): r["dv_mps"] for r in ref
          if r["family"] == "DR-PPO" and r["train_seed"] not in ("mean", "std")}
    policies = {n: load_policy(ROOT / "models", n)[1] for n in names}

    rows = []
    for k in KS:
        for seed, name in enumerate(names):
            summ = summarize(evaluate(policies[name], seeds=TEST_SEEDS, perturbation_k=k))
            for scen, m in summ.items():
                rows.append({"family": FAM, "train_seed": seed, "model": name, "k": k,
                             "scenario": scen, **{key: m[key] for key in mm.METRICS},
                             **mm.compare(m, k, scen, classical),
                             "envelope_ratio": m["dv_mps"] / p8.envelope_dv(
                                 classical, k, scen, m["mean_dist_km"]),
                             "vs_dr_ppo": m["dv_mps"] / dr[(seed, k, scen)]})
        print(f"  k = {k:g} done", flush=True)

    agg = []
    for k in KS:
        for scen in SCENARIOS:
            sel = [r for r in rows if r["k"] == k and r["scenario"] == scen]
            for stat, fn in (("mean", np.mean),
                             ("std", lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0)):
                agg.append({**{key: "" for key in rows[0]}, "family": FAM,
                            "train_seed": stat, "k": k, "scenario": scen,
                            **{key: float(fn([r[key] for r in sel])) for key in KEYS}})
    with open(ROOT / "experiments" / f"{args.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows + agg)

    print(f"\nTest seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1}; mean ± std over "
          f"{len(names)} training seeds. 'x best classical': delta-v / cheapest classical "
          "design at least as accurate (< 1: not dominated)\n")
    print(f"{'k':>4}  {'scenario':<12}{'method':<10}{'kept':>6}{'delta-v [m/s]':>18}"
          f"{'mean [km]':>18}{'x best classical':>18}{'x envelope':>16}{'dominated':>11}")
    for k in KS:
        for scen in SCENARIOS:
            for fam in REF:
                mu = next(r for r in ref if r["family"] == fam and r["k"] == k
                          and r["scenario"] == scen and r["train_seed"] == "mean")
                print(f"{k:>4g}  {scen:<12}{fam:<10}{mu['survival']:>6.0%}"
                      f"{mu['dv_mps']:>18.3f}{mu['mean_dist_km']:>18.2f}"
                      f"{mu['best_classical_ratio']:>18.3f}{mu['envelope_ratio']:>16.3f}"
                      f"{mu['dominated']:>11.0%}")
            mu, sd = (next(a for a in agg if a["k"] == k and a["scenario"] == scen
                           and a["train_seed"] == s) for s in ("mean", "std"))
            cell = lambda key, f: f"{mu[key]:{f}} ± {sd[key]:{f}}"  # noqa: E731
            print(f"{k:>4g}  {scen:<12}{FAM:<10}{mu['survival']:>6.0%}"
                  f"{cell('dv_mps', '.3f'):>18}{cell('mean_dist_km', '.2f'):>18}"
                  f"{cell('best_classical_ratio', '.3f'):>18}"
                  f"{cell('envelope_ratio', '.3f'):>16}{mu['dominated']:>11.0%}")
        print()
    kept = [r for r in rows if r["survival"] == 1.0]
    better = sum(r["vs_dr_ppo"] < 1 for r in kept)
    print(f"LSTM-PPO kept the spacecraft in {len(kept)} of {len(rows)} seed x k x scenario "
          f"cases and used less fuel than DR-PPO (same training seed) in {better} of them")
    plot(rows, ref, classical, args.out)
    print(f"Saved experiments/{args.out}.csv, figures/{args.out}_pareto.png")


def plot(rows, ref, classical, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    ks = (0.0, 1.0, 4.0)
    fig, axes = plt.subplots(2, len(ks), figsize=(4.2 * len(ks), 8))
    for i, scen in enumerate(SCENARIOS):
        for j, k in enumerate(ks):
            ax = axes[i, j]
            for fam in mm.CLASSICAL:
                pts = [c for c in classical if c["k"] == k and c["scenario"] == scen
                       and c["family"] == fam and c["survival"] == 1.0]
                env = mm.pareto_envelope(pts)
                ax.plot([p["mean_dist_km"] for p in env], [p["dv_mps"] for p in env], "-",
                        color=COLORS[fam], lw=1.5, label=f"{fam}, best over rho and r_ref")
            for fam, src in ((f, ref) for f in REF):
                for r in src:
                    if (r["family"] == fam and r["k"] == k and r["scenario"] == scen
                            and r["train_seed"] not in ("mean", "std")
                            and r["survival"] == 1.0):
                        ax.plot(r["mean_dist_km"], r["dv_mps"], p8.MARKERS[int(r["train_seed"])],
                                color=COLORS[fam], ms=5, alpha=0.35)
            for r in rows:
                if r["k"] == k and r["scenario"] == scen and r["survival"] == 1.0:
                    ax.plot(r["mean_dist_km"], r["dv_mps"], p8.MARKERS[r["train_seed"]],
                            color=COLORS[FAM], ms=7, mec="white", mew=0.6,
                            label=FAM if r["train_seed"] == 0 else None)
            ax.set_xscale("log")
            ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_title(f"{'Holding' if scen == 'holding' else 'Recovery'}, k = {k:g}")
            ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("delta-v over one year [m/s]")
            if i == 1:
                ax.set_xlabel("mean distance from L1 [km]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], ls="", marker="o", color=COLORS[f], alpha=0.5) for f in REF]
    lab += [f"{f} (phase 7-8)" for f in REF]
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Phase 9: recurrent PPO (LSTM) vs. the phase 7-8 agents and the classical "
                 "envelopes", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(ROOT / "figures" / f"{out}_pareto.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
