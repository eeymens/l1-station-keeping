"""Phase 8: can RL catch up once it gets memory, or the Kalman filter's estimates?

Phase 7 found that a domain-randomised PPO agent (DR-PPO) comes within a few
per cent of the best classical fuel under solar radiation pressure (SRP), but
is still dominated by the LQR with a Kalman bias estimator (Bias-LQR). One
plausible reason: DR-PPO sees only the current navigation estimate and cannot
tell how strong the SRP is (at k = 0 it thrusts as if an average SRP acted).
Two ways to give RL that information, both trained exactly like DR-PPO
(same hyperparameters, k ~ U(0, 4), 5 training seeds, selection on seeds 0-4):

    MEM-PPO   memory: the observation holds the last 10 navigation estimates
              and the last 10 actions (scripts/08 --history 10)
    RES-PPO   hybrid / residual RL: thrust = Bias-LQR (rho = 1) + agent, and
              the agent also sees the filter's state and force estimates and
              the base thrust (scripts/08 --residual-rho 1)

Both are flown on test seeds 100-119 at k = 0, 0.5, 1, 2, 4 and compared with
the same 345 classical designs as phase 7 (experiments/model_mismatch_classical.csv,
written by scripts/12). The residual agent's own base, Bias-LQR rho = 1 capped
at the agents' thrust limit of 1e-6 m/s^2, is flown too ("RES base"): a
residual agent is only useful if it improves on it.

Columns as in scripts/12 (best_classical_ratio, excess_dv_mps, dominated),
plus for RES-PPO the delta-v and distance relative to its base, and

    envelope_ratio  model delta-v / the classical Pareto envelope (all 345
                    designs) interpolated log-log at the model's mean distance.
                    The staircase best_classical_ratio compares against a design
                    that may be noticeably *more* accurate, which favours the
                    model; a claim that RL beats the classical side needs
                    envelope_ratio < 1. NaN outside the envelope's range.

Writes experiments/phase8.csv and figures/phase8_pareto.png.

Run from the repository root (~15 min; needs scripts/12 to have run):
    python scripts/13_memory_residual.py
"""

import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from cr3bp.control import BiasLQRController
from cr3bp.evaluation import SCENARIOS, TEST_SEEDS, evaluate, summarize
from cr3bp.training import load_policy

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mismatch", ROOT / "scripts" / "12_model_mismatch.py")
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)

KS = mm.KS
RL = {
    "DR-PPO": {s: f"ppo_dr_s{s}" for s in range(5)},
    "MEM-PPO": {s: f"ppo_mem_s{s}" for s in range(5)},
    "RES-PPO": {s: f"ppo_res_s{s}" for s in range(5)},
}
COLORS = {**mm.COLORS, "MEM-PPO": "#8e44ad", "RES-PPO": "#16a085", "RES base": "#17202a"}
MARKERS = mm.MARKERS


def envelope_dv(classical, k, scen, dist):
    pts = [c for c in classical if c["k"] == k and c["scenario"] == scen
           and c["survival"] == 1.0]
    env = mm.pareto_envelope(pts)
    x = np.log10([p["mean_dist_km"] for p in env])
    y = np.log10([p["dv_mps"] for p in env])
    lx = np.log10(dist)
    if lx < x[0] or lx > x[-1]:
        return float("nan")
    return float(10 ** np.interp(lx, x, y))


def main():
    classical = mm.read_csv(ROOT / "experiments" / "model_mismatch_classical.csv")
    policies = {name: load_policy(ROOT / "models", name)[1]
                for models in RL.values() for name in models.values()}
    rows = []
    for k in KS:
        base = summarize(evaluate(BiasLQRController(rho=1.0, u_max_mps2=1e-6), seeds=TEST_SEEDS,
                                  perturbation_k=k))
        for scen, m in base.items():
            rows.append({"family": "RES base", "train_seed": -1, "model": "Bias-LQR rho=1 capped",
                         "k": k, "scenario": scen, **{key: m[key] for key in mm.METRICS},
                         **mm.compare(m, k, scen, classical),
                         "envelope_ratio": m["dv_mps"] / envelope_dv(classical, k, scen,
                                                                     m["mean_dist_km"]),
                         "dv_vs_base": 1.0, "dist_vs_base": 1.0})
        for fam, models in RL.items():
            for seed, name in models.items():
                summ = summarize(evaluate(policies[name], seeds=TEST_SEEDS, perturbation_k=k))
                for scen, m in summ.items():
                    b = base[scen]
                    rows.append({"family": fam, "train_seed": seed, "model": name, "k": k,
                                 "scenario": scen, **{key: m[key] for key in mm.METRICS},
                                 **mm.compare(m, k, scen, classical),
                                 "envelope_ratio": m["dv_mps"] / envelope_dv(
                                     classical, k, scen, m["mean_dist_km"]),
                                 "dv_vs_base": m["dv_mps"] / b["dv_mps"],
                                 "dist_vs_base": m["mean_dist_km"] / b["mean_dist_km"]})
        print(f"  k = {k:g} done", flush=True)

    keys = (*mm.METRICS, "best_classical_ratio", "envelope_ratio", "excess_dv_mps", "dominated",
            "dv_vs_base", "dist_vs_base")
    agg = []
    for fam in RL:
        for k in KS:
            for scen in SCENARIOS:
                sel = [r for r in rows if r["family"] == fam and r["k"] == k
                       and r["scenario"] == scen]
                for stat, fn in (("mean", np.mean), ("std", lambda v: np.std(v, ddof=1))):
                    agg.append({**{key: "" for key in rows[0]}, "family": fam,
                                "train_seed": stat, "k": k, "scenario": scen,
                                **{key: float(fn([r[key] for r in sel])) for key in keys}})
    with open(ROOT / "experiments" / "phase8.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows + agg)

    print(f"\nTest seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1}; mean ± std over 5 "
          "training seeds; 'dominated' = share of seeds\n")
    print(f"{'k':>4}  {'scenario':<12}{'method':<10}{'kept':>6}{'delta-v [m/s]':>18}"
          f"{'mean [km]':>18}{'x best classical':>18}{'x envelope':>16}{'dominated':>11}")
    for k in KS:
        for scen in SCENARIOS:
            r = next(r for r in rows if r["family"] == "RES base" and r["k"] == k
                     and r["scenario"] == scen)
            print(f"{k:>4g}  {scen:<12}{'RES base':<10}{r['survival']:>6.0%}"
                  f"{r['dv_mps']:>18.3f}{r['mean_dist_km']:>18.2f}"
                  f"{r['best_classical_ratio']:>18.3f}{r['envelope_ratio']:>16.3f}"
                  f"{r['dominated']:>11.0f}")
            for fam in RL:
                mu = next(a for a in agg if a["family"] == fam and a["k"] == k
                          and a["scenario"] == scen and a["train_seed"] == "mean")
                sd = next(a for a in agg if a["family"] == fam and a["k"] == k
                          and a["scenario"] == scen and a["train_seed"] == "std")
                cell = lambda key, f: f"{mu[key]:{f}} ± {sd[key]:{f}}"  # noqa: E731
                print(f"{k:>4g}  {scen:<12}{fam:<10}{mu['survival']:>6.0%}"
                      f"{cell('dv_mps', '.3f'):>18}{cell('mean_dist_km', '.2f'):>18}"
                      f"{cell('best_classical_ratio', '.3f'):>18}"
                      f"{cell('envelope_ratio', '.3f'):>16}{mu['dominated']:>11.0%}")
        print()
    plot(rows, classical)
    print("Saved experiments/phase8.csv, figures/phase8_pareto.png")


def plot(rows, classical):
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
            r = next(r for r in rows if r["family"] == "RES base" and r["k"] == k
                     and r["scenario"] == scen)
            ax.plot(r["mean_dist_km"], r["dv_mps"], "X", color=COLORS["RES base"], ms=9,
                    label="RES base (Bias-LQR rho = 1)")
            for fam, models in RL.items():
                for seed in models:
                    r = next(r for r in rows if r["family"] == fam and r["k"] == k
                             and r["scenario"] == scen and r["train_seed"] == seed)
                    if r["survival"] < 1.0:
                        continue
                    ax.plot(r["mean_dist_km"], r["dv_mps"], MARKERS[seed], color=COLORS[fam],
                            ms=6, mec="white", mew=0.6, alpha=0.9,
                            label=f"{fam}" if seed == 0 else None)
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
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Phase 8: RL with memory (MEM-PPO) and residual RL on Bias-LQR (RES-PPO)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(ROOT / "figures" / "phase8_pareto.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
