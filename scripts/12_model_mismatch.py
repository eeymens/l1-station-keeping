"""Phase 7: what happens when the real dynamics leave the controllers' model?

Every controller so far was designed for, or trained in, the planar circular
restricted three-body problem (CR3BP), and the main result of phases 5-6 is
that there LQR is Pareto-optimal: no PPO seed beats its fuel/accuracy curve.
LQR relies on a linearised CR3BP model; RL makes no such assumption. Does the
picture change once the "real" simulator contains physics the model lacks?

The simulator now adds (cr3bp.perturbations), scaled by a severity k:
    Earth's orbital eccentricity  e = k x 0.0167        (elliptic R3BP)
    solar radiation pressure      a = k x 5.9e-8 m/s^2  at 1 AU
k = 0 is the CR3BP of phases 3-6 (bit for bit), k = 1 the realistic values.
All controllers keep their nominal CR3BP model.

Controllers (all on test seeds 100-119, holding and 1,000 km acquisition)
    LQR       u = -K x, swept over rho (same 57 values as scripts/10) and over
              the Bryson position scale r_ref = 1000, 100, 10 km
    LQI       LQR + integral action (cancels a constant force, holds at L1)
    Bias-LQR  Kalman filter estimating the state and a constant force, LQR
              around the cheapest steady state (parks where gravity helps)
    PPO       nominal cubic agents ppo_l1, ppo_l1_s1, ppo_l1_s2 (phase 5),
              flown in the perturbed dynamics without retraining
    DR-PPO    ppo_dr_s0..s4: same PPO hyperparameters, trained with domain
              randomisation k ~ U(0, 4) (scripts/08 --dr-k-max 4)

Comparison metrics, per k and scenario
    lqr_ratio       model delta-v / LQR delta-v at the same mean distance,
                    interpolated on the LQR left branch exactly as in
                    scripts/10_seed_study.py (NaN outside the branch)
    best_classical_ratio
                    model delta-v / delta-v of the cheapest classical
                    controller (any family, any rho) that is at least as
                    accurate (staircase, no interpolation). < 1 means the
                    model beats every classical design at its accuracy.
    excess_dv_mps   the same comparison as a difference in m/s. Near L1 every
                    controller has to cancel SRP (~1.9 k m/s per year), so the
                    ratio tends to 1 for all of them; the excess shows what is
                    left once that common cost is paid.
    dominated       some classical point is both cheaper and at least as accurate

Effects separately: at k = 1 and k = 4 the script also flies "eccentricity
only" and "SRP only" for LQR (rho = 1), Bias-LQR (rho = 1) and every PPO model.

Writes
    experiments/model_mismatch.csv            per model x k x scenario + mean/std rows
    experiments/model_mismatch_classical.csv  the three classical sweeps
    experiments/model_mismatch_parts.csv      eccentricity-only / SRP-only runs
    figures/model_mismatch.png                delta-v and distance vs k
    figures/model_mismatch_pareto.png         fuel/accuracy plane for every k

Run from the repository root (~40 min on one core):
    python scripts/12_model_mismatch.py
    python scripts/12_model_mismatch.py --reuse-classical   # skip the sweeps
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from cr3bp.control import BiasLQRController, LQIController, LQRController
from cr3bp.evaluation import SCENARIOS, TEST_SEEDS, evaluate, summarize
from cr3bp.training import load_policy

ROOT = Path(__file__).resolve().parents[1]
KS = (0.0, 0.5, 1.0, 2.0, 4.0)
RHOS = np.logspace(-3, 4, 57)
#: Classical designs: Bryson position scale r_ref [km] -> fuel weights swept.
#: r_ref = 1000 km is the phase 3-6 design. Under strong mismatch its tightest
#: tracking is limited by the velocity weight, not by rho, so tighter position
#: scales are swept too (a classical design must not be left weak on purpose).
DESIGNS = {1000.0: RHOS, 100.0: np.logspace(-3, 4, 29), 10.0: np.logspace(-3, 4, 29)}
CLASSICAL = {"LQR": LQRController, "LQI": LQIController, "Bias-LQR": BiasLQRController}
RL = {
    "PPO": {0: "ppo_l1", 1: "ppo_l1_s1", 2: "ppo_l1_s2"},
    "DR-PPO": {s: f"ppo_dr_s{s}" for s in range(5)},
}
METRICS = ("survival", "days", "dv_mps", "mean_dist_km", "rms_dist_km")
COLORS = {"LQR": "#2471a3", "LQI": "#1e8449", "Bias-LQR": "#5d6d7e",
          "PPO": "#d68910", "DR-PPO": "#c0392b"}
MARKERS = ["o", "s", "^", "D", "v"]
PARTS = {"eccentricity only": ("e",), "SRP only": ("srp",)}

# the left-branch interpolation of the seed study, reused verbatim
_spec = importlib.util.spec_from_file_location("seed_study", ROOT / "scripts" / "10_seed_study.py")
_seed_study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seed_study)
left_branch, lqr_dv_at = _seed_study.left_branch, _seed_study.lqr_dv_at


def run_classical(designs):
    rows = []
    for r_ref, rhos in designs.items():
        for k in KS:
            for fam, cls in CLASSICAL.items():
                for rho in rhos:
                    summ = summarize(evaluate(cls(rho=rho, r_ref_km=r_ref), seeds=TEST_SEEDS,
                                              perturbation_k=k))
                    for scen, m in summ.items():
                        rows.append({"k": k, "family": fam, "r_ref_km": r_ref, "rho": rho,
                                     "scenario": scen, **{key: m[key] for key in METRICS}})
                print(f"  r_ref = {r_ref:g} km, k = {k:g}: {fam} sweep done", flush=True)
    return rows


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key, v in r.items():
            if key not in ("family", "scenario", "model", "effect", "vs_lqr_branch"):
                r[key] = float(v)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def staircase(points, dist):
    """Cheapest classical delta-v among points at least as accurate as `dist`."""
    ok = [p["dv_mps"] for p in points if p["mean_dist_km"] <= dist]
    return min(ok) if ok else float("nan")


def compare(m, k, scen, classical):
    """Comparison columns for one model result m at severity k."""
    pts = [c for c in classical if c["k"] == k and c["scenario"] == scen
           and c["survival"] == 1.0]
    lqr = sorted((c for c in classical if c["k"] == k and c["scenario"] == scen
                  and c["family"] == "LQR" and c["r_ref_km"] == 1000.0),
                 key=lambda c: c["rho"])
    try:
        branch = left_branch(lqr)
        ref = lqr_dv_at(m["mean_dist_km"], branch)
        x = branch[0]
        where = ("inside" if np.isfinite(ref) else
                 "less accurate" if np.log10(m["mean_dist_km"]) > x[-1] else "more accurate")
    except RuntimeError:                           # LQR curve not monotone at this k
        ref, where = float("nan"), "n/a"
    best = staircase(pts, m["mean_dist_km"])
    best_fam = {fam: staircase([p for p in pts if p["family"] == fam], m["mean_dist_km"])
                for fam in CLASSICAL}
    dominated = m["survival"] < 1.0 or any(
        p["mean_dist_km"] <= m["mean_dist_km"] and p["dv_mps"] <= m["dv_mps"] for p in pts)
    return {"lqr_dv_same_acc_mps": ref, "lqr_ratio": m["dv_mps"] / ref, "vs_lqr_branch": where,
            "best_classical_ratio": m["dv_mps"] / best,
            "excess_dv_mps": m["dv_mps"] - best,
            **{f"{fam}_ratio": m["dv_mps"] / v for fam, v in best_fam.items()},
            "dominated": float(dominated)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-classical", action="store_true",
                    help="read experiments/model_mismatch_classical.csv instead of sweeping")
    ap.add_argument("--classical-only", action="store_true",
                    help="run and save the classical sweeps, then stop")
    args = ap.parse_args()
    exp = ROOT / "experiments"
    cpath = exp / "model_mismatch_classical.csv"

    # ------------------------------------------------------------ classical sweeps
    classical = read_csv(cpath) if args.reuse_classical and cpath.exists() else []
    done = {c["r_ref_km"] for c in classical}
    todo = {r: rhos for r, rhos in DESIGNS.items() if r not in done}
    if todo:
        print(f"Classical sweeps: {list(CLASSICAL)} x r_ref {list(todo)} km x k in {KS}",
              flush=True)
        classical += run_classical(todo)
        write_csv(cpath, classical)
    if args.classical_only:
        return

    # ------------------------------------------------------------ RL models
    rows = []
    policies = {name: load_policy(ROOT / "models", name)[1]
                for models in RL.values() for name in models.values()}
    for k in KS:
        for fam, models in RL.items():
            for seed, name in models.items():
                summ = summarize(evaluate(policies[name], seeds=TEST_SEEDS, perturbation_k=k))
                for scen, m in summ.items():
                    rows.append({"family": fam, "train_seed": seed, "model": name, "k": k,
                                 "scenario": scen, **{key: m[key] for key in METRICS},
                                 **compare(m, k, scen, classical)})
        print(f"  k = {k:g}: RL models evaluated", flush=True)

    # classical reference rows at rho = 1 (the default design) for the tables
    for c in classical:
        if np.isclose(c["rho"], 1.0) and c["r_ref_km"] == 1000.0:
            m = {key: c[key] for key in METRICS}
            rows.append({"family": c["family"], "train_seed": -1, "model": "rho=1",
                         "k": c["k"],
                         "scenario": c["scenario"], **m,
                         **compare(m, c["k"], c["scenario"], classical)})

    keys = (*METRICS, "lqr_ratio", "best_classical_ratio", "excess_dv_mps",
            *[f"{fam}_ratio" for fam in CLASSICAL], "dominated")
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
    write_csv(exp / "model_mismatch.csv", rows + agg)

    # ------------------------------------------------------------ effects separately
    parts_rows = []
    for k in (1.0, 4.0):
        for effect, parts in PARTS.items():
            ctrls = {("LQR", "rho=1"): LQRController(rho=1.0),
                     ("Bias-LQR", "rho=1"): BiasLQRController(rho=1.0),
                     **{(fam, name): policies[name] for fam, models in RL.items()
                        for name in models.values()}}
            for (fam, name), ctrl in ctrls.items():
                summ = summarize(evaluate(ctrl, seeds=TEST_SEEDS, perturbation_k=k,
                                          parts=parts))
                for scen, m in summ.items():
                    parts_rows.append({"k": k, "effect": effect, "family": fam, "model": name,
                                       "scenario": scen, **{key: m[key] for key in METRICS}})
        print(f"  k = {k:g}: effect split done", flush=True)
    write_csv(exp / "model_mismatch_parts.csv", parts_rows)

    # ------------------------------------------------------------ table
    print(f"\nTest seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1}; RL rows are mean ± std "
          "over training seeds; classical rows are rho = 1\n")
    print(f"{'k':>4}  {'scenario':<12}{'method':<10}{'kept':>6}{'delta-v [m/s]':>18}"
          f"{'mean [km]':>20}{'x LQR':>14}{'x best classical':>18}{'dominated':>11}")
    for k in KS:
        for scen in SCENARIOS:
            for fam in CLASSICAL:
                r = next(r for r in rows if r["family"] == fam and r["k"] == k
                         and r["scenario"] == scen)
                print(f"{k:>4g}  {scen:<12}{fam:<10}{r['survival']:>6.0%}{r['dv_mps']:>18.3f}"
                      f"{r['mean_dist_km']:>20.2f}{r['lqr_ratio']:>14.2f}"
                      f"{r['best_classical_ratio']:>18.2f}{r['dominated']:>11.0f}")
            for fam in RL:
                mu = next(a for a in agg if a["family"] == fam and a["k"] == k
                          and a["scenario"] == scen and a["train_seed"] == "mean")
                sd = next(a for a in agg if a["family"] == fam and a["k"] == k
                          and a["scenario"] == scen and a["train_seed"] == "std")
                cell = lambda key, f: f"{mu[key]:{f}} ± {sd[key]:{f}}"  # noqa: E731
                print(f"{k:>4g}  {scen:<12}{fam:<10}{mu['survival']:>6.0%}"
                      f"{cell('dv_mps', '.3f'):>18}{cell('mean_dist_km', '.2f'):>20}"
                      f"{cell('lqr_ratio', '.2f'):>14}{cell('best_classical_ratio', '.2f'):>18}"
                      f"{mu['dominated']:>11.0%}")
        print()

    plot_vs_k(rows)
    plot_pareto(rows, classical)
    print("Saved experiments/model_mismatch.csv, experiments/model_mismatch_classical.csv, "
          "experiments/model_mismatch_parts.csv, figures/model_mismatch.png, "
          "figures/model_mismatch_pareto.png")


def pareto_envelope(points):
    """Points not dominated by any other (cheaper and at least as accurate),
    sorted by distance: the lower-left boundary of a design sweep."""
    pts = sorted(points, key=lambda p: (p["mean_dist_km"], p["dv_mps"]))
    env, best = [], float("inf")
    for p in pts:
        if p["dv_mps"] < best:
            env.append(p)
            best = p["dv_mps"]
    return env


def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})


def plot_vs_k(rows):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    titles = {"holding": "Holding at L1", "acquisition": "Recovering from 1,000 km"}
    offsets = {"LQR": -0.09, "LQI": -0.045, "Bias-LQR": 0.0, "PPO": 0.045, "DR-PPO": 0.09}
    for row_i, scen in enumerate(SCENARIOS):
        for col, key, label in ((0, "dv_mps", "delta-v over one year [m/s]"),
                                (1, "mean_dist_km", "mean distance from L1 [km]")):
            ax = axes[row_i, col]
            for fam in (*CLASSICAL, *RL):
                sel = [r for r in rows if r["family"] == fam and r["scenario"] == scen]
                xk = lambda r: np.log2(r["k"]) if r["k"] > 0 else -2  # noqa: E731
                if fam in CLASSICAL:
                    sel = sorted(sel, key=lambda r: r["k"])
                    ax.plot([xk(r) + offsets[fam] for r in sel], [r[key] for r in sel], "-o",
                            color=COLORS[fam], ms=5, lw=1.5, label=f"{fam} (rho = 1)")
                else:
                    for seed in sorted({r["train_seed"] for r in sel}):
                        s = sorted((r for r in sel if r["train_seed"] == seed),
                                   key=lambda r: r["k"])
                        ax.plot([xk(r) + offsets[fam] for r in s], [r[key] for r in s],
                                MARKERS[int(seed)], color=COLORS[fam], ms=6, mec="white",
                                mew=0.8, alpha=0.9,
                                label=f"{fam}, training seed {int(seed)}")
                    means = [np.mean([r[key] for r in sel if r["k"] == k]) for k in KS]
                    ax.plot([(np.log2(k) if k > 0 else -2) + offsets[fam] for k in KS], means,
                            "-", color=COLORS[fam], lw=1.2, alpha=0.7)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_ylabel(label)
            ax.set_title(titles[scen])
            ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
            ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xticklabels(["0\n(CR3BP)", "0.5", "1\n(realistic)", "2", "4"])
        ax.set_xlabel("model-mismatch severity k (eccentricity and SRP x k)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="center right", frameon=False, fontsize=8)
    fig.suptitle(f"Controllers designed for the CR3BP, flown in perturbed dynamics "
                 f"(test seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(ROOT / "figures" / "model_mismatch.png", dpi=160)
    plt.close(fig)


def plot_pareto(rows, classical):
    _style()
    fig, axes = plt.subplots(2, len(KS), figsize=(4.0 * len(KS), 8))
    for row_i, scen in enumerate(SCENARIOS):
        for col, k in enumerate(KS):
            ax = axes[row_i, col]
            for fam in CLASSICAL:
                pts = [c for c in classical if c["k"] == k and c["scenario"] == scen
                       and c["family"] == fam and c["survival"] == 1.0]
                ax.plot([p["mean_dist_km"] for p in pts], [p["dv_mps"] for p in pts], ".",
                        color=COLORS[fam], ms=2, alpha=0.35)
                env = pareto_envelope(pts)
                ax.plot([p["mean_dist_km"] for p in env], [p["dv_mps"] for p in env], "-",
                        color=COLORS[fam], lw=1.6, label=f"{fam}, best over rho and r_ref")
            for fam, models in RL.items():
                for seed in models:
                    r = next(r for r in rows if r["family"] == fam and r["k"] == k
                             and r["scenario"] == scen and r["train_seed"] == seed)
                    if r["survival"] < 1.0:
                        continue
                    ax.plot(r["mean_dist_km"], r["dv_mps"], MARKERS[seed], color=COLORS[fam],
                            ms=7, mec="white", mew=0.8, label=f"{fam}, seed {seed}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_title(f"{'Holding' if scen == 'holding' else 'Acquisition'}, k = {k:g}",
                         fontsize=10)
            ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel("delta-v over one year [m/s]")
            if row_i == 1:
                ax.set_xlabel("mean distance from L1 [km]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Fuel/accuracy trade-off under growing model mismatch "
                 "(lower left is better)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(ROOT / "figures" / "model_mismatch_pareto.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
