"""Phase 9: is the Moon-trained RL really below the classical frontier?

In scripts/18, some DR-PPO-M and RES-PPO-M agents (trained with the Moon) are
not dominated by any of the 115 designs per classical family while holding, a
few by up to ~2 % against the interpolated envelope. As in phase 8
(scripts/15), margins this small can be an artefact of a coarse design grid,
so the two disturbance-estimating families are searched densely here:

    Harmonic-LQR and Bias-LQR, r_ref in 11 values from 10 to 1000 km (log)
    x rho in 41 values from 1e-3 to 1e4 (log) = 451 designs per family

flown on the same test seeds (100-119), holding scenario, at both Moon
conditions. Every agent is then re-checked against all classical designs
(scripts/18's sweep + this one).

Writes experiments/moon_frontier_designs.csv and experiments/moon_frontier.csv.

Run from the repository root (~25 min on two cores; needs scripts/18):
    python scripts/19_moon_frontier.py [--workers 2]
"""

import argparse
import csv
import importlib.util
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cr3bp.control import BiasLQRController, HarmonicLQRController
from cr3bp.evaluation import SCENARIOS, TEST_SEEDS, evaluate, summarize

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
_spec = importlib.util.spec_from_file_location("moon", ROOT / "scripts" / "18_moon.py")
moon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moon)

FAMILIES = {"Harmonic-LQR": HarmonicLQRController, "Bias-LQR": BiasLQRController}
R_REFS = np.logspace(1, 3, 11)
RHOS = np.logspace(-3, 4, 41)
AGENTS = ("DR-PPO-M", "RES-PPO-M", "RES-PPO", "MEM-PPO", "DR-PPO", "LSTM-PPO")


def _job(job):
    cond, fam, r_ref, rho = job
    k, m = moon.CONDITIONS[cond]
    s = summarize(evaluate(FAMILIES[fam](rho=rho, r_ref_km=r_ref), seeds=TEST_SEEDS,
                           scenarios={"holding": SCENARIOS["holding"]},
                           perturbation_k=k, moon=m))["holding"]
    return {"condition": cond, "family": fam, "r_ref_km": r_ref, "rho": rho,
            "scenario": "holding", **{key: s[key] for key in moon.mm.METRICS}}


def sweep(workers):
    jobs = [(c, f, float(r), float(p)) for c in moon.CONDITIONS for f in FAMILIES
            for r in R_REFS for p in RHOS]
    print(f"{len(jobs)} design x condition runs (holding) on {workers} workers", flush=True)
    dense = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap(_job, jobs, chunksize=8)):
            dense.append(row)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    moon.mm.write_csv(EXP / "moon_frontier_designs.csv", dense)
    return moon.read_classical(EXP / "moon_frontier_designs.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reuse-designs", action="store_true",
                    help="reuse experiments/moon_frontier_designs.csv, only re-check the agents")
    args = ap.parse_args()
    dpath = EXP / "moon_frontier_designs.csv"
    if args.reuse_designs and dpath.exists():
        dense = moon.read_classical(dpath)
    else:
        dense = sweep(args.workers)

    coarse = moon.read_classical(EXP / "moon_classical.csv")
    agents = [r for r in csv.DictReader(open(EXP / "moon.csv", newline=""))
              if r["family"] in AGENTS and r["scenario"] == "holding"
              and r["train_seed"] not in ("mean", "std", "-1")]
    out = []
    for r in agents:
        m = {key: float(r[key]) for key in moon.mm.METRICS}
        pts = [p for p in coarse + dense if p["condition"] == r["condition"]
               and p["scenario"] == "holding"]
        c = moon.compare(m, pts)
        out.append({"condition": r["condition"], "family": r["family"],
                    "train_seed": int(r["train_seed"]), "dv_mps": m["dv_mps"],
                    "mean_dist_km": m["mean_dist_km"],
                    "envelope_ratio_coarse": float(r["envelope_ratio"]),
                    "dominated_coarse": float(r["dominated"]),
                    "best_classical_ratio": c["best_classical_ratio"],
                    "envelope_ratio": c["envelope_ratio"], "dominated": c["dominated"]})
    moon.mm.write_csv(EXP / "moon_frontier.csv", out)

    print(f"\nHolding, test seeds; classical = scripts/18 sweep + {len(dense) // 2} dense "
          "designs per condition\n")
    print(f"{'condition':<12}{'agent':<11}{'seed':>5}{'dv':>8}{'km':>8}{'env (18)':>10}"
          f"{'env (dense)':>13}{'dominated':>11}")
    for r in out:
        if r["family"] in ("DR-PPO-M", "RES-PPO-M"):
            print(f"{r['condition']:<12}{r['family']:<11}{r['train_seed']:>5}"
                  f"{r['dv_mps']:>8.3f}{r['mean_dist_km']:>8.1f}"
                  f"{r['envelope_ratio_coarse']:>10.3f}{r['envelope_ratio']:>13.3f}"
                  f"{'yes' if r['dominated'] else 'no':>11}")
    for fam in dict.fromkeys(r["family"] for r in out):
        sel = [r for r in out if r["family"] == fam]
        print(f"{fam}: not dominated in {sum(r['dominated'] == 0 for r in sel)} of {len(sel)}")
    print("Saved experiments/moon_frontier_designs.csv, experiments/moon_frontier.csv")


if __name__ == "__main__":
    main()
