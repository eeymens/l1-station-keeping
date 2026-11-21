"""Phase 8 follow-up: is RES-PPO really on (or past) the classical frontier?

scripts/13 found that the residual agents (RES-PPO, a PPO correction on top of
Bias-LQR) are not dominated by any of the 345 classical designs while holding
at k = 0.5, 1 and 2, and sit 0.1-0.5 % below the interpolated classical
envelope. Margins that small could come from (a) the 20 test seeds, or (b) the
coarse classical grid (4 designs per decade of rho for r_ref = 100 and 10 km).

This script re-tests the claim on fresh evidence:
    * 100 new confirmation seeds (200-299), never used anywhere before
    * a denser classical sweep of the family that dominated in phase 7,
      Bias-LQR: r_ref in {1000, 300, 100, 30, 10} km x 57 values of rho
      (285 designs per k), plus LQI over the same grid
    * a bootstrap over the 100 seeds (2000 resamples): every resample
      recomputes all design means, the envelope and each agent's ratio, so the
      interval covers the seed noise in both the agent and the classical side

Reported per k and RES-PPO training seed: envelope_ratio (agent delta-v /
classical envelope at the agent's mean distance) with a 95 % bootstrap
interval, and whether any classical design dominates the agent's mean.

Writes experiments/confirmation.csv and experiments/confirmation_classical.csv.

Run from the repository root (~45 min on two cores):
    python scripts/14_confirmation.py
"""

import csv
import importlib.util
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cr3bp.control import BiasLQRController, LQIController
from cr3bp.evaluation import evaluate
from cr3bp.training import load_policy

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mismatch", ROOT / "scripts" / "12_model_mismatch.py")
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)

SEEDS = range(200, 300)
KS = (0.5, 1.0, 2.0)
HOLD = {"holding": (0.0, 0.0)}
RREFS = (1000.0, 300.0, 100.0, 30.0, 10.0)
RHOS = np.logspace(-3, 4, 57)
FAMILIES = {"Bias-LQR": BiasLQRController, "LQI": LQIController}
AGENTS = [f"ppo_res_s{s}" for s in range(5)]
N_BOOT = 2000


def per_seed(rows):
    return (np.array([r["dv_mps"] for r in rows]), np.array([r["mean_dist_km"] for r in rows]),
            np.array([r["survived"] for r in rows], dtype=bool))


def run_design(job):
    k, fam, r_ref, rho = job
    rows = evaluate(FAMILIES[fam](rho=rho, r_ref_km=r_ref), scenarios=HOLD, seeds=SEEDS,
                    perturbation_k=k)
    return job, per_seed(rows)


def run_agent(job):
    k, name = job
    rows = evaluate(load_policy(ROOT / "models", name)[1], scenarios=HOLD, seeds=SEEDS,
                    perturbation_k=k)
    return job, per_seed(rows)


def envelope_at(dist, dvs, d):
    """Pareto envelope of (dist, dv) points, log-log interpolated at d."""
    order = np.lexsort((dvs, dist))
    x, y, best = [], [], np.inf
    for i in order:
        if dvs[i] < best:
            x.append(dist[i])
            y.append(dvs[i])
            best = dvs[i]
    lx = np.log10(d)
    x, y = np.log10(x), np.log10(y)
    if lx < x[0] or lx > x[-1]:
        return np.nan
    return 10 ** np.interp(lx, x, y)


def main():
    jobs = [(k, fam, r, rho) for k in KS for fam in FAMILIES for r in RREFS for rho in RHOS]
    print(f"{len(jobs)} classical designs x {len(SEEDS)} seeds, {len(AGENTS)} agents", flush=True)
    with Pool(2) as pool:
        designs = dict(pool.map(run_design, jobs, chunksize=8))
        agents = dict(pool.map(run_agent, [(k, a) for k in KS for a in AGENTS]))

    with open(ROOT / "experiments" / "confirmation_classical.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "family", "r_ref_km", "rho", "survival", "dv_mps", "mean_dist_km"])
        for (k, fam, r, rho), (dv, d, ok) in designs.items():
            w.writerow([k, fam, r, rho, ok.mean(), dv.mean(), d.mean()])

    rng = np.random.default_rng(0)
    out = []
    for k in KS:
        keys = [j for j in designs if j[0] == k]
        DV = np.array([designs[j][0] for j in keys])          # (designs, seeds)
        DI = np.array([designs[j][1] for j in keys])
        OK = np.array([designs[j][2].all() for j in keys])
        DV, DI = DV[OK], DI[OK]
        boots = rng.integers(0, len(SEEDS), (N_BOOT, len(SEEDS)))
        for name in AGENTS:
            adv, adi, aok = agents[(k, name)]
            m_dv, m_di = adv.mean(), adi.mean()
            cdv, cdi = DV.mean(1), DI.mean(1)
            ratio = m_dv / envelope_at(cdi, cdv, m_di)
            dominated = bool(np.any((cdi <= m_di) & (cdv <= m_dv)))
            br = []
            for b in boots:
                bdv, bdi = DV[:, b].mean(1), DI[:, b].mean(1)
                br.append(adv[b].mean() / envelope_at(bdi, bdv, adi[b].mean()))
            br = np.array(br)
            br = br[np.isfinite(br)]
            lo, hi = np.percentile(br, [2.5, 97.5])
            out.append({"k": k, "model": name, "survival": aok.mean(), "dv_mps": m_dv,
                        "mean_dist_km": m_di, "envelope_ratio": ratio, "ci_low": lo,
                        "ci_high": hi, "p_below_1": float(np.mean(br < 1.0)),
                        "dominated": float(dominated)})
            print(f"k = {k:g}  {name}: {m_dv:.3f} m/s at {m_di:.2f} km  x envelope {ratio:.4f} "
                  f"[{lo:.4f}, {hi:.4f}]  P(<1) = {np.mean(br < 1):.2f}  "
                  f"dominated: {dominated}", flush=True)
    with open(ROOT / "experiments" / "confirmation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)
    print("Saved experiments/confirmation.csv, experiments/confirmation_classical.csv")


if __name__ == "__main__":
    main()
