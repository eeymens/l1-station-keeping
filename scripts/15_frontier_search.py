"""Phase 8, last check: give the classical side its best local tuning.

scripts/14 puts the residual agents (RES-PPO) 0.04-0.6 % *below* the classical
envelope of 1710 designs on confirmation seeds 200-299. The envelope there is
very flat, and the classical grid steps rho by a factor 1.33, so a better
classical design could hide between grid points.

Bias-LQR's Q and R are set by two scales relative to the fuel weight: the
position weight ~ 1 / (rho r_ref^2) and the velocity weight ~ 1 / rho (v_ref
fixed). The envelope designs near the agents line up along roughly constant
c = rho (r_ref / 1000 km)^2. This script therefore searches the 2-D design
space densely around them: r_ref = logspace(10, 1000 km, 21) and, for each,
rho = c / (r_ref / 1000)^2 with c on 21 log-spaced values spanning the c of
the nearby envelope designs widened 3x either way (441 designs per k).

Everything runs on the confirmation seeds 200-299, holding, k = 0.5 and 1
(where scripts/14 found the largest margins). The agents' margin is then
recomputed against the union of these designs and the scripts/14 designs, with
a paired bootstrap (2000 resamples) over the dense designs and the agents.

Writes experiments/frontier_search.csv (per agent) and
experiments/frontier_search_designs.csv.

Run from the repository root (~40 min on two cores; needs scripts/14):
    python scripts/15_frontier_search.py
"""

import csv
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("confirm", ROOT / "scripts" / "14_confirmation.py")
cf = importlib.util.module_from_spec(_spec)
sys.modules["confirm"] = cf                      # so worker processes can pickle its functions
_spec.loader.exec_module(cf)

KS = (0.5, 1.0)
R_GRID = np.logspace(1, 3, 21)


def c_range(stage1, agents, k):
    """c = rho (r_ref/1000)^2 of the envelope designs bracketing the agents."""
    pts = sorted((r for r in stage1 if r["k"] == k and r["family"] == "Bias-LQR"
                  and r["survival"] == 1.0), key=lambda r: (r["mean_dist_km"], r["dv_mps"]))
    env, best = [], np.inf
    for p in pts:
        if p["dv_mps"] < best:
            env.append(p)
            best = p["dv_mps"]
    dmin = min(a["mean_dist_km"] for a in agents if a["k"] == k)
    dmax = max(a["mean_dist_km"] for a in agents if a["k"] == k)
    near = [p for p in env if dmin / 2 <= p["mean_dist_km"] <= dmax * 2]
    cs = [p["rho"] * (p["r_ref_km"] / 1000.0) ** 2 for p in near]
    return min(cs) / 3, max(cs) * 3


def main():
    stage1 = cf.mm.read_csv(ROOT / "experiments" / "confirmation_classical.csv")
    conf = cf.mm.read_csv(ROOT / "experiments" / "confirmation.csv")
    jobs = []
    for k in KS:
        lo, hi = c_range(stage1, conf, k)
        print(f"k = {k:g}: c in [{lo:.3g}, {hi:.3g}]", flush=True)
        for r in R_GRID:
            for c in np.logspace(np.log10(lo), np.log10(hi), 21):
                jobs.append((k, "Bias-LQR", float(r), float(c / (r / 1000.0) ** 2)))
    print(f"{len(jobs)} designs x {len(cf.SEEDS)} seeds", flush=True)
    with Pool(2) as pool:
        designs = dict(pool.map(cf.run_design, jobs, chunksize=8))
        agents = dict(pool.map(cf.run_agent, [(k, a) for k in KS for a in cf.AGENTS]))

    with open(ROOT / "experiments" / "frontier_search_designs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "family", "r_ref_km", "rho", "survival", "dv_mps", "mean_dist_km"])
        for (k, fam, r, rho), (dv, d, ok) in designs.items():
            w.writerow([k, fam, r, rho, ok.mean(), dv.mean(), d.mean()])

    rng = np.random.default_rng(1)
    out = []
    for k in KS:
        keys = [j for j in designs if j[0] == k and designs[j][2].all()]
        DV = np.array([designs[j][0] for j in keys])
        DI = np.array([designs[j][1] for j in keys])
        s1 = [r for r in stage1 if r["k"] == k and r["survival"] == 1.0]
        s1dv = np.array([r["dv_mps"] for r in s1])
        s1di = np.array([r["mean_dist_km"] for r in s1])
        boots = rng.integers(0, len(cf.SEEDS), (cf.N_BOOT, len(cf.SEEDS)))
        for name in cf.AGENTS:
            adv, adi, aok = agents[(k, name)]
            m_dv, m_di = adv.mean(), adi.mean()
            cdv = np.r_[DV.mean(1), s1dv]
            cdi = np.r_[DI.mean(1), s1di]
            ratio = m_dv / cf.envelope_at(cdi, cdv, m_di)
            dom = (cdi <= m_di) & (cdv <= m_dv)
            # cheapest design at least as accurate (staircase), and the margin to it
            acc = cdi <= m_di
            stair = m_dv / cdv[acc].min() if acc.any() else np.nan
            br = []
            for b in boots:
                bdv, bdi = DV[:, b].mean(1), DI[:, b].mean(1)
                br.append(adv[b].mean() / cf.envelope_at(bdi, bdv, adi[b].mean()))
            br = np.array(br)
            br = br[np.isfinite(br)]
            lo, hi = np.percentile(br, [2.5, 97.5])
            out.append({"k": k, "model": name, "dv_mps": m_dv, "mean_dist_km": m_di,
                        "envelope_ratio": ratio, "staircase_ratio": stair,
                        "dense_ci_low": lo, "dense_ci_high": hi,
                        "dominated": float(dom.any()), "n_designs": len(cdv)})
            print(f"k = {k:g}  {name}: x envelope {ratio:.4f}, x staircase {stair:.4f}, "
                  f"dense-only CI [{lo:.4f}, {hi:.4f}], dominated: {bool(dom.any())}",
                  flush=True)
    with open(ROOT / "experiments" / "frontier_search.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)
    print("Saved experiments/frontier_search.csv, experiments/frontier_search_designs.csv")


if __name__ == "__main__":
    main()
