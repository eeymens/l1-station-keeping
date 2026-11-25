"""Phase 9: confirm the Moon-trained DR-PPO agents below the envelope on fresh seeds.

scripts/19 left two DR-PPO-M agents (training seeds 2 and 4) not dominated at
k = 1 + Moon while holding, 1.4-1.7 % below the dense classical envelope on the
test seeds. As in phase 8 (scripts/14), margins this small are re-tested on
100 fresh seeds (200-299) that no model selection or design choice has seen:
the two agents and every dense Harmonic-LQR / Bias-LQR design of scripts/19
whose test-seed mean distance lies between 12 and 400 km (the agents hold at
29 and 170 km) fly the same seeds, holding, k = 1 + Moon.

Writes experiments/moon_confirmation.csv and experiments/moon_confirmation_classical.csv.

Run from the repository root (~25 min on two cores; needs scripts/19):
    python scripts/20_moon_confirmation.py
"""

import csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from cr3bp.control import BiasLQRController, HarmonicLQRController
from cr3bp.evaluation import SCENARIOS, evaluate, summarize
from cr3bp.training import load_policy

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
SEEDS = range(200, 300)
H = {"holding": SCENARIOS["holding"]}
FAM = {"Harmonic-LQR": HarmonicLQRController, "Bias-LQR": BiasLQRController}
def job(d):
    fam, r, rho = d
    s = summarize(evaluate(FAM[fam](rho=rho, r_ref_km=r), seeds=SEEDS, scenarios=H,
                           perturbation_k=1.0, moon=1.0))["holding"]
    return {"family": fam, "r_ref_km": r, "rho": rho, **{k: s[k] for k in ("survival","dv_mps","mean_dist_km")}}
if __name__ == "__main__":
    dense = [r for r in csv.DictReader(open(EXP / "moon_frontier_designs.csv", newline=""))
             if r["condition"] == "k=1 + Moon" and float(r["survival"]) == 1.0]
    lo, hi = 12.0, 400.0                      # around the two agents (29 and 170 km)
    designs = [(r["family"], float(r["r_ref_km"]), float(r["rho"])) for r in dense
               if lo <= float(r["mean_dist_km"]) <= hi]
    print(len(designs), "designs", flush=True)
    with Pool(2) as p:
        res = p.map(job, designs, chunksize=4)
    out = []
    for s in (2, 4):
        c = load_policy(ROOT / "models", f"ppo_drm_s{s}")[1]
        m = summarize(evaluate(c, seeds=SEEDS, scenarios=H, perturbation_k=1.0, moon=1.0))["holding"]
        ok = [x for x in res if x["survival"] == 1.0]
        dom = any(x["mean_dist_km"] <= m["mean_dist_km"] and x["dv_mps"] <= m["dv_mps"] for x in ok)
        # envelope interpolation
        pts = sorted(ok, key=lambda x: x["mean_dist_km"]); env=[]; best=np.inf
        for x in pts:
            if x["dv_mps"] < best: env.append(x); best = x["dv_mps"]
        ex = np.log10([x["mean_dist_km"] for x in env]); ey = np.log10([x["dv_mps"] for x in env])
        e = m["dv_mps"] / 10**np.interp(np.log10(m["mean_dist_km"]), ex, ey)
        print(f"ppo_drm_s{s}: dv {m['dv_mps']:.3f} dist {m['mean_dist_km']:.1f} survival {m['survival']} envelope {e:.3f} dominated {dom}", flush=True)
        out.append({"model": f"ppo_drm_s{s}", **{k: m[k] for k in ("survival","dv_mps","mean_dist_km")}, "envelope_ratio": e, "dominated": float(dom)})
    with open(EXP / "moon_confirmation.csv", "w", newline="") as fh:
        w=csv.DictWriter(fh, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    with open(EXP / "moon_confirmation_classical.csv", "w", newline="") as fh:
        w=csv.DictWriter(fh, fieldnames=res[0].keys()); w.writeheader(); w.writerows(res)
