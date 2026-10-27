"""Common evaluation protocol for every controller (LQR, DQN, PPO).

Each controller flies the same scenarios with the same random seeds, so the
navigation and thrust errors it meets are identical draw for draw.
"""

import numpy as np

from .control.station_keeping import StationKeepingSim, run_episode

#: Scenario name -> starting offset from L1 in km (spacecraft at rest).
SCENARIOS = {
    "holding": (0.0, 0.0),
    "acquisition": (1000.0, 0.0),
}


def evaluate(controller, scenarios=SCENARIOS, seeds=range(5), days=365, noise=None):
    """Fly `controller` on every scenario x seed. Returns a list of result rows."""
    rows = []
    for name, offset in scenarios.items():
        for seed in seeds:
            sim = StationKeepingSim(seed=seed, noise=noise)
            r = run_episode(sim, controller, days=days, offset_km=offset)
            rows.append({
                "scenario": name, "seed": seed,
                "survived": r["survived"], "days_survived": r["days_survived"],
                "dv_mps": r["dv_total_mps"], "mean_dist_km": r["mean_dist_km"],
                "final_dist_km": r["final_dist_km"],
            })
    return rows


def summarize(rows):
    """Per-scenario means: survival rate, delta-v, mean distance."""
    out = {}
    for name in dict.fromkeys(r["scenario"] for r in rows):
        sel = [r for r in rows if r["scenario"] == name]
        out[name] = {
            "survival": float(np.mean([r["survived"] for r in sel])),
            "days": float(np.mean([r["days_survived"] for r in sel])),
            "dv_mps": float(np.mean([r["dv_mps"] for r in sel])),
            "dv_sd": float(np.std([r["dv_mps"] for r in sel])),
            "mean_dist_km": float(np.mean([r["mean_dist_km"] for r in sel])),
            "dist_sd": float(np.std([r["mean_dist_km"] for r in sel])),
        }
    return out
