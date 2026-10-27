"""Common evaluation protocol for every controller (LQR, DQN, PPO).

Each controller flies the same scenarios with the same random seeds, so the
navigation and thrust errors it meets are identical draw for draw.

Metrics per episode
    dv_mps         fuel: total delta-v over the episode
    mean_dist_km   tracking error: mean distance from L1
    rms_dist_km    tracking error: root-mean-square distance (punishes excursions)
    max_dist_km    worst excursion
    days_survived  days before the spacecraft was lost (> 10,000 km), max = episode length
"""

import numpy as np

from .control.station_keeping import StationKeepingSim, run_episode

#: Scenario name -> starting offset from L1 in km (spacecraft at rest).
SCENARIOS = {
    "holding": (0.0, 0.0),
    "acquisition": (1000.0, 0.0),
}

#: Seeds reserved for final tests; never used in training or model selection.
TEST_SEEDS = range(100, 120)


def evaluate(controller, scenarios=SCENARIOS, seeds=range(5), days=365, noise=None):
    """Fly `controller` on every scenario x seed. Returns a list of result rows."""
    rows = []
    for name, offset in scenarios.items():
        for seed in seeds:
            sim = StationKeepingSim(seed=seed, noise=noise)
            r = run_episode(sim, controller, days=days, offset_km=offset)
            d = r["dist_km"]
            rows.append({
                "scenario": name, "seed": seed,
                "survived": r["survived"], "days_survived": r["days_survived"],
                "dv_mps": r["dv_total_mps"], "mean_dist_km": r["mean_dist_km"],
                "final_dist_km": r["final_dist_km"],
                "rms_dist_km": float(np.sqrt(np.mean(d**2))), "max_dist_km": float(d.max()),
            })
    return rows


def summarize(rows):
    """Per-scenario means: survival rate, delta-v, tracking error."""
    out = {}
    for name in dict.fromkeys(r["scenario"] for r in rows):
        sel = [r for r in rows if r["scenario"] == name]
        col = lambda k: np.array([r[k] for r in sel], dtype=float)  # noqa: E731
        out[name] = {
            "survival": float(col("survived").mean()),
            "days": float(col("days_survived").mean()),
            "dv_mps": float(col("dv_mps").mean()),
            "dv_sd": float(col("dv_mps").std()),
            "mean_dist_km": float(col("mean_dist_km").mean()),
            "dist_sd": float(col("mean_dist_km").std()),
        }
        if "rms_dist_km" in sel[0]:
            out[name]["rms_dist_km"] = float(col("rms_dist_km").mean())
            out[name]["max_dist_km"] = float(col("max_dist_km").mean())
    return out
