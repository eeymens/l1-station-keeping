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
from .perturbations import Perturbation

#: Scenario name -> starting offset from L1 in km (spacecraft at rest).
SCENARIOS = {
    "holding": (0.0, 0.0),
    "acquisition": (1000.0, 0.0),
}

#: Seeds reserved for final tests; never used in training or model selection.
TEST_SEEDS = range(100, 120)


def perturbation_for(k, seed, parts=("e", "srp"), moon=0.0):
    """Phase 7: the perturbation flown with episode seed `seed` at severity k.

    Earth's starting true anomaly is drawn from its own generator (seed + 7919),
    so the navigation and thrust errors stay identical draw for draw across
    controllers *and* across k. ``parts`` switches the two effects on
    individually. k = 0 returns None (original CR3BP code path).

    Phase 9: ``moon`` > 0 adds the Moon (see `cr3bp.perturbations`), scaled by
    ``moon``. Its starting angle comes from a third generator (seed + 104729),
    so it changes none of the draws above.
    """
    if k == 0 and moon == 0:
        return None
    f0 = float(np.random.default_rng(seed + 7919).uniform(0.0, 2 * np.pi))
    p = Perturbation.from_k(k, f0=f0)
    if moon:
        p.moon = float(moon)
        p.moon_phase0 = float(np.random.default_rng(seed + 104729).uniform(0.0, 2 * np.pi))
    if "e" not in parts:
        p.e = 0.0
    if "srp" not in parts:
        p.srp_mps2 = 0.0
    return p


def evaluate(controller, scenarios=SCENARIOS, seeds=range(5), days=365, noise=None,
             perturbation_k=0.0, parts=("e", "srp"), moon=0.0):
    """Fly `controller` on every scenario x seed. Returns a list of result rows.

    ``perturbation_k`` > 0 flies the phase-7 perturbed dynamics (see
    `perturbation_for`); the default 0 is the nominal CR3BP of phases 3-6.
    """
    rows = []
    for name, offset in scenarios.items():
        for seed in seeds:
            sim = StationKeepingSim(seed=seed, noise=noise,
                                    perturbation=perturbation_for(perturbation_k, seed, parts,
                                                                 moon))
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
