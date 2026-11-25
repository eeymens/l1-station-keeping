"""Phase 9: the Moon, a disturbance that is not a slowly varying bias.

Phases 7-8 found that under solar radiation pressure (SRP) the classical side
stays on the Pareto frontier, because a Kalman bias estimator captures an
almost constant force. The paper's discussion argued that RL could gain where
the missing physics "cannot be summarised as a slowly varying bias". The Moon
is such a case: near L1 its tidal pull (~1e-7 m/s^2, peaks ~6e-7) oscillates
with the synodic month and its overtones (see cr3bp.perturbations).

No controller was designed or trained with the Moon. All of them meet it here
for the first time:
    classical  LQR, LQI, Bias-LQR (phase 7) and Harmonic-LQR, the textbook
               classical fix for a disturbance of known frequency (internal
               model principle, cr3bp.control.harmonic), each swept over the
               same 115 (rho, r_ref) designs as in phase 7
    RL         DR-PPO, MEM-PPO, RES-PPO (phases 7-8) and LSTM-PPO (phase 9,
               if its models exist), 5 training seeds each

Conditions (test seeds 100-119, one year):
    Moon        k = 0 + Moon: the CR3BP plus the Moon only
    k=1 + Moon  real eccentricity and SRP (phase 7, k = 1) plus the Moon

Columns as in scripts/12 and 13 (best_classical_ratio against all four classical
families, envelope_ratio, dominated) and, for the RL families,
    vs_no_moon   delta-v / the same agent's delta-v without the Moon (same k)

Writes experiments/moon_classical.csv, experiments/moon.csv and
figures/moon_pareto.png.

Run from the repository root (~30 min on two cores):
    python scripts/18_moon.py [--reuse-classical] [--workers 2]
"""

import argparse
import csv
import importlib.util
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from cr3bp.control import (BiasLQRController, HarmonicLQRController, LQIController,
                           LQRController)
from cr3bp.evaluation import SCENARIOS, TEST_SEEDS, evaluate, summarize
from cr3bp.training import load_policy

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mm = _load("mismatch", "12_model_mismatch.py")

CONDITIONS = {"Moon": (0.0, 1.0), "k=1 + Moon": (1.0, 1.0)}     # name -> (k, moon)
CLASSICAL = {"LQR": LQRController, "LQI": LQIController, "Bias-LQR": BiasLQRController,
             "Harmonic-LQR": HarmonicLQRController}
RL = {
    "DR-PPO": [f"ppo_dr_s{s}" for s in range(5)],
    "MEM-PPO": [f"ppo_mem_s{s}" for s in range(5)],
    "RES-PPO": [f"ppo_res_s{s}" for s in range(5)],
    "LSTM-PPO": [f"ppo_lstm_s{s}" for s in range(5)],
}
COLORS = {**mm.COLORS, "Harmonic-LQR": "#7d3c98", "MEM-PPO": "#8e44ad",
          "RES-PPO": "#16a085", "LSTM-PPO": "#d35400"}
NO_MOON = {"DR-PPO": "phase8.csv", "MEM-PPO": "phase8.csv", "RES-PPO": "phase8.csv",
           "LSTM-PPO": "phase9.csv"}


def read_classical(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key, v in r.items():
            if key not in ("condition", "family", "scenario"):
                r[key] = float(v)
    return rows


def _classical_job(job):
    cond, fam, r_ref, rho = job
    k, moon = CONDITIONS[cond]
    summ = summarize(evaluate(CLASSICAL[fam](rho=rho, r_ref_km=r_ref), seeds=TEST_SEEDS,
                              perturbation_k=k, moon=moon))
    return [{"condition": cond, "k": k, "moon": moon, "family": fam, "r_ref_km": r_ref,
             "rho": rho, "scenario": scen, **{key: m[key] for key in mm.METRICS}}
            for scen, m in summ.items()]


def envelope_dv(pts, dist):
    env = mm.pareto_envelope([p for p in pts if p["survival"] == 1.0])
    x = np.log10([p["mean_dist_km"] for p in env])
    y = np.log10([p["dv_mps"] for p in env])
    lx = np.log10(dist)
    if lx < x[0] or lx > x[-1]:
        return float("nan")
    return float(10 ** np.interp(lx, x, y))


def compare(m, pts):
    """best_classical_ratio, envelope_ratio, dominated vs. classical points `pts`
    (one condition and scenario, all four families)."""
    ok = [p for p in pts if p["survival"] == 1.0]
    best = mm.staircase(ok, m["mean_dist_km"])
    fam_best = {fam: mm.staircase([p for p in ok if p["family"] == fam], m["mean_dist_km"])
                for fam in CLASSICAL}
    dominated = m["survival"] < 1.0 or any(
        p["mean_dist_km"] <= m["mean_dist_km"] and p["dv_mps"] <= m["dv_mps"] for p in ok)
    return {"best_classical_ratio": m["dv_mps"] / best,
            **{f"{fam}_ratio": m["dv_mps"] / v for fam, v in fam_best.items()},
            "envelope_ratio": m["dv_mps"] / envelope_dv(pts, m["mean_dist_km"]),
            "dominated": float(dominated)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-classical", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rl", default=",".join(RL), help="comma-separated RL families")
    args = ap.parse_args()
    cpath = EXP / "moon_classical.csv"

    if args.reuse_classical and cpath.exists():
        classical = read_classical(cpath)
    else:
        jobs = [(c, fam, r_ref, float(rho)) for c in CONDITIONS for fam in CLASSICAL
                for r_ref, rhos in mm.DESIGNS.items() for rho in rhos]
        print(f"Classical sweep: {len(jobs)} design x condition runs on {args.workers} workers",
              flush=True)
        classical = []
        with Pool(args.workers) as pool:
            for i, rows in enumerate(pool.imap(_classical_job, jobs, chunksize=4)):
                classical += rows
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(jobs)}", flush=True)
        mm.write_csv(cpath, classical)
        classical = read_classical(cpath)

    pts = {(c, s): [p for p in classical if p["condition"] == c and p["scenario"] == s]
           for c in CONDITIONS for s in SCENARIOS}
    rows = []
    for c, ps in pts.items():                                  # classical rho = 1 references
        for fam in CLASSICAL:
            r = next(p for p in ps if p["family"] == fam and p["r_ref_km"] == 1000.0
                     and np.isclose(p["rho"], 1.0))
            m = {key: r[key] for key in mm.METRICS}
            rows.append({"condition": c[0], "family": fam, "train_seed": -1, "model": "rho=1",
                         "scenario": c[1], **m, **compare(m, ps), "vs_no_moon": float("nan")})

    no_moon = {}
    for fam, fname in NO_MOON.items():
        if (EXP / fname).exists():
            for r in csv.DictReader(open(EXP / fname, newline="")):
                if r["family"] == fam and r["train_seed"] not in ("mean", "std", "-1"):
                    no_moon[(r["model"], float(r["k"]), r["scenario"])] = float(r["dv_mps"])
    fams = [f for f in args.rl.split(",")
            if all((ROOT / "models" / f"{n}.zip").exists() for n in RL[f])]
    for fam in fams:
        for seed, name in enumerate(RL[fam]):
            ctrl = load_policy(ROOT / "models", name)[1]
            for cond, (k, moon) in CONDITIONS.items():
                summ = summarize(evaluate(ctrl, seeds=TEST_SEEDS, perturbation_k=k, moon=moon))
                for scen, m in summ.items():
                    base = no_moon.get((name, k, scen), float("nan"))
                    rows.append({"condition": cond, "family": fam, "train_seed": seed,
                                 "model": name, "scenario": scen,
                                 **{key: m[key] for key in mm.METRICS},
                                 **compare(m, pts[(cond, scen)]),
                                 "vs_no_moon": m["dv_mps"] / base})
        print(f"  {fam} evaluated", flush=True)

    keys = (*mm.METRICS, "best_classical_ratio", "envelope_ratio", "dominated", "vs_no_moon")
    agg = []
    for fam in fams:
        for cond in CONDITIONS:
            for scen in SCENARIOS:
                sel = [r for r in rows if r["family"] == fam and r["condition"] == cond
                       and r["scenario"] == scen]
                for stat, fn in (("mean", np.mean), ("std", lambda v: np.std(v, ddof=1))):
                    agg.append({**{key: "" for key in rows[0]}, "condition": cond,
                                "family": fam, "train_seed": stat, "scenario": scen,
                                **{key: float(fn([r[key] for r in sel])) for key in keys}})
    mm.write_csv(EXP / "moon.csv", rows + agg)

    print(f"\nTest seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1}, one year. RL: mean over "
          "5 training seeds. 'x best classical': delta-v / cheapest classical design (all "
          "four families) at least as accurate. 'not dom.': seeds no classical design "
          "dominates\n")
    print(f"{'condition':<12}{'scenario':<13}{'method':<14}{'kept':>6}{'dv [m/s]':>10}"
          f"{'mean [km]':>11}{'x best cl.':>12}{'x envelope':>12}{'not dom.':>10}"
          f"{'x no Moon':>11}")
    for cond in CONDITIONS:
        for scen in SCENARIOS:
            for fam in CLASSICAL:
                r = next(r for r in rows if r["family"] == fam and r["condition"] == cond
                         and r["scenario"] == scen and r["train_seed"] == -1)
                print(f"{cond:<12}{scen:<13}{fam + ' rho=1':<14}{r['survival']:>6.0%}"
                      f"{r['dv_mps']:>10.3f}{r['mean_dist_km']:>11.2f}"
                      f"{r['best_classical_ratio']:>12.3f}{r['envelope_ratio']:>12.3f}"
                      f"{'':>10}{'':>11}")
            for fam in fams:
                mu = next(a for a in agg if a["family"] == fam and a["condition"] == cond
                          and a["scenario"] == scen and a["train_seed"] == "mean")
                nd = sum(r["dominated"] == 0 for r in rows if r["family"] == fam
                         and r["condition"] == cond and r["scenario"] == scen
                         and r["train_seed"] != -1)
                print(f"{cond:<12}{scen:<13}{fam:<14}{mu['survival']:>6.0%}"
                      f"{mu['dv_mps']:>10.3f}{mu['mean_dist_km']:>11.2f}"
                      f"{mu['best_classical_ratio']:>12.3f}{mu['envelope_ratio']:>12.3f}"
                      f"{nd:>8}/5{mu['vs_no_moon']:>11.2f}")
        print()
    plot(rows, pts, fams)
    print("Saved experiments/moon_classical.csv, experiments/moon.csv, figures/moon_pareto.png")


def plot(rows, pts, fams):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(5.2 * len(CONDITIONS), 8))
    for i, scen in enumerate(SCENARIOS):
        for j, cond in enumerate(CONDITIONS):
            ax = axes[i, j]
            for fam in CLASSICAL:
                env = mm.pareto_envelope([p for p in pts[(cond, scen)]
                                          if p["family"] == fam and p["survival"] == 1.0])
                ax.plot([p["mean_dist_km"] for p in env], [p["dv_mps"] for p in env], "-",
                        color=COLORS[fam], lw=1.5, label=f"{fam}, best over rho and r_ref")
            for fam in fams:
                for r in rows:
                    if (r["family"] == fam and r["condition"] == cond and r["scenario"] == scen
                            and r["survival"] == 1.0):
                        ax.plot(r["mean_dist_km"], r["dv_mps"], mm.MARKERS[r["train_seed"]],
                                color=COLORS[fam], ms=6, mec="white", mew=0.6,
                                label=fam if r["train_seed"] == 0 else None)
            ax.set_xscale("log")
            ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_title(f"{'Holding' if scen == 'holding' else 'Recovery'}, {cond}")
            ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("delta-v over one year [m/s]")
            if i == 1:
                ax.set_xlabel("mean distance from L1 [km]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Phase 9: the Moon, met by every controller for the first time", fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    fig.savefig(ROOT / "figures" / "moon_pareto.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
