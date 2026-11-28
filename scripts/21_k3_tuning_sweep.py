"""Audit K3: does DR-PPO-M still beat Moon-modelling classical designs once
their *filter* is tuned too?

Scripts 18-20 swept the classical designs over (rho, r_ref) only. The Kalman
filter's process noise of Bias-LQR and Harmonic-LQR (``bias_walk_mps2``, how
fast the disturbance estimate may move) stayed at its default 2e-10 m/s^2, and
LQI's integral time scale (``t_int_days``) at 60 days. The independent audit
(25 Nov 2026) re-tuned Harmonic-LQR's bias_walk on 100 extra designs and pushed
DR-PPO-M seed 2 from 0.985 to 1.004 x the envelope. This script does the full
search, with clean seed hygiene:

  grid       Harmonic-LQR, Bias-LQR: r_ref 11 values (10-1000 km, log)
             x rho 41 (1e-3 - 1e4, log) x bias_walk 9 (1e-11 - 1e-6 m/s^2,
             half-decade steps), Riccati solved with riccati="balanced"
             (the legacy solve cannot do bias_walk <= 1e-10, see
             cr3bp.control.riccati);
             LQI: r_ref 11 x rho 41 x t_int 8 (7.5 - 960 days, x2 steps);
             plus, unchanged ("legacy"), every design of scripts 18 and 19 and
             the audit's 100 Harmonic-LQR designs, so the new classical set
             contains every design the paper and the audit used.
             The grid is fixed in advance (log-uniform), not chosen by looking
             at any results.
  screening  every design flies 10 screening seeds (300-309) that nothing else
             in the project uses. Designs within SCREEN_MARGIN of the screening
             envelope (or of the Moon-agnostic one), in a distance window around the agents (also flown on
             300-309), go on to the final comparison. No test seed is used to
             select anything.
  final      the selected designs and DR-PPO-M (5 training seeds) fly 100
             fresh seeds (200-299): envelope ratio, dominated or not, the
             cheapest design of each family at equal accuracy, and a paired
             bootstrap over the 100 seeds (95 % interval of the envelope ratio,
             share of resamples in which the agent is not dominated).
  secondary  the same on the test seeds (100-119), against the selected
             designs plus every design of scripts 18-19 (their test-seed rows).

Condition: k = 1 + Moon, holding (where the paper's claim is made; under the
Moon alone every DR-PPO-M seed was already dominated by the smaller design
set of script 19, so a larger set cannot change that on those seeds).

A design whose filter Riccati equation cannot be solved is not dropped
silently: it is written to experiments/k3_riccati_failures.csv with the
error, and counted in the printed summary.

Writes experiments/k3_screen.csv, k3_riccati_failures.csv, k3_final.csv,
k3_perseed.csv.gz, k3_agents.csv and figures/k3_frontier.png.

Run from the repository root (~2 h on two cores; needs scripts/18-19 outputs):
    python scripts/21_k3_tuning_sweep.py [--workers 2] [--stage screen|final|analyse]
"""

import argparse
import csv
import gzip
import importlib.util
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cr3bp.control import (BiasLQRController, HarmonicLQRController, LQIController,
                           LQRController, RiccatiError)
from cr3bp.evaluation import SCENARIOS, TEST_SEEDS, evaluate

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

K, MOON, COND = 1.0, 1.0, "k=1 + Moon"
HOLD = {"holding": SCENARIOS["holding"]}
SCREEN_SEEDS = range(300, 310)
FRESH_SEEDS = range(200, 300)
SCREEN_MARGIN = 0.02            # keep designs within 2 % of the screening envelope
# (screening vs fresh seeds: relative delta-v of two designs moves by sd 0.15 %,
#  distance by 0.3 %, measured on the 538 designs of scripts 19-20; 2 % is > 6 sd)
WINDOW = 2.0                    # ... whose distance is within x/÷ 2 of the agents' range
N_BOOT = 2000

R_REFS = np.logspace(1, 3, 11)
RHOS = np.logspace(-3, 4, 41)
BIAS_WALKS = np.logspace(-11, -6, 11)              # 1e-11 ... 1e-6, half-decade steps
T_INTS = 7.5 * 2.0 ** np.arange(8)                       # 7.5 ... 960 days
AGENTS = [f"ppo_drm_s{s}" for s in range(5)]           # the claim under test
EXTRA_AGENTS = [f"ppo_resm_s{s}" for s in range(5)]     # also re-checked (not used to select)
AGNOSTIC = ("LQR", "LQI", "Bias-LQR")                    # no model of the Moon
CLASSES = {"LQR": LQRController, "LQI": LQIController, "Bias-LQR": BiasLQRController,
           "Harmonic-LQR": HarmonicLQRController}


# ---------------------------------------------------------------- designs
def design(family, r_ref, rho, riccati="", bias_walk=float("nan"), t_int=float("nan"),
           source="k3 grid"):
    return {"family": family, "riccati": riccati, "r_ref_km": float(r_ref),
            "rho": float(rho), "bias_walk_mps2": float(bias_walk),
            "t_int_days": float(t_int), "source": source}


def design_id(d):
    return (f"{d['family']}|{d['riccati']}|{d['r_ref_km']:.6g}|{d['rho']:.6g}|"
            f"{d['bias_walk_mps2']:.3g}|{d['t_int_days']:.4g}")


def build(d):
    """The controller of design d (raises RiccatiError if its filter can't be solved)."""
    kw = {"rho": d["rho"], "r_ref_km": d["r_ref_km"]}
    if d["family"] in ("Bias-LQR", "Harmonic-LQR"):
        kw["riccati"] = d["riccati"]
        if np.isfinite(d["bias_walk_mps2"]):
            kw["bias_walk_mps2"] = d["bias_walk_mps2"]
    if d["family"] == "LQI" and np.isfinite(d["t_int_days"]):
        kw["t_int_days"] = d["t_int_days"]
    return CLASSES[d["family"]](**kw)


def grid():
    out = []
    for fam in ("Harmonic-LQR", "Bias-LQR"):
        out += [design(fam, r, p, "balanced", bias_walk=b)
                for r in R_REFS for p in RHOS for b in BIAS_WALKS]
    out += [design("LQI", r, p, t_int=t) for r in R_REFS for p in RHOS for t in T_INTS]
    return out


def previous_designs():
    """Every (k = 1 + Moon) design of scripts 18 and 19, plus the audit's 100."""
    out = []
    for fname, src in (("moon_classical.csv", "script 18"),
                       ("moon_frontier_designs.csv", "script 19")):
        for r in csv.DictReader(open(EXP / fname, newline="")):
            if r["condition"] == COND and r["scenario"] == "holding":
                ric = "legacy" if r["family"] in ("Bias-LQR", "Harmonic-LQR") else ""
                out.append(design(r["family"], r["r_ref_km"], r["rho"], ric, source=src))
    for b in (6e-10, 2e-9):                      # audit, 25 Nov 2026
        out += [design("Harmonic-LQR", 1000.0, p, "legacy", bias_walk=b, source="audit")
                for p in np.logspace(np.log10(0.3), np.log10(30), 50)]
    uniq = {}
    for d in out:
        uniq.setdefault(design_id(d), d)
    return list(uniq.values())


# ---------------------------------------------------------------- flying
def fly_design(args):
    """Per-seed results of one design, or a failure row. Never raises RiccatiError."""
    d, seeds = args
    try:
        ctrl = build(d)
    except RiccatiError as exc:
        return {**d, "id": design_id(d), "status": "riccati_failed", "error": str(exc)}, None
    rows = evaluate(ctrl, seeds=seeds, scenarios=HOLD, perturbation_k=K, moon=MOON)
    per = np.array([[r["survived"], r["dv_mps"], r["mean_dist_km"]] for r in rows], float)
    return {**d, "id": design_id(d), "status": "ok", "error": ""}, per


def summary(per):
    return {"survival": float(per[:, 0].mean()), "dv_mps": float(per[:, 1].mean()),
            "mean_dist_km": float(per[:, 2].mean())}


def fly_all(designs, seeds, workers, label):
    t0 = time.time()
    meta, per = [], []
    with Pool(workers) as pool:
        for i, (m, p) in enumerate(pool.imap(fly_design, [(d, seeds) for d in designs],
                                             chunksize=8)):
            meta.append(m)
            per.append(p)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {label}: {i + 1}/{len(designs)}  {el / 60:.1f} min, "
                      f"~{el / (i + 1) * (len(designs) - i - 1) / 60:.0f} min left", flush=True)
    print(f"  {label}: {len(designs)} designs in {(time.time() - t0) / 60:.1f} min", flush=True)
    return meta, per


def fly_agents(seeds, names=AGENTS):
    from cr3bp.training import load_policy
    out = {}
    for name in names:
        ctrl = load_policy(ROOT / "models", name)[1]
        rows = evaluate(ctrl, seeds=seeds, scenarios=HOLD, perturbation_k=K, moon=MOON)
        out[name] = np.array([[r["survived"], r["dv_mps"], r["mean_dist_km"]] for r in rows])
    return out


def split_failures(meta):
    """(ok rows, failed rows); failures are logged, never dropped silently."""
    bad = [m for m in meta if m["status"] != "ok"]
    for m in bad:
        print(f"  RICCATI FAILURE, design skipped and logged: {m['id']}: {m['error']}")
    return [m for m in meta if m["status"] == "ok"], bad


# ---------------------------------------------------------------- frontier maths
def envelope(dv, dist):
    """Lower-left Pareto boundary (indices into dv/dist), sorted by distance."""
    order = np.lexsort((dv, dist))
    env, best = [], np.inf
    for i in order:
        if dv[i] < best:
            env.append(i)
            best = dv[i]
    return np.array(env)


def envelope_dv(dv, dist, at):
    e = envelope(dv, dist)
    x, y = np.log10(dist[e]), np.log10(dv[e])
    la = np.log10(at)
    if la < x[0] or la > x[-1]:
        return np.nan
    return float(10 ** np.interp(la, x, y))


def compare(a_dv, a_dist, dv, dist, fams):
    """Agent (a_dv, a_dist) vs classical designs (arrays; all survived)."""
    out = {"envelope_ratio": a_dv / envelope_dv(dv, dist, a_dist),
           "dominated": float(np.any((dist <= a_dist) & (dv <= a_dv)))}
    acc = dist <= a_dist
    out["best_classical_ratio"] = a_dv / dv[acc].min() if acc.any() else np.nan
    agn = acc & np.isin(fams, AGNOSTIC)
    out["agnostic_ratio"] = a_dv / dv[agn].min() if agn.any() else np.nan
    for f in CLASSES:
        sel = acc & (fams == f)
        out[f"{f}_ratio"] = a_dv / dv[sel].min() if sel.any() else np.nan
    return out


def bootstrap(agent_per, D_dv, D_dist, n=N_BOOT, seed=0):
    """Paired bootstrap over seeds: (2.5 %, 97.5 %) of the envelope ratio, and the
    share of resamples in which no design dominates the agent."""
    rng = np.random.default_rng(seed)
    ns = agent_per.shape[0]
    ratios, nd = [], 0
    for _ in range(n):
        idx = rng.integers(0, ns, ns)
        a_dv, a_dist = agent_per[idx, 1].mean(), agent_per[idx, 2].mean()
        dv, dist = D_dv[:, idx].mean(1), D_dist[:, idx].mean(1)
        ratios.append(a_dv / envelope_dv(dv, dist, a_dist))
        nd += not np.any((dist <= a_dist) & (dv <= a_dv))
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return float(lo), float(hi), nd / n


# ---------------------------------------------------------------- io
def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def write_perseed(path, entries):
    """entries: (id, seed set name, seeds, per-seed array)."""
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "seed_set", "seed", "survived", "dv_mps", "mean_dist_km"])
        for did, sset, seeds, per in entries:
            for s, row in zip(seeds, per):
                w.writerow([did, sset, s, int(row[0]), f"{row[1]:.10g}", f"{row[2]:.10g}"])


def read_perseed(path):
    out = {}
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault((r["id"], r["seed_set"]), []).append(
                [float(r["survived"]), float(r["dv_mps"]), float(r["mean_dist_km"])])
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- stages
def stage_screen(workers):
    designs = grid() + previous_designs()
    path = EXP / "k3_screen.csv"
    done, run = [], {"screen_designs": 0, "screen_failed": 0, "screen_minutes": 0.0}
    if path.exists():                                   # resume: fly only new designs
        done = [r for r in read_csv(path) if r["source"] != "agent"]
        run = json.load(open(EXP / "k3_run.json"))
        have = {r["id"] for r in done}
        designs = [d for d in designs if design_id(d) not in have]
    print(f"Screening {len(designs)} designs ({len(done)} already done) on seeds "
          f"{SCREEN_SEEDS.start}-{SCREEN_SEEDS.stop - 1} ({COND}, holding) on {workers} "
          "workers", flush=True)
    t0 = time.time()
    meta, per = fly_all(designs, SCREEN_SEEDS, workers, "screen")
    ok, bad = split_failures(meta)
    fpath = EXP / "k3_riccati_failures.csv"
    old_bad = [r for r in read_csv(fpath) if r["status"] == "riccati_failed"] if fpath.exists() else []
    bad = old_bad + bad
    write_csv(fpath, bad or [{**design("none", 0, 0), "id": "", "status": "no failures",
                              "error": ""}])
    rows = done + [{**m, **summary(p)} for m, p in zip(meta, per) if m["status"] == "ok"]
    agents = fly_agents(SCREEN_SEEDS)
    rows += [{**design("DR-PPO-M", np.nan, np.nan, source="agent"), "id": n, "status": "ok",
              "error": "", **summary(p)} for n, p in agents.items()]
    write_csv(path, rows)
    run.update({"screen_designs": len(rows) - len(agents) + len(bad),
                "screen_failed": len(bad),
                "screen_minutes": run["screen_minutes"] + (time.time() - t0) / 60})
    json.dump(run, open(EXP / "k3_run.json", "w"), indent=1)
    print(f"Riccati failures: {len(bad)} (logged in experiments/k3_riccati_failures.csv)")


def select(screen):
    """Designs within SCREEN_MARGIN of the screening envelope of all families, or
    (Moon-agnostic families) of the Moon-agnostic envelope, in the distance window
    around the agents."""
    cls = [r for r in screen if r["source"] != "agent" and r["survival"] == 1.0]
    ag = [r for r in screen if r["source"] == "agent"]
    lo = min(r["mean_dist_km"] for r in ag) / WINDOW
    hi = max(r["mean_dist_km"] for r in ag) * WINDOW
    dv = np.array([r["dv_mps"] for r in cls])
    dist = np.array([r["mean_dist_km"] for r in cls])
    agn = np.array([r["family"] in AGNOSTIC for r in cls])
    keep = []
    for r, v, x, a in zip(cls, dv, dist, agn):
        if lo <= x <= hi:
            r["screen_ratio"] = v / envelope_dv(dv, dist, x)
            r["screen_agnostic_ratio"] = (v / envelope_dv(dv[agn], dist[agn], x) if a
                                          else float("nan"))
            if (r["screen_ratio"] <= 1 + SCREEN_MARGIN
                    or r["screen_agnostic_ratio"] <= 1 + SCREEN_MARGIN):
                keep.append(r)
    return keep, (lo, hi)


def stage_final(workers):
    screen = read_csv(EXP / "k3_screen.csv")
    for r in screen:
        for k in ("family", "riccati", "source", "id", "status", "error"):
            r[k] = "" if r[k] == "" else str(r[k])
    keep, (lo, hi) = select(screen)
    print(f"Selected {len(keep)} designs within {SCREEN_MARGIN:.0%} of the screening "
          f"envelope, {lo:.1f}-{hi:.1f} km", flush=True)
    t0 = time.time()
    seeds_all = list(FRESH_SEEDS) + list(TEST_SEEDS)
    designs = [{k: r[k] for k in ("family", "riccati", "r_ref_km", "rho", "bias_walk_mps2",
                                  "t_int_days", "source")} for r in keep]
    for d in designs:
        d["riccati"] = d["riccati"] or ""
    meta, per = fly_all(designs, seeds_all, workers, "final")
    ok, bad = split_failures(meta)
    nf = len(FRESH_SEEDS)
    entries, rows = [], []
    for r, m, p in zip(keep, meta, per):
        if m["status"] != "ok":
            continue
        entries += [(m["id"], "fresh", FRESH_SEEDS, p[:nf]), (m["id"], "test", TEST_SEEDS, p[nf:])]
        f, t = summary(p[:nf]), summary(p[nf:])
        rows.append({**{k: m[k] for k in ("id", "family", "riccati", "r_ref_km", "rho",
                                          "bias_walk_mps2", "t_int_days", "source")},
                     "screen_ratio": r["screen_ratio"],
                     "screen_agnostic_ratio": r["screen_agnostic_ratio"],
                     **{f"fresh_{k}": v for k, v in f.items()},
                     **{f"test_{k}": v for k, v in t.items()}})
    agents = fly_agents(seeds_all)
    for n, p in agents.items():
        entries += [(n, "fresh", FRESH_SEEDS, p[:nf]), (n, "test", TEST_SEEDS, p[nf:])]
    write_csv(EXP / "k3_final.csv", rows)
    write_perseed(EXP / "k3_perseed.csv.gz", entries)
    run = json.load(open(EXP / "k3_run.json"))
    run.update({"final_designs": len(rows), "final_failed": len(bad),
                "final_minutes": (time.time() - t0) / 60, "window_km": [lo, hi]})
    json.dump(run, open(EXP / "k3_run.json", "w"), indent=1)


def stage_analyse():
    final = read_csv(EXP / "k3_final.csv")
    per = read_perseed(EXP / "k3_perseed.csv.gz")
    missing = [n for n in EXTRA_AGENTS if (n, "fresh") not in per]
    if missing:                                  # RES-PPO-M: fly once, add to the per-seed file
        seeds_all = list(FRESH_SEEDS) + list(TEST_SEEDS)
        got = fly_agents(seeds_all, missing)
        nf = len(FRESH_SEEDS)
        entries = [(k[0], k[1], FRESH_SEEDS if k[1] == "fresh" else TEST_SEEDS, v)
                   for k, v in per.items()]
        for n, p in got.items():
            entries += [(n, "fresh", FRESH_SEEDS, p[:nf]), (n, "test", TEST_SEEDS, p[nf:])]
            per[(n, "fresh")], per[(n, "test")] = p[:nf], p[nf:]
        write_perseed(EXP / "k3_perseed.csv.gz", entries)
    old_moon = {}
    for r in csv.DictReader(open(EXP / "moon_frontier.csv", newline="")):
        if r["family"] in ("DR-PPO-M", "RES-PPO-M") and r["condition"] == COND:
            tag = "drm" if r["family"] == "DR-PPO-M" else "resm"
            old_moon[f"ppo_{tag}_s{r['train_seed']}"] = r
    old_conf = {r["model"]: r for r in csv.DictReader(open(EXP / "moon_confirmation.csv"))}
    # previous classical designs on the test seeds (scripts 18-19 rows)
    prev_test = []
    for fname in ("moon_classical.csv", "moon_frontier_designs.csv"):
        for r in csv.DictReader(open(EXP / fname, newline="")):
            if (r["condition"] == COND and r["scenario"] == "holding"
                    and float(r["survival"]) == 1.0):
                prev_test.append((r["family"], float(r["dv_mps"]), float(r["mean_dist_km"])))

    out = []
    for sset in ("fresh", "test"):
        sel = [r for r in final if r[f"{sset}_survival"] == 1.0]
        D_dv = np.array([per[(r["id"], sset)][:, 1] for r in sel])
        D_dist = np.array([per[(r["id"], sset)][:, 2] for r in sel])
        fams = np.array([r["family"] for r in sel])
        dv, dist = D_dv.mean(1), D_dist.mean(1)
        if sset == "test":                       # add every earlier design (summaries only)
            dv_all = np.r_[dv, [p[1] for p in prev_test]]
            dist_all = np.r_[dist, [p[2] for p in prev_test]]
            fam_all = np.r_[fams, [p[0] for p in prev_test]]
        else:
            dv_all, dist_all, fam_all = dv, dist, fams
        for name in AGENTS + EXTRA_AGENTS:
            s = int(name[-1])
            a = per[(name, sset)]
            a_dv, a_dist = a[:, 1].mean(), a[:, 2].mean()
            c = compare(a_dv, a_dist, dv_all, dist_all, fam_all)
            lo, hi, p_nd = bootstrap(a, D_dv, D_dist)
            # the design that dominates the agent with the largest fuel saving, if any
            dom = np.where((dist <= a_dist) & (dv <= a_dv))[0]
            best = sel[dom[np.argmin(dv[dom])]] if len(dom) else None
            prev = (float(old_moon[name]["envelope_ratio"]) if sset == "test"
                    else float(old_conf[name]["envelope_ratio"]) if name in old_conf
                    else float("nan"))
            out.append({"seed_set": sset, "model": name, "train_seed": s,
                        "family": "DR-PPO-M" if "drm" in name else "RES-PPO-M",
                        "survival": float(a[:, 0].mean()), "dv_mps": a_dv,
                        "mean_dist_km": a_dist, **c, "ci_lo": lo, "ci_hi": hi,
                        "p_not_dominated": p_nd, "envelope_ratio_before": prev,
                        "dominating_design": best["id"] if best else "",
                        "dominating_saving_mps": (a_dv - best["fresh_dv_mps" if sset == "fresh"
                                                              else "test_dv_mps"])
                        if best else float("nan")})
    write_csv(EXP / "k3_agents.csv", out)
    run = json.load(open(EXP / "k3_run.json"))
    run.update({"screen_margin": SCREEN_MARGIN, "n_screen_seeds": len(SCREEN_SEEDS),
                "bias_walks": [float(b) for b in BIAS_WALKS],
                "t_ints": [float(t) for t in T_INTS],
                "n_r_ref": len(R_REFS), "n_rho": len(RHOS)})
    json.dump(run, open(EXP / "k3_run.json", "w"), indent=1)
    print(f"\n{COND}, holding. Designs screened: {run['screen_designs']} "
          f"({run['screen_failed']} Riccati failures), {run['screen_minutes']:.0f} min; "
          f"final: {run['final_designs']} designs, {run['final_minutes']:.0f} min")
    print(f"{'seeds':<7}{'model':<12}{'dv':>7}{'km':>8}{'before':>8}{'envelope':>10}"
          f"{'95% CI':>16}{'P(not dom)':>12}{'dominated':>11}")
    for r in out:
        print(f"{r['seed_set']:<7}{r['model']:<12}{r['dv_mps']:>7.3f}{r['mean_dist_km']:>8.1f}"
              f"{r['envelope_ratio_before']:>8.3f}{r['envelope_ratio']:>10.3f}"
              f"   [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]{r['p_not_dominated']:>12.2f}"
              f"{'yes' if r['dominated'] else 'no':>11}")
    for fam in ("DR-PPO-M", "RES-PPO-M"):
        for sset in ("fresh", "test"):
            nd = sum(r["dominated"] == 0 for r in out
                     if r["seed_set"] == sset and r["family"] == fam)
            print(f"{fam}, {sset}: not dominated in {nd} of 5")
    plot(final, per, out)


def plot(final, per, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    colors = {"Harmonic-LQR": "#7d3c98", "Bias-LQR": "#2e86c1", "LQI": "#28b463",
              "LQR": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sel = [r for r in final if r["fresh_survival"] == 1.0]
    for fam, col in colors.items():
        pts = [(r["fresh_mean_dist_km"], r["fresh_dv_mps"]) for r in sel if r["family"] == fam]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, ".", color=col, ms=3, alpha=0.35, label=f"{fam} designs")
    dv = np.array([r["fresh_dv_mps"] for r in sel])
    dist = np.array([r["fresh_mean_dist_km"] for r in sel])
    e = envelope(dv, dist)
    ax.plot(dist[e], dv[e], "-", color="black", lw=1.2, label="classical envelope")
    for r in out:
        if r["seed_set"] == "fresh" and r["family"] == "DR-PPO-M":
            ax.plot(r["mean_dist_km"], r["dv_mps"], "o", color="#e74c3c", ms=6, mec="white")
            ax.annotate(f"s{r['train_seed']}", (r["mean_dist_km"], r["dv_mps"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.plot([], [], "o", color="#e74c3c", label="DR-PPO-M (5 training seeds)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("mean distance from L1 [km]")
    ax.set_ylabel("delta-v over one year [m/s]")
    ax.set_title("k = 1 + Moon, holding, fresh seeds 200-299", fontsize=10)
    ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "k3_frontier.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stage", choices=("all", "screen", "final", "analyse"), default="all")
    args = ap.parse_args()
    if args.stage in ("all", "screen"):
        stage_screen(args.workers)
    if args.stage in ("all", "final"):
        stage_final(args.workers)
    if args.stage in ("all", "analyse"):
        stage_analyse()
    print("Saved experiments/k3_*.csv, figures/k3_frontier.png")


if __name__ == "__main__":
    main()
