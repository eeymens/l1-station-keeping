"""Phase 6: every number and table in the paper, generated from the results.

Nothing in paper/main.tex is typed by hand: this script reads the raw CSVs in
experiments/ (written by scripts 05, 07, 09, 10, 12, 13 and 14), recomputes the few
quantities that need a simulation (thrust statistics, the unstable mode's pull,
the Lagrange-point table), and writes

    paper/generated/numbers.tex     \\newcommand macros used in the text
    paper/generated/tab_*.tex       the tables

Re-run it after any experiment changes and rebuild the PDF; a stale number in
the paper is then impossible.

Run from the repository root (~1 min, needs the ``rl`` extra):
    python scripts/11_paper_assets.py
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from cr3bp import MU_SUN_EARTH, lagrange_points, stability
from cr3bp.constants import AU_KM, TIME_UNIT_DAYS
from cr3bp.control import LQRController, StationKeepingSim, run_episode
from cr3bp.control.station_keeping import MPS2
from cr3bp.evaluation import TEST_SEEDS
from cr3bp.training import load_policy

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "paper" / "generated"

CUBIC = ("ppo_l1", "ppo_l1_s1", "ppo_l1_s2")
LINEAR = ("ppo_l1_linear", "ppo_l1_linear_s1", "ppo_l1_linear_s2")


# ---------------------------------------------------------------- helpers
def load(name):
    with open(EXP / name) as fh:
        return list(csv.DictReader(fh))


def num(v, nd=2):
    """Fixed-point number for LaTeX (unicode minus not needed inside math)."""
    return f"{v:.{nd}f}"


def sig(v, n=2):
    """n significant figures, no scientific notation for moderate values."""
    if v == 0 or not math.isfinite(v):
        return "0" if v == 0 else "--"
    d = max(0, n - 1 - int(math.floor(math.log10(abs(v)))))
    return f"{round(v, d):.{d}f}"


def sci(v, n=2):
    """a \\times 10^{b} for LaTeX math mode."""
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10**e
    if round(m, n - 1) >= 10:
        m, e = m / 10, e + 1
    return rf"{m:.{n - 1}f}\times10^{{{e}}}"


def pm(mean, sd, nd):
    return rf"${mean:.{nd}f} \pm {sd:.{nd}f}$"


def mean_of(rows, key, kept_only=True):
    v = [float(r[key]) for r in rows if not kept_only or r["survived"] == "True"]
    return float(np.mean(v)) if v else float("nan")


macros = {}


def macro(name, value):
    if not name.isalpha():
        raise ValueError(f"LaTeX macro names must be letters only: {name}")
    if name in macros:
        raise ValueError(f"duplicate macro {name}")
    macros[name] = value


def write_table(name, body):
    (OUT / name).write_text(body.strip() + "\n")


# ---------------------------------------------------------------- 1. dynamics
def dynamics():
    mu = MU_SUN_EARTH
    pts = lagrange_points(mu)
    earth_x = 1.0 - mu
    macro("muSE", sci(mu, 5))
    macro("tuDays", num(TIME_UNIT_DAYS, 2))
    rows = []
    for name in ("L1", "L2", "L3", "L4"):
        st = stability(pts[name], mu)
        dist = abs(pts[name][0] - earth_x) * AU_KM if name != "L4" else None
        rows.append((name, dist, st["growth_rate"], st["efold_days"]))
    l1 = rows[0]
    macro("lOneDistKm", f"{l1[1]:,.0f}".replace(",", "{,}"))
    macro("lambdaU", num(l1[2], 3))
    macro("efoldDays", num(l1[3], 1))

    lines = []
    for name, dist, lam, ef in rows:
        if name == "L4":
            lines.append(r"L4, L5 & 1 AU (60\textdegree{} ahead/behind) & 0 & "
                         r"stable (Routh: $\mu < 0.0385$) \\")
            continue
        efs = f"{ef / 365.25:.0f} years" if ef > 1000 else f"{ef:.1f} days"
        lines.append(rf"{name} & {dist:,.0f}~km & {lam:.{3 if lam > 0.01 else 4}f} & "
                     rf"{efs} \\".replace(",", "{,}"))
    write_table("tab_lagrange.tex", r"""
\begin{tabular}{llll}
\toprule
Point & Distance from Earth & Unstable eigenvalue $\lambda$ & $e$-folding time \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

    # Pull of the unstable mode at a given offset: ~ lambda^2 r (dimensional).
    lam_s = l1[2] / (TIME_UNIT_DAYS * 86_400.0)               # 1/s
    for key, r_km in (("Three", 3000.0), ("Five", 5000.0)):
        macro(f"pull{key}", sci(lam_s**2 * r_km * 1e3, 2))


# ---------------------------------------------------------------- 2. LQR sweep (phase 3)
def lqr_phase3():
    rows = load("lqr_pareto.csv")
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], float(r["rho"]))].append(r)
    summ = {k: (np.mean([float(r["dv_mps"]) for r in v]),
                np.mean([float(r["mean_dist_km"]) for r in v])) for k, v in by.items()}
    hold = [v for (s, _), v in summ.items() if s == "holding"]
    macro("holdDvMin", num(min(d for d, _ in hold), 2))
    macro("holdDvMax", num(max(d for d, _ in hold), 2))
    macro("holdDistMin", num(min(x for _, x in hold), 1))
    macro("holdDistMax", num(max(x for _, x in hold), 1))
    acq = {rho: v for (s, rho), v in summ.items() if s == "acquisition"}
    rho_best = min(acq, key=lambda r: acq[r][0])
    macro("acqBestRho", sig(rho_best, 2))
    macro("acqBestDv", num(acq[rho_best][0], 2))
    rhos = sorted({rho for _, rho in summ})
    macro("rhoDecades", f"{math.log10(rhos[-1] / rhos[0]):.0f}")
    macro("rhoCount", str(len(rhos)))


# ---------------------------------------------------------------- 3. main comparison
def comparison():
    rows = load("comparison_phase5.csv")
    agg ={(r["config"], r["train_seed"], r["scenario"]): r
           for r in load("seed_study.csv") if not r["model"]}

    def ctrl(name, scen):
        out = [r for r in rows if r["controller"] == name and r["scenario"] == scen]
        if not out:
            raise RuntimeError(f"no rows for {name} / {scen} in comparison_phase5.csv")
        return out

    coast_h, coast_a = ctrl("coast", "holding"), ctrl("coast", "acquisition")
    days_h = [float(r["days_survived"]) for r in coast_h]
    macro("coastHoldDaysMin", f"{min(days_h):.0f}")
    macro("coastHoldDaysMax", f"{max(days_h):.0f}")
    macro("coastHoldDaysMean", f"{np.mean(days_h):.0f}")
    macro("coastAcqDays", f"{np.mean([float(r['days_survived']) for r in coast_a]):.0f}")

    def single(name):
        h, a = ctrl(name, "holding"), ctrl(name, "acquisition")
        kept = np.mean([r["survived"] == "True" for r in h + a])
        return (mean_of(h, "dv_mps"), mean_of(h, "mean_dist_km"), mean_of(h, "rms_dist_km"),
                mean_of(a, "dv_mps"), mean_of(a, "mean_dist_km"), kept)

    lqr = single("LQR rho=3")
    dqn = single("DQN (dqn_l1_run2)")
    macro("lqrHoldDv", num(lqr[0], 3))
    macro("lqrHoldDist", num(lqr[1], 2))
    macro("lqrAcqDv", num(lqr[3], 2))
    macro("lqrAcqDist", f"{lqr[4]:.0f}")
    macro("dqnHoldDv", num(dqn[0], 2))
    macro("dqnAcqDv", num(dqn[3], 2))
    macro("dqnHoldDist", num(dqn[1], 1))

    c1 = load("comparison_dqn_l1.csv")             # DQN run 1 was only compared in phase 4
    d1 = [r for r in c1 if r["controller"].startswith("DQN")]
    dqn1 = (mean_of([r for r in d1 if r["scenario"] == "holding"], "dv_mps"),
            mean_of([r for r in d1 if r["scenario"] == "holding"], "mean_dist_km"),
            mean_of([r for r in d1 if r["scenario"] == "acquisition"], "dv_mps"),
            mean_of([r for r in d1 if r["scenario"] == "acquisition"], "mean_dist_km"))
    ratios = [dqn1[0] / lqr[0], dqn1[2] / lqr[3], dqn[0] / lqr[0], dqn[3] / lqr[3]]
    macro("dqnRatioMin", f"{min(ratios):.0f}")
    macro("dqnRatioMax", f"{max(ratios):.0f}")

    def cfg(config):
        m = lambda k, s: float(agg[(config, "mean", s)][k])   # noqa: E731
        sd = lambda k, s: float(agg[(config, "std", s)][k])   # noqa: E731
        return m, sd

    for config, key in (("cubic", "cub"), ("linear", "lin")):
        m, sd = cfg(config)
        macro(f"{key}HoldDv", pm(m("dv_mps", "holding"), sd("dv_mps", "holding"), 3))
        macro(f"{key}HoldDist", pm(m("mean_dist_km", "holding"), sd("mean_dist_km", "holding"), 2))
        macro(f"{key}AcqDv", pm(m("dv_mps", "acquisition"), sd("dv_mps", "acquisition"), 2))
        macro(f"{key}HoldRatio", pm(m("lqr_ratio", "holding"), sd("lqr_ratio", "holding"), 2))

    # linear / cubic fuel, per training seed (seed k vs seed k)
    hold ={r["model"]: float(r["dv_mps"]) for r in load("seed_study.csv")
            if r["model"] and r["scenario"] == "holding"}
    acq = {r["model"]: float(r["dv_mps"]) for r in load("seed_study.csv")
           if r["model"] and r["scenario"] == "acquisition"}
    rh = [hold[l_] / hold[c] for l_, c in zip(LINEAR, CUBIC)]
    ra = [acq[l_] / acq[c] for l_, c in zip(LINEAR, CUBIC)]
    macro("linCubHoldMin", num(min(rh), 1))
    macro("linCubHoldMax", num(max(rh), 1))
    macro("linCubAcqMin", num(min(ra), 1))
    macro("linCubAcqMax", num(max(ra), 1))
    # no overlap between configurations in either scenario
    for s, d in (("holding", hold), ("acquisition", acq)):
        if max(d[c] for c in CUBIC) >= min(d[l_] for l_ in LINEAR):
            print(f"WARNING: cubic and linear overlap in {s}; text claims they do not")

    def row(label, v, kept):
        return (rf"{label} & {v[0]} & {v[1]} & {v[2]} & {v[3]} & {kept} \\")

    def fmt_single(t):
        return (num(t[0], 3), f"{num(t[1], 2)} / {num(t[2], 2)}", num(t[3], 2), f"{t[4]:.0f}")

    lines = [
        rf"No control & 0 & lost after {np.mean(days_h):.0f} d & 0 & lost after "
        rf"{np.mean([float(r['days_survived']) for r in coast_a]):.0f} d & 0\,\% \\",
        row(r"LQR, $\rho = 3$", fmt_single(lqr), f"{lqr[5]:.0%}".replace("%", r"\,\%")),
        row(r"DQN (run 2, seed 0)", fmt_single(dqn), f"{dqn[5]:.0%}".replace("%", r"\,\%")),
    ]
    for config, label in (("linear", r"PPO, linear map ($p=1$)"),
                          ("cubic", r"PPO, cubic map ($p=3$)")):
        m, sd = cfg(config)
        kept = min(m("survival", s) for s in ("holding", "acquisition"))
        cells = (pm(m("dv_mps", "holding"), sd("dv_mps", "holding"), 3),
                 pm(m("mean_dist_km", "holding"), sd("mean_dist_km", "holding"), 2) + " / " +
                 pm(m("rms_dist_km", "holding"), sd("rms_dist_km", "holding"), 2),
                 pm(m("dv_mps", "acquisition"), sd("dv_mps", "acquisition"), 2),
                 pm(m("mean_dist_km", "acquisition"), sd("mean_dist_km", "acquisition"), 0))
        lines.append(row(label, cells, f"{kept:.0%}".replace("%", r"\,\%")))
    write_table("tab_main.tex", r"""
\begin{tabular}{lccccc}
\toprule
& \multicolumn{2}{c}{Holding (start at L1)} & \multicolumn{2}{c}{Recovery (start 1000 km)} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Controller & $\Delta v$ [m/s] & mean / RMS dist.\ [km] & $\Delta v$ [m/s] & mean dist.\ [km] & Kept \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ---------------------------------------------------------------- 4. same-accuracy ratios
def ratios():
    rows = load("seed_study.csv")
    lqr = load("seed_study_lqr.csv")
    macro("sweepRhoCount", str(len({r["rho"] for r in lqr})))
    acq = [r for r in lqr if r["scenario"] == "acquisition" and r["left_branch"] == "True"]
    cheapest = min(acq, key=lambda r: float(r["dv_mps"]))
    macro("cheapRho", sig(float(cheapest["rho"]), 2))
    macro("cheapDv", num(float(cheapest["dv_mps"]), 2))
    macro("cheapDist", f"{float(cheapest['mean_dist_km']):.0f}")

    per = {(r["model"], r["scenario"]): r for r in rows if r["model"]}
    lines = []
    for config, models in (("cubic", CUBIC), ("linear", LINEAR)):
        for k, name in enumerate(models):
            cells = []
            for scen in ("holding", "acquisition"):
                r = per[(name, scen)]
                if r["vs_lqr_branch"] == "inside":
                    cells.append(num(float(r["lqr_ratio"]), 2))
                else:
                    cells.append(rf"dominated ($\geq {float(r['lqr_ratio_vs_cheapest']):.2f}$)")
            step = int(float(per[(name, "holding")]["best_ckpt_step"]))
            steps = f"{step / 1e6:.2f}M" if step >= 1e6 else f"{step // 1000}k"
            lines.append(rf"{config} & {k} & {cells[0]} & {cells[1]} & {steps} \\")
        lines.append(r"\addlinespace")
    agg = {(r["config"], r["train_seed"]): r for r in rows
           if not r["model"] and r["scenario"] == "holding"}
    for config in ("cubic", "linear"):
        m, s = float(agg[(config, "mean")]["lqr_ratio"]), float(agg[(config, "std")]["lqr_ratio"])
        lines.append(rf"{config} & mean $\pm$ std & ${m:.2f} \pm {s:.2f}$ & -- & \\")
    hold = [float(per[(n, "holding")]["lqr_ratio"]) for n in CUBIC]
    macro("cubRatioMin", num(min(hold), 1))
    macro("cubRatioMax", num(max(hold), 1))
    acqr = [float(per[(n, "acquisition")]["lqr_ratio"]) for n in CUBIC
            if per[(n, "acquisition")]["vs_lqr_branch"] == "inside"]
    macro("cubAcqRatioMin", num(min(acqr), 1))
    macro("cubAcqRatioMax", num(max(acqr), 1))
    steps = [int(float(per[(n, "holding")]["best_ckpt_step"])) for n in CUBIC]
    macro("cubBestMin", f"{min(steps) // 1000}k")
    macro("cubBestMax", f"{max(steps) // 1000}k")
    steps = [int(float(per[(n, "holding")]["best_ckpt_step"])) for n in LINEAR]
    macro("linBestMin", f"{min(steps) / 1e6:.1f}M")
    macro("linBestMax", f"{max(steps) / 1e6:.2f}M")
    write_table("tab_ratio.tex", r"""
\begin{tabular}{llccc}
\toprule
Thrust map & Training seed & Holding & Recovery & Best checkpoint \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ---------------------------------------------------------------- 5. robustness
def robustness():
    rs = load("robustness_seeds.csv")
    r0 = load("robustness.csv")                  # has DQN (training seed 0 run only)

    def sel(rows, sweep, level, scen, ctrl):
        out = [r for r in rows if r["sweep"] == sweep and float(r["level"]) == level
               and r["scenario"] == scen and r["controller"] == ctrl]
        if not out:
            raise RuntimeError(f"no rows for {ctrl} in {sweep} {level} {scen}")
        return out

    def cell(rows, sweep, level, scen, ctrl):
        s = sel(rows, sweep, level, scen, ctrl)
        return mean_of(s, "dv_mps"), mean_of(s, "rms_late_km")

    def rng(vals, nd):
        lo, hi = min(vals), max(vals)
        return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}--{hi:.{nd}f}"

    def kept(rows, level, ctrl):
        s = sel(rows, "offset", level, "acquisition", ctrl)
        return np.mean([r["survived"] == "True" for r in s])

    def pct(v):
        return f"{100 * v:.0f}" + r"\,\%"

    lines = []
    for label, sweep, level in ((r"All errors $\times 0$", "noise", 0.0),
                                (r"All errors $\times 1$", "noise", 1.0),
                                (r"All errors $\times 8$", "noise", 8.0),
                                (r"Unmodelled force $\times 100$", "disturbance", 100.0)):
        lq = cell(rs, sweep, level, "holding", "LQR")
        pp = [cell(rs, sweep, level, "holding", m) for m in CUBIC]
        dq = cell(r0, sweep, level, "holding", "dqn_l1_run2")
        lines.append(
            rf"{label} & {lq[0]:.2f} / {lq[1]:.1f} & "
            rf"{rng([p[0] for p in pp], 2)} / {rng([p[1] for p in pp], 1)} & "
            rf"{dq[0]:.2f} / {dq[1]:.1f} \\")
    lines.append(r"\midrule")
    for level in (2000.0, 3000.0, 5000.0):
        km = f"{level:,.0f}".replace(",", "{,}")
        lines.append(
            rf"Start {km} km: kept & {pct(kept(rs, level, 'LQR'))} "
            rf"(sat.\ {pct(kept(rs, level, 'LQR (sat.)'))}) & "
            + " / ".join(f"{100 * kept(rs, level, m):.0f}" for m in CUBIC) + r"\,\%"
            + rf" & {pct(kept(r0, level, 'dqn_l1_run2'))} \\")
    write_table("tab_robust.tex", r"""
\begin{tabular}{lccc}
\toprule
Sweep & LQR & PPO cubic (seeds 0 / 1 / 2) & DQN \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

    # numbers used in the text
    nom = [cell(rs, "noise", 1.0, "holding", m)[0] / cell(rs, "noise", 1.0, "holding", "LQR")[0]
           for m in CUBIC]
    hi = [cell(rs, "noise", 8.0, "holding", m)[0] / cell(rs, "noise", 8.0, "holding", "LQR")[0]
          for m in CUBIC]
    macro("noiseOverNomMin", num(min(nom), 1))
    macro("noiseOverNomMax", num(max(nom), 1))
    macro("noiseOverHiMin", num(min(hi), 1))
    macro("noiseOverHiMax", num(max(hi), 1))
    lost = sum(r["survived"] != "True" for r in rs
               if r["sweep"] in ("noise", "disturbance"))
    macro("noiseLost", str(lost))

    # which runs are lost at 3000 km, by direction (angle = seed index mod 4)
    angles = (0, 90, 180, 270)
    seeds = list(TEST_SEEDS)
    lost_dirs = {}
    for ctrl in ("LQR (sat.)", *CUBIC):
        s = sel(rs, "offset", 3000.0, "acquisition", ctrl)
        lost_dirs[ctrl] = sorted({angles[seeds.index(int(r["seed"])) % 4]
                                  for r in s if r["survived"] != "True"})
        nlost = sum(r["survived"] != "True" for r in s)
        print(f"  3000 km, {ctrl}: lost {nlost} runs, directions {lost_dirs[ctrl]}")
    macro("satLostThree", str(sum(r["survived"] != "True"
                                  for r in sel(rs, "offset", 3000.0, "acquisition", "LQR (sat.)"))))
    s0 = [r for r in sel(rs, "offset", 3000.0, "acquisition", "ppo_l1")
          if angles[seeds.index(int(r["seed"])) % 4] in (0, 180)]
    macro("ppoThreeDv", f"{np.mean([float(r['dv_mps']) for r in s0]):.0f}")
    fin = [float(r["rms_late_km"]) for r in s0]
    macro("ppoThreeLateMin", f"{min(fin):.0f}")
    macro("ppoThreeLateMax", f"{max(fin):.0f}")

    # unlimited LQR: largest commanded thrust over the 5000 km runs
    lqr = LQRController(rho=3.0)
    peak = 0.0
    for i, seed in enumerate(seeds):
        a = math.radians(angles[i % 4])
        sim = StationKeepingSim(seed=seed)
        log = []
        run_episode(sim, lambda o: log.append(lqr(o)) or log[-1],
                    offset_km=(5000 * math.cos(a), 5000 * math.sin(a)))
        peak = max(peak, max(float(np.hypot(*u)) for u in log) / MPS2)
    macro("lqrPeakFive", sci(peak, 2))


# ---------------------------------------------------------------- 6. thrust logs
def thrust_logs():
    """Commanded thrust per day, holding scenario, test seed 100."""
    def log_of(ctrl):
        sim = StationKeepingSim(seed=TEST_SEEDS.start)
        log = []
        run_episode(sim, lambda o: log.append(np.asarray(ctrl(o), float)) or log[-1],
                    offset_km=(0.0, 0.0))
        return np.array([np.hypot(*u) for u in log]) / MPS2          # m/s^2

    lqr = log_of(LQRController(rho=3.0))
    ppo = log_of(load_policy(ROOT / "models", "ppo_l1")[1])
    lin = log_of(load_policy(ROOT / "models", "ppo_l1_linear")[1])
    macro("thrLqrMedian", sci(np.median(lqr), 3))
    macro("thrPpoMedian", sci(np.median(ppo), 3))
    macro("thrLinMedian", sci(np.median(lin), 2))
    macro("thrLqrBig", str(int((lqr > 1e-8).sum())))
    macro("thrPpoBig", str(int((ppo > 1e-8).sum())))
    macro("thrLinBig", str(int((lin > 1e-8).sum())))
    macro("thrDays", str(len(lqr)))

    for name, key in (("dqn_l1", "One"), ("dqn_l1_run2", "Two")):
        t = log_of(load_policy(ROOT / "models", name)[1])
        levels = {"Coast": t < 1e-12,
                  "Small": np.isclose(t, 1e-8, rtol=1e-6, atol=0.0),
                  "Mid": np.isclose(t, 1e-7, rtol=1e-6, atol=0.0),
                  "Big": np.isclose(t, 1e-6, rtol=1e-6, atol=0.0)}
        for lv, mask in levels.items():
            macro(f"dqn{key}{lv}", str(int(mask.sum())))
        if sum(int(m.sum()) for m in levels.values()) != len(t):
            raise RuntimeError(f"{name}: commanded thrust outside the discrete levels")
    cubic_a = (np.median(lqr) / 1e-6) ** (1 / 3)
    macro("cubicActionForHold", num(cubic_a, 2))
    macro("linearActionForHold", sci(np.median(lqr) / 1e-6, 1))


# ---------------------------------------------------------------- 7. hyperparameters
def hyperparameters():
    dqn = json.loads((ROOT / "models" / "dqn_l1_run2.json").read_text())
    ppo = json.loads((ROOT / "models" / "ppo_l1.json").read_text())
    for name in (*CUBIC, *LINEAR):
        cfg = json.loads((ROOT / "models" / f"{name}.json").read_text())["train"]
        for k in ("steps", "n_envs", "n_steps", "lr", "gamma", "log_std_init", "w_pos", "w_fuel"):
            if cfg[k] != ppo["train"][k]:
                raise RuntimeError(f"{name}: {k} differs from ppo_l1; table would be wrong")
    macro("ppoSteps", f"{ppo['train']['steps'] / 1e6:.0f}M")
    macro("dqnSteps", f"{dqn['train']['steps'] / 1e6:.0f}M")
    macro("ppoEnvs", str(ppo["train"]["n_envs"]))
    macro("wPos", str(ppo["train"]["w_pos"]))
    macro("wFuel", f"{ppo['train']['w_fuel']:g}")


# ---------------------------------------------------------------- 8. paper figures
COL = {"LQR": "#2471a3", "cubic": "#d68910", "linear": "#7d3c98", "DQN": "#c0392b"}
SEED_MARKERS = ("o", "s", "^")


# ---------------------------------------------------------------- 7. model mismatch
def mismatch():
    """Phase 7 (scripts/12_model_mismatch.py): ER3BP + SRP unknown to the controllers."""
    from scipy.optimize import brentq

    from cr3bp.evaluation import perturbation_for
    from cr3bp.perturbations import (AREA_TO_MASS_DEFAULT, CR_DEFAULT, E_EARTH, P_SUN_NPM2,
                                     SRP_DEFAULT_MPS2, Perturbation)

    mu = MU_SUN_EARTH
    l1x = lagrange_points(mu)["L1"][0]
    a0 = Perturbation(srp_mps2=SRP_DEFAULT_MPS2).a0
    dodx = lambda x: (x - (1 - mu - a0) * (x + mu) / abs(x + mu) ** 3  # noqa: E731
                      - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3)
    shift_km = (brentq(dodx, l1x - 1e-3, l1x + 1e-4, xtol=1e-16) - l1x) * AU_KM
    macro("eEarth", f"{E_EARTH:.4f}")
    macro("srpP", sci(P_SUN_NPM2, 3))
    macro("srpCr", f"{CR_DEFAULT:.1f}")
    macro("srpAm", f"{AREA_TO_MASS_DEFAULT:g}")
    macro("srpAccel", sci(SRP_DEFAULT_MPS2, 2))
    macro("srpPctLimit", f"{100 * SRP_DEFAULT_MPS2 / 1e-6:.0f}")
    macro("srpDvYear", num(SRP_DEFAULT_MPS2 * 365.25 * 86400, 1))
    macro("srpShiftKm", f"{abs(shift_km):.0f}")

    rows = load("model_mismatch.csv")
    cls = load("model_mismatch_classical.csv")
    parts = load("model_mismatch_parts.csv")
    f = lambda r, k: float(r[k])  # noqa: E731

    def one(fam, k, scen, seed):
        return next(r for r in rows if r["family"] == fam and float(r["k"]) == k
                    and r["scenario"] == scen and r["train_seed"] == seed)

    def seeds(fam, k, scen):
        return [r for r in rows if r["family"] == fam and float(r["k"]) == k
                and r["scenario"] == scen and r["train_seed"] not in ("mean", "std")]

    macro("mmDesigns", str(len({(r["family"], r["r_ref_km"], r["rho"]) for r in cls})))
    lqr1 = one("LQR", 1.0, "holding", "-1")
    macro("mmLqrHoldDvKone", num(f(lqr1, "dv_mps"), 2))
    macro("mmLqrHoldDistKone", f"{f(lqr1, 'mean_dist_km'):.0f}")
    macro("mmLqrRatioKone", num(f(lqr1, "best_classical_ratio"), 2))
    lqr0 = one("LQR", 0.0, "holding", "-1")
    macro("mmDvGrowthKone", f"{f(lqr1, 'dv_mps') / f(lqr0, 'dv_mps'):.0f}")
    # cheapest classical design at k = 1 (parks near the shifted equilibrium)
    c1 = [r for r in cls if float(r["k"]) == 1.0 and r["scenario"] == "holding"
          and float(r["survival"]) == 1.0]
    cheap = min(c1, key=lambda r: f(r, "dv_mps"))
    macro("mmCheapFamKone", cheap["family"])
    macro("mmCheapDvKone", num(f(cheap, "dv_mps"), 2))
    macro("mmCheapDistKone", f"{f(cheap, 'mean_dist_km'):.0f}")
    # tightest LQR (any rho / r_ref) vs tightest of all at k = 4, holding
    c4 = [r for r in cls if float(r["k"]) == 4.0 and r["scenario"] == "holding"
          and float(r["survival"]) == 1.0]
    lq4 = [r for r in c4 if r["family"] == "LQR" and float(r["r_ref_km"]) == 1000.0]
    macro("mmLqrTightKfour", f"{min(f(r, 'mean_dist_km') for r in lq4):.0f}")

    # PPO / DR-PPO ratios to the best classical design
    def rng_ratio(fam, scen, ks, key="best_classical_ratio"):
        v = [f(r, key) for k in ks for r in seeds(fam, k, scen)]
        return min(v), max(v)

    pert = (0.5, 1.0, 2.0, 4.0)
    lo, hi = rng_ratio("PPO", "holding", pert)
    macro("mmPpoHoldMin", num(lo, 2))
    macro("mmPpoHoldMax", num(hi, 2))
    lo, hi = rng_ratio("PPO", "acquisition", pert)
    macro("mmPpoAcqMin", num(lo, 2))
    macro("mmPpoAcqMax", num(hi, 2))
    lo, hi = rng_ratio("DR-PPO", "holding", pert)
    macro("mmDrHoldMin", num(lo, 2))
    macro("mmDrHoldMax", num(hi, 2))
    lo, hi = rng_ratio("DR-PPO", "acquisition", pert)
    macro("mmDrAcqMin", num(lo, 2))
    macro("mmDrAcqMax", num(hi, 2))
    lo, hi = rng_ratio("DR-PPO", "holding", pert, "excess_dv_mps")
    macro("mmDrExcessMin", num(lo, 2))
    macro("mmDrExcessMax", num(hi, 2))
    dr1 = one("DR-PPO", 1.0, "holding", "mean")
    macro("mmDrDistKone", num(f(dr1, "mean_dist_km"), 1))
    macro("mmDrDvKone", num(f(dr1, "dv_mps"), 2))
    d = [f(r, "mean_dist_km") for k in pert for r in seeds("DR-PPO", k, "holding")]
    macro("mmDrDistMin", num(min(d), 1))
    macro("mmDrDistMax", num(max(d), 1))
    dr0 = one("DR-PPO", 0.0, "holding", "mean")
    macro("mmDrDistKzero", num(f(dr0, "mean_dist_km"), 1))
    macro("mmDrRatioKzero", num(f(dr0, "best_classical_ratio"), 1))
    rl = [r for r in rows if r["family"] in ("PPO", "DR-PPO")
          and r["train_seed"] not in ("mean", "std")]
    macro("mmRlRuns", str(len(rl)))
    macro("mmRlDominated", str(sum(f(r, "dominated") == 1.0 for r in rl)))
    best_fam = []
    for r in rl:
        pts = [c for c in cls if float(c["k"]) == float(r["k"]) and c["scenario"] == r["scenario"]
               and float(c["survival"]) == 1.0
               and f(c, "mean_dist_km") <= f(r, "mean_dist_km")]
        best_fam.append(min(pts, key=lambda c: f(c, "dv_mps"))["family"])
    macro("mmBiasBest", str(best_fam.count("Bias-LQR")))
    macro("mmRlKept", f"{100 * np.mean([f(r, 'survival') for r in rl]):.0f}")

    # eccentricity alone: LQR rho = 1, holding, k = 4, vs nominal
    e4 = [r for r in parts if float(r["k"]) == 4.0 and r["effect"] == "eccentricity only"
          and r["family"] == "LQR" and r["scenario"] == "holding"][0]
    macro("mmEccDvKfour", num(f(e4, "dv_mps"), 3))
    macro("mmNomDvLqr", num(f(lqr0, "dv_mps"), 3))
    macro("mmEccKfour", f"{4 * E_EARTH:.3f}")

    # DR-PPO's sunward bias at k = 0 (days 100-365, holding, test seeds)
    _, dr_ctrl = load_policy(ROOT / "models", "ppo_dr_s0")
    xs = []
    for seed in TEST_SEEDS:
        sim = StationKeepingSim(seed=seed, perturbation=perturbation_for(0.0, seed))
        xs.append(run_episode(sim, dr_ctrl, offset_km=(0.0, 0.0))["pos_km"][100:, 0].mean())
    macro("mmDrSunwardKzero", num(-float(np.mean(xs)), 1))

    # table
    lines = []
    for scen, title in (("holding", "Holding"), ("acquisition", "Recovery")):
        lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}} \\")
        for k in (0.0, 0.5, 1.0, 2.0, 4.0):
            def cell(r):
                dist = f(r, "mean_dist_km")
                return rf"{f(r, 'dv_mps'):.2f} ({dist:.0f})" if dist >= 10 else \
                    rf"{f(r, 'dv_mps'):.2f} ({dist:.1f})"
            cells = [cell(one("LQR", k, scen, "-1")), cell(one("Bias-LQR", k, scen, "-1")),
                     cell(one("PPO", k, scen, "mean")), cell(one("DR-PPO", k, scen, "mean"))]
            for fam in ("PPO", "DR-PPO"):
                m, sd = one(fam, k, scen, "mean"), one(fam, k, scen, "std")
                cells.append(rf"${f(m, 'best_classical_ratio'):.2f} \pm "
                             rf"{f(sd, 'best_classical_ratio'):.2f}$")
            lines.append(rf"{k:g} & " + " & ".join(cells) + r" \\")
        if scen == "holding":
            lines.append(r"\addlinespace")
    write_table("tab_mismatch.tex", r"""
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{4}{c}{$\dv$ [m/s] (mean distance [km])} & \multicolumn{2}{c}{$\times$ best classical} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
$k$ & LQR & Bias-LQR & PPO & DR-PPO & PPO & DR-PPO \\
\midrule
""" + "\n".join(lines).replace(r"\multicolumn{8}", r"\multicolumn{7}") + r"""
\bottomrule
\end{tabular}""")



def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": 1.4, "pdf.fonttype": 42,
        "font.family": "serif", "mathtext.fontset": "cm",
    })
    return plt


def _plain_log(ax, axis="both"):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    axes = {"x": [ax.xaxis], "y": [ax.yaxis], "both": [ax.xaxis, ax.yaxis]}[axis]
    for a in axes:
        a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a.set_major_formatter(fmt)
        a.set_minor_formatter(NullFormatter())


def figures():
    """Compact, print-sized versions of the seed-study and robustness figures,
    drawn from the CSVs (no simulation), saved as vector PDF in figures/."""
    plt = _style()
    lqr = load("seed_study_lqr.csv")
    per = [r for r in load("seed_study.csv") if r["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    titles = {"holding": "(a) Holding at $L_1$", "acquisition": "(b) Recovery from 1000 km"}
    for ax, scen in zip(axes, ("holding", "acquisition")):
        pts = [r for r in lqr if r["scenario"] == scen]
        x = np.array([float(r["mean_dist_km"]) for r in pts])
        y = np.array([float(r["dv_mps"]) for r in pts])
        left = np.array([r["left_branch"] == "True" for r in pts])
        k = int(np.flatnonzero(left).max())
        ax.plot(x[:k + 1], y[:k + 1], "-", color=COL["LQR"], zorder=1)
        ax.plot(x[k:], y[k:], ":", color=COL["LQR"], lw=1.1, zorder=1)
        for r in per:
            if r["scenario"] != scen:
                continue
            ax.plot(float(r["mean_dist_km"]), float(r["dv_mps"]),
                    SEED_MARKERS[int(r["train_seed"])], color=COL[r["config"]], ms=6,
                    mec="white", mew=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log(ax)
        ax.set_xlabel("mean distance from $L_1$ [km]")
        ax.set_title(titles[scen], loc="left")
        ax.grid(True, which="major", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$\Delta v$ over one year [m/s]")
    axes[0].set_xlim(0.33, 2.6)
    axes[0].text(1.6, 0.034, "LQR", color=COL["LQR"], fontsize=7.5, va="bottom")
    axes[1].text(320, 1.33, "LQR,\nlazier gains", color=COL["LQR"], fontsize=7, ha="center",
                 va="top")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COL["LQR"], label="LQR, $\\rho$ up to the $\\Delta v$ minimum"),
               Line2D([], [], ls="", marker="o", color=COL["cubic"], label="PPO, cubic map"),
               Line2D([], [], ls="", marker="o", color=COL["linear"], label="PPO, linear map")]
    handles += [Line2D([], [], ls="", marker=m, color="#555555", mfc="none",
                       label=f"training seed {i}") for i, m in enumerate(SEED_MARKERS)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(ROOT / "figures" / "paper_seed_study.pdf")
    plt.close(fig)

    # robustness: fuel vs noise, kept vs offset, accuracy vs unmodelled force
    rs = load("robustness_seeds.csv")

    def agg(sweep, scen, ctrl, key):
        out = []
        levels = sorted({float(r["level"]) for r in rs if r["sweep"] == sweep})
        for lv in levels:
            s = [r for r in rs if r["sweep"] == sweep and float(r["level"]) == lv
                 and r["scenario"] == scen and r["controller"] == ctrl]
            if key == "kept":
                out.append(100 * np.mean([r["survived"] == "True" for r in s]))
            else:
                out.append(mean_of(s, key))
        return np.array(levels), np.array(out)

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.35))
    lines = [("LQR", "LQR", "-", COL["LQR"], None),
             ("LQR (sat.)", "LQR, saturated", "--", "#85c1e9", None)]
    lines += [(m, f"PPO cubic, seed {i}", "-", COL["cubic"], SEED_MARKERS[i])
              for i, m in enumerate(CUBIC)]
    panels = (("noise", "holding", "dv_mps", "(a) All errors $\\times k$ (holding)",
               r"$\Delta v$ [m/s]", "error scale $k$"),
              ("disturbance", "holding", "rms_late_km", "(b) Unmodelled force $\\times k$",
               "RMS distance, days 183–365 [km]", "force scale $k$"),
              ("offset", "acquisition", "kept", "(c) Large start offsets",
               "runs kept 1 year [%]", "start offset [km]"))
    for ax, (sweep, scen, key, title, ylabel, xlabel) in zip(axes, panels):
        for ctrl, label, ls, c, mk in lines:
            if sweep != "offset" and ctrl == "LQR (sat.)":
                continue                         # never saturates at these levels
            x, y = agg(sweep, scen, ctrl, key)
            keep = x > 0
            ax.plot(x[keep], y[keep], ls, color=c, marker=mk or "", ms=3.5, lw=1.2,
                    label=label, zorder=4 if "sat" in ctrl else 2)
        ax.set_xscale("log")
        _plain_log(ax, "x")
        if key != "kept":
            ax.set_yscale("log")
            _plain_log(ax, "y")
        else:
            ax.set_ylim(-5, 105)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
    h, lab = axes[2].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    fig.savefig(ROOT / "figures" / "paper_robustness.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- 8. memory and residual RL
def phase8():
    """Phase 8 (scripts/13 and 14): RL with memory and residual RL on Bias-LQR."""
    rows = load("phase8.csv")
    conf = load("confirmation.csv")
    f = lambda r, k: float(r[k])  # noqa: E731
    seeds = lambda fam, k, scen: [r for r in rows if r["family"] == fam  # noqa: E731
                                  and float(r["k"]) == k and r["scenario"] == scen
                                  and r["train_seed"] not in ("mean", "std", "-1")]
    one = lambda fam, k, scen, st: next(r for r in rows if r["family"] == fam  # noqa: E731
                                        and float(r["k"]) == k and r["scenario"] == scen
                                        and r["train_seed"] == st)
    pert = (0.5, 1.0, 2.0, 4.0)
    # memory
    mem_vs_dr = [f(one("MEM-PPO", k, s, "mean"), "best_classical_ratio")
                 / f(one("DR-PPO", k, s, "mean"), "best_classical_ratio")
                 for k in (0.0, *pert) for s in ("holding", "acquisition")]
    macro("peMemWorseCases", str(sum(v > 1 for v in mem_vs_dr)))
    macro("peMemCases", str(len(mem_vs_dr)))
    m0 = one("MEM-PPO", 0.0, "holding", "mean")
    macro("peMemHoldDvKzero", num(f(m0, "dv_mps"), 2))
    macro("peMemHoldRatioKzero", num(f(m0, "best_classical_ratio"), 1))
    d = [f(r, "mean_dist_km") for k in pert for r in seeds("MEM-PPO", k, "holding")]
    macro("peMemDistMin", num(min(d), 1))
    macro("peMemDistMax", f"{max(d):.0f}")
    # residual
    nd = [r for k in (0.5, 1.0, 2.0) for r in seeds("RES-PPO", k, "holding")]
    macro("peResNondomCases", str(sum(f(r, "dominated") == 0 for r in nd)))
    macro("peResNondomOf", str(len(nd)))
    env = [f(one("RES-PPO", k, "holding", "mean"), "envelope_ratio") for k in (0.5, 1.0, 2.0)]
    macro("peResEnvMin", f"{min(env):.3f}")
    macro("peResEnvMax", f"{max(env):.3f}")
    acq = [f(r, "envelope_ratio") for k in (0.0, *pert) for r in seeds("RES-PPO", k, "acquisition")]
    macro("peResAcqMin", num(min(acq), 2))
    macro("peResAcqMax", num(max(acq), 2))
    macro("peResAcqDom", str(sum(f(r, "dominated") == 1 for k in (0.0, *pert)
                                 for r in seeds("RES-PPO", k, "acquisition"))))
    macro("peResAcqOf", str(len(acq)))
    b1, r1 = one("RES base", 1.0, "holding", "-1"), one("RES-PPO", 1.0, "holding", "mean")
    macro("peBaseDistKone", f"{f(b1, 'mean_dist_km'):.0f}")
    macro("peBaseDvKone", num(f(b1, "dv_mps"), 2))
    macro("peResDistKone", num(f(r1, "mean_dist_km"), 1))
    macro("peResDvKone", num(f(r1, "dv_mps"), 2))
    # confirmation (seeds 200-299)
    macro("peConfEnvMin", f"{min(f(r, 'envelope_ratio') for r in conf):.3f}")
    macro("peConfEnvMax", f"{max(f(r, 'envelope_ratio') for r in conf):.3f}")
    macro("peConfLoMin", f"{min(f(r, 'ci_low') for r in conf):.3f}")
    macro("peConfHiMax", f"{max(f(r, 'ci_high') for r in conf):.3f}")
    macro("peConfDominated", str(sum(f(r, "dominated") == 1 for r in conf)))
    macro("peConfCases", str(len(conf)))
    macro("peConfBelow", str(sum(f(r, "ci_high") < 1.0 for r in conf)))
    macro("peConfAbove", str(sum(f(r, "ci_low") > 1.0 for r in conf)))
    fs = load("frontier_search.csv")
    macro("peFsDesigns", str(int(float(fs[0]["n_designs"]))))
    macro("peFsCases", str(len(fs)))
    macro("peFsDominated", str(sum(f(r, "dominated") == 1 for r in fs)))
    macro("peFsBelow", str(sum(f(r, "envelope_ratio") < 1 for r in fs)))
    macro("peFsEnvMin", f"{min(f(r, 'envelope_ratio') for r in fs):.4f}")
    macro("peFsEnvMax", f"{max(f(r, 'envelope_ratio') for r in fs):.4f}")
    macro("peFsMarginMax", f"{100 * (1 - min(f(r, 'envelope_ratio') for r in fs)):.1f}")
    macro("peFsSaveMax", f"{max((1 - f(r, 'envelope_ratio')) * f(r, 'dv_mps') for r in fs):.3f}")

    lines = []
    for scen, title in (("holding", "Holding"), ("acquisition", "Recovery")):
        lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\")
        for k in (0.0, *pert):
            cells = []
            for fam in ("DR-PPO", "MEM-PPO", "RES-PPO"):
                m, sd = one(fam, k, scen, "mean"), one(fam, k, scen, "std")
                cells.append(rf"${f(m, 'best_classical_ratio'):.3f} \pm "
                             rf"{f(sd, 'best_classical_ratio'):.3f}$")
            dom = sum(f(r, "dominated") == 0 for r in seeds("RES-PPO", k, scen))
            lines.append(rf"{k:g} & " + " & ".join(cells) + rf" & {dom}/5 \\")
        if scen == "holding":
            lines.append(r"\addlinespace")
    write_table("tab_phase8.tex", r"""
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{3}{c}{$\dv$ $\times$ best classical at same accuracy} & RES-PPO seeds \\
\cmidrule(lr){2-4}
$k$ & DR-PPO & MEM-PPO & RES-PPO & not dominated \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


def mismatch_figure():
    """Compact fuel/accuracy plane at k = 0, 1, 4 for the paper (from the CSVs)."""
    plt = _style()
    cls = load("model_mismatch_classical.csv")
    rows = [r for r in load("model_mismatch.csv")
            if r["family"] in ("PPO", "DR-PPO") and r["train_seed"] not in ("mean", "std")]
    col = {"LQR": "#2471a3", "LQI": "#1e8449", "Bias-LQR": "#5d6d7e",
           "PPO": "#d68910", "DR-PPO": "#c0392b"}
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.3))
    for i, scen in enumerate(("holding", "acquisition")):
        for j, k in enumerate((0.0, 1.0, 4.0)):
            ax = axes[i, j]
            for fam in ("LQR", "LQI", "Bias-LQR"):
                pts = sorted(((float(c["mean_dist_km"]), float(c["dv_mps"])) for c in cls
                              if float(c["k"]) == k and c["scenario"] == scen
                              and c["family"] == fam and float(c["survival"]) == 1.0))
                env, best = [], float("inf")
                for d, v in pts:
                    if v < best:
                        env.append((d, v))
                        best = v
                ax.plot(*zip(*env), "-", color=col[fam], lw=1.2, label=fam)
            for fam, mk in (("PPO", "o"), ("DR-PPO", "^")):
                sel = [r for r in rows if r["family"] == fam and float(r["k"]) == k
                       and r["scenario"] == scen]
                ax.plot([float(r["mean_dist_km"]) for r in sel], [float(r["dv_mps"]) for r in sel],
                        mk, color=col[fam], ms=3.5, mec="white", mew=0.4, label=fam)
            ax.set_xscale("log")
            ax.set_yscale("log")
            _plain_log(ax)
            ax.set_title(f"({'abc'[j] if i == 0 else 'def'[j]}) "
                         f"{'Holding' if scen == 'holding' else 'Recovery'}, $k = {k:g}$",
                         loc="left")
            ax.grid(True, which="major", color="#e6e6e6", lw=0.5)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(r"$\Delta v$ per year [m/s]")
            if i == 1:
                ax.set_xlabel("mean distance [km]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.8, h_pad=0.8)
    fig.savefig(ROOT / "figures" / "paper_mismatch.pdf")
    plt.close(fig)


def phase8_figure():
    """Paper figure for phase 8: envelopes, RES base, DR/MEM/RES agents (from the CSVs)."""
    plt = _style()
    cls = load("model_mismatch_classical.csv")
    rows = load("phase8.csv")
    col = {"LQR": "#2471a3", "LQI": "#1e8449", "Bias-LQR": "#5d6d7e", "DR-PPO": "#c0392b",
           "MEM-PPO": "#8e44ad", "RES-PPO": "#16a085"}
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.3))
    for i, scen in enumerate(("holding", "acquisition")):
        for j, k in enumerate((0.0, 1.0, 4.0)):
            ax = axes[i, j]
            for fam in ("LQR", "LQI", "Bias-LQR"):
                pts = sorted(((float(c["mean_dist_km"]), float(c["dv_mps"])) for c in cls
                              if float(c["k"]) == k and c["scenario"] == scen
                              and c["family"] == fam and float(c["survival"]) == 1.0))
                env, best = [], float("inf")
                for d, v in pts:
                    if v < best:
                        env.append((d, v))
                        best = v
                ax.plot(*zip(*env), "-", color=col[fam], lw=1.1, label=fam)
            b = next(r for r in rows if r["family"] == "RES base" and float(r["k"]) == k
                     and r["scenario"] == scen)
            ax.plot(float(b["mean_dist_km"]), float(b["dv_mps"]), "x", color="black", ms=5,
                    mew=1.2, label="RES base")
            for fam, mk in (("DR-PPO", "^"), ("MEM-PPO", "s"), ("RES-PPO", "o")):
                sel = [r for r in rows if r["family"] == fam and float(r["k"]) == k
                       and r["scenario"] == scen and r["train_seed"] not in ("mean", "std")]
                ax.plot([float(r["mean_dist_km"]) for r in sel],
                        [float(r["dv_mps"]) for r in sel], mk, color=col[fam], ms=3.2,
                        mec="white", mew=0.3, label=fam)
            ax.set_xscale("log")
            ax.set_yscale("log")
            _plain_log(ax)
            ax.set_title(f"({'abc'[j] if i == 0 else 'def'[j]}) "
                         f"{'Holding' if scen == 'holding' else 'Recovery'}, $k = {k:g}$",
                         loc="left")
            ax.grid(True, which="major", color="#e6e6e6", lw=0.5)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(r"$\Delta v$ per year [m/s]")
            if i == 1:
                ax.set_xlabel("mean distance [km]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.01),
               columnspacing=0.9, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.8, h_pad=0.8)
    fig.savefig(ROOT / "figures" / "paper_phase8.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- 9. the Moon
MOON_FAMS = ("DR-PPO", "MEM-PPO", "RES-PPO", "LSTM-PPO", "DR-PPO-M", "RES-PPO-M")
AGNOSTIC = ("LQR", "LQI", "Bias-LQR")


def moon():
    """Phase 9 (scripts/18, 19): the Moon as unmodelled physics."""
    rows = load("moon.csv")
    dense = {(r["condition"], r["family"], r["train_seed"]): r
             for r in load("moon_frontier.csv")} if (EXP / "moon_frontier.csv").exists() else {}
    cls = load("moon_classical.csv")
    conds = ("Moon", "k=1 + Moon")
    f = lambda r, k: float(r[k])  # noqa: E731
    seeds = lambda fam, c, scen: [r for r in rows if r["family"] == fam  # noqa: E731
                                  and r["condition"] == c and r["scenario"] == scen
                                  and r["train_seed"] not in ("mean", "std", "-1")]
    fams = [fam for fam in MOON_FAMS if seeds(fam, conds[0], "holding")]
    macro("moDesigns", str(len({(r["family"], r["r_ref_km"], r["rho"]) for r in cls})))
    macro("moDenseDesigns", str(len({(r["family"], r["r_ref_km"], r["rho"])
                                     for r in load("moon_frontier_designs.csv")}))
          if dense else "0")

    def best_dv(c, fam_set, dist):
        ok = [f(p, "dv_mps") for p in cls if p["condition"] == c and p["scenario"] == "holding"
              and p["family"] in fam_set and f(p, "survival") == 1.0
              and f(p, "mean_dist_km") <= dist]
        return min(ok) if ok else float("nan")

    # classical: Harmonic vs Bias within 10 km
    for c, tag in zip(conds, ("Only", "Kone")):
        macro(f"moHarmTen{tag}", num(best_dv(c, ("Harmonic-LQR",), 10.0), 2))
        macro(f"moBiasTen{tag}", num(best_dv(c, ("Bias-LQR",), 10.0), 2))
    lines = []
    for fam in fams:
        cells = []
        for c in conds:
            sel = seeds(fam, c, "holding")
            allr = [f(r, "best_classical_ratio") for r in sel]
            agn = [f(r, "dv_mps") / best_dv(c, AGNOSTIC, f(r, "mean_dist_km")) for r in sel]
            nd = sum((f(dense[(c, fam, r["train_seed"])], "dominated") if dense
                      else f(r, "dominated")) == 0 for r in sel)
            cells += [rf"${np.mean(agn):.2f} \pm {np.std(agn, ddof=1):.2f}$",
                      rf"${np.mean(allr):.2f} \pm {np.std(allr, ddof=1):.2f}$", f"{nd}/5"]
        label = fam.replace("-M", r"-M$^{\dagger}$")
        lines.append(rf"{label} & " + " & ".join(cells) + r" \\")
    write_table("tab_moon.tex", r"""
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{Moon} & \multicolumn{3}{c}{$k = 1$ + Moon} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
 & $\times$ agnostic & $\times$ all & not dom. & $\times$ agnostic & $\times$ all & not dom. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")
    # text macros
    for fam, tag in (("DR-PPO", "Dr"), ("MEM-PPO", "Mem"), ("RES-PPO", "Res"),
                     ("DR-PPO-M", "Drm"), ("RES-PPO-M", "Resm")):
        if fam not in fams:
            continue
        for c, ct in zip(conds, ("Only", "Kone")):
            sel = seeds(fam, c, "holding")
            macro(f"mo{tag}All{ct}", num(np.mean([f(r, "best_classical_ratio") for r in sel]), 2))
    drm = seeds("DR-PPO-M", "k=1 + Moon", "holding")
    agn = [f(r, "dv_mps") / best_dv("k=1 + Moon", AGNOSTIC, f(r, "mean_dist_km")) for r in drm]
    macro("moDrmAgnMin", num(min(agn), 2))
    macro("moDrmAgnMax", num(max(agn), 2))
    resm = [r for c in conds for r in seeds("RES-PPO-M", c, "holding")]
    env = [f(dense[(r["condition"], "RES-PPO-M", r["train_seed"])], "envelope_ratio") if dense
           else f(r, "envelope_ratio") for r in resm]
    macro("moResmEnvMin", f"{min(env):.3f}")
    macro("moResmEnvMax", f"{max(env):.3f}")
    if dense:
        d = [r for r in dense.values() if r["family"] == "DR-PPO-M"]
        macro("moDrmDenseNd", str(sum(f(r, "dominated") == 0 for r in d)))
        macro("moDrmDenseOf", str(len(d)))
        e = [f(r, "envelope_ratio") for r in d if f(r, "dominated") == 0]
        macro("moDrmDenseEnvMin", f"{min(e):.3f}" if e else "--")
        macro("moDrmDenseEnvMax", f"{max(e):.3f}" if e else "--")
        dr = [r for r in dense.values() if r["family"] == "DR-PPO-M"
              and r["condition"] == "k=1 + Moon"]
        macro("moDrmCoarseNd", str(sum(f(r, "dominated_coarse") == 0 for r in dr)))
    if (EXP / "moon_confirmation.csv").exists():                  # scripts/20
        conf = load("moon_confirmation.csv")
        macro("moConfNd", str(sum(f(r, "dominated") == 0 for r in conf)))
        macro("moConfOf", str(len(conf)))
        macro("moConfEnvMin", f"{min(f(r, 'envelope_ratio') for r in conf):.3f}")
        macro("moConfEnvMax", f"{max(f(r, 'envelope_ratio') for r in conf):.3f}")
        macro("moConfDesigns", str(len(load("moon_confirmation_classical.csv"))))
        save = [f(r, "dv_mps") * (1 - f(r, "envelope_ratio")) for r in conf]
        macro("moConfSaveMax", num(max(save), 2))
        dists = sorted(f(r, "mean_dist_km") for r in conf)
        macro("moConfDistMin", f"{dists[0]:.0f}")
        macro("moConfDistMax", f"{dists[-1]:.0f}")
    # recovery
    rec = [r for fam in fams for c in conds for r in seeds(fam, c, "acquisition")]
    macro("moRecNd", str(sum(f(r, "dominated") == 0 for r in rec)))
    macro("moRecOf", str(len(rec)))


# ---------------------------------------------------------------- 9. flown missions
def missions():
    """Phase 9 (scripts/17): yearly delta-v of flown missions vs. this work at k = 1."""
    rows = load("mission_benchmark.csv")
    mine = [r for r in rows if r["source"] == "this work"]
    flown = [r for r in rows if r["source"] == "flown mission"]
    f = lambda r: float(r["dv_per_year_mps"])  # noqa: E731
    macro("msFlownMin", num(min(f(r) for r in flown), 2))
    macro("msFlownMax", num(max(f(r) for r in flown), 2))
    for tag, yrs in (("", "k = 1"), ("Moon", "k = 1 + Moon")):
        loose = next(r for r in mine if r["name"] == "classical, cheapest" and r["years"] == yrs)
        tight = next(r for r in mine if r["name"].startswith("classical, cheapest within")
                     and r["years"] == yrs)
        macro(f"ms{tag}LooseDv", num(f(loose), 2))
        macro(f"ms{tag}LooseDist", f"{float(loose['mean_dist_km']):.0f}")
        macro(f"ms{tag}TightDv", num(f(tight), 2))
        macro(f"ms{tag}TightDist", num(float(tight["mean_dist_km"]), 1))
    cite = {"Roberts 2003": r"\cite{roberts2003}", "Roberts 2011": r"\cite{roberts2011}",
            "Dichmann et al. 2014": r"\cite{dichmann2014}"}
    lines = []
    for r in flown:
        how = "" if r["how"] == "reported" else r"$^{*}$"
        orbit = r["orbit"].replace("A_y ~", r"$A_y \sim$").replace(",000", r"\,000")
        lines.append(rf"{r['name']}, {r['years'].replace('-', '--')} & {cite[r['reference']]} & "
                     rf"{f(r):.2f}{how} & {orbit} \\")
    lines.append(r"\addlinespace")
    for r in mine:
        moon_tag = " + Moon" if "Moon" in r["years"] else ""
        name = r["name"].replace("classical, cheapest within 10 km", r"classical, $\le$10~km")
        lines.append(rf"{name}{moon_tag} & this work & {f(r):.2f} & "
                     rf"hold $L_1$, mean {float(r['mean_dist_km']):.1f}~km \\")
    write_table("tab_missions.tex", r"""
\begin{tabular}{llll}
\toprule
Mission / controller & Source & $\dv$ per year [m/s] & Orbit / task \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for step in (dynamics, lqr_phase3, comparison, ratios, robustness, thrust_logs,
                 hyperparameters, mismatch, phase8, moon, missions, figures, mismatch_figure, phase8_figure):
        step()
        print(f"{step.__name__} done", flush=True)
    text = "% Generated by scripts/11_paper_assets.py -- do not edit by hand.\n"
    text += "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in sorted(macros.items()))
    (OUT / "numbers.tex").write_text(text)
    print(f"Wrote {len(macros)} macros and the tables to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
