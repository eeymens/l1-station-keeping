"""Phase 6: every number and table in the paper, generated from the results.

Nothing in paper/main.tex is typed by hand: this script reads the raw CSVs in
experiments/ (written by scripts 05, 07, 09 and 10), recomputes the few
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for step in (dynamics, lqr_phase3, comparison, ratios, robustness, thrust_logs,
                 hyperparameters, figures):
        step()
        print(f"{step.__name__} done", flush=True)
    text = "% Generated by scripts/11_paper_assets.py -- do not edit by hand.\n"
    text += "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in sorted(macros.items()))
    (OUT / "numbers.tex").write_text(text)
    print(f"Wrote {len(macros)} macros and the tables to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
