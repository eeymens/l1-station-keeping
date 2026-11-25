"""Phase 9: are the simulated fuel costs realistic? A check against flown L1/L2 missions.

The yearly delta-v of our controllers is compared with published station-keeping
costs of real libration-point missions. This is a plausibility check, not a
competition: the missions fly large halo / Lissajous orbits (amplitudes of
10^5 km) with about four impulsive burns a year, while our controllers hold a
point within a few km (or a few hundred km) with a small daily thrust, in a
planar model. Same physics, different task. What the check can show is whether
our numbers land in the right order of magnitude once Earth's eccentricity and
solar radiation pressure (SRP) are on, i.e. at k = 1 (phase 7).

Mission values (yearly station-keeping delta-v, m/s):
    SOHO 1996-98   2.4    Roberts (2003), Libration Point Orbits and Applications,
                          doi:10.1142/9789812704849_0009: "an average of 1.2 m/sec
                          per revolution, or 2.4 m/sec per year"
    SOHO 2004-11   0.36   derived: 0.09 m/s per burn x 4 burns a year,
                          Roberts (2011), AAS 11-485 (NTRS 20110022484)
    WIND 2008-11   0.68   derived: 0.17 m/s per burn x 4 burns a year, same source
    ACE  2003-11   1.0    same source: "approximately one meter per second per year"
    JWST (L2)      2.43   planned budget, 25.5 m/s over 10.5 years: Dichmann,
                          Alberding & Yu (2014), ISSFD 24 (NTRS 20140007519)

Our values come from the committed CSVs (test seeds 100-119, one year, holding
scenario): the cheapest classical design overall and the cheapest one that
stays within 10 km of L1 (scripts/12), the RL families (scripts/13, 16), and
the same two classical rows with the Moon added (scripts/18), since the flown
missions of course have the Moon too.

Writes experiments/mission_benchmark.csv and figures/mission_benchmark.png.

Run from the repository root (seconds):
    python scripts/17_mission_benchmark.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

MISSIONS = [
    # name, years, dv per year, how obtained, orbit, reference
    ("SOHO", "1996-98", 2.4, "reported", "halo, A_y ~ 670,000 km", "Roberts 2003"),
    ("SOHO", "2004-11", 0.09 * 4, "0.09 m/s/burn x 4/yr", "halo, A_y ~ 670,000 km",
     "Roberts 2011"),
    ("WIND", "2008-11", 0.17 * 4, "0.17 m/s/burn x 4/yr", "large Lissajous", "Roberts 2011"),
    ("ACE", "2003-11", 1.0, "reported", "Lissajous, A_y ~ 260,000 km", "Roberts 2011"),
    ("JWST (L2)", "planned", 25.5 / 10.5, "25.5 m/s / 10.5 yr", "halo about L2",
     "Dichmann et al. 2014"),
]
K = 1.0                       # real eccentricity and SRP (phase 7)
TIGHT_KM = 10.0


def read(name):
    with open(EXP / name, newline="") as fh:
        return list(csv.DictReader(fh))


def ours():
    out = []
    cl = [r for r in read("model_mismatch_classical.csv")
          if float(r["k"]) == K and r["scenario"] == "holding" and float(r["survival"]) == 1.0]
    cl.sort(key=lambda r: float(r["dv_mps"]))
    loose = cl[0]
    tight = next(r for r in cl if float(r["mean_dist_km"]) <= TIGHT_KM)
    for label, r in (("classical, cheapest", loose),
                     (f"classical, cheapest within {TIGHT_KM:g} km", tight)):
        out.append({"source": "this work", "name": label, "years": f"k = {K:g}",
                    "dv_per_year_mps": float(r["dv_mps"]),
                    "mean_dist_km": float(r["mean_dist_km"]),
                    "how": f"{r['family']}, rho = {float(r['rho']):g}, "
                           f"r_ref = {float(r['r_ref_km']):.0f} km",
                    "orbit": "hold a point", "reference": "scripts/12"})
    fams = [("phase8.csv", f) for f in ("DR-PPO", "RES-PPO")]
    if (EXP / "phase9.csv").exists():
        fams.append(("phase9.csv", "LSTM-PPO"))
    for fname, fam in fams:
        r = next(r for r in read(fname) if r["family"] == fam and float(r["k"]) == K
                 and r["scenario"] == "holding" and r["train_seed"] == "mean")
        out.append({"source": "this work", "name": fam, "years": f"k = {K:g}",
                    "dv_per_year_mps": float(r["dv_mps"]),
                    "mean_dist_km": float(r["mean_dist_km"]),
                    "how": "mean of 5 training seeds", "orbit": "hold a point",
                    "reference": "scripts/13" if fname == "phase8.csv" else "scripts/16"})
    # phase 9: with the Moon too (scripts/18), as the flown missions are
    mo = [r for r in read("moon_classical.csv")
          if r["condition"] == "k=1 + Moon" and r["scenario"] == "holding"
          and float(r["survival"]) == 1.0]
    mo.sort(key=lambda r: float(r["dv_mps"]))
    for label, r in (("classical, cheapest", mo[0]),
                     (f"classical, cheapest within {TIGHT_KM:g} km",
                      next(r for r in mo if float(r["mean_dist_km"]) <= TIGHT_KM))):
        out.append({"source": "this work", "name": label, "years": f"k = {K:g} + Moon",
                    "dv_per_year_mps": float(r["dv_mps"]),
                    "mean_dist_km": float(r["mean_dist_km"]),
                    "how": f"{r['family']}, rho = {float(r['rho']):g}, "
                           f"r_ref = {float(r['r_ref_km']):.0f} km",
                    "orbit": "hold a point", "reference": "scripts/18"})
    return out


def main():
    rows = [{"source": "flown mission", "name": n, "years": y, "dv_per_year_mps": dv,
             "mean_dist_km": "", "how": how, "orbit": orbit, "reference": ref}
            for n, y, dv, how, orbit, ref in MISSIONS] + ours()
    with open(EXP / "mission_benchmark.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"{'':<14}{'':<38}{'m/s per year':>14}{'mean dist [km]':>16}")
    for r in rows:
        d = r["mean_dist_km"]
        print(f"{r['source']:<14}{r['name'] + ' ' + r['years']:<38}"
              f"{r['dv_per_year_mps']:>14.2f}{(f'{d:.1f}' if d != '' else '-'):>16}")
    plot(rows)
    print("Saved experiments/mission_benchmark.csv, figures/mission_benchmark.png")


def plot(rows):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
    for i, r in enumerate(rows):
        mine = r["source"] == "this work"
        ax.barh(i, r["dv_per_year_mps"], color="#16a085" if mine else "#5d6d7e", height=0.6)
        extra = f"  ({r['mean_dist_km']:.0f} km)" if mine else ""
        ax.text(r["dv_per_year_mps"] * 1.06, i, f"{r['dv_per_year_mps']:.2f}{extra}",
                va="center", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['name']} ({r['years']})" for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("station-keeping delta-v per year [m/s]")
    ax.set_title("Flown missions (grey) vs. this work at k = 1 (green, mean distance from L1)",
                 fontsize=10)
    ax.grid(True, axis="x", which="both", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "mission_benchmark.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
