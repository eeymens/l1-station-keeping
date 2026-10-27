"""Phase 4: train a DQN agent to keep the spacecraft at L1.

Every `--eval-every` steps the current greedy policy flies the standard
evaluation protocol (holding + acquisition, 5 seeds; see cr3bp/evaluation.py).
The best policy so far is saved to models/dqn_l1.zip and the learning curve
to experiments/dqn_training.csv + figures/dqn_learning_curve.png.

Run from the repository root:
    python scripts/06_train_dqn.py --steps 500000
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from cr3bp.envs import L1StationKeepingEnv, PolicyController
from cr3bp.evaluation import evaluate, summarize

ROOT = Path(__file__).resolve().parents[1]


def score(summary):
    """One number to rank policies: survival first, then fuel, then accuracy."""
    s = 0.0
    for m in summary.values():
        s += 100.0 * m["days"] / 365.0 - 10.0 * m["dv_mps"] - np.log10(1.0 + m["mean_dist_km"])
    return s / len(summary)


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, every, out_path):
        super().__init__()
        self.eval_env, self.every, self.out_path = eval_env, every, out_path
        self.best, self.log = -np.inf, []

    def _on_step(self):
        if self.n_calls % self.every:
            return True
        ctrl = PolicyController(self.eval_env,
                                lambda f: self.model.predict(f, deterministic=True)[0])
        summ = summarize(evaluate(ctrl))
        sc = score(summ)
        row = {"step": self.num_timesteps, "score": sc}
        for name, m in summ.items():
            row.update({f"{name}_survival": m["survival"], f"{name}_days": m["days"],
                        f"{name}_dv_mps": m["dv_mps"], f"{name}_dist_km": m["mean_dist_km"]})
        self.log.append(row)
        flag = ""
        if sc > self.best:
            self.best = sc
            self.model.save(self.out_path)
            flag = "  <- best"
        h, a = summ["holding"], summ["acquisition"]
        print(f"{self.num_timesteps:>8}  score {sc:7.2f} | hold: {h['survival']:.0%} "
              f"{h['dv_mps']:.3f} m/s {h['mean_dist_km']:8.2f} km | acq: {a['survival']:.0%} "
              f"{a['days']:5.0f} d {a['dv_mps']:.3f} m/s {a['mean_dist_km']:8.2f} km{flag}",
              flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25_000)
    args = p.parse_args()

    env = Monitor(L1StationKeepingEnv(seed=args.seed))
    (ROOT / "models").mkdir(exist_ok=True)
    model_path = ROOT / "models" / "dqn_l1"

    model = DQN(
        "MlpPolicy", env, seed=args.seed, verbose=0,
        learning_rate=5e-4, buffer_size=200_000, learning_starts=10_000,
        batch_size=256, gamma=0.99, train_freq=4, gradient_steps=1,
        target_update_interval=2_000, exploration_fraction=0.3,
        exploration_final_eps=0.02, policy_kwargs={"net_arch": [128, 128]},
    )
    cb = EvalCallback(L1StationKeepingEnv(), args.eval_every, model_path)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    print(f"Training took {(time.time() - t0) / 60:.1f} min; best score {cb.best:.2f}")

    (ROOT / "experiments").mkdir(exist_ok=True)
    with open(ROOT / "experiments" / "dqn_training.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cb.log[0].keys())
        w.writeheader()
        w.writerows(cb.log)

    # learning curve
    steps = np.array([r["step"] for r in cb.log]) / 1000
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, color in (("holding", "#17a589"), ("acquisition", "#2471a3")):
        axes[0].plot(steps, [r[f"{name}_days"] for r in cb.log], "o-", ms=3, color=color, label=name)
        axes[1].plot(steps, [r[f"{name}_dv_mps"] for r in cb.log], "o-", ms=3, color=color, label=name)
        axes[2].semilogy(steps, [r[f"{name}_dist_km"] for r in cb.log], "o-", ms=3, color=color,
                         label=name)
    axes[0].set_ylabel("days kept (max 365)")
    axes[1].set_ylabel("delta-v [m/s]")
    axes[2].set_ylabel("mean distance from L1 [km]")
    for ax, title in zip(axes, ("Survival", "Fuel per episode", "Accuracy")):
        ax.set_xlabel("training steps [thousands]")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "dqn_learning_curve.png", dpi=160)
    print("Saved models/dqn_l1.zip, experiments/dqn_training.csv, figures/dqn_learning_curve.png")


if __name__ == "__main__":
    main()
