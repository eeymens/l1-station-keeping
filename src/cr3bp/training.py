"""Shared pieces for training and loading RL agents (DQN, PPO).

Needs the ``rl`` extra (gymnasium, stable-baselines3).

Every trained model ``models/<name>.zip`` has a sidecar ``models/<name>.json``
with the algorithm and the environment settings it was trained with, so the
comparison scripts rebuild exactly the same action mapping.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from .envs import L1StationKeepingEnv, PolicyController
from .evaluation import evaluate, summarize

ALGOS = {"dqn": DQN, "ppo": PPO}


def score(summary):
    """One number to rank policies: survival first, then fuel, then accuracy."""
    s = 0.0
    for m in summary.values():
        s += 100.0 * m["days"] / 365.0 - 10.0 * m["dv_mps"] - np.log10(1.0 + m["mean_dist_km"])
    return s / len(summary)


class EvalCallback(BaseCallback):
    """Every `every` environment steps, fly the greedy policy through the
    standard evaluation (holding + acquisition, seeds 0-4) and keep the best."""

    def __init__(self, eval_env, every, out_path):
        super().__init__()
        self.eval_env, self.every, self.out_path = eval_env, every, out_path
        self.best, self.log, self._next = -np.inf, [], every

    def _on_step(self):
        # num_timesteps counts steps of all parallel envs, so compare against it
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
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


def save_config(model_path, algo, env_kwargs, train_args):
    Path(model_path).with_suffix(".json").write_text(json.dumps(
        {"algo": algo, "env": env_kwargs, "train": train_args}, indent=2) + "\n")


def load_policy(model_dir, name):
    """Load ``<model_dir>/<name>.zip``. Returns (env, controller) where the
    controller can be flown by `run_episode` / `evaluate` like the LQR."""
    model_dir = Path(model_dir)
    cfg_path = model_dir / f"{name}.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:                                         # phase-4 models predate the sidecar
        cfg = {"algo": "dqn", "env": {}}
    env = L1StationKeepingEnv(**cfg["env"])
    model = ALGOS[cfg["algo"]].load(model_dir / f"{name}.zip", device="cpu")
    return env, PolicyController(env, lambda f: model.predict(f, deterministic=True)[0])


def save_training_log(log, csv_path, fig_path):
    """Write the evaluation log to CSV and plot the learning curve."""
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)

    steps = np.array([r["step"] for r in log]) / 1000
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, color in (("holding", "#17a589"), ("acquisition", "#2471a3")):
        axes[0].plot(steps, [r[f"{name}_days"] for r in log], "o-", ms=3, color=color, label=name)
        axes[1].plot(steps, [r[f"{name}_dv_mps"] for r in log], "o-", ms=3, color=color,
                     label=name)
        axes[2].semilogy(steps, [r[f"{name}_dist_km"] for r in log], "o-", ms=3, color=color,
                         label=name)
    axes[0].set_ylabel("days kept (max 365)")
    axes[1].set_ylabel("delta-v [m/s]")
    axes[2].set_ylabel("mean distance from L1 [km]")
    for ax, title in zip(axes, ("Survival", "Fuel per episode", "Accuracy")):
        ax.set_xlabel("training steps [thousands]")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
