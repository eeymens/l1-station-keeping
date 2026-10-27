"""Phase 4: train a DQN agent to keep the spacecraft at L1.

Every `--eval-every` steps the current greedy policy flies the standard
evaluation protocol (holding + acquisition, 5 seeds; see cr3bp/evaluation.py).
The best policy so far is saved to models/dqn_l1.zip and the learning curve
to experiments/dqn_training.csv + figures/dqn_learning_curve.png.

Run from the repository root:
    python scripts/06_train_dqn.py --steps 1000000                      # run 1
    python scripts/06_train_dqn.py --steps 1000000 --lr 1e-4 \
        --target-update 10000 --w-pos 0.3 --name dqn_l1_run2             # run 2
"""

import argparse
import time
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from cr3bp.envs import L1StationKeepingEnv
from cr3bp.training import EvalCallback, save_config, save_training_log

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--target-update", type=int, default=2_000)
    p.add_argument("--w-pos", type=float, default=0.1, help="distance weight in the reward")
    p.add_argument("--name", default="dqn_l1", help="output name for model, CSV and figure")
    args = p.parse_args()

    env = Monitor(L1StationKeepingEnv(seed=args.seed, w_pos=args.w_pos))
    (ROOT / "models").mkdir(exist_ok=True)
    model_path = ROOT / "models" / args.name

    model = DQN(
        "MlpPolicy", env, seed=args.seed, verbose=0,
        learning_rate=args.lr, buffer_size=200_000, learning_starts=10_000,
        batch_size=256, gamma=0.99, train_freq=4, gradient_steps=1,
        target_update_interval=args.target_update, exploration_fraction=0.3,
        exploration_final_eps=0.02, policy_kwargs={"net_arch": [128, 128]},
    )
    cb = EvalCallback(L1StationKeepingEnv(), args.eval_every, model_path)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    print(f"Training took {(time.time() - t0) / 60:.1f} min; best score {cb.best:.2f}")

    save_config(model_path, "dqn", {"action_type": "discrete", "w_pos": args.w_pos}, vars(args))
    (ROOT / "experiments").mkdir(exist_ok=True)
    save_training_log(cb.log, ROOT / "experiments" / f"{args.name}_training.csv",
                      ROOT / "figures" / f"{args.name}_learning_curve.png")
    print(f"Saved models/{args.name}.zip, experiments/{args.name}_training.csv, "
          f"figures/{args.name}_learning_curve.png")


if __name__ == "__main__":
    main()
