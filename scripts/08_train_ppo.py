"""Phase 5: train a PPO agent with continuous thrust to keep the spacecraft at L1.

Phase 4 showed that DQN's coarse action set (smallest thrust 1e-8 m/s^2)
cannot match LQR's ~2e-9 m/s^2 holding thrust. PPO outputs a continuous
2-D thrust vector instead, so the agent can command any thrust up to the
same maximum (1e-6 m/s^2) that DQN had.

The action map u = u_max |a|^p a/|a| is set by --action-power. p = 3 (default)
gives fine resolution near zero thrust; p = 1 is the plain linear map.

Model selection is the same as for DQN: every --eval-every steps the
deterministic policy flies holding + acquisition on seeds 0-4, and the best
policy is kept. Test seeds (100-119) are untouched until scripts 07 and 09.

Writes models/<name>.zip (+ .json), experiments/<name>_training.csv and
figures/<name>_learning_curve.png.

Run from the repository root:
    python scripts/08_train_ppo.py --steps 2000000 --name ppo_l1
"""

import argparse
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from cr3bp.envs import L1StationKeepingEnv
from cr3bp.training import EvalCallback, save_config, save_training_log

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--action-power", type=float, default=3.0)
    p.add_argument("--w-pos", type=float, default=0.3, help="distance weight in the reward")
    p.add_argument("--w-fuel", type=float, default=10.0, help="delta-v weight in the reward")
    p.add_argument("--name", default="ppo_l1", help="output name for model, CSV and figure")
    args = p.parse_args()
    torch.set_num_threads(1)

    env_kwargs = {"action_type": "continuous", "action_power": args.action_power,
                  "w_pos": args.w_pos, "w_fuel": args.w_fuel}
    env = make_vec_env(L1StationKeepingEnv, n_envs=args.n_envs, seed=args.seed,
                       env_kwargs=env_kwargs)
    (ROOT / "models").mkdir(exist_ok=True)
    model_path = ROOT / "models" / args.name

    model = PPO(
        "MlpPolicy", env, seed=args.seed, verbose=0,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=256, n_epochs=10,
        gamma=args.gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                       "log_std_init": args.log_std_init},
    )
    cb = EvalCallback(L1StationKeepingEnv(**env_kwargs), args.eval_every, model_path)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    print(f"Training took {(time.time() - t0) / 60:.1f} min; best score {cb.best:.2f}")

    save_config(model_path, "ppo", env_kwargs, vars(args))
    (ROOT / "experiments").mkdir(exist_ok=True)
    save_training_log(cb.log, ROOT / "experiments" / f"{args.name}_training.csv",
                      ROOT / "figures" / f"{args.name}_learning_curve.png")
    print(f"Saved models/{args.name}.zip, experiments/{args.name}_training.csv, "
          f"figures/{args.name}_learning_curve.png")


if __name__ == "__main__":
    main()
