"""Gymnasium environment: keep a spacecraft near the Sun-Earth L1 point.

The environment wraps `StationKeepingSim`, so an RL agent faces exactly the
same nonlinear dynamics, navigation errors, thrust errors and fuel
accounting as the LQR baseline.

Observation (8 values, from the noisy navigation estimate)
    [dx, dy] / 1000 km, [vx, vy] / (1 m/s),
    symlog of [dx, dy] in km and of [vx, vy] in mm/s.
    The symlog features keep 0.1 km and 1000 km both visible to the network.

Actions
    discrete:   0 = coast; then 8 directions x 3 thrust levels (25 actions)
    continuous: a 2-vector in [-1, 1]^2, scaled to the maximum thrust
                (norm clipped to 1)

Reward per day
    +1 for staying in the region
    - w_pos * log10(1 + distance / 1 km)
    - w_fuel * delta-v spent this day [m/s]
    and a one-off penalty when the spacecraft is lost (> 10,000 km).

Episodes start at a random offset (up to `max_offset_km`) with a small random
velocity, and last `episode_days` days.
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..control.station_keeping import KM, MPS, MPS2, NoiseModel, StationKeepingSim

THRUST_LEVELS_MPS2 = (1e-8, 1e-7, 1e-6)
N_DIRECTIONS = 8


def symlog(x):
    return np.sign(x) * np.log10(1.0 + np.abs(x))


class L1StationKeepingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, action_type="discrete", episode_days=365, max_offset_km=1000.0,
                 max_vel_mps=0.05, noise=None, w_pos=0.1, w_fuel=10.0, fail_penalty=100.0,
                 u_max_mps2=THRUST_LEVELS_MPS2[-1], dt_days=1.0, seed=None):
        super().__init__()
        self.sim = StationKeepingSim(dt_days=dt_days, noise=noise, seed=seed)
        self.action_type = action_type
        self.episode_steps = int(round(episode_days / dt_days))
        self.max_offset_km = max_offset_km
        self.max_vel_mps = max_vel_mps
        self.w_pos, self.w_fuel, self.fail_penalty = w_pos, w_fuel, fail_penalty
        self.u_max = u_max_mps2 * MPS2

        if action_type == "discrete":
            self.action_space = spaces.Discrete(1 + N_DIRECTIONS * len(THRUST_LEVELS_MPS2))
            table = [np.zeros(2)]
            for level in THRUST_LEVELS_MPS2:
                for k in range(N_DIRECTIONS):
                    ang = 2 * math.pi * k / N_DIRECTIONS
                    table.append(level * MPS2 * np.array([math.cos(ang), math.sin(ang)]))
            self._action_table = np.array(table)
        elif action_type == "continuous":
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            raise ValueError("action_type must be 'discrete' or 'continuous'")

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self._steps = 0
        self._obs_raw = np.zeros(4)

    # ------------------------------------------------------------ helpers
    def action_to_accel(self, action):
        """Map an action to a commanded acceleration (nondimensional)."""
        if self.action_type == "discrete":
            return self._action_table[int(action)]
        a = np.asarray(action, dtype=float)
        n = np.hypot(*a)
        if n > 1.0:
            a = a / n
        return a * self.u_max

    @staticmethod
    def features(rel):
        """8 observation features from a relative state [dx, dy, vx, vy]."""
        pos_km = rel[:2] / KM
        vel_mps = rel[2:] / MPS
        return np.concatenate([pos_km / 1000.0, vel_mps,
                               symlog(pos_km), symlog(vel_mps * 1000.0)]).astype(np.float32)

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
        options = options or {}
        rng = self.np_random
        if "offset_km" in options:
            offset = np.asarray(options["offset_km"], dtype=float)
            vel = np.asarray(options.get("vel_mps", (0.0, 0.0)), dtype=float)
        else:
            r = self.max_offset_km * math.sqrt(rng.uniform())          # uniform over the disc
            ang = rng.uniform(0, 2 * math.pi)
            offset = r * np.array([math.cos(ang), math.sin(ang)])
            vel = rng.normal(0.0, self.max_vel_mps / 2, 2)
        self._obs_raw = self.sim.reset(offset, vel)
        self._steps = 0
        return self.features(self._obs_raw), {}

    def step(self, action):
        dv_before = self.sim.dv
        self._obs_raw, failed = self.sim.step(self.action_to_accel(action))
        self._steps += 1
        dv_mps = (self.sim.dv - dv_before) / MPS
        dist_km = self.sim.distance_km

        reward = 1.0 - self.w_pos * math.log10(1.0 + dist_km) - self.w_fuel * dv_mps
        if failed:
            reward -= self.fail_penalty
        truncated = self._steps >= self.episode_steps
        info = {"distance_km": dist_km, "dv_mps": dv_mps, "dv_total_mps": self.sim.dv / MPS}
        return self.features(self._obs_raw), float(reward), bool(failed), truncated, info


class PolicyController:
    """Wrap a trained policy so `run_episode` can fly it like the LQR.

    ``predict(features) -> action`` is any function, e.g. an SB3 model's
    ``lambda f: model.predict(f, deterministic=True)[0]``.
    """

    def __init__(self, env, predict):
        self.env, self.predict = env, predict

    def __call__(self, obs):
        return self.env.action_to_accel(self.predict(self.env.features(obs)))
