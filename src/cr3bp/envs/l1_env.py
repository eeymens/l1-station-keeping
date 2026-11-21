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
    continuous: a 2-vector a in [-1, 1]^2 (norm clipped to 1), mapped to
                u = u_max * |a|^p * a/|a|. With p = 1 the map is linear; with
                p = 3 small actions give very small thrusts (a = 0.13 ->
                0.002 u_max), so the fine thrust needed for holding is easy
                to reach while the full u_max stays available.

Reward per day
    +1 for staying in the region
    - w_pos * log10(1 + distance / 1 km)
    - w_fuel * delta-v spent this day [m/s]
    and a one-off penalty when the spacecraft is lost (> 10,000 km).

Episodes start at a random offset (up to `max_offset_km`) with a small random
velocity, and last `episode_days` days.

Phase 7 options
    perturbation_k  fixed model-mismatch severity (Earth's eccentricity and
                    solar radiation pressure, see `cr3bp.perturbations`)
    dr_k_max        domain randomisation: every episode draws k ~ U(0, dr_k_max),
                    Earth's starting true anomaly ~ U(0, 2 pi) and an SRP
                    area-to-mass factor ~ U(0.5, 1.5). 0 = off (phases 4-6).

Phase 8 options (both default to the phase 4-7 behaviour)
    history_len     the observation also contains the previous history_len - 1
                    feature vectors and the last history_len actions, so a
                    feed-forward policy can infer slowly varying forces (memory).
    residual        dict of `BiasLQRController` settings: the thrust is
                    u = u_BiasLQR + u_agent (norm capped at u_max), and the
                    observation adds the filter's state estimate, its force
                    estimate and the base thrust (hybrid "residual RL").

The observation is assembled by an `ObsBuilder`, which `PolicyController`
shares, so an agent is flown exactly as it was trained.
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..control.lqi import BiasLQRController
from ..control.station_keeping import KM, MPS, MPS2, NoiseModel, StationKeepingSim
from ..perturbations import Perturbation

THRUST_LEVELS_MPS2 = (1e-8, 1e-7, 1e-6)
N_DIRECTIONS = 8


def symlog(x):
    return np.sign(x) * np.log10(1.0 + np.abs(x))


#: Scale for thrust-like observation entries (phase 8), m/s^2.
U_OBS_SCALE_MPS2 = 1e-7


class ObsBuilder:
    """Turns noisy navigation estimates into observations and actions into thrust.

    Stateless for the phase 4-7 agents (features of the current estimate only).
    With history (phase 8) it keeps the last feature vectors and actions; with
    a residual base it runs a `BiasLQRController` alongside the agent.
    """

    def __init__(self, env):
        self.env = env
        self.h = env.history_len
        self.base = (BiasLQRController(**env.residual) if env.residual is not None else None)
        self.n_act = int(np.prod(env.action_space.shape)) if env.action_type == "continuous" else 1
        self.reset()

    @property
    def dim(self):
        n = 8 * self.h + (self.n_act * self.h if self.h > 1 else 0)
        return n + (12 if self.base is not None else 0)

    def reset(self):
        self.feats, self.acts = None, None
        self.u_base = np.zeros(2)
        if self.base is not None:
            self.base.reset()

    def observe(self, obs_raw):
        f = self.env.features(obs_raw)
        if self.h > 1:
            if self.feats is None:                         # first step: repeat, no actions yet
                self.feats = [f] * self.h
                self.acts = [np.zeros(self.n_act, dtype=np.float32)] * self.h
            else:
                self.feats = self.feats[1:] + [f]
            f = np.concatenate([*self.feats, *self.acts])
        if self.base is not None:
            self.u_base = self.base(obs_raw)
            sc = U_OBS_SCALE_MPS2 * MPS2
            x_hat = self.base.xa[:4]
            d_hat = self.base.xa[4:]
            f = np.concatenate([f, self.env.features(x_hat), self.u_base / sc, d_hat / sc])
        return f.astype(np.float32)

    def act(self, action):
        u = self.env.action_to_accel(action)
        if self.h > 1:
            a = np.asarray(action, dtype=np.float32).reshape(self.n_act)
            self.acts = self.acts[1:] + [a]
        if self.base is not None:
            u = self.u_base + u
            mag = np.hypot(*u)
            if mag > self.env.u_max:
                u = u * (self.env.u_max / mag)
            self.base.u_prev = u          # the filter must predict with the thrust applied
        return u


class L1StationKeepingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, action_type="discrete", episode_days=365, max_offset_km=1000.0,
                 max_vel_mps=0.05, noise=None, w_pos=0.1, w_fuel=10.0, fail_penalty=100.0,
                 u_max_mps2=THRUST_LEVELS_MPS2[-1], action_power=1.0, dt_days=1.0, seed=None,
                 perturbation_k=0.0, dr_k_max=0.0, history_len=1, residual=None):
        super().__init__()
        self.history_len = int(history_len)
        self.residual = residual
        self.sim = StationKeepingSim(dt_days=dt_days, noise=noise, seed=seed)
        self.perturbation_k = float(perturbation_k)
        self.dr_k_max = float(dr_k_max)
        self.action_type = action_type
        self.action_power = float(action_power)
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

        self.builder = ObsBuilder(self)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.builder.dim,),
                                            dtype=np.float32)
        self._steps = 0
        self._obs_raw = np.zeros(4)

    # ------------------------------------------------------------ helpers
    def action_to_accel(self, action):
        """Map an action to a commanded acceleration (nondimensional)."""
        if self.action_type == "discrete":
            return self._action_table[int(action)]
        a = np.asarray(action, dtype=float)
        n = np.hypot(*a)
        if n == 0.0:
            return np.zeros(2)
        if n > 1.0:
            a, n = a / n, 1.0
        return a * n ** (self.action_power - 1.0) * self.u_max

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
        if self.dr_k_max > 0.0:
            self.sim.perturbation = Perturbation.from_k(
                rng.uniform(0.0, self.dr_k_max), f0=rng.uniform(0.0, 2 * math.pi),
                srp_scale=rng.uniform(0.5, 1.5))
        elif self.perturbation_k > 0.0:
            self.sim.perturbation = Perturbation.from_k(
                self.perturbation_k, f0=rng.uniform(0.0, 2 * math.pi))
        self._obs_raw = self.sim.reset(offset, vel)
        self._steps = 0
        self.builder.reset()
        return self.builder.observe(self._obs_raw), {}

    def step(self, action):
        dv_before = self.sim.dv
        self._obs_raw, failed = self.sim.step(self.builder.act(action))
        self._steps += 1
        dv_mps = (self.sim.dv - dv_before) / MPS
        dist_km = self.sim.distance_km

        reward = 1.0 - self.w_pos * math.log10(1.0 + dist_km) - self.w_fuel * dv_mps
        if failed:
            reward -= self.fail_penalty
        truncated = self._steps >= self.episode_steps
        info = {"distance_km": dist_km, "dv_mps": dv_mps, "dv_total_mps": self.sim.dv / MPS}
        return self.builder.observe(self._obs_raw), float(reward), bool(failed), truncated, info


class PolicyController:
    """Wrap a trained policy so `run_episode` can fly it like the LQR.

    ``predict(features) -> action`` is any function, e.g. an SB3 model's
    ``lambda f: model.predict(f, deterministic=True)[0]``.
    """

    def __init__(self, env, predict):
        self.env, self.predict = env, predict
        self.builder = ObsBuilder(env)

    def reset(self):
        self.builder.reset()

    def __call__(self, obs):
        return self.builder.act(self.predict(self.builder.observe(obs)))
