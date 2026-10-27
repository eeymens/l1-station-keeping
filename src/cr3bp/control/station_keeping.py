"""Station-keeping simulator around the Sun-Earth L1 point (planar CR3BP).

The spacecraft follows the full nonlinear CR3BP dynamics. Every control
interval (default 1 day) a controller reads a noisy navigation estimate and
commands a thrust acceleration, which is held constant over the interval
(zero-order hold). Thrust execution has magnitude and pointing errors.

The same simulator is used by the LQR baseline (phase 3) and the RL
environment (phases 4-5), so every controller faces identical physics,
noise and fuel accounting.

Units: everything inside the simulator is nondimensional; helpers convert
to km, m/s and days for reporting.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..constants import (ACCEL_UNIT_MPS2, AU_KM, MU_SUN_EARTH, TIME_UNIT_DAYS,
                         VELOCITY_UNIT_KMS)
from ..dynamics import grad_potential
from ..lagrange import lagrange_points

KM = 1.0 / AU_KM                                  # 1 km in length units
MPS = 1.0 / (VELOCITY_UNIT_KMS * 1000.0)          # 1 m/s in velocity units
MPS2 = 1.0 / ACCEL_UNIT_MPS2                      # 1 m/s^2 in acceleration units


@dataclass
class NoiseModel:
    """1-sigma navigation and thrust-execution errors.

    Defaults are illustrative assumptions for a deep-space mission, not
    values taken from a specific spacecraft.
    """
    pos_km: float = 1.0           # position knowledge error per axis
    vel_mps: float = 0.001        # velocity knowledge error per axis (1 mm/s)
    thrust_mag_frac: float = 0.02 # thrust magnitude error (2 %)
    thrust_angle_deg: float = 1.0 # thrust pointing error
    #: Unmodelled forces (solar radiation pressure changes, the Moon, planets),
    #: drawn per axis for every control interval and held over it.
    disturbance_mps2: float = 1e-9

    @classmethod
    def none(cls):
        return cls(0.0, 0.0, 0.0, 0.0, 0.0)

    def scaled(self, factor):
        return NoiseModel(self.pos_km * factor, self.vel_mps * factor,
                          self.thrust_mag_frac * factor, self.thrust_angle_deg * factor,
                          self.disturbance_mps2 * factor)


def planar_rhs(s, mu, u):
    """d/dt of the planar state [x, y, vx, vy] with control acceleration u."""
    x, y, vx, vy = s
    ox, oy, _ = grad_potential(x, y, 0.0, mu)
    return np.array([vx, vy, 2.0 * vy + ox + u[0], -2.0 * vx + oy + u[1]])


def _rk4_planar(x, y, vx, vy, mu, ux, uy, h, n):
    """n RK4 steps of the planar CR3BP with constant control, on plain floats.

    Same equations as `planar_rhs`, written without numpy so the inner loop
    is fast enough for reinforcement learning (millions of steps).
    """
    m1 = 1.0 - mu
    sq = math.sqrt

    def acc(x, y, vx, vy):
        dx1, dx2 = x + mu, x - m1
        y2 = y * y
        r1 = sq(dx1 * dx1 + y2)
        r2 = sq(dx2 * dx2 + y2)
        k1 = m1 / (r1 * r1 * r1)
        k2 = mu / (r2 * r2 * r2)
        ax = 2.0 * vy + x - k1 * dx1 - k2 * dx2 + ux
        ay = -2.0 * vx + y - k1 * y - k2 * y + uy
        return ax, ay

    h2, h6 = h / 2.0, h / 6.0
    for _ in range(n):
        a1x, a1y = acc(x, y, vx, vy)
        v2x, v2y = vx + h2 * a1x, vy + h2 * a1y
        a2x, a2y = acc(x + h2 * vx, y + h2 * vy, v2x, v2y)
        v3x, v3y = vx + h2 * a2x, vy + h2 * a2y
        a3x, a3y = acc(x + h2 * v2x, y + h2 * v2y, v3x, v3y)
        v4x, v4y = vx + h * a3x, vy + h * a3y
        a4x, a4y = acc(x + h * v3x, y + h * v3y, v4x, v4y)
        x += h6 * (vx + 2.0 * v2x + 2.0 * v3x + v4x)
        y += h6 * (vy + 2.0 * v2y + 2.0 * v3y + v4y)
        vx += h6 * (a1x + 2.0 * a2x + 2.0 * a3x + a4x)
        vy += h6 * (a1y + 2.0 * a2y + 2.0 * a3y + a4y)
    return x, y, vx, vy


class StationKeepingSim:
    """Planar spacecraft near L1 with periodic thrust commands.

    Parameters
    ----------
    mu : float
        Mass ratio.
    dt_days : float
        Length of one control interval.
    substep : float
        Maximum RK4 step inside an interval (nondimensional time).
    noise : NoiseModel
        Navigation and execution errors. ``NoiseModel.none()`` for an ideal run.
    r_fail_km : float
        The episode fails once the spacecraft is farther than this from L1.
    seed : int, optional
        Seed for the random number generator.
    """

    def __init__(self, mu=MU_SUN_EARTH, dt_days=1.0, substep=2e-3, noise=None,
                 r_fail_km=10_000.0, seed=None):
        self.mu = mu
        self.l1 = lagrange_points(mu)["L1"][:2].copy()
        self.dt = dt_days / TIME_UNIT_DAYS
        self.n_sub = max(1, math.ceil(self.dt / substep))
        self.h = self.dt / self.n_sub
        self.noise = NoiseModel() if noise is None else noise
        self.r_fail = r_fail_km * KM
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self, offset_km=(1000.0, 0.0), vel_mps=(0.0, 0.0)):
        """Place the spacecraft at L1 + offset with the given velocity.

        Returns the first (noisy) observation.
        """
        self.state = np.r_[self.l1 + np.asarray(offset_km) * KM, np.asarray(vel_mps) * MPS]
        self.t = 0.0
        self.dv = 0.0
        return self.observe()

    @property
    def rel_state(self):
        """True state relative to L1: [dx, dy, vx, vy] (nondimensional)."""
        return self.state - np.r_[self.l1, 0.0, 0.0]

    @property
    def distance_km(self):
        return float(np.hypot(*self.rel_state[:2]) / KM)

    @property
    def failed(self):
        return np.hypot(*self.rel_state[:2]) > self.r_fail

    def observe(self):
        """Navigation estimate of the relative state (true state + noise)."""
        n = self.noise
        err = np.r_[self.rng.normal(0.0, n.pos_km * KM, 2), self.rng.normal(0.0, n.vel_mps * MPS, 2)]
        return self.rel_state + err

    # ------------------------------------------------------------ dynamics
    def _execute(self, u_cmd):
        """Apply thrust magnitude and pointing errors to a command."""
        u = np.asarray(u_cmd, dtype=float)
        mag = np.hypot(*u)
        if mag == 0.0:
            return u
        n = self.noise
        mag *= max(0.0, 1.0 + self.rng.normal(0.0, n.thrust_mag_frac))
        ang = math.atan2(u[1], u[0]) + math.radians(self.rng.normal(0.0, n.thrust_angle_deg))
        return mag * np.array([math.cos(ang), math.sin(ang)])

    def step(self, u_cmd):
        """Hold the commanded acceleration for one control interval.

        Returns (observation, failed).
        """
        u = self._execute(u_cmd)
        d = self.rng.normal(0.0, self.noise.disturbance_mps2 * MPS2, 2)
        x, y, vx, vy = (float(v) for v in self.state)
        self.state = np.array(_rk4_planar(x, y, vx, vy, self.mu, float(u[0] + d[0]),
                                          float(u[1] + d[1]), self.h, self.n_sub))
        self.t += self.dt
        self.dv += float(np.hypot(*u)) * self.dt     # fuel actually spent
        return self.observe(), self.failed


def run_episode(sim, controller, days=365, offset_km=(1000.0, 0.0), vel_mps=(0.0, 0.0)):
    """Fly one episode and return a log plus summary metrics.

    ``controller(obs) -> u`` maps a noisy relative state to a commanded
    acceleration (nondimensional). Pass ``None`` for an uncontrolled run.
    """
    obs = sim.reset(offset_km, vel_mps)
    n = int(round(days * 1.0 / (sim.dt * TIME_UNIT_DAYS)))
    t_days, dist, dv, pos = [0.0], [sim.distance_km], [0.0], [sim.rel_state[:2] / KM]
    survived = True
    for _ in range(n):
        u = np.zeros(2) if controller is None else controller(obs)
        obs, failed = sim.step(u)
        t_days.append(sim.t * TIME_UNIT_DAYS)
        dist.append(sim.distance_km)
        dv.append(sim.dv / MPS)
        pos.append(sim.rel_state[:2] / KM)
        if failed:
            survived = False
            break
    dist = np.array(dist)
    return {
        "t_days": np.array(t_days),
        "dist_km": dist,
        "dv_mps": np.array(dv),               # cumulative delta-v
        "pos_km": np.array(pos),              # (N, 2) relative to L1
        "survived": survived,
        "days_survived": t_days[-1],
        "dv_total_mps": dv[-1],
        "mean_dist_km": float(dist.mean()),
        "final_dist_km": float(dist[-1]),
    }
