"""Discrete-time LQR station-keeping controller.

Linear model about L1 (planar): x' = A x + B u, with x = [dx, dy, vx, vy]
and u the thrust acceleration. Thrust is held constant over each control
interval, so the model is discretised exactly with a zero-order hold:

    x[k+1] = Ad x[k] + Bd u[k]

The gain minimises  sum_k  x' Q x + u' R u  and comes from the discrete
algebraic Riccati equation. Q and R use Bryson's rule: each state or input
is divided by the largest value we are willing to tolerate.
"""

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from ..constants import MU_SUN_EARTH, TIME_UNIT_DAYS
from ..lagrange import lagrange_points, linearize
from .station_keeping import KM, MPS, MPS2

#: Input matrix: thrust enters the velocity equations.
B_PLANAR = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def discretize(A, B, dt):
    """Exact zero-order-hold discretisation via one matrix exponential."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def dlqr(Ad, Bd, Q, R):
    """Discrete LQR gain K (u = -K x) and the Riccati solution P."""
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


class LQRController:
    """u = -K x, optionally saturated to a maximum acceleration.

    Parameters
    ----------
    rho : float
        Fuel weight. Larger rho -> less thrust, looser tracking.
    r_ref_km, v_ref_mps, u_ref_mps2 : float
        Bryson reference scales for position, velocity and thrust.
    u_max_mps2 : float, optional
        Thrust acceleration limit (norm). ``None`` = unlimited.
    """

    def __init__(self, mu=MU_SUN_EARTH, dt_days=1.0, rho=1.0, r_ref_km=1000.0,
                 v_ref_mps=1.0, u_ref_mps2=1e-6, u_max_mps2=None):
        l1 = lagrange_points(mu)["L1"]
        self.A = linearize(l1, mu)
        self.dt = dt_days / TIME_UNIT_DAYS
        self.Ad, self.Bd = discretize(self.A, B_PLANAR, self.dt)
        r, v, u = r_ref_km * KM, v_ref_mps * MPS, u_ref_mps2 * MPS2
        self.Q = np.diag([1 / r**2, 1 / r**2, 1 / v**2, 1 / v**2])
        self.R = rho * np.eye(2) / u**2
        self.K, self.P = dlqr(self.Ad, self.Bd, self.Q, self.R)
        self.u_max = None if u_max_mps2 is None else u_max_mps2 * MPS2

    @property
    def closed_loop_eigenvalues(self):
        return np.linalg.eigvals(self.Ad - self.Bd @ self.K)

    def __call__(self, obs):
        u = -self.K @ obs
        if self.u_max is not None:
            mag = np.hypot(*u)
            if mag > self.u_max:
                u *= self.u_max / mag
        return u
