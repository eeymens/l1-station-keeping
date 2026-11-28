"""Phase 7: classical controllers that can cope with a constant disturbance.

The plain LQR (u = -K x) is designed for the CR3BP. A constant unmodelled
force d (e.g. solar radiation pressure, SRP) leaves it with a steady-state
offset: the spacecraft settles where gravity, thrust and d balance. That is not
a flaw of LQR as such, it is the textbook limitation of pure state feedback, so
comparing RL only against plain LQR under SRP would be unfair to the classical
side. Two standard fixes are implemented here.

LQIController - LQR with integral action
    Adds the running sum of the position error z[k+1] = z[k] + dt * r[k] to
    the state and designs one LQR for the augmented system. The integral grows
    until the mean position error is zero, so the controller ends up holding
    *exactly* at L1 with a constant thrust that cancels d.

BiasLQRController - LQR with disturbance estimation (offset-free MPC style)
    A steady-state Kalman filter estimates the state and the constant
    disturbance d together, from the same noisy navigation data. Given the
    estimate d_hat it computes the cheapest steady state (x_ss, u_ss) that the
    linear model can hold against d_hat, minimising x' Q x + u' R u with the
    same Q and R as the LQR, and regulates around it:
        u = u_ss - K (x_hat - x_ss).
    With a large fuel weight it parks near the point where gravity alone
    balances d (no thrust); with a small one it cancels d with thrust and stays
    at L1. This is how real missions treat SRP: it is part of the reference.

Both keep state between control steps; `run_episode` calls ``reset()`` at the
start of every episode.
"""

import numpy as np
from ..constants import TIME_UNIT_DAYS
from .lqr import LQRController, dlqr
from .riccati import kalman_dare
from .station_keeping import KM, MPS, MPS2, NoiseModel


class LQIController(LQRController):
    """LQR plus integral action on the position error.

    Parameters (in addition to `LQRController`)
    ----------
    t_int_days : float
        Bryson scale of the integral state: an integrated error of
        r_ref_km x t_int_days counts as much as r_ref_km of position error.
        Smaller = stronger integral action.
    """

    def __init__(self, t_int_days=60.0, **kwargs):
        super().__init__(**kwargs)
        n, m = self.Bd.shape
        C = np.eye(2, n)                           # position part of the state
        self.Aa = np.block([[self.Ad, np.zeros((n, 2))], [self.dt * C, np.eye(2)]])
        self.Ba = np.vstack([self.Bd, np.zeros((2, m))])
        r = kwargs.get("r_ref_km", 1000.0) * KM
        zref = r * t_int_days / TIME_UNIT_DAYS       # position x time, nondimensional
        self.Qa = np.block([[self.Q, np.zeros((n, 2))],
                            [np.zeros((2, n)), np.eye(2) / zref**2]])
        self.Ka, self.Pa = dlqr(self.Aa, self.Ba, self.Qa, self.R)
        self.reset()

    def reset(self):
        self.z = np.zeros(2)

    def __call__(self, obs):
        u = -self.Ka @ np.r_[obs, self.z]
        saturated = False
        if self.u_max is not None:
            mag = np.hypot(*u)
            if mag > self.u_max:
                u *= self.u_max / mag
                saturated = True
        if not saturated:                          # conditional integration (anti-windup)
            self.z = self.z + self.dt * obs[:2]
        return u


class BiasLQRController(LQRController):
    """LQR around the cheapest steady state, with a Kalman filter that
    estimates the state and a constant disturbance acceleration.

    Parameters (in addition to `LQRController`)
    ----------
    noise : NoiseModel
        Navigation / execution error levels the filter is tuned for
        (default: the simulator's default noise).
    bias_walk_mps2 : float
        1-sigma change of the disturbance per control interval (random walk).
        Sets how fast the estimate follows a slowly varying disturbance.
    riccati : {"legacy", "balanced"}
        How the filter's Riccati equation is solved (see `cr3bp.control.riccati`);
        "legacy" (default) reproduces every committed result.
    """

    def __init__(self, noise=None, bias_walk_mps2=2e-10, riccati="legacy", **kwargs):
        super().__init__(**kwargs)
        noise = NoiseModel() if noise is None else noise
        n, m = self.Bd.shape
        # ---- augmented model: [x; d],  x+ = Ad x + Bd (u + d),  d+ = d
        self.Fa = np.block([[self.Ad, self.Bd], [np.zeros((m, n)), np.eye(m)]])
        self.Ga = np.vstack([self.Bd, np.zeros((m, m))])
        H = np.hstack([np.eye(n), np.zeros((n, m))])
        self.H = H
        # process noise: unmodelled force + thrust errors act like an acceleration
        # held over the interval; the bias performs a slow random walk
        sig_acc = max(noise.disturbance_mps2, 1e-12) * MPS2
        Qx = self.Bd @ self.Bd.T * sig_acc**2
        Qd = np.eye(m) * (bias_walk_mps2 * MPS2) ** 2
        Qn = np.block([[Qx, np.zeros((n, m))], [np.zeros((m, n)), Qd]])
        Qn += np.eye(n + m) * 1e-30                 # keep the Riccati equation well posed
        Rn = np.diag([max(noise.pos_km, 1e-3) * KM] * 2 + [max(noise.vel_mps, 1e-6) * MPS] * 2) ** 2
        scale = np.r_[[KM] * 2, [1e-3 * MPS] * 2, [1e-9 * MPS2] * m]   # 1 km, 1 mm/s, 1e-9 m/s^2
        P, self.riccati_info = kalman_dare(self.Fa, H, Qn, Rn, scale=scale, method=riccati)
        self.bias_walk_mps2 = bias_walk_mps2
        S = H @ P @ H.T + Rn
        self.L = P @ H.T @ np.linalg.inv(S)         # Kalman gain (predicted -> filtered)
        # ---- steady state (x_ss, u_ss) as a linear function of d:
        #      min x'Qx + u'Ru  s.t.  (Ad - I) x + Bd u = -Bd d
        G = np.hstack([self.Ad - np.eye(n), self.Bd])
        Hq = np.block([[self.Q, np.zeros((n, m))], [np.zeros((m, n)), self.R]])
        KKT = np.block([[2 * Hq, G.T], [G, np.zeros((n, n))]])
        rhs = np.vstack([np.zeros((n + m, m)), -self.Bd])
        sol = np.linalg.solve(KKT, rhs)
        self.Mx, self.Mu = sol[:n], sol[n:n + m]    # x_ss = Mx d, u_ss = Mu d
        self.reset()

    def reset(self):
        self.xa = None                               # filtered [x; d]
        self.u_prev = np.zeros(2)

    @property
    def bias_estimate_mps2(self):
        return None if self.xa is None else self.xa[4:] / MPS2

    def __call__(self, obs):
        if self.xa is None:
            self.xa = np.r_[obs, 0.0, 0.0]
        else:
            pred = self.Fa @ self.xa + self.Ga @ self.u_prev
            self.xa = pred + self.L @ (obs - self.H @ pred)
        x_hat, d_hat = self.xa[:4], self.xa[4:]
        x_ss, u_ss = self.Mx @ d_hat, self.Mu @ d_hat
        u = u_ss - self.K @ (x_hat - x_ss)
        if self.u_max is not None:
            mag = np.hypot(*u)
            if mag > self.u_max:
                u *= self.u_max / mag
        self.u_prev = u
        return u
