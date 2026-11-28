"""Phase 9: a classical controller for a periodic disturbance (the Moon).

Bias-LQR (lqi.py) models the unknown force as a constant d and estimates it
with a Kalman filter. The Moon's pull near L1 is not constant: in the rotating
frame it repeats every synodic month (29.53 days), mostly at twice that
frequency (see `cr3bp.perturbations`). A constant-bias filter keeps chasing
it. The textbook fix is the internal model principle: put a model of the
disturbance's known frequencies into the estimator. The frequencies are known
in advance (the Moon's orbit is), the amplitudes and phases are not.

HarmonicLQRController
    Disturbance model, per axis: a constant d0 plus one oscillator per harmonic
    j of the synodic rate w_s,
        xi_j+ = Rot(j w_s dt) xi_j,     d = d0 + sum_j (first component of xi_j).
    A steady-state Kalman filter estimates the spacecraft state, d0 and the
    oscillators together from the noisy navigation data.

    Control: the cheapest steady response to each disturbance part, in the
    same Q, R metric as the LQR. For d0 this is Bias-LQR's (x_ss, u_ss). For an
    oscillator with complex amplitude Z (d_k = Re(Z z^k), z = e^{i j w_s dt}),
    the periodic steady state x_k = Re(X z^k), u_k = Re(U z^k) of
    x+ = Ad x + Bd (u + d) has X = T (U + Z), T = (z I - Ad)^-1 Bd, and the U
    that minimises the mean of x'Qx + u'Ru is
        U = -(T^H Q T + R)^-1 T^H Q T Z.
    With a large fuel weight the controller lets the spacecraft ride the
    oscillation; with a small one it cancels it with thrust. The feedback
    u = u_ss - K (x_hat - x_ss) then regulates around the sum.

The class reuses LQRController's discretisation, K, Q and R.
"""

import numpy as np
from ..perturbations import MOON_RATE
from .lqr import LQRController
from .riccati import kalman_dare
from .station_keeping import KM, MPS, MPS2, NoiseModel


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


class HarmonicLQRController(LQRController):
    """LQR around the cheapest (periodic) steady state, with a Kalman filter for
    the state, a constant force and oscillating forces at known frequencies.

    Parameters (in addition to `LQRController`)
    ----------
    harmonics : tuple of int
        Multiples of the synodic rate to model (default 1, 2, 3).
    noise : NoiseModel
        Error levels the filter is tuned for (default: the simulator's).
    bias_walk_mps2 : float
        1-sigma change per control interval of the constant force and of each
        oscillator's components (lets the estimate adapt).
    riccati : {"legacy", "balanced"}
        How the filter's Riccati equation is solved (see `cr3bp.control.riccati`).
        "legacy" (default) reproduces every committed result but cannot solve
        bias_walk_mps2 <= 1e-10; "balanced" solves the same equation in scaled
        coordinates, accurately, for any bias_walk_mps2 down to 1e-11. A design
        that cannot be solved raises `RiccatiError`.
    """

    def __init__(self, harmonics=(1, 2, 3), noise=None, bias_walk_mps2=2e-10,
                 riccati="legacy", **kwargs):
        super().__init__(**kwargs)
        noise = NoiseModel() if noise is None else noise
        n, m = self.Bd.shape
        self.harmonics = tuple(harmonics)
        h = len(self.harmonics)
        nd = m + 4 * h                                   # d0 (2) + per harmonic 2 axes x (p, q)
        # disturbance dynamics and output map d = Cd @ dist
        Fd = np.eye(nd)
        Cd = np.zeros((m, nd))
        Cd[:, :m] = np.eye(m)
        for j, k in enumerate(self.harmonics):
            R = _rot(k * MOON_RATE * self.dt)
            for ax in range(m):
                i = m + 4 * j + 2 * ax                   # [p, q] of axis ax, harmonic k
                Fd[i:i + 2, i:i + 2] = R
                Cd[ax, i] = 1.0
        self.Cd, self.m, self.nd = Cd, m, nd
        self.Fa = np.block([[self.Ad, self.Bd @ Cd], [np.zeros((nd, n)), Fd]])
        self.Ga = np.vstack([self.Bd, np.zeros((nd, m))])
        H = np.hstack([np.eye(n), np.zeros((n, nd))])
        self.H = H
        sig_acc = max(noise.disturbance_mps2, 1e-12) * MPS2
        Qn = np.zeros((n + nd, n + nd))
        Qn[:n, :n] = self.Bd @ self.Bd.T * sig_acc**2
        Qn[n:, n:] = np.eye(nd) * (bias_walk_mps2 * MPS2) ** 2
        Qn += np.eye(n + nd) * 1e-30
        Rn = np.diag([max(noise.pos_km, 1e-3) * KM] * 2
                     + [max(noise.vel_mps, 1e-6) * MPS] * 2) ** 2
        # state scale for the balanced solve: 1 km, 1 mm/s, 1e-9 m/s^2
        scale = np.r_[[KM] * 2, [1e-3 * MPS] * 2, [1e-9 * MPS2] * nd]
        P, self.riccati_info = kalman_dare(self.Fa, H, Qn, Rn, scale=scale, method=riccati)
        self.bias_walk_mps2 = bias_walk_mps2
        self.L = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rn)
        # ---- constant part: as Bias-LQR
        G = np.hstack([self.Ad - np.eye(n), self.Bd])
        Hq = np.block([[self.Q, np.zeros((n, m))], [np.zeros((m, n)), self.R]])
        KKT = np.block([[2 * Hq, G.T], [G, np.zeros((n, n))]])
        sol = np.linalg.solve(KKT, np.vstack([np.zeros((n + m, m)), -self.Bd]))
        self.Mx, self.Mu = sol[:n], sol[n:n + m]
        # ---- periodic parts: complex gains X = Gx Z, U = Gu Z
        self.Gx, self.Gu = [], []
        for k in self.harmonics:
            z = np.exp(1j * k * MOON_RATE * self.dt)
            T = np.linalg.solve(z * np.eye(n) - self.Ad, self.Bd)
            TQT = T.conj().T @ self.Q @ T
            Gu = -np.linalg.solve(TQT + self.R, TQT)
            self.Gu.append(Gu)
            self.Gx.append(T @ (np.eye(m) + Gu))
        self.reset()

    def reset(self):
        self.xa = None
        self.u_prev = np.zeros(2)

    def steady_state(self, dist):
        """(x_ss, u_ss) for a disturbance-state estimate `dist`."""
        m = self.m
        x_ss, u_ss = self.Mx @ dist[:m], self.Mu @ dist[:m]
        for j in range(len(self.harmonics)):
            i = m + 4 * j
            Z = dist[i:i + 4:2] + 1j * dist[i + 1:i + 4:2]     # (p + i q) per axis
            x_ss = x_ss + np.real(self.Gx[j] @ Z)
            u_ss = u_ss + np.real(self.Gu[j] @ Z)
        return x_ss, u_ss

    @property
    def disturbance_estimate_mps2(self):
        return None if self.xa is None else self.Cd @ self.xa[4:] / MPS2

    def __call__(self, obs):
        if self.xa is None:
            self.xa = np.r_[obs, np.zeros(self.nd)]
        else:
            pred = self.Fa @ self.xa + self.Ga @ self.u_prev
            self.xa = pred + self.L @ (obs - self.H @ pred)
        x_hat, dist = self.xa[:4], self.xa[4:]
        x_ss, u_ss = self.steady_state(dist)
        u = u_ss - self.K @ (x_hat - x_ss)
        if self.u_max is not None:
            mag = np.hypot(*u)
            if mag > self.u_max:
                u *= self.u_max / mag
        self.u_prev = u
        return u
