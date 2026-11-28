"""Steady-state Kalman filter Riccati equation for the disturbance-estimating LQRs.

Bias-LQR and Harmonic-LQR (lqi.py, harmonic.py) estimate the state and the
disturbance with a steady-state Kalman filter, i.e. the predicted covariance P
of

    x+ = F x + w,  w ~ N(0, Qn),     y = H x + v,  v ~ N(0, Rn)

solves the discrete algebraic Riccati equation (DARE)

    P = F P F' - F P H' (H P H' + Rn)^-1 H P F' + Qn.

In nondimensional units this problem is very badly scaled: positions are
~1e-8, velocities ~1e-5, disturbance states ~1e-5 and the process noise of the
disturbance states (``bias_walk_mps2``) can be 1e-36 on the diagonal. The
disturbance model of Harmonic-LQR also has many eigenvalues on the unit circle
(the oscillators). With a small ``bias_walk_mps2`` the filter's poles for those
modes approach the unit circle, and the Schur method of
`scipy.linalg.solve_discrete_are` can no longer separate the stable from the
unstable eigenvalues of the symplectic pencil: it raises ``ValueError``
("Reordering ... failed") for every Harmonic-LQR design with
``bias_walk_mps2 <= 1e-10``.

Two methods are offered:

``"legacy"``
    `solve_discrete_are` on the raw problem, exactly as phases 7-9 did. Every
    committed result (and the base controller of the RES-PPO / RES-PPO-M
    models) uses it, so it stays the default and reproduces them bit for bit.
    Where it fails it now raises `RiccatiError` instead of a bare ValueError.
    Note: even where it returns, its DARE residual is not small (see
    `dare_residual`).
``"balanced"``
    The same equation solved after a diagonal change of variables
    x = T x_s that makes every state O(1) (T = ``scale``); P = T P_s T'. This
    is mathematically identical and numerically well posed: the residual is
    ~1e-10 relative and it solves every ``bias_walk_mps2`` down to 1e-11.
    The result is checked (finite, symmetric positive semidefinite, small
    residual, stable filter); a design that still fails raises `RiccatiError`,
    so a sweep can log and skip it explicitly.
"""

import numpy as np
from scipy.linalg import solve_discrete_are

METHODS = ("legacy", "balanced")
#: largest accepted relative DARE residual of the balanced solve
RESIDUAL_TOL = 1e-6


class RiccatiError(RuntimeError):
    """The filter Riccati equation of a controller design could not be solved."""


def dare_residual(F, H, Qn, Rn, P, scale=None):
    """Max-abs DARE residual relative to max-abs P, both in the scaled coordinates."""
    G = F @ P @ H.T
    res = F @ P @ F.T - G @ np.linalg.solve(H @ P @ H.T + Rn, G.T) + Qn - P
    if scale is not None:
        Ti = np.diag(1.0 / np.asarray(scale, dtype=float))
        res, P = Ti @ res @ Ti, Ti @ P @ Ti
    return float(np.abs(res).max() / np.abs(P).max())


def kalman_dare(F, H, Qn, Rn, scale=None, method="legacy"):
    """Predicted steady-state covariance P (see module docstring).

    Returns (P, info) with info = {"method", "residual"}; ``residual`` is
    `dare_residual` in the ``scale`` coordinates (NaN when no scale is given).
    Raises `RiccatiError` when no acceptable solution is found.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "legacy":
        try:
            P = solve_discrete_are(F.T, H.T, Qn, Rn)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise RiccatiError(f"legacy DARE solve failed ({exc}); "
                               "use riccati='balanced'") from exc
    else:
        if scale is None:
            raise ValueError("method='balanced' needs the state scale")
        s = np.asarray(scale, dtype=float)
        T, Ti = np.diag(s), np.diag(1.0 / s)
        try:
            Ps = solve_discrete_are((Ti @ F @ T).T, (H @ T).T, Ti @ Qn @ Ti, Rn)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise RiccatiError(f"balanced DARE solve failed ({exc})") from exc
        if not np.all(np.isfinite(Ps)):
            raise RiccatiError("balanced DARE solution is not finite")
        Ps = 0.5 * (Ps + Ps.T)
        if np.linalg.eigvalsh(Ps).min() < -1e-9 * np.abs(Ps).max():
            raise RiccatiError("balanced DARE solution is not positive semidefinite")
        P = T @ Ps @ T
    residual = float("nan") if scale is None else dare_residual(F, H, Qn, Rn, P, scale)
    if method == "balanced" and not residual <= RESIDUAL_TOL:
        raise RiccatiError(f"balanced DARE residual {residual:.2e} > {RESIDUAL_TOL:g}")
    L = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rn)
    rad = np.abs(np.linalg.eigvals((np.eye(len(F)) - L @ H) @ F)).max()
    if not rad < 1.0:
        raise RiccatiError(f"Kalman filter is not stable (spectral radius {rad:.6f})")
    return P, {"method": method, "residual": residual, "filter_radius": float(rad)}
