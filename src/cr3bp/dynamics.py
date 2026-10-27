"""Equations of motion of the circular restricted three-body problem.

State vector: s = [x, y, z, vx, vy, vz] in the rotating (synodic) frame.
The larger primary (Sun) sits at (-mu, 0, 0), the smaller (Earth) at (1 - mu, 0, 0).

    x'' - 2 y' = dOmega/dx + ux
    y'' + 2 x' = dOmega/dy + uy
    z''        = dOmega/dz + uz

    Omega = (x^2 + y^2) / 2 + (1 - mu) / r1 + mu / r2

where u is an optional control acceleration (thrust / mass).
"""

import numpy as np


def distances(x, y, z, mu):
    """Distances r1 (to the Sun) and r2 (to the Earth)."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    return r1, r2


def effective_potential(x, y, z, mu):
    """Omega(x, y, z): gravity of both primaries plus the centrifugal term."""
    r1, r2 = distances(x, y, z, mu)
    return 0.5 * (x ** 2 + y ** 2) + (1.0 - mu) / r1 + mu / r2


def grad_potential(x, y, z, mu):
    """Gradient (dOmega/dx, dOmega/dy, dOmega/dz)."""
    r1, r2 = distances(x, y, z, mu)
    k1 = (1.0 - mu) / r1 ** 3
    k2 = mu / r2 ** 3
    ox = x - k1 * (x + mu) - k2 * (x - 1.0 + mu)
    oy = y - k1 * y - k2 * y
    oz = -k1 * z - k2 * z
    return ox, oy, oz


def rhs(t, s, mu, u=None):
    """Time derivative of the state, ds/dt.

    Parameters
    ----------
    t : float
        Time (unused; the CR3BP is autonomous, kept for solver signatures).
    s : array_like, shape (6,)
        State [x, y, z, vx, vy, vz].
    mu : float
        Mass ratio.
    u : array_like, shape (3,), optional
        Control acceleration in the rotating frame.
    """
    x, y, z, vx, vy, vz = s
    ox, oy, oz = grad_potential(x, y, z, mu)
    ax = 2.0 * vy + ox
    ay = -2.0 * vx + oy
    az = oz
    if u is not None:
        ax, ay, az = ax + u[0], ay + u[1], az + u[2]
    return np.array([vx, vy, vz, ax, ay, az])


def jacobi_constant(s, mu):
    """Jacobi constant C = 2*Omega - v^2. Conserved when u = 0.

    Accepts a single state of shape (6,) or a trajectory of shape (N, 6).
    """
    s = np.asarray(s, dtype=float)
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    v2 = s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2
    return 2.0 * effective_potential(x, y, z, mu) - v2
