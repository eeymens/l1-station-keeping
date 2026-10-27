"""Lagrange (libration) points and their linear stability.

L1, L2, L3 lie on the x axis and are roots of dOmega/dx = 0 (found with
Brent's method). L4 and L5 form equilateral triangles with the primaries.

Linearising the equations of motion about a Lagrange point gives
    d(delta s)/dt = A delta s.
A positive real eigenvalue lambda of A means small deviations grow like
exp(lambda * t): the point is unstable.
"""

import math

import numpy as np
from scipy.optimize import brentq

from .constants import TIME_UNIT_DAYS
from .dynamics import grad_potential

#: Routh's critical mass ratio. L4/L5 are linearly stable for mu < MU_ROUTH.
MU_ROUTH = 0.5 * (1.0 - math.sqrt(23.0 / 27.0))


def _dodx_on_axis(x, mu):
    return grad_potential(x, 0.0, 0.0, mu)[0]


def lagrange_points(mu):
    """Return {'L1': array([x, y, z]), ..., 'L5': ...}."""
    eps = 1e-12
    f = lambda x: _dodx_on_axis(x, mu)
    x_l1 = brentq(f, -mu + eps, 1.0 - mu - eps, xtol=1e-15)      # between Sun and Earth
    x_l2 = brentq(f, 1.0 - mu + eps, 2.0, xtol=1e-15)            # beyond Earth
    x_l3 = brentq(f, -2.0, -mu - eps, xtol=1e-15)                # beyond the Sun
    h = math.sqrt(3.0) / 2.0
    return {
        "L1": np.array([x_l1, 0.0, 0.0]),
        "L2": np.array([x_l2, 0.0, 0.0]),
        "L3": np.array([x_l3, 0.0, 0.0]),
        "L4": np.array([0.5 - mu, h, 0.0]),
        "L5": np.array([0.5 - mu, -h, 0.0]),
    }


def potential_hessian(x, y, z, mu):
    """3x3 Hessian of Omega at (x, y, z), computed analytically."""
    hess = np.diag([1.0, 1.0, 0.0])  # centrifugal term (x^2 + y^2) / 2
    for k, cx in ((1.0 - mu, -mu), (mu, 1.0 - mu)):
        d = np.array([x - cx, y, z])
        r = np.linalg.norm(d)
        hess += k * (3.0 * np.outer(d, d) / r ** 5 - np.eye(3) / r ** 3)
    return hess


def linearize(point, mu, spatial=False):
    """State matrix A of the linearised dynamics about `point`.

    Planar (spatial=False): state [x, y, vx, vy], A is 4x4.
    Spatial (spatial=True): state [x, y, z, vx, vy, vz], A is 6x6.
    """
    hess = potential_hessian(*point, mu)
    if spatial:
        coriolis = np.array([[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        return np.block([[np.zeros((3, 3)), np.eye(3)], [hess, coriolis]])
    coriolis = np.array([[0.0, 2.0], [-2.0, 0.0]])
    return np.block([[np.zeros((2, 2)), np.eye(2)], [hess[:2, :2], coriolis]])


def stability(point, mu, spatial=False):
    """Eigenvalues of A and the growth rate of the most unstable mode.

    Returns a dict with
    * ``eigenvalues``: complex array
    * ``growth_rate``: largest real part (0 for a linearly stable point)
    * ``efold_days``: time for a deviation to grow by a factor e (inf if stable)
    """
    eig = np.linalg.eigvals(linearize(point, mu, spatial=spatial))
    growth = float(np.max(eig.real))
    if growth < 1e-9:
        growth = 0.0
    efold = math.inf if growth == 0.0 else TIME_UNIT_DAYS / growth
    return {"eigenvalues": eig, "growth_rate": growth, "efold_days": efold}
