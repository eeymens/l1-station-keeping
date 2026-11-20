"""Phase 7: perturbations that the controllers' internal model does not know.

Every controller in this project (LQR, DQN, PPO) is designed for, or trained
in, the planar *circular* restricted three-body problem (CR3BP). Real Sun-Earth
dynamics differ from it. This module adds two of the largest differences to the
simulator only, so that the controllers meet dynamics they were not built for:

1. Earth's elliptic orbit (elliptic restricted three-body problem, ER3BP),
   eccentricity e = 0.0167.
2. Solar radiation pressure (SRP): sunlight pushes the spacecraft away from
   the Sun with an acceleration that falls off as 1 / (distance to the Sun)^2.

Formulation: pulsating rotating coordinates
-------------------------------------------
In the ER3BP the Sun-Earth distance r changes over the year,

    r(f) = a (1 - e^2) / (1 + e cos f),        f = Earth's true anomaly.

The standard trick (Szebehely 1967, "Theory of Orbits", ch. 10) is to rotate
with the Sun-Earth line *and* scale lengths by r(f), and to use f instead of
time as the independent variable. In these "pulsating" coordinates the Sun and
Earth stay at (-mu, 0) and (1 - mu, 0), the five Lagrange points stay at their
CR3BP positions, and the planar equations of motion become (' = d/df)

    x'' - 2 y' = w(f) dOmega/dx + g(f) ux
    y'' + 2 x' = w(f) dOmega/dy + g(f) uy

    w(f) = 1 / (1 + e cos f)
    g(f) = (1 - e^2)^2 / (1 + e cos f)^3

Omega is the same CR3BP potential as in `dynamics.py`. With e = 0 we get
w = g = 1, f = t, and the CR3BP exactly. We chose this formulation because
(i) its e = 0 limit *is* the existing simulator, which gives an exact
regression test, (ii) "L1" keeps a fixed position, so "distance from L1" means
distance from the instantaneous (moving) Sun-Earth L1, and (iii) it is the
standard form in libration-point literature.

Where g(f) comes from: a physical acceleration a (in nondimensional units: AU,
mean motion = 1) becomes a / (r * df/dt^2) in pulsating units. Kepler's second
law gives df/dt = sqrt(1 - e^2) / r^2, so r * (df/dt)^2 = (1 - e^2) / r^3 and
the factor is r^3 / (1 - e^2) = g(f). The same factor turns the Sun's and the
Earth's gravity into w(f) dOmega/dx, which is how the equations above arise.
The test `tests/test_perturbations.py::test_matches_inertial_integration`
checks all of this against a brute-force integration in an inertial frame.

SRP in pulsating coordinates: with a0 the SRP acceleration at 1 AU, the
physical acceleration is a0 / (r r1)^2 pointing away from the Sun (r1 = Sun
distance in pulsating units). Multiplying by g(f) gives w(f) a0 / r1^2, i.e.
SRP simply lowers the Sun's effective gravity from (1 - mu) to (1 - mu - a0).

Timing: controllers still act once per *day* of physical time. Each interval
[t, t + dt] is mapped to [f(t), f(t + dt)] with Kepler's equation and
integrated in f with RK4.

Observations and metrics: the navigation estimate a controller sees is the
pulsating state relative to L1 (positions in AU-scaled pulsating units, and
d/df velocities), which is what a controller designed in the CR3BP would use.
Reported distances are *physical* km, i.e. pulsating distance x r(f) x 1 AU.
Delta-v is physical: integral of |thrust acceleration| dt.
"""

import math
from dataclasses import dataclass

from .constants import ACCEL_UNIT_MPS2

#: Eccentricity of Earth's orbit (J2000 mean value).
E_EARTH = 0.0167086

#: Solar radiation pressure at 1 AU on a perfect absorber, N/m^2
#: (solar flux 1367 W/m^2 divided by the speed of light; Montenbruck & Gill,
#: "Satellite Orbits", 2000, sec. 3.4).
P_SUN_NPM2 = 4.56e-6

#: Reflectivity coefficient (1 = absorber, 2 = mirror). Assumed value.
CR_DEFAULT = 1.3

#: Area-to-mass ratio, m^2/kg. Assumed value for a ~1 t spacecraft with ~10 m^2
#: of sunlit area (solar panels + body); illustrative, not a specific mission.
AREA_TO_MASS_DEFAULT = 0.01

#: Resulting SRP acceleration at 1 AU, m/s^2 (~5.9e-8, i.e. ~6 % of the
#: 1e-6 m/s^2 thrust limit and ~1.9 m/s of delta-v per year if fully cancelled).
SRP_DEFAULT_MPS2 = P_SUN_NPM2 * CR_DEFAULT * AREA_TO_MASS_DEFAULT


@dataclass
class Perturbation:
    """Unmodelled physics added to the simulator.

    e         : eccentricity of Earth's orbit (0 = circular CR3BP)
    srp_mps2  : SRP acceleration at 1 AU, m/s^2 (0 = no SRP)
    f0        : Earth's true anomaly at t = 0 (rad), i.e. where in the year we start
    """
    e: float = 0.0
    srp_mps2: float = 0.0
    f0: float = 0.0

    @classmethod
    def from_k(cls, k, f0=0.0, srp_scale=1.0):
        """Mismatch severity k: k = 0 is the CR3BP, k = 1 the realistic values
        (e = 0.0167, default SRP), k = 4 four times both."""
        return cls(e=k * E_EARTH, srp_mps2=k * srp_scale * SRP_DEFAULT_MPS2, f0=f0)

    @property
    def is_null(self):
        return self.e == 0.0 and self.srp_mps2 == 0.0

    @property
    def a0(self):
        """SRP acceleration at 1 AU in nondimensional units."""
        return self.srp_mps2 / ACCEL_UNIT_MPS2


# ------------------------------------------------------------------ Kepler
def mean_from_true(f, e):
    """Mean anomaly M from true anomaly f."""
    E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(f / 2.0),
                         math.sqrt(1.0 + e) * math.cos(f / 2.0))
    return E - e * math.sin(E)


def true_from_mean(M, e):
    """True anomaly f from mean anomaly M (Newton on Kepler's equation).

    The returned angle is continuous in M: f - M stays small, so f grows
    with M instead of wrapping at 2 pi.
    """
    E = M
    for _ in range(30):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-15:
            break
    f = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                         math.sqrt(1.0 - e) * math.cos(E / 2.0))
    # atan2 wraps to (-pi, pi]; put f on the same branch as M
    f += 2.0 * math.pi * round((M - f) / (2.0 * math.pi))
    return f


def radius(f, e):
    """Sun-Earth distance in AU at true anomaly f (semi-major axis = 1 AU)."""
    return (1.0 - e * e) / (1.0 + e * math.cos(f))


# ------------------------------------------------------------------ dynamics
def er3bp_accel(f, x, y, vx, vy, mu, e, a0, ux, uy):
    """Planar ER3BP + SRP accelerations (d2x/df2, d2y/df2) in pulsating coordinates.

    (ux, uy) is the *physical* control/disturbance acceleration in
    nondimensional units; it is scaled by g(f) here.
    """
    c = 1.0 + e * math.cos(f)
    w = 1.0 / c
    g = (1.0 - e * e) ** 2 * w * w * w
    m1 = 1.0 - mu
    dx1, dx2 = x + mu, x - m1
    y2 = y * y
    r1 = math.sqrt(dx1 * dx1 + y2)
    r2 = math.sqrt(dx2 * dx2 + y2)
    k1 = (m1 - a0) / (r1 * r1 * r1)      # Sun gravity minus SRP
    k2 = mu / (r2 * r2 * r2)
    ax = 2.0 * vy + w * (x - k1 * dx1 - k2 * dx2) + g * ux
    ay = -2.0 * vx + w * (y - k1 * y - k2 * y) + g * uy
    return ax, ay


def rk4_er3bp(x, y, vx, vy, mu, e, a0, ux, uy, f_start, f_end, n):
    """n RK4 steps in true anomaly from f_start to f_end, constant (ux, uy)."""
    h = (f_end - f_start) / n
    h2, h6 = h / 2.0, h / 6.0
    f = f_start
    for _ in range(n):
        a1x, a1y = er3bp_accel(f, x, y, vx, vy, mu, e, a0, ux, uy)
        v2x, v2y = vx + h2 * a1x, vy + h2 * a1y
        a2x, a2y = er3bp_accel(f + h2, x + h2 * vx, y + h2 * vy, v2x, v2y, mu, e, a0, ux, uy)
        v3x, v3y = vx + h2 * a2x, vy + h2 * a2y
        a3x, a3y = er3bp_accel(f + h2, x + h2 * v2x, y + h2 * v2y, v3x, v3y, mu, e, a0, ux, uy)
        v4x, v4y = vx + h * a3x, vy + h * a3y
        a4x, a4y = er3bp_accel(f + h, x + h * v3x, y + h * v3y, v4x, v4y, mu, e, a0, ux, uy)
        x += h6 * (vx + 2.0 * v2x + 2.0 * v3x + v4x)
        y += h6 * (vy + 2.0 * v2y + 2.0 * v3y + v4y)
        vx += h6 * (a1x + 2.0 * a2x + 2.0 * a3x + a4x)
        vy += h6 * (a1y + 2.0 * a2y + 2.0 * a3y + a4y)
        f += h
    return x, y, vx, vy
