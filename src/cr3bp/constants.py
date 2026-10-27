"""Physical constants and unit conversions for the Sun-Earth CR3BP.

The circular restricted three-body problem (CR3BP) is solved in
nondimensional units:

* length: the Sun-Earth distance (1 AU)
* mass:   the total mass of Sun + Earth
* time:   chosen so that the primaries' angular velocity is 1,
          i.e. one orbital period of the Earth is 2*pi time units.
"""

import math

# Masses in kg (IAU nominal values)
M_SUN_KG = 1.98847e30
M_EARTH_KG = 5.9722e24

#: Mass ratio mu = m2 / (m1 + m2) for the Sun-Earth system (~3.0035e-6).
MU_SUN_EARTH = M_EARTH_KG / (M_SUN_KG + M_EARTH_KG)

AU_KM = 149_597_870.7          # length unit, km
SIDEREAL_YEAR_DAYS = 365.256363

#: One nondimensional time unit, in days (~58.13 days).
TIME_UNIT_DAYS = SIDEREAL_YEAR_DAYS / (2.0 * math.pi)

#: One nondimensional velocity unit, in km/s (~29.8 km/s).
VELOCITY_UNIT_KMS = AU_KM / (TIME_UNIT_DAYS * 86_400.0)


def to_km(length):
    """Nondimensional length -> km."""
    return length * AU_KM


def to_days(time):
    """Nondimensional time -> days."""
    return time * TIME_UNIT_DAYS


def to_mps(velocity):
    """Nondimensional velocity (or delta-v) -> m/s."""
    return velocity * VELOCITY_UNIT_KMS * 1000.0
