"""Circular restricted three-body problem (CR3BP) toolkit for the Sun-Earth system."""

from .constants import (ACCEL_UNIT_MPS2, AU_KM, MU_SUN_EARTH, TIME_UNIT_DAYS, VELOCITY_UNIT_KMS,
                        to_days, to_km, to_mps)
from .dynamics import (distances, effective_potential, grad_potential,
                       jacobi_constant, rhs)
from .integrators import dop853, rk4, rk4_step
from .lagrange import (MU_ROUTH, lagrange_points, linearize, potential_hessian,
                       stability)

__version__ = "0.1.0"
