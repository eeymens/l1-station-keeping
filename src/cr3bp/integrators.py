"""Numerical integrators.

* ``rk4``: classic fixed-step 4th-order Runge-Kutta. Simple and fast,
  used inside the RL environment where every step has a fixed length.
* ``dop853``: adaptive 8th-order Dormand-Prince (scipy). Used as the
  high-accuracy reference solution.

Note: velocity Verlet is not used here. The Coriolis force depends on
velocity, so the standard Verlet scheme loses its symplectic property.
"""

import numpy as np
from scipy.integrate import solve_ivp


def rk4_step(f, t, s, h):
    """Advance one RK4 step of size h for ds/dt = f(t, s)."""
    k1 = f(t, s)
    k2 = f(t + h / 2, s + h / 2 * k1)
    k3 = f(t + h / 2, s + h / 2 * k2)
    k4 = f(t + h, s + h * k3)
    return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4(f, s0, t_end, h, t0=0.0):
    """Integrate with fixed-step RK4 from t0 to t_end.

    The last step is shortened so the run ends exactly at t_end.
    Returns (t, states) with states of shape (N, len(s0)).
    """
    n_full = int(np.floor((t_end - t0) / h + 1e-12))
    times = [t0 + i * h for i in range(n_full + 1)]
    if times[-1] < t_end - 1e-12:
        times.append(t_end)
    t = np.array(times)
    states = np.empty((len(t), len(s0)))
    states[0] = s0
    for i in range(1, len(t)):
        states[i] = rk4_step(f, t[i - 1], states[i - 1], t[i] - t[i - 1])
    return t, states


def dop853(f, s0, t_end, t0=0.0, rtol=1e-12, atol=1e-12, t_eval=None, **kwargs):
    """Integrate with scipy's adaptive DOP853. Returns (t, states)."""
    sol = solve_ivp(f, (t0, t_end), s0, method="DOP853",
                    rtol=rtol, atol=atol, t_eval=t_eval, **kwargs)
    if not sol.success:
        raise RuntimeError(f"DOP853 failed: {sol.message}")
    return sol.t, sol.y.T
