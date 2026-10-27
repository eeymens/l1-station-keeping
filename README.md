# L1 Station-Keeping: LQR vs Reinforcement Learning

Can a reinforcement learning agent keep a spacecraft at the unstable Sun–Earth L1 point using less fuel (Δv) than a classical LQR controller?

The Sun–Earth L1 point sits about 1.49 million km sunward of Earth. It is an equilibrium of the circular restricted three-body problem (CR3BP), but an unstable one: a spacecraft left there with a 150 m error drifts away by a factor of *e* every 23 days and is more than a million kilometres off within a year. Missions such as SOHO (at L1) and JWST (at L2) fire their thrusters regularly to stay in place. This project builds the dynamics from scratch, verifies them, and compares an LQR controller with DQN and PPO agents on the station-keeping task.

![Uncontrolled drift away from L1](figures/l1_escape.png)

*Left: distance from L1 grows exactly at the rate predicted by the unstable eigenvalue (λ = 2.533). Right: the escape trajectory in the rotating frame.*

## Status

| Phase | Content | Status |
| --- | --- | --- |
| 1 | CR3BP simulator, RK4 + DOP853, Jacobi-constant validation | Done |
| 2 | Lagrange points, linear stability, uncontrolled escape from L1 | Done |
| 3 | LQR baseline controller | Planned |
| 4 | Gymnasium environment, DQN agent | Planned |
| 5 | PPO agent, comparison experiments (Δv, tracking error, robustness) | Planned |
| 6 | Paper | Planned |

## Results so far

**Lagrange points (Sun–Earth, μ = 3.0034 × 10⁻⁶)**

| Point | Distance from Earth | Unstable growth rate | e-folding time |
| --- | --- | --- | --- |
| L1 | 1,491,539 km | 2.533 | 23.0 days |
| L2 | 1,501,519 km | 2.484 | 23.4 days |
| L3 | 299,195,479 km | 0.0028 | ~57 years |
| L4, L5 | 1 AU (60° ahead / behind) | 0 (stable, Routh: μ < 0.0385) | — |

![Lagrange points and zero-velocity curves](figures/lagrange_points.png)

**Integrator validation.** Over 10 years, DOP853 keeps the relative Jacobi-constant error below 10⁻¹³; fixed-step RK4 converges with measured order 3.92 (theory: 4).

![Jacobi constant drift and RK4 convergence](figures/jacobi_validation.png)

## The model

Nondimensional units: the Sun–Earth distance is 1, the total mass is 1, and one year is 2π time units (1 time unit ≈ 58.1 days). In the frame rotating with the Earth:

```
x'' - 2y' = ∂Ω/∂x + uₓ
y'' + 2x' = ∂Ω/∂y + u_y
z''       = ∂Ω/∂z + u_z

Ω = (x² + y²)/2 + (1 - μ)/r₁ + μ/r₂
```

`u` is the control acceleration supplied by the thrusters. With `u = 0` the Jacobi constant `C = 2Ω − v²` is conserved, which is the main accuracy check.

Velocity Verlet is not used because the Coriolis term depends on velocity, which breaks the scheme's symplectic property. RK4 (fixed step) drives the RL environment; DOP853 (adaptive, scipy) is the reference.

## Quick start

```bash
git clone https://github.com/<your-username>/l1-station-keeping.git
cd l1-station-keeping
pip install -e ".[dev]"

pytest                                  # 12 correctness tests
python scripts/01_lagrange_points.py    # table + figures/lagrange_points.png
python scripts/02_jacobi_validation.py  # figures/jacobi_validation.png
python scripts/03_l1_escape.py          # figures/l1_escape.png
```

```python
import numpy as np
from cr3bp import MU_SUN_EARTH, lagrange_points, stability, rhs, dop853

mu = MU_SUN_EARTH
l1 = lagrange_points(mu)["L1"]
print(stability(l1, mu)["efold_days"])       # ~23.0

s0 = np.r_[l1 + [1e-9, 0, 0], 0, 0, 0]       # 150 m off L1, at rest
t, states = dop853(lambda t, s: rhs(t, s, mu), s0, t_end=6.0)
```

## Repository layout

```
src/cr3bp/
  constants.py     mass ratio and unit conversions (km, days, m/s)
  dynamics.py      equations of motion, effective potential, Jacobi constant
  integrators.py   RK4 and DOP853
  lagrange.py      L1–L5, Hessian, linearisation, eigenvalue stability
scripts/           one script per figure
tests/             pytest suite
figures/           generated plots
```

## References

- W. S. Koon, M. W. Lo, J. E. Marsden, S. D. Ross, *Dynamical Systems, the Three-Body Problem and Space Mission Design*.
- V. Szebehely, *Theory of Orbits: The Restricted Problem of Three Bodies*. Academic Press, 1967.
- S. H. Strogatz, *Nonlinear Dynamics and Chaos*.
- R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.

## License

MIT © 2026 Emir Eymen Sargın
