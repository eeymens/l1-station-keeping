# L1 Station-Keeping: LQR vs Reinforcement Learning

Can a reinforcement learning agent keep a spacecraft at the unstable Sun–Earth L1 point using less fuel (Δv) than a classical LQR controller?

The Sun–Earth L1 point sits about 1.49 million km sunward of Earth. It is an equilibrium of the circular restricted three-body problem (CR3BP), but an unstable one: a spacecraft left there with a 150 m error drifts away by a factor of *e* every 23 days and is more than a million kilometres off within a year. Missions such as SOHO (at L1) and JWST (at L2) fire their thrusters regularly to stay in place. This project builds the dynamics from scratch, verifies them, and compares an LQR controller with DQN and PPO agents on the station-keeping task.

**Short answer: no.** PPO with continuous thrust gets within 1.4–1.6× of LQR's fuel at the same accuracy (DQN: 2–12×), and it handles the thrust limit better than a saturated LQR, but it never beats LQR on fuel.

![Uncontrolled drift away from L1](figures/l1_escape.png)

*Left: distance from L1 grows exactly at the rate predicted by the unstable eigenvalue (λ = 2.533). Right: the escape trajectory in the rotating frame.*

## Status

| Phase | Content | Status |
| --- | --- | --- |
| 1 | CR3BP simulator, RK4 + DOP853, Jacobi-constant validation | Done |
| 2 | Lagrange points, linear stability, uncontrolled escape from L1 | Done |
| 3 | LQR baseline controller, fuel/accuracy trade-off | Done |
| 4 | Gymnasium environment, DQN agent, comparison with LQR | Done |
| 5 | PPO agent, comparison experiments (Δv, tracking error, robustness) | Done |
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

**LQR station-keeping (phase 3).** A discrete-time LQR updates the thrust once per day from a noisy navigation estimate. Starting 1,000 km from L1, the uncontrolled spacecraft is lost (> 10,000 km) after 61 days; LQR brings it back and holds it for a full year.

![LQR station-keeping](figures/lqr_station_keeping.png)

Sweeping the fuel weight ρ over nine orders of magnitude (5 seeds each) gives the baseline the RL agents must beat:

- **Holding** (start at L1): 0.03–0.95 m/s per year of Δv, for a mean distance of 2.5–0.4 km. Below ~0.4 km, navigation errors set the floor and extra fuel buys almost nothing. Without control, unmodelled forces alone push the spacecraft out within 230–260 days.
- **Acquisition** (start 1,000 km away): the cheapest gain (ρ = 3) needs 1.12 m/s. Lazier gains cost *more*, because the unstable mode keeps growing while the controller waits.

![LQR fuel/accuracy trade-off](figures/lqr_pareto.png)

Error model (1σ, illustrative assumptions): navigation errors of 1 km and 1 mm/s, thrust errors of 2 % in magnitude and 1° in pointing, and a random unmodelled acceleration of 10⁻⁹ m/s² per axis per day (solar radiation pressure changes, the Moon, planets). Raw results: `experiments/lqr_pareto.csv`.

**DQN vs LQR (phase 4).** A DQN agent picks one of 25 actions each day: coast, or thrust in one of 8 directions at 10⁻⁸, 10⁻⁷ or 10⁻⁶ m/s². Two training runs of 1M steps each; the best checkpoint of each run is tested on 20 seeds never used in training or model selection.

| Controller | Holding: Δv / year | Holding: mean distance | Recovery from 1,000 km: Δv | Recovery: mean distance | Runs kept 1 year |
| --- | --- | --- | --- | --- | --- |
| No control | 0 | lost after ~250 days | 0 | lost after 61 days | 0 % |
| LQR (ρ = 3) | 0.062 m/s | 0.5 km | 1.12 m/s | 76 km | 100 % |
| DQN run 1 | 0.42 m/s | 10.9 km | 2.11 m/s | 118 km | 100 % |
| DQN run 2 (lower learning rate, stronger distance penalty) | 0.72 m/s | 1.4 km | 2.17 m/s | 68 km | 100 % |

Both agents learned to keep the spacecraft in every test episode, but LQR uses 2–12× less fuel. The thrust logs show why: while holding, LQR's thrust is about 2 × 10⁻⁹ m/s², five times below the smallest discrete DQN action. Run 1 fires that smallest level on 351 of 365 days; run 2 coasts on 214 days and over-corrects with 10⁻⁷ m/s² on 97. The gap comes from the coarse action set rather than from learning, which is what phase 5 tests with continuous-thrust PPO. Run 1 also collapsed after 700k steps (a known DQN instability); run 2's lower learning rate and slower target updates kept training stable.

![LQR vs DQN](figures/lqr_vs_dqn_l1_run2.png)

**PPO vs LQR (phase 5).** PPO outputs a continuous 2-D thrust vector, capped at the same 10⁻⁶ m/s² as DQN's largest action. The action *a* (|*a*| ≤ 1) is mapped to thrust *u* = *u*ₘₐₓ |*a*|³ *a*/|*a*|: the cubic map makes the ~2 × 10⁻⁹ m/s² holding thrust a moderate action (|*a*| ≈ 0.13) instead of 0.002 of full scale. A second run with the plain linear map (|*a*|¹) isolates the effect of this choice. Both runs: 3M steps, 8 parallel environments, same reward and model selection as DQN run 2 (~11 min each). Test seeds 100–119:

| Controller | Holding: Δv / year | Holding: mean / RMS distance | Recovery: Δv | Recovery: mean distance | Runs kept 1 year |
| --- | --- | --- | --- | --- | --- |
| LQR (ρ = 3) | 0.062 m/s | 0.5 / 0.6 km | 1.12 m/s | 76 km | 100 % |
| DQN run 2 | 0.72 m/s | 1.4 / 1.6 km | 2.17 m/s | 68 km | 100 % |
| PPO, linear thrust map | 0.41 m/s | 0.4 / 0.5 km | 1.86 m/s | 56 km | 100 % |
| **PPO, cubic thrust map** | **0.12 m/s** | **0.5 / 0.5 km** | **1.60 m/s** | **65 km** | 100 % |

PPO tracks a little more tightly than LQR ρ = 3, so the fair comparison is against LQR tuned to the *same* accuracy (interpolating the ρ sweep): there LQR needs 0.074 m/s while holding and 1.11 m/s for recovery, so **PPO uses 1.6× and 1.4× more fuel**. Both PPO points sit above the LQR curve: LQR is still Pareto-optimal.

![LQR vs DQN vs PPO](figures/lqr_vs_phase5.png)

The thrust logs (holding, seed 100) show what changed since phase 4. PPO's median daily thrust is 1.73 × 10⁻⁹ m/s², almost exactly LQR's 1.79 × 10⁻⁹, so the agent did learn the right thrust scale. The extra fuel comes from the tail: PPO thrusts above 10⁻⁸ m/s² on 27 days, LQR on none. With the linear map the median is 1.0 × 10⁻⁸ m/s² (185 days above 10⁻⁸): the network cannot output 0.002 of full scale precisely enough, and fuel is 3.5× higher. **Most of the DQN–LQR gap was the action representation, not learning.** The best PPO checkpoint came after only 250k steps; longer training traded fuel for slightly tighter tracking and never raised the selection score again (`figures/ppo_l1_learning_curve.png`).

**Robustness (phase 5).** All controllers were trained or tuned with the nominal error model and start offsets ≤ 1,000 km. `scripts/09_robustness.py` pushes them past that on the test seeds. Because the agents can thrust at most 10⁻⁶ m/s² and plain LQR has no limit, LQR is also flown saturated at 10⁻⁶ m/s² ("sat.").

| Sweep (Δv / RMS distance in days 183–365) | LQR | PPO | DQN |
| --- | --- | --- | --- |
| All errors ×0 (holding) | 0.00 m/s / 0.0 km | 0.00 m/s / 0.7 km | 0.36 m/s / 2.7 km |
| All errors ×1 (holding) | 0.06 m/s / 0.6 km | 0.12 m/s / 0.5 km | 0.72 m/s / 1.6 km |
| All errors ×8 (holding) | 0.50 m/s / 4.9 km | 2.81 m/s / 4.2 km | 5.49 m/s / 9.6 km |
| Unmodelled force ×100 (holding) | 1.13 m/s / 33 km | 2.18 m/s / 6.1 km | 4.10 m/s / 9.6 km |
| Start 2,000 km away: kept | 100 % (sat.: 100 %) | 100 % | 65 % |
| Start 3,000 km away: kept | 100 % (sat.: 50 %) | 100 % | 50 % |
| Start 5,000 km away: kept | 100 % (sat.: 50 %) | 50 % | 25 % |

- **Noise.** No controller lost a single run at up to 8× the nominal errors. PPO's fuel overhead grows with noise, from 2× LQR at nominal to 5.6× at ×8: it reacts to navigation noise instead of averaging it out, while LQR's gain is linear in the error by construction.
- **Unmodelled forces.** With a 100× stronger unmodelled force, LQR ρ = 3 drifts to 33 km RMS because its gain was tuned for nominal conditions; PPO holds 6 km with twice the fuel. The two are no longer at the same point of the trade-off, so neither "wins" here.
- **Large offsets and the thrust limit.** Apart from DQN at 5,000 km, every lost run started on the Sun–Earth line (0° or 180°), the direction of the unstable mode. A rough estimate of the mode's pull, λ²r, is 0.76 × 10⁻⁶ m/s² at 3,000 km and 1.27 × 10⁻⁶ m/s² at 5,000 km: at 5,000 km it exceeds the 10⁻⁶ m/s² limit, which is why all bounded controllers lose exactly those runs. At 3,000 km recovery is possible but only just: saturated LQR, whose gain ignores the limit, loses all 10 runs on the Sun–Earth line, while PPO keeps them, at a cost of ~13 m/s and still 60–215 km off after a year. Unlimited LQR keeps everything but needs up to 4.9 × 10⁻⁶ m/s².

![Robustness sweeps](figures/robustness.png)

Raw results: `experiments/comparison_phase5.csv`, `experiments/robustness.csv`.

**Current simplifications.** The spacecraft holds the L1 point itself; real missions fly halo or Lissajous orbits around it (to keep the Sun out of the antenna beam). Motion is planar, and thrust is a low, continuous push updated daily rather than impulsive burns weeks apart. These are deliberate, so LQR and RL are compared on identical, controlled conditions.

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
pip install -e ".[rl]"                  # phase 4+: gymnasium, stable-baselines3

pytest                                  # correctness tests
python scripts/01_lagrange_points.py    # table + figures/lagrange_points.png
python scripts/02_jacobi_validation.py  # figures/jacobi_validation.png
python scripts/03_l1_escape.py          # figures/l1_escape.png
python scripts/04_lqr_station_keeping.py # figures/lqr_station_keeping.png
python scripts/05_lqr_pareto.py         # figures/lqr_pareto.png (~30 s)
python scripts/06_train_dqn.py --steps 1000000 --lr 1e-4 --target-update 10000 \
    --w-pos 0.3 --name dqn_l1_run2      # ~22 min on one CPU core
python scripts/07_compare_controllers.py --model dqn_l1_run2
python scripts/08_train_ppo.py --steps 3000000 --name ppo_l1          # ~11 min, 8 envs
python scripts/08_train_ppo.py --steps 3000000 --action-power 1 --name ppo_l1_linear
python scripts/07_compare_controllers.py --model dqn_l1_run2 ppo_l1 ppo_l1_linear --out phase5
python scripts/09_robustness.py --model dqn_l1_run2 ppo_l1           # ~2 min
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
  control/
    station_keeping.py   simulator: daily thrust, navigation/thrust errors, delta-v log
    lqr.py               ZOH discretisation, discrete Riccati, LQR controller
  envs/l1_env.py         gymnasium environment (discrete or continuous thrust)
  evaluation.py          common test protocol: same scenarios and seeds for every controller
  training.py            shared RL pieces: evaluation callback, model config, load_policy
models/            trained agents (best checkpoint of each run) + .json with env settings
experiments/       raw CSV results
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
