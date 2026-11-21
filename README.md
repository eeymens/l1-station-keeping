# L1 Station-Keeping: LQR vs Reinforcement Learning

Can a reinforcement learning agent keep a spacecraft at the unstable Sun–Earth L1 point using less fuel (Δv) than a classical LQR controller?

The Sun–Earth L1 point sits about 1.49 million km sunward of Earth. It is an equilibrium of the circular restricted three-body problem (CR3BP), but an unstable one: a spacecraft left there with a 150 m error drifts away by a factor of *e* every 23 days and is more than a million kilometres off within a year. Missions such as SOHO (at L1) and JWST (at L2) fire their thrusters regularly to stay in place. This project builds the dynamics from scratch, verifies them, and compares an LQR controller with DQN and PPO agents on the station-keeping task.

**Short answer: no.** Across three training seeds, PPO with continuous thrust needs 1.3–2.7× LQR's fuel at the same accuracy while holding (DQN: 2–12×) and never beats LQR on fuel. One of the three PPO seeds also handled the thrust limit better than a saturated LQR, but the other two did not.

**Phase 7: what if the real physics differs from the model?** With solar radiation pressure (SRP) and Earth's elliptic orbit added to the simulator only, plain LQR degrades (≈30× the nominal fuel at realistic SRP) and PPO trained with domain randomisation gets within 1.01–1.13× of the best classical fuel while holding L1 to 0.8–3 km. But once the classical side gets its textbook fixes (integral action, a Kalman disturbance estimator), all 80 RL test cases are still dominated. The eccentricity alone has almost no effect; SRP is what matters.

**Phase 8: give RL the missing information.** A memory of the last 10 observations makes PPO *worse*. A residual PPO correction on top of the Kalman bias-estimating LQR is the first learned controller that is **not dominated**: on 100 fresh seeds and after a dense search of 1,011 classical designs per severity, no classical design beats it while holding under SRP. Its edge is at most 0.6 % (≤ 0.01 m/s per year), though, so RL *reaches* the classical frontier rather than moving it.

📄 **Paper draft:** [`paper/main.pdf`](paper/main.pdf) · **Reproduce everything:** `make reproduce` (see [Reproducing the results](#reproducing-the-results))

![Uncontrolled drift away from L1](figures/l1_escape.png)

*Left: distance from L1 grows exactly at the rate predicted by the unstable eigenvalue (λ = 2.533). Right: the escape trajectory in the rotating frame.*

## Status

| Phase | Content | Status |
| --- | --- | --- |
| 1 | CR3BP simulator, RK4 + DOP853, Jacobi-constant validation | Done |
| 2 | Lagrange points, linear stability, uncontrolled escape from L1 | Done |
| 3 | LQR baseline controller, fuel/accuracy trade-off | Done |
| 4 | Gymnasium environment, DQN agent, comparison with LQR | Done |
| 5 | PPO agent, comparison experiments (Δv, tracking error, robustness), 3 training seeds | Done |
| 6 | Paper draft (numbers generated from the CSVs), one-command reproduction, CI | Done |
| 7 | Model mismatch: elliptic Earth orbit + solar radiation pressure, LQI and Kalman bias-estimating LQR, domain-randomised PPO (5 seeds) | Done |
| 8 | RL with memory and residual RL on Bias-LQR (5 seeds each), confirmation on fresh seeds and a dense classical frontier search | Done |

## Results

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

**PPO vs LQR (phase 5).** PPO outputs a continuous 2-D thrust vector, capped at the same 10⁻⁶ m/s² as DQN's largest action. The action *a* (|*a*| ≤ 1) is mapped to thrust *u* = *u*ₘₐₓ |*a*|³ *a*/|*a*|: the cubic map makes the ~2 × 10⁻⁹ m/s² holding thrust a moderate action (|*a*| ≈ 0.13) instead of 0.002 of full scale. A second configuration with the plain linear map (|*a*|¹) isolates the effect of this choice. Every run: 3M steps, 8 parallel environments, same reward and model selection as DQN run 2 (~11 min each). Each configuration was trained with three training seeds (0, 1, 2) and otherwise identical hyperparameters, to check that the results are not one lucky run. Test seeds 100–119; the "3 seeds" rows are mean ± standard deviation (n = 3, ddof = 1) across training seeds:

| Controller | Holding: Δv / year | Holding: mean / RMS distance | Recovery: Δv | Recovery: mean distance | Runs kept 1 year |
| --- | --- | --- | --- | --- | --- |
| LQR (ρ = 3) | 0.062 m/s | 0.5 / 0.6 km | 1.12 m/s | 76 km | 100 % |
| DQN run 2 | 0.72 m/s | 1.4 / 1.6 km | 2.17 m/s | 68 km | 100 % |
| PPO, linear thrust map (training seed 0) | 0.41 m/s | 0.4 / 0.5 km | 1.86 m/s | 56 km | 100 % |
| PPO, cubic thrust map (training seed 0) | 0.12 m/s | 0.5 / 0.5 km | 1.60 m/s | 65 km | 100 % |
| PPO, linear thrust map (3 seeds) | 0.39 ± 0.04 m/s | 0.45 ± 0.04 / 0.51 ± 0.04 km | 1.72 ± 0.12 m/s | 72 ± 14 km | 100 % |
| **PPO, cubic thrust map (3 seeds)** | **0.110 ± 0.007 m/s** | **0.57 ± 0.17 / 0.64 ± 0.18 km** | **1.46 ± 0.18 m/s** | **84 ± 30 km** | 100 % |

PPO does not track at exactly the accuracy of LQR ρ = 3, so the fair comparison is against LQR tuned to the *same* accuracy. `scripts/10_seed_study.py` sweeps LQR over 57 values of ρ (10⁻³–10⁴) on the test seeds, keeps the branch from the tightest gain up to the Δv minimum (beyond it, lazier gains cost more fuel again), and interpolates log-log at each model's mean distance:

| PPO model | Holding: × LQR fuel at same accuracy | Recovery: × LQR fuel at same accuracy | Best checkpoint |
| --- | --- | --- | --- |
| cubic, seed 0 | 1.54 | 1.44 | 250k steps |
| cubic, seed 1 | 1.31 | 1.13 | 250k steps |
| cubic, seed 2 | 2.74 | dominated (≥ 1.37) | 200k steps |
| linear, seed 0 | 4.48 | 1.61 | 2.95M steps |
| linear, seed 1 | 5.92 | dominated (≥ 1.48) | 1.2M steps |
| linear, seed 2 | 2.90 | dominated (≥ 1.50) | 2.9M steps |
| **cubic, mean ± std** | **1.87 ± 0.77** | — | |
| linear, mean ± std | 4.43 ± 1.51 | — | |

"Dominated" means the model tracks *less* tightly than the cheapest LQR (ρ ≈ 1.8: 1.11 m/s at 69 km), which is therefore both cheaper and more accurate; the bracket is the ratio to that cheapest LQR, a lower bound. The single-seed figure quoted before ("1.6× / 1.4×", training seed 0, from a coarser interpolation) does not carry over as one number: across three training seeds the cubic agent needs 1.3–2.7× LQR's fuel while holding and 1.1–1.4× (or is dominated) for recovery. What does hold in all three seeds is the conclusion: **every PPO model sits above the LQR curve, so LQR is still Pareto-optimal.**

![PPO training seeds vs the LQR trade-off](figures/seed_study.png)

![LQR vs DQN vs PPO](figures/lqr_vs_phase5.png)

The thrust logs (holding, seed 100) show what changed since phase 4. PPO's median daily thrust is 1.73 × 10⁻⁹ m/s², almost exactly LQR's 1.79 × 10⁻⁹, so the agent did learn the right thrust scale. The extra fuel comes from the tail: PPO thrusts above 10⁻⁸ m/s² on 27 days, LQR on none. With the linear map the median is 1.0 × 10⁻⁸ m/s² (185 days above 10⁻⁸): the network cannot output 0.002 of full scale precisely enough. Across the three training seeds the linear map needs 3.1–4.1× the cubic map's holding fuel (0.39 ± 0.04 vs 0.110 ± 0.007 m/s, while tracking slightly tighter: 0.45 vs 0.57 km) and 1.1–1.3× its recovery fuel, with no overlap between the two configurations in either scenario. **In all 3 training seeds, most of the DQN–LQR gap was the action representation, not learning.** With the cubic map the best checkpoint came early in all 3 training seeds (200k–250k of 3M steps); longer training traded fuel for slightly tighter tracking and never raised the selection score again (`figures/ppo_l1_learning_curve.png`). The linear map is different: its best checkpoints came at 1.2M–2.95M steps.

**Robustness (phase 5).** All controllers were trained or tuned with the nominal error model and start offsets ≤ 1,000 km. `scripts/09_robustness.py` pushes them past that on the test seeds. Because the agents can thrust at most 10⁻⁶ m/s² and plain LQR has no limit, LQR is also flown saturated at 10⁻⁶ m/s² ("sat.").

The PPO column covers the cubic agent's three training seeds (`scripts/09_robustness.py --model ppo_l1 ppo_l1_s1 ppo_l1_s2 --out robustness_seeds`): a range is min–max over the seeds, and kept rates are listed as seed 0 / 1 / 2.

| Sweep (Δv / RMS distance in days 183–365) | LQR | PPO cubic, 3 training seeds | DQN |
| --- | --- | --- | --- |
| All errors ×0 (holding) | 0.00 m/s / 0.0 km | 0.00–0.01 m/s / 0.7–1.5 km | 0.36 m/s / 2.7 km |
| All errors ×1 (holding) | 0.06 m/s / 0.6 km | 0.10–0.12 m/s / 0.5–0.8 km | 0.72 m/s / 1.6 km |
| All errors ×8 (holding) | 0.50 m/s / 4.9 km | 2.02–2.81 m/s / 4.2–4.3 km | 5.49 m/s / 9.6 km |
| Unmodelled force ×100 (holding) | 1.13 m/s / 33 km | 1.87–2.18 m/s / 6.1–9.6 km | 4.10 m/s / 9.6 km |
| Start 2,000 km away: kept | 100 % (sat.: 100 %) | 100 % / 100 % / 100 % | 65 % |
| Start 3,000 km away: kept | 100 % (sat.: 50 %) | 100 % / 50 % / 50 % | 50 % |
| Start 5,000 km away: kept | 100 % (sat.: 50 %) | 50 % / 50 % / 50 % | 25 % |

- **Noise.** No controller lost a single run at up to 8× the nominal errors (all 3 PPO training seeds). PPO's fuel overhead grows with noise, from 1.6–1.9× LQR at nominal to 4.0–5.6× at ×8: it reacts to navigation noise instead of averaging it out, while LQR's gain is linear in the error by construction.
- **Unmodelled forces.** With a 100× stronger unmodelled force, LQR ρ = 3 drifts to 33 km RMS because its gain was tuned for nominal conditions; PPO holds 6–10 km with 1.7–1.9× the fuel. The two are no longer at the same point of the trade-off, so neither "wins" here.
- **Large offsets and the thrust limit.** Apart from DQN at 5,000 km, every lost run started on the Sun–Earth line (0° or 180°), the direction of the unstable mode. A rough estimate of the mode's pull, λ²r, is 0.76 × 10⁻⁶ m/s² at 3,000 km and 1.27 × 10⁻⁶ m/s² at 5,000 km: at 5,000 km it exceeds the 10⁻⁶ m/s² limit, which is why all bounded controllers lose exactly those runs. At 3,000 km recovery is possible but only just: saturated LQR, whose gain ignores the limit, loses all 10 runs on the Sun–Earth line. PPO training seed 0 keeps them, at a cost of ~13 m/s and still 56–215 km off (RMS, second half of the year), **but training seeds 1 and 2 lose exactly the same 10 runs as saturated LQR.** So the earlier claim that PPO handles the thrust limit better than saturated LQR does not hold across training seeds; it was a property of one run. Unlimited LQR keeps everything but needs up to 4.9 × 10⁻⁶ m/s².

![Robustness sweeps](figures/robustness.png)

Raw results: `experiments/comparison_phase5.csv`, `experiments/robustness.csv` (training seed 0 only), `experiments/seed_study.csv` (+ `seed_study_lqr.csv`, the 57-point LQR sweep), `experiments/robustness_seeds.csv`.

**Model mismatch (phase 7).** Everything above uses the circular restricted three-body problem (CR3BP) for both the controllers and the simulator. Phase 7 adds two pieces of physics to the *simulator only* (`src/cr3bp/perturbations.py`), so every controller meets dynamics it was not designed for:

- **Earth's elliptic orbit** (e = 0.0167), in pulsating rotating coordinates (Szebehely 1967): lengths scale with the instantaneous Sun–Earth distance and true anomaly replaces time, so L1 stays at its CR3BP position and e = 0 gives back the CR3BP exactly (tested bit for bit). A brute-force integration in an inertial frame with the Sun and Earth on Kepler ellipses agrees to < 1 m after 30 days.
- **Solar radiation pressure**: 4.56 × 10⁻⁶ N/m² × C_r 1.3 × A/m 0.01 m²/kg ≈ 5.9 × 10⁻⁸ m/s² at 1 AU (C_r and A/m are assumptions). Only 6 % of the thrust limit, but always in the same direction: cancelling it costs ~1.9 m/s per year, and it moves the equilibrium 167 km sunward.

A severity k scales both (e = k × 0.0167, SRP × k; k = 1 realistic, k = 4 four times). New controllers, so the classical side is not left weak against a constant force: **LQI** (LQR + integral action, holds exactly at L1) and **Bias-LQR** (Kalman filter estimating state + constant force, LQR around the cheapest steady state; it parks where gravity balances SRP). **DR-PPO**: 5 PPO agents (training seeds 0–4, unchanged hyperparameters) trained with domain randomisation, k ~ U(0, 4), random season and SRP factor per episode. `scripts/12_model_mismatch.py` flies everything on test seeds 100–119 and compares against **345 classical designs** (3 families × ρ × position scale r_ref = 1000/100/10 km):

| k | Holding: LQR ρ=1 | Bias-LQR ρ=1 | PPO (3 seeds) | DR-PPO (5 seeds) | PPO × best classical | DR-PPO × best classical |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.09 m/s (0.5 km) | 0.04 (0.6) | 0.11 (0.6) | 0.18 (7.9) | 1.99 ± 0.96 | 5.92 ± 1.87 |
| 0.5 | 1.28 (29) | 0.88 (9.2) | 1.12 (11) | 1.04 (2.1) | 1.31 ± 0.02 | 1.10 ± 0.03 |
| 1 | 2.55 (58) | 1.75 (18) | 2.20 (22) | 2.02 (1.4) | 1.26 ± 0.02 | 1.06 ± 0.02 |
| 2 | 5.10 (117) | 3.50 (37) | 4.40 (48) | 3.95 (0.9) | 1.28 ± 0.05 | 1.03 ± 0.01 |
| 4 | 10.26 (234) | 7.00 (74) | 8.99 (118) | 7.80 (1.9) | 1.34 ± 0.07 | 1.02 ± 0.01 |

"× best classical" = model Δv ÷ Δv of the cheapest classical design (any family, ρ, r_ref) that is at least as accurate. Recovery from 1,000 km and every per-seed number: `experiments/model_mismatch.csv`.

- **Eccentricity barely matters, SRP does.** Flown separately, e = 0.067 (k = 4) leaves LQR's holding cost at 0.085 m/s, unchanged from nominal: in pulsating coordinates L1 is still an equilibrium and the eccentricity only scales forces proportional to the distance from it (a few km here; it would matter for 10⁵ km halo orbits). All the change comes from SRP (`experiments/model_mismatch_parts.csv`).
- **The LQR trade-off turns around.** Under SRP, looser LQR gains drift *away* from the Sun, where L1's instability adds to the load, so they are both farther and more expensive. Bias-LQR parks on the sunward side instead (k = 1: 0.40 m/s at 222 km).
- **Nominal PPO** (never trained on SRP) loses no runs but needs 1.23–1.42× the best classical fuel while holding and 1.30–3.92× in recovery.
- **DR-PPO** holds L1 to 0.8–3.0 km at every k > 0, tighter than any ρ = 1 classical design, at 1.01–1.13× the best classical fuel (1.06–1.48× in recovery). But near L1 every controller has to pay the SRP cancellation cost, which pushes all ratios towards 1; in absolute terms DR-PPO still spends 0.06–0.24 m/s per year more. **All 80 RL test cases are dominated** by a classical design that is both cheaper and at least as accurate (73 of them by a Bias-LQR).
- **A near-miss worth reporting.** With only the r_ref = 1000 km designs, DR-PPO at k = 4 looked more accurate than every classical controller. That came from tuning the classical side too narrowly; a tighter position scale removes it.
- **Robustness has a price.** At k = 0, DR-PPO needs 5.9× the best classical fuel and holds 9.7 km sunward of L1. Without memory it cannot tell whether SRP is present, so it thrusts as if an average SRP were acting. Bias-LQR's Kalman filter solves exactly this; an RL agent with memory (frame stacking or LSTM) is the natural next step.

![Fuel/accuracy under model mismatch](figures/model_mismatch_pareto.png)

![Controllers vs mismatch severity](figures/model_mismatch.png)

Explanatory notes in Turkish (equations, design choices, which line does what): [`docs/faz7_notlar.md`](docs/faz7_notlar.md).

**Memory and residual RL (phase 8).** DR-PPO cannot tell how strong the SRP is; Bias-LQR, which dominates it, estimates exactly that. Two ways to give RL the same information, trained exactly like DR-PPO (same hyperparameters and domain randomisation, 5 training seeds each):

- **MEM-PPO**: the observation holds the last 10 navigation estimates and the last 10 actions (`--history 10`).
- **RES-PPO** (residual RL): thrust = Bias-LQR (ρ = 1) + agent, capped at 10⁻⁶ m/s², and the agent also sees the filter's state and force estimates and the base thrust (`--residual-rho 1`). With a zero action it *is* the Bias-LQR (unit-tested). Writing that test caught a real bug: the filter has to be told the total thrust actually applied, not only its own.

`scripts/13_memory_residual.py` (test seeds 100–119, Δv ÷ cheapest classical design at least as accurate, mean ± std over 5 seeds):

| k | Holding: DR-PPO | MEM-PPO | RES-PPO | RES-PPO seeds not dominated |
| --- | --- | --- | --- | --- |
| 0 | 5.92 ± 1.87 | 12.20 ± 5.10 | 0.987 ± 0.093 | 3/5 |
| 0.5 | 1.095 ± 0.029 | 1.332 ± 0.200 | 0.990 ± 0.005 | 5/5 |
| 1 | 1.057 ± 0.016 | 1.151 ± 0.104 | 0.994 ± 0.001 | 5/5 |
| 2 | 1.028 ± 0.009 | 1.071 ± 0.056 | 0.996 ± 0.003 | 5/5 |
| 4 | 1.017 ± 0.008 | 1.045 ± 0.017 | 0.999 ± 0.003 | 3/5 |

- **Memory did not help.** MEM-PPO is worse than DR-PPO in all 10 k/scenario combinations and much less consistent across seeds. Stacked observations carry the information but also 10× the navigation noise; with the unchanged training budget the network did not learn to use it.
- **Residual RL reaches the frontier.** RES-PPO slides its base *along* the Bias-LQR frontier towards tighter tracking (k = 1: base 18 km / 1.75 m/s → 6.4 km / 1.87 m/s). In recovery it is dominated in all 25 cases (1.03–1.07× the envelope).
- **Confirmation** (`scripts/14_confirmation.py`): 100 fresh seeds (200–299), 1,710 Bias-LQR and LQI designs (r_ref = 1000/300/100/30/10 km × 57 ρ), paired bootstrap. RES-PPO is not dominated in 13 of 15 holding cases, at 0.994–1.001× the envelope.
- **Dense frontier search** (`scripts/15_frontier_search.py`): because the envelope is very flat, Bias-LQR's two tuning parameters were searched on a 21 × 21 grid around the agents (1,011 designs per severity in total). **None dominates any of the 10 agents; 9 of 10 lie below the envelope, by at most 0.6 % (≤ 0.010 m/s per year).**

So residual RL *matches* the best classical controller while holding under SRP, with a margin too small to matter physically. It learned to re-tune its classical base along the frontier, not a better control law. A practical upside: a zero action recovers the trusted classical controller.

![Memory and residual RL vs the classical envelopes](figures/phase8_pareto.png)

Turkish notes: [`docs/faz8_notlar.md`](docs/faz8_notlar.md). Raw results: `experiments/phase8.csv`, `experiments/confirmation*.csv`, `experiments/frontier_search*.csv`.

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

## Reproducing the results

```bash
git clone https://github.com/eeymens/l1-station-keeping.git
cd l1-station-keeping
make install        # pip install -e ".[dev,rl]"  (CPU-only PyTorch is enough)
make test           # pytest, 36 tests
make reproduce      # every figure, CSV and paper number from the trained models (~4 h on two cores, mostly the phase-7/8 classical sweeps)
make paper          # paper/main.pdf (needs LaTeX)
```

`make reproduce` evaluates the trained agents in `models/`; it does not retrain them. Re-running it on the committed models reproduces the CSVs in `experiments/` exactly. `make train` retrains all 23 agents from scratch (~8 h on one core); PyTorch's nondeterminism across machines means retrained agents will differ slightly from the committed ones. `make -n <target>` prints the individual commands, one script per figure:

| Script | Output |
| --- | --- |
| `01_lagrange_points.py` | Lagrange-point table, `figures/lagrange_points.png` |
| `02_jacobi_validation.py` | `figures/jacobi_validation.png` |
| `03_l1_escape.py` | `figures/l1_escape.png` |
| `04_lqr_station_keeping.py` | `figures/lqr_station_keeping.png` |
| `05_lqr_pareto.py` | `experiments/lqr_pareto.csv`, `figures/lqr_pareto.png` |
| `06_train_dqn.py` | `models/dqn_*`, training log and learning curve |
| `07_compare_controllers.py` | `experiments/comparison_*.csv`, `figures/lqr_vs_*.png` |
| `08_train_ppo.py` | `models/ppo_*`, training log and learning curve |
| `09_robustness.py` | `experiments/robustness*.csv`, `figures/robustness*.png` |
| `10_seed_study.py` | `experiments/seed_study*.csv`, `figures/seed_study.png` |
| `12_model_mismatch.py` | `experiments/model_mismatch*.csv`, `figures/model_mismatch*.png` (phase 7) |
| `13_memory_residual.py` | `experiments/phase8.csv`, `figures/phase8_pareto.png` (phase 8) |
| `14_confirmation.py` | `experiments/confirmation*.csv` (phase 8, seeds 200–299) |
| `15_frontier_search.py` | `experiments/frontier_search*.csv` (phase 8, dense classical search) |
| `11_paper_assets.py` | `paper/generated/` — every number and table in the paper |

Package versions verified to reproduce the committed results: [`requirements-lock.txt`](requirements-lock.txt).

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
    lqi.py               phase 7: LQI and Kalman bias-estimating LQR
  perturbations.py       phase 7: elliptic Earth orbit (pulsating coords) + solar radiation pressure
  envs/l1_env.py         gymnasium environment (discrete or continuous thrust)
  evaluation.py          common test protocol: same scenarios and seeds for every controller
  training.py            shared RL pieces: evaluation callback, model config, load_policy
models/            trained agents (best checkpoint of each run) + .json with env settings
experiments/       raw CSV results
scripts/           one script per figure
tests/             pytest suite
docs/              explanatory notes (phases 7 and 8, in Turkish)
figures/           generated plots
paper/             manuscript (main.tex, references.bib, generated/ numbers and tables)
Makefile           install, test, reproduce, paper, train
```

## References

- W. S. Koon, M. W. Lo, J. E. Marsden, S. D. Ross, *Dynamical Systems, the Three-Body Problem and Space Mission Design*.
- V. Szebehely, *Theory of Orbits: The Restricted Problem of Three Bodies*. Academic Press, 1967.
- S. H. Strogatz, *Nonlinear Dynamics and Chaos*.
- R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.
- Full reference list: [`paper/references.bib`](paper/references.bib).

## Citing

If you use this code or its results, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

## License

MIT © 2026 Emir Eymen Sargın
