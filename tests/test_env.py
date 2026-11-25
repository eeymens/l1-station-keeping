"""Tests for the RL environment and the fast integrator inside the simulator."""

import math

import numpy as np
import pytest

from cr3bp import MU_SUN_EARTH, lagrange_points
from cr3bp.control import KM, MPS2, NoiseModel, StationKeepingSim, planar_rhs, run_episode
from cr3bp.control.station_keeping import _rk4_planar

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

from cr3bp.envs import THRUST_LEVELS_MPS2, L1StationKeepingEnv, PolicyController  # noqa: E402

MU = MU_SUN_EARTH


def test_fast_rk4_matches_reference_rk4():
    s = np.r_[lagrange_points(MU)["L1"][:2] + [300 * KM, -100 * KM], 1e-6, -2e-6]
    u = np.array([3e-7, 1e-7]) * MPS2
    h, n = 1e-3, 50
    ref = s.copy()
    for _ in range(n):
        k1 = planar_rhs(ref, MU, u)
        k2 = planar_rhs(ref + h / 2 * k1, MU, u)
        k3 = planar_rhs(ref + h / 2 * k2, MU, u)
        k4 = planar_rhs(ref + h * k3, MU, u)
        ref = ref + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    fast = np.array(_rk4_planar(*s, MU, *u, h, n))
    assert np.allclose(fast, ref, rtol=0, atol=1e-15)


@pytest.mark.parametrize("action_type", ["discrete", "continuous"])
def test_gymnasium_api(action_type):
    check_env(L1StationKeepingEnv(action_type=action_type), skip_render_check=True)


def test_discrete_action_table():
    env = L1StationKeepingEnv()
    assert env.action_space.n == 1 + 8 * len(THRUST_LEVELS_MPS2)
    assert np.all(env.action_to_accel(0) == 0)
    mags = [np.hypot(*env.action_to_accel(a)) / MPS2 for a in range(1, env.action_space.n)]
    assert sorted(set(np.round(mags, 15))) == pytest.approx(sorted(THRUST_LEVELS_MPS2))


def test_continuous_action_is_norm_clipped():
    env = L1StationKeepingEnv(action_type="continuous")
    u = env.action_to_accel(np.array([1.0, 1.0]))
    assert math.hypot(*u) == pytest.approx(env.u_max)


def test_reset_with_fixed_offset():
    env = L1StationKeepingEnv(noise=NoiseModel.none())
    obs, _ = env.reset(seed=0, options={"offset_km": (500.0, 0.0)})
    assert obs.shape == (8,) and obs.dtype == np.float32
    assert env.sim.distance_km == pytest.approx(500.0)


def test_coasting_costs_no_fuel_and_episode_truncates():
    env = L1StationKeepingEnv(noise=NoiseModel.none(), episode_days=5)
    env.reset(seed=0, options={"offset_km": (0.0, 0.0)})
    for i in range(5):
        _, reward, terminated, truncated, info = env.step(0)
        assert info["dv_mps"] == 0.0 and reward <= 1.0
        assert not terminated
    assert truncated


def test_losing_the_spacecraft_terminates_with_penalty():
    env = L1StationKeepingEnv(noise=NoiseModel.none())
    env.reset(seed=0, options={"offset_km": (5000.0, 0.0)})
    terminated, reward = False, 0.0
    for _ in range(365):
        _, reward, terminated, _, _ = env.step(0)
        if terminated:
            break
    assert terminated and reward < -env.fail_penalty / 2


def test_policy_controller_runs_in_evaluation_loop():
    env = L1StationKeepingEnv()
    coast = PolicyController(env, lambda features: 0)
    res = run_episode(StationKeepingSim(noise=NoiseModel.none()), coast)
    assert not res["survived"] and res["dv_total_mps"] == 0.0


@pytest.mark.parametrize("power", [1.0, 3.0])
def test_continuous_action_power_map(power):
    env = L1StationKeepingEnv(action_type="continuous", action_power=power)
    for a in ([0.1, 0.0], [0.0, -0.5], [0.3, 0.4]):
        u = env.action_to_accel(np.array(a))
        n = math.hypot(*a)
        assert math.hypot(*u) == pytest.approx(env.u_max * n**power)
        assert u @ np.array(a) == pytest.approx(math.hypot(*u) * n)   # same direction
    assert np.all(env.action_to_accel(np.zeros(2)) == 0)
    # p = 3 reaches the ~2e-9 m/s^2 holding thrust with a moderate action
    if power == 3.0:
        a = (2e-9 / 1e-6) ** (1 / 3)
        assert math.hypot(*env.action_to_accel(np.array([a, 0.0]))) / MPS2 == pytest.approx(2e-9)


def test_evaluation_reports_tracking_error():
    from cr3bp.control import LQRController
    from cr3bp.evaluation import evaluate, summarize
    rows = evaluate(LQRController(rho=3.0), seeds=range(1), days=30)
    for r in rows:
        assert r["max_dist_km"] >= r["rms_dist_km"] >= r["mean_dist_km"] > 0
    s = summarize(rows)
    assert set(s) == {"holding", "acquisition"} and "rms_dist_km" in s["holding"]


def test_saved_policy_round_trip(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from cr3bp.training import load_policy, save_config

    kw = {"action_type": "continuous", "action_power": 3.0}
    model = PPO("MlpPolicy", L1StationKeepingEnv(**kw), n_steps=64, batch_size=32, seed=0)
    model.save(tmp_path / "ppo_test")
    save_config(tmp_path / "ppo_test", "ppo", kw, {})
    env, ctrl = load_policy(tmp_path, "ppo_test")
    assert env.action_type == "continuous" and env.action_power == 3.0
    obs = StationKeepingSim(noise=NoiseModel.none()).reset((100.0, 0.0))
    u = ctrl(obs)
    assert u.shape == (2,) and math.hypot(*u) <= env.u_max * (1 + 1e-12)


# ------------------------------------------------------------ phase 8
@pytest.mark.parametrize("kw", [{"history_len": 10}, {"residual": {"rho": 1.0}},
                                {"history_len": 4, "residual": {"rho": 1.0}}])
def test_phase8_envs_pass_the_gymnasium_checker(kw):
    check_env(L1StationKeepingEnv(action_type="continuous", dr_k_max=4.0, **kw),
              skip_render_check=True)


@pytest.mark.parametrize("kw", [{"history_len": 10}, {"residual": {"rho": 1.0}}])
def test_policy_controller_sees_what_the_env_shows(kw):
    """Flying a fixed action sequence through the env and through
    PolicyController + run_episode must give identical observations."""
    env = L1StationKeepingEnv(action_type="continuous", **kw)
    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.3, 0.3, (30, 2))
    obs, _ = env.reset(seed=11, options={"offset_km": (200.0, 0.0)})
    env_obs = [obs]
    for a in actions:
        obs, *_ = env.step(a)
        env_obs.append(obs)
    seen = []

    def predict(f):
        seen.append(f.copy())
        return actions[len(seen) - 1]

    env2 = L1StationKeepingEnv(action_type="continuous", **kw)
    sim = StationKeepingSim()
    sim.rng = np.random.default_rng(11)        # same draws as env.reset(seed=11)
    run_episode(sim, PolicyController(env2, predict), days=30, offset_km=(200.0, 0.0))
    for a, b in zip(env_obs[:30], seen):
        assert np.allclose(a, b, rtol=0, atol=1e-6)


def test_history_observation_layout():
    env = L1StationKeepingEnv(action_type="continuous", history_len=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8 * 3 + 2 * 3,)
    assert np.array_equal(obs[:8], obs[8:16])          # first step repeats the estimate
    obs, *_ = env.step(np.array([0.5, -0.25]))
    assert np.allclose(obs[-2:], [0.5, -0.25])          # newest action last


def test_residual_zero_action_is_the_bias_lqr():
    from cr3bp.control import BiasLQRController
    env = L1StationKeepingEnv(action_type="continuous", residual={"rho": 1.0})
    ctrl = PolicyController(env, lambda f: np.zeros(2))
    a = run_episode(StationKeepingSim(seed=4), ctrl, days=100)
    b = run_episode(StationKeepingSim(seed=4), BiasLQRController(rho=1.0, u_max_mps2=1e-6), days=100)
    assert np.allclose(a["dist_km"], b["dist_km"], rtol=1e-12)


# ------------------------------------------------------------ phase 9
def _lstm_policy(tmp_path):
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import RecurrentPPO
    from cr3bp.training import load_policy, save_config

    kw = {"action_type": "continuous", "action_power": 3.0, "dr_k_max": 4.0}
    model = RecurrentPPO("MlpLstmPolicy", L1StationKeepingEnv(**kw), n_steps=64, batch_size=32,
                         seed=0, policy_kwargs={"lstm_hidden_size": 8, "log_std_init": 1.0})
    model.save(tmp_path / "lstm_test")
    save_config(tmp_path / "lstm_test", "rppo", kw, {})
    return load_policy(tmp_path, "lstm_test")


def test_lstm_policy_round_trip(tmp_path):
    env, ctrl = _lstm_policy(tmp_path)
    obs = StationKeepingSim(noise=NoiseModel.none()).reset((100.0, 0.0))
    ctrl.reset()
    u = ctrl(obs)
    assert u.shape == (2,) and math.hypot(*u) <= env.u_max * (1 + 1e-12)


def test_lstm_state_carries_over_and_resets(tmp_path):
    """The LSTM must remember within an episode and forget between episodes,
    so two evaluations of the same seeds give identical results."""
    env, ctrl = _lstm_policy(tmp_path)
    f = env.features(StationKeepingSim(noise=NoiseModel.none()).reset((100.0, 0.0)))
    pred = ctrl.predict                                     # the RecurrentPredictor
    ctrl.reset()
    first = [pred(f) for _ in range(3)]
    assert not np.array_equal(first[0], first[2])          # same input, new memory
    ctrl.reset()
    assert np.array_equal(pred(f), first[0])               # reset forgets
    from cr3bp.evaluation import evaluate
    a = evaluate(ctrl, seeds=range(2), days=30, perturbation_k=1.0)
    b = evaluate(ctrl, seeds=range(2), days=30, perturbation_k=1.0)
    assert [r["dv_mps"] for r in a] == [r["dv_mps"] for r in b]


HARM = {"rho": 1.0, "base": "harmonic"}


@pytest.mark.parametrize("kw", [{"dr_moon_max": 2.0}, {"residual": HARM},
                                {"residual": HARM, "dr_moon_max": 2.0}])
def test_moon_envs_pass_the_gymnasium_checker(kw):
    check_env(L1StationKeepingEnv(action_type="continuous", dr_k_max=4.0, **kw),
              skip_render_check=True)


def test_moon_randomisation_leaves_the_phase7_draws_alone():
    a = L1StationKeepingEnv(action_type="continuous", dr_k_max=4.0)
    b = L1StationKeepingEnv(action_type="continuous", dr_k_max=4.0, dr_moon_max=2.0)
    a.reset(seed=3)
    b.reset(seed=3)
    pa, pb = a.sim.perturbation, b.sim.perturbation
    assert (pa.e, pa.srp_mps2, pa.f0) == (pb.e, pb.srp_mps2, pb.f0)
    assert pa.moon == 0.0 and 0.0 < pb.moon < 2.0


def test_residual_zero_action_is_the_harmonic_lqr():
    from cr3bp.control import HarmonicLQRController
    from cr3bp.evaluation import perturbation_for
    env = L1StationKeepingEnv(action_type="continuous", residual=dict(HARM))
    ctrl = PolicyController(env, lambda f: np.zeros(2))
    p = perturbation_for(1.0, 4, moon=1.0)
    a = run_episode(StationKeepingSim(seed=4, perturbation=p), ctrl, days=100)
    b = run_episode(StationKeepingSim(seed=4, perturbation=perturbation_for(1.0, 4, moon=1.0)),
                    HarmonicLQRController(rho=1.0, u_max_mps2=1e-6), days=100)
    assert np.allclose(a["dist_km"], b["dist_km"], rtol=1e-12)
