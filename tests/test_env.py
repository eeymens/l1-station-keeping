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
