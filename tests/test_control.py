"""Tests for the station-keeping simulator and the LQR controller."""

import numpy as np
import pytest
from scipy.linalg import expm

from cr3bp import MU_SUN_EARTH, lagrange_points, linearize
from cr3bp.control import (B_PLANAR, KM, MPS, MPS2, LQRController, NoiseModel,
                           StationKeepingSim, discretize, run_episode)

MU = MU_SUN_EARTH


def test_zoh_discretization():
    A = linearize(lagrange_points(MU)["L1"], MU)
    dt = 0.05
    Ad, Bd = discretize(A, B_PLANAR, dt)
    assert np.allclose(Ad, expm(A * dt))
    # Bd = integral_0^dt expm(A s) ds @ B, checked with the trapezoid rule
    s = np.linspace(0, dt, 2001)
    integral = np.trapezoid(np.array([expm(A * si) for si in s]), s, axis=0)
    assert np.allclose(Bd, integral @ B_PLANAR, rtol=1e-6, atol=1e-12)


def test_simulator_matches_linear_model_near_l1():
    sim = StationKeepingSim(noise=NoiseModel.none())
    ctrl = LQRController()
    x0 = sim.reset(offset_km=(50.0, -20.0), vel_mps=(0.01, 0.0))
    u = np.array([2e-7, -1e-7]) * MPS2
    x1, _ = sim.step(u)
    x1_lin = ctrl.Ad @ x0 + ctrl.Bd @ u
    assert np.allclose(x1, x1_lin, rtol=1e-3, atol=1e-3 * np.abs(x1).max())


@pytest.mark.parametrize("rho", [1e-3, 1.0, 1e3])
def test_closed_loop_is_stable(rho):
    ctrl = LQRController(rho=rho)
    assert np.abs(ctrl.closed_loop_eigenvalues).max() < 1.0
    # ...while the open loop is not
    assert np.abs(np.linalg.eigvals(ctrl.Ad)).max() > 1.0


def test_uncontrolled_spacecraft_is_lost():
    res = run_episode(StationKeepingSim(noise=NoiseModel.none()), None)
    assert not res["survived"]
    assert res["days_survived"] < 90


def test_disturbances_push_a_coasting_spacecraft_off_l1():
    # Exactly at L1 and at rest, only the unmodelled forces can move it
    res = run_episode(StationKeepingSim(seed=0), None, offset_km=(0.0, 0.0))
    assert not res["survived"]


def test_lqr_recovers_from_1000_km_without_noise():
    res = run_episode(StationKeepingSim(noise=NoiseModel.none()), LQRController())
    assert res["survived"]
    assert res["final_dist_km"] < 1.0


def test_lqr_holds_station_with_noise():
    res = run_episode(StationKeepingSim(seed=1), LQRController(), offset_km=(0.0, 0.0))
    assert res["survived"]
    assert res["dist_km"].max() < 20.0


def test_delta_v_accounting():
    sim = StationKeepingSim(noise=NoiseModel.none())
    sim.reset(offset_km=(0.0, 0.0))
    u = np.array([3e-7, 4e-7]) * MPS2          # |u| = 5e-7 m/s^2
    for _ in range(10):
        sim.step(u)
    expected_mps = 5e-7 * 10 * 86_400.0          # |u| * 10 days
    assert sim.dv / MPS == pytest.approx(expected_mps, rel=1e-9)


def test_thrust_saturation():
    ctrl = LQRController(rho=1e-4, u_max_mps2=1e-7)
    u = ctrl(np.array([5000 * KM, 0.0, 0.0, 0.0]))
    assert np.hypot(*u) <= 1e-7 * MPS2 * (1 + 1e-12)
