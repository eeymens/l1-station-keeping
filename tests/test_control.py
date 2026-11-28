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


# ------------------------------------------------------------ phase 9: Harmonic-LQR
def test_harmonic_lqr_without_harmonics_is_the_bias_lqr():
    from cr3bp.control import (BiasLQRController, HarmonicLQRController, StationKeepingSim,
                               run_episode)
    from cr3bp.evaluation import perturbation_for
    p = perturbation_for(1.0, 5)
    a = run_episode(StationKeepingSim(seed=5, perturbation=p), BiasLQRController(rho=3.0), days=60)
    b = run_episode(StationKeepingSim(seed=5, perturbation=perturbation_for(1.0, 5)),
                    HarmonicLQRController(rho=3.0, harmonics=()), days=60)
    assert np.allclose(a["dist_km"], b["dist_km"], rtol=1e-9, atol=1e-9)
    assert np.isclose(a["dv_total_mps"], b["dv_total_mps"], rtol=1e-9)


def test_harmonic_lqr_tracks_the_moon_and_bias_lqr_does_not():
    """Estimated vs true lunar acceleration over a year (after 60 days)."""
    from cr3bp.control import BiasLQRController, HarmonicLQRController, StationKeepingSim
    from cr3bp.evaluation import perturbation_for
    from cr3bp.perturbations import MOON_RATE, moon_tidal_accel
    err = {}
    for cls in (BiasLQRController, HarmonicLQRController):
        c, p = cls(rho=1.0), perturbation_for(0.0, 101, moon=1.0)
        sim = StationKeepingSim(seed=101, perturbation=p)
        obs, e, tru = sim.reset((0.0, 0.0)), [], []
        c.reset()
        for _ in range(365):
            u = c(obs)
            tru.append(np.array(moon_tidal_accel(sim.state[0] - (1 - sim.mu), sim.state[1],
                                                 sim.mu, p.moon_phase0 + MOON_RATE * sim.t)))
            d = c.xa[4:6] if cls is BiasLQRController else c.Cd @ c.xa[4:]
            e.append(d - tru[-1])
            obs, _ = sim.step(u)
        e, tru = np.array(e)[60:], np.array(tru)[60:]
        err[cls.__name__] = np.sqrt((e ** 2).sum(1).mean() / (tru ** 2).sum(1).mean())
    assert err["HarmonicLQRController"] < 0.4 < 0.9 < err["BiasLQRController"]


# ------------------------------------------------- K3 fix: the filter Riccati equation
def _iterate_filter_riccati(c, scale, steps):
    """Reference: run the Kalman covariance recursion to convergence, in scaled units."""
    from cr3bp.control.station_keeping import NoiseModel as NM
    s = np.asarray(scale)
    T, Ti = np.diag(s), np.diag(1 / s)
    F, H = Ti @ c.Fa @ T, c.H @ T
    n, nd = 4, c.Fa.shape[0] - 4
    noise = NM()
    sig = noise.disturbance_mps2 * MPS2
    Qn = np.zeros((n + nd, n + nd))
    Qn[:n, :n] = c.Bd @ c.Bd.T * sig**2
    Qn[n:, n:] = np.eye(nd) * (c.bias_walk_mps2 * MPS2) ** 2
    Qn = Ti @ (Qn + np.eye(n + nd) * 1e-30) @ Ti
    Rn = np.diag([noise.pos_km * KM] * 2 + [noise.vel_mps * MPS] * 2) ** 2
    P = np.eye(n + nd)
    for _ in range(steps):
        G = F @ P @ H.T
        P = F @ P @ F.T - G @ np.linalg.solve(H @ P @ H.T + Rn, G.T) + Qn
        P = 0.5 * (P + P.T)
    return T @ P @ T


@pytest.mark.parametrize("bias_walk", [1e-11, 6e-11])
def test_legacy_harmonic_riccati_failure_is_explicit(bias_walk):
    """harmonic.py used to raise a bare scipy ValueError here (audit K3 / harmonic.py 89)."""
    from cr3bp.control import HarmonicLQRController, RiccatiError
    with pytest.raises(RiccatiError, match="riccati='balanced'"):
        HarmonicLQRController(rho=1.0, bias_walk_mps2=bias_walk)


@pytest.mark.parametrize("bias_walk", [1e-11, 6e-11, 2e-10, 1e-8])
def test_balanced_riccati_solves_small_bias_walk(bias_walk):
    """The balanced solve agrees with the converged covariance recursion and gives a
    stable filter, also where the legacy solve fails."""
    from cr3bp.control import HarmonicLQRController
    from cr3bp.control.station_keeping import MPS as V
    c = HarmonicLQRController(rho=1.0, bias_walk_mps2=bias_walk, riccati="balanced")
    assert c.riccati_info["residual"] < 1e-8
    assert c.riccati_info["filter_radius"] < 1.0
    scale = np.r_[[KM] * 2, [1e-3 * V] * 2, [1e-9 * MPS2] * (c.Fa.shape[0] - 4)]
    # the slowest filter mode decays like radius^k: iterate long enough to converge
    steps = int(40 / (1 - c.riccati_info["filter_radius"]))
    P_ref = _iterate_filter_riccati(c, scale, steps)
    H = c.H
    L_ref = P_ref @ H.T @ np.linalg.inv(H @ P_ref @ H.T + np.diag(
        [NoiseModel().pos_km * KM] * 2 + [NoiseModel().vel_mps * MPS] * 2) ** 2)
    Ti, D = np.diag(1 / scale), np.diag([KM] * 2 + [MPS] * 2)
    a, b = Ti @ c.L @ D, Ti @ L_ref @ D                  # gain in km, mm/s, 1e-9 m/s^2 units
    assert np.abs(a - b).max() < 1e-7 * np.abs(b).max()


def test_legacy_riccati_is_still_the_default_and_unchanged():
    """Committed results (and the RES-PPO-M base) must reproduce bit for bit."""
    from scipy.linalg import solve_discrete_are
    from cr3bp.control import BiasLQRController, HarmonicLQRController
    for cls in (BiasLQRController, HarmonicLQRController):
        c = cls(rho=3.0, r_ref_km=100.0)
        assert c.riccati_info["method"] == "legacy"
        n = c.Fa.shape[0]
        sig = NoiseModel().disturbance_mps2 * MPS2
        Qn = np.zeros((n, n))
        Qn[:4, :4] = c.Bd @ c.Bd.T * sig**2
        Qn[4:, 4:] = np.eye(n - 4) * (2e-10 * MPS2) ** 2
        Qn += np.eye(n) * 1e-30
        Rn = np.diag([NoiseModel().pos_km * KM] * 2 + [NoiseModel().vel_mps * MPS] * 2) ** 2
        P = solve_discrete_are(c.Fa.T, c.H.T, Qn, Rn)
        assert np.array_equal(c.L, P @ c.H.T @ np.linalg.inv(c.H @ P @ c.H.T + Rn))


def test_balanced_harmonic_lqr_holds_station_with_small_bias_walk():
    from cr3bp.control import HarmonicLQRController
    from cr3bp.evaluation import perturbation_for
    sim = StationKeepingSim(seed=7, perturbation=perturbation_for(1.0, 7, moon=1.0))
    r = run_episode(sim, HarmonicLQRController(rho=1.0, bias_walk_mps2=1e-11,
                                               riccati="balanced"), days=120, offset_km=(0, 0))
    assert r["survived"] and r["mean_dist_km"] < 1000 and np.isfinite(r["dv_total_mps"])


def test_unknown_riccati_method_is_rejected():
    from cr3bp.control import BiasLQRController
    with pytest.raises(ValueError):
        BiasLQRController(riccati="schur")


def test_k3_sweep_logs_unsolvable_designs_instead_of_skipping_them(capsys):
    """scripts/21: a design whose filter cannot be solved becomes a logged failure row."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "k3", Path(__file__).resolve().parents[1] / "scripts" / "21_k3_tuning_sweep.py")
    k3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k3)
    bad = k3.design("Harmonic-LQR", 1000.0, 1.0, "legacy", bias_walk=1e-11)
    good = k3.design("Harmonic-LQR", 1000.0, 1.0, "balanced", bias_walk=1e-11)
    res = [k3.fly_design((d, range(300, 301))) for d in (bad, good)]
    assert res[0][0]["status"] == "riccati_failed" and res[0][1] is None
    assert "DARE" in res[0][0]["error"]
    assert res[1][0]["status"] == "ok" and res[1][1].shape == (1, 3)
    ok, failed = k3.split_failures([m for m, _ in res])
    assert len(ok) == 1 and len(failed) == 1
    assert "RICCATI FAILURE" in capsys.readouterr().out
    # and the frontier maths: an agent below a two-point envelope
    dv, dist = np.array([2.0, 1.0]), np.array([10.0, 100.0])
    c = k3.compare(1.4, 31.6227766, dv, dist, np.array(["LQR", "LQR"]))
    assert np.isclose(c["envelope_ratio"], 1.4 / np.sqrt(2.0)) and c["dominated"] == 0.0
