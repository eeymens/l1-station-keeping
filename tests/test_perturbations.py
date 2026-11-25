"""Phase 7 tests: elliptic Earth orbit (ER3BP) and solar radiation pressure."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from cr3bp import AU_KM, MU_SUN_EARTH, lagrange_points
from cr3bp.control import KM, MPS, MPS2, LQRController, StationKeepingSim, run_episode
from cr3bp.control.station_keeping import _rk4_planar
from cr3bp.perturbations import (E_EARTH, SRP_DEFAULT_MPS2, Perturbation, mean_from_true,
                                 radius, rk4_er3bp, true_from_mean)

MU = MU_SUN_EARTH
L1 = lagrange_points(MU)["L1"]


# ------------------------------------------------------------ e = 0 regression
@pytest.mark.parametrize("pert", [None, Perturbation(), Perturbation.from_k(0.0, f0=1.3)])
def test_null_perturbation_reproduces_cr3bp_bit_for_bit(pert):
    ref = run_episode(StationKeepingSim(seed=7), LQRController(), days=120)
    out = run_episode(StationKeepingSim(seed=7, perturbation=pert), LQRController(), days=120)
    for key in ("dist_km", "dv_mps", "pos_km"):
        assert np.array_equal(ref[key], out[key])


def test_er3bp_integrator_reduces_to_cr3bp_when_e_is_zero():
    s = (L1[0] + 800 * KM, -300 * KM, 0.02 * MPS, -0.01 * MPS)
    u = (3e-7 * MPS2, -2e-7 * MPS2)
    a = _rk4_planar(*s, MU, *u, 2e-3, 50)
    b = rk4_er3bp(*s, MU, 0.0, 0.0, *u, 0.0, 0.1, 50)
    assert np.allclose(a, b, rtol=0, atol=1e-15)


# ------------------------------------------------------------ Kepler helpers
@pytest.mark.parametrize("e", [0.0, E_EARTH, 0.2])
def test_kepler_round_trip(e):
    for f in np.linspace(-3.0, 9.0, 25):
        M = mean_from_true(f, e)
        f2 = true_from_mean(M, e)
        assert math.cos(f2) == pytest.approx(math.cos(f), abs=1e-12)
        assert math.sin(f2) == pytest.approx(math.sin(f), abs=1e-12)


def test_true_anomaly_is_continuous_over_a_year():
    e = E_EARTH
    fs = [true_from_mean(M, e) for M in np.linspace(0, 4 * math.pi, 400)]
    assert np.all(np.diff(fs) > 0)


# ------------------------------------------------------------ ER3BP physics
def test_l1_is_still_an_equilibrium_in_pulsating_coordinates():
    """The Lagrange points do not move in pulsating coordinates (Szebehely 1967)."""
    x, y, vx, vy = rk4_er3bp(L1[0], 0.0, 0.0, 0.0, MU, E_EARTH, 0.0, 0.0, 0.0,
                             0.4, 0.4 + 0.5, 300)            # ~29 days
    assert abs(x - L1[0]) < 1e-14 and abs(y) < 1e-14
    assert abs(vx) < 1e-13 and abs(vy) < 1e-13


def test_physical_l1_distance_pulsates_with_earths_orbit():
    d = lambda f: (1 - MU - L1[0]) * radius(f, E_EARTH) * AU_KM  # noqa: E731
    assert d(0.0) / d(math.pi) == pytest.approx((1 - E_EARTH) / (1 + E_EARTH), rel=1e-12)
    assert d(math.pi) - d(0.0) == pytest.approx(2 * E_EARTH * 1.4917e6, rel=2e-3)


def _pulsating_to_inertial(f, s, e):
    """Pulsating state [x, y, x', y'] (' = d/df) -> inertial barycentric [X, Y, VX, VY]."""
    r = radius(f, e)
    fdot = math.sqrt(1 - e * e) / r ** 2
    rdot = r * e * math.sin(f) / (1 + e * math.cos(f)) * fdot
    c, sn = math.cos(f), math.sin(f)
    R = np.array([[c, -sn], [sn, c]])
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    xi, dxi = np.asarray(s[:2]), np.asarray(s[2:])
    pos = r * R @ xi
    vel = rdot * R @ xi + r * fdot * (R @ J @ xi + R @ dxi)
    return np.r_[pos, vel]


@pytest.mark.parametrize("e", [E_EARTH, 0.1])
def test_matches_inertial_integration(e):
    """Integrate the same spacecraft (with thrust and SRP) by brute force in an
    inertial frame, with the Sun and Earth on Kepler ellipses, and compare."""
    a0_mps2 = 10 * SRP_DEFAULT_MPS2
    pert = Perturbation(e=e, srp_mps2=a0_mps2)
    u = np.array([4e-7, -3e-7]) * MPS2                  # thrust, rotating-frame axes
    f0 = 0.7
    s0 = (L1[0] + 2000 * KM, 1500 * KM, 0.05 * MPS, -0.03 * MPS)
    M0 = mean_from_true(f0, e)
    t_end = 30 / 58.1309                                 # ~30 days
    f_end = true_from_mean(M0 + t_end, e)

    puls = rk4_er3bp(*s0, MU, e, pert.a0, *u, f0, f_end, 2000)

    def rhs(t, X):
        f = true_from_mean(M0 + t, e)
        r = radius(f, e)
        Rse = r * np.array([math.cos(f), math.sin(f)])  # Sun -> Earth
        sun, earth = -MU * Rse, (1 - MU) * Rse
        d1, d2 = X[:2] - sun, X[:2] - earth
        acc = (-(1 - MU - pert.a0) * d1 / np.linalg.norm(d1) ** 3
               - MU * d2 / np.linalg.norm(d2) ** 3)
        c, sn = math.cos(f), math.sin(f)
        acc = acc + np.array([[c, -sn], [sn, c]]) @ u
        return np.r_[X[2:], acc]

    X0 = _pulsating_to_inertial(f0, s0, e)
    sol = solve_ivp(rhs, (0, t_end), X0, method="DOP853", rtol=1e-12, atol=1e-14)
    X_ref = sol.y[:, -1]
    X_puls = _pulsating_to_inertial(f_end, puls, e)
    assert np.linalg.norm(X_puls[:2] - X_ref[:2]) * AU_KM < 1e-3          # < 1 m
    assert np.linalg.norm(X_puls[2:] - X_ref[2:]) / MPS < 1e-8            # < 10 nm/s


# ------------------------------------------------------------ SRP physics
def test_srp_moves_the_equilibrium_sunward():
    """With SRP the Sun is effectively lighter, so the L1 equilibrium moves
    toward the Sun by ~a0 / Omega_xx (a few hundred km at realistic SRP)."""
    a0 = Perturbation(srp_mps2=SRP_DEFAULT_MPS2).a0

    def dodx(x, a0):
        return x - (1 - MU - a0) * (x + MU) / abs(x + MU) ** 3 - MU * (x - 1 + MU) / abs(x - 1 + MU) ** 3

    x_srp = brentq(dodx, L1[0] - 1e-3, L1[0] + 1e-4, args=(a0,), xtol=1e-16)
    shift_km = (x_srp - L1[0]) * AU_KM
    assert -400 < shift_km < -100
    # the simulator with SRP and no noise stays put at the shifted point
    from cr3bp.control import NoiseModel
    sim = StationKeepingSim(noise=NoiseModel.none(), perturbation=Perturbation(srp_mps2=SRP_DEFAULT_MPS2))
    sim.reset(offset_km=(shift_km, 0.0))
    for _ in range(20):
        sim.step(np.zeros(2))
    assert abs(sim.distance_km - abs(shift_km)) < 0.01


def test_realistic_srp_is_well_below_thrust_limit():
    assert 1e-8 < SRP_DEFAULT_MPS2 < 1e-7


def test_physical_distance_uses_sun_earth_distance():
    from cr3bp.control import NoiseModel
    p = Perturbation(e=E_EARTH, f0=0.0)                 # perihelion
    sim = StationKeepingSim(noise=NoiseModel.none(), perturbation=p)
    sim.reset(offset_km=(1000.0, 0.0))
    assert sim.distance_km == pytest.approx(1000.0 * (1 - E_EARTH), rel=1e-12)


# ------------------------------------------------------------ phase-7 controllers / env
def test_lqi_removes_steady_offset_under_constant_srp():
    from cr3bp.control import LQIController, NoiseModel
    sim = StationKeepingSim(noise=NoiseModel.none(), perturbation=Perturbation(srp_mps2=SRP_DEFAULT_MPS2))
    r = run_episode(sim, LQIController(), days=365, offset_km=(0.0, 0.0))
    assert r["survived"]
    assert r["dist_km"][-30:].max() < 1.0          # integral action: back at L1


def test_bias_lqr_estimates_the_srp_acceleration():
    from cr3bp.control import BiasLQRController
    from cr3bp.control.station_keeping import NoiseModel
    sim = StationKeepingSim(seed=3, perturbation=Perturbation(srp_mps2=SRP_DEFAULT_MPS2))
    ctrl = BiasLQRController(rho=10.0)
    r = run_episode(sim, ctrl, days=365, offset_km=(0.0, 0.0))
    assert r["survived"]
    d = ctrl.bias_estimate_mps2
    # SRP points away from the Sun, i.e. +x in the rotating frame
    assert d[0] == pytest.approx(SRP_DEFAULT_MPS2, rel=0.15)
    assert abs(d[1]) < 0.15 * SRP_DEFAULT_MPS2
    assert NoiseModel  # imported for clarity


def test_stateful_controllers_reset_between_episodes():
    from cr3bp.control import BiasLQRController, LQIController
    for make in (LQIController, BiasLQRController):
        ctrl = make()
        a = run_episode(StationKeepingSim(seed=5), ctrl, days=60)
        b = run_episode(StationKeepingSim(seed=5), ctrl, days=60)
        assert np.array_equal(a["dist_km"], b["dist_km"])


def test_domain_randomised_env_draws_new_physics_each_episode():
    pytest.importorskip("gymnasium")
    from cr3bp.envs import L1StationKeepingEnv
    env = L1StationKeepingEnv(action_type="continuous", dr_k_max=4.0)
    seen = set()
    for i in range(5):
        env.reset(seed=i)
        p = env.sim.perturbation
        assert 0.0 <= p.e <= 4 * E_EARTH and p.srp_mps2 >= 0.0
        seen.add(round(p.e, 9))
        env.step(env.action_space.sample())
    assert len(seen) == 5


def test_nominal_env_is_unchanged():
    pytest.importorskip("gymnasium")
    from cr3bp.envs import L1StationKeepingEnv
    env = L1StationKeepingEnv(action_type="continuous")
    env.reset(seed=0)
    assert env.sim.perturbation is None


# ------------------------------------------------------------ phase 9: the Moon
def _two_bodies_minus_one(X, Y, theta):
    """Independent check of moon_tidal_accel with numpy vectors."""
    from cr3bp.perturbations import EARTH_MOON_AU, Q_MOON
    u = np.array([math.cos(theta), math.sin(theta)])
    p = np.array([X, Y])
    bodies = [((1 - Q_MOON) * MU, -Q_MOON * EARTH_MOON_AU * u),
              (Q_MOON * MU, (1 - Q_MOON) * EARTH_MOON_AU * u)]
    a = sum(-m * (p - b) / np.linalg.norm(p - b) ** 3 for m, b in bodies)
    return a + MU * p / np.linalg.norm(p) ** 3


@pytest.mark.parametrize("theta", [0.0, 0.7, 2.0, math.pi, 4.5])
def test_moon_tidal_accel_matches_point_masses(theta):
    from cr3bp.perturbations import moon_tidal_accel
    X, Y = L1[0] - (1 - MU), 3e-4
    assert np.allclose(moon_tidal_accel(X, Y, MU, theta), _two_bodies_minus_one(X, Y, theta),
                       rtol=1e-9, atol=0.0)


def test_moon_keeps_total_mass_so_far_field_vanishes():
    """Same total mass as the point it replaces: no 1/R^2 term, so far away the
    extra pull falls off at least as fast as 1/R^4 (tidal)."""
    from cr3bp.perturbations import moon_tidal_accel
    a1 = np.hypot(*moon_tidal_accel(0.01, 0.0, MU, 0.3))
    a2 = np.hypot(*moon_tidal_accel(0.02, 0.0, MU, 0.3))
    assert a2 < a1 / 12                                   # ~1/16 for a pure quadrupole


def test_moon_near_l1_is_about_1e7_mps2():
    from cr3bp.constants import ACCEL_UNIT_MPS2
    from cr3bp.perturbations import moon_tidal_accel
    X = L1[0] - (1 - MU)
    a = np.array([moon_tidal_accel(X, 0.0, MU, t)
                  for t in np.linspace(0, 2 * np.pi, 361)]) * ACCEL_UNIT_MPS2
    peak = np.hypot(a[:, 0], a[:, 1]).max()
    assert 1e-7 < peak < 1e-6                             # below the 1e-6 thrust limit


def test_moon_off_is_the_old_code_path_and_moon_on_changes_the_orbit():
    from cr3bp.control import StationKeepingSim, run_episode
    from cr3bp.evaluation import perturbation_for
    assert perturbation_for(0.0, 3) is None
    p0, p1 = perturbation_for(1.0, 3), perturbation_for(1.0, 3, moon=1.0)
    assert (p0.e, p0.srp_mps2, p0.f0) == (p1.e, p1.srp_mps2, p1.f0) and p1.moon == 1.0
    r = [run_episode(StationKeepingSim(seed=3, perturbation=p), None, days=20)
         for p in (p0, p1)]
    assert r[0]["final_dist_km"] != r[1]["final_dist_km"]


def test_moon_phase_turns_once_per_synodic_month():
    from cr3bp.perturbations import MOON_RATE, SYNODIC_MONTH_DAYS
    from cr3bp.constants import TIME_UNIT_DAYS
    assert math.isclose(MOON_RATE * SYNODIC_MONTH_DAYS / TIME_UNIT_DAYS, 2 * math.pi)


@pytest.mark.parametrize("e", [0.0, E_EARTH])
def test_moon_matches_inertial_integration(e):
    """Same spacecraft, integrated by brute force in an inertial frame with the
    Earth and the Moon as two bodies circling their barycentre (which follows
    the Kepler ellipse), vs. the pulsating integrator stepped day by day as the
    simulator does."""
    from cr3bp.constants import TIME_UNIT_DAYS
    from cr3bp.perturbations import EARTH_MOON_AU, MOON_RATE, Q_MOON
    pert = Perturbation(e=e, srp_mps2=SRP_DEFAULT_MPS2, moon=1.0, moon_phase0=1.1)
    u = np.array([4e-7, -3e-7]) * MPS2
    f0, s0 = 0.7, (L1[0] + 2000 * KM, 1500 * KM, 0.05 * MPS, -0.03 * MPS)
    M0, dt, days = mean_from_true(f0, e), 1.0 / TIME_UNIT_DAYS, 30

    s, f = s0, f0
    for i in range(days):
        f_next = true_from_mean(M0 + (i + 1) * dt, e)
        s = rk4_er3bp(*s, MU, e, pert.a0, *u, f, f_next, 80, pert.moon,
                      pert.moon_phase0 + MOON_RATE * i * dt)
        f = f_next

    def rhs(t, X):
        fa = true_from_mean(M0 + t, e)
        r = radius(fa, e)
        Rse = r * np.array([math.cos(fa), math.sin(fa)])
        sun, emb = -MU * Rse, (1 - MU) * Rse
        th = fa + pert.moon_phase0 + MOON_RATE * t
        um = np.array([math.cos(th), math.sin(th)])
        earth = emb - Q_MOON * EARTH_MOON_AU * um
        moon = emb + (1 - Q_MOON) * EARTH_MOON_AU * um
        acc = -(1 - MU - pert.a0) * (X[:2] - sun) / np.linalg.norm(X[:2] - sun) ** 3
        for m, b in (((1 - Q_MOON) * MU, earth), (Q_MOON * MU, moon)):
            acc = acc - m * (X[:2] - b) / np.linalg.norm(X[:2] - b) ** 3
        c, sn = math.cos(fa), math.sin(fa)
        return np.r_[X[2:], acc + np.array([[c, -sn], [sn, c]]) @ u]

    sol = solve_ivp(rhs, (0, days * dt), _pulsating_to_inertial(f0, s0, e), method="DOP853",
                    rtol=1e-12, atol=1e-14)
    X_puls = _pulsating_to_inertial(f, s, e)
    assert np.linalg.norm(X_puls[:2] - sol.y[:2, -1]) * AU_KM < 1e-3        # < 1 m
    assert np.linalg.norm(X_puls[2:] - sol.y[2:, -1]) / MPS < 1e-8          # < 10 nm/s
