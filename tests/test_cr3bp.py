"""Correctness tests for the CR3BP toolkit. Run with:  pytest"""

from functools import partial

import numpy as np
import pytest

from cr3bp import (AU_KM, MU_ROUTH, MU_SUN_EARTH, dop853, grad_potential,
                   jacobi_constant, lagrange_points, rhs, rk4, stability)

MU = MU_SUN_EARTH


@pytest.mark.parametrize("name", ["L1", "L2", "L3", "L4", "L5"])
def test_lagrange_points_are_equilibria(name):
    p = lagrange_points(MU)[name]
    assert np.allclose(grad_potential(*p, MU), 0.0, atol=1e-12)
    # zero velocity at an equilibrium -> zero acceleration
    assert np.allclose(rhs(0.0, np.r_[p, 0, 0, 0], MU), 0.0, atol=1e-12)


def test_l4_forms_equilateral_triangle():
    p = lagrange_points(MU)["L4"]
    r_sun = np.linalg.norm(p - [-MU, 0, 0])
    r_earth = np.linalg.norm(p - [1 - MU, 0, 0])
    assert r_sun == pytest.approx(1.0, abs=1e-12)
    assert r_earth == pytest.approx(1.0, abs=1e-12)


def test_l1_distance_matches_known_value():
    # Sun-Earth L1 is about 1.49-1.5 million km sunward of Earth
    p = lagrange_points(MU)["L1"]
    d_km = (1 - MU - p[0]) * AU_KM
    assert 1.48e6 < d_km < 1.51e6


def test_collinear_points_are_unstable():
    for name in ("L1", "L2"):
        st = stability(lagrange_points(MU)[name], MU)
        assert st["growth_rate"] > 2.0          # saddle: fast exponential escape


def test_routh_criterion_for_l4():
    # Sun-Earth: mu << MU_ROUTH, so L4 is linearly stable
    assert stability(lagrange_points(MU)["L4"], MU)["growth_rate"] == 0.0
    # Above the Routh limit, L4 becomes unstable
    mu_big = 0.1
    assert mu_big > MU_ROUTH
    assert stability(lagrange_points(mu_big)["L4"], mu_big)["growth_rate"] > 0.0


def test_jacobi_constant_conserved_dop853():
    s0 = np.r_[lagrange_points(MU)["L4"] + [0.01, 0, 0], 0, 0, 0]
    _, S = dop853(partial(rhs, mu=MU), s0, 10 * 2 * np.pi)
    c = jacobi_constant(S, MU)
    assert np.max(np.abs(c - c[0])) / abs(c[0]) < 1e-10


def test_rk4_is_fourth_order():
    f = partial(rhs, mu=MU)
    s0 = np.r_[lagrange_points(MU)["L4"] + [0.01, 0, 0], 0, 0, 0]
    T = 2 * np.pi
    _, S_ref = dop853(f, s0, T, rtol=1e-13, atol=1e-13)
    steps = [0.08, 0.04, 0.02]
    errs = [np.linalg.norm(rk4(f, s0, T, h)[1][-1] - S_ref[-1]) for h in steps]
    order = np.polyfit(np.log(steps), np.log(errs), 1)[0]
    assert 3.7 < order < 4.3


def test_control_input_adds_acceleration():
    s = np.r_[lagrange_points(MU)["L1"], 0, 0, 0]
    u = np.array([1e-3, -2e-3, 5e-4])
    assert np.allclose(rhs(0.0, s, MU, u)[3:] - rhs(0.0, s, MU)[3:], u)
