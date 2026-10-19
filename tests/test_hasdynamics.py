"""Tests for ``hasdynamics``: kept by the mass and motor inertia setters, and
required by the dynamics methods."""

import numpy as np
import pytest

import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH


def _bare_dh(n=2):
    return DHRobot([RevoluteDH(a=1.0) for _ in range(n)], name="bare")


@pytest.mark.parametrize("model", ["Panda", "Puma560", "UR5"])
def test_dh_models_have_dynamics(model):
    assert getattr(rtb.models.DH, model)().hasdynamics


@pytest.mark.parametrize("model", [rtb.models.URDF.Panda, rtb.models.ETS.Puma560])
def test_models_without_dynamics(model):
    assert not model().hasdynamics


def test_hand_built_dh_has_no_dynamics():
    # kinematic setters must not set the flag
    robot = _bare_dh()
    robot[0].a = 2.0
    assert not robot.hasdynamics
    assert not robot[0].hasdynamics


def test_zero_mass_is_not_dynamics():
    robot = _bare_dh()
    robot[0].m = 0.0
    assert not robot.hasdynamics


def test_set_then_reset_mass():
    robot = _bare_dh()
    robot[1].m = 2.0
    assert robot[1].hasdynamics
    assert robot.hasdynamics
    robot[1].m = 0.0
    assert not robot[1].hasdynamics
    assert not robot.hasdynamics


def test_motor_inertia_only():
    robot = _bare_dh()
    robot[0].Jm = 1e-4
    assert robot.hasdynamics
    robot[0].Jm = 0.0
    assert not robot.hasdynamics


def test_link_shared_by_two_robots():
    link = RevoluteDH(a=1.0)
    r1 = DHRobot([link], name="r1")
    r2 = DHRobot([link], name="r2")
    assert not r1.hasdynamics and not r2.hasdynamics
    link.m = 1.0
    assert r1.hasdynamics and r2.hasdynamics


def test_hasdynamics_is_read_only():
    robot = _bare_dh()
    with pytest.raises(AttributeError):
        robot.hasdynamics = True
    with pytest.raises(AttributeError):
        robot[0].hasdynamics = True


@pytest.mark.parametrize("name", ["r", "I", "Tc"])
def test_array_views_are_read_only(name):
    link = RevoluteDH(a=1.0, m=1.0)
    arr = getattr(link, name)
    with pytest.raises(ValueError):
        arr[0] = 1.0


@pytest.mark.parametrize("name, value", [("r", [1, 2, 3]), ("Tc", [0.1, -0.1])])
def test_setters_copy_the_callers_array(name, value):
    link = RevoluteDH(a=1.0, m=1.0)
    v = np.array(value, dtype=float)
    setattr(link, name, v)
    v[:] = 99
    assert not np.any(getattr(link, name) == 99)


def test_setting_whole_array_works():
    link = RevoluteDH(a=1.0, m=1.0)
    link.r = [0, 5, 0]
    np.testing.assert_array_equal(link.r, [0, 5, 0])


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.gravload(np.zeros(r.n)),
        lambda r: r.inertia(np.zeros(r.n)),
        lambda r: r.coriolis(np.zeros(r.n), np.zeros(r.n)),
        lambda r: r.accel(np.zeros(r.n), np.zeros(r.n), np.zeros(r.n)),
        lambda r: r.rne(np.zeros(r.n), np.zeros(r.n), np.zeros(r.n)),
        lambda r: r.itorque(np.zeros(r.n), np.zeros(r.n)),
    ],
)
def test_dynamics_methods_raise(call):
    robot = _bare_dh()
    with pytest.raises(ValueError, match="bare"):
        call(robot)


def test_dynamics_methods_work_with_parameters():
    robot = rtb.models.DH.Puma560()
    assert robot.gravload(robot.qn).shape == (6,)
    assert robot.inertia(robot.qn).shape == (6, 6)


def test_friction_needs_no_mass():
    robot = _bare_dh()
    robot[0].B = 0.1
    robot.friction(np.ones(robot.n))


def test_copy_is_independent():
    robot = _bare_dh()
    other = robot.copy()
    robot[0].m = 1.0
    assert robot.hasdynamics
    assert not other.hasdynamics


def test_default_gear_ratio_is_one():
    assert RevoluteDH(a=1.0).G == 1.0


def test_motor_inertia_and_friction_count_without_explicit_gear_ratio():
    link = RevoluteDH(a=1.0, Jm=1e-3, B=0.5, Tc=[0.2, -0.2])
    assert link.friction(1.0) == pytest.approx(-(0.5 + 0.2))

    robot = DHRobot([RevoluteDH(a=1.0, m=1.0, r=[0.5, 0, 0], Jm=0.0)], name="one")
    with_jm = DHRobot([RevoluteDH(a=1.0, m=1.0, r=[0.5, 0, 0], Jm=1.0)], name="one")
    q = np.zeros(1)
    assert with_jm.inertia(q)[0, 0] == pytest.approx(robot.inertia(q)[0, 0] + 1.0)
