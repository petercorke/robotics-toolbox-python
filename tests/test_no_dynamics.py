"""Dynamics computations must reject models without inertial data clearly."""

import re

import numpy as np
import numpy.testing as npt
import pytest

import roboticstoolbox as rtb


def _call_dynamics(robot: rtb.Robot, method: str) -> object:
    q = robot.q
    velocity = np.ones(robot.n)
    acceleration = np.ones(robot.n)
    arguments = {
        "gravload": (q,),
        "rne": (q, velocity, acceleration),
        "rne_python": (q, velocity, acceleration),
        "inertia": (q,),
        "friction": (velocity,),
        "coriolis": (q, velocity),
        "accel": (q, velocity, acceleration),
        "fdyn": (0.01, q),
        "itorque": (q, acceleration),
        "jointdynamics": (q,),
        "inertia_x": (q,),
        "coriolis_x": (q, velocity),
        "gravload_x": (q,),
        "accel_x": (q, np.ones(6), np.ones(6)),
        "paycap": (np.ones(6), np.tile([100.0, -100.0], (robot.n, 1))),
    }
    return getattr(robot, method)(*arguments[method])


@pytest.mark.parametrize("model", ["Panda", "ETS-Puma560", "DH"])
@pytest.mark.parametrize(
    "method",
    [
        "gravload",
        "rne",
        "inertia",
        "friction",
        "coriolis",
        "accel",
        "fdyn",
        "itorque",
        "jointdynamics",
        "inertia_x",
        "coriolis_x",
        "gravload_x",
        "accel_x",
        "paycap",
    ],
)
def test_dynamics_without_parameters(model: str, method: str) -> None:
    if model == "Panda":
        robot = rtb.models.Panda()
    elif model == "ETS-Puma560":
        robot = rtb.models.ETS.Puma560()
    else:
        robot = rtb.DHRobot([rtb.RevoluteDH()], name="kinematics only")

    message = (
        f"model '{robot.name}' has no dynamic parameters; "
        "use a model with dynamics, such as rtb.models.DH.Panda(), "
        "or set m, r, I on the links"
    )
    assert not robot.has_dynamics
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _call_dynamics(robot, method)


@pytest.mark.parametrize(
    "method",
    [
        "gravload",
        "rne",
        "rne_python",
        "inertia",
        "friction",
        "coriolis",
        "accel",
        "fdyn",
    ],
)
def test_dynamics_with_parameters(method: str) -> None:
    robot = rtb.models.DH.Puma560()
    robot.q = robot.qn
    assert robot.links[0].m == 0
    assert robot.has_dynamics

    result = _call_dynamics(robot, method)
    if method == "fdyn":
        values = result.q
    else:
        values = np.asarray(result)
    assert np.all(np.isfinite(values))
    assert np.any(values != 0)

    npt.assert_allclose(
        robot.gravload(robot.qn), [0, 31.6399, 6.0351, 0, 0.0283, 0], atol=5e-5
    )
    assert np.all(np.linalg.eigvalsh(robot.inertia(robot.qn)) > 0)


def test_explicit_zero_parameters_and_mass_changes() -> None:
    robot = rtb.DHRobot([rtb.PrismaticDH(m=0, r=[0, 0, 0], I=[0, 0, 0])])
    # The legacy flag records supplied parameters, including explicit zeros.
    assert robot._hasdynamics
    assert not robot.has_dynamics
    with pytest.raises(ValueError, match="has no dynamic parameters"):
        robot.rne_python([0], [0], [0])

    robot.links[0].m = 2
    assert robot.has_dynamics
    npt.assert_allclose(robot.gravload([0]), [19.62])
    robot.links[0].m = 0
    assert not robot.has_dynamics
    with pytest.raises(ValueError, match="has no dynamic parameters"):
        robot.gravload([0])


def test_adding_dynamics_to_kinematics_only_model() -> None:
    robot = rtb.Robot([rtb.Link(rtb.ET.tz())])
    assert not robot._hasdynamics
    assert not robot.has_dynamics
    robot.links[0].m = 2
    robot.links[0].r = [0, 0, 0]
    robot.links[0].I = [0.1, 0.1, 0.1]
    assert robot.has_dynamics
    npt.assert_allclose(robot.gravload([0]), [19.62])


def test_adding_mass_to_link_shared_by_robots() -> None:
    link = rtb.PrismaticDH()
    # DH kinematic setters mark this flag even without inertial parameters.
    link._hasdynamics = False
    first = rtb.DHRobot([link])
    second = rtb.DHRobot([link])
    assert not first.has_dynamics
    assert not second.has_dynamics

    link.m = 2
    # Only the most recently constructed robot receives dynchanged().
    assert not first._hasdynamics
    assert second._hasdynamics
    for robot in (first, second):
        assert robot.has_dynamics
        npt.assert_allclose(robot.gravload([0]), [19.62])
