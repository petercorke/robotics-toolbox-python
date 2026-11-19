"""Joint-limit constraints must address the same coordinates as robot.q."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import LinearConstraint, minimize

from roboticstoolbox import ET, Link, Robot, models


def indexed_robot(indices, mixed=False, first_limits=(-1, 1)):
    base = Link(ET.tx(0.4), name="base")
    first = Link(ET.Rz(jindex=indices[0], qlim=first_limits), parent=base, name="first")
    fixed = Link(ET.tx(0.6), parent=first, name="fixed")
    transform = ET.tx if mixed else ET.Ry
    second = Link(
        transform(jindex=indices[1], qlim=[-2, 2]), parent=fixed, name="second"
    )
    third = Link(ET.Rz(jindex=indices[2], qlim=[-3, 3]), parent=second, name="third")
    return Robot([base, first, fixed, second, third])


def coordinates_and_constraints(indices, gain=1.0):
    q = np.empty(3)
    q[indices] = [0.94, -1.92, 0.0]
    diagonal = np.empty(3)
    diagonal[indices] = [1, -1, 0]
    bound = np.empty(3)
    # Distances 0.06 and 0.08 to the upper/lower limit with ps=.05, pi=.1.
    bound[indices] = gain * np.array([0.2, 0.6, 0.0])
    return q, np.diag(diagonal), bound


@pytest.mark.parametrize("indices", [[0, 1, 2], [1, 0, 2], [2, 0, 1], [1, 2, 0]])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("explicit_q", [False, True])
@pytest.mark.parametrize("gain", [0.5, 1.5])
def test_damper_uses_global_joint_coordinates(indices, mixed, explicit_q, gain):
    robot = indexed_robot(indices, mixed=mixed)
    q, expected_a, expected_b = coordinates_and_constraints(indices, gain)
    if not explicit_q:
        robot.q = q
    original_state = robot.q.copy()
    original_argument = q.copy()

    actual_a, actual_b = robot.joint_velocity_damper(
        q=q if explicit_q else None, gain=gain
    )

    assert_allclose(actual_a, expected_a, atol=1e-14)
    assert_allclose(actual_b, expected_b, atol=1e-14)
    assert_array_equal(robot.q, original_state)
    assert_array_equal(q, original_argument)


@pytest.mark.parametrize("indices", [[2, 0, 1], [1, 2, 0]])
@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_n_selects_a_prefix_of_the_configuration_vector(indices, n):
    robot = indexed_robot(indices, mixed=True)
    q, expected_a, expected_b = coordinates_and_constraints(indices)

    actual_a, actual_b = robot.joint_velocity_damper(q=q, n=n)

    assert actual_a.shape == (n, n)
    assert actual_b.shape == (n,)
    assert_allclose(actual_a, expected_a[:n, :n], atol=1e-14)
    assert_allclose(actual_b, expected_b[:n], atol=1e-14)


def test_unspecified_revolute_limits_keep_the_existing_pi_default():
    robot = indexed_robot([2, 0, 1], first_limits=None)
    q = np.array([0.0, 0.0, np.pi - 0.06])

    a, b = robot.joint_velocity_damper(q=q)

    assert_allclose(a, np.diag([0, 0, 1]), atol=1e-14)
    assert_allclose(b, [0, 0, 0.2], atol=1e-14)


def test_empty_prefix_does_not_require_limits_for_unused_joints():
    robot = Robot([Link(ET.tx())])
    a, b = robot.joint_velocity_damper(n=0)
    assert a.shape == (0, 0)
    assert b.shape == (0,)


@pytest.mark.parametrize("factory", [models.DH.Puma560, models.URDF.Panda])
def test_existing_ordered_robot_models_keep_their_constraints(factory):
    robot = factory()
    q = np.mean(robot.qlim, axis=0)
    q[0] = robot.qlim[1, 0] - 0.06
    q[-1] = robot.qlim[0, -1] + 0.08
    expected = np.zeros(robot.n)
    expected[0], expected[-1] = 1, -1
    bound = np.zeros(robot.n)
    bound[0], bound[-1] = 0.2, 0.6

    a, b = robot.joint_velocity_damper(q=q)

    assert_allclose(a, np.diag(expected), atol=1e-14)
    assert_allclose(b, bound, atol=1e-14)


def test_qp_motion_respects_the_physical_joint_limit():
    robot = Robot(
        ET.Rz(jindex=1, qlim=[-1, 1])
        * ET.tx(1)
        * ET.Rz(jindex=0, qlim=[-2, 2])
        * ET.tx(1)
    )
    q = np.array([0.0, 0.94])
    desired = np.array([0.4, 2.0])
    a, b = robot.joint_velocity_damper(q=q)
    active = np.any(a != 0, axis=1)
    constraints = (
        [LinearConstraint(a[active], -np.inf, b[active])] if np.any(active) else []
    )
    result = minimize(
        lambda velocity: 0.5 * np.sum((velocity - desired) ** 2),
        np.zeros(2),
        jac=lambda velocity: velocity - desired,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12},
    )
    assert result.success

    following = q + 0.05 * result.x
    # The first physical joint is q[1]. Its safety margin is 0.05 below 1.
    # Without its constraint, the requested step reaches 1.04 and crosses
    # even the hard joint limit; the constrained step ends at 0.95.
    assert following[1] <= 0.95 + 1e-12
    assert_allclose(result.x, [0.4, 0.2], atol=1e-12)
    assert_allclose(
        robot.fkine(following).t[:2],
        [
            np.cos(following[1]) + np.cos(following.sum()),
            np.sin(following[1]) + np.sin(following.sum()),
        ],
        atol=1e-12,
    )
