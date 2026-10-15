"""Operational gravity must honour gravity overrides and Cartesian dimensions."""

import numpy as np
import numpy.testing as npt
import pytest

import roboticstoolbox as rtb

_PUMA_Q = np.array([0.2, -0.7, 0.4, 0.3, 0.5, -0.2])


def test_prismatic_gravity_wrench_has_analytic_magnitude() -> None:
    robot = rtb.DHRobot([rtb.PrismaticDH(m=2)], gravity=[0, 0, -9.81])
    npt.assert_allclose(robot.gravload_x([0.2]), [0, 0, 19.62, 0, 0, 0])
    npt.assert_allclose(robot.gravload_x([0.2], gravity=[0, 0, -3]), [0, 0, 6, 0, 0, 0])
    npt.assert_allclose(robot.gravload_x([0.2], gravity=[0, 0, 0]), np.zeros(6))


@pytest.mark.parametrize("model", ["Puma560", "Panda"])
def test_gravity_override_matches_joint_load_and_preserves_state(model: str) -> None:
    robot = getattr(rtb.models.DH, model)()
    q = np.array(robot.qr if model == "Panda" else _PUMA_Q)
    gravity = np.array([1.2, -3.4, -7.8])
    initial_gravity = robot.gravity.copy()
    initial_q = robot.q.copy()
    inverse = np.linalg.pinv(robot.jacob0_analytical(q, representation="rpy/xyz"))
    expected = inverse.T @ robot.gravload(q, gravity=gravity)

    npt.assert_allclose(robot.gravload_x(q, gravity=gravity), expected, atol=1e-12)
    npt.assert_array_equal(robot.gravity, initial_gravity)
    npt.assert_array_equal(robot.q, initial_q)
    npt.assert_array_equal(gravity, [1.2, -3.4, -7.8])


@pytest.mark.parametrize("model", ["prismatic", "Puma560", "Panda"])
def test_gravity_trajectory_returns_six_components_per_pose(model: str) -> None:
    if model == "prismatic":
        robot = rtb.DHRobot([rtb.PrismaticDH(m=2)])
        q = np.array([[0.2], [0.4]])
    else:
        robot = getattr(rtb.models.DH, model)()
        q0 = np.array(robot.qr if model == "Panda" else _PUMA_Q)
        q = np.vstack((q0, q0 + 0.1))
    gravity = [1.2, -3.4, -7.8]
    expected = np.stack(
        [
            np.linalg.pinv(robot.jacob0_analytical(qk, representation="rpy/xyz")).T
            @ robot.gravload(qk, gravity=gravity)
            for qk in q
        ]
    )

    result = robot.gravload_x(q, gravity=gravity)

    npt.assert_equal(result.shape, (2, 6))
    npt.assert_allclose(result, expected, atol=1e-12)


def test_precomputed_inverse_applies_to_gravity_trajectory() -> None:
    robot = rtb.models.DH.Puma560()
    q = np.vstack((_PUMA_Q, _PUMA_Q + 0.1))
    inverse = np.linalg.inv(robot.jacob0_analytical(q[0], representation="rpy/xyz"))
    gravity = [1.2, -3.4, -7.8]
    expected = np.stack([inverse.T @ robot.gravload(qk, gravity=gravity) for qk in q])

    npt.assert_allclose(
        robot.gravload_x(q, gravity=gravity, Ji=inverse), expected, atol=1e-12
    )
