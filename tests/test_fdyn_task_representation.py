"""Task-space outputs describe one pose representation and its derivatives."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath.base import tr2x

from roboticstoolbox import models
from roboticstoolbox.blocks.arm import FDyn_X


@pytest.mark.parametrize("representation", ["rpy/xyz", "rpy/zyx", "eul", "exp"])
@pytest.mark.parametrize("after_deriv", [False, True])
def test_pose_velocity_and_acceleration_share_representation(
    representation: str, after_deriv: bool
) -> None:
    robot = models.DH.Puma560()
    q = np.array([0.3, -0.6, 0.7, 0.2, 0.8, -0.4])
    qd = np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.3])
    state = np.r_[q, qd]
    block = FDyn_X(robot, q, representation=representation)
    wrench = np.array([0.5, -0.2, 0.3, 0.1, -0.1, 0.2])
    if after_deriv:
        qdd = block.test_deriv(wrench, x=state)[robot.n :]
    else:
        qdd = np.zeros(robot.n)

    # output() may run before its wrench input has been resolved by bdsim.
    result = block.output(0.0, [None], state)

    def pose(t: float):
        qt = q + t * qd + 0.5 * t**2 * qdd
        return tr2x(robot.fkine(qt).A, representation=representation)

    h = 1e-4
    expected_pose = pose(0.0)
    expected_velocity = (pose(h) - pose(-h)) / (2 * h)
    expected_acceleration = (pose(h) - 2 * expected_pose + pose(-h)) / h**2

    nt.assert_array_equal(result[0], q)
    nt.assert_array_equal(result[1], qd)
    nt.assert_allclose(result[2], expected_pose, atol=1e-12)
    nt.assert_allclose(result[3], expected_velocity, atol=2e-6)
    nt.assert_allclose(result[4], expected_acceleration, atol=2e-5)
    nt.assert_array_equal(state, np.r_[q, qd])


def test_stationary_default_output_remains_unchanged() -> None:
    robot = models.DH.Puma560()
    q = np.array([0.3, -0.6, 0.7, 0.2, 0.8, -0.4])
    block = FDyn_X(robot, q)

    result = block.output(0.0, [None], np.r_[q, np.zeros(robot.n)])

    nt.assert_allclose(result[2], tr2x(robot.fkine(q).A))
    nt.assert_array_equal(result[3], np.zeros(6))
    nt.assert_array_equal(result[4], np.zeros(6))


def test_representation_parameter_changes_all_task_outputs() -> None:
    robot = models.DH.Puma560()
    q = np.array([0.3, -0.6, 0.7, 0.2, 0.8, -0.4])
    block = FDyn_X(robot, q)
    block.set_param("representation", "eul")

    result = block.output(0.0, [None], np.r_[q, np.zeros(robot.n)])

    nt.assert_allclose(result[2], tr2x(robot.fkine(q).A, representation="eul"))
    nt.assert_array_equal(result[3], np.zeros(6))
    nt.assert_array_equal(result[4], np.zeros(6))
