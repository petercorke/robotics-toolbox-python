"""Operational acceleration requires aligned trajectory rows."""

import numpy as np
import numpy.testing as npt
import pytest

from roboticstoolbox.models.DH import Panda, Puma560


@pytest.mark.parametrize("robot_type", [Puma560, Panda])
@pytest.mark.parametrize(
    "rows", [(3, 1, 3), (3, 3, 1), (1, 3, 3), (3, 2, 3), (3, 3, 2), (2, 3, 3)]
)
def test_mismatched_trajectories_raise(
    robot_type: type[Puma560] | type[Panda], rows: tuple[int, int, int]
) -> None:
    robot = robot_type()
    pose = robot.qn if isinstance(robot, Puma560) else robot.qr
    q = np.tile(pose, (rows[0], 1))
    xd = np.full((rows[1], 6), 0.01)
    wrench = np.full((rows[2], 6), 0.2)
    with pytest.raises(ValueError, match="same number of rows"):
        robot.accel_x(q, xd, wrench)


@pytest.mark.parametrize("robot_type", [Puma560, Panda])
@pytest.mark.parametrize("count", [1, 3])
def test_matching_trajectories_match_individual_calls(
    robot_type: type[Puma560] | type[Panda], count: int
) -> None:
    robot = robot_type()
    pose = robot.qn if isinstance(robot, Puma560) else robot.qr
    q = np.tile(pose, (count, 1)) + np.arange(count)[:, None] * 0.01
    xd = np.arange(count * 6).reshape(count, 6) * 0.001
    wrench = np.arange(count * 6).reshape(count, 6) * 0.01
    saved = [value.copy() for value in (q, xd, wrench)]
    expected = np.stack(
        [robot.accel_x(qk, vk, wk) for qk, vk, wk in zip(q, xd, wrench)]
    )
    actual = robot.accel_x(q, xd, wrench)
    assert actual.shape == ((6,) if count == 1 else (count, 6))
    npt.assert_allclose(
        actual, expected[0] if count == 1 else expected, atol=1e-10, rtol=1e-10
    )
    for old, current in zip(saved, (q, xd, wrench)):
        npt.assert_array_equal(current, old)


@pytest.mark.parametrize("robot_type", [Puma560, Panda])
def test_empty_matching_trajectories(robot_type: type[Puma560] | type[Panda]) -> None:
    robot = robot_type()
    result = robot.accel_x(np.empty((0, robot.n)), np.empty((0, 6)), np.empty((0, 6)))
    assert result.shape == (0, 6)
