"""Range-bearing observation rows for small maps and pose batches."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import EKF, Bicycle, LandmarkMap, RangeBearingSensor


@pytest.mark.parametrize("nlandmarks", [0, 1, 2])
def test_all_landmark_observation_rows(nlandmarks: int) -> None:
    robot = Bicycle(x0=[1.0, -2.0, 0.4])
    points = np.array([[3.0, 7.0], [4.0, -4.0]])[:, :nlandmarks]
    sensor = RangeBearingSensor(robot, LandmarkMap(points))
    delta = points.T - robot.x[:2]
    expected = np.column_stack(
        (np.linalg.norm(delta, axis=1), np.arctan2(delta[:, 1], delta[:, 0]) - 0.4)
    )

    observed = sensor.h(robot.x)
    nt.assert_equal(observed.shape, (nlandmarks, 2))
    nt.assert_allclose(observed, expected, atol=1e-12)


@pytest.mark.parametrize(
    "limits, expected_visible",
    [
        ({}, True),
        ({"range": 10.0}, True),
        ({"angle": np.pi}, True),
        ({"range": 10.0, "angle": np.pi}, True),
        ({"range": 1.0}, False),
        ({"angle": 0.2}, False),
    ],
)
def test_single_landmark_visibility_and_reading(
    limits: dict[str, float], expected_visible: bool
) -> None:
    robot = Bicycle(x0=[1.0, -2.0, 0.4])
    sensor = RangeBearingSensor(robot, LandmarkMap(np.array([[3.0], [4.0]])), **limits)
    expected = [np.sqrt(40.0), np.arctan2(6.0, 2.0) - 0.4]

    visible = sensor.visible()
    nt.assert_equal(len(visible), int(expected_visible))
    if expected_visible:
        nt.assert_allclose(visible[0][0], expected, atol=1e-12)
        nt.assert_equal(visible[0][1], 0)

    observation, landmark_id = sensor.reading()
    if expected_visible:
        nt.assert_equal(landmark_id, 0)
        nt.assert_allclose(observation, expected, atol=1e-12)
    else:
        nt.assert_equal(observation is None, True)
        nt.assert_equal(landmark_id is None, True)


@pytest.mark.parametrize("nposes", [0, 1, 3])
def test_explicit_landmark_pose_batch_rows(nposes: int) -> None:
    robot = Bicycle(x0=[1.0, -2.0, 0.4])
    sensor = RangeBearingSensor(robot, LandmarkMap(np.array([[3.0], [4.0]])))
    poses = np.tile(robot.x, (nposes, 1))
    expected = np.tile([np.sqrt(40.0), np.arctan2(6.0, 2.0) - 0.4], (nposes, 1))

    observed = sensor.h(poses, 0)
    nt.assert_equal(observed.shape, (nposes, 2))
    nt.assert_allclose(observed, expected, atol=1e-12)


def test_explicit_landmark_scalar_pose_shape() -> None:
    robot = Bicycle(x0=[1.0, -2.0, 0.4])
    point = np.array([3.0, 4.0])
    sensor = RangeBearingSensor(robot, LandmarkMap(point[:, None]))
    expected = [np.sqrt(40.0), np.arctan2(6.0, 2.0) - 0.4]
    for landmark in [0, point]:
        observed = sensor.h(robot.x, landmark)
        nt.assert_equal(observed.shape, (2,))
        nt.assert_allclose(observed, expected, atol=1e-12)


@pytest.mark.parametrize("slam", [False, True])
def test_single_landmark_ekf_observations(slam: bool) -> None:
    start = [1.0, -2.0, 0.4]
    robot = Bicycle(x0=start)
    robot.control = [0.0, 0.0]
    point = np.array([[3.0], [4.0]])
    sensor = RangeBearingSensor(robot, LandmarkMap(point))
    estimator = EKF(
        robot=(robot, np.eye(2) * 0.01 if slam else None),
        sensor=(sensor, np.eye(2) * 0.01),
        P0=np.eye(3) * 0.01 if slam else None,
        x0=start,
        animate=False,
    )
    estimator.run(T=1.0)

    nt.assert_equal(all(item.lm == 0 for item in estimator.history), True)
    nt.assert_allclose(estimator.landmark_x(0), point[:, 0], atol=1e-12)
