"""Regression tests for estimated landmark map coordinates."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import EKF, Bicycle, LandmarkMap, RangeBearingSensor


@pytest.mark.parametrize("slam", [False, True])
@pytest.mark.parametrize("nobserved", [0, 1, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_get_map_landmark_coordinates(
    slam: bool, nobserved: int, reverse: bool
) -> None:
    points = np.array([[2.0, 7.0, -3.0], [3.0, -4.0, 8.0]])
    if reverse:
        points = points[:, ::-1]
    start = [1.0, -2.0, 0.4]
    robot = Bicycle(x0=start)
    robot.control = [0.0, 0.0]
    landmarks = LandmarkMap(points)
    sensor = RangeBearingSensor(robot, landmarks, covar=np.zeros((2, 2)))
    estimator = EKF(
        robot=(robot, np.eye(2) * 0.01 if slam else None),
        sensor=(sensor, np.eye(2) * 0.01),
        P0=np.eye(3) * 0.01 if slam else None,
        x0=start,
        animate=False,
    )

    if nobserved:
        estimator.run(T=robot.dt if nobserved == 1 else 2.0)
    else:
        estimator.init()

    observed = list(
        dict.fromkeys(item.lm for item in estimator.history if item.lm >= 0)
    )
    assert len(observed) == nobserved
    exported = estimator.get_map()
    if not nobserved:
        assert exported.size == 0
        return

    expected = points[:, observed].T
    individual = np.array([estimator.landmark_x(lm_id) for lm_id in observed])
    nt.assert_allclose(individual, expected, atol=1e-12)
    nt.assert_allclose(exported[0], expected[0], atol=1e-12)
    nt.assert_allclose(exported, expected, atol=1e-12)

    exported[:] = 999.0
    nt.assert_allclose(
        np.array([estimator.landmark_x(lm_id) for lm_id in observed]),
        expected,
        atol=1e-12,
    )
