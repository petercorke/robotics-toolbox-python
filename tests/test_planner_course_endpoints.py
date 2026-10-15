"""A zero path coordinate is valid data, including at a terminal pose."""

import numpy as np
import pytest
from spatialmath import SE2
from spatialmath.base import wrap_mpi_pi

from roboticstoolbox import DubinsPlanner, ReedsSheppPlanner


def assert_endpoint(actual, expected):
    np.testing.assert_allclose(actual[:2], expected[:2], atol=1e-9)
    np.testing.assert_array_less(abs(wrap_mpi_pi(actual[2] - expected[2])), 1e-9)


@pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("stepsize", [0.05, 0.3])
@pytest.mark.parametrize("relative_goal", [(0, 1, 0), (0, -1, 0), (0, 0, np.pi)])
def test_reeds_shepp_keeps_zero_x_terminal_pose(curvature, stepsize, relative_goal):
    start = np.zeros(3)
    goal = np.asarray(relative_goal)
    path, status = ReedsSheppPlanner(curvature, stepsize).query(start, goal)
    np.testing.assert_array_less(0, len(path))
    assert_endpoint(path[0], start)
    assert_endpoint(path[-1], goal)
    np.testing.assert_equal(len(status.direction), len(path))
    np.testing.assert_equal(np.isfinite(path).all(), True)


@pytest.mark.parametrize("start", [(3, -4, 0.7), (-2, 5, -1.2)])
def test_reeds_shepp_terminal_pose_is_frame_invariant(start):
    goal = (SE2(start) * SE2(0, 1, 0)).xyt()
    path, status = ReedsSheppPlanner().query(start, goal)
    assert_endpoint(path[0], start)
    assert_endpoint(path[-1], goal)
    np.testing.assert_equal(len(status.direction), len(path))


@pytest.mark.parametrize("start", [(0, 0, 0), (3, -4, 0.7), (-2, 5, -1.2)])
def test_dubins_keeps_stationary_path_pose(start):
    path, status = DubinsPlanner().query(start, start)
    np.testing.assert_array_less(0, len(path))
    assert_endpoint(path[0], start)
    assert_endpoint(path[-1], start)
    np.testing.assert_equal(status.length == pytest.approx(0), True)


@pytest.mark.parametrize("planner", [DubinsPlanner, ReedsSheppPlanner])
@pytest.mark.parametrize("goal", [(1, 0, 0), (2, -1, 0.4), (-1, 2, -0.8)])
def test_nonzero_terminal_coordinate_controls(planner, goal):
    start = (0, 0, 0)
    path, _ = planner().query(start, goal)
    assert_endpoint(path[0], start)
    assert_endpoint(path[-1], goal)
