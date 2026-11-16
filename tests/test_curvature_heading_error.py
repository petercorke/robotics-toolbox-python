"""Curvature-planner objectives measure heading error on the circle."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import CurvaturePolyPlanner
from roboticstoolbox.mobile.CurvaturePolyPlanner import costfunc


@pytest.mark.parametrize("turns", [-2, -1, 1, 2])
def test_cost_is_invariant_to_full_heading_turns(turns: int) -> None:
    start = np.array([1.0, 2.0, 0.3])
    goal = start + [2 * np.cos(0.3), 2 * np.sin(0.3), turns * 2 * np.pi]

    cost = costfunc([0, 0, 0, 0, 2], start, goal)

    nt.assert_allclose(cost, 0, atol=1e-12)


def test_cost_uses_short_heading_error_across_angle_cut() -> None:
    start = np.array([0.0, 0.0, np.pi - 0.02])
    goal = np.array([2 * np.cos(start[2]), 2 * np.sin(start[2]), -np.pi + 0.01])

    cost = costfunc([0, 0, 0, 0, 2], start, goal)

    nt.assert_allclose(cost, 0.03, atol=1e-12)


@pytest.mark.timeout(15)
def test_query_does_not_add_a_loop_for_equivalent_goal_heading() -> None:
    planner = CurvaturePolyPlanner()

    path, status = planner.query((0, 0, 0), (2, 0, 2 * np.pi))

    nt.assert_allclose(status.length, 2, atol=1e-8)
    nt.assert_allclose(path[:, 1:], 0, atol=1e-8)
    nt.assert_allclose(path[-1, :2], [2, 0], atol=1e-8)
