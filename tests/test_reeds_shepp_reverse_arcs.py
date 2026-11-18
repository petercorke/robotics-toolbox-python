"""A short reversing arc must not be replaced by a forward full-circle detour."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from spatialmath.base import angdiff

from roboticstoolbox import ReedsSheppPlanner
from roboticstoolbox.mobile.ReedsSheppPlanner import left_right_left


def endpoint(start, lengths, modes, curvature=1.0):
    """Compose exact straight/circular motions independently of the planner."""
    pose = np.array(start, dtype=float)
    for length, mode in zip(lengths, modes):
        heading = pose[2]
        if mode == "S":
            pose[:2] += length * np.array([np.cos(heading), np.sin(heading)])
        else:
            k = curvature if mode == "L" else -curvature
            final = heading + k * length
            pose[0] += (np.sin(final) - np.sin(heading)) / k
            pose[1] += (np.cos(heading) - np.cos(final)) / k
            pose[2] = final
    return pose


def assert_arrival(path, status, start, goal, curvature):
    assert np.all(np.isfinite(path))
    assert_allclose(path[0, :2], start[:2], atol=1e-11)
    assert_allclose(path[-1, :2], goal[:2], atol=1e-11)
    assert abs(angdiff(path[-1, 2], goal[2])) < 1e-11
    # The selected segment sequence also reaches the goal when integrated
    # analytically, without using the planner's sampling implementation.
    exact = endpoint(start, status.seglengths, status.segments, curvature)
    assert_allclose(exact[:2], goal[:2], atol=1e-11)
    assert abs(angdiff(exact[2], goal[2])) < 1e-11
    assert_allclose(status.length, np.sum(np.abs(status.seglengths)), atol=1e-12)


@pytest.mark.parametrize("curvature", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("start", [(0.0, 0.0, 0.0), (1.0, -2.0, 0.7)])
@pytest.mark.parametrize("modes", ["LSL", "RSR"])
@pytest.mark.parametrize("gear", [-1.0, 1.0])
def test_short_reverse_arc_beats_a_known_feasible_route(curvature, start, modes, gear):
    lengths = gear * np.array([0.2, 0.2, -0.2]) / curvature
    goal = endpoint(start, lengths, modes, curvature)
    planner = ReedsSheppPlanner(curvature=curvature, stepsize=0.02 / curvature)

    path, status = planner.query(start, goal)

    assert status.length <= np.sum(np.abs(lengths)) + 1e-10
    assert np.any(np.asarray(status.seglengths) < 0)
    assert np.any(np.asarray(status.seglengths) > 0)
    assert_arrival(path, status, start, goal, curvature)


@pytest.mark.parametrize(
    "last",
    [
        -0.4,
        -0.1,
        -1e-12,
        0.0,
        1e-12,
        0.1,
        np.pi - 1e-12,
        np.pi,
        np.pi + 1e-12,
        -np.pi - 1e-12,
        -np.pi,
        -np.pi + 1e-12,
    ],
)
def test_final_curve_uses_a_signed_principal_angle(last):
    goal = endpoint([0, 0, 0], [0.3, -0.2, last], "LRL")

    found, first, middle, final = left_right_left(*goal)

    assert found
    assert_allclose([first, middle], [0.3, -0.2], atol=1e-12)
    assert abs(final) <= np.pi + 1e-14
    assert abs(angdiff(final, last)) < 1e-12
    assert_allclose(abs(final), abs(np.arctan2(np.sin(last), np.cos(last))), atol=1e-12)


@pytest.mark.parametrize("turns", [-3, 0, 2])
def test_equivalent_goal_headings_preserve_the_short_route(turns):
    goal = endpoint([0, 0, 0], [0.2, 0.2, -0.2], "LSL")
    goal[2] += turns * 2 * np.pi
    path, status = ReedsSheppPlanner(stepsize=0.02).query([0, 0, 0], goal)
    assert status.length <= 0.6 + 1e-10
    assert_arrival(path, status, [0, 0, 0], goal, 1.0)


@pytest.mark.parametrize("heading", [-np.pi, np.pi])
def test_half_turn_keeps_either_equivalent_shortest_sequence(heading):
    goal = [0, 0, heading]
    path, status = ReedsSheppPlanner(stepsize=0.02).query([0, 0, 0], goal)
    assert_allclose(status.length, np.pi, atol=1e-12)
    assert_arrival(path, status, [0, 0, 0], goal, 1.0)


@pytest.mark.parametrize("modes", ["LSR", "RSL"])
@pytest.mark.parametrize("gear", [-1.0, 1.0])
def test_long_curve_straight_curve_candidates_remain_available(modes, gear):
    # Restricting the separate CSC formulas to principal positive turns would
    # lose this existing coverage. Their last turn is deliberately above pi.
    lengths = gear * np.array([0.2, 4.0, 3.5])
    start = [1, -2, 0.7]
    goal = endpoint(start, lengths, modes)
    path, status = ReedsSheppPlanner(stepsize=0.05).query(start, goal)
    assert status.length <= np.sum(np.abs(lengths)) + 1e-10
    assert_arrival(path, status, start, goal, 1.0)
