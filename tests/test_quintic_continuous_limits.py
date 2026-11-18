"""Quintic path limits apply between the returned time samples too."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BPoly

from roboticstoolbox import QuinticPolyPlanner
from roboticstoolbox.mobile.PlannerBase import NoPathError


def boundary_polynomial(start, goal, duration, sv=0, sa=0, gv=0, ga=0):
    """Independent Bernstein interpolation of the six boundary conditions."""
    start_direction = np.array([np.cos(start[2]), np.sin(start[2])])
    goal_direction = np.array([np.cos(goal[2]), np.sin(goal[2])])
    return BPoly.from_derivatives(
        [0, duration],
        [
            [start[:2], sv * start_direction, sa * start_direction],
            [goal[:2], gv * goal_direction, ga * goal_direction],
        ],
    )


@pytest.mark.parametrize("heading", [0, 0.7, -2.1])
@pytest.mark.parametrize("dt", [0.1, 1, 5])
def test_acceleration_peak_changes_duration_even_without_sample_at_peak(heading, dt):
    start = (0.3, -0.4, heading)
    goal = (start[0] + np.cos(heading), start[1] + np.sin(heading), heading)
    planner = QuinticPolyPlanner(dt=dt, max_acc=0.1, max_jerk=100, max_t=20)
    path, status = planner.query(start, goal)

    # For rest-to-rest motion over one unit, max |q''| = 10*sqrt(3)/(3*T**2).
    assert status.t[-1] == 10
    assert 10 * np.sqrt(3) / (3 * status.t[-1] ** 2) <= planner.max_acc
    assert_allclose(path[[0, -1], :2], np.array([start[:2], goal[:2]]), atol=2e-13)


@pytest.mark.parametrize("dt", [1, 5, 9])
def test_peak_just_above_limit_is_not_rounded_down_to_sample_maximum(dt):
    planner = QuinticPolyPlanner(dt=dt, max_acc=0.2305, max_jerk=100, max_t=20)
    _, status = planner.query((0, 0, 0), (1, 0, 0))
    # At T=5, the t=1,4 samples give 0.2304, but the peak is 0.2309401.
    assert status.t[-1] == 10


@pytest.mark.parametrize("limit", [0.1, 0.2305])
def test_no_feasible_duration_rejects_an_intersample_violation(limit):
    planner = QuinticPolyPlanner(dt=5, max_acc=limit, max_jerk=100, max_t=10)
    with pytest.raises(NoPathError, match="acceleration and jerk limits"):
        planner.query((0, 0, 0), (1, 0, 0))


@pytest.mark.parametrize("heading", [0, 0.6, -1.4])
@pytest.mark.parametrize("dt", [0.2, 1, 3])
def test_interior_jerk_peak_rejects_the_first_duration(heading, dt):
    direction = np.array([np.cos(heading), np.sin(heading)])
    start = (0, 0, heading)
    goal = (*(0.15 * direction), heading)
    planner = QuinticPolyPlanner(
        dt=dt,
        goal_vel=0.5,
        goal_acc=1,
        max_acc=2,
        max_jerk=1.2,
        min_t=1,
        max_t=3,
    )
    _, status = planner.query(start, goal)
    # T=1 gives q(t)=t**4/4-t**5/10, with jerk 6*t*(1-t).
    # Its endpoint jerks are zero, but its interior peak is 1.5.
    assert status.t[-1] == 2
    curve = boundary_polynomial(start, goal, 2, gv=0.5, ga=1)
    samples = np.linspace(0, 2, 2001)
    assert np.linalg.norm(curve.derivative(3)(samples), axis=1).max() < 1.2


@pytest.mark.parametrize("time_scale", [0.001, 1, 1000])
def test_duration_selection_is_invariant_to_time_units(time_scale):
    planner = QuinticPolyPlanner(
        dt=5 * time_scale,
        min_t=5 * time_scale,
        max_t=20 * time_scale,
        max_acc=0.1 / time_scale**2,
        max_jerk=100 / time_scale**3,
    )
    _, status = planner.query((0, 0, 0), (1, 0, 0))
    assert_allclose(status.t[-1], 10 * time_scale, rtol=2e-14)


@pytest.mark.parametrize("distance_scale", [1e-170, 1e170])
def test_finite_derivative_magnitudes_do_not_overflow_or_underflow_when_squared(
    distance_scale,
):
    planner = QuinticPolyPlanner(
        dt=5,
        max_t=20,
        max_acc=0.1 * distance_scale,
        max_jerk=100 * distance_scale,
    )
    path, status = planner.query((0, 0, 0), (distance_scale, 0, 0))
    assert status.t[-1] == 10
    assert_allclose(path[-1, 0] / distance_scale, 1, atol=2e-14)


@pytest.mark.parametrize("dt", [0.07, 1.3, 20])
@pytest.mark.parametrize(
    "start,goal,sv,sa,gv,ga,max_acc,max_jerk,expected_time",
    [
        ((0, 0, 0.4), (1.7, -0.5, -0.9), 0.3, 0.2, 0.1, -0.1, 0.5, 0.4, 6),
        ((-0.3, 0.7, 1), (1.5, 2.1, -0.2), 0.4, -0.1, 0.2, 0.2, 0.6, 0.3, 8),
        ((0.2, -0.8, -1.2), (-1.1, 0.9, 2), 0.2, 0.1, 0.5, -0.15, 0.5, 0.5, 7),
    ],
)
def test_planar_vector_limits_against_independent_bernstein_curve(
    dt, start, goal, sv, sa, gv, ga, max_acc, max_jerk, expected_time
):
    planner = QuinticPolyPlanner(
        dt=dt,
        start_vel=sv,
        start_acc=sa,
        goal_vel=gv,
        goal_acc=ga,
        max_acc=max_acc,
        max_jerk=max_jerk,
        min_t=1,
        max_t=15,
    )
    path, status = planner.query(start, goal)
    assert status.t[-1] == expected_time
    curve = boundary_polynomial(start, goal, expected_time, sv, sa, gv, ga)
    assert_allclose(path[:, :2], curve(status.t), atol=2e-13)
    samples = np.linspace(0, expected_time, 20001)
    acceleration = np.linalg.norm(curve.derivative(2)(samples), axis=1)
    jerk = np.linalg.norm(curve.derivative(3)(samples), axis=1)
    assert acceleration.max() <= max_acc
    assert jerk.max() <= max_jerk
    for previous_time in range(1, expected_time):
        previous = boundary_polynomial(start, goal, previous_time, sv, sa, gv, ga)
        times = np.linspace(0, previous_time, 20001)
        assert (
            np.linalg.norm(previous.derivative(2)(times), axis=1).max() > max_acc
            or np.linalg.norm(previous.derivative(3)(times), axis=1).max() > max_jerk
        )


@pytest.mark.parametrize("moving", [False, True])
def test_zero_derivatives_accept_stationary_and_constant_velocity_paths(moving):
    speed = 1 if moving else 0
    planner = QuinticPolyPlanner(
        dt=0.4,
        start_vel=speed,
        goal_vel=speed,
        max_acc=0,
        max_jerk=0,
        min_t=1,
        max_t=2,
    )
    path, status = planner.query((0, 0, 0), (speed, 0, 0))
    assert status.t[-1] == 1
    assert_allclose(path[:, 0], speed * np.asarray(status.t), atol=1e-14)
    assert_allclose(status.acc, 0, atol=1e-14)
    assert_allclose(status.jerk, 0, atol=1e-14)
