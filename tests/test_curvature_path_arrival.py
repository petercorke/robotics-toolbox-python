"""A locally converged polynomial is not necessarily a path to the goal."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from spatialmath.base import angdiff

from roboticstoolbox import CurvaturePolyPlanner, NoPathError
from roboticstoolbox.mobile.CurvaturePolyPlanner import solvepath


@pytest.mark.parametrize("shift", [0.0, 1e3, 1e9])
def test_real_optimizer_cannot_return_a_path_in_the_opposite_direction(shift):
    start = (shift, -shift, 0.0)
    goal = (shift - 1, -shift, 0.0)
    with pytest.raises(NoPathError, match="reach the goal"):
        CurvaturePolyPlanner().query(start, goal)


@pytest.mark.parametrize("heading", [-np.pi / 2, np.pi / 2, np.pi])
def test_real_optimizer_cannot_return_a_stationary_path_for_a_turn(heading):
    with pytest.raises(NoPathError, match="reach the goal"):
        CurvaturePolyPlanner().query((0, 0, 0), (0, 0, heading))


@pytest.mark.parametrize("shift", [0.0, 1e3, 1e6, 1e9])
@pytest.mark.parametrize("turns", [-1, 0, 2])
def test_real_straight_paths_preserve_map_translation_and_heading_wrap(shift, turns):
    start = np.array([shift, -shift, 0.3])
    goal = start + [2 * np.cos(0.3), 2 * np.sin(0.3), turns * 2 * np.pi]
    path, status = CurvaturePolyPlanner().query(start, goal)
    assert_allclose(path[0], start, atol=1e-10, rtol=0)
    error = path[-1] - goal
    error[2] = angdiff(path[-1, 2], goal[2])
    assert np.linalg.norm(error) < 1e-3
    assert_allclose(status.length, 2, atol=2e-7)


@pytest.mark.parametrize(
    "start,goal",
    [
        ((0, 0, np.pi / 2), (1, 0, np.pi / 2)),
        ((0, 0, 0), (1, 1, np.pi / 2)),
        ((0, 0, 0.3), (2, 1, 0.5)),
    ],
)
def test_real_curved_paths_still_reach_the_goal(start, goal):
    path, status = CurvaturePolyPlanner().query(start, goal)
    error = path[-1] - goal
    error[2] = angdiff(path[-1, 2], goal[2])
    assert np.linalg.norm(error) < 1e-3
    assert status.length >= np.linalg.norm(np.subtract(goal[:2], start[:2]))
    assert np.all(np.isfinite(path))


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("overshoot", [0.0005, 0.002])
def test_actual_arrival_controls_acceptance_not_the_optimizer_flag(
    monkeypatch, success, overshoot
):
    parameters = np.array([0, 0, 0, 0, 1 + overshoot])
    monkeypatch.setattr(
        "scipy.optimize.minimize",
        lambda *args, **kwargs: SimpleNamespace(x=parameters, success=success, fun=0.0),
    )
    planner = CurvaturePolyPlanner()
    if overshoot < 1e-3:
        path, status = planner.query((0, 0, 0), (1, 0, 0))
        assert_allclose(path[-1], [1 + overshoot, 0, 0], atol=1e-12)
        assert status.length == 1 + overshoot
    else:
        with pytest.raises(NoPathError, match="reach the goal"):
            planner.query((0, 0, 0), (1, 0, 0))


def test_coarse_objective_does_not_override_final_trajectory(monkeypatch):
    # A real constant-curvature arc is integrated, with the optimizer's
    # reported objective deliberately inconsistent with that trajectory.
    parameters = np.array([0, 0, 0, 1, np.pi / 2])
    monkeypatch.setattr(
        "scipy.optimize.minimize",
        lambda *args, **kwargs: SimpleNamespace(x=parameters, success=True, fun=0.0),
    )
    with pytest.raises(NoPathError, match="reach the goal"):
        CurvaturePolyPlanner().query((0, 0, 0), (1, 0.9, np.pi / 2))
    path, _ = CurvaturePolyPlanner().query((0, 0, 0), (1, 1, 5 * np.pi / 2))
    assert_allclose(path[-1], [1, 1, np.pi / 2], atol=1e-9)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_optimizer_output_is_a_planning_failure(monkeypatch, bad):
    monkeypatch.setattr(
        "scipy.optimize.minimize",
        lambda *args, **kwargs: SimpleNamespace(x=np.array([0, 0, 0, bad, 1])),
    )
    with pytest.raises(NoPathError, match="finite path parameters"):
        CurvaturePolyPlanner().query((0, 0, 0), (1, 0, 0))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("endpoint", ["start", "goal"])
def test_nonfinite_endpoints_are_invalid_input(bad, endpoint):
    points = {"start": [0, 0, 0], "goal": [1, 0, 0]}
    points[endpoint][0] = bad
    with pytest.raises(ValueError, match="finite"):
        CurvaturePolyPlanner().query(**points)


def test_nonfinite_integrated_path_is_a_planning_failure(monkeypatch):
    import importlib

    module = importlib.import_module("roboticstoolbox.mobile.CurvaturePolyPlanner")
    path, curvature = solvepath([0, 0, 0, 0, 1])
    path[0, -1] = np.nan
    monkeypatch.setattr(module, "solvepath", lambda *args, **kwargs: (path, curvature))
    monkeypatch.setattr(
        "scipy.optimize.minimize",
        lambda *args, **kwargs: SimpleNamespace(x=np.array([0, 0, 0, 0, 1])),
    )
    with pytest.raises(NoPathError, match="non-finite poses"):
        CurvaturePolyPlanner().query((0, 0, 0), (1, 0, 0))


def test_finite_endpoints_cannot_hide_a_nonfinite_residual(monkeypatch):
    import importlib

    module = importlib.import_module("roboticstoolbox.mobile.CurvaturePolyPlanner")
    large = np.finfo(float).max
    path = np.array([[0, 1], [0, 0], [large, large]])
    monkeypatch.setattr(module, "solvepath", lambda *args, **kwargs: (path, 0.0))
    monkeypatch.setattr(
        "scipy.optimize.minimize",
        lambda *args, **kwargs: SimpleNamespace(x=np.array([0, 0, 0, 0, 1])),
    )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(NoPathError, match="reach the goal"),
    ):
        CurvaturePolyPlanner().query((0, 0, large), (1, 0, -large))
