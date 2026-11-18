"""Weighted grassfire distances must converge after reachability is known."""

import heapq

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import BinaryOccupancyGrid, DistanceTransformPlanner
from roboticstoolbox.mobile.DistanceTransformPlanner import distancexform
from roboticstoolbox.tools.types import NDArray


def two_routes() -> NDArray:
    # The eight-edge zigzag arrives first but costs 1 + 7*sqrt(2).
    # The ten-edge route up the left and across the top costs 9 + sqrt(2).
    return np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 1],
        ],
        dtype=bool,
    )


def shortest_distances(grid: NDArray, goal: tuple[int, int], metric: str) -> NDArray:
    """Independent Dijkstra oracle with the planner's grid connectivity."""
    distances = np.full(grid.shape, np.inf)
    distances[grid] = np.nan
    y, x = goal
    distances[y, x] = 0.0
    queue = [(0.0, y, x)]
    while queue:
        cost, y, x = heapq.heappop(queue)
        if cost > distances[y, x]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx) or (metric != "euclidean" and dy and dx):
                    continue
                row, column = y + dy, x + dx
                if not (0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]):
                    continue
                if grid[row, column]:
                    continue
                candidate = cost + np.hypot(dx, dy)
                if candidate < distances[row, column]:
                    distances[row, column] = candidate
                    heapq.heappush(queue, (candidate, row, column))
    return distances


@pytest.mark.parametrize("turns", range(4))
def test_longer_route_can_have_lower_euclidean_cost(turns: int) -> None:
    grid = two_routes()
    goals = np.zeros_like(grid)
    goals[6, 0] = True
    targets = np.zeros_like(grid)
    targets[0, 5] = True
    grid = np.rot90(grid, turns)
    goal = tuple(np.argwhere(np.rot90(goals, turns))[0])
    target = tuple(np.argwhere(np.rot90(targets, turns))[0])

    result = distancexform(grid, goal[::-1], metric="euclidean")

    nt.assert_allclose(result[target], 9 + np.sqrt(2), atol=1e-12)
    nt.assert_allclose(result, shortest_distances(grid, goal, "euclidean"))


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cityblock"])
def test_unreachable_cells_and_obstacles_still_terminate(
    metric: str, capsys: pytest.CaptureFixture[str]
) -> None:
    grid = np.zeros((7, 9), dtype=bool)
    grid[3, :] = True
    grid[0, 1] = True
    goal = (1, 1)

    result = distancexform(grid, goal[::-1], metric=metric, summary=True)

    nt.assert_allclose(result, shortest_distances(grid, goal, metric))
    assert np.isnan(result[grid]).all()
    assert np.isinf(result[4:, :]).all()
    assert "27 unreachable cells" in capsys.readouterr().out


def test_isolated_goal_terminates() -> None:
    grid = np.ones((5, 5), dtype=bool)
    grid[0, 0] = False
    grid[3, 3] = False

    result = distancexform(grid, (0, 0), metric="euclidean")

    nt.assert_allclose(result, shortest_distances(grid, (0, 0), "euclidean"))
    assert result[0, 0] == 0
    assert np.isinf(result[3, 3])


def test_planner_distance_field_retains_world_units() -> None:
    grid = BinaryOccupancyGrid(two_routes(), origin=(-3.0, 1.5), cellsize=0.25)
    planner = DistanceTransformPlanner(grid, metric="euclidean")

    planner.plan(goal=grid.g2w((0, 6)))

    expected = shortest_distances(grid.grid, (6, 0), "euclidean") * grid.cellsize
    nt.assert_allclose(planner.distancemap, expected)
