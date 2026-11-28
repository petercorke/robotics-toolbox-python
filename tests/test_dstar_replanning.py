"""D* replanning must use the current goal, costs and ordered frontier."""

import heapq
from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import numpy as np
import numpy.testing as nt
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from roboticstoolbox import DstarPlanner, OccupancyGrid
from roboticstoolbox.mobile.DstarPlanner import _Dstar, _Map
from roboticstoolbox.mobile.PlannerBase import NoPathError


def path_cost(path: np.ndarray, grid: OccupancyGrid) -> float:
    """Sum the documented mean-cell-cost times world-distance edge costs."""
    total = 0.0
    for start, end in pairwise(path):
        c0, r0 = grid.w2g(start)
        c1, r1 = grid.w2g(end)
        total += (
            (grid.grid[r0, c0] + grid.grid[r1, c1]) / 2 * np.linalg.norm(end - start)
        )
    return float(total)


@pytest.mark.parametrize("cellsize,origin", [(1.0, (0, 0)), (0.25, (-2, 3))])
def test_plan_can_replace_the_goal(
    cellsize: float, origin: tuple[float, float]
) -> None:
    grid = OccupancyGrid(np.ones((4, 5)), cellsize=cellsize, origin=origin)
    planner = DstarPlanner(grid)
    old_goal = grid.g2w((4, 3))
    new_goal = grid.g2w((0, 0))
    planner.plan(old_goal)
    planner.query(new_goal)
    planner.plan(new_goal)

    path, status = planner.query(old_goal)

    nt.assert_allclose(path[0], old_goal)
    nt.assert_allclose(path[-1], new_goal)
    assert status.cost == pytest.approx((3 * np.sqrt(2) + 1) * cellsize)
    assert status.cost == pytest.approx(path_cost(path, grid))


def test_plan_reads_revised_costs_even_for_the_same_goal() -> None:
    grid = OccupancyGrid(np.ones((5, 6)))
    planner = DstarPlanner(grid)
    planner.plan((4, 2))
    planner.costmap[2, 2] = 100
    planner.plan()

    path, status = planner.query((1, 2))

    assert not np.any(np.all(path == [2, 2], axis=1))
    assert status.cost == pytest.approx(1 + 2 * np.sqrt(2))
    assert status.cost == pytest.approx(path_cost(path, grid))


@pytest.mark.parametrize("cellsize,origin", [(1.0, (0, 0)), (0.25, (-2, 3))])
def test_status_cost_includes_the_sensor_detour(
    cellsize: float, origin: tuple[float, float]
) -> None:
    grid = OccupancyGrid(np.ones((5, 6)), cellsize=cellsize, origin=origin)
    planner = DstarPlanner(grid)
    planner.plan(grid.g2w((4, 2)))
    called = False

    def sensor(position: tuple[float, float]) -> list[tuple[float, float, float]]:
        nonlocal called
        if called:
            return []
        called = True
        return [(*grid.g2w((2, 2)), np.inf)]

    path, status = planner.query(grid.g2w((1, 2)), sensor=sensor)

    assert status.cost == pytest.approx((1 + 2 * np.sqrt(2)) * cellsize)
    assert status.cost == pytest.approx(path_cost(path, grid))


def test_unreachable_goal_does_not_return_a_path_through_obstacles() -> None:
    costs = np.ones((5, 6))
    costs[2, :] = np.inf
    planner = DstarPlanner(costs)
    planner.plan((4, 4))

    with pytest.raises(NoPathError):
        planner.query((1, 0))


def test_sensor_can_make_the_goal_unreachable() -> None:
    planner = DstarPlanner(np.ones((5, 6)))
    planner.plan((4, 4))

    def sensor(position: tuple[float, float]) -> list[tuple[float, float, float]]:
        return [(x, 2, np.inf) for x in range(6)]

    with pytest.raises(NoPathError):
        planner.query((1, 0), sensor=sensor)


def test_query_at_goal_has_zero_cost() -> None:
    planner = DstarPlanner(np.ones((3, 4)))
    planner.plan((2, 1))
    path, status = planner.query((2, 1))
    nt.assert_array_equal(path, [[2, 1]])
    assert status.cost == 0


@pytest.mark.parametrize("seed", [0, 10, 17])
def test_weighted_paths_match_independent_shortest_paths(seed: int) -> None:
    costs = np.random.default_rng(seed).integers(1, 20, size=(5, 6)).astype(float)
    grid = OccupancyGrid(costs, cellsize=0.25, origin=(-2, 3))
    planner = DstarPlanner(grid)
    planner.plan(grid.g2w((4, 3)))
    rows, columns = costs.shape
    edges = np.zeros((costs.size, costs.size))
    for y, x in np.ndindex(costs.shape):
        for dy, dx in np.ndindex(3, 3):
            yy, xx = y + dy - 1, x + dx - 1
            if (yy, xx) == (y, x) or not (0 <= yy < rows and 0 <= xx < columns):
                continue
            edges[y * columns + x, yy * columns + xx] = (
                (costs[y, x] + costs[yy, xx]) / 2 * np.hypot(xx - x, yy - y) * 0.25
            )
    expected = dijkstra(csr_matrix(edges), indices=3 * columns + 4)

    for start in [(0, 0), (5, 0), (0, 4), (5, 4)]:
        path, status = planner.query(grid.g2w(start))
        assert status.cost == pytest.approx(expected[start[1] * columns + start[0]])
        assert status.cost == pytest.approx(path_cost(path, grid))


def test_decreased_frontier_key_preserves_priority_order() -> None:
    grid = _Map(1, 20)
    planner = _Dstar(grid)
    priorities = [10, 6, 17, 15, 12, 8, 7, 2, 5, 16, 11, 20, 4, 19, 3, 13, 18, 9, 1, 14]
    for state, priority in zip(grid._Map[0], priorities):
        planner.insert(state, priority)

    planner.insert(grid._Map[0][18], 0)

    observed = [heapq.heappop(planner.open_list)[0] for _ in range(20)]
    assert observed == [0, *range(2, 21)]
