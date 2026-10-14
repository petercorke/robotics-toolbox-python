"""Occupancy queries must not wrap negative grid indices."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import BinaryOccupancyGrid, PRMPlanner


@pytest.mark.parametrize("origin", [(0, 0), (10, -7)])
@pytest.mark.parametrize("cellsize", [0.25, 1.0, 2.0])
def test_outside_grid_is_occupied(origin: tuple[int, int], cellsize: float) -> None:
    """Check all sides and corners in grid coordinates, including wraparound."""
    grid = BinaryOccupancyGrid(np.zeros((3, 5)), origin=origin, cellsize=cellsize)
    for column, row in [
        (-1, 1),
        (5, 1),
        (2, -1),
        (2, 3),
        (-1, -1),
        (-1, 3),
        (5, -1),
        (5, 3),
        (-5, 1),
        (2, -3),
        (-6, 1),
        (2, -4),
    ]:
        nt.assert_equal(grid.isoccupied(grid.g2w((column, row))), True)


@pytest.mark.parametrize("origin", [(0, 0), (10, -7)])
@pytest.mark.parametrize("cellsize", [0.25, 1.0, 2.0])
def test_grid_bounds_preserve_rounding_and_cell_values(
    origin: tuple[int, int], cellsize: float
) -> None:
    """Bounds apply after world-to-grid rounding, with unchanged valid cells."""
    cells = np.zeros((3, 5))
    cells[1, 2] = 1
    grid = BinaryOccupancyGrid(cells, origin=origin, cellsize=cellsize)
    for row in range(3):
        for column in range(5):
            nt.assert_equal(
                grid.isoccupied(grid.g2w((column, row))), bool(cells[row, column])
            )
    for column, row in [(-0.49, 1), (4.49, 1), (1, -0.49), (1, 2.49)]:
        nt.assert_equal(grid.isoccupied(grid.g2w((column, row))), False)
    for column, row in [(-0.51, 1), (4.51, 1), (1, -0.51), (1, 2.51)]:
        nt.assert_equal(grid.isoccupied(grid.g2w((column, row))), True)


@pytest.mark.parametrize("outside", [(-5, 2), (20, 2), (2, -5), (2, 20)])
@pytest.mark.parametrize("endpoint", ["start", "goal"])
def test_prm_rejects_outside_endpoint(outside: tuple[int, int], endpoint: str) -> None:
    """The real planner must reject invalid endpoints before finding a path."""
    planner = PRMPlanner(np.zeros((20, 20)), npoints=10, dist_thresh=40, seed=0)
    planner.plan()
    endpoints = {"start": (2, 2), "goal": (17, 17)}
    path = planner.query(**endpoints)
    nt.assert_array_equal(path[0], endpoints["start"])
    nt.assert_array_equal(path[-1], endpoints["goal"])
    endpoints[endpoint] = outside
    with pytest.raises(ValueError, match="inside obstacle"):
        planner.query(**endpoints)
