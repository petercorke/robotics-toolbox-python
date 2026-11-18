"""Inflation uses a circular radius in world units without rounding it."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import BinaryOccupancyGrid, DistanceTransformPlanner


def inflated_points(cells, cellsize, radius):
    """Independent oracle: test every cell against all occupied cell centres."""
    occupied = np.argwhere(cells)
    if occupied.size == 0:
        return np.zeros(cells.shape, dtype=bool)
    points = np.stack(np.indices(cells.shape), axis=-1)
    distances = np.linalg.norm((points[..., None, :] - occupied) * cellsize, axis=-1)
    return np.any(distances <= np.nextafter(radius, np.inf), axis=-1)


@pytest.mark.parametrize("cellsize", [0.1, 0.25, 1.0])
@pytest.mark.parametrize("radius_cells", [0, 0.4, 0.75, 1, 1.45, 1.55, 2, 2.25])
def test_inflation_matches_world_distance_oracle(cellsize, radius_cells):
    cells = np.zeros((7, 9), dtype=bool)
    cells[0, 0] = cells[3, 4] = cells[6, 1] = True
    grid = BinaryOccupancyGrid(cells, origin=(-2.5, 4.75), cellsize=cellsize)
    radius = radius_cells * cellsize
    original = grid.grid.copy()

    grid.inflate(radius)

    nt.assert_array_equal(grid.grid, inflated_points(cells, cellsize, radius))
    nt.assert_array_equal(cells, original)
    assert grid.grid.dtype == np.bool_
    assert grid.cellsize == cellsize
    nt.assert_allclose(grid.g2w((0, 0)), [-2.5, 4.75])


@pytest.mark.parametrize("cellsize", [0.1, 0.25, 1.0])
@pytest.mark.parametrize("above", [False, True])
def test_diagonal_transition_uses_actual_radius(cellsize, above):
    cells = np.zeros((5, 5), dtype=bool)
    cells[2, 2] = True
    radius = (np.sqrt(2) + (1e-6 if above else -1e-6)) * cellsize
    grid = BinaryOccupancyGrid(cells, cellsize=cellsize)

    grid.inflate(radius)

    assert bool(grid.grid[3, 3]) is above
    assert grid.grid.sum() == (9 if above else 5)


def test_scaled_integer_boundary_is_included():
    # 0.3 / 0.1 is just below 3 in floating point; preserve the exact-cell
    # radius behavior while using a fractional radius elsewhere.
    cells = np.zeros((9, 9), dtype=bool)
    cells[4, 4] = True
    grid = BinaryOccupancyGrid(cells, cellsize=0.1)

    grid.inflate(0.3)

    assert grid.grid[4, 7]
    assert grid.grid[1, 4]
    assert not grid.grid[4, 8]


@pytest.mark.parametrize("value", [False, True])
def test_empty_and_full_maps_are_preserved(value):
    grid = BinaryOccupancyGrid(workspace=[0, 2, 0, 1], cellsize=0.25, value=value)
    before = grid.grid.copy()

    grid.inflate(0.36)

    nt.assert_array_equal(grid.grid, before)


def test_successive_inflations_compound():
    cells = np.zeros((9, 9), dtype=bool)
    cells[4, 4] = True
    grid = BinaryOccupancyGrid(cells, cellsize=0.1)
    expected = inflated_points(cells, 0.1, 0.145)
    expected = inflated_points(expected, 0.1, 0.145)

    grid.inflate(0.145)
    grid.inflate(0.145)

    nt.assert_array_equal(grid.grid, expected)


@pytest.mark.parametrize("cellsize", [0.1, 1.0])
def test_planned_path_obeys_fractional_clearance(cellsize):
    cells = np.zeros((9, 9), dtype=bool)
    cells[4, 4] = True
    grid = BinaryOccupancyGrid(cells, origin=(-2, 5), cellsize=cellsize)
    planner = DistanceTransformPlanner(
        grid, inflate=1.45 * cellsize, goal=grid.g2w((8, 8))
    )
    planner.plan()

    path = planner.query(start=grid.g2w((0, 0)))

    clearances = np.linalg.norm(path - grid.g2w((4, 4)), axis=1)
    assert clearances.min() > 1.45 * cellsize


@pytest.mark.parametrize("cellsize", [0.1, 1.0])
def test_fractional_radius_keeps_a_valid_corridor_open(cellsize):
    cells = np.zeros((9, 7), dtype=bool)
    cells[:, [1, 5]] = True
    grid = BinaryOccupancyGrid(cells, origin=(-2, 5), cellsize=cellsize)
    planner = DistanceTransformPlanner(
        grid, inflate=1.55 * cellsize, goal=grid.g2w((3, 7))
    )
    planner.plan()

    path = planner.query(start=grid.g2w((3, 1)))

    nt.assert_allclose(path[:, 0], grid.g2w((3, 0))[0])
    nt.assert_allclose(path[0], grid.g2w((3, 1)))
    nt.assert_allclose(path[-1], grid.g2w((3, 7)))


@pytest.mark.parametrize("radius", [-0.1, np.nan, np.inf, -np.inf])
def test_invalid_radius_does_not_modify_map(radius):
    cells = np.eye(3, dtype=bool)
    grid = BinaryOccupancyGrid(cells)

    with pytest.raises(ValueError, match="finite and non-negative"):
        grid.inflate(radius)

    nt.assert_array_equal(grid.grid, cells)
