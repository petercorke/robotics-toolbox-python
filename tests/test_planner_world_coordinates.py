"""Grid planners and their backgrounds share the map's world coordinates."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import (
    BinaryOccupancyGrid,
    Bug2,
    DistanceTransformPlanner,
    DstarPlanner,
    OccupancyGrid,
    PRMPlanner,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(
    params=[
        (1, (0, 0)),
        (0.5, (0, 0)),
        (1, (10, 20)),
        (0.5, (10, 20)),
        (0.1, (-1.3, 2.7)),
    ]
)
def geometry(request):
    return request.param


def floorplan():
    grid = np.zeros((10, 12), dtype=bool)
    grid[3:7, 5:7] = True
    return grid


@pytest.mark.parametrize("metric", ["manhattan", "euclidean"])
def test_distance_transform_world_path(geometry, metric):
    cellsize, origin = geometry
    grid = BinaryOccupancyGrid(floorplan(), cellsize=cellsize, origin=origin)
    planner = DistanceTransformPlanner(grid, metric=metric)
    reference = DistanceTransformPlanner(floorplan(), metric=metric)
    planner.plan(grid.g2w((10, 8)))
    reference.plan((10, 8))
    path = planner.query(grid.g2w((1, 1)))
    expected = reference.query((1, 1)) * cellsize + origin
    nt.assert_allclose(path, expected)
    nt.assert_allclose(planner.distancemap, reference.distancemap * cellsize)
    assert all(not grid.isoccupied(p) for p in path)
    nt.assert_allclose(planner.start, grid.g2w((1, 1)))
    nt.assert_allclose(planner.goal, grid.g2w((10, 8)))


def test_distance_transform_rounds_endpoints_without_truncating_world_values():
    grid = BinaryOccupancyGrid(np.zeros((4, 5)), cellsize=0.5, origin=(10, 20))
    planner = DistanceTransformPlanner(grid)
    planner.plan((11.1, 21.1))
    path = planner.query((10.6, 20.6))
    nt.assert_allclose(path, [[10.5, 20.5], [11, 21]])
    nt.assert_allclose(planner.query((11.05, 21.05)), [[11, 21]])
    assert planner.next((11.05, 21.05)) is None


@pytest.mark.parametrize("cost_grid", [False, True])
def test_dstar_world_path_and_cost(geometry, cost_grid):
    cellsize, origin = geometry
    grid = BinaryOccupancyGrid(floorplan(), cellsize=cellsize, origin=origin)
    costs = np.where(floorplan(), np.inf, 1.0)
    if cost_grid:
        planner = DstarPlanner(OccupancyGrid(costs, cellsize=cellsize, origin=origin))
    else:
        planner = DstarPlanner(occgrid=grid)
    reference = DstarPlanner(costs)
    planner.plan(grid.g2w((10, 8)))
    reference.plan((10, 8))
    path, status = planner.query(grid.g2w((1, 1)))
    reference_path, reference_status = reference.query((1, 1))
    nt.assert_allclose(path[[0, -1]], reference_path[[0, -1]] * cellsize + origin)
    assert status.cost == pytest.approx(reference_status.cost * cellsize)
    assert all(not grid.isoccupied(p) for p in path)
    nt.assert_allclose(planner.goal, grid.g2w((10, 8)))


def test_dstar_sensor_uses_world_coordinates():
    grid = OccupancyGrid(np.ones((5, 6)), cellsize=0.5, origin=(10, 20))
    planner = DstarPlanner(grid)
    planner.plan(grid.g2w((4, 3)))
    positions = []

    def sensor(position):
        positions.append(position)
        # Increase the cost of a distant cell once, then report no changes.
        return [(10, 22, 4)] if len(positions) == 1 else []

    path, _ = planner.query((10.5, 20.5), sensor=sensor)
    nt.assert_allclose(positions, path[:-1])
    assert planner._Map.costmap[4, 0] == 2


def test_bug2_world_path_and_m_line(geometry):
    cellsize, origin = geometry
    grid = BinaryOccupancyGrid(floorplan(), cellsize=cellsize, origin=origin)
    planner = Bug2(occgrid=grid)
    reference = Bug2(occgrid=floorplan())
    path = planner.run(grid.g2w((1, 4)), grid.g2w((10, 4)))
    expected = reference.run((1, 4), (10, 4)) * cellsize + origin
    nt.assert_allclose(path, expected)
    assert all(not grid.isoccupied(p) for p in path)
    for point in path[[0, -1]]:
        assert np.r_[point, 1] @ planner.m_line == pytest.approx(0, abs=1e-12)
    nt.assert_allclose(planner.H, np.array(reference.H) * cellsize + origin)


def test_bug2_diagonal_m_line_at_obstacle_corners(geometry):
    cellsize, origin = geometry
    obstacles = np.zeros((20, 20))
    obstacles[8:12, 8:12] = 1
    grid = BinaryOccupancyGrid(obstacles, cellsize=cellsize, origin=origin)
    planner = Bug2(occgrid=grid)
    path = planner.run(grid.g2w((2, 2)), grid.g2w((17, 17)))
    reference = Bug2(occgrid=obstacles).run((2, 2), (17, 17))
    nt.assert_allclose(path, reference * cellsize + origin)


def test_dstar_rejects_inconsistent_map_geometry():
    costs = OccupancyGrid(np.ones((5, 6)), cellsize=0.5, origin=(10, 20))
    obstacles = BinaryOccupancyGrid(np.zeros((5, 6)))
    with pytest.raises(ValueError, match="same geometry"):
        DstarPlanner(costs, occgrid=obstacles)


def test_prm_roadmap_scales_with_grid(geometry):
    cellsize, origin = geometry
    grid = BinaryOccupancyGrid(floorplan(), cellsize=cellsize, origin=origin)
    planner = PRMPlanner(grid, npoints=30, seed=1, progress=False)
    reference = PRMPlanner(floorplan(), npoints=30, seed=1, progress=False)
    planner.plan()
    reference.plan()
    assert planner.graph.ne == reference.graph.ne
    for actual, expected in zip(planner.graph, reference.graph):
        nt.assert_allclose(actual.coord, expected.coord * cellsize + origin)


def test_field_and_obstacles_share_world_extent_and_axes(geometry):
    cellsize, origin = geometry
    grid = BinaryOccupancyGrid(np.zeros((3, 4)), cellsize=cellsize, origin=origin)
    grid.grid[1, 2] = True
    field = np.arange(12.0).reshape(3, 4)
    field[1, 2] = np.nan
    field[2, 3] = np.inf
    before = field.copy()
    fig, ax = plt.subplots()
    other_fig, other_ax = plt.subplots()  # the requested axes is not current
    grid.plot(field=field, labels=True, ax=ax, colorbar_style={"label": "Cost"})
    expected = grid.workspace + cellsize * np.array([-0.5, 0.5, -0.5, 0.5])
    assert len(ax.images) == 2
    for image in ax.images:
        nt.assert_allclose(image.get_extent(), expected)
    nt.assert_allclose(ax.get_xlim(), expected[:2])
    nt.assert_allclose(ax.get_ylim(), expected[2:])
    assert len(other_ax.images) == 0
    assert len(other_fig.axes) == 1
    assert fig.axes[1].get_ylabel() == "Cost"
    for text in ax.texts:
        c, r = grid.w2g(text.get_position())
        nt.assert_allclose(text.get_position(), grid.g2w((c, r)))
        assert float(text.get_text()) == field[r, c]
    assert len(ax.texts) == 10
    nt.assert_equal(field, before)
    fig.canvas.draw()


@pytest.mark.parametrize("with_field", [False, True])
@pytest.mark.parametrize("show_inflated", [False, True])
def test_inflated_obstacle_colors(with_field, show_inflated):
    grid = BinaryOccupancyGrid(np.zeros((7, 7)), cellsize=0.5, origin=(10, 20))
    grid.grid[3, 3] = True
    planner = DistanceTransformPlanner(grid, inflate=0.5)
    if with_field:
        planner.plan((10.5, 20.5))
    fig, ax = plt.subplots()
    planner.plot_bg(ax=ax, inflated=show_inflated, colorbar=False)
    image = ax.images[-1]
    rgba = image.cmap(image.norm(image.get_array()))
    nt.assert_allclose(rgba[3, 3], [1, 0, 0, 1])
    nt.assert_allclose(rgba[3, 2], [1, 0.75, 0.8, 1] if show_inflated else [1, 1, 1, 0])
    nt.assert_allclose(rgba[0, 0], [1, 1, 1, 0])
    fig.canvas.draw()


@pytest.mark.parametrize(
    "field", [np.array([[0, np.inf]]), np.full((1, 2), np.nan), np.full((1, 2), np.inf)]
)
def test_field_with_no_positive_finite_distance(field):
    grid = BinaryOccupancyGrid(np.zeros((1, 2)), cellsize=0.5, origin=(10, 20))
    fig, ax = plt.subplots()
    grid.plot(field=field, ax=ax, labels=True)
    fig.canvas.draw()
    assert len(ax.texts) == np.isfinite(field).sum()
    image = ax.images[0]
    if np.isinf(field).any():
        rgba = image.cmap(image.norm(image.get_array()))
        nt.assert_allclose(
            rgba[np.isinf(field)], np.tile([0, 0, 1, 1], (np.isinf(field).sum(), 1))
        )


@pytest.mark.parametrize("argument", ["map", "field", "inflated"])
def test_overlay_shape_is_checked(argument):
    grid = BinaryOccupancyGrid(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="same shape"):
        grid.plot(**{argument: np.zeros((4, 3))})


def test_replacement_map_uses_grid_geometry():
    grid = OccupancyGrid(np.zeros((3, 4)), cellsize=0.5, origin=(10, 20))
    _, ax = plt.subplots()
    grid.plot(map=np.ones((3, 4)), ax=ax, cmap="viridis")
    nt.assert_allclose(ax.images[0].get_extent(), [9.75, 11.75, 19.75, 21.25])
    assert ax.images[0].get_cmap().name == "viridis"


def test_labels_skip_inflated_cells_and_preserve_inputs():
    grid = BinaryOccupancyGrid(np.zeros((3, 4)), cellsize=0.5, origin=(10, 20))
    field = np.arange(12.0).reshape(3, 4)
    inflated = np.zeros(grid.shape, dtype=bool)
    inflated[1, 2] = True
    style = {"color": "purple", "fontsize": 12}
    colorbar_style = {"label": "Cost"}
    _, ax = plt.subplots()
    grid.plot(
        field=field,
        inflated=inflated,
        labels=True,
        label_style=style,
        colorbar_style=colorbar_style,
        ax=ax,
    )
    assert len(ax.texts) == 11
    assert all(
        tuple(text.get_position()) != tuple(grid.g2w((2, 1))) for text in ax.texts
    )
    assert {text.get_color() for text in ax.texts} == {"purple"}
    nt.assert_array_equal(field, np.arange(12.0).reshape(3, 4))
    assert inflated.sum() == 1
    assert style == {"color": "purple", "fontsize": 12}
    assert colorbar_style == {"label": "Cost"}


def test_world_path_on_3d_distance_surface():
    grid = BinaryOccupancyGrid(np.zeros((5, 6)), cellsize=0.5, origin=(10, 20))
    planner = DistanceTransformPlanner(grid)
    planner.plan((12, 21.5))
    path = planner.query((10.5, 20.5))
    ax = planner.plot_3d(path, block=None)
    x, y, z = ax.lines[0].get_data_3d()
    nt.assert_allclose(np.c_[x, y], path)
    assert z[-1] == 0
    ax.figure.canvas.draw()


def test_field_accepts_explicit_color_limits():
    grid = BinaryOccupancyGrid(np.zeros((3, 4)))
    _, ax = plt.subplots()
    grid.plot(field=np.arange(12).reshape(3, 4), ax=ax, vmin=0, vmax=20)
    assert ax.images[0].get_clim() == (0, 20)
    ax.figure.canvas.draw()


def test_dstar_sensor_replans_around_world_cell(geometry):
    cellsize, origin = geometry
    grid = OccupancyGrid(np.ones((5, 6)), cellsize=cellsize, origin=origin)
    planner = DstarPlanner(grid)
    planner.plan(grid.g2w((4, 2)))
    blocked = grid.g2w((2, 2))
    calls = 0

    def sensor(position):
        nonlocal calls
        calls += 1
        return [(*blocked, np.inf)] if calls == 1 else []

    path, _ = planner.query(grid.g2w((1, 2)), sensor=sensor)
    nt.assert_allclose(path[-1], grid.g2w((4, 2)))
    assert not np.any(np.all(path == blocked, axis=1))
    assert np.isinf(planner.costmap[2, 2])
