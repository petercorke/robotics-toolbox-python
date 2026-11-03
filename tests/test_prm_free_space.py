"""PRM planning on maps where rejection sampling cannot find free space."""

import numpy as np
import pytest

from roboticstoolbox import NoPathError, PRMPlanner


@pytest.mark.timeout(2)
@pytest.mark.parametrize("inflate", [0, 1])
def test_plan_rejects_maps_without_free_cells(inflate: int) -> None:
    grid = np.ones((5, 7))
    if inflate:
        grid[2, 3] = 0
    planner = PRMPlanner(grid, inflate=inflate, progress=False, seed=0)

    with pytest.raises(NoPathError, match="no free cells"):
        planner.plan(npoints=1)


def test_plan_with_free_cells_builds_a_roadmap() -> None:
    grid = np.ones((5, 7))
    grid[1:4, 2:5] = 0
    planner = PRMPlanner(grid, progress=False, seed=0)

    planner.plan(npoints=5)

    assert planner.graph.n == 5
    assert all(not planner.occgrid.isoccupied(vertex.coord) for vertex in planner.graph)
