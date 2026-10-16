"""
Regression tests for mobile planning optimisations.

Compare sparse pose-graph assembly with the original dense equations, and
check that RRT rejects overlong connections before doing collision work.
Wall-clock timings belong in benchmarks, not assertions on a shared CI runner.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.testing as nt
import pgraph
import pytest
import scipy.sparse
import scipy.sparse.linalg
import spatialmath.base as smb
from spatialmath import Polygon2

from roboticstoolbox import Bicycle, PolygonMap, PoseGraph, RRTPlanner
from roboticstoolbox.mobile.PoseGraph import PGEdge, PGVertex
from roboticstoolbox.tools.types import NDArray


def dense_system(pg: PoseGraph) -> tuple[NDArray, NDArray, float]:
    """
    Assemble the original dense normal equations as a reference

    :param pg: pose graph at the current iteration
    :returns: system matrix, right-hand side and total edge error
    """
    H = np.zeros((3 * pg.graph.n, 3 * pg.graph.n))
    b = np.zeros((3 * pg.graph.n, 1))
    energy = 0.0
    for edge in pg.graph.edges():
        e, A, B = edge.linear_factors()
        omega = edge.info
        i, j = [vertex.index for vertex in edge.endpoints]
        si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
        H[si, si] += A.T @ omega @ A
        H[sj, sj] += B.T @ omega @ B
        Hij = A.T @ omega @ B
        H[si, sj] += Hij
        H[sj, si] += Hij.T
        b[si, 0] += -A.T @ omega @ e
        b[sj, 0] += -B.T @ omega @ e
        energy += np.inner(e, e)
    H[:3, :3] += np.eye(3)
    return H, b, energy


@pytest.mark.parametrize("repeated_edge", [False, True])
@pytest.mark.parametrize("offset", [0.0, 3.0])
def test_sparse_posegraph_matches_dense_iterations(
    repeated_edge: bool, offset: float
) -> None:
    pg = PoseGraph("data/pg1.g2o")
    for vertex in pg.graph:
        vertex.coord[2] += offset
    if repeated_edge:
        # Repeated constraints must accumulate into the same blocks, including
        # when the endpoints occur in the opposite order.
        first = pg.graph.edges()[0]
        v1, v2 = first.endpoints
        v1.connect(v2, edge=PGEdge(v1, v2, first.mean.copy(), first.info.copy()))
        mean = smb.tr2xyt(smb.trinv2(smb.xyt2tr(first.mean)))
        v2.connect(v1, edge=PGEdge(v2, v1, mean, first.info.copy()))

    solve = scipy.sparse.linalg.spsolve
    for _ in range(3):
        H, b, energy = dense_system(pg)
        expected = np.array([v.coord.copy() for v in pg.graph])
        expected += solve(scipy.sparse.csr_matrix(H), b).reshape((-1, 3))
        expected[:, 2] = smb.angdiff(expected[:, 2])
        with patch.object(scipy.sparse.linalg, "spsolve", wraps=solve) as solver:
            assert pg.linearize_and_solve() == energy
        matrix, vector = solver.call_args.args
        nt.assert_array_equal(matrix.toarray(), H)
        nt.assert_array_equal(vector, b)
        nt.assert_array_equal(np.array([v.coord for v in pg.graph]), expected)


def test_posegraph_does_not_allocate_a_dense_system() -> None:
    pg = PoseGraph("data/killian-small.toro")
    zeros = np.zeros

    def bounded_zeros(
        shape: int | tuple[int, ...], *args: Any, **kwargs: Any
    ) -> NDArray:
        if isinstance(shape, tuple) and shape == (3 * pg.graph.n, 3 * pg.graph.n):
            pytest.fail("pose-graph assembly allocated a quadratic dense matrix")
        return zeros(shape, *args, **kwargs)

    with patch.object(np, "zeros", side_effect=bounded_zeros):
        assert np.isfinite(pg.linearize_and_solve())


def test_single_pose_without_edges_is_anchored() -> None:
    pg = PoseGraph("data/pg1.g2o")
    pg.graph = pgraph.UGraph()
    vertex = PGVertex("vertex", 0, coord=np.array([2.0, -1.0, 0.4]))
    pg.graph.add_vertex(vertex)
    before = vertex.coord.copy()
    assert pg.linearize_and_solve() == 0
    nt.assert_allclose(vertex.coord, before, atol=1e-15)


@pytest.mark.parametrize("length", [6.01, 20.0])
def test_rrt_rejects_overlong_paths_before_collision_checks(length: float) -> None:
    outline = Polygon2([(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)])
    vehicle = Bicycle(polygon=outline)
    planner = RRTPlanner(
        map=PolygonMap(workspace=[0, 10]), vehicle=vehicle, npoints=1, progress=False
    )
    path = np.array([[1.0, 0, 0], [0.0, 0, 0]])
    # The second candidate is exactly at the length limit and must still have
    # every configuration checked before it can enter the tree.
    candidates = [
        (path, SimpleNamespace(length=length)),
        (path, SimpleNamespace(length=6)),
    ]
    with (
        patch.object(planner, "qrandom_free", return_value=path[0]),
        patch.object(planner.dubins, "query", side_effect=candidates),
        patch.object(vehicle, "polygon", wraps=vehicle.polygon) as polygon,
        patch.object(planner.map, "iscollision", return_value=False) as collision,
    ):
        planner.plan(goal=path[-1], showsamples=False, showvalid=False)
    assert polygon.call_count == len(path)
    assert collision.call_count == len(path)
    assert planner.g.n == 2
    nt.assert_array_equal(planner.g[1].path, path)
