"""
Regression tests for the planners and SLAM code that are built on pgraph.

These were previously untested: ``PRMPlanner``, ``RRTPlanner``,
``LatticePlanner`` and ``PoseGraph`` account for roughly 600 statements and had
no tests at all.  They are the whole of RTB's use of pgraph, so they are also
the safety net for pgraph releases.

The assertions are on RTB's own results (endpoints, collision freedom,
determinism, graph sizes, energy decrease) and not on pgraph internals, so that
they hold for any pgraph version.  Graph sizes that are pinned are decided by
RTB's own logic (random seed, move set, lattice merge tolerance).

Tests marked ``expectedFailure`` document defects found while writing this file;
remove the decorator when the defect is fixed.
"""

import contextlib
import io
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pgraph

from roboticstoolbox import (
    BinaryOccupancyGrid,
    LatticePlanner,
    PRMPlanner,
    PoseGraph,
    RRTPlanner,
)
from roboticstoolbox.mobile import Bicycle, PolygonMap
from spatialmath import Polygon2

# pgraph >= 1.0 has a single Vertex class and refuses duplicate vertex names,
# 0.6.x silently replaced the earlier vertex
PGRAPH_REJECTS_DUPLICATES = hasattr(pgraph, "Vertex")


T = TypeVar("T")


def quiet(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call ``func`` discarding anything it prints (the planners are chatty)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def write_graph_file(text: str, suffix: str = ".g2o") -> str:
    """Write a g2o/toro file to a temporary directory and return its path."""
    directory = tempfile.mkdtemp()
    path = Path(directory) / f"graph{suffix}"
    path.write_text(text)
    return str(path)


class TestPRMPlanner(unittest.TestCase):
    START = (2, 2)
    GOAL = (17, 17)

    @staticmethod
    def grid() -> np.ndarray:
        g = np.zeros((20, 20))
        g[5:12, 8:14] = 1  # a block in the middle
        return g

    def planned(self, seed: int = 0, npoints: int = 60) -> PRMPlanner:
        prm = PRMPlanner(self.grid(), seed=seed)
        quiet(prm.plan, npoints=npoints)
        return prm

    def test_roadmap_size(self) -> None:
        prm = self.planned(npoints=60)
        self.assertEqual(prm.graph.n, 60)
        self.assertGreater(prm.graph.ne, 60)  # connected enough to be useful

    def test_query_endpoints_and_shape(self) -> None:
        prm = self.planned()
        path = prm.query(self.START, self.GOAL)

        self.assertEqual(path.ndim, 2)
        self.assertEqual(path.shape[1], 2)
        self.assertGreaterEqual(path.shape[0], 3)
        nt.assert_array_equal(path[0], self.START)
        nt.assert_array_equal(path[-1], self.GOAL)

    def test_roadmap_waypoints_are_in_free_space(self) -> None:
        # waypoints 1..N-2 are roadmap vertices and so must be obstacle free
        prm = self.planned()
        path = prm.query(self.START, self.GOAL)
        for point in path[1:-1]:
            with self.subTest(point=point):
                self.assertFalse(prm.occgrid.isoccupied(point))

    def test_same_seed_same_plan(self) -> None:
        a = self.planned(seed=3).query(self.START, self.GOAL)
        b = self.planned(seed=3).query(self.START, self.GOAL)
        nt.assert_array_equal(a, b)

    def test_different_seed_different_roadmap(self) -> None:
        a = np.array([v.coord for v in self.planned(seed=1).graph])
        b = np.array([v.coord for v in self.planned(seed=2).graph])
        self.assertEqual(a.shape, b.shape)
        self.assertFalse(np.allclose(a, b))

    def test_goal_inside_obstacle(self) -> None:
        prm = self.planned()
        with self.assertRaises(ValueError):
            prm.query(self.START, (10, 8))

    def test_no_path_across_a_wall(self) -> None:
        wall = np.zeros((20, 20))
        wall[:, 9:11] = 1  # splits the map in two
        prm = PRMPlanner(wall, seed=0)
        quiet(prm.plan, npoints=80)
        with self.assertRaises(RuntimeError):
            prm.query(self.START, self.GOAL)

    def test_query_before_plan_is_a_clear_error(self) -> None:
        prm = PRMPlanner(self.grid(), seed=0)
        with self.assertRaises(RuntimeError):
            prm.query(self.START, self.GOAL)

    def test_plot(self) -> None:
        prm = self.planned(npoints=30)
        prm.plot()
        plt.close("all")


class TestRRTPlanner(unittest.TestCase):
    GOAL = (8, 2, -np.pi / 2)
    START = (2, 8, -np.pi / 2)

    def planner(self, seed: int = 0, npoints: int = 15) -> RRTPlanner:
        workspace = PolygonMap(workspace=[0, 10])
        workspace.add([(5, 50), (5, 6), (6, 6), (6, 50)])
        workspace.add([(5, 4), (5, -50), (6, -50), (6, 4)])
        outline = Polygon2([(-1.5, 0.75), (-1.5, -0.75), (1.5, -0.75), (1.5, 0.75)])
        vehicle = Bicycle(steer_max=0.4, L=2, polygon=outline)
        return RRTPlanner(map=workspace, vehicle=vehicle, npoints=npoints, seed=seed)

    def planned(self, **kwargs: Any) -> RRTPlanner:
        rrt = self.planner(**kwargs)
        quiet(rrt.plan, goal=self.GOAL, showsamples=False, showvalid=False)
        return rrt

    def test_tree_size(self) -> None:
        rrt = self.planned(npoints=15)
        # the goal is the root, then one vertex and one edge per sample
        self.assertEqual(rrt.g.n, 16)
        self.assertEqual(rrt.g.ne, 15)

    def test_root_is_goal(self) -> None:
        rrt = self.planned()
        nt.assert_allclose(rrt.g[0].coord, self.GOAL)

    def test_query_reaches_goal(self) -> None:
        rrt = self.planned()
        path, status = quiet(rrt.query, self.START)

        self.assertEqual(path.ndim, 2)
        self.assertEqual(path.shape[1], 3)
        nt.assert_allclose(path[-1], self.GOAL, atol=1e-6)
        nt.assert_allclose(status.vertices[-1].coord, self.GOAL)
        self.assertGreater(status.length, 0)
        self.assertGreaterEqual(status.initial_d, 0)

    def test_path_starts_at_nearest_vertex(self) -> None:
        rrt = self.planned()
        path, status = quiet(rrt.query, self.START)
        nt.assert_allclose(path[0], status.vertices[0].coord, atol=1e-6)

    def test_path_is_collision_free(self) -> None:
        rrt = self.planned()
        path, _ = quiet(rrt.query, self.START)
        for config in path:
            self.assertFalse(rrt.iscollision(config))

    def test_same_seed_same_tree(self) -> None:
        a = self.planned(seed=7)
        b = self.planned(seed=7)
        pa, _ = quiet(a.query, self.START)
        pb, _ = quiet(b.query, self.START)
        nt.assert_array_equal(pa, pb)

    def test_plot(self) -> None:
        rrt = self.planned(npoints=8)
        path, _ = quiet(rrt.query, self.START)
        rrt.plot(path)
        plt.close("all")


class TestLatticePlanner(unittest.TestCase):
    def planned(self, iterations: int) -> LatticePlanner:
        lattice = LatticePlanner()
        quiet(lattice.plan, iterations=iterations)
        return lattice

    def test_graph_growth(self) -> None:
        # (vertices, edges): fixed by the move set and the merge tolerance
        for iterations, expected in {1: (4, 3), 2: (13, 12), 4: (97, 114)}.items():
            with self.subTest(iterations=iterations):
                lattice = self.planned(iterations)
                self.assertEqual((lattice.graph.n, lattice.graph.ne), expected)

    def test_vertex_names_unique(self) -> None:
        lattice = self.planned(6)
        names = [v.name for v in lattice.graph]
        self.assertEqual(len(names), len(set(names)))

    def test_query(self) -> None:
        lattice = self.planned(4)
        start, goal = (0, 0, 0), (1, 2, np.pi / 2)
        path, status = lattice.query(start=start, goal=goal)

        nt.assert_allclose(path[0], start, atol=1e-9)
        nt.assert_allclose(path[-1], goal, atol=1e-9)
        self.assertEqual(len(status.segments), path.shape[0] - 1)
        self.assertEqual(len(status.edges), path.shape[0] - 1)
        self.assertTrue(set(status.segments) <= {"S", "L", "R"})
        # cost is the sum of the edge costs
        self.assertAlmostEqual(status.cost, sum(e.cost for e in status.edges))
        # and a path cannot be shorter than the straight line
        self.assertGreaterEqual(status.cost, np.hypot(1, 2) - 1e-9)

    def test_query_to_self(self) -> None:
        lattice = self.planned(2)
        path, status = lattice.query(start=(0, 0, 0), goal=(0, 0, 0))
        self.assertEqual(path.shape, (1, 3))
        self.assertEqual(status.cost, 0)
        self.assertEqual(status.segments, [])

    def test_start_not_in_lattice(self) -> None:
        lattice = self.planned(4)
        with self.assertRaises(ValueError):
            lattice.query(start=(0.5, 0, 0), goal=(1, 2, np.pi / 2))

    def test_goal_not_in_lattice(self) -> None:
        lattice = self.planned(4)
        with self.assertRaises(ValueError):
            lattice.query(start=(0, 0, 0), goal=(40, 0, 0))

    def test_plan_needs_iterations_without_occgrid(self) -> None:
        with self.assertRaises(ValueError):
            LatticePlanner().plan()

    def test_vertices_avoid_obstacles(self) -> None:
        grid = BinaryOccupancyGrid(workspace=[-5, 5, -5, 5], value=False)
        grid.set([3, 3, -2, 3], True)
        lattice = LatticePlanner(occgrid=grid)
        quiet(lattice.plan, iterations=10)

        self.assertGreater(lattice.graph.n, 100)
        for vertex in lattice.graph:
            self.assertFalse(grid.isoccupied(vertex.coord[:2]))

    def test_plot(self) -> None:
        lattice = self.planned(3)
        path, _ = lattice.query(start=(0, 0, 0), goal=(1, 1, np.pi / 2))
        lattice.plot(path=path)
        lattice.plot(configspace=True)
        plt.close("all")


class TestPoseGraph(unittest.TestCase):
    def test_load_g2o(self) -> None:
        pg = quiet(PoseGraph, "data/pg1.g2o")
        self.assertEqual(pg.graph.n, 4)
        self.assertEqual(pg.graph.ne, 4)
        self.assertEqual([v.name for v in pg.graph], ["1", "2", "3", "4"])
        nt.assert_allclose(pg.graph["2"].coord, [11, 2, 0.1])

    def test_load_toro(self) -> None:
        pg = quiet(PoseGraph, "data/killian-small.toro")
        self.assertEqual(pg.graph.n, 1941)
        self.assertEqual(pg.graph.ne, 3995)

    def test_load_lidar(self) -> None:
        pg = quiet(PoseGraph, "data/killian.g2o.zip", lidar=True)
        self.assertEqual(pg.graph.n, 3873)
        self.assertEqual(pg.graph.ne, 4987)

        scanned = [v for v in pg.graph if hasattr(v, "range")]
        self.assertGreater(len(scanned), 0)
        v = scanned[0]
        self.assertEqual(v.range.shape, v.theta.shape)
        self.assertEqual(v.range.ndim, 1)
        self.assertGreater(v.time, 0)

    def test_lidar_ignored_unless_requested(self) -> None:
        pg = quiet(PoseGraph, "data/killian.g2o.zip")
        self.assertFalse(any(hasattr(v, "range") for v in pg.graph))

    def test_energy_decreases(self) -> None:
        pg = quiet(PoseGraph, "data/pg1.g2o")
        energies = [quiet(pg.linearize_and_solve) for _ in range(5)]

        for before, after in zip(energies[:-1], energies[1:]):
            self.assertLess(after, before)
        self.assertLess(energies[-1], 1e-2 * energies[0])

    def test_optimize_moves_vertices(self) -> None:
        pg = quiet(PoseGraph, "data/pg1.g2o")
        before = np.array([v.coord.copy() for v in pg.graph])
        quiet(pg.optimize, iterations=10)
        after = np.array([v.coord for v in pg.graph])

        self.assertFalse(np.allclose(before, after))
        # a second pass finds nothing left to do
        self.assertLess(quiet(pg.linearize_and_solve), 1e-2)

    def test_optimize_toro(self) -> None:
        pg = quiet(PoseGraph, "data/killian-small.toro")
        first = quiet(pg.linearize_and_solve)
        second = quiet(pg.linearize_and_solve)
        self.assertLess(second, first)

    def test_plot(self) -> None:
        pg = quiet(PoseGraph, "data/pg1.g2o")
        pg.plot()
        plt.close("all")

    def test_unexpected_record(self) -> None:
        path = write_graph_file("VERTEX_SE2 1 0 0 0\nFOO 1 2\n")
        with self.assertRaises(RuntimeError):
            quiet(PoseGraph, path)

    def test_missing_file(self) -> None:
        with self.assertRaises((ValueError, FileNotFoundError)):
            quiet(PoseGraph, "no-such-file.g2o")

    def test_edge_to_unknown_vertex(self) -> None:
        for suffix, vertex_record, edge_record in (
            (".g2o", "VERTEX_SE2", "EDGE_SE2"),
            (".toro", "VERTEX2", "EDGE2"),
        ):
            for start_id, end_id in (("1", "9"), ("9", "1")):
                with self.subTest(suffix=suffix, start=start_id, end=end_id):
                    path = write_graph_file(
                        f"# graph header\n{vertex_record} 1 0 0 0\n"
                        f"{edge_record} {start_id} {end_id} 1 0 0 500 0 0 500 0 500\n",
                        suffix=suffix,
                    )
                    with self.assertRaises(ValueError) as raised:
                        quiet(PoseGraph, path)
                    self.assertIn(path, str(raised.exception))
                    self.assertIn("line 3", str(raised.exception))
                    self.assertIn("'9'", str(raised.exception))
                    self.assertIsInstance(raised.exception.__cause__, KeyError)

    @unittest.skipUnless(
        PGRAPH_REJECTS_DUPLICATES, "pgraph < 1.0 silently replaces the vertex"
    )
    def test_duplicate_vertex_id_rejected(self) -> None:
        path = write_graph_file("VERTEX_SE2 1 0 0 0\nVERTEX_SE2 1 1 1 0\n")
        with self.assertRaises(ValueError):
            quiet(PoseGraph, path)

    def test_blank_line_ignored(self) -> None:
        path = write_graph_file("VERTEX_SE2 1 0 0 0\n\nVERTEX_SE2 2 1 0 0\n")
        pg = quiet(PoseGraph, path)
        self.assertEqual(pg.graph.n, 2)


if __name__ == "__main__":  # pragma nocover
    unittest.main()
