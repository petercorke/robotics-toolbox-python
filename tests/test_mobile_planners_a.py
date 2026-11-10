"""
Tests of the DistanceTransform, PRM, QuinticPoly and CurvaturePoly planners.
"""

import collections
import contextlib
import io
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from roboticstoolbox import (
    CurvaturePolyPlanner,
    DistanceTransformPlanner,
    PRMPlanner,
    QuinticPolyPlanner,
)
from roboticstoolbox.mobile.DistanceTransformPlanner import distancexform


def quiet(func, *args, **kwargs):
    """Call ``func`` returning what it returned and what it printed"""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = func(*args, **kwargs)
    return result, out.getvalue()


class TestDistanceTransform(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_next_does_not_wrap_around(self):
        # the cell at x = -1 is the last column, which has the goal in it, so
        # a negative index used to give a step off the left edge of the map
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        dx.plan((9, 5))
        position = np.array([0, 5])
        for _ in range(30):
            position = dx.next(position)
            if position is None:
                break
            self.assertTrue(0 <= position[0] < 10 and 0 <= position[1] < 10)
        self.assertIsNone(position)

    def test_unknown_metric(self):
        with self.assertRaises(ValueError):
            distancexform(np.zeros((5, 5)), (1, 1), metric="chebyshev")

    def test_metrics(self):
        grid = np.zeros((5, 5))
        manhattan = distancexform(grid, (0, 0), metric="manhattan")
        euclidean = distancexform(grid, (0, 0), metric="euclidean")
        self.assertEqual(manhattan[4, 4], 8)
        self.assertAlmostEqual(float(euclidean[4, 4]), 4 * np.sqrt(2), places=5)

    def test_plot_3d(self):
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        dx.plan((9, 5))
        path = dx.query((0, 5))
        ax = dx.plot_3d(path, path_style=dict(color="r"), block=False)
        self.assertEqual(ax.name, "3d")
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(ax.lines[0].get_color(), "r")

    def test_plot_3d_ls_is_deprecated(self):
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        dx.plan((9, 5))
        path = dx.query((0, 5))
        with self.assertWarns(DeprecationWarning):
            ax = dx.plot_3d(path, ls=dict(color="g"), block=False)
        self.assertEqual(ax.lines[0].get_color(), "g")

    def test_plot_3d_needs_a_plan(self):
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        with self.assertRaises(ValueError):
            dx.plot_3d(block=False)

    def test_plan_needs_a_goal(self):
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        with self.assertRaises(ValueError):
            dx.plan()


class TestPRM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = np.zeros((30, 30))
        cls.grid[10:20, 10:20] = 1

    def planner(self, **kwargs):
        prm = PRMPlanner(self.grid, seed=0, progress=False, dist_thresh=15, **kwargs)
        prm.plan(npoints=25)
        return prm

    def tearDown(self):
        plt.close("all")

    def test_needs_an_occupancy_grid(self):
        with self.assertRaises(ValueError):
            PRMPlanner()

    def test_plot_prints_nothing(self):
        prm = self.planner()
        _, out = quiet(prm.plot, block=False)
        self.assertEqual(out, "")

    def test_style_names(self):
        prm = self.planner()
        fig, ax = plt.subplots()
        prm.plot(
            ax=ax,
            block=False,
            vertex_style=dict(markersize=9),
            edge_style=dict(linewidth=2),
        )
        self.assertEqual({line.get_markersize() for line in ax.lines}, {9})
        self.assertEqual({p.get_linewidth() for p in ax.patches}, {2})

    def test_style_defaults(self):
        prm = self.planner()
        fig, ax = plt.subplots()
        prm.plot(ax=ax, block=False)
        self.assertEqual({line.get_markersize() for line in ax.lines}, {4})
        self.assertEqual({p.get_linewidth() for p in ax.patches}, {0.5})

    def test_old_style_names_are_deprecated(self):
        prm = self.planner()
        fig, ax = plt.subplots()
        with self.assertWarns(DeprecationWarning):
            prm.plot(
                ax=ax, block=False, vertex=dict(markersize=9), edge=dict(linewidth=2)
            )
        self.assertEqual({line.get_markersize() for line in ax.lines}, {9})
        self.assertEqual({p.get_linewidth() for p in ax.patches}, {2})

    def test_style_dicts_are_not_modified(self):
        prm = self.planner()
        vertex = dict(markersize=9)
        prm.plot(block=False, vertex_style=vertex)
        self.assertEqual(vertex, dict(markersize=9))

    def test_each_connection_is_one_edge(self):
        # the roadmap is undirected, but each connection used to be added from
        # both of its ends, giving two parallel edges
        prm = self.planner()
        graph = prm.graph
        pairs = collections.Counter(
            frozenset((id(e.v1), id(e.v2))) for e in graph.edges()
        )
        self.assertGreater(len(pairs), 0)
        self.assertEqual(set(pairs.values()), {1})
        self.assertEqual(graph.ne, len(pairs))

    def test_edges_are_drawn_straight(self):
        # parallel edges are drawn as arcs, single edges as straight lines
        prm = self.planner()
        fig, ax = plt.subplots()
        prm.plot(ax=ax, block=False)
        curvature = {
            round(p.get_connectionstyle().rad, 3)
            for p in ax.patches
            if hasattr(p, "get_connectionstyle")
        }
        self.assertEqual(curvature, {0.0})

    def test_query_uses_the_constructor_start_and_goal(self):
        prm = PRMPlanner(
            self.grid,
            seed=0,
            progress=False,
            dist_thresh=15,
            start=(2, 2),
            goal=(27, 27),
        )
        prm.plan(npoints=60)
        path = prm.query()
        nt.assert_array_equal(path[0], [2, 2])
        nt.assert_array_equal(path[-1], [27, 27])

    def test_query_needs_a_plan(self):
        prm = PRMPlanner(self.grid, seed=0, progress=False)
        with self.assertRaises(RuntimeError):
            prm.query((2, 2), (27, 27))

    def test_animate_is_deprecated(self):
        prm = PRMPlanner(self.grid, seed=0, progress=False)
        with self.assertWarns(DeprecationWarning):
            prm.plan(npoints=5, animate=True)


class TestQuinticPoly(unittest.TestCase):
    start = (10, 10, np.deg2rad(10.0))
    goal = (30, -10, np.deg2rad(20.0))

    def query(self, **kwargs):
        (path, status), out = quiet(
            QuinticPolyPlanner(**kwargs).query, self.start, self.goal
        )
        return path, status, out

    def test_boundary_velocities(self):
        path, status, _ = self.query(start_vel=1, goal_vel=2)
        self.assertAlmostEqual(status.vel[0], 1, places=6)
        self.assertAlmostEqual(status.vel[-1], 2, places=6)

    def test_goal_velocity_is_independent_of_the_start(self):
        # the goal velocity used to be the start velocity
        _, status, _ = self.query(start_vel=0, goal_vel=1)
        self.assertAlmostEqual(status.vel[0], 0, places=6)
        self.assertAlmostEqual(status.vel[-1], 1, places=6)

    def test_prints_nothing(self):
        _, _, out = self.query(start_vel=1)
        self.assertEqual(out, "")

    def test_impossible_limits(self):
        planner = QuinticPolyPlanner(max_acc=1e-6, max_jerk=1e-6, max_t=20)
        with self.assertRaises(RuntimeError):
            planner.query(self.start, self.goal)

    def test_start_and_goal_are_arrays(self):
        planner = QuinticPolyPlanner(start_vel=1)
        quiet(planner.query, self.start, self.goal)
        self.assertIsInstance(planner.start, np.ndarray)
        nt.assert_allclose(planner.goal, self.goal)

    def test_path_ends(self):
        path, _, _ = self.query(start_vel=1)
        nt.assert_allclose(path[0, :2], self.start[:2])
        nt.assert_allclose(path[-1, :2], self.goal[:2], atol=1e-6)


class TestCurvaturePoly(unittest.TestCase):
    def test_prints_nothing(self):
        planner = CurvaturePolyPlanner()
        (path, status), out = quiet(planner.query, (0, 0, np.pi / 2), (1, 0, np.pi / 2))
        self.assertEqual(out, "")
        nt.assert_allclose(path[0], [0, 0, np.pi / 2], atol=1e-6)
        nt.assert_allclose(path[-1], [1, 0, np.pi / 2], atol=1e-2)
        self.assertGreater(status.length, 1)


if __name__ == "__main__":
    unittest.main()
