"""
Tests of the Matplotlib style arguments of the plotting methods in
``roboticstoolbox.mobile``.

These pin down what each style argument does today, so that the planned renaming
of the arguments (to ``<thing>_style``, with the old names deprecated) and the
merging of user styles over defaults can be made safely.  Each test draws into
an Agg figure and examines the Matplotlib artists, so they are headless.
"""

import contextlib
import io
import importlib
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from roboticstoolbox import (
    EKF,
    Bicycle,
    DistanceTransformPlanner,
    LandmarkMap,
    PRMPlanner,
    RandomPath,
    RangeBearingSensor,
)

V = np.diag([0.02, np.radians(0.5)]) ** 2  # odometry covariance
W = np.diag([0.1, np.radians(1)]) ** 2  # sensor covariance
P0 = np.diag([0.05, 0.05, np.radians(0.5)]) ** 2  # initial vehicle covariance


def quiet(func, *args, **kwargs):
    """Call ``func`` discarding anything it prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def lines_with(ax, **props):
    """Lines in ``ax`` whose Line2D getters match, e.g. ``marker="s"``."""
    found = []
    for line in ax.lines:
        if all(getattr(line, f"get_{k}")() == v for k, v in props.items()):
            found.append(line)
    return found


# roboticstoolbox.mobile.EKF is the class; the module is needed to patch it
ekf_module = importlib.import_module("roboticstoolbox.mobile.EKF")


class PlannerPlotTestCase(unittest.TestCase):
    def setUp(self):
        floorplan = np.zeros((20, 20), dtype=int)
        floorplan[8:12, 8:12] = 1
        self.planner = DistanceTransformPlanner(floorplan)
        self.planner.plan((18, 18))
        self.path = self.planner.query((1, 1))
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.fig)

    def plot(self, **kwargs):
        # the background is the occupancy grid and distance field, not under test
        self.planner.plot(
            self.path, ax=self.ax, block=False, background=False, **kwargs
        )


class TestPlannerPlot(PlannerPlotTestCase):
    def test_default_path_style(self):
        # two layers, a wide black line with a dashed yellow line on top
        self.plot()
        black = lines_with(self.ax, color="black", linewidth=4)
        yellow = lines_with(self.ax, color="yellow", linewidth=3)
        self.assertEqual(len(black), 1)
        self.assertEqual(len(yellow), 1)
        self.assertEqual(len(black[0].get_xdata()), len(self.path))

    def test_line_is_a_sequence_of_layers(self):
        self.plot(line=(dict(color="m", linewidth=5), dict(color="c", linewidth=2)))
        self.assertEqual(len(lines_with(self.ax, color="m", linewidth=5)), 1)
        self.assertEqual(len(lines_with(self.ax, color="c", linewidth=2)), 1)
        # and replaces the default
        self.assertEqual(len(lines_with(self.ax, color="black")), 0)

    def test_line_r_is_used_for_reverse_motion(self):
        n = len(self.path)
        direction = np.r_[np.ones(n // 2), -np.ones(n - n // 2)]
        self.plot(direction=direction, line_r=(dict(color="g", linewidth=6),))
        self.assertEqual(len(lines_with(self.ax, color="g", linewidth=6)), 1)
        # forward motion still uses the default line
        self.assertEqual(len(lines_with(self.ax, color="black", linewidth=4)), 1)

    def test_default_start_and_goal_markers(self):
        self.plot()
        (start,) = lines_with(self.ax, marker="o")
        (goal,) = lines_with(self.ax, marker="*")
        self.assertEqual(start.get_markersize(), 10)
        self.assertEqual(goal.get_markersize(), 16)

    def test_start_and_goal_marker_arguments(self):
        self.plot(
            start_marker=dict(marker="s", color="b", markersize=7),
            goal_marker=dict(marker="^", markersize=9),
        )
        (start,) = lines_with(self.ax, marker="s")
        (goal,) = lines_with(self.ax, marker="^")
        self.assertEqual(start.get_markersize(), 7)
        self.assertEqual(goal.get_markersize(), 9)
        # the user's dict is merged over the defaults: its marker replaces the
        # default marker, the other defaults remain
        self.assertEqual(len(lines_with(self.ax, marker="o")), 0)
        self.assertEqual(len(lines_with(self.ax, marker="*")), 0)

    def test_path_marker_is_a_deprecated_alias(self):
        # ``path_marker`` was in the signature, and the book's figure scripts pass
        # it, but it was never read: the path was drawn from ``line``.  It is now
        # a deprecated alias of ``path_style``.
        self.plot(path_marker=(dict(color="m", linewidth=5),))
        self.assertEqual(len(lines_with(self.ax, color="m", linewidth=5)), 1)


class TestPRMPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prm = PRMPlanner(np.zeros((30, 30)), seed=0, progress=False, dist_thresh=15)
        cls.prm.plan(npoints=15)

    def setUp(self):
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.fig)

    def plot(self, **kwargs):
        quiet(self.prm.plot, ax=self.ax, block=False, **kwargs)

    def test_default_style(self):
        self.plot()
        # one marker per vertex, one patch per edge
        self.assertEqual(len(self.ax.lines), self.prm.graph.n)
        self.assertEqual(len(self.ax.patches), self.prm.graph.ne)
        self.assertEqual({line.get_markersize() for line in self.ax.lines}, {4})
        self.assertEqual({p.get_linewidth() for p in self.ax.patches}, {0.5})

    def test_vertex_and_edge_style(self):
        self.plot(vertex=dict(markersize=9), edge=dict(linewidth=2, color="k"))
        self.assertEqual({line.get_markersize() for line in self.ax.lines}, {9})
        self.assertEqual({p.get_linewidth() for p in self.ax.patches}, {2})
        for p in self.ax.patches:
            nt.assert_allclose(p.get_edgecolor()[:3], (0, 0, 0))

    def test_style_dicts_are_not_modified(self):
        vertex = dict(markersize=9)
        edge = dict(linewidth=2)
        self.plot(vertex=vertex, edge=edge)
        self.assertEqual(vertex, dict(markersize=9))
        self.assertEqual(edge, dict(linewidth=2))


class TestEKFPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        robot = Bicycle(covar=V, seed=0)
        robot.control = RandomPath(workspace=10, seed=0)
        map = LandmarkMap(20, workspace=10, seed=0, verbose=False)
        sensor = RangeBearingSensor(
            robot, map, covar=W, angle=[-np.pi / 2, np.pi / 2], range=4
        )
        cls.ekf = EKF(robot=(robot, V), P0=P0, sensor=(sensor, W), animate=False)
        quiet(cls.ekf.run, T=20)

    def setUp(self):
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.fig)

    def test_plot_xy_accepts_format_string_and_keywords(self):
        self.ekf.plot_xy("r--", label="a")
        self.ekf.plot_xy(color="g", linewidth=3)
        self.assertEqual(len(self.ax.lines), 2)
        self.assertEqual(self.ax.lines[0].get_color(), "r")
        self.assertEqual(self.ax.lines[0].get_linestyle(), "--")
        self.assertEqual(self.ax.lines[1].get_color(), "g")
        self.assertEqual(self.ax.lines[1].get_linewidth(), 3)

    def test_plot_map_default_marker(self):
        self.ekf.plot_map()
        (line,) = self.ax.lines
        self.assertEqual(line.get_marker(), "+")
        self.assertEqual(line.get_markersize(), 10)
        # no ellipses unless asked for
        self.assertEqual(len(self.ax.patches), 0)

    def test_plot_map_marker_argument(self):
        self.ekf.plot_map(marker_style=dict(marker="x", markersize=5, linewidth=0))
        (line,) = self.ax.lines
        self.assertEqual(line.get_marker(), "x")
        self.assertEqual(line.get_markersize(), 5)

    def test_plot_map_ellipse_argument(self):
        # an ellipse for every landmark is drawn only if ``ellipse`` is given
        self.ekf.plot_map(ellipse_style=dict(color="g"))
        nlandmarks = len(self.ekf.landmarks)
        self.assertGreater(nlandmarks, 0)
        # plot_ellipse draws a line per ellipse
        green = lines_with(self.ax, color="g")
        self.assertEqual(len(green), nlandmarks)

    def test_plot_map_ellipse_uses_landmark_covariance(self):
        # The covariance for landmark i is the 2x2 block at rows and columns
        # 3+2i.. (after the vehicle state).  plot_map used to take the block
        # at rows and columns i.., so ellipses after the first were wrong.
        with mock.patch.object(ekf_module.base, "plot_ellipse") as plot_ellipse:
            self.ekf.plot_map(ellipse_style=dict(color="g"))
        P = self.ekf.P_est
        for i, call in enumerate(plot_ellipse.call_args_list):
            expected = P[3 + 2 * i : 5 + 2 * i, 3 + 2 * i : 5 + 2 * i]
            nt.assert_allclose(call.args[0], expected)

    def test_plot_ellipse_forwards_keywords(self):
        self.ekf.plot_ellipse(N=3, color="m", linewidth=2)
        magenta = lines_with(self.ax, color="m", linewidth=2)
        self.assertEqual(len(magenta), 3)


if __name__ == "__main__":
    unittest.main()
