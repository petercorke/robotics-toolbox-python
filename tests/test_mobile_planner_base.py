"""
Tests of PlannerBase: argument checking, start and goal handling, and the
plotting style arguments.
"""

import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from roboticstoolbox import DistanceTransformPlanner
from roboticstoolbox.mobile.PlannerBase import (
    GOAL_STYLE,
    PATH_STYLE,
    START_STYLE,
    PlannerBase,
    _progress,
)


class Planner(PlannerBase):
    """A planner with nothing to plan, to test the superclass"""


def floorplan():
    fp = np.zeros((20, 20), dtype=int)
    fp[8:12, 8:12] = 1
    return fp


def lines_with(ax, **props):
    return [
        line
        for line in ax.lines
        if all(getattr(line, f"get_{k}")() == v for k, v in props.items())
    ]


class TestConstruction(unittest.TestCase):
    def test_occgrid_must_be_supported(self):
        with self.assertRaises(ValueError):
            Planner(occgrid=[[0, 1], [1, 0]], ndims=2)
        with self.assertRaises(ValueError):
            Planner(occgrid=np.zeros((3, 3, 3)), ndims=2)
        self.assertIsNotNone(Planner(occgrid=np.zeros((3, 3)), ndims=2).occgrid)
        self.assertIsNone(Planner(ndims=2).occgrid)

    def test_unknown_argument_is_ignored_with_a_warning(self):
        # a mistyped seed is not silent, but code printed in the book passes
        # arguments that were always ignored, so it is not an error either
        with self.assertWarns(DeprecationWarning) as cm:
            planner = Planner(ndims=2, sead=1)
        self.assertIn("sead", str(cm.warning))
        self.assertIsNone(planner._seed)

    @unittest.skipUnless(_progress, "progress package not installed")
    def test_progress_false_is_honoured(self):
        planner = Planner(ndims=2, progress=False)
        planner.progress_start(5)
        self.assertFalse(hasattr(planner, "_bar"))
        planner.progress_next()
        planner.progress_end()

    def test_start_and_goal_length(self):
        planner = Planner(ndims=2)
        with self.assertRaises(ValueError):
            planner.start = [1, 2, 3]
        with self.assertRaises(ValueError):
            planner.goal = [1]
        planner.start = [1, 2]
        nt.assert_array_equal(planner.start, [1, 2])

        planner3 = Planner(ndims=3)
        with self.assertRaises(ValueError):
            planner3.start = [1, 2]


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.planner = DistanceTransformPlanner(floorplan())
        self.planner.plan((18, 18))

    def tearDown(self):
        plt.close("all")

    def test_path_dtype(self):
        path = self.planner.query((1, 1))
        self.assertTrue(np.issubdtype(path.dtype, np.integer))
        path = self.planner.query((1, 1), dtype=float)
        self.assertTrue(np.issubdtype(path.dtype, np.floating))

    def test_animate_has_no_stray_point(self):
        # the marker size used to be passed as a positional argument, which
        # Matplotlib took as another data series: a point at (0, 12)
        with mock.patch.object(plt, "pause"):
            self.planner.query((1, 1), animate=True)
        singles = [
            (line.get_xdata()[0], line.get_ydata()[0])
            for line in plt.gca().lines
            if len(line.get_xdata()) == 1
        ]
        self.assertNotIn((0, 12), [(float(x), float(y)) for x, y in singles])


class TestPlot(unittest.TestCase):
    def setUp(self):
        self.planner = DistanceTransformPlanner(floorplan())
        self.planner.plan((18, 18))
        self.path = self.planner.query((1, 1))
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.fig)

    def plot(self, **kwargs):
        # the background is not under test here
        self.planner.plot(
            self.path, ax=self.ax, block=False, background=False, **kwargs
        )

    def test_plot_does_not_change_the_goal(self):
        goal = self.planner.goal.copy()
        self.plot(goal=(3, 3))
        nt.assert_array_equal(self.planner.goal, goal)

    def test_plot_bg_does_not_modify_the_distance_field(self):
        distance = np.arange(400.0).reshape(20, 20)
        distance[0, 0] = np.inf
        self.planner.plot_bg(distance=distance, ax=self.ax, colorbar=False)
        self.assertTrue(np.isinf(distance[0, 0]))

    def test_defaults(self):
        self.plot()
        self.assertEqual(len(lines_with(self.ax, color="black", linewidth=4)), 1)
        self.assertEqual(len(lines_with(self.ax, color="yellow", linewidth=3)), 1)
        (start,) = lines_with(self.ax, marker="o")
        (goal,) = lines_with(self.ax, marker="*")
        self.assertEqual(start.get_markersize(), START_STYLE["markersize"])
        self.assertEqual(goal.get_markersize(), GOAL_STYLE["markersize"])

    def test_path_style_layers(self):
        self.plot(
            path_style=(dict(color="m", linewidth=5), dict(color="c", linewidth=2))
        )
        self.assertEqual(len(lines_with(self.ax, color="m", linewidth=5)), 1)
        self.assertEqual(len(lines_with(self.ax, color="c", linewidth=2)), 1)
        self.assertEqual(len(lines_with(self.ax, color="black")), 0)

    def test_path_style_single_dict(self):
        self.plot(path_style=dict(color="m", linewidth=5))
        self.assertEqual(len(lines_with(self.ax, color="m", linewidth=5)), 1)

    def test_path_reverse_style(self):
        n = len(self.path)
        direction = np.r_[np.ones(n // 2), -np.ones(n - n // 2)]
        self.plot(direction=direction, path_reverse_style=dict(color="g", linewidth=6))
        self.assertEqual(len(lines_with(self.ax, color="g", linewidth=6)), 1)

    def test_start_and_goal_style_are_merged_over_the_defaults(self):
        self.plot(start_style=dict(marker="s"), goal_style=dict(markersize=9))
        (start,) = lines_with(self.ax, marker="s")
        self.assertEqual(start.get_markersize(), START_STYLE["markersize"])
        (goal,) = lines_with(self.ax, marker="*")
        self.assertEqual(goal.get_markersize(), 9)
        # the module defaults are not modified
        self.assertEqual(START_STYLE["marker"], "o")
        self.assertEqual(GOAL_STYLE["markersize"], 16)
        self.assertEqual(PATH_STYLE[0]["linewidth"], 4)

    def test_deprecated_names(self):
        cases = [
            ("line", (dict(color="m", linewidth=5),), dict(color="m", linewidth=5)),
            (
                "path_marker",
                (dict(color="m", linewidth=5),),
                dict(color="m", linewidth=5),
            ),
            (
                "start_marker",
                dict(marker="s", markersize=7),
                dict(marker="s", markersize=7),
            ),
            (
                "goal_marker",
                dict(marker="^", markersize=9),
                dict(marker="^", markersize=9),
            ),
        ]
        for old, value, expect in cases:
            with self.subTest(old=old):
                self.ax.clear()
                with self.assertWarns(DeprecationWarning):
                    self.plot(**{old: value})
                self.assertEqual(len(lines_with(self.ax, **expect)), 1)

    def test_deprecated_reverse_names(self):
        n = len(self.path)
        direction = np.r_[np.ones(n // 2), -np.ones(n - n // 2)]
        for old in ("line_r", "path_marker_reverse"):
            with self.subTest(old=old):
                self.ax.clear()
                with self.assertWarns(DeprecationWarning):
                    self.plot(
                        direction=direction, **{old: (dict(color="g", linewidth=6),)}
                    )
                self.assertEqual(len(lines_with(self.ax, color="g", linewidth=6)), 1)


class TestColorbar(unittest.TestCase):
    def setUp(self):
        self.planner = DistanceTransformPlanner(floorplan())
        self.planner.plan((18, 18))
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.fig)

    def test_colorbar_style(self):
        self.planner.plot_bg(ax=self.ax, colorbar=True, colorbar_style=dict(label="d"))
        self.assertEqual(len(self.fig.axes), 2)
        self.assertEqual(self.fig.axes[1].get_ylabel(), "d")

    def test_colorbar_dict_is_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            self.planner.plot_bg(ax=self.ax, colorbar=dict(label="old"))
        self.assertEqual(self.fig.axes[1].get_ylabel(), "old")

    def test_no_colorbar(self):
        self.planner.plot_bg(ax=self.ax, colorbar=False)
        self.assertEqual(len(self.fig.axes), 1)


if __name__ == "__main__":
    unittest.main()
