"""
Animation in the mobile planners: the PlannerBase helper owns the pacing and
the movie file, the planner owns what is drawn.
"""

import importlib
import unittest
import warnings
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboticstoolbox import Bug2, DistanceTransformPlanner, DstarPlanner
from roboticstoolbox.mobile.PlannerBase import PlannerBase

base_module = importlib.import_module("roboticstoolbox.mobile.PlannerBase")


def free_grid(n=12):
    return np.zeros((n, n), dtype=int)


class Planner(PlannerBase):
    pass


class TestHelper(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_defaults(self):
        p = Planner(ndims=2)
        self.assertFalse(p.animate)
        self.assertIsNone(p.movie)

    def test_constructor_options(self):
        p = Planner(ndims=2, animate=True, movie="a.mp4")
        self.assertTrue(p.animate)
        self.assertEqual(p.movie, "a.mp4")

    def test_movie_implies_animate(self):
        p = Planner(ndims=2, movie="a.mp4")
        self.assertTrue(p.animate)

    def test_not_animating_does_nothing(self):
        p = Planner(ndims=2)
        draw = mock.Mock()
        with mock.patch.object(base_module, "MovieWriter") as writer:
            self.assertFalse(p.animation_start(draw=draw))
        draw.assert_not_called()
        writer.assert_not_called()
        self.assertEqual(plt.get_fignums(), [])

    def test_constructor_default_and_call_override(self):
        p = Planner(ndims=2, animate=True)
        with mock.patch.object(base_module, "MovieWriter"):
            self.assertTrue(p.animation_start())
            self.assertFalse(p.animation_start(animate=False))
        p = Planner(ndims=2)
        with mock.patch.object(base_module, "MovieWriter"):
            self.assertTrue(p.animation_start(animate=True))

    def test_draw_then_movie_then_frames(self):
        p = Planner(ndims=2)
        order = []
        with mock.patch.object(base_module, "MovieWriter") as writer:
            writer.side_effect = lambda m: order.append(("writer", m)) or mock.Mock()
            with mock.patch.object(plt, "pause") as pause:
                self.assertTrue(
                    p.animation_start(movie="x.mp4", draw=lambda: order.append("draw"))
                )
                p.animation_frame(0.2)
                p.animation_end()
        self.assertEqual(order, ["draw", ("writer", "x.mp4")])
        pause.assert_called_once_with(0.2)

    def test_call_movie_overrides_constructor_movie(self):
        p = Planner(ndims=2, movie="a.mp4")
        with mock.patch.object(base_module, "MovieWriter") as writer:
            p.animation_start(movie="b.mp4")
        writer.assert_called_once_with("b.mp4")

    def test_end_is_idempotent(self):
        p = Planner(ndims=2)
        p.animation_end()
        with mock.patch.object(base_module, "MovieWriter") as writer:
            p.animation_start(animate=True)
            p.animation_end()
            p.animation_end()
        writer.return_value.done.assert_called_once()


class TestPlanners(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_dx_query_not_animating_draws_nothing(self):
        dx = DistanceTransformPlanner(free_grid(), goal=(10, 10))
        dx.plan()
        dx.query((1, 1))
        self.assertEqual(plt.get_fignums(), [])

    def test_dx_constructor_animate(self):
        dx = DistanceTransformPlanner(free_grid(), goal=(10, 10), animate=True)
        with mock.patch.object(plt, "pause") as pause:
            dx.plan()  # the distance transform animates as well
            before = pause.call_count
            path = dx.query((1, 1))
        self.assertEqual(pause.call_count - before, len(path) - 1)

    def test_dx_movie_is_not_deprecated(self):
        dx = DistanceTransformPlanner(free_grid(), goal=(10, 10))
        dx.plan()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=DeprecationWarning)
            with mock.patch.object(base_module, "MovieWriter") as writer:
                with mock.patch.object(plt, "pause"):
                    path = dx.query((1, 1), movie="dx.mp4")
        writer.assert_called_once_with("dx.mp4")
        self.assertEqual(writer.return_value.add.call_count, len(path) - 1)
        writer.return_value.done.assert_called_once()

    def test_dx_movie_closed_on_failure(self):
        g = free_grid()
        g[:, 5] = 1
        dx = DistanceTransformPlanner(g, goal=(10, 5))
        dx.plan()
        with mock.patch.object(base_module, "MovieWriter") as writer:
            with mock.patch.object(plt, "pause"):
                with self.assertRaises(RuntimeError):
                    dx.query((1, 5), movie="dx.mp4")
        writer.return_value.done.assert_called_once()

    def test_bug2_constructor_animate(self):
        bug = Bug2(occgrid=free_grid(20), animate=True)
        with mock.patch.object(plt, "pause") as pause:
            path = bug.run((1, 1), (15, 15), pause=0.001)
        self.assertGreater(pause.call_count, 0)

    def test_bug2_call_override(self):
        bug = Bug2(occgrid=free_grid(20), animate=True)
        with mock.patch.object(plt, "pause") as pause:
            bug.run((1, 1), (15, 15), animate=False)
        pause.assert_not_called()

    def test_dstar_animate(self):
        costmap = np.ones((8, 8))
        ds = DstarPlanner(costmap, goal=(6, 6))
        ds.plan()
        with mock.patch.object(plt, "pause") as pause:
            path, _ = ds.query((1, 1), animate=True)
        self.assertEqual(pause.call_count, len(path))

    def test_dstar_not_animating(self):
        ds = DstarPlanner(np.ones((8, 8)), goal=(6, 6))
        ds.plan()
        ds.query((1, 1))
        self.assertEqual(plt.get_fignums(), [])

    def test_dstar_plan_animate_is_deprecated(self):
        ds = DstarPlanner(np.ones((8, 8)), goal=(6, 6))
        with self.assertWarns(DeprecationWarning):
            ds.plan(animate=True)

    def test_dstar_does_not_print(self):
        import contextlib
        import io

        costmap = np.ones((8, 8))
        ds = DstarPlanner(costmap, goal=(6, 6))
        ds.plan()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ds.query((1, 1), sensor=lambda p: [(3, 3, np.inf)] if p == (2, 2) else None)
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
