"""
Failure contract of the mobile planners: ``NoPathError`` (a ``RuntimeError``)
for a valid problem with no solution, ``ValueError`` for invalid input, and one
``RuntimeError`` for a query made before a plan.
"""

import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

import roboticstoolbox as rtb
from roboticstoolbox import (
    Bug2,
    DistanceTransformPlanner,
    PRMPlanner,
    QuinticPolyPlanner,
)
from roboticstoolbox.mobile import NoPathError
from roboticstoolbox.mobile.PlannerBase import NoPathError as BaseNoPathError


def walled_grid():
    """10x10 with a full-height wall 3 cells wide, left and right are disconnected"""
    g = np.zeros((10, 10), dtype=int)
    g[:, 4:7] = 1
    return g


class TestNoPathError(unittest.TestCase):
    def test_is_a_runtime_error(self):
        self.assertTrue(issubclass(NoPathError, RuntimeError))
        self.assertIs(NoPathError, BaseNoPathError)
        self.assertIs(rtb.NoPathError, NoPathError)

    def test_distance_transform_unreachable(self):
        dx = DistanceTransformPlanner(walled_grid(), goal=(8, 4))
        dx.plan()
        with self.assertRaises(NoPathError):
            dx.query((1, 1))

    def test_prm_unreachable(self):
        prm = PRMPlanner(
            walled_grid(), npoints=60, dist_thresh=2, seed=0, progress=False
        )
        prm.plan()
        with self.assertRaises(NoPathError):
            prm.query((1, 1), (8, 8))

    def test_quintic_infeasible(self):
        q = QuinticPolyPlanner(max_acc=1e-6, max_jerk=1e-6, max_t=10)
        with self.assertRaises(NoPathError):
            q.query((0, 0, 0), (10, 10, 0))

    def test_bug2_trapped(self):
        g = np.zeros((20, 20), dtype=int)
        g[5:15, 5:15] = 1
        g[7:13, 7:13] = 0  # a hollow, closed box with the start inside
        bug = Bug2(occgrid=g)
        with self.assertRaises(NoPathError):
            bug.run((10, 10), (1, 10))

    def test_bug2_goal_enclosed_by_the_obstacle(self):
        # used to loop forever: leave the wall on the far side, step toward
        # the goal, hit the wall again, and repeat
        g = np.zeros((20, 20), dtype=int)
        g[5:15, 5:15] = 1
        g[7:13, 7:13] = 0  # a closed box with the goal inside
        bug = Bug2(occgrid=g)
        with self.assertRaises(NoPathError):
            bug.run((1, 10), (10, 10))


class TestQueryBeforePlan(unittest.TestCase):
    def test_distance_transform(self):
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        with self.assertRaises(RuntimeError) as cm:
            dx.query((0, 5))
        self.assertNotIsInstance(cm.exception, NoPathError)
        self.assertIn("plan", str(cm.exception))

    def test_prm(self):
        prm = PRMPlanner(np.zeros((10, 10), dtype=int), progress=False)
        with self.assertRaises(RuntimeError) as cm:
            prm.query((1, 1), (8, 8))
        self.assertNotIsInstance(cm.exception, NoPathError)
        self.assertIn("plan", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
