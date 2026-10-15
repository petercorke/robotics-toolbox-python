"""
Tests of the planner base class contract: what PlannerBase declares and how a
planner's start and goal are validated.
"""

import hashlib
import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from roboticstoolbox import (
    Bug2,
    DistanceTransformPlanner,
    DubinsPlanner,
    PRMPlanner,
    ReedsSheppPlanner,
    rtb_load_matfile,
)
from roboticstoolbox.mobile.PlannerBase import PlannerBase


class Planner(PlannerBase):
    """A planner with nothing in it, to test the superclass"""


def grid_with_obstacle():
    g = np.zeros((10, 10), dtype=int)
    g[4:6, 4:6] = 1
    return g


class TestBaseDeclaresNoVerbs(unittest.TestCase):
    def test_not_an_abc(self):
        import abc

        self.assertNotIn(abc.ABC, PlannerBase.__mro__)

    def test_no_plan_next_or_query(self):
        # a planner has the verbs that make sense for it, and only those
        for verb in ("plan", "next", "query", "run"):
            with self.subTest(verb=verb):
                self.assertFalse(hasattr(PlannerBase, verb))

    def test_closed_form_planners_have_no_plan(self):
        # DubinsPlanner().plan() used to be a silent no-op
        for cls in (DubinsPlanner, ReedsSheppPlanner):
            with self.subTest(planner=cls.__name__):
                self.assertFalse(hasattr(cls, "plan"))

    def test_bug2_has_no_plan_or_query(self):
        # it used to inherit them and raise NotImplementedError
        self.assertFalse(hasattr(Bug2, "plan"))
        self.assertFalse(hasattr(Bug2, "query"))
        self.assertTrue(hasattr(Bug2, "run"))


class TestValidateStartGoal(unittest.TestCase):
    def planner(self, **kwargs):
        return Planner(occgrid=grid_with_obstacle(), ndims=2, **kwargs)

    def test_stores_both(self):
        p = self.planner()
        self.assertIsNone(p.validate_start_goal((1, 1), (8, 8)))
        nt.assert_array_equal(p.start, [1, 1])
        nt.assert_array_equal(p.goal, [8, 8])

    def test_none_keeps_the_previous_value(self):
        p = self.planner(start=(1, 1), goal=(8, 8))
        p.validate_start_goal(goal=(7, 7))
        nt.assert_array_equal(p.start, [1, 1])
        nt.assert_array_equal(p.goal, [7, 7])
        p.validate_start_goal()
        nt.assert_array_equal(p.goal, [7, 7])

    def test_dtype(self):
        p = self.planner()
        p.validate_start_goal((1, 1), (8, 8), dtype=int)
        self.assertTrue(np.issubdtype(p.start.dtype, np.integer))

    def test_wrong_length(self):
        p = self.planner()
        with self.assertRaises(ValueError):
            p.validate_start_goal((1, 1, 1), (8, 8))
        with self.assertRaises(ValueError):
            p.validate_start_goal((1, 1), (8,))

    def test_inside_an_obstacle(self):
        p = self.planner()
        with self.assertRaises(ValueError):
            p.validate_start_goal((4, 4), (8, 8))
        with self.assertRaises(ValueError):
            p.validate_start_goal((1, 1), (5, 5))

    def test_nothing_is_stored_if_either_is_invalid(self):
        p = self.planner(start=(1, 1), goal=(8, 8))
        with self.assertRaises(ValueError):
            p.validate_start_goal((2, 2), (5, 5))
        nt.assert_array_equal(p.start, [1, 1])
        nt.assert_array_equal(p.goal, [8, 8])


class TestPlannersStillValidate(unittest.TestCase):
    def test_prm(self):
        prm = PRMPlanner(grid_with_obstacle(), seed=0, progress=False, dist_thresh=20)
        prm.plan(npoints=30)
        with self.assertRaises(ValueError):
            prm.query((4, 4), (8, 8))

    def test_dubins(self):
        with self.assertRaises(ValueError):
            DubinsPlanner().query((0, 0), (1, 0, 0))

    def test_reedsshepp(self):
        with self.assertRaises(ValueError):
            ReedsSheppPlanner().query((0, 0, 0), (1, 0))


class TestDistanceTransformQuery(unittest.TestCase):
    """The stepping loop moved from PlannerBase to DistanceTransformPlanner"""

    @classmethod
    def setUpClass(cls):
        house = rtb_load_matfile("data/house.mat")
        cls.floorplan = house["floorplan"]
        cls.places = house["places"]

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def sha(path):
        return hashlib.sha1(np.ascontiguousarray(path).tobytes()).hexdigest()[:12]

    def test_book_paths_are_unchanged(self):
        for metric in ("euclidean", "manhattan"):
            dx = DistanceTransformPlanner(occgrid=self.floorplan, metric=metric)
            dx.plan(self.places.kitchen)
            with self.subTest(metric=metric, start="br3"):
                p = dx.query(self.places.br3)
                self.assertEqual(p.shape, (338, 2))
                self.assertEqual(self.sha(p), "a20babd0e3e0")
                self.assertEqual(p.dtype, np.int64)
            with self.subTest(metric=metric, start="br2"):
                p = dx.query(self.places.br2)
                self.assertEqual(p.shape, (268, 2))
                self.assertEqual(self.sha(p), "6910b68a0494")

    def test_small_example(self):
        # the example in the class docstring
        g = np.zeros((10, 10), dtype=int)
        g[2:5, 3:5] = 1
        dx = DistanceTransformPlanner(g, goal=(1, 1), metric="manhattan")
        dx.plan()
        path = dx.query(start=(5, 4))
        self.assertEqual(
            path.tolist(), [[5, 4], [5, 3], [5, 2], [4, 1], [3, 1], [2, 1], [1, 1]]
        )

    def test_dtype(self):
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        dx.plan((9, 5))
        self.assertTrue(np.issubdtype(dx.query((0, 5)).dtype, np.integer))
        self.assertTrue(np.issubdtype(dx.query((0, 5), dtype=float).dtype, np.floating))

    def test_no_start(self):
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        dx.plan((9, 5))
        with self.assertRaises(ValueError):
            dx.query()

    def test_next_is_deprecated(self):
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        dx.plan((9, 5))
        with self.assertWarns(DeprecationWarning):
            dx.query((0, 5), next=True)

    def test_query_before_plan(self):
        dx = DistanceTransformPlanner(np.zeros((10, 10), dtype=int))
        with self.assertRaises((ValueError, RuntimeError)):
            dx.query((0, 5))


if __name__ == "__main__":
    unittest.main()
