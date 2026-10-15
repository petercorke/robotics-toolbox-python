"""Regression tests for PRM constructor distance thresholds."""

import unittest
from unittest.mock import patch

import numpy as np

from roboticstoolbox.mobile import PRMPlanner


class TestPRMConstructor(unittest.TestCase):
    def test_default_distance_threshold(self):
        """The default threshold is derived from the occupancy grid size."""
        planner = PRMPlanner(np.zeros((10, 20)), progress=False)

        expected = 0.3 * planner.occgrid.maxdim
        self.assertEqual(planner.dist_thresh, expected)
        self.assertEqual(planner._dist_thresh0, expected)

    def test_explicit_distance_threshold_reaches_planner(self):
        """Explicit thresholds, including zero, remain available during planning."""
        grid = np.zeros((10, 20))
        for threshold in (0.0, 2.5):
            with self.subTest(threshold=threshold):
                planner = PRMPlanner(grid, dist_thresh=threshold, progress=False)
                self.assertEqual(planner.dist_thresh, threshold)
                self.assertEqual(planner._dist_thresh0, threshold)

                with patch.object(planner, "_create_roadmap") as create_roadmap:
                    planner.plan(npoints=1)
                create_roadmap.assert_called_once_with(1, threshold)

                with patch.object(planner, "_create_roadmap") as create_roadmap:
                    planner.plan(npoints=1, dist_thresh=3.0)
                create_roadmap.assert_called_once_with(1, 3.0)
                self.assertEqual(planner.dist_thresh, threshold)
