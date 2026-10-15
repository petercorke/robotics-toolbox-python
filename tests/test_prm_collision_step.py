"""PRMPlanner connection test: the optional ``collision_step`` spacing"""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from roboticstoolbox import PRMPlanner


def wall_grid(cellsize=1.0):
    # one-cell wide wall at column 5 of a 12 x 12 grid
    g = np.zeros((12, 12), dtype=int)
    g[:, 5] = 1
    return g


def vertex(x, y):
    return SimpleNamespace(coord=np.array([x, y], dtype=float))


class TestCollisionStep(unittest.TestCase):
    def planner(self, **kwargs):
        return PRMPlanner(wall_grid(), progress=False, **kwargs)

    def test_default_is_the_existing_test(self):
        # edge of length 1.9 crossing the wall: 2 samples, both endpoints,
        # so it steps over the one-cell wall.  This is the existing behaviour.
        p = self.planner()
        self.assertTrue(p._test_path(vertex(4.2, 3), vertex(6.1, 3)))

    def test_step_catches_the_thin_wall(self):
        p = self.planner(collision_step=0.5)
        self.assertFalse(p._test_path(vertex(4.2, 3), vertex(6.1, 3)))

    def test_free_edge_is_free(self):
        p = self.planner(collision_step=0.5)
        self.assertTrue(p._test_path(vertex(1, 3), vertex(3.5, 4)))

    def test_short_edges_test_both_ends(self):
        p = self.planner(collision_step=0.5)
        # length 0.3, both ends free, and endpoint inside the wall
        self.assertTrue(p._test_path(vertex(1, 1), vertex(1.3, 1)))
        self.assertFalse(p._test_path(vertex(4.0, 1), vertex(5.0, 1)))

    def test_large_step_is_not_clamped(self):
        # the user's choice, a step of 4 cells is allowed and tests only the ends
        p = self.planner(collision_step=4)
        self.assertEqual(p.collision_step, 4)
        self.assertTrue(p._test_path(vertex(3.2, 3), vertex(6.8, 3)))

    def test_step_is_in_cells(self):
        # same world geometry on a grid of 2-unit cells: 0.5 cells = 1 unit
        from roboticstoolbox import BinaryOccupancyGrid

        g = np.zeros((12, 12), dtype=int)
        g[:, 3] = 1  # world x in [5, 7] for cellsize 2
        occ = BinaryOccupancyGrid(g, cellsize=2)
        p = PRMPlanner(occ, progress=False, collision_step=0.5)
        samples = []
        orig = occ.isoccupied
        occ.isoccupied = lambda q: samples.append(np.array(q)) or orig(q)
        p._test_path(vertex(0, 0), vertex(8, 0))
        spacing = np.diff([s[0] for s in samples])
        self.assertTrue(np.all(spacing <= 1.0 + 1e-9))

    def test_invalid(self):
        for bad in (0, -1):
            with self.subTest(step=bad):
                with self.assertRaises(ValueError):
                    self.planner(collision_step=bad)


if __name__ == "__main__":
    unittest.main()
