"""
Tests of the map classes: occupancy grids, the polygon map and the landmark map.
"""

import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
from spatialmath import Polygon2

from roboticstoolbox import BinaryOccupancyGrid, LandmarkMap, PolygonMap


class TestOccupancyGrid(unittest.TestCase):
    def test_unknown_argument_is_an_error(self):
        # ``size`` used to be silently ignored, the argument is ``cellsize``
        with self.assertRaises(TypeError):
            BinaryOccupancyGrid(np.zeros((3, 3)), size=0.1)

    def test_needs_a_grid_or_a_workspace(self):
        with self.assertRaises(ValueError):
            BinaryOccupancyGrid()

    def test_set_region_order(self):
        # the region is [xmin, xmax, ymin, ymax], as for a workspace
        g = BinaryOccupancyGrid(workspace=[0, 9, 0, 9], cellsize=1, value=False)
        g.set([2, 4, 5, 6], True)
        for x in (2, 3, 4):
            for y in (5, 6):
                self.assertTrue(g.isoccupied((x, y)), (x, y))
        self.assertFalse(g.isoccupied((5, 5)))
        self.assertFalse(g.isoccupied((3, 4)))
        self.assertEqual(g.grid.sum(), 6)

    def test_width_and_height(self):
        g = BinaryOccupancyGrid(workspace=[0, 10, 2, 8], cellsize=1, value=False)
        self.assertEqual(g.width, 10)
        self.assertEqual(g.height, 6)

    def test_world_grid_names(self):
        g = BinaryOccupancyGrid(workspace=[0, 10, 2, 8], cellsize=2, value=False)
        nt.assert_array_equal(g.world_to_grid((4, 6)), g.w2g((4, 6)))
        nt.assert_array_equal(g.grid_to_world((1, 2)), g.g2w((1, 2)))
        nt.assert_array_equal(g.grid_to_world((1, 2)), [2, 6])
        nt.assert_array_equal(g.line_world((0, 2), (4, 2)), g.line_w((0, 2), (4, 2)))


class TestPolygonMap(unittest.TestCase):
    def test_collision_accepts_an_array(self):
        pmap = PolygonMap(workspace=10)
        pmap.add([(1, 1), (3, 1), (3, 3), (1, 3)])
        hit = np.array([(2, 2), (4, 2), (4, 4), (2, 4)]).T
        miss = np.array([(5, 5), (6, 5), (6, 6), (5, 6)]).T
        self.assertTrue(pmap.iscollision(hit))
        self.assertFalse(pmap.iscollision(miss))
        self.assertTrue(pmap.iscollision(Polygon2(hit)))


class TestLandmarkMap(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_numpy_integer_count(self):
        m = LandmarkMap(np.int64(7), workspace=10)
        self.assertEqual(len(m), 7)

    def test_plot_default_style(self):
        LandmarkMap(5, workspace=10).plot()
        (line,) = plt.gca().lines
        self.assertEqual(line.get_marker(), "x")
        self.assertEqual(line.get_color(), "black")

    def test_plot_style_is_merged_over_defaults(self):
        # any keyword used to drop the whole default style
        LandmarkMap(5, workspace=10).plot(label="mine", color="r")
        (line,) = plt.gca().lines
        self.assertEqual(line.get_label(), "mine")
        self.assertEqual(line.get_color(), "r")
        self.assertEqual(line.get_marker(), "x")

    def test_verbose_is_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            LandmarkMap(5, workspace=10, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LandmarkMap(5, workspace=10)

    def test_landmarks_argument(self):
        xy = np.array([[1.0, 2, 3], [4, 5, 6]])
        m = LandmarkMap(landmarks=xy, workspace=10)
        self.assertEqual(len(m), 3)
        nt.assert_array_equal(m.landmarks, xy)
        # positional use is unchanged
        self.assertEqual(len(LandmarkMap(20, workspace=10)), 20)

    def test_map_argument_is_deprecated(self):
        xy = np.array([[1.0, 2, 3], [4, 5, 6]])
        with self.assertWarns(DeprecationWarning):
            m = LandmarkMap(map=xy, workspace=10)
        nt.assert_array_equal(m.landmarks, xy)

    def test_no_landmarks_is_an_error(self):
        with self.assertRaises(ValueError):
            LandmarkMap(workspace=10)

    def test_bad_landmarks(self):
        with self.assertRaises(ValueError):
            LandmarkMap("twenty", workspace=10)


if __name__ == "__main__":
    unittest.main()
