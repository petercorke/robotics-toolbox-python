"""
Tests of the Bug2 reactive navigator and its helper functions.
"""

import importlib
import unittest
import warnings
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from roboticstoolbox.mobile.Bug2 import Bug2, adjacent_point, edgelist, hom_line

# the package exports the class Bug2, which hides the module of the same name
bug2_module = importlib.import_module("roboticstoolbox.mobile.PlannerBase")


def block_map():
    occ = np.zeros((20, 20), dtype=int)
    occ[8:12, 8:12] = 1
    return occ


def ring_map():
    # a hollow square, with the start inside it
    occ = np.zeros((9, 9), dtype=int)
    occ[2:7, 2:7] = 1
    occ[3:6, 3:6] = 0
    return occ


class TestRun(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_path_shape_and_ends(self):
        path = Bug2(occgrid=block_map()).run((2, 10), (17, 10))
        self.assertEqual(path.ndim, 2)
        self.assertEqual(path.shape[1], 2)
        nt.assert_array_equal(path[0], [2, 10])
        nt.assert_array_equal(path[-1], [17, 10])

    def test_no_graphics_without_animation(self):
        # the trail artists used to be created on the current axes regardless,
        # leaving a stray marker at the origin
        plt.close("all")
        Bug2(occgrid=block_map()).run((2, 10), (17, 10))
        self.assertEqual(plt.get_fignums(), [])

    def test_start_equals_goal(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            path = Bug2(occgrid=block_map()).run((2, 2), (2, 2))
        nt.assert_array_equal(path, [[2, 2]])

    def test_unknown_argument_is_an_error(self):
        with self.assertRaises(TypeError):
            Bug2(occgrid=block_map()).run((2, 10), (17, 10), speed=3)

    def test_label_names_the_trail(self):
        with mock.patch.object(plt, "pause"):
            Bug2(occgrid=block_map()).run((2, 10), (17, 10), animate=True, label="mine")
        self.assertIn("mine", [line.get_label() for line in plt.gca().lines])

    def test_diagonal_m_line_through_corners(self):
        # a 45 degree m-line passes between two diagonal neighbours, 0.707 from
        # each, and was missed: the robot was reported as trapped
        path = Bug2(occgrid=block_map()).run((2, 2), (17, 17))
        nt.assert_array_equal(path[-1], [17, 17])

    def test_house_path_length(self):
        # the example in the book, Fig 5.5: 1302 points since 1.4.5, 1307 before
        from roboticstoolbox import rtb_load_matfile

        house = rtb_load_matfile("data/house.mat")
        places = house["places"]
        path = Bug2(occgrid=house["floorplan"]).run(places.br3, places.kitchen)
        self.assertEqual(path.shape, (1302, 2))

    def test_path_has_no_repeated_points(self):
        # a point used to be repeated where the obstacle was first hit, and
        # where the robot left it and found the next step blocked
        from roboticstoolbox import rtb_load_matfile

        house = rtb_load_matfile("data/house.mat")
        places = house["places"]
        path = Bug2(occgrid=house["floorplan"]).run(places.br3, places.kitchen)
        repeats = np.all(np.diff(path, axis=0) == 0, axis=1)
        self.assertEqual(int(repeats.sum()), 0)

    def test_trapped_does_not_block(self):
        # plt.show(block=True) used to be called before the exception
        with mock.patch.object(plt, "show") as show:
            with self.assertRaises(RuntimeError):
                Bug2(occgrid=ring_map()).run((4, 4), (8, 8))
        show.assert_not_called()

    def test_movie(self):
        with mock.patch.object(bug2_module, "MovieWriter") as writer:
            with mock.patch.object(plt, "pause"):
                path = Bug2(occgrid=block_map()).run(
                    (2, 10), (17, 10), movie="bug.mp4", pause=0
                )
        writer.assert_called_once_with("bug.mp4")
        self.assertGreaterEqual(writer.return_value.add.call_count, len(path) - 1)
        writer.return_value.done.assert_called_once()

    def test_query_and_plan_are_not_provided(self):
        bug = Bug2(occgrid=block_map())
        with self.assertRaises(AttributeError):
            bug.query()
        with self.assertRaises(AttributeError):
            bug.plan()


class TestPlotMLine(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_needs_a_run(self):
        with self.assertRaises(ValueError):
            Bug2(occgrid=block_map()).plot_m_line()

    def test_style_for_a_regular_line(self):
        bug = Bug2(occgrid=block_map())
        bug.run((2, 2), (17, 10))
        plt.figure()
        bug.plot_m_line("r-")
        self.assertEqual(plt.gca().lines[-1].get_color(), "r")

    def test_style_for_a_vertical_line(self):
        # the style used to be ignored for a vertical m-line
        bug = Bug2(occgrid=block_map())
        bug.run((2, 2), (2, 15))
        plt.figure()
        bug.plot_m_line("r-")
        self.assertEqual(plt.gca().lines[-1].get_color(), "r")

    def test_keyword_style(self):
        bug = Bug2(occgrid=block_map())
        bug.run((2, 2), (17, 10))
        plt.figure()
        bug.plot_m_line(color="g", linewidth=3)
        line = plt.gca().lines[-1]
        self.assertEqual(line.get_color(), "g")
        self.assertEqual(line.get_linewidth(), 3)


class TestEdgelist(unittest.TestCase):
    def test_region_touching_the_border(self):
        # the edge of an image that is all one region is its border; reading
        # outside the image used to raise, or to wrap around to the far side
        im = np.ones((5, 5), dtype=bool)
        edge, dirs = edgelist(im, (0, 0))
        pts = {tuple(int(v) for v in p) for p in edge}
        border = {
            (x, y) for x in range(5) for y in range(5) if x in (0, 4) or y in (0, 4)
        }
        self.assertEqual(pts, border)
        # the initial point is the first and the last element of the edge list
        nt.assert_array_equal(edge[0], edge[-1])
        self.assertEqual(len(edge), len(dirs))

    def test_blob_in_the_corner(self):
        im = np.zeros((6, 6), dtype=bool)
        im[0:3, 0:3] = True
        edge, dirs = edgelist(im, (0, 0))
        for p in edge:
            self.assertTrue(0 <= p[0] < 6 and 0 <= p[1] < 6)
            self.assertTrue(im[int(p[1]), int(p[0])])

    def test_point_outside_the_image(self):
        with self.assertRaises(ValueError):
            edgelist(np.ones((5, 5), dtype=bool), (9, 9))

    def test_off_the_image_is_outside_the_blob(self):
        # a point on the border of a region that fills the image has a
        # neighbour outside the blob: the one off the edge of the image.  It
        # must not wrap around to the far side of the image.
        im = np.ones((4, 4), dtype=bool)
        q = adjacent_point(im, np.array([0, 1]), True)
        self.assertIsNotNone(q)
        self.assertEqual(int(q[0]), -1)

    def test_interior_point_has_no_outside_neighbour(self):
        im = np.ones((5, 5), dtype=bool)
        self.assertIsNone(adjacent_point(im, np.array([2, 2]), True))
        with self.assertRaises(ValueError):
            edgelist(im, (2, 2))


class TestHomLine(unittest.TestCase):
    def test_line_through_two_points(self):
        line = hom_line((0, 0), (4, 3))
        # a point on the line gives zero, the scale is a unit normal
        self.assertAlmostEqual(float(line @ np.r_[2, 1.5, 1]), 0)
        self.assertAlmostEqual(float(np.linalg.norm(line[:2])), 1)
        # the value at a point is its distance from the line
        self.assertAlmostEqual(abs(float(line @ np.r_[0, 5, 1])), 4.0)

    def test_coincident_points(self):
        with self.assertRaises(ValueError):
            hom_line((1, 1), (1, 1))


if __name__ == "__main__":
    unittest.main()
