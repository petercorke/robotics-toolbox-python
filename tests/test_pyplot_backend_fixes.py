"""
PyPlot backend: things that raised on a headless backend or before launch.
"""

import unittest
import unittest.mock as mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spatialgeometry as sg

import roboticstoolbox as rtb
from roboticstoolbox.backends.PyPlot import PyPlot, PyPlot2
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot, ShapePlot


class TestReprBeforeLaunch(unittest.TestCase):
    def test_pyplot(self):
        # sim_time was only set by launch()
        self.assertIn("t = 0", repr(PyPlot()))

    def test_pyplot2(self):
        self.assertIn("t = 0", repr(PyPlot2()))


class TestHeadlessLaunch(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_pyplot2_without_a_figure_manager(self):
        with mock.patch.object(plt, "get_current_fig_manager", return_value=None):
            PyPlot2().launch()

    def test_pyplot2_manager_without_set_window_title(self):
        class Manager:
            pass

        with mock.patch.object(plt, "get_current_fig_manager", return_value=Manager()):
            PyPlot2().launch()


class TestShapePlot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_creates_axes(self):
        # used an undefined name ax: NameError
        shape = sg.Cuboid(scale=[1, 2, 3])
        sp = ShapePlot(shape)
        sp.plot()
        self.assertIsNotNone(sp.ax)
        self.assertIsNotNone(sp.mpl)

    def test_plot_into_given_axes(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        sp = ShapePlot(sg.Sphere(radius=0.5))
        sp.plot(ax=ax)
        self.assertIs(sp.ax, ax)

    def test_draw_before_plot(self):
        sp = ShapePlot(sg.Cuboid(scale=[1, 1, 1]))
        sp.draw()
        self.assertIsNotNone(sp.mpl)


class TestEllipsePlot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_unknown_type(self):
        # self.vell was never set: AttributeError at draw time
        robot = rtb.models.DH.Puma560()
        with self.assertRaises(ValueError):
            EllipsePlot(robot, None, "x")

    def test_default_centre_is_not_shared(self):
        robot = rtb.models.DH.Puma560()
        a = EllipsePlot(robot, None, "v")
        b = EllipsePlot(robot, None, "v")
        self.assertIsNot(a.centre, b.centre)

    def test_plot(self):
        robot = rtb.models.DH.Puma560()
        e = EllipsePlot(robot, robot.qn, "v", centre="ee")
        e.plot()
        self.assertIsNotNone(e.ell)


if __name__ == "__main__":
    unittest.main()
