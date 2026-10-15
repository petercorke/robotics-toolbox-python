"""
PyPlot backend: things that raised on a headless backend or before launch.
"""

import importlib
import unittest
import unittest.mock as mock
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spatialgeometry as sg

import roboticstoolbox as rtb
from roboticstoolbox.backends.Connector import Connector
from roboticstoolbox.backends.PyPlot import PyPlot, PyPlot2
from roboticstoolbox.tools.plot import set_window_title
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot, ShapePlot


class TestReprBeforeLaunch(unittest.TestCase):
    def test_pyplot(self):
        # sim_time was only set by launch()
        self.assertIn("t = 0", repr(PyPlot()))

    def test_pyplot2(self):
        self.assertIn("t = 0", repr(PyPlot2()))


class TestWindowTitle(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def figure_with(manager):
        return SimpleNamespace(canvas=SimpleNamespace(manager=manager))

    def test_sets_the_title(self):
        manager = mock.Mock()
        self.assertTrue(set_window_title(self.figure_with(manager), "a title"))
        manager.set_window_title.assert_called_once_with("a title")

    def test_no_manager(self):
        self.assertFalse(set_window_title(self.figure_with(None), "a title"))

    def test_manager_without_set_window_title(self):
        class Manager:
            pass

        self.assertFalse(set_window_title(self.figure_with(Manager()), "a title"))

    def test_figure_without_a_canvas_manager_attribute(self):
        self.assertFalse(
            set_window_title(SimpleNamespace(canvas=SimpleNamespace()), "a title")
        )

    def test_both_backends_launch_headless(self):
        # Agg: the manager exists but is not a window
        for cls in (PyPlot, PyPlot2):
            with self.subTest(backend=cls.__name__):
                # the package attribute of the same name is the class, so the
                # module is fetched explicitly (Python 3.10 mock resolves the
                # dotted name to the class)
                module = importlib.import_module(cls.__module__)
                with mock.patch.object(module, "set_window_title", return_value=False):
                    cls().launch()
                plt.close("all")


class TestVehicleWindow(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_animation_without_a_window_manager(self):
        # the title used to be set by reaching into canvas.manager directly
        veh = rtb.Bicycle(workspace=10, animation="car")
        with mock.patch("roboticstoolbox.mobile.Vehicle.set_window_title") as title:
            veh.init(animate=True)
        title.assert_called_once()
        self.assertIn("Robotics Toolbox", title.call_args.args[1])


class TestConnectorProtocol(unittest.TestCase):
    def test_backends_are_connectors(self):
        self.assertIsInstance(PyPlot(), Connector)
        self.assertIsInstance(PyPlot2(), Connector)

    def test_a_class_need_not_inherit(self):
        # Swift does not depend on RTB, so it cannot inherit from Connector
        class Standalone:
            supports_teach = False
            supports_ellipse = False

            def launch(self, *a, **k): ...
            def step(self, *a, **k): ...
            def reset(self, *a, **k): ...
            def restart(self, *a, **k): ...
            def close(self, *a, **k): ...
            def add(self, *a, **k): ...
            def remove(self, *a, **k): ...
            def hold(self, *a, **k): ...
            def _add_teach_panel(self, *a, **k): ...

        self.assertIsInstance(Standalone(), Connector)

    def test_an_object_without_the_methods_is_not(self):
        self.assertNotIsInstance(object(), Connector)

    def test_inheriting_gets_the_defaults(self):
        class Minimal(Connector):
            def launch(self, *a, **k): ...
            def step(self, *a, **k): ...
            def reset(self, *a, **k): ...
            def restart(self, *a, **k): ...
            def close(self, *a, **k): ...
            def add(self, *a, **k): ...
            def remove(self, *a, **k): ...
            def hold(self, *a, **k): ...

        m = Minimal()
        self.assertTrue(m.supports_teach)
        self.assertFalse(m.supports_ellipse)
        with self.assertRaises(NotImplementedError):
            m._add_teach_panel(None, None)

    def test_inheriting_without_a_method_is_an_error(self):
        class Incomplete(Connector):
            def launch(self, *a, **k): ...

        with self.assertRaises(TypeError):
            Incomplete()

    def test_swift_satisfies_the_protocol(self):
        swift = __import__("importlib").util.find_spec("swift")
        if swift is None:
            self.skipTest("swift is not installed")
        from roboticstoolbox.backends.swift import Swift

        for name in (
            "launch",
            "step",
            "reset",
            "restart",
            "close",
            "add",
            "remove",
            "hold",
        ):
            self.assertTrue(callable(getattr(Swift, name, None)), name)


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

    def test_plot_reuses_the_current_3d_axes(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        sp = ShapePlot(sg.Cuboid(scale=[1, 1, 1]))
        sp.plot()
        self.assertIs(sp.ax, ax)
        self.assertEqual(len(plt.get_fignums()), 1)

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
