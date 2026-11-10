"""
Tests of EKF: its description, state, sensor type, map plot and arguments.
"""

import importlib
import unittest
import unittest.mock as mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from roboticstoolbox import (
    EKF,
    Bicycle,
    LandmarkMap,
    RandomPath,
    RangeBearingSensor,
)
from roboticstoolbox.mobile.sensors import SensorBase

V = np.diag([0.02, np.radians(0.5)]) ** 2
W = np.diag([0.1, np.radians(1)]) ** 2
P0 = np.diag([0.05, 0.05, np.radians(0.5)]) ** 2


def make(vehicle=True, map_given=False, sensor=True, p0=None, **kwargs):
    robot = Bicycle(covar=V if vehicle else None)
    robot.control = RandomPath(workspace=10, seed=0)
    landmarks = LandmarkMap(10, workspace=10, seed=0)
    s = (RangeBearingSensor(robot, landmarks, covar=W, seed=0), W) if sensor else None
    return EKF(
        robot=(robot, V if vehicle else None),
        sensor=s,
        map=landmarks if map_given else None,
        P0=(P0 if p0 is None else p0) if vehicle else None,
        animate=False,
        **kwargs,
    )


# roboticstoolbox.mobile.EKF is the class; the module is needed to patch it
ekf_module = importlib.import_module("roboticstoolbox.mobile.EKF")


class TestDescription(unittest.TestCase):
    def estimating(self, ekf):
        line = str(ekf).splitlines()[0]
        return line.split("estimating: ")[1] if "estimating: " in line else ""

    def test_slam(self):
        self.assertEqual(self.estimating(make()), "vehicle pose, map")

    def test_localization(self):
        self.assertEqual(self.estimating(make(map_given=True)), "vehicle pose")

    def test_mapping(self):
        # perfect odometry, the map is estimated
        self.assertEqual(self.estimating(make(vehicle=False)), "map")

    def test_dead_reckoning(self):
        self.assertEqual(self.estimating(make(sensor=False)), "vehicle pose")


class TestState(unittest.TestCase):
    def test_no_camel_case_flag(self):
        # init() set _estVehicle, which nothing read
        self.assertFalse(hasattr(make(), "_estVehicle"))

    def test_init_does_not_alias_p0(self):
        p0 = P0.copy()
        ekf = make(p0=p0)
        self.assertIsNot(ekf.P_est, p0)
        ekf.run(T=2)
        ekf.init()
        self.assertIsNot(ekf.P_est, p0)
        nt.assert_array_equal(ekf.P_est, P0)
        nt.assert_array_equal(p0, P0)

    def test_state_is_an_array(self):
        ekf = make(x0=[0, 0, 0])
        self.assertIsInstance(ekf.x_est, np.ndarray)

    def test_default_x0_is_not_shared(self):
        import inspect

        default = inspect.signature(EKF.__init__).parameters["x0"].default
        self.assertNotIsInstance(default, list)


class TestSensorType(unittest.TestCase):
    def test_only_range_bearing_sensor(self):
        class Other(SensorBase):
            pass

        robot = Bicycle()
        with self.assertRaises(TypeError):
            EKF(
                robot=(robot, None),
                sensor=(Other(robot, LandmarkMap(5, workspace=10)), W),
            )


class TestPlotMap(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @classmethod
    def setUpClass(cls):
        cls.ekf = make()
        cls.ekf.run(T=8)

    def test_ellipses_use_each_landmarks_covariance(self):
        with mock.patch.object(ekf_module.base, "plot_ellipse") as plot_ellipse:
            self.ekf.plot_map(ellipse_style=dict(color="g"))
        self.assertGreater(plot_ellipse.call_count, 1)
        P = self.ekf.P_est
        for i, call in enumerate(plot_ellipse.call_args_list):
            expected = P[3 + 2 * i : 5 + 2 * i, 3 + 2 * i : 5 + 2 * i]
            nt.assert_allclose(call.args[0], expected)

    def test_ellipses_centred_on_the_landmarks(self):
        with mock.patch.object(ekf_module.base, "plot_ellipse") as plot_ellipse:
            self.ekf.plot_map(ellipse_style=dict(color="g"))
        xy = self.ekf.get_map()
        for i, call in enumerate(plot_ellipse.call_args_list):
            nt.assert_allclose(call.kwargs["centre"], xy[i])

    def test_marker_style(self):
        self.ekf.plot_map(marker_style=dict(marker="x", markersize=5, linewidth=0))
        (line,) = plt.gca().lines
        self.assertEqual(line.get_marker(), "x")
        self.assertEqual(line.get_markersize(), 5)

    def test_deprecated_names(self):
        with self.assertWarns(DeprecationWarning):
            self.ekf.plot_map(marker=dict(marker="x", linewidth=0))
        with self.assertWarns(DeprecationWarning):
            self.ekf.plot_map(ellipse=dict(color="g"))
        self.assertGreater(len(plt.gca().patches) + len(plt.gca().lines), 0)

    def test_style_dicts_are_not_modified(self):
        style = dict(color="g")
        self.ekf.plot_map(ellipse_style=style)
        self.assertEqual(style, dict(color="g"))


if __name__ == "__main__":
    unittest.main()
