"""
Tests of how a ``workspace`` is resolved by the mobile robot classes.

A workspace is a rectangle ``[xmin, xmax, ymin, ymax]``.  It is given as a
scalar, a 2-vector, a 4-vector or any object that has a ``workspace``
attribute.  When several are available the order of precedence is

1. an explicit ``workspace`` argument,
2. a map that has a concrete extent,
3. the vehicle's workspace,
4. the default.
"""

import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np
import numpy.testing as nt

from roboticstoolbox import (
    EKF,
    Bicycle,
    BinaryOccupancyGrid,
    LandmarkMap,
    PolygonMap,
    RandomPath,
    RangeBearingSensor,
)
from roboticstoolbox.mobile.workspace import resolve_workspace

V = np.diag([0.02, np.radians(0.5)]) ** 2
W = np.diag([0.1, np.radians(1)]) ** 2
P0 = np.diag([0.05, 0.05, np.radians(0.5)]) ** 2


class HasWorkspace:
    def __init__(self, workspace):
        self.workspace = workspace


class TestResolve(unittest.TestCase):
    def test_forms(self):
        nt.assert_array_equal(resolve_workspace(5), [-5, 5, -5, 5])
        nt.assert_array_equal(resolve_workspace([1, 4]), [1, 4, 1, 4])
        nt.assert_array_equal(resolve_workspace((1, 4, 2, 3)), [1, 4, 2, 3])
        nt.assert_array_equal(resolve_workspace(np.r_[1, 4, 2, 3]), [1, 4, 2, 3])

    def test_object_with_workspace(self):
        nt.assert_array_equal(resolve_workspace(HasWorkspace(3)), [-3, 3, -3, 3])
        nt.assert_array_equal(
            resolve_workspace(HasWorkspace([1, 2, 3, 4])), [1, 2, 3, 4]
        )

    def test_none(self):
        self.assertIsNone(resolve_workspace())
        self.assertIsNone(resolve_workspace(None))
        self.assertIsNone(resolve_workspace(None, HasWorkspace(None)))

    def test_precedence_is_argument_order(self):
        nt.assert_array_equal(resolve_workspace(None, 2, 7), [-2, 2, -2, 2])
        # an object that has no workspace is skipped
        nt.assert_array_equal(
            resolve_workspace(HasWorkspace(None), HasWorkspace(4)), [-4, 4, -4, 4]
        )
        # the first wins
        nt.assert_array_equal(resolve_workspace(1, 2), [-1, 1, -1, 1])

    def test_bad(self):
        with self.assertRaises(ValueError):
            resolve_workspace([1, 2, 3])


class TestClasses(unittest.TestCase):
    def test_vehicle_accepts_array(self):
        # a 2-element ndarray used to be an error (truth value of an array)
        veh = Bicycle(workspace=np.array([-5, 5]))
        nt.assert_array_equal(veh.workspace, [-5, 5, -5, 5])

    def test_vehicle_accepts_object(self):
        map = LandmarkMap(5, workspace=[0, 10, 2, 8])
        veh = Bicycle(workspace=map)
        nt.assert_array_equal(veh.workspace, [0, 10, 2, 8])

    def test_vehicle_workspace_is_settable(self):
        veh = Bicycle()
        self.assertIsNone(veh.workspace)
        veh.workspace = 6
        nt.assert_array_equal(veh.workspace, [-6, 6, -6, 6])

    def test_map_accepts_object(self):
        # occupancy grids and polygon maps accept an object with a workspace
        map = LandmarkMap(5, workspace=[0, 10, 2, 8])
        grid = BinaryOccupancyGrid(workspace=map, cellsize=1, value=False)
        nt.assert_array_equal(grid.workspace, [0, 10, 2, 8])
        polys = PolygonMap(workspace=map)
        nt.assert_array_equal(polys.workspace, [0, 10, 2, 8])

    def test_polygonmap_polygons_not_shared(self):
        a = PolygonMap(workspace=10)
        b = PolygonMap(workspace=10)
        a.add([(0, 0), (1, 0), (1, 1)])
        self.assertEqual(len(a.polygons), 1)
        self.assertEqual(len(b.polygons), 0)

    def test_driver_uses_vehicle_workspace(self):
        veh = Bicycle(workspace=8)
        veh.control = RandomPath()
        veh.init(animate=False)
        nt.assert_array_equal(veh.control.workspace, [-8, 8, -8, 8])
        veh.control.demand()  # chooses a goal inside the workspace
        goal = veh.control._goal
        self.assertTrue(np.all(np.abs(goal) <= 8))

    def test_driver_own_workspace_wins(self):
        veh = Bicycle(workspace=8)
        veh.control = RandomPath(workspace=2)
        veh.init(animate=False)
        nt.assert_array_equal(veh.control.workspace, [-2, 2, -2, 2])

    def test_driver_without_any_workspace(self):
        veh = Bicycle()
        veh.control = RandomPath()
        with self.assertRaises(ValueError):
            veh.init(animate=False)


class TestEKFWorkspace(unittest.TestCase):
    def robot(self):
        return Bicycle(covar=V, seed=0)

    def test_explicit_workspace(self):
        # EKF(workspace=...) used to raise NameError
        ekf = EKF(robot=(self.robot(), V), P0=P0, workspace=[0, 4, 0, 6], animate=False)
        nt.assert_array_equal(ekf.workspace, [0, 4, 0, 6])

    def test_precedence(self):
        robot = self.robot()
        robot.workspace = 3
        map = LandmarkMap(10, workspace=[0, 10, 2, 8])
        sensor = RangeBearingSensor(robot, map, covar=W, range=4)

        # the map beats the vehicle
        ekf = EKF(robot=(robot, V), P0=P0, sensor=(sensor, W), animate=False)
        nt.assert_array_equal(ekf.workspace, [0, 10, 2, 8])

        # an explicit workspace beats the map
        ekf = EKF(
            robot=(robot, V), P0=P0, sensor=(sensor, W), workspace=5, animate=False
        )
        nt.assert_array_equal(ekf.workspace, [-5, 5, -5, 5])

    def test_vehicle_only(self):
        robot = self.robot()
        robot.workspace = 3
        ekf = EKF(robot=(robot, V), P0=P0, animate=False)
        nt.assert_array_equal(ekf.workspace, [-3, 3, -3, 3])

    def test_animation_gives_the_vehicle_the_map_workspace(self):
        # through the public property, not the private attribute
        robot = self.robot()
        map = LandmarkMap(10, workspace=[0, 10, 2, 8])
        sensor = RangeBearingSensor(robot, map, covar=W, range=4)
        EKF(robot=(robot, V), P0=P0, sensor=(sensor, W), animate=True)
        nt.assert_array_equal(robot.workspace, [0, 10, 2, 8])


if __name__ == "__main__":
    unittest.main()
