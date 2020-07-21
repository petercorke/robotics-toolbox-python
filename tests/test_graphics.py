#!/usr/bin/env python3

import unittest

from spatialmath import SE3
from vpython import vector, box
from numpy import array
from math import pi

import graphics.common_functions as common
import graphics.graphics_canvas as canvas


class TestCommonFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.se3 = SE3().Tx(3)

    # @classmethod
    # def setUpClass(cls) -> None:
    #     # Load a blank scene, to ensure that the connection is made before tests may be run with it.
    #     scene = canvas.GraphicsCanvas()
    #     scene.scene.waitfor("draw_complete")

    def test_get_pose_x_vector(self):
        self.assertEqual(common.get_pose_x_vec(self.se3), vector(1, 0, 0))

    def test_get_pose_y_vector(self):
        self.assertEqual(common.get_pose_y_vec(self.se3), vector(0, 1, 0))

    def test_get_pose_z_vector(self):
        self.assertEqual(common.get_pose_z_vec(self.se3), vector(0, 0, 1))

    def test_get_pose_pos(self):
        self.assertEqual(common.get_pose_pos(self.se3), vector(3, 0, 0))

    def test_vpython_to_se3(self):
        # Create a scene
        scene = canvas.GraphicsCanvas(title="TEST VPYTHON TO SE3")

        # Create a basic entity
        # pos = 1, 2, 3
        # X = 0, 0, -1
        # Y = -1, 0, 0
        # Z = 0, 1, 0
        entity = box(
            pos=vector(1, 2, 3),
            axis=vector(0, 0, -1),
            up=vector(-1, 0, 0)
        )
        scene.scene.waitfor("draw_complete")

        # Check resulting SE3
        arr = array([
            [ 0, -1, 0, 1],
            [ 0,  0, 1, 2],
            [-1,  0, 0, 3],
            [ 0,  0, 0, 1]
        ])
        expected = SE3(arr)
        self.assertEqual(common.vpython_to_se3(entity), expected)

    def test_wrap_to_pi(self):
        tests = [
            # type, actual, expected
            ['deg', 0, 0],
            ['deg', 50, 50],
            ['deg', 180, 180],
            ['deg', -180, 180],
            ['deg', -181, 179],
            ['deg', 270, -90],
            ['deg', -270, 90],
            ['deg', 360, 0],
            ['deg', -360, 0],
            ['rad', 0, 0],
            ['rad', -3*pi/2, pi/2],
            ['rad', pi/2, pi/2],
            ['rad', pi/4, pi/4],
            ['rad', 10*pi/2, pi],
            ['rad', -5*pi/2, -pi/2]
        ]
        for test in tests:
            self.assertEqual(common.wrap_to_pi(test[0], test[1]), test[2])


class TestCanvas(unittest.TestCase):
    pass


class TestGrid(unittest.TestCase):
    pass


class TestRobot(unittest.TestCase):
    pass


class TestStl(unittest.TestCase):
    pass


class TestText(unittest.TestCase):
    pass


class TestPuma(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
