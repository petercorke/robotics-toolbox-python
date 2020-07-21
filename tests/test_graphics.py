#!/usr/bin/env python3

import unittest

from spatialmath import SE3
from vpython import vector, box
from numpy import array
from math import pi

import graphics.common_functions as common
import graphics.graphics_canvas as canvas
import graphics.graphics_robot as robot


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
    def test_graphics_canvas_init(self):
        # Create a canvas with all options being used (different to defaults)
        scene = canvas.GraphicsCanvas(
            height=360,
            width=480,
            title="Test Graphics Canvas Creation",
            caption="Caption text here",
            grid=False
        )
        try:
            # Put a box in the created scene
            box(canvas=scene.scene)
        except:
            # Something went wrong
            self.assertEqual(False, True)

    def test_grid_visibility(self):
        # Create a scene, with grid=True (default)
        scene = canvas.GraphicsCanvas(title="Test Grid Visibility", grid=True)

        # Check all objects in scene are visible (default scene will just have grid, assuming init has grid=True)
        self.assertGreater(len(scene.scene.objects), 0)

        # Change visibility
        scene.grid_visibility(False)

        # Check all are invisible
        # Invisible objects are not shown in the objects list
        self.assertEqual(len(scene.scene.objects), 0)

    def test_add_robot(self):
        # Create a scene (no grid visible)
        scene = canvas.GraphicsCanvas(title="Test Add Robot", grid=False)

        # Save number of objects
        num_objs = len(scene.scene.objects)

        # Create a 3-link robot
        r = robot.GraphicalRobot(scene, 'robot 1')
        r.append_link('r', SE3(), 1.0)
        r.append_link('r', SE3().Tx(1), 1.0)
        r.append_link('r', SE3().Tx(2), 1.0)
        # Hide reference frames to only have robot joints in visible list
        r.set_reference_visibility(False)

        # Check number of new graphics
        self.assertEqual(len(scene.scene.objects) - num_objs, 3)

    def test_draw_reference_axes(self):
        # Create a scene, no grid
        scene = canvas.GraphicsCanvas(title="Test Draw Reference Frame", grid=False)

        # Check objects is empty
        self.assertEqual(len(scene.scene.objects), 0)

        # Add a reference frame
        arr = array([
            [-1,  0,  0, 3],
            [ 0,  0, -1, 2],
            [ 0, -1,  0, 3],
            [ 0,  0,  0, 1]
        ])
        expected = SE3(arr)
        canvas.draw_reference_frame_axes(expected, scene.scene)

        # Through objects, get position, and vectors
        self.assertEqual(len(scene.scene.objects), 1)
        obj = scene.scene.objects[0]

        pos = obj.pos
        x_vec = obj.axis
        y_vec = obj.up
        z_vec = x_vec.cross(y_vec)

        # Recreate the SE3
        arr = array([
            [x_vec.x, y_vec.x, z_vec.x, pos.x],
            [x_vec.y, y_vec.y, z_vec.y, pos.y],
            [x_vec.z, y_vec.z, z_vec.z, pos.z],
            [0, 0, 0, 1]
        ])
        actual = SE3(arr)

        # Check SE3 are equal
        self.assertEqual(actual, expected)


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
