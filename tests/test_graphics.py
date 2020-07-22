#!/usr/bin/env python3

import unittest

from spatialmath import SE3
from vpython import vector, box
from numpy import array
from math import pi
# Can't import the graphics the usual way as it needs to make use of functions not imported through __init__
import graphics.common_functions as common
import graphics.graphics_canvas as canvas
import graphics.graphics_robot as robot


class TestCommonFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.se3 = SE3().Tx(3)

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
    def test_grid_init(self):
        # Create a scene
        scene = canvas.GraphicsCanvas(title="Test Grid Init", grid=False)

        # Create a (technically second) graphics grid for the scene
        grid = canvas.GraphicsGrid(scene.scene)


class TestRobot(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = canvas.GraphicsCanvas()

        #    0.707107 -0.707107  0         0
        #    0.707107  0.707107  0         1
        #    0         0         1         0.4
        #    0         0         0         1
        self.se3 = SE3().Ty(1) * SE3().Tz(0.4) * SE3().Rz(45, 'deg')
        self.structure = 1.0

    def check_obj_pose(self, obj, pose):
        self.assertEqual(common.vpython_to_se3(obj.get_graphic_object()), pose)

    def check_joint_type(self, obj, typ):
        self.assertEqual(obj.get_joint_type(), typ)

    ##################################################
    # Init functions
    ##################################################
    def test_default_joint_init(self):
        self.scene.scene.title = "Test Default Joint init"
        joint = robot.DefaultJoint(self.se3, self.structure, self.scene.scene)
        self.check_obj_pose(joint, self.se3)

        # has int not float
        self.assertRaises(TypeError, robot.DefaultJoint, self.se3, 1, self.scene.scene)
        # has vars in wrong order
        self.assertRaises(TypeError, robot.DefaultJoint, 1.0, self.se3, self.scene.scene)

    def test_rotational_joint_init(self):
        self.scene.scene.title = "Test Rotational Joint init"
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)
        self.check_obj_pose(joint, self.se3)
        self.check_joint_type(joint, "R")

        # has int not float
        self.assertRaises(TypeError, robot.RotationalJoint, self.se3, 1, self.scene.scene)
        # has vars in wrong order
        self.assertRaises(TypeError, robot.RotationalJoint, 1.0, self.se3, self.scene.scene)

    def test_prismatic_joint_init(self):
        self.scene.scene.title = "Test Prismatic Joint init"
        joint = robot.PrismaticJoint(self.se3, self.structure, self.scene.scene)
        self.check_obj_pose(joint, self.se3)
        self.check_joint_type(joint, "P")

        # has int not float
        self.assertRaises(TypeError, robot.PrismaticJoint, self.se3, 1, self.scene.scene)
        # has vars in wrong order
        self.assertRaises(TypeError, robot.PrismaticJoint, 1.0, self.se3, self.scene.scene)

    def test_static_joint_init(self):
        self.scene.scene.title = "Test Static Joint init"
        joint = robot.StaticJoint(self.se3, self.structure, self.scene.scene)
        self.check_obj_pose(joint, self.se3)
        self.check_joint_type(joint, "S")

        # has int not float
        self.assertRaises(TypeError, robot.StaticJoint, self.se3, 1, self.scene.scene)
        # has vars in wrong order
        self.assertRaises(TypeError, robot.StaticJoint, 1.0, self.se3, self.scene.scene)

    def test_gripper_joint_init(self):
        self.scene.scene.title = "Test Gripper Joint init"
        joint = robot.Gripper(self.se3, self.structure, self.scene.scene)
        self.check_obj_pose(joint, self.se3)
        self.check_joint_type(joint, "G")

        # has int not float
        self.assertRaises(TypeError, robot.Gripper, self.se3, 1, self.scene.scene)
        # has vars in wrong order
        self.assertRaises(TypeError, robot.Gripper, 1.0, self.se3, self.scene.scene)

    def test_graphical_robot_init(self):
        self.scene.scene.title = "Test Graphical Robot init"
        robot.GraphicalRobot(self.scene, "Robot 1")

        # Canvas obj given not scene
        self.assertRaises(Exception, robot.GraphicalRobot, self.scene.scene, "Robot 2")

    ##################################################
    # Joint Functions
    ##################################################

    def test_set_joint_position(self):
        # Create a scene
        self.scene.scene.title = "Test Set Joint Position"

        # Create a joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Move joint x+3, y, z-2
        joint.update_position(self.se3 * SE3().Tx(3) * SE3().Tz(-2))

        # Check position
        self.check_obj_pose(joint, self.se3 * SE3().Tx(3) * SE3().Tz(-2))

    def test_set_joint_orientation(self):
        # Create a scene
        self.scene.scene.title = "Test Set Joint Orientation"

        # Create a joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Rotate joint x+30d, y, z+45d
        joint.update_orientation(self.se3 * SE3().Rx(30, 'deg') * SE3().Rz(45, 'deg'))

        # Check position
        self.check_obj_pose(joint, self.se3 * SE3().Rx(30, 'deg') * SE3().Rz(45, 'deg'))

    def test_set_joint_pose(self):
        # Create a scene
        self.scene.scene.title = "Test Set Joint Pose"

        # Create a joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Move joint x+30d, y, z-2
        joint.update_pose(self.se3 * SE3().Rx(30, 'deg') * SE3().Tz(-2))

        # Check position
        self.check_obj_pose(joint, self.se3 * SE3().Rx(30, 'deg') * SE3().Tz(-2))

    def test_draw_reference_frame(self):
        # Scene update
        self.scene.scene.title = "Test Draw Reference Frame"
        self.scene.grid_visibility(False)

        # Create a joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Count num objects
        num_obj_initial = len(self.scene.scene.objects)

        # Turn off reference frame
        joint.draw_reference_frame(False)

        # Count num objects
        num_obj_off = len(self.scene.scene.objects)
        self.assertEqual(num_obj_initial-num_obj_off, 1)

        # Turn on
        joint.draw_reference_frame(True)

        # Count num objects
        num_obj_on = len(self.scene.scene.objects)
        self.assertEqual(num_obj_on - num_obj_off, 1)
        self.assertEqual(num_obj_on, num_obj_initial)

    def test_joint_visibility(self):
        # Scene update
        self.scene.scene.title = "Test Joint Visibility"
        self.scene.grid_visibility(False)

        # Create a joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Count num objects
        num_obj_initial = len(self.scene.scene.objects)

        # Turn off joint graphic
        joint.set_joint_visibility(False)

        # Count num objects
        num_obj_off = len(self.scene.scene.objects)
        self.assertEqual(num_obj_initial - num_obj_off, 1)

        # Turn on
        joint.set_joint_visibility(True)

        # Count num objects
        num_obj_on = len(self.scene.scene.objects)
        self.assertEqual(num_obj_on - num_obj_off, 1)
        self.assertEqual(num_obj_on, num_obj_initial)

    def test_joint_texture(self):
        # Scene update
        self.scene.scene.title = "Test Joint Texture"

        # Create joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Apply texture and colour
        joint.set_texture(
            colour=[0.5, 0, 1],
            texture_link="https://s3.amazonaws.com/glowscript/textures/flower_texture.jpg"
        )

        # Ensure texture is not none, and colour is not white
        gph_obj = joint.get_graphic_object()
        self.assertNotEqual(gph_obj.color, vector(0.5, 0, 1))
        self.assertIsNotNone(gph_obj.texture)

        # Remove Texture and colour
        joint.set_texture()

        # Ensure colour is white, texture is none
        self.assertEqual(gph_obj.color, vector(1, 1, 1))
        self.assertIsNotNone(gph_obj.texture)

        # Apply bad colour
        # Should assert Value Error
        self.assertRaises(ValueError, joint.set_texture, color=[127, 0, 255])

    def test_joint_transparency(self):
        # Scene update
        self.scene.scene.title = "Test Joint Transparency"

        # Create joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Apply texture and colour
        opc_val = 0.34
        joint.set_transparency(opc_val)

        # Ensure texture is not none, and colour is not white
        gph_obj = joint.get_graphic_object()
        self.assertEqual(gph_obj.opacity, opc_val)

        # Set transparency out of range
        # Should throw value error
        self.assertRaises(ValueError, joint.set_transparency, 1.5)

    def test_set_origin(self):
        # Update scene
        self.scene.scene.title = "Test Set Origin"

        # Create a joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Save origin pos
        first_pos = joint.get_graphic_object().origin

        # Move origin
        current_pos = vector(1, 0, -0.5)
        new_pos = vector(0, 0, 0.5)
        joint.set_stl_joint_origin(current_pos, new_pos)

        # Save new origin
        second_pos = joint.get_graphic_object().origin

        # Object should go from along +x-axis, to along -x-axis slightly above z=0 plane
        # Compare
        self.assertEqual(first_pos, vector(0, 0, 0))  # Check original origin is at 0, 0, 0 (Default)
        self.assertEqual(second_pos, new_pos)  # Check new set origin is at the new position

    def test_joint_get_pose(self):
        # Update scene
        self.scene.scene.title = "Test Get Joint Pose"

        # Create a joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Get pose
        pose = joint.get_pose()

        # Check it's equal (proves get returned correctly)
        self.assertEqual(self.se3, pose)

    def test_joint_get_axis_vector(self):
        # Update scene
        self.scene.scene.title = "Test Get Joint Pose"

        # Create a joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)

        # Get axis vector
        x_vec = joint.get_axis_vector(common.x_axis_vector)
        y_vec = joint.get_axis_vector(common.y_axis_vector)
        z_vec = joint.get_axis_vector(common.z_axis_vector)

        # Check it's equal (proves get returned correctly)
        self.assertEqual(common.get_pose_x_vec(self.se3), x_vec)
        self.assertEqual(common.get_pose_y_vec(self.se3), y_vec)
        self.assertEqual(common.get_pose_z_vec(self.se3), z_vec)

        # Check error is thrown
        self.assertRaises(ValueError, joint.get_axis_vector, vector(0.5, 2, 3))

    def test_joint_get_type(self):
        # Update scene
        self.scene.scene.title = "Test Joint Get Type"

        # Create one of each joint
        r = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)
        p = robot.PrismaticJoint(self.se3, self.structure, self.scene.scene)
        s = robot.StaticJoint(self.se3, self.structure, self.scene.scene)
        g = robot.Gripper(self.se3, self.structure, self.scene.scene)

        # Check each is correct
        self.check_joint_type(r, "R")
        self.check_joint_type(p, "P")
        self.check_joint_type(s, "S")
        self.check_joint_type(g, "G")

    def test_joint_get_graphic(self):
        # Update scene
        self.scene.scene.title = "Test Joint Get Graphic"
        self.scene.grid_visibility(False)

        # Create a joint
        joint = robot.RotationalJoint(self.se3, self.structure, self.scene.scene)
        joint.draw_reference_frame(False)

        # Get graphic obj
        gph_obj = joint.get_graphic_object()

        # If obj equal only obj in scene
        self.assertEqual(len(self.scene.scene.objects), 1)
        self.assertEqual(gph_obj, self.scene.scene.objects[0])

    ##################################################
    # Robot functions
    ##################################################

    # def test_robot_append_made_link(self):
    #     raise NotImplementedError
    #
    # def test_robot_append_link(self):
    #     raise NotImplementedError
    #
    # def test_robot_detach_link(self):
    #     raise NotImplementedError
    #
    # def test_robot_reference_visibility(self):
    #     raise NotImplementedError
    #
    # def test_robot_visibility(self):
    #     raise NotImplementedError
    #
    # def test_robot_transparency(self):
    #     raise NotImplementedError
    #
    # def test_robot_set_poses(self):
    #     raise NotImplementedError
    #
    # def test_robot_animate(self):
    #     raise NotImplementedError


class TestStl(unittest.TestCase):
    # def test_import_object(self):
    #     raise NotImplementedError
    pass


class TestText(unittest.TestCase):
    # No functions to really test here
    pass


class TestPuma(unittest.TestCase):
    # def test_import_puma560(self):
    #     raise NotImplementedError
    pass


if __name__ == '__main__':
    unittest.main()
