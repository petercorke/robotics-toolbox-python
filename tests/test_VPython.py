# #!/usr/bin/env python3

# import unittest
# import warnings

# from spatialmath import SE3
# from vpython import vector, box
# from numpy import array
# from math import pi
# import time

# from roboticstoolbox.backends import VPython
# from roboticstoolbox.models.DH.Puma560 import Puma560

# from roboticstoolbox.backends.VPython.common_functions import \
#     get_pose_x_vec, get_pose_y_vec, get_pose_z_vec, get_pose_pos, \
#     vpython_to_se3, wrap_to_pi, close_localhost_session, \
#     x_axis_vector, y_axis_vector, z_axis_vector
# from roboticstoolbox.backends.VPython.canvas import GraphicsCanvas3D, \
#     draw_reference_frame_axes
# from roboticstoolbox.backends.VPython.graphicalrobot import GraphicalRobot, \
#     DefaultJoint, RotationalJoint, PrismaticJoint, StaticJoint, Gripper
# from roboticstoolbox.backends.VPython.stl import import_object_from_numpy_stl
# from roboticstoolbox.backends.VPython.grid import GraphicsGrid


# class TestVPython(unittest.TestCase):

#     env = None

#     @classmethod
#     def setUpClass(cls):
#         cls.env = VPython()

#     def setUp(self):
#         self.env.launch()
#         # Get the last scene
#         self.robot_scene = self.env.canvases[len(self.env.canvases)-1]

#         #    0.707107 -0.707107  0         0
#         #    0.707107  0.707107  0         1
#         #    0         0         1         0.4
#         #    0         0         0         1
#         self.robot_se3 = SE3().Ty(1) * SE3().Tz(0.4) * SE3().Rz(45, 'deg')
#         self.robot_structure = 1.0
#         self.se3 = SE3().Tx(3)
#         warnings.simplefilter('ignore', category=ResourceWarning)

#     @classmethod
#     def tearDownClass(cls):
#         with cls.assertRaises(cls, SystemExit):
#             cls.env.close()
#             # Give time for VPython to exit
#             time.sleep(1)

#     #  TODO
#     #   COMMENTED OUT UNTIL CAN RUN ON GITHUB ACTIONS
#     #   CAN STILL BE EXECUTED ON PERSONAL MACHINE LOCALLY
#     #####################################################################
#     # def test_get_pose_x_vector(self):
#     #     self.assertEqual(get_pose_x_vec(self.se3), vector(1, 0, 0))

#     # def test_get_pose_y_vector(self):
#     #     self.assertEqual(get_pose_y_vec(self.se3), vector(0, 1, 0))

#     # def test_get_pose_z_vector(self):
#     #     self.assertEqual(get_pose_z_vec(self.se3), vector(0, 0, 1))

#     # def test_get_pose_pos(self):
#     #     self.assertEqual(get_pose_pos(self.se3), vector(3, 0, 0))

#     # def test_vpython_to_se3(self):
#     #     # Create a scene
#     #     scene = GraphicsCanvas3D(title="TEST VPYTHON TO SE3")

#     #     # Create a basic entity
#     #     # pos = 1, 2, 3
#     #     # X = 0, 0, -1
#     #     # Y = -1, 0, 0
#     #     # Z = 0, 1, 0
#     #     entity = box(
#     #         pos=vector(1, 2, 3),
#     #         axis=vector(0, 0, -1),
#     #         up=vector(-1, 0, 0)
#     #     )
#     #     scene.scene.waitfor("draw_complete")

#     #     # Check resulting SE3
#     #     arr = array([
#     #         [0, -1, 0, 1],
#     #         [0, 0, 1, 2],
#     #         [-1, 0, 0, 3],
#     #         [0, 0, 0, 1]
#     #     ])
#     #     expected = SE3(arr)
#     #     self.assertEqual(vpython_to_se3(entity), expected)

#     # def test_wrap_to_pi(self):
#     #     tests = [
#     #         # type, actual, expected
#     #         ['deg', 0, 0],
#     #         ['deg', 50, 50],
#     #         ['deg', 180, 180],
#     #         ['deg', -180, 180],
#     #         ['deg', -181, 179],
#     #         ['deg', 270, -90],
#     #         ['deg', -270, 90],
#     #         ['deg', 360, 0],
#     #         ['deg', -360, 0],
#     #         ['rad', 0, 0],
#     #         ['rad', -3 * pi / 2, pi / 2],
#     #         ['rad', pi / 2, pi / 2],
#     #         ['rad', pi / 4, pi / 4],
#     #         ['rad', 10 * pi / 2, pi],
#     #         ['rad', -5 * pi / 2, -pi / 2]
#     #     ]
#     #     for test in tests:
#     #         self.assertEqual(wrap_to_pi(test[0], test[1]), test[2])

#     # ########################################################################
#     # def test_graphics_canvas_init(self):
#     #     # Create a canvas with all options being used (different to defaults)
#     #     scene = GraphicsCanvas3D(
#     #         height=360,
#     #         width=480,
#     #         title="Test Graphics Canvas Creation",
#     #         caption="Caption text here",
#     #         grid=False
#     #     )
#     #     try:
#     #         # Put a box in the created scene
#     #         box(canvas=scene.scene)
#     #     except Exception:
#     #         # Something went wrong
#     #         self.assertEqual(False, True)

#     # def test_grid_visibility(self):
#     #     # Create a scene, with grid=True (default)
#     #     scene = GraphicsCanvas3D(title="Test Grid Visibility", grid=True)

#     #     # Check all objects in scene are visible (default scene will just have
#     #     # grid, assuming init has grid=True)
#     #     self.assertGreater(len(scene.scene.objects), 0)

#     #     # Change visibility
#     #     scene.grid_visibility(False)

#     #     # Check all are invisible
#     #     # Invisible objects are not shown in the objects list
#     #     self.assertEqual(len(scene.scene.objects), 0)

#     # def test_add_robot(self):
#     #     # Create a scene (no grid visible)
#     #     scene = GraphicsCanvas3D(title="Test Add Robot", grid=False)

#     #     # Save number of objects
#     #     num_objs = len(scene.scene.objects)

#     #     # Create a 3-link robot
#     #     r = GraphicalRobot(scene, 'robot 1')
#     #     r.append_link('r', SE3(), 1.0, [0, 0], 0)
#     #     r.append_link('r', SE3().Tx(1), 1.0, [0, 0], 0)
#     #     r.append_link('r', SE3().Tx(2), 1.0, [0, 0], 0)
#     #     # Hide reference frames to only have robot joints in visible list
#     #     r.set_reference_visibility(False)

#     #     # Check number of new graphics
#     #     self.assertEqual(len(scene.scene.objects) - num_objs, 3)

#     # def test_draw_reference_axes(self):
#     #     # Create a scene, no grid
#     #     scene = GraphicsCanvas3D(title="Test Draw Reference Frame", grid=False)

#     #     # Check objects is empty
#     #     self.assertEqual(len(scene.scene.objects), 0)

#     #     # Add a reference frame
#     #     arr = array([
#     #         [-1, 0, 0, 3],
#     #         [0, 0, -1, 2],
#     #         [0, -1, 0, 3],
#     #         [0, 0, 0, 1]
#     #     ])
#     #     expected = SE3(arr)
#     #     draw_reference_frame_axes(expected, scene.scene)

#     #     # Through objects, get position, and vectors
#     #     self.assertEqual(len(scene.scene.objects), 1)
#     #     obj = scene.scene.objects[0]

#     #     pos = obj.pos
#     #     x_vec = obj.axis
#     #     y_vec = obj.up
#     #     z_vec = x_vec.cross(y_vec)

#     #     # Recreate the SE3
#     #     arr = array([
#     #         [x_vec.x, y_vec.x, z_vec.x, pos.x],
#     #         [x_vec.y, y_vec.y, z_vec.y, pos.y],
#     #         [x_vec.z, y_vec.z, z_vec.z, pos.z],
#     #         [0, 0, 0, 1]
#     #     ])
#     #     actual = SE3(arr)

#     #     # Check SE3 are equal
#     #     self.assertEqual(actual, expected)

#     # def test_grid_init(self):
#     #     # Create a scene
#     #     scene = GraphicsCanvas3D(title="Test Grid Init", grid=False)

#     #     # Create a (technically second) graphics grid for the scene
#     #     # grid = GraphicsGrid(scene.scene)
#     #     GraphicsGrid(scene.scene)

#     # ##########################################################################
#     # def check_obj_pose(self, obj, pose):
#     #     self.assertEqual(vpython_to_se3(obj.get_graphic_object()), pose)

#     # def check_joint_type(self, obj, typ):
#     #     self.assertEqual(obj.get_joint_type(), typ)

#     # ##################################################
#     # # Init functions
#     # ##################################################
#     # def test_default_joint_init(self):
#     #     self.robot_scene.scene.title = "Test Default Joint init"
#     #     joint = DefaultJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     self.check_obj_pose(joint, self.robot_se3)

#     #     # has int not float
#     #     self.assertRaises(
#     #         TypeError, DefaultJoint, self.robot_se3, 1, self.robot_scene)
#     #     # has vars in wrong order
#     #     self.assertRaises(
#     #         TypeError, DefaultJoint, 1.0, self.robot_se3, self.robot_scene)

#     # def test_rotational_joint_init(self):
#     #     self.robot_scene.scene.title = "Test Rotational Joint init"
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     self.check_obj_pose(joint, self.robot_se3)
#     #     self.check_joint_type(joint, "R")

#     #     # has int not float
#     #     self.assertRaises(
#     #         TypeError, RotationalJoint, self.robot_se3, 1, self.robot_scene)
#     #     # has vars in wrong order
#     #     self.assertRaises(
#     #         TypeError, RotationalJoint, 1.0, self.robot_se3, self.robot_scene)

#     # def test_prismatic_joint_init(self):
#     #     self.robot_scene.scene.title = "Test Prismatic Joint init"
#     #     joint = PrismaticJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     self.check_obj_pose(joint, self.robot_se3)
#     #     self.check_joint_type(joint, "P")

#     #     # has int not float
#     #     self.assertRaises(
#     #         TypeError, PrismaticJoint, self.robot_se3, 1, self.robot_scene)
#     #     # has vars in wrong order
#     #     self.assertRaises(
#     #         TypeError, PrismaticJoint, 1.0, self.robot_se3, self.robot_scene)

#     # def test_static_joint_init(self):
#     #     self.robot_scene.scene.title = "Test Static Joint init"
#     #     joint = StaticJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     self.check_obj_pose(joint, self.robot_se3)
#     #     self.check_joint_type(joint, "S")

#     #     # has int not float
#     #     self.assertRaises(
#     #         TypeError, StaticJoint, self.robot_se3, 1, self.robot_scene)
#     #     # has vars in wrong order
#     #     self.assertRaises(
#     #         TypeError, StaticJoint, 1.0, self.robot_se3, self.robot_scene)

#     # def test_gripper_joint_init(self):
#     #     self.robot_scene.scene.title = "Test Gripper Joint init"
#     #     joint = Gripper(self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     self.check_obj_pose(joint, self.robot_se3)
#     #     self.check_joint_type(joint, "G")

#     #     # has int not float
#     #     self.assertRaises(
#     #         TypeError, Gripper, self.robot_se3, 1, self.robot_scene)
#     #     # has vars in wrong order
#     #     self.assertRaises(
#     #         TypeError, Gripper, 1.0, self.robot_se3, self.robot_scene)

#     # def test_graphical_robot_init(self):
#     #     self.robot_scene.scene.title = "Test Graphical Robot init"
#     #     GraphicalRobot(self.robot_scene, "Robot 1")

#     #     # Scene obj given not canvas
#     #     self.assertRaises(
#     #         Exception, GraphicalRobot, self.robot_scene.scene, "Robot 2")

#     # ##################################################
#     # # Joint Functions
#     # ##################################################
#     # def test_set_joint_position(self):
#     #     # Create a scene
#     #     self.robot_scene.scene.title = "Test Set Joint Position"

#     #     # Create a joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Move joint x+3, y, z-2
#     #     joint.update_position(self.robot_se3 * SE3().Tx(3) * SE3().Tz(-2))

#     #     # Check position
#     #     self.check_obj_pose(joint, self.robot_se3 * SE3().Tx(3) * SE3().Tz(-2))

#     # def test_set_joint_orientation(self):
#     #     # Create a scene
#     #     self.robot_scene.scene.title = "Test Set Joint Orientation"

#     #     # Create a joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Rotate joint x+30d, y, z+45d
#     #     joint.update_orientation(
#     #         self.robot_se3 * SE3().Rx(30, 'deg') * SE3().Rz(45, 'deg'))

#     #     # Check position
#     #     self.check_obj_pose(
#     #         joint, self.robot_se3 * SE3().Rx(30, 'deg') * SE3().Rz(45, 'deg'))

#     # def test_set_joint_pose(self):
#     #     # Create a scene
#     #     self.robot_scene.scene.title = "Test Set Joint Pose"

#     #     # Create a joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Move joint x+30d, y, z-2
#     #     joint.update_pose(
#     #         self.robot_se3 * SE3().Rx(30, 'deg') * SE3().Tz(-2))

#     #     # Check position
#     #     self.check_obj_pose(
#     #         joint, self.robot_se3 * SE3().Rx(30, 'deg') * SE3().Tz(-2))

#     # def test_draw_reference_frame(self):
#     #     # Scene update
#     #     self.robot_scene.scene.title = "Test Draw Reference Frame"
#     #     self.robot_scene.grid_visibility(False)

#     #     # Create a joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Count num objects
#     #     num_obj_initial = len(self.robot_scene.scene.objects)

#     #     # Turn off reference frame
#     #     joint.draw_reference_frame(False)

#     #     # Count num objects
#     #     num_obj_off = len(self.robot_scene.scene.objects)
#     #     self.assertEqual(num_obj_initial - num_obj_off, 1)

#     #     # Turn on
#     #     joint.draw_reference_frame(True)

#     #     # Count num objects
#     #     num_obj_on = len(self.robot_scene.scene.objects)
#     #     self.assertEqual(num_obj_on - num_obj_off, 1)
#     #     self.assertEqual(num_obj_on, num_obj_initial)

#     # def test_joint_visibility(self):
#     #     # Scene update
#     #     self.robot_scene.scene.title = "Test Joint Visibility"
#     #     self.robot_scene.grid_visibility(False)

#     #     # Create a joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Count num objects
#     #     num_obj_initial = len(self.robot_scene.scene.objects)

#     #     # Turn off joint graphic
#     #     joint.set_joint_visibility(False)

#     #     # Count num objects
#     #     num_obj_off = len(self.robot_scene.scene.objects)
#     #     self.assertEqual(num_obj_initial - num_obj_off, 1)

#     #     # Turn on
#     #     joint.set_joint_visibility(True)

#     #     # Count num objects
#     #     num_obj_on = len(self.robot_scene.scene.objects)
#     #     self.assertEqual(num_obj_on - num_obj_off, 1)
#     #     self.assertEqual(num_obj_on, num_obj_initial)

#     # def test_joint_texture(self):
#     #     # Scene update
#     #     self.robot_scene.scene.title = "Test Joint Texture"

#     #     # Create joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Apply texture and colour
#     #     joint.set_texture(
#     #         colour=[0.5, 0, 1],
#     #         texture_link="https://s3.amazonaws.com/glowscript/textures/flower_texture.jpg"  # noqa
#     #     )

#     #     # Ensure texture is not none, and colour is not white
#     #     gph_obj = joint.get_graphic_object()
#     #     self.assertEqual(gph_obj.color, vector(0.5, 0, 1))
#     #     self.assertIsNotNone(gph_obj.texture)

#     #     # Remove Texture and colour
#     #     joint.set_texture()

#     #     # Ensure colour is white, texture is none
#     #     self.assertEqual(gph_obj.color, vector(1, 1, 1))
#     #     self.assertIsNone(gph_obj.texture)

#     #     # Apply bad colour
#     #     # Should assert Value Error
#     #     self.assertRaises(ValueError, joint.set_texture, colour=[127, 0, 255])

#     # def test_joint_transparency(self):
#     #     # Scene update
#     #     self.robot_scene.scene.title = "Test Joint Transparency"

#     #     # Create joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Apply texture and colour
#     #     opc_val = 0.34
#     #     joint.set_transparency(opc_val)

#     #     # Ensure texture is not none, and colour is not white
#     #     gph_obj = joint.get_graphic_object()
#     #     self.assertEqual(gph_obj.opacity, opc_val)

#     #     # Set transparency out of range
#     #     # Should throw value error
#     #     self.assertRaises(ValueError, joint.set_transparency, 1.5)

#     # def test_set_origin(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Set Origin"

#     #     # Create a joint
#     #     joint = RotationalJoint(SE3(), self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Save origin pos (copy of)
#     #     first_pos = vector(joint.get_graphic_object().origin)

#     #     # Move origin
#     #     current_pos = vector(1, 0, -0.5)
#     #     new_pos = vector(0, 0, 0.5)
#     #     joint.set_stl_joint_origin(current_pos, new_pos)

#     #     # Save new origin
#     #     second_pos = joint.get_graphic_object().origin

#     #     # Object should go from along +x-axis, to along -x-axis slightly
#     #     # above z=0 plane
#     #     # Compare
#     #     # Check original origin is at 0, 0, 0 (Default)
#     #     self.assertEqual(first_pos, vector(0, 0, 0))
#     #     # Check new set origin is at the new position
#     #     self.assertEqual(second_pos, new_pos - current_pos)

#     # def test_joint_get_pose(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Get Joint Pose"

#     #     # Create a joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Get pose
#     #     pose = joint.get_pose()

#     #     # Check it's equal (proves get returned correctly)
#     #     self.assertEqual(self.robot_se3, pose)

#     # def test_joint_get_axis_vector(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Get Joint Pose"

#     #     # Create a joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Get axis vector
#     #     x_vec = joint.get_axis_vector(x_axis_vector)
#     #     y_vec = joint.get_axis_vector(y_axis_vector)
#     #     z_vec = joint.get_axis_vector(z_axis_vector)

#     #     # Check it's equal (proves get returned correctly)
#     #     self.assertEqual(get_pose_x_vec(self.robot_se3), x_vec)
#     #     self.assertEqual(get_pose_y_vec(self.robot_se3), y_vec)
#     #     self.assertEqual(get_pose_z_vec(self.robot_se3), z_vec)

#     #     # Check error is thrown
#     #     self.assertRaises(ValueError, joint.get_axis_vector, vector(0.5, 2, 3))

#     # def test_joint_get_type(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Joint Get Type"

#     #     # Create one of each joint
#     #     r = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     p = PrismaticJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     s = StaticJoint(self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     g = Gripper(self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Check each is correct
#     #     self.check_joint_type(r, "R")
#     #     self.check_joint_type(p, "P")
#     #     self.check_joint_type(s, "S")
#     #     self.check_joint_type(g, "G")

#     # def test_joint_get_graphic(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Joint Get Graphic"
#     #     self.robot_scene.grid_visibility(False)

#     #     # Create a joint
#     #     joint = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     joint.draw_reference_frame(False)

#     #     # Get graphic obj
#     #     gph_obj = joint.get_graphic_object()

#     #     # If obj equal only obj in scene
#     #     self.assertEqual(len(self.robot_scene.scene.objects), 1)
#     #     self.assertEqual(gph_obj, self.robot_scene.scene.objects[0])

#     # ##################################################
#     # # Robot functions
#     # ##################################################
#     # def test_robot_append_made_link(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Robot Append Made Link"

#     #     # Create 2 joints
#     #     joint1 = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     joint2 = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Create robot
#     #     robot1 = GraphicalRobot(self.robot_scene, "Robot 1")

#     #     # Add 1 joint
#     #     robot1.append_made_link(joint1)

#     #     # Print joint poses to prove its added
#     #     robot1.print_joint_poses()

#     #     # Create a new scene
#     #     scene2 = GraphicsCanvas3D(title="Test Robot Append Made Link 2")

#     #     # Create a new robot in new scene
#     #     robot2 = GraphicalRobot(scene2, "Robot 2")

#     #     # Add other joint to new scene
#     #     # Expecting an error (can't add joint to robot in different scene
#     #     self.assertRaises(RuntimeError, robot2.append_made_link, joint2)

#     # def test_robot_append_link(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Robot Append Link"

#     #     # Create robot
#     #     robot1 = GraphicalRobot(self.robot_scene, "Robot 1")

#     #     # Add link
#     #     robot1.append_link("r", self.robot_se3, self.robot_structure, [0, 0], 0)
#     #     robot1.append_link("R", self.robot_se3, self.robot_structure, [0, 0], 0)

#     #     # Print poses to verify
#     #     robot1.print_joint_poses()

#     #     # Try wrong inputs, expecting errors
#     #     # bad joint type
#     #     self.assertRaises(
#     #         ValueError, robot1.append_link, "x", self.robot_se3,
#     #         self.robot_structure, [0, 0], 0)
#     #     # incorrect param order
#     #     self.assertRaises(
#     #         TypeError, robot1.append_link, "p", self.robot_structure,
#     #         self.robot_se3, [0, 0], 0)

#     # def test_robot_detach_link(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Robot Detach Link"
#     #     self.robot_scene.grid_visibility(False)

#     #     # Create robot
#     #     robot1 = GraphicalRobot(self.robot_scene, "Robot 1")

#     #     # Add two links
#     #     robot1.append_link(
#     #         "r", self.robot_se3, self.robot_structure, [0, 0], 0)
#     #     robot1.append_link(
#     #         "r", self.robot_se3 * SE3().Tx(1), self.robot_structure, [0, 0], 0)

#     #     # Count num objects
#     #     num_obj = len(self.robot_scene.scene.objects)

#     #     # Count num joints
#     #     num_joints = robot1.num_joints

#     #     # Detach
#     #     robot1.detach_link()

#     #     # Verify new object count
#     #     # 2 = one for joint, 1 for ref frame
#     #     self.assertEqual(len(self.robot_scene.scene.objects), num_obj - 2)

#     #     # Verify new joint count
#     #     # Taken away 1 joint
#     #     self.assertEqual(robot1.num_joints, num_joints - 1)

#     #     # Create new empty robot
#     #     robot2 = GraphicalRobot(self.robot_scene, "Robot 2")

#     #     # Attempt to detach from empty
#     #     self.assertRaises(UserWarning, robot2.detach_link)

#     # def test_robot_reference_visibility(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Robot Reference Visibility"
#     #     self.robot_scene.grid_visibility(False)

#     #     # Create two link robot
#     #     robot1 = GraphicalRobot(self.robot_scene, "Robot 1")
#     #     robot1.append_link("r", self.robot_se3, self.robot_structure, [0, 0], 0)
#     #     robot1.append_link("r", self.robot_se3, self.robot_structure, [0, 0], 0)

#     #     # Count num obj visible
#     #     num_obj = len(self.robot_scene.scene.objects)

#     #     # Turn off ref frames
#     #     robot1.set_reference_visibility(False)

#     #     # Verify new amount
#     #     # Take 1 for each link
#     #     self.assertEqual(len(self.robot_scene.scene.objects), num_obj - 2)

#     #     # Turn on ref frames
#     #     robot1.set_reference_visibility(True)

#     #     # Verify amount
#     #     # Original amount
#     #     self.assertEqual(len(self.robot_scene.scene.objects), num_obj)

#     # def test_robot_visibility(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Robot Visibility"
#     #     self.robot_scene.grid_visibility(False)

#     #     # Create two link robot
#     #     robot1 = GraphicalRobot(self.robot_scene, "Robot 1")
#     #     robot1.append_link("r", self.robot_se3, self.robot_structure, [0, 0], 0)
#     #     robot1.append_link("r", self.robot_se3, self.robot_structure, [0, 0], 0)

#     #     # Count num obj visible
#     #     num_obj = len(self.robot_scene.scene.objects)

#     #     # Turn off ref frames
#     #     robot1.set_robot_visibility(False)

#     #     # Verify new amount
#     #     # Take 1 for each link
#     #     self.assertEqual(len(self.robot_scene.scene.objects), num_obj - 2)

#     #     # Turn on ref frames
#     #     robot1.set_robot_visibility(True)

#     #     # Verify amount
#     #     # Original amount
#     #     self.assertEqual(len(self.robot_scene.scene.objects), num_obj)

#     # def test_robot_transparency(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Robot Transparency"

#     #     # Create two joints
#     #     joint1 = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)
#     #     joint2 = RotationalJoint(
#     #         self.robot_se3, self.robot_structure, self.robot_scene, [0, 0], 0)

#     #     # Create robot
#     #     robot1 = GraphicalRobot(self.robot_scene, "Robot 1")

#     #     # Add joints
#     #     robot1.append_made_link(joint1)
#     #     robot1.append_made_link(joint2)

#     #     # Change opacity value
#     #     opc_val = 0.33
#     #     robot1.set_transparency(opc_val)

#     #     # Verify change in all joints
#     #     self.assertEqual(joint1.get_graphic_object().opacity, opc_val)
#     #     self.assertEqual(joint2.get_graphic_object().opacity, opc_val)

#     #     # Test bad opc val
#     #     self.assertRaises(ValueError, robot1.set_transparency, -0.2)

#     # def test_robot_set_poses(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Robot Set Poses"
#     #     self.robot_scene.grid_visibility(False)

#     #     # Create a two link robot
#     #     robot1 = GraphicalRobot(self.robot_scene, "Robot 1")
#     #     robot1.append_link(
#     #         "r", self.robot_se3, self.robot_structure, [0, 0], 0)
#     #     robot1.append_link(
#     #         "r", self.robot_se3 * SE3().Tx(1), self.robot_structure, [0, 0], 0)

#     #     s1 = SE3().Tx(2) * SE3().Tz(0.3) * SE3().Ry(23, 'deg')
#     #     s2 = SE3().Ty(0.5) * SE3().Tx(1.2) * SE3().Rz(-34, 'deg')

#     #     # Set each joint to a known location
#     #     robot1.set_joint_poses([s1, s2])

#     #     # For each obj in scene, make sure it is one of the two locations
#     #     # Ensure objects visible are just reference frames (they have same
#     #     # pose as the graphic itself)
#     #     robot1.set_robot_visibility(False)
#     #     # Should only have 2 reference frames in the scene
#     #     self.assertEqual(len(self.robot_scene.scene.objects), 2)
#     #     # Both objects must be in either of the poses (but not the same one)
#     #     self.assertTrue(
#     #         # 0 in s1, 1 in s2
#     #         (vpython_to_se3(self.robot_scene.scene.objects[0]) == s1 and
#     #          vpython_to_se3(self.robot_scene.scene.objects[1]) == s2)
#     #         or
#     #         # 1 in s1, 0 in s2
#     #         (vpython_to_se3(self.robot_scene.scene.objects[1]) == s1 and
#     #          vpython_to_se3(self.robot_scene.scene.objects[0]) == s2)
#     #     )

#     #     # Try giving not enough poses
#     #     self.assertRaises(UserWarning, robot1.set_joint_poses, [])

#     #     # Create new robot
#     #     robot2 = GraphicalRobot(self.robot_scene, "Robot 2")

#     #     # Try setting poses on empty robot
#     #     self.assertRaises(UserWarning, robot2.set_joint_poses, [s1, s2])

#     # def test_robot_animate(self):
#     #     # Update scene
#     #     self.robot_scene.scene.title = "Test Robot Animate"
#     #     self.robot_scene.grid_visibility(False)

#     #     # Create a two link robot
#     #     robot1 = GraphicalRobot(self.robot_scene, "Robot 1")
#     #     robot1.append_link(
#     #         "r", self.robot_se3, self.robot_structure, [0, 0], 0)
#     #     robot1.append_link(
#     #         "r", self.robot_se3 * SE3().Tx(1), self.robot_structure, [0, 0], 0)

#     #     s1 = SE3().Tx(2) * SE3().Tz(0.3) * SE3().Ry(23, 'deg')
#     #     s2 = SE3().Ty(0.5) * SE3().Tx(1.2) * SE3().Rz(-34, 'deg')

#     #     # Set each joint to a known location
#     #     robot1.animate([[s2, s1], [s1, s2]], 1)
#     #     # As it can't test positions mid frame, just check final positions
#     #     # are correct

#     #     # For each obj in scene, make sure it is one of the two locations
#     #     # Ensure objects visible are just reference frames (they have same
#     #     # pose as the graphic itself)
#     #     robot1.set_robot_visibility(False)
#     #     # Should only have 2 reference frames in the scene
#     #     self.assertEqual(len(self.robot_scene.scene.objects), 2)
#     #     # Both objects must be in either of the poses (but not the same one)
#     #     self.assertTrue(
#     #         # 0 in s1, 1 in s2
#     #         (vpython_to_se3(self.robot_scene.scene.objects[0]) == s1 and
#     #          vpython_to_se3(self.robot_scene.scene.objects[1]) == s2)
#     #         or
#     #         # 1 in s1, 0 in s2
#     #         (vpython_to_se3(self.robot_scene.scene.objects[1]) == s1 and
#     #          vpython_to_se3(self.robot_scene.scene.objects[0]) == s2)
#     #     )

#     #     # Try giving no frames
#     #     self.assertRaises(ValueError, robot1.animate, [], 1)

#     #     # Try giving bad frame count
#     #     self.assertRaises(ValueError, robot1.animate, [[s1, s2]], -1)

#     #     # Try giving wrong number SE3s
#     #     self.assertRaises(UserWarning, robot1.animate, [[]], 1)

#     # ##########################################################################
#     # def test_import_object(self):
#     #     # Update Scene
#     #     self.robot_scene.scene.title = "Test Import Object"
#     #     self.robot_scene.grid_visibility(False)

#     #     # Check num objects
#     #     # num_obj = len(scene.scene.objects)

#     #     # Import an object
#     #     graphic_obj = import_object_from_numpy_stl(
#     #         './roboticstoolbox/models/DH/meshes/UNIMATE/puma560/link0.stl',
#     #         scene.scene
#     #     )

#     #     # Verify object was added
#     #     # Object is at origin
#     #     self.assertEqual(graphic_obj.pos, vector(0, 0, 0))
#     #     # Can't check how many objects, as each triangle counts as one. No way
#     #     # to know correct amount
#     #     # self.assertEqual(len(scene.scene.objects), num_obj + 1)  # 1 object
#     #     # was added to the scene

#     # ##########################################################################
#     # def test_backend_add(self):
#     #     # Create robot
#     #     p = Puma560()
#     #
#     #     # Count num objects
#     #     num_prior = len(self.robot_scene.scene.objects)
#     #
#     #     # Add robot
#     #     self.env.add(len(self.env.canvases)-1, 'puma', p)
#     #
#     #     # Verify num objects
#     #     num_post = len(self.robot_scene.scene.objects)
#     #     self.assertGreater(num_post, num_prior)
#     #
#     # def test_backend_step(self):
#     #     # Create robot
#     #     p = Puma560()
#     #
#     #     # Add robot
#     #     self.env.add(len(self.env.canvases)-1, 'puma', p)
#     #
#     #     # Note positions of objects
#     #     positions_prior = [obj.pos for obj in self.robot_scene.scene.objects]
#     #
#     #     # Step
#     #     self.env.step(p, q=p.qr, fig_num=len(self.env.canvases)-1)
#     #
#     #     # Verify positions changed
#     #     positions_after = [obj.pos for obj in self.robot_scene.scene.objects]
#     #     self.assertNotEqual(positions_prior, positions_after)
#     #
#     # def test_backend_remove(self):
#     #     # Create robot
#     #     p = Puma560()
#     #
#     #     # Count objects
#     #     num_prior = len(self.robot_scene.scene.objects)
#     #
#     #     # Add robot
#     #     self.env.add(len(self.env.canvases) - 1, 'puma', p)
#     #
#     #     # Count objects
#     #     num_mid = len(self.robot_scene.scene.objects)
#     #
#     #     # Remove
#     #     self.env.remove(p, fig_num=len(self.env.canvases)-1)
#     #
#     #     # Count objects
#     #     num_post = len(self.robot_scene.scene.objects)
#     #     self.assertGreater(num_mid, num_prior)
#     #     self.assertGreater(num_post, num_mid)
#     #     self.assertEqual(num_prior, num_post)
#     #
#     # # ##########################################################################
#     # def test_2d_init(self):
#     #     # Create a scene
#     #     self.env.launch(is_3d=False)
#     #
#     #     try:
#     #         # Put a box in the created scene
#     #         box(canvas=self.env.canvases[len(self.env.canvases) - 1])
#     #     except Exception:
#     #         # Something went wrong
#     #         self.assertEqual(False, True)
#     #
#     # def test_2d_grid_visiblity(self):
#     #     # Create a scene, with grid=True (default)
#     #     self.env.launch(is_3d=False, grid=True)
#     #
#     #     # Check all objects in scene are visible (default scene will just have
#     #     # grid, assuming init has grid=True)
#     #     self.assertGreater(len(self.env.canvases[len(self.env.canvases) - 1].scene.objects), 0)
#     #
#     #     # Change visibility
#     #     self.env.canvases[len(self.env.canvases) - 1].grid_visibility(False)
#     #
#     #     # Check all are invisible
#     #     # Invisible objects are not shown in the objects list
#     #     self.assertEqual(len(self.env.canvases[len(self.env.canvases) - 1].scene.objects), 0)
#     #
#     # def test_2d_plot(self):
#     #     # Create scene
#     #     self.env.launch(is_3d=False)
#     #     scene = self.env.canvases[len(self.env.canvases) - 1]
#     #
#     #     # Correct input
#     #     scene.plot([1, 2, 3, 4], [3, 2, 5, 2], 'rs-')
#     #
#     #     # Rearranged correct input
#     #     scene.plot([1, 2, 3, 4], [3, 2, 5, 2], 's-r')
#     #
#     #     # Empty inputs
#     #     scene.plot([], [], '')  # Shouldn't draw or throw errors
#     #
#     #     # Error checks
#     #     self.assertRaises(ValueError, scene.plot, [1, 1], 's7:')  # Unknown Character
#     #     self.assertRaises(ValueError, scene.plot, [1, 1], '-:r')  # Too many line segments
#     #     self.assertRaises(ValueError, scene.plot, [1, 1], 'rcs')  # Too many colour segments
#     #     self.assertRaises(ValueError, scene.plot, [1, 1], 'dsg')  # Too many marker segments
#     #     self.assertRaises(ValueError, scene.plot, [1, 1], [1, 2, 3])  # Different num values


# if __name__ == '__main__':
#     unittest.main()
