#!/usr/bin/env python
"""
@author Micah Huth
"""

import numpy as np
from vpython import box, compound, color, sqrt
from roboticstoolbox.backends.VPython.canvas import draw_reference_frame_axes
from roboticstoolbox.backends.VPython.common_functions import \
    array, x_axis_vector, y_axis_vector, z_axis_vector, \
    get_pose_pos, get_pose_x_vec, get_pose_y_vec, get_pose_z_vec, \
    vector
from roboticstoolbox.backends.VPython.stl import set_stl_origin, \
    import_object_from_numpy_stl
from time import perf_counter
from spatialmath import SE3
from pathlib import PurePath


class DefaultJoint:  # pragma nocover
    """
    This class forms a base for all the different types of joints
    - Rotational
    - Prismatic
    - Static
    - Gripper

    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: A variable representing the joint length (float) or
        a file path to an STL (str)
    :type structure: `float`, `str`
    :param g_canvas: The canvas in which to add the link
    :type g_canvas: class:`graphics.graphics_canvas.graphicscanvas3d`
    :param qlim: A list of the angle limits for the joint
    :type qlim: `list`
    :param theta: The current angle of the joint in radians
    :type theta: `float`
    :param axis_through: The axis that the longest side goes through
    :type axis_through: class:`numpy.ndarray`
    """

    def __init__(
            self, initial_se3, structure, g_canvas,
            qlim, theta, axis_through=array([1, 0, 0])):

        if not isinstance(
                structure, float) and not isinstance(structure, PurePath):
            error_str = "structure must be of type {0} or {1}. Given {2}. " \
                "Either give a length (float)," \
                "or meshdata [filepath, scale, origindata]"
            raise TypeError(error_str.format(float, str, type(structure)))

        self.qlim = qlim
        self.theta = theta

        self.__scene = g_canvas.scene
        self.__pose = initial_se3

        # Set the graphic
        # self.stl_offset = None
        self.__graphic_obj = self.__set_graphic(structure, axis_through)
        self.visible = True

        # Set the other reference frame vectors
        self.__graphic_ref = draw_reference_frame_axes(
            self.__pose, self.__scene)

        # Apply the initial pose
        self.update_pose(self.__pose)

    def update_position(self, se_object):
        """
        Given an SE object, update just the orientation of the joint.

        :param se_object: SE3 pose representation of the joint
        :type se_object: class:`spatialmath.pose3d.SE3`
        """
        new_position = get_pose_pos(se_object)
        self.__graphic_obj.pos = new_position

        # Update the reference frame
        self.__update_reference_frame()
        self.draw_reference_frame(self.__graphic_ref.visible)

    def update_orientation(self, se_object):
        """
        Given an SE object, update just the orientation of the joint.

        :param se_object: SE3 pose representation of the joint
        :type se_object: class:`spatialmath.pose3d.SE3`
        """
        # Get the new pose details
        new_x_axis = get_pose_x_vec(se_object)
        new_y_axis = get_pose_y_vec(se_object)
        # new_z_axis = get_pose_z_vec(se_object)  # not needed

        # Update the graphic object
        self.__graphic_obj.axis = new_x_axis
        self.__graphic_obj.up = new_y_axis

        # Update the reference frame
        self.__update_reference_frame()
        self.draw_reference_frame(self.__graphic_ref.visible)

    def update_pose(self, se_object):
        """
        Given an SE object, update the pose of the joint.

        :param se_object: SE3 pose representation of the joint
        :type se_object: class:`spatialmath.pose3d.SE3`
        """
        # if self.stl_offset is not None:
        #     calc_se3 = se_object * self.stl_offset
        # else:
        #     calc_se3 = se_object

        # Get the new pose details
        new_x_axis = get_pose_x_vec(se_object)
        new_y_axis = get_pose_y_vec(se_object)
        # new_z_axis = get_pose_z_vec(se_object)  # not needed
        new_position = get_pose_pos(se_object)

        # Update the graphic object
        self.__graphic_obj.pos = new_position
        self.__graphic_obj.axis = new_x_axis
        self.__graphic_obj.up = new_y_axis

        # Update the reference frame
        self.__pose = se_object
        self.__update_reference_frame()
        self.draw_reference_frame(self.__graphic_ref.visible)

    def __update_reference_frame(self):
        """
        Update the reference frame axis vectors
        """
        self.__x_vector = get_pose_x_vec(self.__pose)
        self.__y_vector = get_pose_y_vec(self.__pose)
        self.__z_vector = get_pose_z_vec(self.__pose)

    def draw_reference_frame(self, is_visible):
        """
        Draw a reference frame at the tool point position.

        :param is_visible: Whether the reference frame should be drawn or not
        :type is_visible: `bool`
        """
        self.__graphic_ref.pos = get_pose_pos(self.__pose)
        self.__graphic_ref.axis = get_pose_x_vec(self.__pose)
        self.__graphic_ref.up = get_pose_y_vec(self.__pose)
        self.__graphic_ref.visible = is_visible

    def set_stl_joint_origin(self, current_location, required_location):
        """
        Modify the origin position of the graphical object.
        This is mainly used when loading STL objects. If the origin (point of
        rotation/placement) does not align with
        reality, it can be set.
        It translates the object to place the origin at the required location,
        and save the new origin to the object.

        :param current_location: 3D coordinate of where the real origin is in
            space.
        :type current_location: class:`vpython.vector`
        :param required_location: 3D coordinate of where the real origin
            should be in space
        :type required_location: class:`vpython.vector`
        """
        set_stl_origin(
            self.__graphic_obj, current_location,
            required_location, self.__scene)

    def set_joint_visibility(self, is_visible):
        """
        Choose whether or not the joint is displayed in the canvas.

        :param is_visible: Whether the joint should be drawn or not
        :type is_visible: `bool`
        """
        # If the option is different to current setting
        if is_visible is not self.visible:
            # Update
            self.__graphic_obj.visible = is_visible
            # self.__graphic_ref.visible = is_visible
            self.visible = is_visible

    def __set_graphic(self, structure, axis_through):
        """
        Set the graphic object depending on if one was given. If no object was
        given, create a box and return it

        :param structure: `float` or `str` representing the joint length or
            STL path to load from
        :type structure: `float`, `str`
        :param axis_through: The axis that the longest side goes through
        :type axis_through: class:`numpy.ndarray`
        :raises ValueError: Joint length must be greater than 0
        :return: Graphical object for the joint
        :rtype: class:`vpython.compound`
        """
        if isinstance(structure, float):
            length = structure
            if length <= 0.0:
                raise ValueError("Joint length must be greater than 0")
            axis = vector(axis_through[0], axis_through[1], axis_through[2])
            axis.mag = length

            box_midpoint = axis / 2
            box_tooltip = axis

            # Create a box along the +x axis, with the origin
            # (point of rotation) at (0, 0, 0)
            graphic_obj = box(
                canvas=self.__scene,
                pos=vector(box_midpoint.x, box_midpoint.y, box_midpoint.z),
                axis=axis,
                size=vector(length, 0.1, 0.1),  # L, W, H
                # up=y_axis_vector
            )

            # Set the boxes new origin
            graphic_obj = compound(
                [graphic_obj], origin=box_tooltip, axis=axis)

            return graphic_obj
        else:
            # self.stl_offset = structure[2]
            return import_object_from_numpy_stl(structure, self.__scene)

    def set_texture(self, colour=None, texture_link=None):
        """
        Apply the texture/colour to the object. If both are given, both are
        applied.
        Texture link can either be a link to an online image, or local file.

        WARNING: If the texture can't be loaded, the object will have n
        texture
        (appear invisible, but not set as invisible).

        WARNING: If the image has a width or height that is not a power of 2
        (that is, not 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, etc.),
        the image is stretched to the next larger width or height that is
        a power of 2.

        :param colour: List of RGB values
        :type colour: `list`, optional
        :param texture_link: Path/website to a texture image
        :type texture_link: `str`, optional
        :raises ValueError: RGB values must be normalised between 0 and 1
        """
        # Apply the texture
        if texture_link is not None:
            self.__graphic_obj.texture = {
                'file': texture_link
                # 'bumpmap', 'place', 'flipx', 'flipy', 'turn'
            }
            # Wait for the texture to load
            self.__scene.waitfor("textures")
        else:
            # Remove any texture
            self.__graphic_obj.texture = None

        # Apply the colour
        if colour is not None:
            if colour[0] > 1.0 or colour[1] > 1.0 or colour[2] > 1.0 or \
               colour[0] < 0.0 or colour[1] < 0.0 or colour[2] < 0.0:
                raise ValueError(
                    "RGB values must be normalised between 0 and 1")
            colour_vec = vector(colour[0], colour[1], colour[2])
            self.__graphic_obj.color = colour_vec
        else:
            # Set to white if none given
            self.__graphic_obj.color = color.white

    def set_transparency(self, opacity):
        """
        Sets the transparency of the joint.

        :param opacity: Normalised value (0 -> 1) to set the opacity.
            0 = transparent, 1 = opaque
        :type opacity: `float`
        :raises ValueError: Value must be between 0 and 1 inclusively
        """
        if opacity < 0 or opacity > 1:
            raise ValueError("Value must be between 0 and 1 inclusively")
        self.__graphic_obj.opacity = opacity

    def get_axis_vector(self, axis):
        """
        Get the current vector of a specified X, Y, or Z axis

        :param axis: Specified joint axis to get the angle of rotation of
        :type axis: class:`vpython.vector`
        :return: Current vector representation of the joints X, Y, or Z axis
        :rtype: class:`vpython.vector`
        :raise ValueError: The given axis_of_rotation must be one of the
            default X, Y, Z vectors (e.g. x_axis_vector)
        """
        # Return new vectors to avoid pass by reference
        if axis.equals(x_axis_vector):
            return self.__x_vector
        elif axis.equals(y_axis_vector):
            return self.__y_vector
        elif axis.equals(z_axis_vector):
            return self.__z_vector
        else:
            error_str = "Bad input vector given ({0}). " \
                        "Must be either x_axis_vector ({1}), " \
                        "y_axis_vector ({2})," \
                        "or z_axis_vector ({3})."
            raise ValueError(
                error_str.format(
                    axis, x_axis_vector, y_axis_vector, z_axis_vector))

    def get_pose(self):
        """
        Return the current pose of the joint

        :return: SE3 representation of the current joint pose
        :rtype: class:`spatialmath.pose3d.SE3`
        """
        return self.__pose

    def get_joint_type(self):
        """
        Return the type of joint (To Be Overridden by Child classes)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return ""

    def get_graphic_object(self):
        """
        Getter function that returns the graphical object of the joint

        :return: VPython graphical entity of the joint
        :rtype: class:`vpython.object`
        """
        return self.__graphic_obj

    def get_scene(self):
        """
        Getter function that returns the scene the object is in

        :return: The scene the object is in
        :rtype: class:`vpython.canvas`
        """
        return self.__scene


class RotationalJoint(DefaultJoint):  # pragma nocover
    """
    A rotational joint based off the default joint class

    :param g_canvas: The canvas in which to add the link
    :type g_canvas: class:`graphics.graphics_canvas.graphicscanvas3d`
    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: either a float of the length of the joint, or a list of
        str of the filepath and scale
    :type structure: `float`, `list`
    :param qlim: A list of the angle limits for the joint
    :type qlim: `list`
    :param theta: The current angle of the joint in radians
    :type theta: `float`
    :param axis_through: The axis that the longest side goes through
    :type axis_through: class:`numpy.ndarray`
    """

    def __init__(
            self, initial_se3, structure, g_canvas, qlim,
            theta, axis_through=array([1, 0, 0])):

        # Call super init function
        super().__init__(
            initial_se3, structure, g_canvas, qlim, theta, axis_through)

    def get_joint_type(self):
        """
        Return the type of joint (R for Rotational)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return "R"


class PrismaticJoint(DefaultJoint):  # pragma nocover
    """
    A prismatic joint based from the default joint class

    :param g_canvas: The canvas in which to add the link
    :type g_canvas: class:`graphics.graphics_canvas.graphicscanvas3d`
    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: either a float of the length of the joint, or a list of
        str of the filepath and scale
    :type structure: `float`, `list`
    :param qlim: A list of the angle limits for the joint
    :type qlim: `list`
    :param theta: The current angle of the joint in radians
    :type theta: `float`
    :param axis_through: The axis that the longest side goes through
    :type axis_through: class:`numpy.ndarray`
    """
    def __init__(
            self, initial_se3, structure, g_canvas, qlim, theta,
            axis_through=array([1, 0, 0])):

        super().__init__(
            initial_se3, structure, g_canvas, qlim, theta, axis_through)
        self.min_translation = None
        self.max_translation = None

    def translate_joint(self, new_translation):
        # TODO
        pass

    def get_joint_type(self):
        """
        Return the type of joint (P for Prismatic)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return "P"


class StaticJoint(DefaultJoint):  # pragma nocover
    """
    This class represents a static joint (one that doesn't translate or
    rotate on it's own).
    It has no extra functions to utilise.

    :param g_canvas: The canvas in which to add the link
    :type g_canvas: class:`graphics.graphics_canvas.graphicscanvas3d`
    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: either a float of the length of the joint, or a list of
        str of the filepath and scale
    :type structure: `float`, `list`
    :param qlim: A list of the angle limits for the joint
    :type qlim: `list`
    :param theta: The current angle of the joint in radians
    :type theta: `float`
    :param axis_through: The axis that the longest side goes through
    :type axis_through: class:`numpy.ndarray`
    """

    def __init__(
            self, initial_se3, structure, g_canvas, qlim, theta,
            axis_through=array([1, 0, 0])):

        super().__init__(
            initial_se3, structure, g_canvas, qlim, theta, axis_through)

    def get_joint_type(self):
        """
        Return the type of joint (S for Static)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return "S"


class Gripper(DefaultJoint):  # pragma nocover
    """
    This class represents a gripper joint with a moving gripper
    (To Be Implemented).
    Usually the end joint of a robot.

    :param g_canvas: The canvas in which to add the link
    :type g_canvas: class:`graphics.graphics_canvas.graphicscanvas3d`
    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: either a float of the length of the joint, or a list
        of str of the filepath and scale
    :type structure: `float`, `list`
    :param qlim: A list of the angle limits for the joint
    :type qlim: `list`
    :param theta: The current angle of the joint in radians
    :type theta: `float`
    :param axis_through: The axis that the longest side goes through
    :type axis_through: class:`numpy.ndarray`
    """

    def __init__(
            self, initial_se3, structure, g_canvas,
            qlim, theta, axis_through=array([1, 0, 0])):

        super().__init__(
            initial_se3, structure, g_canvas, qlim, theta, axis_through)

    # TODO close/open gripper

    def get_joint_type(self):
        """
        Return the type of joint (G for Gripper)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return "G"


class GraphicalRobot:  # pragma nocover
    """
    The GraphicalRobot class holds all of the different joints to easily
    control the robot arm.

    :param graphics_canvas: The canvas to add the robot to
    :type graphics_canvas: class:`GraphicsCanvas`
    :param name: The name of the robot to identify it
    :type name: `str`
    :param robot: A serial link object to create a robot on
    :type robot: class:`roboticstoolbox.robot.DHRobot`
    """
    def __init__(self, graphics_canvas, name, robot=None):
        self.joints = []
        self.num_joints = 0
        self.rob_shown = True
        self.ref_shown = True
        self.opacity = 1
        self.name = name
        self.__scene = graphics_canvas

        self.angles = []
        self.robot = robot

        # If Robot given, create the robot
        if self.robot is not None:
            # Update name
            self.name = self.robot.name
            # Get initial poses
            zero_angles = np.zeros((self.robot.n,))
            all_poses = self.robot.fkine_all(zero_angles, old=False)
            # Create the base
            if robot.basemesh is not None:
                self.append_link("s", all_poses[0], robot.basemesh, [0, 0], 0)
            # else: assume no base joint
            robot_colours = robot.linkcolormap()
            # Create the joints
            for i, link in enumerate(self.robot.links):
                # Get info
                if link.isprismatic:
                    j_type = 'p'
                elif link.isrevolute:
                    j_type = 'r'
                else:
                    j_type = 's'
                pose = all_poses[i+1]   # link frame pose
                if link.mesh is None:
                    if i == 0:
                        x1, x2 = SE3().t[0], all_poses[i].t[0]
                        y1, y2 = SE3().t[1], all_poses[i].t[1]
                        z1, z2 = SE3().t[2], all_poses[i].t[2]
                        length = sqrt(
                            (x2 - x1) * (x2 - x1) + (y2 - y1)
                            * (y2 - y1) + (z2 - z1) * (z2 - z1))  # Length
                    else:
                        x1, x2 = all_poses[i - 1].t[0], all_poses[i].t[0]
                        y1, y2 = all_poses[i - 1].t[1], all_poses[i].t[1]
                        z1, z2 = all_poses[i - 1].t[2], all_poses[i].t[2]
                        length = sqrt(
                            (x2 - x1) * (x2 - x1) + (y2 - y1)
                            * (y2 - y1) + (z2 - z1) * (z2 - z1))  # Length
                else:
                    length = link.mesh
                angle_lims = link.qlim  # Angle limits
                theta = link.theta  # Current angle
                self.append_link(j_type, pose, length, angle_lims, theta)

            # Apply colours
            for i, joint in enumerate(self.joints):
                link_colour = list(robot_colours(i))[:3]
                joint.set_texture(colour=link_colour)

        # Add the robot to the canvas UI
        graphics_canvas.add_robot(self)

    def append_made_link(self, joint):
        """
        Append an already made joint to the end of the robot
        (Useful for links manually created)

        :param joint: A joint object already constructed
        :type joint: class:`graphics.graphics_robot.RotationalJoint`,
            class:`graphics.graphics_robot.PrismaticJoint`,
        class:`graphics.graphics_robot.StaticJoint`,
            class:`graphics.graphics_robot.Gripper`
        :raises RuntimeError: Ensure the link is in the same scene as the robot
        """
        if joint.get_scene() != self.__scene.scene:
            raise RuntimeError(
                "The given made link is not in the same scene "
                "as the robot is.")

        self.joints.append(joint)
        self.num_joints += 1
        self.angles.append(joint.theta)

    def append_link(
            self, typeof, pose, structure, qlim, theta,
            axis_through=array([1, 0, 0])):
        """
        Append a joint to the end of the robot.

        :param typeof: String character of the joint type. e.g. 'R', 'P',
            'S', 'G'
        :type typeof: `str`
        :param pose: SE3 object for the pose of the joint
        :type pose: class:`spatialmath.pose3d.SE3`
        :param structure: either a float of the length of the joint, or a list
            of str of the filepath and scale
        :type structure: `float`, `list`
        :param qlim: A list of the angle limits for the joint
        :type qlim: `list`
        :param theta: The current angle of the joint in radians
        :type theta: `float`
        :param axis_through: The axis that the longest side goes through
        :type axis_through: class:`numpy.ndarray`
        :raises ValueError: typeof must be a valid character
        """
        # Capitalise the type for case-insensitive use
        typeof = typeof.upper()

        if typeof == 'R':
            link = RotationalJoint(
                pose, structure, self.__scene, qlim, theta, axis_through)
        elif typeof == 'P':
            link = PrismaticJoint(
                pose, structure, self.__scene, qlim, theta, axis_through)
        elif typeof == 'S':
            link = StaticJoint(
                pose, structure, self.__scene, qlim, theta, axis_through)
        elif typeof == 'G':
            link = Gripper(
                pose, structure, self.__scene, qlim, theta, axis_through)
        else:
            raise ValueError(
                "typeof should be (case-insensitive) either 'R' (Rotational)"
                ", 'P' (Prismatic), 'S' (Static), or 'G' (Gripper)")

        # Append the joint to the robot
        self.joints.append(link)
        self.num_joints += 1
        self.angles.append(theta)

    def detach_link(self):
        """
        Detach the end link of the robot.

        :raises UserWarning: Must have a joint available to detach
        """
        # Check if no joints to detach
        if self.num_joints == 0:
            raise UserWarning("No robot joints to detach")

        # Turn off the graphics in the canvas
        self.joints[self.num_joints - 1].set_joint_visibility(False)
        self.joints[self.num_joints - 1].draw_reference_frame(False)
        # Ensure deletion
        self.joints[self.num_joints - 1] = None

        # Resize lists
        self.joints = self.joints[0:self.num_joints - 1]
        self.angles = self.angles[0:self.num_joints - 1]
        self.num_joints -= 1

    def set_robot_visibility(self, is_visible):
        """
        Set the entire robots visibility inside the canvas.

        :param is_visible: Whether the robot should be visible or not.
        :type is_visible: `bool`
        """
        if is_visible is not self.rob_shown:
            for joint in self.joints:
                joint.set_joint_visibility(is_visible)
                self.rob_shown = is_visible

    def set_reference_visibility(self, is_visible):
        """
        Set the visibility of the reference frame for all joints.

        :param is_visible: Whether the reference frames should be
            visible or not.
        :type is_visible: `bool`
        """
        if is_visible is not self.ref_shown:
            for joint in self.joints:
                joint.draw_reference_frame(is_visible)
            self.ref_shown = is_visible

    def set_transparency(self, opacity):
        """
        Set the transparency of the robot.
        Allows for easier identification of the reference frames
        (if hidden by the robot itself)

        :param opacity: Normalised value (0 -> 1) to set the opacity.
            0 = transparent, 1 = opaque
        :type opacity: `float`
        """
        if opacity is not self.opacity:
            for joint in self.joints:
                joint.set_transparency(opacity)
            self.opacity = opacity

    def set_joint_poses(self, all_poses):
        """
        Set the joint poses.

        :param all_poses: List of all the new poses to set
        :type all_poses: class:`spatialmath.pose3d.SE3` list
        :raises UserWarning: Robot must not have 0 joints, and given poses
            length must equal number of joints.
        """
        # Sanity checks
        if self.num_joints == 0:
            raise UserWarning(
                "Robot has 0 joints. Create some using append_link()")

        # If given a base pose, when there isn't one
        if self.num_joints == len(all_poses) - 1 and all_poses[0] == SE3():
            # Update the joints
            for idx in range(self.num_joints):
                self.joints[idx].update_pose(all_poses[idx+1])
            return

        # If given all poses excluding a base pose
        if self.num_joints - 1 == len(all_poses):
            for idx in range(1, self.num_joints):
                self.joints[idx].update_pose(all_poses[idx-1])
            return

        # If incorrect number of joints
        if self.num_joints != len(all_poses):
            err = "Number of given poses {0} does not equal number " \
                "of joints {1}"
            raise UserWarning(err.format(len(all_poses), self.num_joints))

        # Update the joints
        for idx in range(self.num_joints):
            self.joints[idx].update_pose(all_poses[idx])

    def animate(self, frame_poses, fps):
        """
        Calling this function will animate the robot through its frames.

        :param frame_poses: A 2D list of each joint pose for each frame.
        :type frame_poses: `list`
        :param fps: Number of frames per second to render at
            (limited by number of graphics trying to update)
        :type fps: `int`
        :raises ValueError: Number of frames and fps must be greater than 0
        """
        num_frames = len(frame_poses)
        # Validate num_frames
        if num_frames == 0:
            raise ValueError(
                "0 frames were given. Supply at least 1 iteration of poses.")

        if fps <= 0:
            raise ValueError(
                "fps must be greater than 0.")
        f = 1 / fps

        for poses in frame_poses:
            # Get current time
            t_start = perf_counter()

            self.set_joint_poses(poses)  # Validation done in set_joint_poses
            # Wait for scene to finish drawing
            self.__scene.scene.waitfor("draw_complete")

            # Get current time
            t_stop = perf_counter()

            # Wait for time of frame to finish
            # If drawing takes longer than frame frequency,
            # this while is skipped
            while t_stop - t_start < f:
                t_stop = perf_counter()

    def fkine(self, joint_angles):
        """
        Call fkine for the robot. If it is based on a seriallink object,
        run it's fkine function.

        :param joint_angles: List of the joint angles
        :type joint_angles: `list`
        """
        # If seriallink object, run it's fkine
        if self.robot is not None:
            return self.robot.fkine_all(joint_angles)
        # Else TODO
        else:
            pass

    def print_joint_poses(self):
        """
        Print all of the current joint poses
        """
        # For each joint
        num = 0
        print(self.name)
        for joint in self.joints:
            print("Joint", num, "| Type:", joint.get_joint_type(), "| Pose:")
            print(joint.get_pose(), "\n")
            num += 1
