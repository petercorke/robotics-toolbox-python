from vpython import box, compound, scene, color
from graphics.graphics_canvas import draw_reference_frame_axes
from graphics.common_functions import *
from graphics.graphics_stl import set_stl_origin, import_object_from_numpy_stl
from time import perf_counter


class DefaultJoint:
    """
    This class forms a base for all the different types of joints
    - Rotational
    - Prismatic
    - Static
    - Gripper

    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float`, `str`
    """

    def __init__(self, initial_se3, structure):

        if not isinstance(structure, float) and not isinstance(structure, str):
            error_str = "structure must be of type {0} or {1}. Given {2}. Either give a length (float)," \
                        "or a file path to an STL (str)"
            raise TypeError(error_str.format(float, str, type(structure)))

        self.__pose = initial_se3

        # Set the graphic
        self.__graphic_obj = self.__set_graphic(structure)
        self.visible = True

        # Calculate the length of the link (Generally longest side is the length)
        self.__length = max(self.__graphic_obj.length, self.__graphic_obj.width, self.__graphic_obj.height)

        # Set the other reference frame vectors
        self.__graphic_ref = draw_reference_frame_axes(self.__pose)

        # Apply the initial pose if not given an STL (STL may need origin updating)
        if isinstance(structure, float):
            self.update_pose(self.__pose)
        
    def rotate_around_joint_axis(self, angle_of_rotation, axis_of_rotation):
        """
        Rotate the joint by a specific amount around one of the joints xyz axes

        :param angle_of_rotation: +/- angle of rotation to apply
        :type angle_of_rotation: float (radians)
        :param axis_of_rotation: X, Y, or Z axis to apply around the objects specific X, Y, or Z axes
        :type axis_of_rotation: class:`vpython.vector`
        :raise ValueError: The given axis_of_rotation must be one of the default X, Y, Z vectors (e.g. x_axis_vector)
        """
        raise PendingDeprecationWarning("Currently out of date. Will be updated soon.")
        # Determine the axis of rotation based on the given joint axis direction
        # Then add the rotation amount to the axis counter
        """
        if axis_of_rotation.equals(x_axis_vector):
            rotation_axis = self.__x_vector
            self.__x_rotation = wrap_to_pi("rad", self.__x_rotation + angle_of_rotation)
        elif axis_of_rotation.equals(y_axis_vector):
            rotation_axis = self.__y_vector
            self.__y_rotation = wrap_to_pi("rad", self.__y_rotation + angle_of_rotation)
        elif axis_of_rotation.equals(z_axis_vector):
            rotation_axis = self.__z_vector
            self.__z_rotation = wrap_to_pi("rad", self.__z_rotation + angle_of_rotation)
        else:
            error_str = "Bad input vector given ({0}). Must be either x_axis_vector ({1}), y_axis_vector ({2})," \
                        "or z_axis_vector ({3}). Use rotate_around_vector for rotation about an arbitrary vector."
            raise ValueError(error_str.format(axis_of_rotation), x_axis_vector, y_axis_vector, z_axis_vector)

        # Rotate the graphic object of the link to automatically transform the xyz axes and the graphic
        self.__graphic_obj.rotate(angle=angle_of_rotation, axis=rotation_axis, origin=self.__connect_from)
        # Update the vectors and reference frames
        self.__update_reference_frame()
        # Calculate the updated toolpoint location
        self.__connect_dir.rotate(angle=angle_of_rotation, axis=rotation_axis)
        self.__connect_to = self.__connect_dir.pos + self.__connect_dir.axis
        # If the reference frame exists, redraw it
        if self.__graphic_ref is not None:
            self.draw_reference_frame(self.__graphic_ref.visible)
        self.__draw_graphic()
        """

    def rotate_around_vector(self, angle_of_rotation, axis_of_rotation):
        """
        Rotate the object around a particular vector. Is utilised when rotating a different joint,
        and updating the subsequent joints.

        :param angle_of_rotation: +/- angle of rotation to apply
        :type angle_of_rotation: float (radians)
        :param axis_of_rotation: X, Y, or Z axis to apply around the objects specific X, Y, or Z axes
        :type axis_of_rotation: class:`vpython.vector`
        """
        raise PendingDeprecationWarning("Currently out of date. Will be updated soon.")

        """
        x_prev, y_prev, z_prev = vector(self.__x_vector), vector(self.__y_vector), vector(self.__z_vector)

        # Rotate the graphic object of the link to automatically transform the xyz axes and the graphic
        self.__graphic_obj.rotate(angle=angle_of_rotation, axis=axis_of_rotation, origin=self.__connect_from)
        # Update the vectors and reference frames
        self.__update_reference_frame()
        x_new, y_new, z_new = self.__x_vector, self.__y_vector, self.__z_vector

        angle_diff_x, angle_diff_y, angle_diff_z = diff_angle(x_prev, x_new), \
                                                   diff_angle(y_prev, y_new), \
                                                   diff_angle(z_prev, z_new)

        # Works out which axis of rotation the local one closest represents
        # (i.e. if facing (+x) straight up, world rotations closely resembles +z)
        # axis of rotation will have the smallest (it's less affected by the rotation)
        min_angle_diff = min(angle_diff_x, angle_diff_y, angle_diff_z)
        if min_angle_diff == angle_diff_x:
            self.__x_rotation = wrap_to_pi("rad", self.__x_rotation + angle_of_rotation)
        elif min_angle_diff == angle_diff_y:
            self.__y_rotation = wrap_to_pi("rad", self.__y_rotation + angle_of_rotation)
        else:
            self.__z_rotation = wrap_to_pi("rad", self.__z_rotation + angle_of_rotation)

        # Calculate the updated toolpoint location
        self.__connect_dir.rotate(angle=angle_of_rotation, axis=axis_of_rotation)
        self.__connect_to = self.__connect_dir.pos + self.__connect_dir.axis
        # If the reference frame exists, redraw it
        if self.__graphic_ref is not None:
            self.draw_reference_frame(self.__graphic_ref.visible)
        self.__draw_graphic()
        """

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
        This is mainly used when loading STL objects. If the origin (point of rotation/placement) does not align with
        reality, it can be set.
        It translates the object to place the origin at the required location, and save the new origin to the object.

        :param current_location: 3D coordinate of where the real origin is in space.
        :type current_location: class:`vpython.vector`
        :param required_location: 3D coordinate of where the real origin should be in space
        :type required_location: class:`vpython.vector`
        """
        set_stl_origin(self.__graphic_obj, current_location, required_location)

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

    def __set_graphic(self, structure):
        """
        Set the graphic object depending on if one was given. If no object was given, create a box and return it

        :param structure: `float` or `str` representing the joint length or STL path to load from
        :type structure: `float`, `str`
        :raises ValueError: Joint length must be greater than 0
        :return: Graphical object for the joint
        :rtype: class:`vpython.compound`
        """
        if isinstance(structure, float):
            length = structure
            if length <= 0.0:
                raise ValueError("Joint length must be greater than 0")

            box_midpoint = vector(length / 2, 0, 0)

            # Create a box along the +x axis, with the origin (point of rotation) at (0, 0, 0)
            graphic_obj = box(
                pos=vector(box_midpoint.x, box_midpoint.y, box_midpoint.z),
                axis=x_axis_vector,
                size=vector(length, 0.1, 0.1),
                up=y_axis_vector
            )

            # Set the boxes new origin
            graphic_obj = compound([graphic_obj], origin=vector(0, 0, 0), axis=x_axis_vector, up=y_axis_vector)

            return graphic_obj
        else:
            return import_object_from_numpy_stl(structure)

    def set_texture(self, colour=None, texture_link=None):
        """
        Apply the texture/colour to the object. If both are given, both are applied.
        Texture link can either be a link to an online image, or local file.

        WARNING: If the texture can't be loaded, the object will have no texture
        (appear invisible, but not set as invisible).

        WARNING: If the image has a width or height that is not a power of 2
        (that is, not 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, etc.),
        the image is stretched to the next larger width or height that is a power of 2.

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
            scene.waitfor("textures")
        else:
            # Remove any texture
            self.__graphic_obj.texture = None

        # Apply the colour
        if colour is not None:
            if colour[0] > 1.0 or colour[1] > 1.0 or colour[2] > 1.0 or \
               colour[0] < 0.0 or colour[1] < 0.0 or colour[2] < 0.0:
                raise ValueError("RGB values must be normalised between 0 and 1")
            colour_vec = vector(colour[0], colour[1], colour[2])
            self.__graphic_obj.color = colour_vec
        else:
            # Set to white if none given
            self.__graphic_obj.color = color.white

    def set_transparency(self, opacity):
        """
        Sets the transparency of the joint.

        :param opacity: Normalised value (0 -> 1) to set the opacity. 0 = transparent, 1 = opaque
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
        :raise ValueError: The given axis_of_rotation must be one of the default X, Y, Z vectors (e.g. x_axis_vector)
        """
        # Return new vectors to avoid pass by reference
        if axis.equals(x_axis_vector):
            return self.__x_vector
        elif axis.equals(y_axis_vector):
            return self.__y_vector
        elif axis.equals(z_axis_vector):
            return self.__z_vector
        else:
            error_str = "Bad input vector given ({0}). Must be either x_axis_vector ({1}), y_axis_vector ({2})," \
                        "or z_axis_vector ({3})."
            raise ValueError(error_str.format(axis, x_axis_vector, y_axis_vector, z_axis_vector))

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


class RotationalJoint(DefaultJoint):
    """
    A rotational joint based off the default joint class

    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float`, `str`
    """

    def __init__(self, initial_se3, structure):
        # Call super init function
        super().__init__(initial_se3, structure)
        self.rotation_axis = z_axis_vector
        # self.rotation_angle = radians(0)

    def rotate_joint(self, new_angle):
        """
        Rotate the joint to a given angle in range [-pi pi] (radians)

        :param new_angle: The new angle (radians) in range [-pi pi] that the link is to be rotated to.
        :type new_angle: `float`
        """
        raise PendingDeprecationWarning("Will likely be unused")

        # Wrap given angle to -pi to pi
        new_angle = wrap_to_pi("rad", new_angle)
        current_angle = self.rotation_angle
        # Calculate amount to rotate the link
        angle_diff = wrap_to_pi("rad", new_angle - current_angle)
        # Update the link
        self.rotate_around_joint_axis(angle_diff, self.rotation_axis)
        self.rotation_angle = new_angle

    def get_joint_type(self):
        """
        Return the type of joint (R for Rotational)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return "R"


class PrismaticJoint(DefaultJoint):
    """
    A prismatic joint based from the default joint class

    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float`, `str`
    """
    def __init__(self, initial_se3, structure):
        super().__init__(initial_se3, structure)
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


class StaticJoint(DefaultJoint):
    """
    This class represents a static joint (one that doesn't translate or rotate on it's own).
    It has no extra functions to utilise.

    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float`, `str`
    """

    def __init__(self, initial_se3, structure):
        super().__init__(initial_se3, structure)

    def get_joint_type(self):
        """
        Return the type of joint (S for Static)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return "S"


class Gripper(DefaultJoint):
    """
    This class represents a gripper joint with a moving gripper (To Be Implemented).
    Usually the end joint of a robot.

    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: class:`spatialmath.pose3d.SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float`, `str`
    """

    def __init__(self, initial_se3, structure):
        super().__init__(initial_se3, structure)

    # TODO close/open gripper

    def get_joint_type(self):
        """
        Return the type of joint (G for Gripper)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return "G"


class GraphicalRobot:
    """
    The GraphicalRobot class holds all of the different joints to easily control the robot arm.
    :param graphics_canvas: The canvas to add the robot to
    :type graphics_canvas: class:`GraphicsCanvas`
    """
    def __init__(self, graphics_canvas, name):
        self.joints = []
        self.num_joints = 0
        self.rob_shown = True
        self.ref_shown = True
        self.opacity = 1
        self.name = name
        # Add the robot to the canvas
        graphics_canvas.add_robot(self)

    def append_made_link(self, joint):
        """
        Append an already made joint to the end of the robot (Useful for links manually created)

        :param joint: A joint object already constructed
        :type joint: class:`graphics.graphics_robot.RotationalJoint`, class:`graphics.graphics_robot.PrismaticJoint`,
        class:`graphics.graphics_robot.StaticJoint`, class:`graphics.graphics_robot.Gripper`
        """
        self.joints.append(joint)
        self.num_joints += 1

    def append_link(self, typeof, pose, structure):
        """
        Append a joint to the end of the robot.

        :param typeof: String character of the joint type. e.g. 'R', 'P', 'S', 'G'
        :type typeof: `str`
        :param pose: SE3 object for the pose of the joint
        :type pose: class:`spatialmath.pose3d.SE3`
        :param structure: either a float of the length of the joint, or a str of the filepath to an STL to load
        :type structure: `float`, `str`
        :raises ValueError: typeof must be a valid character
        """
        # Capitalise the type for case-insensitive use
        typeof = typeof.upper()

        if typeof == 'R':
            link = RotationalJoint(pose, structure)
        elif typeof == 'P':
            link = PrismaticJoint(pose, structure)
        elif typeof == 'S':
            link = StaticJoint(pose, structure)
        elif typeof == 'G':
            link = Gripper(pose, structure)
        else:
            raise ValueError("typeof should be (case-insensitive) either 'R' (Rotational), 'P' (Prismatic), "
                             "'S' (Static), or 'G' (Gripper)")

        # Append the joint to the robot
        self.joints.append(link)
        self.num_joints += 1

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
        # Ensure deletion
        self.joints[self.num_joints - 1] = None

        # Resize list
        self.joints = self.joints[0:self.num_joints - 1]
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

        :param is_visible: Whether the reference frames should be visible or not.
        :type is_visible: `bool`
        """
        if is_visible is not self.ref_shown:
            for joint in self.joints:
                joint.draw_reference_frame(is_visible)
            self.ref_shown = is_visible

    def set_transparency(self, opacity):
        """
        Set the transparency of the robot.
        Allows for easier identification of the reference frames (if hidden by the robot itself)

        :param opacity: Normalised value (0 -> 1) to set the opacity. 0 = transparent, 1 = opaque
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
        :raises UserWarning: Robot must not have 0 joints, and given poses length must equal number of joints.
        """
        # Sanity checks
        if self.num_joints == 0:
            raise UserWarning("Robot has 0 joints. Create some using append_link()")

        if self.num_joints != len(all_poses):
            err = "Number of given poses {0} does not equal number of joints {1}"
            raise UserWarning(err.format(len(all_poses), self.num_joints))

        # Update the joints
        for idx in range(0, self.num_joints):
            self.joints[idx].update_pose(all_poses[idx])

    def animate(self, frame_poses, fps):
        """
        Calling this function will animate the robot through its frames.

        :param frame_poses: A 2D list of each joint pose for each frame.
        :type frame_poses: `list`
        :param fps: Number of frames per second to render at (limited by number of graphics trying to update)
        :type fps: `int`
        :raises ValueError: Number of frames and fps must be greater than 0
        """
        num_frames = len(frame_poses)
        # Validate num_frames
        if num_frames == 0:
            raise ValueError("0 frames were given. Supply at least 1 iteration of poses.")

        if fps <= 0:
            raise ValueError("fps must be greater than 0.")
        f = 1 / fps

        for poses in frame_poses:
            # Get current time
            t_start = perf_counter()

            self.set_joint_poses(poses)  # Validation done in set_joint_poses
            # Wait for scene to finish drawing
            scene.waitfor("draw_complete")

            # Get current time
            t_stop = perf_counter()

            # Wait for time of frame to finish
            # If drawing takes longer than frame frequency, this while is skipped
            while t_stop - t_start < f:
                t_stop = perf_counter()

    def set_joint_angle(self, link_num, new_angle):
        """
        Set the angle (radians) for a specific joint in the robot.

        :param link_num: Index of the joint in the robot arm (Base = 0, Gripper = end)
        :type link_num: `int`
        :param new_angle: The required angle to set the arm rotated towards
        :type new_angle: `float`
        :raise IndexError: Link index must be between 0 (inclusive) and number of joints (exclusive)
        :raise TypeError: The joint index chosen must be indexing a revolute joint
        """
        raise PendingDeprecationWarning("Will likely be unused")
        if (link_num < 0) or (link_num >= self.num_joints):
            error_str = "link number given ({0}) is not between range of 0 (inclusive) and {1} (exclusive)"
            raise IndexError(error_str.format(link_num, self.num_joints))

        # If the joint is a revolute
        if self.joints[link_num].get_joint_type() == "R":
            # If the angle already is as required, return
            if self.joints[link_num].rotation_angle == new_angle:
                return
            # Save prev angle to calculate rotation amount
            prev_angle = self.joints[link_num].rotation_angle
            # Rotate
            self.joints[link_num].rotate_joint(new_angle)
            # Calculate the vector representation of the axis rotated around
            rot_axis = self.joints[link_num].get_axis_vector(self.joints[link_num].rotation_axis)
            # For each next joint, apply the same rotation
            for affected_joint in range(link_num + 1, self.num_joints):
                rot_angle = new_angle - prev_angle
                self.joints[affected_joint].rotate_around_vector(rot_angle, rot_axis)
            # Reposition joints to connect back to each other
            self.__position_joints()
        else:
            error_str = "Given joint {0} is not a revolute joint. It is a {1} joint"
            raise TypeError(error_str.format(link_num, self.joints[link_num].get_joint_type()))

    def set_all_joint_angles(self, new_angles):
        """
        Set all of the angles for each joint in the robot.

        :param new_angles: List of new angles (radians) to set each joint to.
        Must have the same length as number of joints in robot arm, even if the joints aren't revolute
        :type new_angles: float list (radians)
        :raise IndexError: The length of the given list must equal the number of joints.
        """
        raise PendingDeprecationWarning("Will likely be unused")
        # Raise error if lengths don't match
        if len(new_angles) != len(self.joints):
            error_str = "Length of given angles ({0}) does not match number of joints ({1})."
            raise IndexError(error_str.format(len(new_angles), len(self.joints)))

        # For each joint
        for joint_num in range(0, self.num_joints):
            # If joint is a revolute
            if self.joints[joint_num].get_joint_type() == "R":
                new_angle = new_angles[joint_num]
                # If the angle is the already the same, return
                if self.joints[joint_num].rotation_angle == new_angle:
                    continue
                # Save prev angle to calculate rotation amount
                prev_angle = self.joints[joint_num].rotation_angle
                # Rotate
                self.joints[joint_num].rotate_joint(new_angle)
                # Calculate the vector representation of the axis that was rotated around
                rot_axis = self.joints[joint_num].get_axis_vector(self.joints[joint_num].rotation_axis)
                # For each successive joint, rotate it the same
                for affected_joint in range(joint_num + 1, self.num_joints):
                    rot_angle = new_angle - prev_angle
                    self.joints[affected_joint].rotate_around_vector(rot_angle, rot_axis)
            else:
                # Skip over any non-revolute joints. Print message saying so
                print("Joint", joint_num, "is not a revolute. Skipping...")
        # Reposition all joints to connect to the previous segment
        self.__position_joints()

    def print_joint_poses(self):
        """
        Print all of the current joint poses
        """
        # For each joint
        num = 0
        for joint in self.joints:
            print("Joint", num, "| Type:", joint.get_joint_type(), "| Pose:")
            print(joint.get_pose(), "\n")
            num += 1
