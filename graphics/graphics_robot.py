from graphics.graphics_canvas import *
from graphics.graphics_stl import *


class DefaultJoint:
    """
    This class forms a base for all the different types of joints
    - Rotational
    - Prismatic
    - Static
    - Gripper

    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: `SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float` or `str`
    """

    # DONE
    def __init__(self,
                 initial_se3,
                 structure=None):

        if not isinstance(structure, float) or not isinstance(structure, str):
            error_str = "structure must be of type {0} or {1}. Given {2}. Either give a length (float)," \
                        "or a file path to an STL (str)"
            raise TypeError(error_str.format(float, str, type(structure)))

        self.__pose = initial_se3

        # Set the graphic
        self.__graphic_obj = self.__set_graphic(structure)
        self.visible = True
        self.update_pose(self.__pose)

        # Calculate the length of the link (Generally longest side is the length)
        self.__length = max(self.__graphic_obj.length, self.__graphic_obj.width, self.__graphic_obj.height)

        # Set the other reference frame vectors
        self.__graphic_ref = draw_reference_frame_axes(self.__pose)

    # Keep, but will be for private use??
    def rotate_around_joint_axis(self, angle_of_rotation, axis_of_rotation):
        """
        Rotate the joint by a specific amount around one of the joints xyz axes

        :param angle_of_rotation: +/- angle of rotation to apply
        :type angle_of_rotation: float (radians)
        :param axis_of_rotation: X, Y, or Z axis to apply around the objects specific X, Y, or Z axes
        :type axis_of_rotation: class:`vpython.vector`
        :raise ValueError: The given axis_of_rotation must be one of the default X, Y, Z vectors (e.g. x_axis_vector)
        """
        raise DeprecationWarning("Currently out of date. Will be updated soon.")
        # Determine the axis of rotation based on the given joint axis direction
        # Then add the rotation amount to the axis counter
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

    # Keep, but will be for private use??
    def rotate_around_vector(self, angle_of_rotation, axis_of_rotation):
        """
        Rotate the object around a particular vector. Is utilised when rotating a different joint,
        and updating the subsequent joints.

        :param angle_of_rotation: +/- angle of rotation to apply
        :type angle_of_rotation: float (radians)
        :param axis_of_rotation: X, Y, or Z axis to apply around the objects specific X, Y, or Z axes
        :type axis_of_rotation: class:`vpython.vector`
        """
        raise DeprecationWarning("Currently out of date. Will be updated soon.")

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

    # DONE
    def update_position(self, se_object):
        """
        Given an SE object, update just the orientation of the joint.

        :param se_object: SE3 pose representation of the joint
        :type se_object: `SE3`
        """
        new_position = get_pose_pos(se_object)
        self.__graphic_obj.pos = new_position

        # Update the reference frame
        self.__update_reference_frame()
        self.draw_reference_frame(self.__graphic_ref.visible)

    # DONE
    def update_orientation(self, se_object):
        """
        Given an SE object, update just the orientation of the joint.

        :param se_object: SE3 pose representation of the joint
        :type se_object: `SE3`
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

    # DONE
    def update_pose(self, se_object):
        """
        Given an SE object, update the pose of the joint.

        :param se_object: SE3 pose representation of the joint
        :type se_object: `SE3`
        """
        # Get the new pose details
        new_x_axis = get_pose_x_vec(se_object)
        new_y_axis = get_pose_y_vec(se_object)
        # new_z_axis = get_pose_z_vec(se_object)  # not needed
        new_position = get_pose_pos(se_object)

        # Update the graphic object
        self.__graphic_obj.axis = new_x_axis
        self.__graphic_obj.up = new_y_axis
        self.__graphic_obj.pos = new_position

        # Update the reference frame
        self.__update_reference_frame()
        self.draw_reference_frame(self.__graphic_ref.visible)

    # DONE
    def __update_reference_frame(self):
        """
        Update the reference frame axis vectors
        """
        self.__x_vector = get_pose_x_vec(self.__pose)
        self.__y_vector = get_pose_y_vec(self.__pose)
        self.__z_vector = get_pose_z_vec(self.__pose)

    # DONE
    def draw_reference_frame(self, is_visible):
        """
        Draw a reference frame at the tool point position.

        :param is_visible: Whether the reference frame should be drawn or not
        :type is_visible: bool
        """
        # If not visible, turn off
        if not is_visible:
            # If a reference frame exists
            if self.__graphic_ref is not None:
                # Set invisible, and also update its orientations
                self.__graphic_ref.visible = False
                self.__graphic_ref.pos = get_pose_pos(self.__pose)
                self.__graphic_ref.axis = get_pose_x_vec(self.__pose)
                self.__graphic_ref.up = get_pose_y_vec(self.__pose)
        # Else: draw
        else:
            # If graphic does not currently exist
            if self.__graphic_ref is None:
                # Create one
                self.__graphic_ref = draw_reference_frame_axes(self.__pose)
            # Else graphic does exist
            else:
                self.__graphic_ref.pos = get_pose_pos(self.__pose)
                self.__graphic_ref.axis = get_pose_x_vec(self.__pose)
                self.__graphic_ref.up = get_pose_y_vec(self.__pose)

    # DONE
    def set_joint_visibility(self, is_visible):
        """
        Choose whether or not the joint is displayed in the canvas.

        :param is_visible: Whether the joint should be drawn or not
        :type is_visible: bool
        """
        # If the option is different to current setting
        if is_visible is not self.visible:
            # Update
            self.__graphic_obj.visible = is_visible
            self.__graphic_ref.visible = is_visible
            self.visible = is_visible

    # DONE - May need modifying
    def __set_graphic(self, structure):
        """
        Set the graphic object depending on if one was given. If no object was given, create a box and return it

        :param structure: `float` or `str` representing the joint length or STL path to load from
        :type structure: `float` or `str`
        :return: Graphical object for the joint
        :rtype: class:`vpython.compound`
        """
        if isinstance(structure, float):
            length = structure

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

    # DONE
    def __import_texture(self):
        # TODO (much later)
        pass

    # DONE
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

    # DONE
    def get_pose(self):
        """
        Return the current pose of the joint

        :return: SE3 representation of the current joint pose
        :rtype: `SE3`
        """
        return self.__pose

    # DONE
    def get_joint_type(self):
        """
        Return the type of joint (To Be Overridden by Child classes)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return ""


class RotationalJoint(DefaultJoint):
    """
    A rotational joint based off the default joint class

    :param connection_from_prev_seg: Origin point of the joint (Where it connects to the previous segment)
    :type connection_from_prev_seg: class:`vpython.vector`
    :param connection_to_next_seg: Tooltip point of the joint (Where it connects to the next segment)
    :type connection_to_next_seg: class:`vpython.vector`
    :param x_axis: Vector representation of the joints +x axis, defaults to +x axis (1, 0, 0)
    :type x_axis: class:`vpython.vector`
    :param rotation_axis: Vector representation of the joint axis that it rotates around, defaults to +y axis (0, 1, 0)
    :type rotation_axis: class:`vpython.vector`
    :param graphic_obj: Graphical object for which the joint will use. If none given, auto generates an object,
    defaults to `None`
    :type graphic_obj: class:`vpython.compound`
    """

    # CHANGE - internals referenced by SE3
    def __init__(self,
                 connection_from_prev_seg,
                 connection_to_next_seg,
                 x_axis=x_axis_vector,
                 rotation_axis=y_axis_vector,
                 graphic_obj=None):
        # Call super init function
        super().__init__(connection_from_prev_seg, connection_to_next_seg, x_axis, graphic_obj)
        self.rotation_axis = rotation_axis
        self.rotation_angle = radians(0)

    # Keep, but might be unused
    def rotate_joint(self, new_angle):
        """
        Rotate the joint to a given angle in range [-pi pi] (radians)

        :param new_angle: The new angle in range [-pi pi] that the link is to be rotated to.
        :type new_angle: float (radians)
        """
        # Wrap given angle to -pi to pi
        new_angle = wrap_to_pi("rad", new_angle)
        current_angle = self.rotation_angle
        # Calculate amount to rotate the link
        angle_diff = wrap_to_pi("rad", new_angle - current_angle)
        # Update the link
        self.rotate_around_joint_axis(angle_diff, self.rotation_axis)
        self.rotation_angle = new_angle

    # DONE
    def get_joint_type(self):
        """
        Return the type of joint (R for Rotational)

        :return: String representation of the joint type.
        :rtype: `str`
        """
        return "R"


class PrismaticJoint(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, x_axis, graphic_obj=None):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, x_axis, graphic_obj)
        self.min_translation = None
        self.max_translation = None

    def translate_joint(self, new_translation):
        # TODO calculate new connectTo point, update relevant super() params
        # TODO Update graphic
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

    :param connection_from_prev_seg: Origin point of the joint (Where it connects to the previous segment)
    :type connection_from_prev_seg: class:`vpython.vector`
    :param connection_to_next_seg: Tooltip point of the joint (Where it connects to the next segment)
    :type connection_to_next_seg: class:`vpython.vector`
    :param x_axis: Vector representation of the joints +x axis, defaults to +x axis (1, 0, 0)
    :type x_axis: class:`vpython.vector`
    :param graphic_obj: Graphical object for which the joint will use. If none given, auto generates an object,
    defaults to `None`
    :type graphic_obj: class:`vpython.compound`
    """

    def __init__(self, connection_from_prev_seg, connection_to_next_seg, x_axis, graphic_obj=None):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, x_axis, graphic_obj)

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

    :param connection_from_prev_seg: Origin point of the joint (Where it connects to the previous segment)
    :type connection_from_prev_seg: class:`vpython.vector`
    :param connection_to_next_seg: Tooltip point of the joint (Where it connects to the next segment)
    :type connection_to_next_seg: class:`vpython.vector`
    :param x_axis: Vector representation of the joints +x axis, defaults to +x axis (1, 0, 0)
    :type x_axis: class:`vpython.vector`
    :param graphic_obj: Graphical object for which the joint will use. If none given, auto generates an object,
    defaults to `None`
    :type graphic_obj: class:`vpython.compound`
    """

    def __init__(self, connection_from_prev_seg, connection_to_next_seg, x_axis, graphic_obj=None):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, x_axis, graphic_obj)

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
    The GraphicalRobot class encapsulates all of the different joint types to easily control the robot arm.

    :param joints: A list of the joints in order from base (0) to gripper (end), or other types.
    :type joints: list
    :raise ValueError: The given length of joints must not be 0
    """

    def __init__(self, joints):
        if len(joints) == 0:
            raise ValueError("Robot was given", len(joints), "joints. Must have at least 1.")
        self.joints = joints
        self.num_joints = len(joints)
        self.is_shown = True
        self.__create_robot()

    def __create_robot(self):
        """
        Upon creation of the robot, orient all objects correctly.
        """
        self.__position_joints()

    # CHANGE - internals referenced by SE3
    def __position_joints(self):
        """
        Position all joints based upon each respective connect_from and connect_to points.
        """
        # For each joint in the robot (exclude base)
        for joint_num in range(1, self.num_joints):
            # Place the joint connect_from (origin) to the previous segments connect_to
            self.joints[joint_num].update_position(self.joints[joint_num - 1].get_connection_to_pos())

    # DONE
    def set_robot_visibility(self, is_visible):
        """
        Set the entire robots visibility inside the canvas.

        :param is_visible: Whether the robot should be visible or not.
        :type is_visible: bool
        """
        if is_visible is not self.is_shown:
            for joint in self.joints:
                joint.set_joint_visibility(is_visible)
                self.is_shown = is_visible

    # DONE
    def set_reference_visibility(self, is_visible):
        """
        Set the visibility of the reference frame for all joints.

        :param is_visible: Whether the reference frames should be visible or not.
        :type is_visible: bool
        """
        for joint in self.joints:
            joint.draw_reference_frame(is_visible)

    # CHANGE - internals referenced by SE3
    def set_joint_angle(self, link_num, new_angle):
        """
        Set the angle (radians) for a specific joint in the robot.

        :param link_num: Index of the joint in the robot arm (Base = 0, Gripper = end)
        :type link_num: int
        :param new_angle: The required angle to set the arm rotated towards
        :type new_angle: float (radians)
        :raise IndexError: Link index must be between 0 (inclusive) and number of joints (exclusive)
        :raise TypeError: The joint index chosen must be indexing a revolute joint
        """
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

    # CHANGE - internals referenced by SE3
    def set_all_joint_angles(self, new_angles):
        """
        Set all of the angles for each joint in the robot.

        :param new_angles: List of new angles (radians) to set each joint to.
        Must have the same length as number of joints in robot arm, even if the joints aren't revolute
        :type new_angles: float list (radians)
        :raise IndexError: The length of the given list must equal the number of joints.
        """
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

    # CHANGE - internals referenced by SE3
    def move_base(self, position):
        """
        Move the base around to a particular position.

        :param position: 3D position to move the base's origin to
        :type position: class:`vpython.vector`
        """
        # Move the base, then update all of the joints
        self.joints[0].update_position(position)
        self.__position_joints()

    # CHANGE - internals referenced by SE3
    def print_joint_angles(self, is_degrees=False):
        """
        Print all of the current joint angles (Local rotation and total rotation (rotation from other joints))

        :param is_degrees: Whether or not to display angles as degrees or radians (default)
        :type is_degrees: bool, optional
        """
        # For each joint
        for joint in range(0, self.num_joints):
            total_x = self.joints[joint].get_rotation_angle(x_axis_vector)
            total_y = self.joints[joint].get_rotation_angle(y_axis_vector)
            total_z = self.joints[joint].get_rotation_angle(z_axis_vector)

            if is_degrees:
                total_x = round(degrees(total_x), 3)
                total_y = round(degrees(total_y), 3)
                total_z = round(degrees(total_z), 3)

            # If revolute
            if self.joints[joint].get_joint_type() == "R":
                local_angle = self.joints[joint].rotation_angle
                if is_degrees:
                    local_angle = round(degrees(local_angle), 3)
                print("Joint", joint,
                      "\n\tLocal angle =", local_angle,
                      "\n\tTotal angles (x,y,z)= (", total_x, ",", total_y, ",", total_z, ")", )
            # If not a revolute
            else:
                print("Joint", joint,
                      "\n\tLocal angle = <Not a rotating joint>",
                      "\n\tTotal angles (x,y,z)= (", total_x, ",", total_y, ",", total_z, ")", )
