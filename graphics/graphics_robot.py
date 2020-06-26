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

        #self.update_pose(self.__pose)

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
        raise PendingDeprecationWarning("Currently out of date. Will be updated soon.")
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
        raise PendingDeprecationWarning("Currently out of date. Will be updated soon.")

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
        self.__pose = se_object
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
    def set_stl_joint_origin(self, current_location, required_location):
        set_stl_origin(self.__graphic_obj, current_location, required_location)

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

    # DONE
    def get_graphic_object(self):
        """
        Getter function that returns the graphical object of the joint

        :return: VPython graphical entity of the joint
        :rtype: `vpython.object`
        """
        return self.__graphic_obj


class RotationalJoint(DefaultJoint):
    """
    A rotational joint based off the default joint class

    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: `SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float` or `str`
    """

    # DONE
    def __init__(self, initial_se3, structure):
        # Call super init function
        super().__init__(initial_se3, structure)
        self.rotation_axis = z_axis_vector
        # self.rotation_angle = radians(0)

    # Keep, but might be unused
    def rotate_joint(self, new_angle):
        """
        Rotate the joint to a given angle in range [-pi pi] (radians)

        :param new_angle: The new angle in range [-pi pi] that the link is to be rotated to.
        :type new_angle: float (radians)
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

    # DONE
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
    :type initial_se3: `SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float` or `str`
    """
    # TODO
    def __init__(self, initial_se3, structure):
        super().__init__(initial_se3, structure)
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

    :param initial_se3: Pose to set the joint to initially
    :type initial_se3: `SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float` or `str`
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
    :type initial_se3: `SE3`
    :param structure: A variable representing the joint length (float) or a file path to an STL (str)
    :type structure: `float` or `str`
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
    The GraphicalRobot class encapsulates all of the different joint types to easily control the robot arm.
    """

    def __init__(self):
        self.joints = []
        self.num_joints = 0
        self.is_shown = True

    def append_made_link(self, joint):
        """
        Append an already made joint to the end of the robot (Useful for links manually created)
        :param joint: A joint object already constructed
        :type joint: `graphics.graphics_robot.DefaultJoint` or inherited classes
        """
        self.joints.append(joint)
        self.num_joints += 1

    def append_link(self, typeof, pose, structure):
        """
        Append a joint to the end of the robot.

        :param typeof: String character of the joint type. e.g. 'R', 'P', 'S', 'G'
        :type typeof: `str`
        :param pose: SE3 object for the pose of the joint
        :type pose: `SE3`
        :param structure: either a float of the length of the joint, or a str of the filepath to an STL to load
        :type structure: `float` or `str`
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

        # TODO handle clearing it from canvas

        # Keep all but the last joint
        self.joints = self.joints[0:self.num_joints-1]
        self.num_joints -= 1

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

    def set_joint_poses(self, all_poses):
        """
        Set the joint poses.

        :param all_poses: List of all the new poses to set
        :type all_poses: `SE3` list
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

    # CHANGE - internals referenced by SE3
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

    # DONE
    def print_joint_poses(self):
        """
        Print all of the current joint poses
        """
        # For each joint
        for joint in self.joints:
            print("Type:", joint.get_joint_type())
            print("\tPose:\n\t", joint.get_pose(), "\n")
