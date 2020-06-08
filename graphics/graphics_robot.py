from graphics.graphics_canvas import *
from graphics.common_functions import *

class DefaultJoint:
    """
    This class forms a base for all the different types of joints
    - Rotational
    - Prismatic
    - Static
    - Gripper

    :param connection_from_prev_seg: Origin point of the joint (Where it connects to the previous segment)
    :type connection_from_prev_seg: class:`vpython.vector`
    :param connection_to_next_seg: Tooltip point of the joint (Where it connects to the next segment)
    :type connection_to_next_seg: class:`vpython.vector`
    :param axis: Vector representation of the joints +x axis, defaults to +x axis (1, 0, 0)
    :type axis: class:`vpython.vector`
    :param graphic_object: Graphical object for which the joint will use. If none given, auto generates an object,
    defaults to `None`
    :type graphic_object: class:`vpython.compound`
    """

    def __init__(self,
                 connection_from_prev_seg,
                 connection_to_next_seg,
                 axis=x_axis_vector,
                 graphic_object=None):

        # Set connection points
        self.__connect_from = connection_from_prev_seg
        self.__connect_to = connection_to_next_seg
        # Set an arrow to track position and direction for easy updates (auto applies transforms)
        self.__connect_dir = arrow(pos=self.__connect_from,
                                   axis=(self.__connect_to - self.__connect_from),
                                   visible=False)
        # Set the x vector direction
        self.__x_vector = axis

        # Set the rotation angles
        self.__x_rotation = radians(0)
        self.__y_rotation = radians(0)
        self.__z_rotation = radians(0)

        # Set the graphic
        self.__graphic_obj = self.__set_graphic(graphic_object)
        self.visible = True

        # Calculate the length of the link (Generally longest side is the length)
        self.__length = max(self.__graphic_obj.length, self.__graphic_obj.width, self.__graphic_obj.height)

        # Set the other reference frame vectors
        self.__graphic_ref = draw_reference_frame_axes(self.__connect_to, self.__x_vector, self.__x_rotation)
        self.__update_reference_frame()

    def update_position(self, new_pos):
        """
        Move the position of the link to the specified location

        :param new_pos: 3D vector representing the new location for the origin (connection_from) of the link
        :type new_pos: class:`vpython.vector`
        """
        # Calculate translational movement amount
        axes_movement = new_pos - self.__connect_from
        # Update each position
        self.__connect_from += axes_movement
        self.__connect_to += axes_movement
        self.__connect_dir.pos += axes_movement
        # If the reference frame exists, redraw it
        if self.__graphic_ref is not None:
            self.draw_reference_frame(self.__graphic_ref.visible)
        self.__draw_graphic()

    def rotate_around_joint_axis(self, angle_of_rotation, axis_of_rotation):
        """
        Rotate the joint by a specific amount around one of the joints xyz axes

        :param angle_of_rotation: +/- angle of rotation to apply
        :type angle_of_rotation: float (radians)
        :param axis_of_rotation: X, Y, or Z axis to apply around the objects specific X, Y, or Z axes
        :type axis_of_rotation: class:`vpython.vector`
        """
        # Determine the axis of rotation based on the given joint axis direction
        # Then add the rotation amount to the axis counter
        if axis_of_rotation.equals(x_axis_vector):
            rotation_axis = self.__x_vector
            self.__x_rotation = wrap_to_pi(self.__x_rotation + angle_of_rotation)
        elif axis_of_rotation.equals(y_axis_vector):
            rotation_axis = self.__y_vector
            self.__y_rotation = wrap_to_pi(self.__y_rotation + angle_of_rotation)
        elif axis_of_rotation.equals(z_axis_vector):
            rotation_axis = self.__z_vector
            self.__z_rotation = wrap_to_pi(self.__z_rotation + angle_of_rotation)
        else:
            # Default to the y-axis
            rotation_axis = self.__y_vector
            self.__y_rotation = wrap_to_pi(self.__y_rotation + angle_of_rotation)

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

    def rotate_around_vector(self, angle_of_rotation, axis_of_rotation):
        """
        Rotate the object around a particular vector. Is utilised when rotating a different joint,
        and updating the subsequent joints.

        :param angle_of_rotation: +/- angle of rotation to apply
        :type angle_of_rotation: float (radians)
        :param axis_of_rotation: X, Y, or Z axis to apply around the objects specific X, Y, or Z axes
        :type axis_of_rotation: class:`vpython.vector`
        """
        # TODO
        #  calculate amount to update x,y,z angles by based on vectors and angle
        x_prev, y_prev, z_prev = vector(self.__x_vector), vector(self.__y_vector), vector(self.__z_vector)

        # Rotate the graphic object of the link to automatically transform the xyz axes and the graphic
        self.__graphic_obj.rotate(angle=angle_of_rotation, axis=axis_of_rotation, origin=self.__connect_from)

        # Update the vectors and reference frames
        self.__update_reference_frame()
        x_new, y_new, z_new = self.__x_vector, self.__y_vector, self.__z_vector

        angle_diff_x, angle_diff_y, angle_diff_z = diff_angle(x_prev, x_new), \
                                                   diff_angle(y_prev, y_new), \
                                                   diff_angle(z_prev, z_new)

        print("DIFF", round(degrees(angle_diff_x)), round(degrees(angle_diff_y)), round(degrees(angle_diff_z)), "\n")
        min_angle_diff = min(angle_diff_x, angle_diff_y, angle_diff_z)
        if min_angle_diff == angle_diff_x:
            self.__x_rotation = wrap_to_pi(self.__x_rotation + angle_of_rotation)
        elif min_angle_diff == angle_diff_y:
            self.__y_rotation = wrap_to_pi(self.__y_rotation + angle_of_rotation)
        else:
            self.__z_rotation = wrap_to_pi(self.__z_rotation + angle_of_rotation)

        # Calculate the updated toolpoint location
        self.__connect_dir.rotate(angle=angle_of_rotation, axis=axis_of_rotation)
        self.__connect_to = self.__connect_dir.pos + self.__connect_dir.axis
        # If the reference frame exists, redraw it
        if self.__graphic_ref is not None:
            self.draw_reference_frame(self.__graphic_ref.visible)
        self.__draw_graphic()

    def __update_reference_frame(self):
        """
        Update the reference frame axis vectors
        """
        # X vector is through the tooltip
        self.__x_vector = self.__graphic_obj.axis
        # self.x_vector.mag = self.__length
        # Y vector is in the 'up' direction of the object
        self.__y_vector = self.__graphic_obj.up
        # self.y_vector.mag = self.__length
        # Z vector is the cross product of the two
        self.__z_vector = self.__x_vector.cross(self.__y_vector)
        # self.z_vector.mag = self.__length

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
                self.__graphic_ref.pos = self.__connect_to
                self.__graphic_ref.axis = self.__x_vector
                self.__graphic_ref.up = self.__y_vector
        # Else: draw
        else:
            # If graphic does not currently exist
            if self.__graphic_ref is None:
                # Create one
                self.__graphic_ref = draw_reference_frame_axes(self.__connect_to, self.__x_vector, self.__x_rotation)
            # Else graphic does exist
            else:
                self.__graphic_ref.pos = self.__connect_to
                self.__graphic_ref.axis = self.__x_vector
                self.__graphic_ref.up = self.__y_vector

    def __draw_graphic(self):
        """
        Draw the objects graphic on screen
        """
        self.__graphic_obj.pos = self.__connect_from
        self.__graphic_obj.axis = self.__x_vector
        self.__graphic_obj.up = self.__y_vector

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

    def __set_graphic(self, given_obj):
        """
        Set the graphic object depending on if one was given. If no object was given, create a box and return it

        :param given_obj: Graphical object for the joint
        :type given_obj: class:`vpython.compound`
        :return: New graphical object for the joint
        :rtype: class:`vpython.compound`
        """
        # If stl_filename is None, use a box
        if given_obj is None:
            box_midpoint = vector(
                (self.__connect_to - self.__connect_from).mag / 2,
                0,
                0
            )
            # Create a box along the +x axis, with the origin (point of rotation) at (0, 0, 0)
            graphic_obj = box(pos=vector(box_midpoint.x, box_midpoint.y, box_midpoint.z),
                              axis=x_axis_vector,
                              size=vector((self.__connect_to - self.__connect_from).mag, 0.1, 0.1),
                              up=z_axis_vector)
            # Set the boxes new origin
            graphic_obj = compound([graphic_obj], origin=vector(0, 0, 0), axis=x_axis_vector)
            return graphic_obj
        else:
            # TODO When texture application available, put it here
            return given_obj

    def __import_texture(self):
        # TODO (much later)
        pass

    def get_connection_to_pos(self):
        """
        Return the private variable containing the connection position (toolpoint)

        :return: Connect_to (toolpoint) position
        :rtype: class:`vpython.vector`
        """
        return self.__connect_to

    def get_rotation_angle(self, axis):
        """
        Get the current angle of rotation around a specified X, Y, or Z axis

        :param axis: Specified joint axis to get the angle of rotation of
        :type axis: class:`vpython.vector`
        :return: Current angle of rotation with respect to world (includes rotation from previous joints)
        :rtype: float (radians)
        """
        if axis.equals(x_axis_vector):
            return self.__x_rotation
        elif axis.equals(y_axis_vector):
            return self.__y_rotation
        elif axis.equals(z_axis_vector):
            return self.__z_rotation
        else:
            return self.__y_rotation

    def get_axis_vector(self, axis):
        """
        Get the current vector of a specified X, Y, or Z axis

        :param axis: Specified joint axis to get the angle of rotation of
        :type axis: class:`vpython.vector`
        :return: Current vector representation of the joints X, Y, or Z axis
        :rtype: class:`vpython.vector`
        """
        if axis.equals(x_axis_vector):
            return self.__x_vector
        elif axis.equals(y_axis_vector):
            return self.__y_vector
        elif axis.equals(z_axis_vector):
            return self.__z_vector
        else:
            return self.__y_vector

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

    def __init__(self,
                 connection_from_prev_seg,
                 connection_to_next_seg,
                 x_axis=x_axis_vector,
                 rotation_axis=y_axis_vector,
                 graphic_obj=None):
        # Call super init function
        super().__init__(connection_from_prev_seg, connection_to_next_seg, x_axis, graphic_obj)
        # TODO
        #  sanity check input
        self.rotation_axis = rotation_axis
        self.rotation_angle = radians(0)

    def rotate_joint(self, new_angle):
        """
        Rotate the joint to a given angle in range [-pi pi] (radians)

        :param new_angle: The new angle in range [-pi pi] that the link is to be rotated to.
        :type new_angle: float (radians)
        """
        # Wrap given angle to -pi to pi
        new_angle = wrap_to_pi(new_angle)
        current_angle = self.rotation_angle
        # Calculate amount to rotate the link
        angle_diff = wrap_to_pi(new_angle - current_angle)
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
    # TODO:
    #  Have functions to update links,
    #  take in rotation, translation, etc, params
    """
    The GraphicalRobot class encapsulates all of the different joint types to easily control the robot arm.

    :param joints: A list of the joints in order from base (0) to gripper (end), or other types.
    :type joints: list
    """

    def __init__(self, joints):
        # TODO sanity check input
        self.joints = joints
        self.num_joints = len(joints)
        self.is_shown = True
        self.__create_robot()

    def __create_robot(self):
        """
        Upon creation of the robot, orient all objects correctly.
        """
        self.__position_joints()

    def __position_joints(self):
        """
        Position all joints based upon each respective connect_from and connect_to points.
        """
        # For each joint in the robot (exclude base)
        for joint_num in range(1, self.num_joints):
            # Place the joint connect_from (origin) to the previous segments connect_to
            self.joints[joint_num].update_position(self.joints[joint_num - 1].get_connection_to_pos())

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

    def set_reference_visibility(self, is_visible):
        """
        Set the visibility of the reference frame for all joints.

        :param is_visible: Whether the reference frames should be visible or not.
        :type is_visible: bool
        """
        for joint in self.joints:
            joint.draw_reference_frame(is_visible)

    def set_joint_angle(self, link_num, new_angle):
        """
        Set the angle (radians) for a specific joint in the robot.

        :param link_num: Index of the joint in the robot arm (Base = 0, Gripper = end)
        :type link_num: int
        :param new_angle: The required angle to set the arm rotated towards
        :type new_angle: float (radians)
        """
        # If the joint is a revolute
        if self.joints[link_num].get_joint_type() == "R":
            # If the angle already is as required, return
            if self.joints[link_num].rotation_angle == new_angle:
                return
            # Rotate
            self.joints[link_num].rotate_joint(new_angle)
            # Calculate the vector representation of the axis rotated around
            rot_axis = self.joints[link_num].get_axis_vector(self.joints[link_num].rotation_axis)
            # For each next joint, apply the same rotation
            for affected_joint in range(link_num + 1, self.num_joints):
                self.joints[affected_joint].rotate_around_vector(new_angle, rot_axis)
            # Reposition joints to connect back to each other
            self.__position_joints()
        else:
            # TODO error handling
            pass

    def set_all_joint_angles(self, new_angles):
        """
        Set all of the angles for each joint in the robot.

        :param new_angles: List of new angles (radians) to set each joint to. Must have the same length as number of joints in
        robot arm, even if the joints aren't revolute
        :type new_angles: float list (radians)
        """
        # TODO error handling (Out of bounds, not revolute)
        # For each joint
        for joint_num in range(0, self.num_joints):
            # If joint is a revolute
            if self.joints[joint_num].get_joint_type() == "R":
                # If the angle is the already the same, return
                if self.joints[joint_num].rotation_angle == new_angles[joint_num]:
                    continue
                # Rotate
                self.joints[joint_num].rotate_joint(new_angles[joint_num])
                # Calculate the vector representation of the axis that was rotated around
                rot_axis = self.joints[joint_num].get_axis_vector(self.joints[joint_num].rotation_axis)
                # For each successive joint, rotate it the same
                for affected_joint in range(joint_num + 1, self.num_joints):
                    self.joints[affected_joint].rotate_around_vector(new_angles[joint_num], rot_axis)
            else:
                pass
        # Reposition all joints to connect to the previous segment
        self.__position_joints()

    def move_base(self, position):
        """
        Move the base around to a particular position.

        :param position: 3D position to move the base's origin to
        :type position: class:`vpython.vector`
        """
        # TODO sanity check input
        # Move the base, then update all of the joints
        self.joints[0].update_position(position)
        self.__position_joints()

    def print_joint_angles(self, is_degrees=False):
        """
        Print all of the current joint angles (Local rotation and total rotation (rotation from other joints))

        :param is_degrees: Whether or not to display angles as degrees or radians (default)
        :type is_degrees: bool, optional
        """
        # TODO degrees conversion
        # For each joint
        for joint in range(0, self.num_joints):
            # If revolute
            if self.joints[joint].get_joint_type() == "R":
                print("Joint", joint,
                      "\n\tLocal angle =", self.joints[joint].rotation_angle,
                      "\n\tTotal angles (x,y,z)= (",
                      self.joints[joint].get_rotation_angle(x_axis_vector), ",",
                      self.joints[joint].get_rotation_angle(y_axis_vector), ",",
                      self.joints[joint].get_rotation_angle(z_axis_vector), ")", )
            # If not a revolute
            else:
                print("Joint", joint,
                      "\n\tLocal angle = <Not a rotating joint>",
                      "\n\tTotal angles (x,y,z)= (",
                      self.joints[joint].get_rotation_angle(x_axis_vector), ",",
                      self.joints[joint].get_rotation_angle(y_axis_vector), ",",
                      self.joints[joint].get_rotation_angle(z_axis_vector), ")", )
