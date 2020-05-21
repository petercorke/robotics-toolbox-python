from graphics.graphics_canvas import *
from graphics.graphics_stl import *
from graphics.common_functions import wrap_to_pi


class DefaultJoint:
    """
    This class forms a base for all the different types of joints
    - Rotational
    - Translational
    - Static
    - Gripper

    :param connection_from_prev_seg: Origin point of the joint (Where it connects to the previous segment)
    :type connection_from_prev_seg: class:`vpython.vector`
    :param connection_to_next_seg: Tooltip point of the joint (Where it connects to the next segment)
    :type connection_to_next_seg: class:`vpython.vector`
    """
    def __init__(self, connection_from_prev_seg, connection_to_next_seg):
        # Set connection points
        self.__connect_from = connection_from_prev_seg
        self.__connect_to = connection_to_next_seg

        # Calculate the length of the link
        self.__length = mag(self.__connect_to - self.__connect_from)

        # Change the directional vector magnitude to match the length
        self.x_vector = self.__connect_to - self.__connect_from
        self.x_vector.mag = self.__length

        # Set the other reference frame vectors
        self.__graphic_ref = draw_reference_frame_axes(self.__connect_to, self.x_vector, radians(0))
        self.__update_reference_frame()

        # Calculate the arm angle
        self.arm_angle = self.calculate_arm_angle()
        # TODO self.x_rotation = None

        # Set the graphic
        # TODO using default box for now
        box_midpoint = vector(
            (self.__connect_from.x + self.__connect_to.x) / 2,
            (self.__connect_from.y + self.__connect_to.y) / 2,
            (self.__connect_from.z + self.__connect_to.z) / 2
        )
        # NB: Set XY axis first, as vpython is +y up bias, objects rotate respective to this bias when setting axis
        self.__graphic_obj = box(pos=vector(box_midpoint.x, box_midpoint.y, 0),
                                 axis=vector(self.x_vector.x, self.x_vector.y, 0),
                                 size=vector(self.__length, 0.1, 0.1), up=vector(0, 0, 1))
        self.__graphic_obj.axis = self.x_vector
        self.__graphic_obj.pos = box_midpoint
        self.__graphic_obj.rotate(radians(0))

    def update_position(self, new_pos):
        """
        Move the position of the link to the specified location

        :param new_pos: 3D vector representing the new location for the origin (connection_from) of the link
        :type new_pos: class:`vpython.vector`
        """
        # Calculate translational movement amount
        axes_movement = self.__connect_from + new_pos
        # Update each position
        self.__connect_from += axes_movement
        self.__connect_to += axes_movement
        # If the reference frame exists, redraw it
        if self.__graphic_ref is not None:
            self.draw_reference_frame(self.__graphic_ref.visible)
            self.__update_reference_frame()
        self.__draw_graphic()

    def update_orientation(self, new_direction):
        """
        Orient the link to face the direction of the given vector (respective from the link origin (connect_from))

        :param new_direction: vector representation of the direction the link now represents
        :type new_direction: class:`vpython.vector`
        """
        # Set magnitude to reflect link length
        new_direction.mag = self.__length
        # Set the new direction and connection end point (tool tip)
        self.x_vector = new_direction
        self.__connect_to = self.__connect_from + new_direction
        # If the reference frame exists, redraw it
        if self.__graphic_ref is not None:
            self.draw_reference_frame(self.__graphic_ref.visible)
            self.__update_reference_frame()
        self.__draw_graphic()
        # Calculate the updated arm angle
        self.arm_angle = self.calculate_arm_angle()

    def __update_reference_frame(self):
        """
        Update the reference frame axis vectors
        """
        # X vector is through the tooltip
        self.x_vector = self.__connect_to - self.__connect_from
        self.x_vector.mag = self.__length
        # Y vector is in the 'up' direction of the object
        self.y_vector = self.__graphic_ref.up
        self.y_vector.mag = self.__length
        # Z vector is the cross product of the two
        self.z_vector = self.x_vector.cross(self.y_vector)
        self.z_vector.mag = self.__length

    def draw_reference_frame(self, is_visible):
        """
        Draw a reference frame at the tool point position
        :param is_visible: Whether the reference frame should be drawn or not
        :type is_visible: bool
        """
        # TODO update with 'rotate'
        # If not visible, turn off
        if not is_visible:
            # If a reference frame exists
            if self.__graphic_ref is not None:
                # Set invisible, and also update its orientations
                self.__graphic_ref.visible = False
                self.__graphic_ref.pos = self.__connect_to
                self.__graphic_ref.axis = self.x_vector
                self.__graphic_ref.rotate(angle=radians(0))
        # Else: draw
        else:
            # If graphic does not currently exist
            if self.__graphic_ref is None:
                # Create one
                self.__graphic_ref = draw_reference_frame_axes(self.__connect_to, self.x_vector, radians(0))
            # Else graphic does exist
            else:
                self.__graphic_ref.pos = self.__connect_to
                self.__graphic_ref.axis = self.x_vector
                self.__graphic_ref.rotate(angle=radians(0))

    def __draw_graphic(self):
        """
        Draw the objects graphic on screen
        """
        # Midpoint of the box is it's origin/position
        box_midpoint = vector(
            (self.__connect_from.x + self.__connect_to.x) / 2,
            (self.__connect_from.y + self.__connect_to.y) / 2,
            (self.__connect_from.z + self.__connect_to.z) / 2
        )
        self.__graphic_obj.pos = box_midpoint
        self.__graphic_obj.axis = self.x_vector
        self.__graphic_obj.size = vector(self.__length, 0.1, 0.1)

    def calculate_arm_angle(self):
        """
        Calculate the arm angle respective to the XY (ground) plane

        :return: Angle of the arm from the XY (ground) plane
        :rtype: float (radians)
        """
        # TODO
        #  Do checks for x-axis rotation (will affect Z ref direction)

        # Angle between arm and XY plane
        xy_plane_angle = asin(self.x_vector.z / (sqrt(1) * self.x_vector.mag))
        # Is the arm angled above or below the horizontal plane
        xy_plane_sign = sign(xy_plane_angle) == 1
        # If the arm is facing up or down
        if abs(self.z_vector.z) < 0.0001:
            # Reference frame Z axis is opposite to the XY plane angle
            ref_z_sign = not xy_plane_sign
        else:
            # Arm is facing in the direction of the reference frame Z axis
            ref_z_sign = sign(self.z_vector.z) == 1

        # Quadrant modifications to the angle between the arm and XY plane
        # X-Y | Z | ans
        #  +  | + | ans
        #  +  | - | 180-ans
        #  -  | + | ans
        #  -  | - | -(180+ans)
        if xy_plane_sign and ref_z_sign:
            xy_plane_angle += 0
        elif xy_plane_sign and not ref_z_sign:
            xy_plane_angle = radians(180) - xy_plane_angle
        elif not xy_plane_sign and ref_z_sign:
            xy_plane_angle += 0
        elif not xy_plane_sign and not ref_z_sign:
            xy_plane_angle = -(radians(180) + xy_plane_angle)

        return xy_plane_angle

    # TODO MAY NOT BE NEEDED
    def __rotate(self, axis_of_rotation, angle_of_rotation):
        # TODO
        #  (Rotate around a given axis, by a given angle,
        #  in case stl has loaded sideways for example)
        pass

    def __set_graphic(self):
        # TODO: STL or Line/Box
        pass

    def __import_texture(self):
        # TODO (much later)
        pass


class RotationalJoint(DefaultJoint):
    """
    A rotational joint based off the default joint class

    :param connection_from_prev_seg: Origin point of the joint (Where it connects to the previous segment)
    :type connection_from_prev_seg: class:`vpython.vector`
    :param connection_to_next_seg: Tooltip point of the joint (Where it connects to the next segment)
    :type connection_to_next_seg: class:`vpython.vector`
    """
    # TODO
    #  1. Add input parameters to determine the rotation axis
    #  2. Update functions to rotate around the correct axis

    def __init__(self, connection_from_prev_seg, connection_to_next_seg):
        # Call super init function
        super().__init__(connection_from_prev_seg, connection_to_next_seg)

    def rotate_joint(self, new_angle):
        """
        Rotate the joint to a given angle in range [-pi pi] (radians)

        :param new_angle: The new angle in range [-pi pi] that the link is to be rotated to.
        :type new_angle: float (radians)
        """
        # Wrap given angle to -pi to pi
        new_angle = wrap_to_pi(new_angle)
        current_angle = self.arm_angle
        # Calculate amount to rotate the link
        angle_diff = wrap_to_pi(current_angle - new_angle)
        # Calculate the new vector representation the link will be at for the new angle
        new_vector = self.x_vector.rotate(angle=angle_diff, axis=self.y_vector)
        # Update the link
        self.update_orientation(new_vector)


class TranslationalJoint(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg):
        super().__init__(connection_from_prev_seg, connection_to_next_seg)
        self.min_translation = None
        self.max_translation = None

    def translate_joint(self, new_translation):
        # TODO calculate new connectTo point, update relevant super() params
        # TODO Update graphic
        pass


class StaticJoint(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg):
        super().__init__(connection_from_prev_seg, connection_to_next_seg)


class Gripper(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg):
        super().__init__(connection_from_prev_seg, connection_to_next_seg)


class Robot:
    # TODO:
    #  Have functions to update links,
    #  take in rotation, translation, etc, params
    def __init__(self, joints):
        pass
