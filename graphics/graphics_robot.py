from graphics.graphics_canvas import *
from graphics.graphics_stl import *


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
        self.y_vector = self.__graphic_ref.up
        self.y_vector.mag = self.__length
        self.z_vector = self.x_vector.cross(self.y_vector)
        self.z_vector.mag = self.__length
        # TODO self.x_rotation = None
        # TODO using default box for now
        box_midpoint = vector(
            (self.__connect_from.x + self.__connect_to.x) / 2,
            (self.__connect_from.y + self.__connect_to.y) / 2,
            (self.__connect_from.z + self.__connect_to.z) / 2
        )
        self.__graphic_obj = box(pos=box_midpoint, axis=self.x_vector, size=vector(self.__length, 0.1, 0.1))

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

    def __update_reference_frame(self):
        self.x_vector = self.__connect_to - self.__connect_from
        self.x_vector.mag = self.__length
        self.y_vector = self.__graphic_ref.up
        self.y_vector.mag = self.__length
        self.z_vector = self.x_vector.cross(self.y_vector)
        self.z_vector.mag = self.__length

    def draw_reference_frame(self, is_visible):
        """
        Draw a reference frame at the tool point position
        :param is_visible: Whether the reference frame should be drawn or not
        :type is_visible: bool
        """
        # TODO
        #  Instead of replacing the object, move and rotate it and set visibility on
        # If not visible, turn off
        if not is_visible:
            # If a reference frame exists
            if self.__graphic_ref is not None:
                self.__graphic_ref.visible = False
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
        box_midpoint = vector(
            (self.__connect_from.x + self.__connect_to.x) / 2,
            (self.__connect_from.y + self.__connect_to.y) / 2,
            (self.__connect_from.z + self.__connect_to.z) / 2
        )
        self.__graphic_obj.pos = box_midpoint
        self.__graphic_obj.axis = self.x_vector
        self.__graphic_obj.size = vector(self.__length, 0.1, 0.1)

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
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg):
        super().__init__(connection_from_prev_seg, connection_to_next_seg)

    def rotate_joint(self, angle_amount):
        new_vector = self.x_vector.rotate(angle=angle_amount, axis=self.y_vector)
        self.update_orientation(new_vector)
        """
        # Find angle relative to ground plane
        ground_reference_vector = vector(self.vector.x, self.vector.y, 0)
        # If vector is pointing negatively in the z-axis, subtract from 360deg
        if self.vector.z < 0:
            current_angle = radians(360) - ground_reference_vector.diff_angle(self.vector)
        else:
            current_angle = ground_reference_vector.diff_angle(self.vector)
        # Find difference in angles to rotate by that amount
        required_rotation = new_angle - current_angle
        # Rotate the vector
        rotation_axis = ground_reference_vector.cross(self.vector)
        new_vector = self.vector.rotate(angle=required_rotation, axis=rotation_axis)
        # Update graphics
        self.update_orientation(new_vector)
        """


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
