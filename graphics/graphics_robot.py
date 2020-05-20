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
    :param direction_vector: Vector direction from the connection_from to the connection_to, defaults to +z (up)
    :type direction_vector: class:`vpython.vector`
    """
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, direction_vector):
        # Set connection points
        self.connect_from = connection_from_prev_seg
        self.connect_to = connection_to_next_seg
        # Calculate the length of the link
        self.length = mag(self.connect_to - self.connect_from)
        # Change the directional vector magnitude to match the length
        self.vector = direction_vector
        self.vector.mag = self.length
        # TODO
        #  self.x_rotation = None
        #  self.graphic_obj = None
        self.graphic_ref = None

    def update_position(self, new_pos):
        """
        Move the position of the link to the specified location

        :param new_pos: 3D vector representing the new location for the origin (connection_from) of the link
        :type new_pos: class:`vpython.vector`
        """
        # Calculate translational movement amount
        axes_movement = self.connect_from + new_pos
        # Update each position
        self.connect_from += axes_movement
        self.connect_to += axes_movement
        # If the reference frame is drawn, redraw it
        self.draw_reference_frame(self.graphic_ref.visible)

    def update_orientation(self, new_direction):
        """
        Orient the link to face the direction of the given vector (respective from the link origin (connect_from))

        :param new_direction: vector representation of the direction the link now represents
        :type new_direction: class:`vpython.vector`
        """
        # Set magnitude to reflect link length
        new_direction.mag = self.length
        # Set the new direction and connection end point (tool tip)
        self.vector = new_direction
        self.connect_to = self.connect_from + new_direction
        # If the reference frame is drawn, redraw it
        self.draw_reference_frame(self.graphic_ref.visible)

    def draw_reference_frame(self, is_visible):
        """
        Draw a reference frame at the tool point position
        :param is_visible: Whether the reference frame should be drawn or not
        :type is_visible: bool
        """
        # If not visible, turn off
        if not is_visible:
            # If a reference frame exists
            if self.graphic_ref is not None:
                self.graphic_ref.visible = False
        # Else: draw
        else:
            # If graphic does not currently exist
            if self.graphic_ref is None:
                # Create one
                self.graphic_ref = draw_reference_frame_axes(self.connect_to, self.vector, radians(0))
            # Else graphic does exist
            else:
                # Set previous to invisible
                self.graphic_ref.visible = False
                # Rewrite over it
                self.graphic_ref = draw_reference_frame_axes(self.connect_to, self.vector, radians(0))

    def __draw_graphic(self):
        # TODO
        pass

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
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, direction_vector=vector(0, 0, 1)):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, direction_vector)

    def rotate_joint(self, new_angle):
        # TODO calculate vector representation of new angle. call update orientation
        # Find angle relative to ground plane
        ground_reference_vector = self.vector - vector(0, 0, self.vector.z)
        # If vector is pointing negatively in the z-axis, subtract from 360deg
        if self.vector.z < 0:
            current_angle = radians(360) - ground_reference_vector.diff_angle(self.vector)
        else:
            current_angle = ground_reference_vector.diff_angle(self.vector)
        # Find difference in angles to rotate by that amount
        required_rotation = new_angle - current_angle
        # Rotate the vector
        rotation_axis = self.vector.cross(ground_reference_vector)
        new_vector = self.vector.rotate(angle=required_rotation, axis=rotation_axis)
        # Update graphics
        self.update_orientation(new_vector)


class TranslationalJoint(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, direction_vector=vector(0, 0, 1)):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, direction_vector)
        self.min_translation = None
        self.max_translation = None

    def translate_joint(self, new_translation):
        # TODO calculate new connectTo point, update relevant super() params
        # TODO Update graphic
        pass


class StaticJoint(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, direction_vector=vector(0, 0, 1)):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, direction_vector)


class Gripper(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, direction_vector=vector(0, 0, 1)):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, direction_vector)


class Robot:
    # TODO:
    #  Have functions to update links,
    #  take in rotation, translation, etc, params
    def __init__(self, joints):
        pass
