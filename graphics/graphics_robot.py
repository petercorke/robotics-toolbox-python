from vpython import *
from numpy import sign

# TODO:
#  1. Create a link class (for each joint type (rotate/translate) that contains:
#       a. Visual object
#           - Will contain link pos, vector, rotation, etc
#       b. Positional variables (where it connects from, and to, reference frame local, etc)
#       c. Update position methods
#           - World axis are different to that of regulation (x=forward, z=up). Need to adjust for this
#  2. Create a robot class that contains:
#       a. Collection of links
#       b. Update position methods
#  3. Texture import


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
        self.connectFrom = connection_from_prev_seg
        self.connectTo = connection_to_next_seg
        # Calculate the length of the link
        self.length = mag(self.connectTo - self.connectFrom)
        # Change the directional vector magnitude to match the length
        self.vector = direction_vector
        self.vector.mag = self.length

    def update_position(self, new_pos):
        """
        Move the position of the link to the specified location

        :param new_pos: 3D vector representing the new location for the origin (connection_from) of the link
        :type new_pos: class:`vpython.vector`
        """
        # Calculate translational movement amount
        axes_movement = self.connectFrom + new_pos
        # Update each position
        self.connectFrom += axes_movement
        self.connectTo += axes_movement

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
        self.connectTo = self.connectFrom + new_direction


class RotationalJoint(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, direction_vector=vector(0, 0, 1)):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, direction_vector)


class TranslationalJoint(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, direction_vector=vector(0, 0, 1)):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, direction_vector)


class StaticJoint(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, direction_vector=vector(0, 0, 1)):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, direction_vector)


class Gripper(DefaultJoint):
    # TODO
    def __init__(self, connection_from_prev_seg, connection_to_next_seg, direction_vector=vector(0, 0, 1)):
        super().__init__(connection_from_prev_seg, connection_to_next_seg, direction_vector)


class Robot:
    def __init__(self, joints):
        pass


def import_object_from_stl(filename):
    """
    Import an stl object and convert it into a usable vpython object.
    Function not directly part of the vpython package, but can by found as part of vpython git repo.
    Code was based on it.
    https://github.com/vpython/vpython-jupyter/blob/master/convert_stl.zip

    :param filename: Name of the stl file to import (Exclude path and extension).
    :type filename: str
    :return: Compound object of a collection of triangles formed from an stl file.
    :rtype: class:`vpython.compound`
    """
    # TODO: put error handling in case binary stl file used instead of ascii

    # TODO: put error handling in case of bad file

    # Open the file
    filepath = './graphics/models/' + filename + '.stl'
    stl_file = open(filepath, mode='rb')
    stl_file.seek(0)
    stl_text = stl_file.readlines()

    # Initial Conditions
    triangles = []
    vertices = []

    # For every line in the file
    for line in stl_text:
        file_line = line.split()
        # If blank line (skip)
        if not file_line:
            pass
        # If a face
        elif file_line[0] == b'facet':
            N = vec(
                float(file_line[2]),
                float(file_line[3]),
                float(file_line[4])
            )
        # If a vertex
        elif file_line[0] == b'vertex':
            vertices.append(
                vertex(
                    pos=vec(
                        float(file_line[1]),
                        float(file_line[2]),
                        float(file_line[3])
                    ),
                    normal=N,
                    color=color.white
                )
            )
            if len(vertices) == 3:
                triangles.append(triangle(vs=vertices))
                vertices = []

    return compound(triangles)


def set_stl_origin(stl_obj, current_obj_origin, required_obj_origin):
    """
    Move the object so the required origin is at (0, 0, 0). Then set the origin for the generated stl object.
    Origin can't be changed, so creating a compound of itself allows setting an origin location
    :param stl_obj: The generated stl object.
    :type stl_obj: class:`vpython.compound`
    :param current_obj_origin: Current coordinates of the origin of the model
    :type current_obj_origin: class:`vpython.vector`
    :param required_obj_origin: Required coordinates to place the origin at (0, 0, 0)
    :type required_obj_origin: class:`vpython.vector`
    :return: Compound object of itself, with origin set respective to the joint
    :rtype: class:`vpython.compound`
    """
    # Move the object to put the origin at 0, 0, 0
    movement = required_obj_origin - current_obj_origin
    stl_obj.pos += movement

    # Set invisible to return an overwritten copy
    stl_obj.visible = False

    # Return a compound of itself with the origin at (0, 0, 0)
    return compound([stl_obj], origin=vector(0, 0, 0))


def import_puma_560():
    pass
