from vpython import *


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


class Joint:
    def __init__(self):
        self.position = None
        self.vector = None
        self.rotation = None
        self.connectFrom = None
        self.connectTo = None
        self.toolpoint = None

    def update_position(self, new_pos):
        pass

    def update_orientation(self, new_direction, new_angle):
        pass


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


def set_stl_origin(stl_obj, object_origin, required_obj_origin):
    # Z axis movement
    required_obj_z_origin = required_obj_origin.z
    current_obj_z_origin = object_origin.z
    z_movement = required_obj_z_origin - current_obj_z_origin

    stl_obj.pos.z += z_movement

    return


def import_puma_560():
    pass
