from vpython import *


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
    normal = None

    # For every line in the file
    for line in stl_text:
        file_line = line.split()
        # If blank line (skip)
        if not file_line:
            pass
        # If a face
        elif file_line[0] == b'facet':
            normal = vec(
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
                    normal=normal,
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
    # TODO add axis in here (1, 0, 0). Then update calculate_arm_angle with correct params for XY plane angle
    return compound([stl_obj], origin=vector(0, 0, 0), vector=(1, 0, 0))
