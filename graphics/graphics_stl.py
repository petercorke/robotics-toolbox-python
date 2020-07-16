from vpython import vec, vertex, color, triangle, compound
from graphics.common_functions import *
from stl import mesh


def import_object_from_stl(filename):
    """
    Import an stl object and convert it into a usable vpython object.
    Function not directly part of the vpython package, but can by found as part of vpython git repo.
    Code was based on it.
    https://github.com/vpython/vpython-jupyter/blob/master/convert_stl.zip

    :param filename: Name of the stl file to import (Exclude path and extension).
    :type filename: `str`
    :return: Compound object of a collection of triangles formed from an stl file.
    :rtype: class:`vpython.compound`

    .. deprecated::
        A new function using numpy import is available. It accepts both ASCII and BINARY formats.
    """
    raise DeprecationWarning("This function is outdated. Use import_object_from_numpy_stl")

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


def import_object_from_numpy_stl(filename, scene):
    """
    Import either an ASCII or BINARY file format of an STL file.
    The triangles will be combined into a single compound entity.

    :param filename: Path of the stl file to import.
    :type filename: `str`
    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Compound object of a collection of triangles formed from an stl file.
    :rtype: class:`vpython.compound`
    """
    # Load the mesh using NumPy-STL
    the_mesh = mesh.Mesh.from_file(filename)

    num_faces = len(the_mesh.vectors)
    triangles = []

    # For every face in the model
    for face in range(0, num_faces):
        # Get the (3) 3D points
        point0 = the_mesh.vectors[face][0]
        point1 = the_mesh.vectors[face][1]
        point2 = the_mesh.vectors[face][2]

        # Get the normal direction for the face
        normal0 = the_mesh.normals[face][0]
        normal1 = the_mesh.normals[face][1]
        normal2 = the_mesh.normals[face][2]
        normal = vec(normal0, normal1, normal2)

        # Create the VPython 3D points
        vertex0 = vertex(
            pos=vec(point0[0], point0[1], point0[2]),
            normal=normal,
            color=color.white
        )
        vertex1 = vertex(
            pos=vec(point1[0], point1[1], point1[2]),
            normal=normal,
            color=color.white
        )
        vertex2 = vertex(
            pos=vec(point2[0], point2[1], point2[2]),
            normal=normal,
            color=color.white
        )

        # Combine them in a list
        vertices = [vertex0, vertex1, vertex2]

        # Create a triangle using the points, and add it to the list
        triangles.append(triangle(canvas=scene, vs=vertices))

    # Return a compound of the triangles
    visual_mesh = compound(triangles, origin=vector(0, 0, 0), canvas=scene)
    return visual_mesh


def set_stl_origin(stl_obj, current_obj_origin, required_obj_origin, scene):
    """
    Move the object so the required origin is at (0, 0, 0). Then set the origin for the generated stl object.
    Origin can't be changed, so creating a compound of itself allows setting an origin location

    :param stl_obj: The generated stl object.
    :type stl_obj: class:`vpython.compound`
    :param current_obj_origin: Current coordinates of the origin of the model
    :type current_obj_origin: class:`vpython.vector`
    :param required_obj_origin: Required coordinates to place the origin at (0, 0, 0)
    :type required_obj_origin: class:`vpython.vector`
    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Compound object of itself, with origin set respective to the joint
    :rtype: class:`vpython.compound`
    """
    # Move the object to put the origin at 0, 0, 0
    movement = required_obj_origin - current_obj_origin
    stl_obj.pos += movement

    # Set invisible to return an overwritten copy
    stl_obj.visible = False

    # Return a compound of itself with the origin at (0, 0, 0)
    return compound([stl_obj], origin=vector(0, 0, 0), vector=x_axis_vector, canvas=scene)
