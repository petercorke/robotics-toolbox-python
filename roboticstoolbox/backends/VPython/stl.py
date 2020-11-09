#!/usr/bin/env python
"""
@author Micah Huth
"""

from vpython import vec, vertex, color, triangle, compound
from roboticstoolbox.backends.VPython.common_functions import \
    vector, x_axis_vector
from io import StringIO

try:
    from roboticstoolbox.tools.stdout_supress import pipes
except Exception:  # pragma nocover
    from contextlib import contextmanager

    @contextmanager
    def pipes(stdout=None, stderr=None):
        pass

_out = StringIO()
_err = StringIO()
try:
    with pipes(stdout=_out, stderr=_err):
        from stl import mesh
except Exception:  # pragma nocover
    from stl import mesh


def import_object_from_numpy_stl(filename, scene):  # pragma nocover
    """
    Import either an ASCII or BINARY file format of an STL file.
    The triangles will be combined into a single compound entity.

    :param filename: Path of the stl file to import.
    :type filename: `str`
    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: Compound object of a collection of triangles formed from an
        stl file.
    :rtype: class:`vpython.compound`
    """
    # filename = meshdata[0]
    # scale = meshdata[1]
    # origin = meshdata[2]
    # origin_pos = get_pose_pos(origin)
    # origin_x = get_pose_x_vec(origin)
    # origin_y = get_pose_y_vec(origin)
    # origin_z = get_pose_z_vec(origin)

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
    visual_mesh = compound(triangles, origin=vec(0, 0, 0), canvas=scene)
    # visual_mesh.length *= scale[0]
    # visual_mesh.height *= scale[1]
    # visual_mesh.width *= scale[2]

    return visual_mesh


def set_stl_origin(
        stl_obj, current_obj_origin,
        required_obj_origin, scene):  # pragma nocover
    """
    Move the object so the required origin is at (0, 0, 0). Then set the
    origin for the generated stl object. Origin can't be changed, so creating
    a compound of itself allows setting an origin location

    :param stl_obj: The generated stl object.
    :type stl_obj: class:`vpython.compound`
    :param current_obj_origin: Current coordinates of the origin of the model
    :type current_obj_origin: class:`vpython.vector`
    :param required_obj_origin: Required coordinates to place the origin
        at (0, 0, 0)
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
    return compound(
        [stl_obj], origin=vector(0, 0, 0), vector=x_axis_vector, canvas=scene)
