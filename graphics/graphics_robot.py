from vpython import *


def import_object_from_stl(filename):
    """
    Import an stl object and convert it into a usable vpython object.
    Function not directly par tof the vpython package, but can by found as part of vpython git repo.
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
