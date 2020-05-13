from vpython import *


def import_object_from_stl(filename):
    """
    https://github.com/vpython/vpython-jupyter/blob/master/convert_stl.zip
    """
    filepath = './graphics/models/' + filename + '.stl'
    stl_file = open(filepath, mode='rb')
    stl_file.seek(0)
    stl_text = stl_file.readlines()

    triangles = []
    vertices = []

    for line in stl_text:
        file_line = line.split()
        if file_line[0] == b'facet':
            N = vec(
                float(file_line[2]),
                float(file_line[3]),
                float(file_line[4])
            )
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
