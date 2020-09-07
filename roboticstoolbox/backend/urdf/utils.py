"""Utilities for URDF parsing.
"""
import numpy as np


def rpy_to_matrix(coords):
    """Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.
    The roll-pitch-yaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.
    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    Parameters
    ----------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    Returns
    -------
    R : (3,3) float
        The corresponding homogenous 3x3 rotation matrix.
    """
    coords = np.asanyarray(coords, dtype=np.float64)
    c3, c2, c1 = np.cos(coords)
    s3, s2, s1 = np.sin(coords)

    return np.array([
        [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
        [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
        [-s2, c2 * s3, c2 * c3]
    ], dtype=np.float64)


def parse_origin(node):
    """Find the ``origin`` subelement of an XML node and convert it
    into a 4x4 homogenous transformation matrix.
    Parameters
    ----------
    node : :class`lxml.etree.Element`
        An XML node which (optionally) has a child node with the ``origin``
        tag.
    Returns
    -------
    matrix : (4,4) float
        The 4x4 homogneous transform matrix that corresponds to this node's
        ``origin`` child. Defaults to the identity matrix if no ``origin``
        child was found.
    """
    matrix = np.eye(4, dtype=np.float64)
    origin_node = node.find('origin')
    if origin_node is not None:
        if 'xyz' in origin_node.attrib:
            matrix[:3, 3] = np.fromstring(origin_node.attrib['xyz'], sep=' ')
        if 'rpy' in origin_node.attrib:
            rpy = np.fromstring(origin_node.attrib['rpy'], sep=' ')
            matrix[:3, :3] = rpy_to_matrix(rpy)
    return matrix


# def get_filename(base_path, file_path, makedirs=False):
#     """Formats a file path correctly for URDF loading.
#     Parameters
#     ----------
#     base_path : str
#         The base path to the URDF's folder.
#     file_path : str
#         The path to the file.
#     makedirs : bool, optional
#         If ``True``, the directories leading to the file will be created
#         if needed.
#     Returns
#     -------
#     resolved : str
#         The resolved filepath -- just the normal ``file_path`` if it was an
#         absolute path, otherwise that path joined to ``base_path``.
#     """
#     fn = file_path
#     if not os.path.isabs(file_path):
#         fn = os.path.join(base_path, file_path)
#     if makedirs:
#         d, _ = os.path.split(fn)
#         if not os.path.exists(d):
#             os.makedirs(d)
#     return fn


def configure_origin(value):
    """Convert a value into a 4x4 transform matrix.
    Parameters
    ----------
    value : None, (6,) float, or (4,4) float
        The value to turn into the matrix.
        If (6,), interpreted as xyzrpy coordinates.
    Returns
    -------
    matrix : (4,4) float or None
        The created matrix.
    """
    if value is None:
        value = np.eye(4, dtype=np.float64)
    return value
