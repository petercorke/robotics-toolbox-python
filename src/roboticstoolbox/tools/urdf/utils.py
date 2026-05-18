"""Utilities for URDF parsing.
"""
import numpy as np
import spatialmath as sm


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
    matrix = sm.SE3()
    rpy = np.array([0, 0, 0])
    origin_node = node.find('origin')
    if origin_node is not None:
        if 'xyz' in origin_node.attrib:
            t = np.fromstring(origin_node.attrib['xyz'], sep=' ')
            matrix = sm.SE3(t)
        if 'rpy' in origin_node.attrib:
            rpy = np.fromstring(origin_node.attrib['rpy'], sep=' ')
            matrix.A[:3, :3] = sm.SE3.RPY(rpy).R

    return matrix.A, rpy


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
