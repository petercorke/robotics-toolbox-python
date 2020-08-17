#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

import numpy as np
from spatialmath import SE3
from spatialmath.base.argcheck import getvector, verifymatrix, isscalar
import ropy as rp


class Shape(object):

    def __init__(self):

        primitive = True



class Mesh(Shape):
    """
    A triangular mesh object.

    :param filename: The path to the mesh that contains this object.
        This is the absolute path.
    :type filename: str
        
    :param scale: The scaling value for the mesh along the XYZ axes. If
        ``None``, assumes no scale is applied.
    :type scale: list (3) float, optional

    """

    def __init__(self, filename, scale=None):
        self.filename = filename
        self.scale = scale

    @property
    def filename(self):
        """str : The path to the mesh file for this object.
        """
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    @property
    def scale(self):
        """(3,) float : A scaling for the mesh along its local XYZ axes.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not None:
            value = np.asanyarray(value).astype(np.float64)
        self._scale = value


class Cylinder(Shape):
    """A cylinder whose center is at the local origin.
    Parameters
    ----------
    :param radius: The radius of the cylinder in meters.
    :type radius: float
        The radius of the cylinder in meters.
    :param length: The length of the cylinder in meters.
    :type length: float

    """

    def __init__(self, radius, length):
        self.radius = radius
        self.length = length

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = float(value)


class Sphere(Shape):
    """
    A sphere whose center is at the local origin.

    :param radius: The radius of the sphere in meters.
    :type radius: float

    """

    def __init__(self, radius):
        self.radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)
