#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from spatialmath import SE3
from spatialmath.base.argcheck import getvector


class Shape(object):

    def __init__(
            self,
            primitive,
            base=None,
            radius=0,
            length=0,
            scale=[1, 1, 1],
            filename=None):

        self.base = base
        self.primitive = primitive
        self.scale = scale
        self.filename = filename

    @property
    def base(self):
        return self._base

    @property
    def primitive(self):
        return self._primitive

    @primitive.setter
    def primitive(self, value):
        self._primitive = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not None:
            value = getvector(value, 3)
        self._scale = value

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

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

    @base.setter
    def base(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._base = T

    @classmethod
    def Box(cls, scale, base=None):
        return cls(True, base=base, scale=scale)

    @classmethod
    def Cylinder(cls, radius, length, base=None):
        return cls(True, base=base, radius=radius, length=length)

    @classmethod
    def Sphere(cls, radius, base=None):
        return cls(True, base=base, radius=radius)

    @classmethod
    def Mesh(cls, filename, base=None, scale=None):
        return cls(False, filename=filename, base=base, scale=scale)


# class Mesh(Shape):
#     """
#     A mesh object.

#     :param filename: The path to the mesh that contains this object.
#         This is the absolute path.
#     :type filename: str
#     :param scale: The scaling value for the mesh along the XYZ axes. If
#         ``None``, assumes no scale is applied.
#     :type scale: list (3) float, optional

#     """

#     def __init__(self, filename, base=None, scale=None):
#         super(Box, self).__init__(
#             False, filename=filename, base=base, scale=scale)


# class Cylinder(Shape):
#     """A cylinder whose center is at the local origin.
#     Parameters
#     ----------
#     :param radius: The radius of the cylinder in meters.
#     :type radius: float
#         The radius of the cylinder in meters.
#     :param length: The length of the cylinder in meters.
#     :type length: float

#     """

#     def __init__(self, radius, length, base=None):
#         super(Box, self).__init__(

#             True, base=base, radius=radius, length=length)


# class Sphere(Shape):
#     """
#     A sphere whose center is at the local origin.

#     :param radius: The radius of the sphere in meters.
#     :type radius: float

#     """

#     def __init__(self, radius, base=None):
#         super(Box, self).__init__(True, base=base, radius=radius)


# class Box(Shape):
#     """
#     A rectangular prism whose center is at the local origin.

#     :param scale: The length, width, and height of the box in meters.
#     :type scale: list (3) float

#     """

#     def __init__(self, scale, base=None):

#         super(Box, self).__init__(True, base=base, scale=scale)
