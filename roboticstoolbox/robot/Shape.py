#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from spatialmath import SE3
from spatialmath.base.argcheck import getvector
from spatialmath.base import r2q
import numpy as np

try:
    import fcl
    _fcl = True
except ImportError:
    _fcl = False


class Shape(object):

    def __init__(
            self,
            primitive,
            base=None,
            radius=0,
            length=0,
            scale=[1, 1, 1],
            filename=None,
            co=None,
            stype=None):

        self.co = co
        self.base = base
        self.wT = None
        self.primitive = primitive
        self.scale = scale
        self.radius = radius
        self.length = length
        self.filename = filename
        self.stype = stype

    def to_dict(self):

        if self.stype == 'cylinder':
            fk = self.wT * SE3.Rx(np.pi/2)
        else:
            fk = self.wT

        shape = {
            'stype': self.stype,
            'scale': self.scale.tolist(),
            'filename': self.filename,
            'radius': self.radius,
            'length': self.length,
            't': fk.t.tolist(),
            'q': r2q(fk.R).tolist()
        }

        return shape

    def fk_dict(self):

        if self.stype == 'cylinder':
            fk = self.wT * SE3.Rx(np.pi/2)
        else:
            fk = self.wT

        shape = {
            't': fk.t.tolist(),
            'q': r2q(fk.R).tolist()
        }

        return shape

    def __repr__(self):
        return f'{self.stype},\n{self.base}'

    @property
    def wT(self):
        return self._wT

    @wT.setter
    def wT(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._wT = T * self.base

        if _fcl and self.co is not None:
            tf = fcl.Transform(self._wT.R, self._wT.t)
            self.co.setTransform(tf)

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
        else:
            value = getvector([1, 1, 1], 3)
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

        if _fcl:
            obj = fcl.Box(scale[0], scale[1], scale[2])
            co = fcl.CollisionObject(obj, fcl.Transform())
        else:
            co = None

        return cls(
            True, base=base, scale=scale, co=co, stype='box')

    @classmethod
    def Cylinder(cls, radius, length, base=None):

        if _fcl:
            obj = fcl.Cylinder(radius, length)
            co = fcl.CollisionObject(obj, fcl.Transform())
        else:
            co = None

        return cls(
            True, base=base, radius=radius, length=length, co=co,
            stype='cylinder')

    @classmethod
    def Sphere(cls, radius, base=None):

        if _fcl:
            obj = fcl.Sphere(radius)
            co = fcl.CollisionObject(obj, fcl.Transform())
        else:
            co = None

        return cls(True, base=base, radius=radius, co=co, stype='sphere')

    @classmethod
    def Mesh(cls, filename, base=None, scale=None):
        return cls(
            False, filename=filename, base=base, scale=scale, stype='mesh')


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
