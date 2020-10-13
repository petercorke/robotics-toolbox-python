#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from spatialmath import SE3
from spatialmath.base.argcheck import getvector
from spatialmath.base import r2q
import numpy as np
import os

try:
    import fcl
    _fcl = True
except ImportError:
    _fcl = False

try:
    import pybullet as p
    cid = p.connect(p.SHARED_MEMORY)
    if (cid < 0):
        p.connect(p.DIRECT)
    import trimesh
    _trimesh = True
except ImportError:
    _trimesh = False


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

        self._wT = SE3()
        self.co = co
        self.base = base
        self.wT = None
        self.primitive = primitive
        self.scale = scale
        self.radius = radius
        self.length = length
        self.filename = filename
        self.stype = stype
        self.v = np.zeros(6)

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
            'q': r2q(fk.R).tolist(),
            'v': self.v.tolist()
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

    def _update_fcl(self):
        if _fcl and self.co is not None:
            q = r2q(self.wT.R)
            rot = [q[1], q[2], q[3], q[0]]
            p.resetBasePositionAndOrientation(self.co, self.wT.t, rot)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = getvector(value, 6)

    @property
    def wT(self):
        return self._wT * self.base

    @wT.setter
    def wT(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._wT = T
        self._update_fcl()

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
        self._update_fcl()

    @classmethod
    def Box(cls, scale, base=None):

        if base is None:
            base = SE3()

        if _fcl:
            col = p.createCollisionShape(
                shapeType=p.GEOM_BOX, halfExtents=np.array(scale)/2)

            co = p.createMultiBody(
                baseMass=1,
                baseInertialFramePosition=[0, 0, 0],
                baseCollisionShapeIndex=col)
        else:
            co = None

        return cls(
            True, base=base, scale=scale, co=co, stype='box')

    @classmethod
    def Cylinder(cls, radius, length, base=None):

        if base is None:
            base = SE3()

        if _fcl:
            col = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER, radius=radius, height=length)

            co = p.createMultiBody(
                baseMass=1,
                baseInertialFramePosition=[0, 0, 0],
                baseCollisionShapeIndex=col)
        else:
            co = None

        return cls(
            True, base=base, radius=radius, length=length, co=co,
            stype='cylinder')

    @classmethod
    def Sphere(cls, radius, base=None):

        if base is None:
            base = SE3()

        if _fcl:
            col = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE, radius=radius)

            co = p.createMultiBody(
                baseMass=1,
                baseInertialFramePosition=[0, 0, 0],
                baseCollisionShapeIndex=col)
        else:
            co = None

        return cls(True, base=base, radius=radius, co=co, stype='sphere')

    @classmethod
    def Mesh(cls, filename, base=None, scale=[1, 1, 1]):

        if base is None:
            base = SE3()

        name, file_extension = os.path.splitext(filename)

        if _fcl and _trimesh and \
                (file_extension == '.stl' or file_extension == '.STL'):

            col = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=filename,
                meshScale=scale)

            co = p.createMultiBody(
                baseMass=1,
                baseInertialFramePosition=[0, 0, 0],
                baseCollisionShapeIndex=col)

        else:
            co = None

        return cls(
            False, filename=filename, base=base, co=co,
            scale=scale, stype='mesh')


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
