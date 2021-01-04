#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from spatialmath import SE3
from spatialmath.base.argcheck import getvector
from spatialmath.base import r2q
import numpy as np
from functools import wraps
from io import StringIO
import os

_mpl = False

try:
    from matplotlib import colors as mpc
    _mpl = True
except ImportError:    # pragma nocover
    pass

p = None
_pyb = None


def _import_pyb():
    import importlib
    global _pyb
    global p

    try:
        from roboticstoolbox.tools.stdout_supress import pipes
    except Exception:  # pragma nocover
        from contextlib import contextmanager

        @contextmanager
        def pipes(stdout=None, stderr=None):
            pass

    try:
        out = StringIO()
        try:
            with pipes(stdout=out, stderr=None):
                p = importlib.import_module('pybullet')
        except Exception:  # pragma nocover
            p = importlib.import_module('pybullet')

        cid = p.connect(p.SHARED_MEMORY)
        if (cid < 0):
            p.connect(p.DIRECT)
        _pyb = True
    except ImportError:   # pragma nocover
        _pyb = False


class Shape(object):

    def __init__(
            self,
            base=None,
            radius=0,
            length=0,
            scale=[1, 1, 1],
            color=None,
            filename=None,
            stype=None):

        self._wT = SE3()
        self.co = None
        self.base = base
        self.scale = scale
        self.radius = radius
        self.length = length
        self.filename = filename
        self.stype = stype
        self.v = np.zeros(6)
        self.color = color

        self.pinit = False

    def to_dict(self):
        '''
        to_dict() returns the shapes information in dictionary form

        :returns: All information about the shape
        :rtype: dict
        '''

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
            'v': self.v.tolist(),
            'color': list(self.color)
        }

        return shape

    def fk_dict(self):
        '''
        fk_dict() outputs shapes pose in dictionary form

        :returns: The shape pose in translation and quternion form
        :rtype: dict
        '''

        if self.stype == 'cylinder':
            fk = self.wT * SE3.Rx(np.pi/2)
        else:
            fk = self.wT

        shape = {
            't': fk.t.tolist(),
            'q': r2q(fk.R).tolist()
        }

        return shape

    def __repr__(self):   # pragma nocover
        return f'{self.stype},\n{self.base}'

    def _update_pyb(self):
        if _pyb and self.co is not None:
            q = r2q(self.wT.R)
            rot = [q[1], q[2], q[3], q[0]]
            p.resetBasePositionAndOrientation(self.co, self.wT.t, rot)

    def _init_pob(self):   # pragma nocover
        pass

    def _check_pyb(func):   # pragma nocover
        @wraps(func)
        def wrapper_check_pyb(*args, **kwargs):
            if _pyb is None:
                _import_pyb()
            return func(*args, **kwargs)
        return wrapper_check_pyb

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = getvector(value, 6)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):

        default_color = (0.95, 0.5, 0.25, 1.0)

        if isinstance(value, tuple):
            value = list(value)

        if isinstance(value, list):
            if len(value) == 3:
                value.append(1.0)
                value = tuple(value)
        elif isinstance(value, str):
            if _mpl:
                try:
                    value = mpc.to_rgba(value)
                except ValueError:
                    print(f'{value} is an invalid color name, using default color')
                    value = default_color
            else:  # pragma nocover
                value = default_color
                print(
                    'Color only supported when matplotlib is installed\n'
                    'Install using: pip install matplotlib')
        elif value is None:
            value = default_color

        self._color = value

    @property
    def wT(self):
        return self._wT * self.base

    @wT.setter
    def wT(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._wT = T
        self._update_pyb()

    @property
    def base(self):
        return self._base

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
        self._update_pyb()

    @_check_pyb
    def closest_point(self, shape, inf_dist=1.0):
        '''
        closest_point(shape, inf_dist) returns the minimum euclidean
        distance between self and shape, provided it is less than inf_dist.
        It will also return the points on self and shape in the world frame
        which connect the line of length distance between the shapes. If the
        distance is negative then the shapes are collided.

        :param shape: The shape to compare distance to
        :type shape: Shape
        :param inf_dist: The minimum distance within which to consider
            the shape
        :type inf_dist: float
        :returns: d, p1, p2 where d is the distance between the shapes,
            p1 and p2 are the points in the world frame on the respective
            shapes
        :rtype: float, SE3, SE3
        '''

        if not self.pinit:
            self._init_pob()
            self._update_pyb()

        if not shape.pinit:
            shape._init_pob()
            shape._update_pyb()

        if not _pyb:  # pragma nocover
            raise ImportError(
                'The package PyBullet is required for collision '
                'functionality. Install using pip install pybullet')

        ret = p.getClosestPoints(self.co, shape.co, inf_dist)

        if len(ret) == 0:
            d = None
            p1 = None
            p2 = None
        else:
            ret = ret[0]
            d = ret[8]
            p1 = SE3(ret[5])
            p2 = SE3(ret[6])

        return d, p1, p2

    def collided(self, shape):
        '''
        collided(shape) checks if self and shape have collided

        :param shape: The shape to compare distance to
        :type shape: Shape
        :returns: True if shapes have collided
        :rtype: bool
        '''

        d, _, _ = self.closest_point(shape)

        if d is not None and d <= 0:
            return True
        else:
            return False


class Mesh(Shape):
    """
    A mesh object described by an stl or collada file.

    :param filename: The path to the mesh that contains this object.
        This is the absolute path.
    :type filename: str
    :param scale: The scaling value for the mesh along the XYZ axes. If
        ``None``, assumes no scale is applied.
    :type scale: list (3) float, optional
    :param base: Local reference frame of the shape
    :type base: SE3

    """

    def __init__(self, filename=None, scale=[1, 1, 1], base=None):
        super(Mesh, self).__init__(
            filename=filename, base=base,
            scale=scale, stype='mesh')

    def _init_pob(self):
        name, file_extension = os.path.splitext(self.filename)
        if (file_extension == '.stl' or file_extension == '.STL'):

            col = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=self.filename,
                meshScale=self.scale)

            self.co = p.createMultiBody(
                baseMass=1,
                baseInertialFramePosition=[0, 0, 0],
                baseCollisionShapeIndex=col)

            self.pinit = True


class Cylinder(Shape):
    """A cylinder whose center is at the local origin.
    Parameters

    :param radius: The radius of the cylinder in meters.
    :type radius: float
    :param length: The length of the cylinder in meters.
    :type length: float
    :param base: Local reference frame of the shape
    :type base: SE3

    """

    def __init__(self, radius, length, base=None):
        super(Cylinder, self).__init__(
            base=base, radius=radius, length=length,
            stype='cylinder')

    def _init_pob(self):
        col = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius, height=self.length)

        self.co = p.createMultiBody(
            baseMass=1,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=col)

        self.pinit = True


class Sphere(Shape):
    """
    A sphere whose center is at the local origin.

    :param radius: The radius of the sphere in meters.
    :type radius: float
    :param base: Local reference frame of the shape
    :type base: SE3

    """

    def __init__(self, radius, base=None):
        super(Sphere, self).__init__(
            base=base, radius=radius, stype='sphere')

    def _init_pob(self):
        col = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE, radius=self.radius)

        self.co = p.createMultiBody(
            baseMass=1,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=col)

        self.pinit = True


class Box(Shape):
    """
    A rectangular prism whose center is at the local origin.

    :param scale: The length, width, and height of the box in meters.
    :type scale: list (3) float
    :param base: Local reference frame of the shape
    :type base: SE3

    """

    def __init__(self, scale, base=None):
        super(Box, self).__init__(base=base, scale=scale, stype='box')

    def _init_pob(self):

        col = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=np.array(self.scale)/2)

        self.co = p.createMultiBody(
            baseMass=1,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=col)

        self.pinit = True
