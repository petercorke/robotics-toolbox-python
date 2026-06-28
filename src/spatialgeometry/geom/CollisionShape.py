#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

import sys
import os
import numpy as np
from spatialmath.base.argcheck import getvector
from spatialgeometry.geom import Shape
from spatialgeometry.geom.Shape import update
from warnings import warn

from typing import Tuple, Union

# Module-level coal reference — populated on first use, never in Pyodide.
_coal = None


def _require_coal():
    """Import coal on first use; raise clearly if unavailable."""
    global _coal
    if _coal is not None:
        return
    if sys.platform == "emscripten":
        raise RuntimeError(
            "Collision detection is not available in the browser (Pyodide) "
            "environment. Use a native Python installation for collision checking."
        )
    try:
        import coal as _c
        _coal = _c
    except ImportError:
        raise ImportError(
            "The 'coal' package is required for collision functionality. "
            "Install with:  pip install coal"
        )


class CollisionShape(Shape):
    def __init__(self, collision=True, **kwargs):
        self.co = None      # coal.CollisionObject, created on first use
        self._cinit = False
        super().__init__(**kwargs)
        self._collision = collision

    def _update_coal(self):
        """Push the current world transform into the Coal collision object."""
        if self.co is not None:
            self.co.setTranslation(self._wT[:3, 3])
            self.co.setRotation(self._wT[:3, :3])

    def _init_coal(self):  # overridden by each subclass
        pass

    def _ensure_coal(self):
        """Guarantee Coal is loaded and this object's Coal twin is current."""
        _require_coal()
        if not self._cinit:
            self._init_coal()
        self._update_coal()

    def closest_point(
        self, shape: "CollisionShape", inf_dist: float = 1.0
    ) -> Tuple[Union[float, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
        """
        Return the minimum euclidean distance between self and shape.

        :param shape: The shape to compare distance to
        :param inf_dist: Only return a result when distance < inf_dist
        :returns: (d, p1, p2) — distance and closest points in world frame,
            or (None, None, None) when the shapes are farther than inf_dist.
            d is negative when the shapes are penetrating.
        """
        self._ensure_coal()
        shape._ensure_coal()

        req = _coal.DistanceRequest()
        req.enable_signed_distance = True
        res = _coal.DistanceResult()
        _coal.distance(self.co, shape.co, req, res)

        d = res.min_distance
        if d > inf_dist:
            return None, None, None
        return d, np.array(res.getNearestPoint1()), np.array(res.getNearestPoint2())

    def iscollided(self, shape: "CollisionShape") -> bool:
        """
        Return True if self and shape have collided (distance ≤ 0).

        :param shape: The shape to check against
        """
        d, _, _ = self.closest_point(shape)
        return d is not None and d <= 0

    def collided(self, shape: "CollisionShape") -> bool:
        """Deprecated — use iscollided instead."""
        warn("collided is deprecated, use iscollided instead", FutureWarning)
        return self.iscollided(shape)


class Mesh(CollisionShape):
    """
    A mesh object described by an STL, OBJ, or DAE file.

    :param filename: Absolute path to the mesh file.
    :param scale: Scale factors along XYZ axes (default [1, 1, 1]).
    :param collision: Whether this shape participates in collision checking.
    """

    def __init__(self, filename=None, scale=[1, 1, 1], **kwargs):
        super().__init__(stype="mesh", **kwargs)
        self.filename = filename
        self.scale = scale

    def _init_coal(self):
        if not self.collision:
            raise ValueError(
                "This shape has collision=False and cannot be used as a collision object"
            )
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "The 'trimesh' package is required for mesh collision objects. "
                "Install with:  pip install trimesh"
            )

        mesh = trimesh.load(self.filename, force="mesh")
        vertices = (mesh.vertices * self.scale).astype(np.float64, order="C")
        triangles = mesh.faces.astype(np.int64, order="C")

        bvh = _coal.BVHModelOBBRSS()
        bvh.beginModel(len(triangles), len(vertices))
        bvh.addVertices(vertices)
        bvh.addTriangles(triangles)
        bvh.endModel()

        self.co = _coal.CollisionObject(bvh)
        self._cinit = True

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @scale.setter
    @update
    def scale(self, value):
        value = getvector(value if value is not None else [1, 1, 1], 3)
        self._scale = np.array(value)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    @update
    def filename(self, value):
        self._filename = value

    def to_dict(self):
        shape = super().to_dict()
        shape["filename"] = self.filename
        shape["scale"] = self.scale.tolist()
        return shape


class Cylinder(CollisionShape):
    """
    A cylinder whose centre is at the local origin, axis along Z.

    :param radius: Radius in metres.
    :param length: Total length in metres.
    :param collision: Whether this shape participates in collision checking.
    """

    def __init__(self, radius, length, **kwargs):
        super().__init__(stype="cylinder", **kwargs)
        self.radius = radius
        self.length = length

    def _init_coal(self):
        if not self.collision:
            raise ValueError(
                "This shape has collision=False and cannot be used as a collision object"
            )
        # Coal Cylinder(radius, halfLength)
        geom = _coal.Cylinder(self.radius, self.length / 2.0)
        self.co = _coal.CollisionObject(geom)
        self._cinit = True

    @property
    def radius(self):
        return self._radius

    @radius.setter
    @update
    def radius(self, value):
        self._radius = float(value)

    @property
    def length(self):
        return self._length

    @length.setter
    @update
    def length(self, value):
        self._length = float(value)

    def to_dict(self):
        shape = super().to_dict()
        shape["radius"] = self.radius
        shape["length"] = self.length
        return shape


class Sphere(CollisionShape):
    """
    A sphere whose centre is at the local origin.

    :param radius: Radius in metres.
    :param collision: Whether this shape participates in collision checking.
    """

    def __init__(self, radius, **kwargs):
        super().__init__(stype="sphere", **kwargs)
        self.radius = radius

    def _init_coal(self):
        if not self.collision:
            raise ValueError(
                "This shape has collision=False and cannot be used as a collision object"
            )
        self.co = _coal.CollisionObject(_coal.Sphere(self.radius))
        self._cinit = True

    @property
    def radius(self):
        return self._radius

    @radius.setter
    @update
    def radius(self, value):
        self._radius = float(value)

    def to_dict(self):
        shape = super().to_dict()
        shape["radius"] = self.radius
        return shape


class Cuboid(CollisionShape):
    """
    A rectangular prism whose centre is at the local origin.

    :param scale: [length, width, height] in metres.
    :param collision: Whether this shape participates in collision checking.
    """

    def __init__(self, scale, **kwargs):
        super().__init__(stype="cuboid", **kwargs)
        self.scale = scale

    def _init_coal(self):
        if not self.collision:
            raise ValueError(
                "This shape has collision=False and cannot be used as a collision object"
            )
        s = self.scale
        # Coal Box(x, y, z) takes full dimensions (not half-extents)
        self.co = _coal.CollisionObject(_coal.Box(s[0], s[1], s[2]))
        self._cinit = True

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @scale.setter
    @update
    def scale(self, value):
        value = getvector(value if value is not None else [1, 1, 1], 3)
        self._scale = np.array(value)

    def to_dict(self):
        shape = super().to_dict()
        shape["scale"] = self.scale.tolist()
        return shape


class Box(Cuboid):
    def __init__(self, scale, **kwargs):
        warn("Box is deprecated, use Cuboid instead", FutureWarning)
        super().__init__(scale, **kwargs)
