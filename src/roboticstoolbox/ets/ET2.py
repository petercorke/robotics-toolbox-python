#!/usr/bin/env python3

"""
@author: Jesse Haviland
"""

import roboticstoolbox as rtb
from numpy import array, ndarray
from spatialmath.base import trot2, transl2
# Aliased: the bottom of this module rebinds the bare name `SE2` to the
# ET2.SE2 classmethod (the free-function alias), so the spatialmath class
# needs its own name here.
from spatialmath import SE2 as SE2T

from roboticstoolbox.ets._ET import BaseET, Sym, _resolve_param


class ET2(BaseET):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __mul__(self, other: "ET2") -> "rtb.ETS2":
        return rtb.ETS2([self, other])

    def __add__(self, other: "ET2") -> "rtb.ETS2":
        return self.__mul__(other)

    @property
    def s(self) -> ndarray:  # pragma: nocover
        if self.kind[0] == "R":
            return array([0, 0, 0, 1])
        if self.kind[1] == "x":
            return array([1, 0, 0, 0])
        elif self.kind[1] == "y":
            return array([0, 1, 0, 0])
        else:
            return array([0, 0, 1, 0])

    @classmethod
    def R(
        cls,
        param: float | Sym | str | None = None,
        unit: str = "rad",
        *,
        eta: float | None = None,
        **kwargs,
    ) -> "ET2":
        """
        Pure rotation

        :param param: rotation angle
        :param unit: angular unit, "rad" [default] or "deg"
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET2

        - ``ET2.R(param)`` is an elementary rotation by a constant angle
        - ``ET2.R()`` is an elementary rotation by a variable angle, i.e. a
          revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        .. rubric:: Notes

        - In the 2D case this is rotation around the normal to the
            xy-plane.

        See Also
        --------
        :func:`ET2`, :func:`isrotation`

        """
        param = _resolve_param(param, eta)
        return cls(
            axis="R", param=param, axis_func=lambda theta: trot2(theta), unit=unit, **kwargs
        )

    @classmethod
    def tx(
        cls,
        param: float | Sym | str | None = None,
        unit: str = "rad",
        *,
        eta: float | None = None,
        **kwargs,
    ) -> "ET2":
        """
        Pure translation along the x-axis

        :param param: translation distance along the x-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET2

        - ``ET2.tx(param)`` is an elementary translation along the x-axis by a
          distance constant
        - ``ET2.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET2`
        :func:`istranslation`

        """
        param = _resolve_param(param, eta)
        return cls(axis="tx", param=param, axis_func=lambda x: transl2(x, 0), **kwargs)

    @classmethod
    def ty(
        cls,
        param: float | Sym | str | None = None,
        unit: str = "rad",
        *,
        eta: float | None = None,
        **kwargs,
    ) -> "ET2":
        """
        Pure translation along the y-axis

        :param param: translation distance along the y-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET2

        - ``ET2.ty(param)`` is an elementary translation along the y-axis by a
          distance constant
        - ``ET2.ty()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET2`

        """
        param = _resolve_param(param, eta)
        return cls(axis="ty", param=param, axis_func=lambda y: transl2(0, y), **kwargs)

    @classmethod
    def SE2(cls, T: ndarray | SE2T, **kwargs) -> "ET2":
        """
        A static SE2

        :param T: The SE2 transformation matrix
        :returns: An elementary transform
        :rtype: ET2

        See Also
        --------
        :func:`ET2`
        :func:`istranslation`

        :SymPy: supported
        """

        trans = T.A if isinstance(T, SE2T) else T

        return cls(axis="SE2", T=trans, **kwargs)

    # A() is inherited from BaseET: ET2 has no compiled acceleration, so
    # the shared pure-Python evaluation is all it ever needed.


# ---------------------------------------------------------------------------
# Bare free-function aliases, so `from roboticstoolbox.ets.ET2 import *` pulls
# in exactly the 2D names. Note tx/ty here are 2D and mean something
# different to roboticstoolbox.ets.ET's tx/ty (3D) - importing both modules'
# wildcards into the same namespace will have the second import's tx/ty win.
# ---------------------------------------------------------------------------
R = ET2.R
tx = ET2.tx
ty = ET2.ty
SE2 = ET2.SE2

__all__ = ["ET2", "R", "tx", "ty", "SE2"]
