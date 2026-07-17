#!/usr/bin/env python3

"""
@author: Jesse Haviland
"""

import roboticstoolbox as rtb
from numpy import array, ndarray, pi
from spatialmath.base import trotx, troty, trotz
# Aliased: the bottom of this module rebinds the bare name `SE3` to the
# ET.SE3 classmethod (the free-function alias), so the spatialmath class
# needs its own name here.
from spatialmath import SE3 as SE3T

from roboticstoolbox.ets._ET import BaseET, Sym, _AXIS_TO_INT, _resolve_param
from roboticstoolbox.ets.fknm import ET_T, ET_init, ET_update


class ET(BaseET):
    # See BaseET._deepcopy_skip: the compiled acceleration handle is
    # rebuilt by _accel_init() rather than deep-copied.
    _deepcopy_skip = ("_ET__fknm",)

    def __init__(self, **kwargs):
        # Set before super().__init__() runs: BaseET.__init__ may invoke
        # the `param` setter (for a static ET), which calls _accel_update()
        # below. `None` here tells _accel_update() the compiled struct
        # doesn't exist yet, so it skips the sync instead of touching an
        # attribute that isn't there yet.
        self.__fknm = None
        super().__init__(**kwargs)
        # Now that BaseET.__init__ has finished (axis/param/T/joint/etc. are
        # all final), do the one real build of the compiled struct.
        self._accel_init()

    def __mul__(self, other: "ET") -> "rtb.ETS":
        return rtb.ETS([self, other])

    def __add__(self, other: "ET") -> "rtb.ETS":
        return self.__mul__(other)

    # ------------------------------------------------------------------
    # Compiled C++ acceleration. ET2 has none of this - see BaseET's
    # no-op _accel_init/_accel_update and pure-Python A().
    # ------------------------------------------------------------------
    def __axis_to_number(self, axis: str) -> int:
        """
        Private convenience function which converts the axis string to an
        integer for faster processing in the C extensions
        """
        return _AXIS_TO_INT.get(axis, 0)

    def _accel_init(self) -> None:
        """
        Build the compiled struct that holds this ET's data, from the
        current (final) Python-side state.
        """
        if self.jindex is None:
            jindex = 0
        else:
            jindex = self.jindex

        if self.qlim is None:
            if self.kind[0] == "R":
                qlim = array([-pi, pi])
            else:
                qlim = array([0, 1])
        else:
            qlim = self.qlim

        self.__fknm = ET_init(
            self._isstaticsym,
            self.isjoint,
            self.isflip,
            jindex,
            self.__axis_to_number(self.kind),
            self._T,
            qlim,
        )

    def _accel_update(self) -> None:
        """
        Push current Python-side state to the compiled struct. Called
        whenever param/qlim/jindex change after construction. A no-op while
        the struct doesn't exist yet (i.e. mid-__init__, before
        _accel_init() has run for the first time).
        """
        if self.__fknm is None:
            return

        if self.jindex is None:
            jindex = 0
        else:
            jindex = self.jindex

        if self.qlim is None:
            if self.kind[0] == "R":
                qlim = array([-pi, pi])
            else:
                qlim = array([0, 1])
        else:
            qlim = self.qlim

        ET_update(
            self.__fknm,
            self._isstaticsym,
            self.isjoint,
            self.isflip,
            jindex,
            self.__axis_to_number(self.kind),
            self._T,
            qlim,
        )

    @property
    def fknm(self):
        return self.__fknm

    def A(self, q: float | Sym = 0.0) -> ndarray:
        """
        Evaluate an elementary transformation

        :param q: Is used if this ET is variable (a joint)
        :returns: The SE(3) matrix value of the ET
        :rtype: ndarray

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.A()
            >>> e = ET.tx()
            >>> e.A(0.7)

        """
        try:
            # Try and use the C implementation, flip is handled in C
            return ET_T(self.__fknm, q)
        except TypeError:
            # We can't use the fast version (e.g. symbolic q), fall back
            # to the pure-Python evaluation shared with ET2
            return super().A(q)

    @property
    def s(self) -> ndarray:  # pragma: nocover
        if self.kind[1] == "x":
            if self.kind[0] == "R":
                return array([0, 0, 0, 1, 0, 0])
            else:
                return array([1, 0, 0, 0, 0, 0])
        elif self.kind[1] == "y":
            if self.kind[0] == "R":
                return array([0, 0, 0, 0, 1, 0])
            else:
                return array([0, 1, 0, 0, 0, 0])
        else:
            if self.kind[0] == "R":
                return array([0, 0, 0, 0, 0, 1])
            else:
                return array([0, 0, 1, 0, 0, 0])

    @classmethod
    def Rx(
        cls,
        param: float | Sym | str | None = None,
        unit: str = "rad",
        *,
        eta: float | None = None,
        **kwargs,
    ) -> "ET":
        """
        Pure rotation about the x-axis

        :param param: rotation about the x-axis
        :param unit: angular unit, "rad" [default] or "deg"
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.Rx(param)`` is an elementary rotation about the x-axis by a
          constant angle
        - ``ET.Rx()`` is an elementary rotation about the x-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        See Also
        --------
        :func:`ET`
        :func:`isrotation`

        :SymPy: supported
        """
        param = _resolve_param(param, eta)
        return cls(axis="Rx", param=param, axis_func=trotx, unit=unit, **kwargs)

    @classmethod
    def Ry(
        cls,
        param: float | Sym | str | None = None,
        unit: str = "rad",
        *,
        eta: float | None = None,
        **kwargs,
    ) -> "ET":
        """
        Pure rotation about the y-axis

        :param param: rotation about the y-axis
        :param unit: angular unit, "rad" [default] or "deg"
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.Ry(param)`` is an elementary rotation about the y-axis by a
          constant angle
        - ``ET.Ry()`` is an elementary rotation about the y-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        See Also
        --------
        :func:`ET`
        :func:`isrotation`

        :SymPy: supported
        """
        param = _resolve_param(param, eta)
        return cls(axis="Ry", param=param, axis_func=troty, unit=unit, **kwargs)

    @classmethod
    def Rz(
        cls,
        param: float | Sym | str | None = None,
        unit: str = "rad",
        *,
        eta: float | None = None,
        **kwargs,
    ) -> "ET":
        """
        Pure rotation about the z-axis

        :param param: rotation about the z-axis
        :param unit: angular unit, "rad" [default] or "deg"
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.Rz(param)`` is an elementary rotation about the z-axis by a
          constant angle
        - ``ET.Rz()`` is an elementary rotation about the z-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        See Also
        --------
        :func:`ET`
        :func:`isrotation`

        :SymPy: supported
        """
        param = _resolve_param(param, eta)
        return cls(axis="Rz", param=param, axis_func=trotz, unit=unit, **kwargs)

    @classmethod
    def tx(
        cls,
        param: float | Sym | str | None = None,
        *,
        eta: float | None = None,
        **kwargs,
    ) -> "ET":
        """
        Pure translation along the x-axis

        :param param: translation distance along the x-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.tx(param)`` is an elementary translation along the x-axis by a
          distance constant
        - ``ET.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET`
        :func:`istranslation`

        :SymPy: supported
        """
        param = _resolve_param(param, eta)

        # this method is 3x faster than using lambda x: transl(x, 0, 0)
        def axis_func(param):
            # fmt: off
            return array([
                [1, 0, 0, param],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            # fmt: on

        return cls(axis="tx", axis_func=axis_func, param=param, **kwargs)

    @classmethod
    def ty(
        cls,
        param: float | Sym | str | None = None,
        *,
        eta: float | None = None,
        **kwargs,
    ) -> "ET":
        """
        Pure translation along the y-axis

        :param param: translation distance along the y-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.ty(param)`` is an elementary translation along the y-axis by a
          distance constant
        - ``ET.ty()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET`
        :func:`istranslation`

        :SymPy: supported
        """
        param = _resolve_param(param, eta)

        def axis_func(param):
            # fmt: off
            return array([
                [1, 0, 0, 0],
                [0, 1, 0, param],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            # fmt: on

        return cls(axis="ty", param=param, axis_func=axis_func, **kwargs)

    @classmethod
    def tz(
        cls,
        param: float | Sym | str | None = None,
        *,
        eta: float | None = None,
        **kwargs,
    ) -> "ET":
        """
        Pure translation along the z-axis

        :param param: translation distance along the z-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.tz(param)`` is an elementary translation along the z-axis by a
          distance constant
        - ``ET.tz()`` is an elementary translation along the z-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET`
        :func:`istranslation`

        :SymPy: supported
        """
        param = _resolve_param(param, eta)

        def axis_func(param):
            # fmt: off
            return array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, param],
                [0, 0, 0, 1]
            ])
            # fmt: on

        return cls(axis="tz", axis_func=axis_func, param=param, **kwargs)

    @classmethod
    def SE3(cls, T: ndarray | SE3T, **kwargs) -> "ET":
        """
        A static SE3

        :param T: The SE3 transformation matrix
        :returns: An elementary transform
        :rtype: ET

        See Also
        --------
        :func:`ET`
        :func:`istranslation`

        :SymPy: supported
        """

        trans = T.A if isinstance(T, SE3T) else T

        return cls(axis="SE3", T=trans, **kwargs)


# ---------------------------------------------------------------------------
# Bare free-function aliases, so `from roboticstoolbox.ets.ET import *` pulls
# in exactly the 3D names (no ET2 tx/ty collision - see roboticstoolbox.ets.ET2
# for the 2D equivalents, which can't be wildcard-imported alongside these
# since they share the tx/ty names with different meanings).
# ---------------------------------------------------------------------------
Rx = ET.Rx
Ry = ET.Ry
Rz = ET.Rz
tx = ET.tx
ty = ET.ty
tz = ET.tz
SE3 = ET.SE3

__all__ = ["ET", "Rx", "Ry", "Rz", "tx", "ty", "tz", "SE3"]
