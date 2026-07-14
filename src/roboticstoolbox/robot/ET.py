#!/usr/bin/env python3

"""
@author: Jesse Haviland
"""

from numpy import array, ndarray, deg2rad, eye, pi
from numpy.linalg import inv as npinv
import roboticstoolbox as rtb
from spatialmath.base import (
    trotx,
    troty,
    trotz,
    issymbol,
    tr2rpy,
    trot2,
    transl2,
    tr2xyt,
)
from copy import deepcopy
from roboticstoolbox.robot.fknm import ET_T, ET_init, ET_update
from spatialmath.base import getvector
from spatialmath import SE3, SE2
from typing import Callable, TYPE_CHECKING

# from spatialmath.base.types import ArrayLike
from roboticstoolbox.tools.types import ArrayLike, NDArray

_AXIS_TO_INT: dict[str, int] = {"Rx": 0, "Ry": 1, "Rz": 2, "tx": 3, "ty": 4, "tz": 5}

if TYPE_CHECKING:  # pragma: nocover
    import sympy

    Sym = sympy.core.symbol.Symbol  # type: ignore
else:  # pragma: nocover
    Sym = None


class BaseET:
    def __init__(
        self,
        axis: str,
        eta: float | Sym | None = None,
        axis_func: Callable[[float | Sym], ndarray] | None = None,
        T: ndarray | None = None,
        jindex: int | None = None,
        unit: str = "rad",
        flip: bool = False,
        qlim: ArrayLike | None = None,
    ):
        self._axis = axis

        # A flag to check if the ET is a static joint with a symbolic value
        # Defaults to False as is set to True if eta is a symbol below
        self._isstaticsym = False

        # axis_func/flip/jindex/qlim must all be set before `eta` below:
        # the `eta` setter (for the static, eta-is-not-None case) reads
        # `self.axis_func` to (re)compute `self._T`, and subclasses that add
        # compiled acceleration (see ET._accel_update) need jindex/qlim/flip
        # already in place too.
        self._axis_func = axis_func
        self._flip = flip
        self._jindex = jindex

        if qlim is not None:
            self._qlim: NDArray | None = getvector(qlim, 2, out="array")
        else:
            self._qlim: NDArray | None = None

        if eta is None:
            self._eta = None
            if T is None:
                self._joint = True
                self._T = eye(4).copy(order="F")
                if axis_func is None:
                    raise TypeError("For a variable joint, axis_func must be specified")
            else:
                self._joint = False
                self._T = T.copy(order="F")
        else:
            if axis[0] == "R" and unit.lower().startswith("deg"):
                if not issymbol(eta):
                    eta = deg2rad(float(eta))
            # This is a static joint. The `eta` setter validates axis_func,
            # computes `_T`, sets `_isstaticsym`/`_joint`, and (for ET)
            # syncs the compiled acceleration struct.
            self.eta = eta

    def __str__(self):
        eta_str = ""

        if self.isjoint:
            if self.jindex is None:
                eta_str = "q"
            else:
                eta_str = f"q{self.jindex}"
        elif issymbol(self.eta):
            # Check if symbolic
            eta_str = f"{self.eta}"
        elif self.isrotation and self.eta is not None:
            eta_str = f"{self.eta * (180.0 / pi):.4g}°"
        elif not self.iselementary:
            if isinstance(self, ET):
                T = self.A()
                rpy = tr2rpy(T) * 180.0 / pi
                if T[:3, -1].any() and rpy.any():
                    eta_str = (
                        f"{T[0, -1]:.4g}, {T[1, -1]:.4g}, {T[2, -1]:.4g};"
                        f" {rpy[0]:.4g}°, {rpy[1]:.4g}°, {rpy[2]:.4g}°"
                    )
                elif T[:3, -1].any():
                    eta_str = f"{T[0, -1]:.4g}, {T[1, -1]:.4g}, {T[2, -1]:.4g}"
                elif rpy.any():
                    eta_str = f"{rpy[0]:.4g}°, {rpy[1]:.4g}°, {rpy[2]:.4g}°"
                else:
                    eta_str = ""  # pragma: nocover
            elif isinstance(self, ET2):
                T = self.A()
                xyt = tr2xyt(T)
                xyt[2] *= 180 / pi
                eta_str = f"{xyt[0]:.4g}, {xyt[1]:.4g}; {xyt[2]:.4g}°"

        else:
            eta_str = f"{self.eta:.4g}"

        return f"{self.axis}({eta_str})"

    def __repr__(self):
        s_eta = "" if self.eta is None else f"eta={self.eta}"
        s_T = (
            f"T={repr(self._T)}"
            if (self.eta is None and self.axis_func is None)
            else ""
        )
        s_flip = "" if not self.isflip else f"flip={self.isflip}"
        s_qlim = "" if self.qlim is None else f"qlim={repr(self.qlim)}"
        s_jindex = "" if self.jindex is None else f"jindex={self.jindex}"

        kwargs = [s_eta, s_T, s_jindex, s_flip, s_qlim]
        s_kwargs = ", ".join(filter(None, kwargs))

        start = "ET" if isinstance(self, ET) else "ET2"

        return f"{start}.{self.axis}({s_kwargs})"

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print stringified version when variable is displayed in IPython, ie. on
        a line by itself.

        Example::

            [In [1]: e
            Out [1]: tx(1)
        """
        p.text(str(self))  # pragma: nocover

    # Attribute names to exclude from the generic __dict__ copy below, e.g.
    # an opaque compiled-acceleration handle that can't be deep-copied and
    # must instead be rebuilt fresh by _accel_init(). Empty for the plain
    # Python ET2; overridden by ET.
    _deepcopy_skip: tuple[str, ...] = ()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k not in self._deepcopy_skip:
                setattr(result, k, deepcopy(v, memo))

        result._accel_init()
        return result

    def __eq__(self, other):
        return repr(self) == repr(other)

    # ------------------------------------------------------------------
    # Compiled-acceleration hooks. BaseET (and so ET2) is pure Python; ET
    # overrides both to build/refresh the compiled C++ struct. Keeping
    # these as no-op hooks here means the eta/qlim/jindex setters and
    # inv()/__deepcopy__ below don't need to know or care whether the
    # concrete class has acceleration at all.
    # ------------------------------------------------------------------
    def _accel_init(self) -> None:
        pass

    def _accel_update(self) -> None:
        pass

    @property
    def eta(self) -> float | Sym | None:
        """
        Get the transform constant

        :returns: The constant η if set
        :rtype: float or Sym or None

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.eta
            >>> e = ET.Rx(90, 'deg')
            >>> e.eta
            >>> e = ET.ty()
            >>> e.eta

        .. rubric:: Notes

        - If the value was given in degrees it will be converted and
            stored internally in radians
        """
        return self._eta

    @eta.setter
    def eta(self, value: float | Sym) -> None:
        """
        Set the transform constant

        :param value: The transform constant η

        .. rubric:: Notes

        - No unit conversions are applied, it is assumed to be in
            radians.
        - Setting `eta` always makes the ET a static (non-joint) transform:
            `_T` is recomputed from `axis_func`, and (for ET) the compiled
            acceleration struct is refreshed. This is also what ETS.merge()
            relies on when it combines two adjacent static ETs.
        """
        if self.axis_func is None:
            raise TypeError(
                "For a static joint either both `eta` and `axis_func` "
                "must be specified otherwise `T` must be supplied"
            )

        self._eta = value if issymbol(value) else float(value)
        self._isstaticsym = issymbol(value)
        self._joint = False
        self._T = self.axis_func(self._eta).copy(order="F")

        self._accel_update()

    @property
    def axis_func(
        self,
    ) -> Callable[[float | Sym], ndarray] | None:
        return self._axis_func

    @property
    def axis(self) -> str:
        """
        The transform type and axis

        :returns: The transform type and axis
        :rtype: str

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.axis
            >>> e = ET.Rx(90, 'deg')
            >>> e.axis

        """
        return self._axis

    @property
    def isjoint(self) -> bool:
        """
        Test if ET is a joint

        :returns: True if a joint
        :rtype: bool

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.isjoint
            >>> e = ET.tx()
            >>> e.isjoint

        """
        return self._joint

    @property
    def isflip(self) -> bool:
        """
        Test if ET joint is flipped

        :returns: True if joint is flipped
        :rtype: bool

        A flipped joint uses the negative of the joint variable, ie. it rotates
        or moves in the opposite direction.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx()
            >>> e.T(1)
            >>> eflip = ET.tx(flip=True)
            >>> eflip.T(1)

        """

        return self._flip

    @property
    def isrotation(self) -> bool:
        """
        Test if ET is a rotation

        :returns: True if a rotation
        :rtype: bool

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.isrotation
            >>> e = ET.rx()
            >>> e.isrotation

        """

        return self.axis[0] == "R"

    @property
    def istranslation(self) -> bool:
        """
        Test if ET is a translation

        :returns: True if a translation
        :rtype: bool

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.istranslation
            >>> e = ET.rx()
            >>> e.istranslation

        """

        return self.axis[0] == "t"

    @property
    def qlim(self) -> ndarray | None:
        return self._qlim

    @qlim.setter
    def qlim(self, qlim_new: ArrayLike | None) -> None:
        if qlim_new is not None:
            qlim_new = getvector(qlim_new, 2, out="array")
        self._qlim = qlim_new
        self._accel_update()

    @property
    def jindex(self) -> int | None:
        """
        Get ET joint index

        :returns: The assigned joint index
        :rtype: int or None

        Allows an ET to be associated with a numbered joint in a robot.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx()
            >>> print(e)
            >>> e = ET.tx(j=3)
            >>> print(e)
            >>> print(e.jindex)

        """

        return self._jindex

    @jindex.setter
    def jindex(self, j):
        if not isinstance(j, int) or j < 0:
            raise ValueError(f"jindex is {j}, must be an int >= 0")
        self._jindex = j
        self._accel_update()

    @property
    def iselementary(self) -> bool:
        """
        Test if ET is an elementary transform

        :returns: True if an elementary transform
        :rtype: bool

        .. rubric:: Notes

        - ET's may not actually be "elementary", it can be a complex
            mix of rotations and translations.

        See Also
        --------
        :func:`compile`

        """

        return self.axis[0] != "S"

    def inv(self):
        r"""
        Inverse of ET

        :returns: Inverse of the ET
        :rtype: ET

        The inverse of a given ET.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(2.5)
            >>> print(e)
            >>> print(e.inv())

        """  # noqa

        inv = deepcopy(self)

        if inv.isjoint:
            inv._flip ^= True
        elif not inv.iselementary:
            inv._T = npinv(inv._T).copy(order="F")
        elif inv._eta is not None:
            inv._T = npinv(inv._T).copy(order="F")
            inv._eta = -inv._eta

        inv._accel_update()

        return inv

    def A(self, q: float | Sym = 0.0) -> ndarray:
        """
        Evaluate an elementary transformation

        :param q: Is used if this ET is variable (a joint)
        :returns: The SE(3) or SE(2) matrix value of the ET
        :rtype: ndarray

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.A()
            >>> e = ET.tx()
            >>> e.A(0.7)

        Pure-Python evaluation, shared by ET2 and used by ET as the
        fallback when the compiled acceleration struct can't be used.
        """
        if self.isjoint:
            if self.isflip:
                q = -q  # type: ignore

            if self.axis_func is not None:
                return self.axis_func(q)
            else:  # pragma: no cover
                raise TypeError("axis_func not defined")
        else:  # pragma: no cover
            return self._T


class ET(BaseET):
    # See BaseET._deepcopy_skip: the compiled acceleration handle is
    # rebuilt by _accel_init() rather than deep-copied.
    _deepcopy_skip = ("_ET__fknm",)

    def __init__(self, **kwargs):
        # Set before super().__init__() runs: BaseET.__init__ may invoke
        # the `eta` setter (for a static ET), which calls _accel_update()
        # below. `None` here tells _accel_update() the compiled struct
        # doesn't exist yet, so it skips the sync instead of touching an
        # attribute that isn't there yet.
        self.__fknm = None
        super().__init__(**kwargs)
        # Now that BaseET.__init__ has finished (axis/eta/T/joint/etc. are
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
            if self.axis[0] == "R":
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
            self.__axis_to_number(self.axis),
            self._T,
            qlim,
        )

    def _accel_update(self) -> None:
        """
        Push current Python-side state to the compiled struct. Called
        whenever eta/qlim/jindex change after construction. A no-op while
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
            if self.axis[0] == "R":
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
            self.__axis_to_number(self.axis),
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
        if self.axis[1] == "x":
            if self.axis[0] == "R":
                return array([0, 0, 0, 1, 0, 0])
            else:
                return array([1, 0, 0, 0, 0, 0])
        elif self.axis[1] == "y":
            if self.axis[0] == "R":
                return array([0, 0, 0, 0, 1, 0])
            else:
                return array([0, 1, 0, 0, 0, 0])
        else:
            if self.axis[0] == "R":
                return array([0, 0, 0, 0, 0, 1])
            else:
                return array([0, 0, 1, 0, 0, 0])

    @classmethod
    def Rx(
        cls, eta: float | Sym | None = None, unit: str = "rad", **kwargs
    ) -> "ET":
        """
        Pure rotation about the x-axis

        :param η: rotation about the x-axis
        :param unit: angular unit, "rad" [default] or "deg"
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.Rx(η)`` is an elementary rotation about the x-axis by a
          constant angle η
        - ``ET.Rx()`` is an elementary rotation about the x-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        See Also
        --------
        :func:`ET`
        :func:`isrotation`

        :SymPy: supported
        """

        return cls(axis="Rx", eta=eta, axis_func=trotx, unit=unit, **kwargs)

    @classmethod
    def Ry(
        cls, eta: float | Sym | None = None, unit: str = "rad", **kwargs
    ) -> "ET":
        """
        Pure rotation about the y-axis

        :param η: rotation about the y-axis
        :param unit: angular unit, "rad" [default] or "deg"
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.Ry(η)`` is an elementary rotation about the y-axis by a
          constant angle η
        - ``ET.Ry()`` is an elementary rotation about the y-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        See Also
        --------
        :func:`ET`
        :func:`isrotation`

        :SymPy: supported
        """
        return cls(axis="Ry", eta=eta, axis_func=troty, unit=unit, **kwargs)

    @classmethod
    def Rz(
        cls, eta: float | Sym | None = None, unit: str = "rad", **kwargs
    ) -> "ET":
        """
        Pure rotation about the z-axis

        :param η: rotation about the z-axis
        :param unit: angular unit, "rad" [default] or "deg"
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.Rz(η)`` is an elementary rotation about the z-axis by a
          constant angle η
        - ``ET.Rz()`` is an elementary rotation about the z-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        See Also
        --------
        :func:`ET`
        :func:`isrotation`

        :SymPy: supported
        """
        return cls(axis="Rz", eta=eta, axis_func=trotz, unit=unit, **kwargs)

    @classmethod
    def tx(cls, eta: float | Sym | None = None, **kwargs) -> "ET":
        """
        Pure translation along the x-axis

        :param η: translation distance along the x-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.tx(η)`` is an elementary translation along the x-axis by a
          distance constant η
        - ``ET.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET`
        :func:`istranslation`

        :SymPy: supported
        """

        # this method is 3x faster than using lambda x: transl(x, 0, 0)
        def axis_func(eta):
            # fmt: off
            return array([
                [1, 0, 0, eta],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            # fmt: on

        return cls(axis="tx", axis_func=axis_func, eta=eta, **kwargs)

    @classmethod
    def ty(cls, eta: float | Sym | None = None, **kwargs) -> "ET":
        """
        Pure translation along the y-axis

        :param η: translation distance along the y-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.ty(η)`` is an elementary translation along the y-axis by a
          distance constant η
        - ``ET.ty()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET`
        :func:`istranslation`

        :SymPy: supported
        """

        def axis_func(eta):
            # fmt: off
            return array([
                [1, 0, 0, 0],
                [0, 1, 0, eta],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            # fmt: on

        return cls(axis="ty", eta=eta, axis_func=axis_func, **kwargs)

    @classmethod
    def tz(cls, eta: float | Sym | None = None, **kwargs) -> "ET":
        """
        Pure translation along the z-axis

        :param η: translation distance along the z-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET

        - ``ET.tz(η)`` is an elementary translation along the z-axis by a
          distance constant η
        - ``ET.tz()`` is an elementary translation along the z-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET`
        :func:`istranslation`

        :SymPy: supported
        """

        def axis_func(eta):
            # fmt: off
            return array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, eta],
                [0, 0, 0, 1]
            ])
            # fmt: on

        return cls(axis="tz", axis_func=axis_func, eta=eta, **kwargs)

    @classmethod
    def SE3(cls, T: ndarray | SE3, **kwargs) -> "ET":
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

        trans = T.A if isinstance(T, SE3) else T

        return cls(axis="SE3", T=trans, **kwargs)


class ET2(BaseET):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __mul__(self, other: "ET2") -> "rtb.ETS2":
        return rtb.ETS2([self, other])

    def __add__(self, other: "ET2") -> "rtb.ETS2":
        return self.__mul__(other)

    @property
    def s(self) -> ndarray:  # pragma: nocover
        if self.axis[0] == "R":
            return array([0, 0, 0, 1])
        if self.axis[1] == "x":
            return array([1, 0, 0, 0])
        elif self.axis[1] == "y":
            return array([0, 1, 0, 0])
        else:
            return array([0, 0, 1, 0])

    @classmethod
    def R(
        cls, eta: float | Sym | None = None, unit: str = "rad", **kwargs
    ) -> "ET2":
        """
        Pure rotation

        :param η: rotation angle
        :param unit: angular unit, "rad" [default] or "deg"
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET2

        - ``ET2.R(η)`` is an elementary rotation by a constant angle η
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

        return cls(
            axis="R", eta=eta, axis_func=lambda theta: trot2(theta), unit=unit, **kwargs
        )

    @classmethod
    def tx(
        cls, eta: float | Sym | None = None, unit: str = "rad", **kwargs
    ) -> "ET2":
        """
        Pure translation along the x-axis

        :param η: translation distance along the x-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET2

        - ``ET2.tx(η)`` is an elementary translation along the x-axis by a
          distance constant η
        - ``ET2.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET2`
        :func:`istranslation`

        """

        return cls(axis="tx", eta=eta, axis_func=lambda x: transl2(x, 0), **kwargs)

    @classmethod
    def ty(
        cls, eta: float | Sym | None = None, unit: str = "rad", **kwargs
    ) -> "ET2":
        """
        Pure translation along the y-axis

        :param η: translation distance along the y-axis
        :param j: Explicit joint number within the robot
        :param flip: Joint moves in opposite direction
        :returns: An elementary transform
        :rtype: ET2

        - ``ET2.ty(η)`` is an elementary translation along the y-axis by a
          distance constant η
        - ``ET2.ty()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        See Also
        --------
        :func:`ET2`

        """

        return cls(axis="ty", eta=eta, axis_func=lambda y: transl2(0, y), **kwargs)

    @classmethod
    def SE2(cls, T: ndarray | SE2, **kwargs) -> "ET2":
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

        trans = T.A if isinstance(T, SE2) else T

        return cls(axis="SE2", T=trans, **kwargs)

    # A() is inherited from BaseET: ET2 has no compiled acceleration, so
    # the shared pure-Python evaluation is all it ever needed.
