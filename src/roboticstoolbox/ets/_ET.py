#!/usr/bin/env python3

"""
@author: Jesse Haviland

Shared base for ET (3D) and ET2 (2D) - see roboticstoolbox.ets.ET /
roboticstoolbox.ets.ET2 for the concrete classes. This module exists
separately so ET.py and ET2.py can each import BaseET without importing
each other (BaseET is depended on by both, so it can't depend on either
without a cycle).
"""

import re
import warnings
from copy import deepcopy

from numpy import array, ndarray, deg2rad, eye, pi
from numpy.linalg import inv as npinv
from spatialmath.base import getvector, issymbol, tr2rpy, tr2xyt
from typing import Callable, TYPE_CHECKING

from roboticstoolbox.tools.types import ArrayLike, NDArray

_AXIS_TO_INT: dict[str, int] = {"Rx": 0, "Ry": 1, "Rz": 2, "tx": 3, "ty": 4, "tz": 5}

if TYPE_CHECKING:  # pragma: nocover
    import sympy

    Sym = sympy.core.symbol.Symbol  # type: ignore
else:  # pragma: nocover
    Sym = None


def _resolve_param(
    param: "float | Sym | None", eta: "float | None"
) -> "float | Sym | None":
    """
    Merge the `param` kwarg with the deprecated `eta` kwarg.

    `eta` (η) is the name used in the original Elementary Transform Sequence
    paper; `param` is its replacement. If `eta` is passed, warn and use it
    as `param` - this keeps every existing `eta=...` call working
    unchanged, since `param` didn't exist before 1.4.0.
    """
    if eta is not None:
        warnings.warn(
            "the `eta` keyword is deprecated since 1.4.0, use `param` instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return eta
    return param


def _parse_joint_descriptor(s: str) -> "tuple[int | None, bool]":
    """
    Parse a joint descriptor string into (jindex, flip).

    A leading '-' sets flip (and is stripped); a leading '+' is stripped and
    ignored. The first run of digits found anywhere in what remains becomes
    the joint index - this treats 'theta2', 'q2', 'q(3)' and 'θ_3' the same
    regardless of how the index is set off from the rest of the name. If no
    digit is found, jindex is None, so the joint falls through to ETS's
    existing auto-numbering for unassigned joints.
    """
    flip = s.startswith("-")
    if s[:1] in "+-":
        s = s[1:]

    match = re.search(r"\d+", s)
    jindex = int(match.group()) if match else None

    return jindex, flip


class BaseET:
    def __init__(
        self,
        axis: str,
        param: float | Sym | str | None = None,
        axis_func: Callable[[float | Sym], ndarray] | None = None,
        T: ndarray | None = None,
        jindex: int | None = None,
        unit: str = "rad",
        flip: bool | None = None,
        qlim: ArrayLike | None = None,
        *,
        eta: float | None = None,
    ):
        param = _resolve_param(param, eta)

        self._kind = axis

        # A custom joint display name, e.g. "theta2" from a string `param`
        # descriptor below - printed by __str__ in place of the generic
        # "q2" when set.
        self._joint_name = None

        # A string `param` that doesn't parse as a plain number is a joint
        # descriptor (e.g. "theta2", "-q(3)", "θ_3"): regex-parsed for
        # jindex/flip and remembered for __str__, then treated as a
        # variable joint (param=None) from here on. `jindex`/`flip` must
        # not also be given explicitly in this case - one form or the
        # other, not a silent merge of both.
        if isinstance(param, str):
            try:
                param = float(param)
            except ValueError:
                if jindex is not None or flip is not None:
                    raise ValueError(
                        "cannot specify `jindex` or `flip` alongside a string "
                        "joint descriptor for `param`"
                    )
                jindex, flip = _parse_joint_descriptor(param)
                self._joint_name = param
                param = None

        flip = bool(flip)  # None (not given, and not set above) -> False

        # A flag to check if the ET is a static joint with a symbolic value
        # Defaults to False as is set to True if param is a symbol below
        self._isstaticsym = False

        # axis_func/flip/jindex/qlim must all be set before `param` below:
        # the `param` setter (for the static, param-is-not-None case) reads
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

        if param is None:
            self._param = None
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
                if not issymbol(param):
                    param = deg2rad(float(param))
            # This is a static joint. The `param` setter validates axis_func,
            # computes `_T`, sets `_isstaticsym`/`_joint`, and (for ET)
            # syncs the compiled acceleration struct.
            self.param = param

    def __str__(self):
        param_str = ""

        if self.isjoint:
            if self._joint_name is not None:
                param_str = self._joint_name
            elif self.jindex is None:
                param_str = "q"
            else:
                param_str = f"q{self.jindex}"
        elif issymbol(self.param):
            # Check if symbolic
            param_str = f"{self.param}"
        elif self.isrotation and self.param is not None:
            param_str = f"{self.param * (180.0 / pi):.4g}°"
        elif not self.iselementary:
            # Compound/arbitrary transform - kind is guaranteed to be
            # exactly "SE3" or "SE2" here (the only "S"-prefixed kinds).
            if self.kind == "SE3":
                T = self.A()
                rpy = tr2rpy(T) * 180.0 / pi
                if T[:3, -1].any() and rpy.any():
                    param_str = (
                        f"{T[0, -1]:.4g}, {T[1, -1]:.4g}, {T[2, -1]:.4g};"
                        f" {rpy[0]:.4g}°, {rpy[1]:.4g}°, {rpy[2]:.4g}°"
                    )
                elif T[:3, -1].any():
                    param_str = f"{T[0, -1]:.4g}, {T[1, -1]:.4g}, {T[2, -1]:.4g}"
                elif rpy.any():
                    param_str = f"{rpy[0]:.4g}°, {rpy[1]:.4g}°, {rpy[2]:.4g}°"
                else:
                    param_str = ""  # pragma: nocover
            elif self.kind == "SE2":
                T = self.A()
                xyt = tr2xyt(T)
                xyt[2] *= 180 / pi
                param_str = f"{xyt[0]:.4g}, {xyt[1]:.4g}; {xyt[2]:.4g}°"

        else:
            param_str = f"{self.param:.4g}"

        return f"{self.kind}({param_str})"

    def __repr__(self):
        s_param = "" if self.param is None else f"param={self.param}"
        s_T = (
            f"T={repr(self._T)}"
            if (self.param is None and self.axis_func is None)
            else ""
        )
        s_flip = "" if not self.isflip else f"flip={self.isflip}"
        s_qlim = "" if self.qlim is None else f"qlim={repr(self.qlim)}"
        s_jindex = "" if self.jindex is None else f"jindex={self.jindex}"

        kwargs = [s_param, s_T, s_jindex, s_flip, s_qlim]
        s_kwargs = ", ".join(filter(None, kwargs))

        # self.__class__.__name__ rather than isinstance(self, ET)/(self, ET2):
        # ET/ET2 can't be imported here without a circular import (they both
        # depend on BaseET), and this is a display label anyway - it's also
        # more correct for any future subclass than a hardcoded "ET"/"ET2".
        start = self.__class__.__name__

        return f"{start}.{self.kind}({s_kwargs})"

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

    def __radd__(self, other):
        # lets sum() work without an explicit start value, since its
        # default start is the int 0, which has no idea how to compose
        # with an ET/ET2
        if other == 0:
            return self
        return NotImplemented

    # ------------------------------------------------------------------
    # Compiled-acceleration hooks. BaseET (and so ET2) is pure Python; ET
    # overrides both to build/refresh the compiled C++ struct. Keeping
    # these as no-op hooks here means the param/qlim/jindex setters and
    # inv()/__deepcopy__ below don't need to know or care whether the
    # concrete class has acceleration at all.
    # ------------------------------------------------------------------
    def _accel_init(self) -> None:
        pass

    def _accel_update(self) -> None:
        pass

    @property
    def param(self) -> float | Sym | None:
        """
        Get the transform constant

        :returns: The constant value if set
        :rtype: float or Sym or None

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.param
            >>> e = ET.Rx(90, 'deg')
            >>> e.param
            >>> e = ET.ty()
            >>> e.param

        .. rubric:: Notes

        - If the value was given in degrees it will be converted and
            stored internally in radians
        - Historically called `eta` (η), after the notation used in the
            original Elementary Transform Sequence paper (Haviland & Corke,
            "Manipulator Differential Kinematics"). `eta` is kept as a
            deprecated alias below.
        """
        return self._param

    @param.setter
    def param(self, value: float | Sym) -> None:
        """
        Set the transform constant

        :param value: The transform constant

        .. rubric:: Notes

        - No unit conversions are applied, it is assumed to be in
            radians.
        - Setting `param` always makes the ET a static (non-joint) transform:
            `_T` is recomputed from `axis_func`, and (for ET) the compiled
            acceleration struct is refreshed. This is also what ETS.merge()
            relies on when it combines two adjacent static ETs.
        """
        if self.axis_func is None:
            raise TypeError(
                "For a static joint either both `param` and `axis_func` "
                "must be specified otherwise `T` must be supplied"
            )

        self._param = value if issymbol(value) else float(value)
        self._isstaticsym = issymbol(value)
        self._joint = False
        self._T = self.axis_func(self._param).copy(order="F")

        self._accel_update()

    @property
    def eta(self) -> float | Sym | None:
        """
        Get the transform constant

        .. deprecated:: 1.4.0
            `eta` (η) is the name used in the original Elementary Transform
            Sequence paper; kept as a permanent alias for :attr:`param`,
            which is otherwise identical.

        :returns: The constant value if set
        :rtype: float or Sym or None
        """
        warnings.warn(
            "ET.eta is deprecated since 1.4.0, use .param instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._param

    @eta.setter
    def eta(self, value: float | Sym) -> None:
        warnings.warn(
            "ET.eta is deprecated since 1.4.0, use .param instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.param = value

    @property
    def axis_func(
        self,
    ) -> Callable[[float | Sym], ndarray] | None:
        return self._axis_func

    @property
    def kind(self) -> str:
        """
        The transform type and axis

        :returns: The transform type and axis, e.g. ``"Rx"``, ``"tx"``, ``"SE3"``
        :rtype: str

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.kind
            >>> e = ET.Rx(90, 'deg')
            >>> e.kind

        """
        return self._kind

    @property
    def axis(self) -> str:
        """
        The transform type and axis

        .. deprecated:: 1.4.0
            Use :attr:`kind` instead. ``axis`` is kept as an alias and will
            not be repurposed to mean something else in a future release.

        :returns: The transform type and axis
        :rtype: str
        """
        warnings.warn(
            "ET.axis is deprecated since 1.4.0, use .kind instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._kind

    @property
    def ax(self) -> str | None:
        """
        The Cartesian axis this transform acts along/about

        :returns: ``"x"``, ``"y"``, or ``"z"`` for an elementary transform,
            otherwise ``None`` (e.g. ``ET2``'s rotation, which has no axis
            letter, or a compound/arbitrary ``SE3``/``SE2`` transform)
        :rtype: str or None

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.ax
            >>> e = ET.Rx(90, 'deg')
            >>> e.ax

        """
        letter = self._kind[-1]
        return letter if letter in "xyz" else None

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

        return self.kind[0] == "R"

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

        return self.kind[0] == "t"

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

        return self.kind[0] != "S"

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
        elif inv._param is not None:
            inv._T = npinv(inv._T).copy(order="F")
            inv._param = -inv._param

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
