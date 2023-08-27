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
from roboticstoolbox.fknm import ET_T, ET_init, ET_update
from spatialmath.base import getvector
from spatialmath import SE3, SE2
from typing import Optional, Callable, Union, TYPE_CHECKING

# from spatialmath.base.types import ArrayLike
from roboticstoolbox.tools.types import ArrayLike, NDArray

if TYPE_CHECKING:  # pragma: nocover
    import sympy

    Sym = sympy.core.symbol.Symbol
else:  # pragma: nocover
    Sym = None


class BaseET:
    def __init__(
        self,
        axis: str,
        eta: Union[float, Sym, None] = None,
        axis_func: Optional[Callable[[Union[float, Sym]], ndarray]] = None,
        T: Optional[ndarray] = None,
        jindex: Optional[int] = None,
        unit: str = "rad",
        flip: bool = False,
        qlim: Optional[ArrayLike] = None,
    ):
        self._axis = axis

        # A flag to check if the ET is a static joint with a symbolic value
        # Defaults to False as is set to True if eta is a symbol below
        self._isstaticsym = False

        if eta is None:
            self._eta = None
        else:
            if axis[0] == "R" and unit.lower().startswith("deg"):
                if not issymbol(eta):
                    self.eta = deg2rad(float(eta))
            else:
                self.eta = eta

        self._axis_func = axis_func
        self._flip = flip
        self._jindex = jindex

        if qlim is not None:
            self._qlim: Union[NDArray, None] = getvector(qlim, 2, out="array")
        else:
            self._qlim: Union[NDArray, None] = None

        if self.eta is None:
            if T is None:
                self._joint = True
                self._T = eye(4).copy(order="F")
                if axis_func is None:
                    raise TypeError("For a variable joint, axis_func must be specified")
            else:
                self._joint = False
                self._T = T.copy(order="F")
        else:
            # This is a static joint
            if issymbol(eta):
                self._isstaticsym = True

            self._joint = False
            if axis_func is not None:
                self._T = axis_func(self.eta).copy(order="F")
            else:
                raise TypeError(
                    "For a static joint either both `eta` and `axis_func` "
                    "must be specified otherwise `T` must be supplied"
                )

        # Initialise the C object which holds ET data
        # This returns a reference to said C data
        self.__fknm = self.__init_c()

    def __init_c(self):
        """
        Super Private method which initialises a C object to hold ET Data
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

        return ET_init(
            self._isstaticsym,
            self.isjoint,
            self.isflip,
            jindex,
            self.__axis_to_number(self.axis),
            self._T,
            qlim,
        )

    def __update_c(self):
        """
        Super Private method which updates the C object which holds ET Data
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

        ET_update(
            self.fknm,
            self._isstaticsym,
            self.isjoint,
            self.isflip,
            jindex,
            self.__axis_to_number(self.axis),
            self._T,
            qlim,
        )

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
            eta_str = f"{self.eta * (180.0/pi):.4g}°"
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

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k != "_BaseET__fknm":
                setattr(result, k, deepcopy(v, memo))

        result.__fknm = result.__init_c()
        return result

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __axis_to_number(self, axis: str) -> int:
        """
        Private convenience function which converts the axis string to an
        integer for faster processing in the C extensions
        """
        if isinstance(self, ET2):
            return 0

        if axis[0] == "R":
            if axis[1] == "x":
                return 0
            elif axis[1] == "y":
                return 1
            elif axis[1] == "z":
                return 2
        elif axis[0] == "t":
            if axis[1] == "x":
                return 3
            elif axis[1] == "y":
                return 4
            elif axis[1] == "z":
                return 5
        return 0

    @property
    def fknm(self):
        return self.__fknm

    @property
    def eta(self) -> Union[float, Sym, None]:
        """
        Get the transform constant

        Returns
        -------
        ets
            The constant η if set

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

        Notes
        -----
        - If the value was given in degrees it will be converted and
            stored internally in radians
        """
        return self._eta

    @eta.setter
    def eta(self, value: Union[float, Sym]) -> None:
        """
        Set the transform constant

        Parameters
        ----------
        value
            The transform constant η

        Notes
        -----
        - No unit conversions are applied, it is assumed to be in
            radians.
        """
        self._eta = value if issymbol(value) else float(value)

    @property
    def axis_func(
        self,
    ) -> Union[Callable[[Union[float, Sym]], ndarray], None]:
        return self._axis_func

    @property
    def axis(self) -> str:
        """
        The transform type and axis

        Returns
        -------
        axis
            The transform type and axis

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

        Returns
        -------
        isjoint
            True if a joint

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

        A flipped joint uses the negative of the joint variable, ie. it rotates
        or moves in the opposite direction.

        Returns
        -------
        isflip
            True if joint is flipped

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

        Returns
        -------
        isrotation
            True if a rotation

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

        Returns
        -------
        istranslation
            True if a translation

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
    def qlim(self) -> Union[ndarray, None]:
        return self._qlim

    @qlim.setter
    def qlim(self, qlim_new: Union[ArrayLike, None]) -> None:
        if qlim_new is not None:
            qlim_new = getvector(qlim_new, 2, out="array")
        self._qlim = qlim_new
        self.__update_c()

    @property
    def jindex(self) -> Union[int, None]:
        """
        Get ET joint index

        Returns
        -------
        jindex
            The assigmed joint index

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
        self.__update_c()

    @property
    def iselementary(self) -> bool:
        """
        Test if ET is an elementary transform

        Returns
        -------
        iselementary
            True if an elementary transform

        Notes
        -----
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

        The inverse of a given ET.

        Returns
        -------
        inv
            Inverse of the ET

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

        inv.__update_c()

        return inv

    def A(self, q: Union[float, Sym] = 0.0) -> ndarray:
        """
        Evaluate an elementary transformation

        Parameters
        ----------
        q
            Is used if this ET is variable (a joint)

        Returns
        -------
        T
            The SE(3) or SE(2) matrix value of the ET

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
            # We can't use the fast version, lets use Python instead
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __mul__(self, other: "ET") -> "rtb.ETS":
        return rtb.ETS([self, other])

    def __add__(self, other: "ET") -> "rtb.ETS":
        return self.__mul__(other)

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
        cls, eta: Union[float, Sym, None] = None, unit: str = "rad", **kwargs
    ) -> "ET":
        """
        Pure rotation about the x-axis

        - ``ET.Rx(η)`` is an elementary rotation about the x-axis by a
          constant angle η
        - ``ET.Rx()`` is an elementary rotation about the x-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        Parameters
        ----------
        η
            rotation about the x-axis
        unit
            angular unit, "rad" [default] or "deg"
        j
            Explicit joint number within the robot
        flip
            Joint moves in opposite direction

        Returns
        -------
        Rx
            An elementary transform

        See Also
        --------
        :func:`ET`
        :func:`isrotation`

        :SymPy: supported
        """

        return cls(axis="Rx", eta=eta, axis_func=trotx, unit=unit, **kwargs)

    @classmethod
    def Ry(
        cls, eta: Union[float, Sym, None] = None, unit: str = "rad", **kwargs
    ) -> "ET":
        """
        Pure rotation about the y-axis

        - ``ET.Ry(η)`` is an elementary rotation about the y-axis by a
          constant angle η
        - ``ET.Ry()`` is an elementary rotation about the y-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        Parameters
        ----------
        η
            rotation about the y-axis
        unit
            angular unit, "rad" [default] or "deg"
        j
            Explicit joint number within the robot
        flip
            Joint moves in opposite direction

        Returns
        -------
        Ry
            An elementary transform

        See Also
        --------
        :func:`ET`
        :func:`isrotation`

        :SymPy: supported
        """
        return cls(axis="Ry", eta=eta, axis_func=troty, unit=unit, **kwargs)

    @classmethod
    def Rz(
        cls, eta: Union[float, Sym, None] = None, unit: str = "rad", **kwargs
    ) -> "ET":
        """
        Pure rotation about the z-axis

        - ``ET.Rz(η)`` is an elementary rotation about the z-axis by a
          constant angle η
        - ``ET.Rz()`` is an elementary rotation about the z-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        Parameters
        ----------
        η
            rotation about the z-axis
        unit
            angular unit, "rad" [default] or "deg"
        j
            Explicit joint number within the robot
        flip
            Joint moves in opposite direction

        Returns
        -------
        Rz
            An elementary transform

        See Also
        --------
        :func:`ET`
        :func:`isrotation`

        :SymPy: supported
        """
        return cls(axis="Rz", eta=eta, axis_func=trotz, unit=unit, **kwargs)

    @classmethod
    def tx(cls, eta: Union[float, Sym, None] = None, **kwargs) -> "ET":
        """
        Pure translation along the x-axis

        - ``ET.tx(η)`` is an elementary translation along the x-axis by a
          distance constant η
        - ``ET.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        Parameters
        ----------
        η
            translation distance along the z-axis
        j
            Explicit joint number within the robot
        flip
            Joint moves in opposite direction

        Returns
        -------
        tx
            An elementary transform

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
    def ty(cls, eta: Union[float, Sym, None] = None, **kwargs) -> "ET":
        """
        Pure translation along the y-axis

        - ``ET.ty(η)`` is an elementary translation along the y-axis by a
          distance constant η
        - ``ET.ty()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        Parameters
        ----------
        η
            translation distance along the y-axis
        j
            Explicit joint number within the robot
        flip
            Joint moves in opposite direction

        Returns
        -------
        ty
            An elementary transform

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
    def tz(cls, eta: Union[float, Sym, None] = None, **kwargs) -> "ET":
        """
        Pure translation along the z-axis

        - ``ET.tz(η)`` is an elementary translation along the z-axis by a
          distance constant η
        - ``ET.tz()`` is an elementary translation along the z-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        Parameters
        ----------
        η
            translation distance along the z-axis
        j
            Explicit joint number within the robot
        flip
            Joint moves in opposite direction

        Returns
        -------
        tz
            An elementary transform

        See Also
        --------
        :func:`ET`
        func:`istranslation`

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
    def SE3(cls, T: Union[ndarray, SE3], **kwargs) -> "ET":
        """
        A static SE3

        - ``ET.T(η)`` is an elementary translation along the z-axis by a
          distance constant η
        - ``ET.tz()`` is an elementary translation along the z-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        Parameters
        ----------
        T
            The SE3 trnasformation matrix

        Returns
        -------
        SE3
            An elementary transform

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
        cls, eta: Union[float, Sym, None] = None, unit: str = "rad", **kwargs
    ) -> "ET2":
        """
        Pure rotation

        - ``ET2.R(η)`` is an elementary rotation by a constant angle η
        - ``ET2.R()`` is an elementary rotation by a variable angle, i.e. a
          revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        Parameters
        ----------
        η
            rotation angle
        unit
            angular unit, "rad" [default] or "deg"
        j
            Explicit joint number within the robot
        flip
            Joint moves in opposite direction

        Returns
        -------
        R
            An elementary transform

        Notes
        -----
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
        cls, eta: Union[float, Sym, None] = None, unit: str = "rad", **kwargs
    ) -> "ET2":
        """
        Pure translation along the x-axis

        - ``ET2.tx(η)`` is an elementary translation along the x-axis by a
          distance constant η
        - ``ET2.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        Parameters
        ----------
        η
            translation distance along the x-axis
        j
            Explicit joint number within the robot
        flip
            Joint moves in opposite direction

        Returns
        -------
        tx
            An elementary transform

        See Also
        --------
        :func:`ET2`
        :func:`istranslation`

        """

        return cls(axis="tx", eta=eta, axis_func=lambda x: transl2(x, 0), **kwargs)

    @classmethod
    def ty(
        cls, eta: Union[float, Sym, None] = None, unit: str = "rad", **kwargs
    ) -> "ET2":
        """
        Pure translation along the y-axis

        - ``ET2.tx(η)`` is an elementary translation along the y-axis by a
          distance constant η
        - ``ET2.tx()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        Parameters
        ----------
        η
            translation distance along the y-axis
        j
            Explicit joint number within the robot
        flip
            Joint moves in opposite direction

        Returns
        -------
        ty
            An elementary transform

        See Also
        --------
        :func:`ET2`

        """

        return cls(axis="ty", eta=eta, axis_func=lambda y: transl2(0, y), **kwargs)

    @classmethod
    def SE2(cls, T: Union[ndarray, SE2], **kwargs) -> "ET2":
        """
        A static SE2

        - ``ET2.T(η)`` is an elementary translation along the z-axis by a
          distance constant η
        - ``ET2.tz()`` is an elementary translation along the z-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        Parameters
        ----------
        T
            The SE2 trnasformation matrix

        Returns
        -------
        SE2
            An elementary transform

        See Also
        --------
        :func:`ET2`
        :func:`istranslation`

        :SymPy: supported
        """

        trans = T.A if isinstance(T, SE2) else T

        return cls(axis="SE2", T=trans, **kwargs)

    def A(self, q: Union[float, Sym] = 0.0) -> ndarray:
        """
        Evaluate an elementary transformation

        Parameters
        ----------
        q
            Is used if this ET2 is variable (a joint)

        Returns
        -------
        T
            The SE(2) matrix value of the ET2

        Examples
        --------
        .. runblock:: pycon
        >>> from roboticstoolbox import ET2
        >>> e = ET2.tx(1)
        >>> e.A()
        >>> e = ET2.tx()
        >>> e.A(0.7)

        """

        if self.isjoint:
            if self.isflip:
                q = -1.0 * q

            if self.axis_func is not None:
                return self.axis_func(q)
            else:  # pragma: no cover
                raise TypeError("axis_func not defined")
        else:  # pragma: no cover
            return self._T
