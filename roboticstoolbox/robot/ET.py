from collections import UserList
import numpy as np
from spatialmath.base import trotx, troty, trotz, issymbol, getmatrix, tr2rpy
import fknm
import sympy
from copy import deepcopy
from roboticstoolbox import rtb_get_param

# from spatialmath import base

from spatialmath.base import getvector

from spatialmath import SE3

from typing import Optional, Callable, Union
from numpy.typing import ArrayLike, NDArray

try:  # pragma: no cover
    import sympy

    # Sym = sympy.Expr
    Sym = sympy.core.symbol.Symbol

except ImportError:  # pragma: no cover
    Sym = float


class BaseET:
    def __init__(
        self,
        axis: str,
        eta: Union[float, Sym, None] = None,
        axis_func: Optional[Callable[[Union[float, Sym]], NDArray[np.float64]]] = None,
        T: Optional[NDArray[np.float64]] = None,
        jindex: Optional[int] = None,
        unit: str = "rad",
        flip: bool = False,
        qlim: Optional[ArrayLike] = None,
    ):
        self._axis = axis

        if eta is None:
            self._eta = None
        else:
            if axis[0] == "R" and unit.lower().startswith("deg"):
                if not issymbol(eta):
                    self.eta = np.deg2rad(float(eta))
            else:
                self.eta = eta

        self._axis_func = axis_func
        self._flip = flip
        self._jindex = jindex
        if qlim is not None:
            qlim = np.array(getvector(qlim, 2, out="array"))
        self._qlim = qlim

        if self.eta is None:
            if T is None:
                self._joint = True
                self._T = np.eye(4)
                if axis_func is None:
                    raise TypeError("For a variable joint, axis_func must be specified")
            else:
                self._joint = False
                self._T = T
        else:
            self._joint = False
            if axis_func is not None:
                self._T = axis_func(self.eta)
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
            qlim = np.array([0, 0])
        else:
            qlim = self.qlim

        return fknm.ET_init(
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
            qlim = np.array([0, 0])
        else:
            qlim = self.qlim

        fknm.ET_update(
            self.fknm,
            self.isjoint,
            self.isflip,
            jindex,
            self.__axis_to_number(self.axis),
            self._T,
            qlim,
        )

    def __mul__(self, other: Union["ET", "ETS"]) -> "ETS":
        return ETS([self, other])

    def __add__(self, other: Union["ET", "ETS"]) -> "ETS":
        return self.__mul__(other)

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
            eta_str = f"{self.eta * (180.0/np.pi):.2f}°"
        elif not self.iselementary:
            T = self.T()
            rpy = tr2rpy(T) * 180.0 / np.pi
            zeros = np.zeros(3)
            if T[:3, -1].any() and rpy.any():
                eta_str = f"xyzrpy: {T[0, -1]}, {T[1, -1]}, {T[2, -1]}, {rpy[0]}°, {rpy[1]}°, {rpy[2]}°"
            elif T[:3, -1].any():
                eta_str = f"xyz: {T[0, -1]}, {T[1, -1]}, {T[2, -1]}"
            elif rpy.any():
                eta_str = f"rpy: {rpy[0]}°, {rpy[1]}°, {rpy[2]}°"
            else:
                eta_str = ""
        else:
            eta_str = f"{self.eta}"

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

        return f"ET.{self.axis}({s_kwargs})"

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

        :return: The constant η if set
        :rtype: float, symbolic or None

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.eta
            >>> e = ET.Rx(90, 'deg')
            >>> e.eta
            >>> e = ET.ty()
            >>> e.eta

        .. note:: If the value was given in degrees it will be converted and
            stored internally in radians
        """
        return self._eta

    @eta.setter
    def eta(self, value: Union[float, Sym]) -> None:
        """
        Set the transform constant

        :param value: The transform constant η
        :type value: float, symbolic or None

        .. note:: No unit conversions are applied, it is assumed to be in
            radians.
        """
        self._eta = value if issymbol(value) else float(value)

    @property
    def axis_func(
        self,
    ) -> Union[Callable[[Union[float, Sym]], NDArray[np.float64]], None]:
        return self._axis_func

    @property
    def axis(self) -> str:
        """
        The transform type and axis

        :return: The transform type and axis
        :rtype: str

        Example:

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

        :return: True if a joint
        :rtype: bool

        Example:

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

        :return: True if joint is flipped
        :rtype: bool

        A flipped joint uses the negative of the joint variable, ie. it rotates
        or moves in the opposite direction.

        Example:

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

        :return: True if a rotation
        :rtype: bool

        Example:

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

        :return: True if a translation
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.istranslation
            >>> e = ET.rx()
            >>> e.istranslation
        """
        return self.axis[0] == "t"

    @property
    def qlim(self) -> Union[NDArray[np.float64], None]:
        return self._qlim

    @property
    def jindex(self) -> Union[int, None]:
        """
        Get ET joint index

        :return: The assigmed joint index
        :rtype: int or None

        Allows an ET to be associated with a numbered joint in a robot.

        Example:

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
            raise TypeError(f"jindex is {j}, must be an int >= 0")
        self._jindex = j
        self.__update_c()

    @property
    def iselementary(self) -> bool:
        """
        Test if ET is an elementary transform

        :return: True if an elementary transform
        :rtype: bool

        .. note:: ET's may not actually be "elementary", it can be a complex
            mix of rotations and translations.

        :seealso: :func:`compile`
        """
        return self.axis[0] != "S"

    def inv(self):
        r"""
        Inverse of ET

        :return: [description]
        :rtype: ET instance

        The inverse of a given ET.

        Example:

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
            inv._T = np.linalg.inv(inv._T)
        elif inv._eta is not None:
            inv._T = np.linalg.inv(inv._T)
            inv._eta = -inv._eta

        inv.__update_c()

        return inv

    def T(self, q: Union[float, Sym] = 0.0) -> NDArray[np.float64]:
        """
        Evaluate an elementary transformation

        :param q: Is used if this ET is variable (a joint)
        :type q: float (radians), required for variable ET's
        :return: The SE(3) or SE(2) matrix value of the ET
        :rtype:  ndarray(4,4) or ndarray(3,3)

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tx(1)
            >>> e.T()
            >>> e = ET.tx()
            >>> e.T(0.7)

        """
        try:
            # Try and use the C implementation, flip is handled in C
            return fknm.ET_T(self.__fknm, q)
        except TypeError:
            # We can't use the fast version, lets use Python instead
            if self.isjoint:
                if self.isflip:
                    q = -1.0 * q

                if self.axis_func is not None:
                    return self.axis_func(q)
                else:  # pragma: no cover
                    raise TypeError("axis_func not defined")
            else:  # pragma: no cover
                return self._T


class ET(BaseET):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def Rx(
        cls, eta: Union[float, Sym, None] = None, unit: str = "rad", **kwargs
    ) -> "ET":
        """
        Pure rotation about the x-axis

        :param η: rotation about the x-axis
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.rx(η)`` is an elementary rotation about the x-axis by a
          constant angle η
        - ``ETS.rx()`` is an elementary rotation about the x-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        :seealso: :func:`ETS`, :func:`isrotation`
        :SymPy: supported
        """

        # def axis_func(theta):
        #     ct = base.sym.cos(theta)
        #     st = base.sym.sin(theta)

        #     return np.array(
        #         [[1, 0, 0, 0], [0, ct, -st, 0], [0, st, ct, 0], [0, 0, 0, 1]]
        #     )

        return cls(axis="Rx", eta=eta, axis_func=trotx, unit=unit, **kwargs)

    @classmethod
    def Ry(
        cls, eta: Union[float, Sym, None] = None, unit: str = "rad", **kwargs
    ) -> "ET":
        """
        Pure rotation about the y-axis

        :param η: rotation about the y-axis
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.ry(η)`` is an elementary rotation about the y-axis by a
          constant angle η
        - ``ETS.ry()`` is an elementary rotation about the y-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        :seealso: :func:`ETS`, :func:`isrotation`
        :SymPy: supported
        """
        return cls(axis="Ry", eta=eta, axis_func=troty, unit=unit, **kwargs)

    @classmethod
    def Rz(
        cls, eta: Union[float, Sym, None] = None, unit: str = "rad", **kwargs
    ) -> "ET":
        """
        Pure rotation about the z-axis

        :param η: rotation about the z-axis
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.rz(η)`` is an elementary rotation about the z-axis by a
          constant angle η
        - ``ETS.rz()`` is an elementary rotation about the z-axis by a variable
          angle, i.e. a revolute robot joint. ``j`` or ``flip`` can be set in
          this case.

        :seealso: :func:`ETS`, :func:`isrotation`
        :SymPy: supported
        """
        return cls(axis="Rz", eta=eta, axis_func=trotz, unit=unit, **kwargs)

    @classmethod
    def tx(cls, eta: Union[float, Sym, None] = None, **kwargs) -> "ET":
        """
        Pure translation along the x-axis

        :param η: translation distance along the z-axis
        :type η: float
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.tx(η)`` is an elementary translation along the x-axis by a
          distance constant η
        - ``ETS.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        :seealso: :func:`ETS`, :func:`istranslation`
        :SymPy: supported
        """

        # this method is 3x faster than using lambda x: transl(x, 0, 0)
        def axis_func(eta):
            # fmt: off
            return np.array([
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

        :param η: translation distance along the y-axis
        :type η: float
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.ty(η)`` is an elementary translation along the y-axis by a
          distance constant η
        - ``ETS.ty()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        :seealso: :func:`ETS`, :func:`istranslation`
        :SymPy: supported
        """

        def axis_func(eta):
            # fmt: off
            return np.array([
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

        :param η: translation distance along the z-axis
        :type η: float
        :param j: Explicit joint number within the robot
        :type j: int, optional
        :param flip: Joint moves in opposite direction
        :type flip: bool
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.tz(η)`` is an elementary translation along the z-axis by a
          distance constant η
        - ``ETS.tz()`` is an elementary translation along the z-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        :seealso: :func:`ETS`, :func:`istranslation`
        :SymPy: supported
        """

        def axis_func(eta):
            # fmt: off
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, eta],
                [0, 0, 0, 1]
            ])
            # fmt: on

        return cls(axis="tz", axis_func=axis_func, eta=eta, **kwargs)

    @classmethod
    def SE3(cls, T: Union[NDArray[np.float64], SE3], **kwargs) -> "ET":
        """
        A static SE3

        :param T: The SE3 trnasformation matrix
        :type T: float
        :return: An elementary transform
        :rtype: ET instance

        - ``ET.T(η)`` is an elementary translation along the z-axis by a
          distance constant η
        - ``ETS.tz()`` is an elementary translation along the z-axis by a
          variable distance, i.e. a prismatic robot joint. ``j`` or ``flip``
          can be set in this case.

        :seealso: :func:`ETS`, :func:`istranslation`
        :SymPy: supported
        """

        trans = T.A if isinstance(T, SE3) else T

        return cls(axis="SE3", T=trans, **kwargs)


class ETS(UserList):
    # This is essentially just a container for a list of ET instances

    def __init__(self, arg=None):
        super().__init__()
        if isinstance(arg, list):
            if not all([isinstance(a, ET) for a in arg]):
                raise ValueError("bad arg")
            # Deep copy the ET's before saving them
            self.data = [deepcopy(et) for et in arg]
        else:
            # Initialise with identity ET
            self.data = [ET.SE3(np.eye(4))]

        self.__fknm = [et.fknm for et in self.data]

        self._m = len(self.data)
        self._n = len([True for et in self.data if et.isjoint])

    # def __str__(self):
    #     return " ⊕ ".join([str(e) for e in self.data])

    def __str__(self, q=None):
        """
        Pretty prints the ETS

        :param q: control how joint variables are displayed
        :type q: str
        :return: Pretty printed ETS
        :rtype: str

        ``q`` controls how the joint variables are displayed:

        - None, format depends on number of joint variables
            - one, display joint variable as q
            - more, display joint variables as q0, q1, ...
            - if a joint index was provided, use this value
        - "", display all joint variables as empty parentheses ``()``
        - "θ", display all joint variables as ``(θ)``
        - format string with passed joint variables ``(j, j+1)``, so "θ{0}"
          would display joint variables as θ0, θ1, ... while "θ{1}" would
          display joint variables as θ1, θ2, ...  ``j`` is either the joint
          index, if provided, otherwise a sequential value.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz()
            >>> print(e[:2])
            >>> print(e)
            >>> print(e.__str__(""))
            >>> print(e.__str__("θ{0}"))  # numbering from 0
            >>> print(e.__str__("θ{1}"))  # numbering from 1
            >>> # explicit joint indices
            >>> e = ET.Rz(j=3) * ET.tx(1) * ET.Rz(j=4)
            >>> print(e)
            >>> print(e.__str__("θ{0}"))

        .. note:: Angular parameters are converted to degrees, except if they
            are symbolic.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> from spatialmath.base import symbol
            >>> theta, d = symbol('theta, d')
            >>> e = ET.Rx(theta) * ET.tx(2) * ET.Rx(45, 'deg') * \
            >>>     ET.Ry(0.2) * ET.ty(d)
            >>> str(e)

        :SymPy: supported
        """
        es = []
        j = 0
        c = 0
        s = None
        unicode = rtb_get_param("unicode")

        if q is None:
            if len(self.joints()) > 1:
                q = "q{0}"
            else:
                q = "q"

        # For et in the object, display it, data comes from properties
        # which come from the named tuple
        for et in self:

            if et.isjoint:
                if q is not None:
                    if et.jindex is None:
                        _j = j
                    else:
                        _j = et.jindex
                    qvar = q.format(
                        _j, _j + 1
                    )  # lgtm [py/str-format/surplus-argument]  # noqa
                else:
                    qvar = ""
                if et.isflip:
                    s = f"{et.axis}(-{qvar})"
                else:
                    s = f"{et.axis}({qvar})"
                j += 1

            elif et.isrotation:
                if issymbol(et.eta):
                    s = f"{et.axis}({et.eta:.4g})"
                else:
                    s = f"{et.axis}({et.eta * 180 / np.pi:.4g}°)"

            elif et.istranslation:
                s = f"{et.axis}({et.eta:.4g})"

            elif et.isconstant:
                s = f"C{c}"
                c += 1

            es.append(s)

        if unicode:
            return " \u2295 ".join(es)
        else:
            return " * ".join(es)

    def __mul__(self, other: Union["ET", "ETS"]) -> "ETS":
        if isinstance(other, ET):
            return ETS([*self.data, other])
        elif isinstance(other, ETS):
            return ETS([*self.data, *other.data])

    def __rmul__(self, other: Union["ET", "ETS"]) -> "ETS":
        return ETS([other, self.data])

    def __imul__(self, rest: "ETS"):
        return self + rest

    def __add__(self, rest):
        self.__mul__(rest)

    # redefine so that indexing returns an ET type
    def __getitem__(self, i) -> Union[list[ET], ET]:
        """
        Index or slice an ETS

        :param i: the index or slince
        :type i: int or slice
        :return: Elementary transform
        :rtype: ET

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e[0]
            >>> e[1]
            >>> e[1:3]

        """
        return self.data[i]  # can be [2] or slice, eg. [3:5]

    @property
    def structure(self) -> str:
        """
        Joint structure string

        :return: A string indicating the joint types
        :rtype: str

        A string comprising the characters 'R' or 'P' which indicate the types
        of joints in order from left to right.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.structure

        """
        return "".join(["R" if self.data[i].isrotation else "P" for i in self.joints()])

    @property
    def n(self) -> int:
        """
        Number of joints

        :return: the number of joints in the ETS
        :rtype: int

        Counts the number of joints in the ETS.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rx() * ETS.tx(1) * ETS.tz()
            >>> e.n

        :seealso: :func:`joints`
        """

        return self._n

    def fkine(
        self,
        q: ArrayLike,
        base: Union[NDArray[np.float64], SE3, None] = None,
        tool: Union[NDArray[np.float64], SE3, None] = None,
        include_base: bool = True,
    ) -> NDArray[np.float64]:
        """
        Forward kinematics
        :param q: Joint coordinates
        :type q: ndarray(n) or ndarray(m,n)
        :param base: base transform, optional
        :type base: ndarray(4,4) or SE3
        :param tool: tool transform, optional
        :type tool: ndarray(4,4) or SE3
        :return: The transformation matrix representing the pose of the
            end-effector
        :rtype: SE3 instance
        - ``T = ets.fkine(q)`` evaluates forward kinematics for the robot at
          joint configuration ``q``.
        **Trajectory operation**:
        If ``q`` has multiple rows (mxn), it is considered a trajectory and the
        result is an ``SE3`` instance with ``m`` values.
        .. note::
            - The robot's base tool transform, if set, is incorporated
              into the result.
            - A tool transform, if provided, is incorporated into the result.
            - Works from the end-effector link to the base
        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        # try:
        #     return fknm.ETS_fkine(self._m, self.__fknm, q, base, tool, include_base)
        # except:
        #     pass

        q = getmatrix(q, (None, self.n))
        l, _ = q.shape  # type: ignore
        end = self.data[-1]

        if base is None:
            base = np.eye(4)
        elif isinstance(base, SE3):
            base = np.array(base.A)

        if tool is None:
            tool = np.eye(4)
        elif isinstance(tool, SE3):
            tool = np.array(tool.A)

        if l > 1:
            T = np.zeros((l, 4, 4))
        else:
            T = np.zeros((4, 4))
        Tk = np.eye(4)

        for k, qk in enumerate(q):  # type: ignore
            link = end  # start with last link

            # add tool if provided
            A = link.T(qk[link.jindex])
            if A is None:
                Tk = tool
            else:
                if tool is None:
                    Tk = A
                elif A is not None:
                    Tk = A @ tool

            # add remaining links, back toward the base
            for i in range(self.n - 2, -1, -1):
                link = self.data[i]

                A = link.T(qk[link.jindex])

                if A is not None:
                    Tk = A @ Tk

            # add base transform if it is set
            if include_base == True:
                Tk = base @ Tk

            # cast to pose class and append
            if l > 1:
                T[k, :, :] = Tk
            else:
                T = Tk

        return T

    def jacob0(
        self,
        q: ArrayLike,
        tool: Union[NDArray[np.float64], SE3, None] = None,
    ):
        r"""
        Manipulator geometric Jacobian in the base frame
        :param q: Joint coordinate vector
        :type q: ndarray(n)
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None
        :type tool: SE3, optional
        :return J: Manipulator Jacobian in the base frame
        :rtype: ndarray(6,n)
        - ``robot.jacob0(q)`` is the manipulator Jacobian matrix which maps
          joint  velocity to end-effector spatial velocity expressed in the
          end-effector frame.
        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        Example:
        .. runblock:: pycon
            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.ETS.Puma560()
            >>> puma.jacob0([0, 0, 0, 0, 0, 0])
        """  # noqa

        # Use c extension
        try:
            return fknm.ETS_jacob0(self._m, self._n, self.__fknm, q, tool)
        except TypeError:
            pass

        # Otherwise use Python
        if tool is None:
            tool = np.eye(4)
        elif isinstance(tool, SE3):
            tool = np.array(tool.A)

        q = getvector(q, self.n)

        T = self.fkine(q, include_base=False) @ tool

        U = np.eye(4)
        j = 0
        J = np.zeros((6, self.n))
        zero = np.array([0, 0, 0])
        end = self.data[-1]

        for link in self.data:

            if link.isjoint:
                U = U @ link.T(q[link.jindex])  # type: ignore

                if link == end:
                    U = U @ tool

                Tu = np.linalg.inv(U) @ T
                n = U[:3, 0]
                o = U[:3, 1]
                a = U[:3, 2]
                x = Tu[0, 3]
                y = Tu[1, 3]
                z = Tu[2, 3]

                if link.v.axis == "Rz":
                    J[:3, j] = (o * x) - (n * y)
                    J[3:, j] = a

                elif link.v.axis == "Ry":
                    J[:3, j] = (n * z) - (a * x)
                    J[3:, j] = o

                elif link.v.axis == "Rx":
                    J[:3, j] = (a * y) - (o * z)
                    J[3:, j] = n

                elif link.v.axis == "tx":
                    J[:3, j] = n
                    J[3:, j] = zero

                elif link.v.axis == "ty":
                    J[:3, j] = o
                    J[3:, j] = zero

                elif link.v.axis == "tz":
                    J[:3, j] = a
                    J[3:, j] = zero

                j += 1
            else:
                A = link.T()
                if A is not None:
                    U = U @ A

        return J

    def pop(self, i=-1):
        """
        Pop value

        :param i: item in the list to pop, default is last
        :type i: int
        :return: the popped value
        :rtype: instance of same type
        :raises IndexError: if there are no values to pop

        Removes a value from the value list and returns it.  The original
        instance is modified.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> tail = e.pop()
            >>> tail
            >>> e
        """
        item = super().pop(i)
        return item

    @property
    def m(self):
        """
        Number of transforms

        :return: the number of transforms in the ETS
        :rtype: int

        Counts the number of transforms in the ETS.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rx() * ET.tx(1) * ET.tz()
            >>> e.m

        """

        return self._m

    def inv(self) -> "ETS":
        r"""
        Inverse of ETS

        :return: [description]
        :rtype: ETS instance

        The inverse of a given ETS.  It is computed as the inverse of the
        individual ETs in the reverse order.

        .. math::

            (\mathbf{E}_0, \mathbf{E}_1 \cdots \mathbf{E}_{n-1} )^{-1} = (\mathbf{E}_{n-1}^{-1}, \mathbf{E}_{n-2}^{-1} \cdots \mathbf{E}_0^{-1}{n-1} )

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ET.Rz(j=2) * ET.tx(1) * ET.Rx(j=3,flip=True) * ET.tx(1)
            >>> print(e)
            >>> print(e.inv())
            >>> q = [1,2,3,4]

        .. note:: It is essential to use explicit joint indices to account for
            the reversed order of the transforms.
        """  # noqa

        return ETS([et.inv() for et in reversed(self.data)])

    def joints(self) -> list[int]:
        """
        Get index of joint transforms

        :return: indices of transforms that are joints
        :rtype: list of int

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.joints()

        """
        return np.where([e.isjoint for e in self])[0]

    def jointset(self) -> set[int]:
        """
        Get set of joint indices

        :return: set of unique joint indices
        :rtype: set

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(j=1) * ET.tx(j=2) * ET.Rz(j=1) * ET.tx(1)
            >>> e.jointset()
        """
        return set([self[j].jindex for j in self.joints()])  # type: ignore

    def split(self) -> list["ETS"]:
        """
        Split ETS into link segments

        Returns a list of ETS, each one, apart from the last,
        ends with a variable ET.
        """
        segments = []
        start = 0
        for j, k in enumerate(self.joints()):
            ets_j = self[start : k + 1]
            start = k + 1
            segments.append(ets_j)
        tail = self[start:]

        if isinstance(tail, list):
            tail_len = len(tail)
        elif tail is not None:
            tail_len = 1
        else:
            tail_len = 0

        if tail_len > 0:
            segments.append(tail)

        return segments

    def compile(self) -> "ETS":
        """
        Compile an ETS

        :return: optimised ETS
        :rtype: ETS

        Perform constant folding for faster evaluation.  Consecutive constant
        ETs are compounded, leading to a constant ET which is denoted by
        ``SE3`` when displayed.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.ETS.Panda()
            >>> ets = robot.ets()
            >>> ets
            >>> ets.compile()

        :seealso: :func:`isconstant`
        """
        const = None
        ets = ETS()

        for et in self:

            if et.isjoint:
                # a joint
                if const is not None:
                    # flush the constant
                    if not np.all(const == np.eye(4)):
                        ets *= ET.SE3(const)
                    const = None
                ets *= et  # emit the joint ET
            else:
                # not a joint
                if const is None:
                    const = et.T()
                else:
                    const = const @ et.T()

        if const is not None:
            # flush the constant, tool transform
            if not np.all(const == np.eye(4)):
                ets *= ET.SE3(const)
        return ets

    def insert(self, i: int = -1, et: ET = None) -> None:
        """
        Insert value

        :param i: insert an ET into the ETS, default is at the end
        :type i: int
        :param et: the elementary transform to insert
        :type et: ET

        Inserts an ET into the ET sequence.  The inserted value is at position
        ``i``.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> f = ET.Ry()
            >>> e.insert(2, f)
            >>> e
        """
        self.data.insert(i, et)
