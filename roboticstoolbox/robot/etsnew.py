from collections import UserList
from abc import ABC
import numpy as np
from spatialmath.base import trotx, troty, trotz
import fknm
import sympy
import math
from spatialmath import base


class BaseET:
    # all the ET goodness goes in here, simpler because it's a singleton
    # optimize the b***z out of this

    def __init__(
        self,
        axis=None,
        eta=None,
        axis_func=None,
        jindex=None,
        unit="rad",
        flip=False,
        qlim=None,
    ):
        self._axis = axis
        self._eta = eta
        self._axis_func = axis_func
        self._flip = flip
        self._qlim = qlim
        self._jindex = jindex

        if eta is None:
            self._joint = True
            self._T = np.eye(4)
        else:
            self._joint = False
            self._T = axis_func(eta)

        # self.__axis_number = self.__axis_to_number(axis)

        # Initialise the C object which holds ET data
        # This returns a reference to said C data
        self.__fknm = self.__init_c()

    def __init_c(self):
        if self.jindex is None:
            jindex = 0
        else:
            jindex = self.jindex

        return fknm.ET_init(
            self.isjoint, self.isflip, jindex, self.__axis_to_number(self.axis), self._T
        )

    def __mul__(self, other):
        return ETS([self, other])

    def __str__(self):
        return f"{self.axis}({self.eta if self.eta is not None else 'q'})"

    def __axis_to_number(self, axis):
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

    @property
    def fknm(self):
        return self.__fknm

    @property
    def eta(self):
        """
        Get the transform constant

        :return: The constant η if set
        :rtype: float, symbolic or None

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
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
    def eta(self, value):
        """
        Set the transform constant

        :param value: The transform constant η
        :type value: float, symbolic or None

        .. note:: No unit conversions are applied, it is assumed to be in
            radians.
        """
        self._eta = value

    @property
    def axis_func(self):
        return self._axis_func

    @property
    def axis(self):
        """
        The transform type and axis

        :return: The transform type and axis
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ET.tx(1)
            >>> e.axis
            >>> e = ET.rx(90, 'deg')
            >>> e.axis

        """
        return self._axis

    @property
    def isjoint(self):
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
    def isflip(self):
        """
        Test if ET joint is flipped

        :return: True if joint is flipped
        :rtype: bool

        A flipped joint uses the negative of the joint variable, ie. it rotates
        or moves in the opposite direction.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ET.tx()
            >>> e.T(1)
            >>> eflip = ET.tx(flip=True)
            >>> eflip.T(1)
        """
        return self._flip

    @property
    def isrotation(self):
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
    def istranslation(self):
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
    def qlim(self):
        return self._qlim

    @property
    def jindex(self):
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

    def T(self, q=None):
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
                    q = -q
                return self.axis_func(q)
            else:
                return self._T


class ET(BaseET):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def Rx(cls, eta=None, unit="rad", **kwargs):
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

        def axis_func(theta):
            ct = base.sym.cos(theta)
            st = base.sym.sin(theta)

            return np.array(
                [[1, 0, 0, 0], [0, ct, -st, 0], [0, st, ct, 0], [0, 0, 0, 1]]
            )

        return cls(axis="Rx", eta=eta, axis_func=axis_func, unit=unit, **kwargs)

    @classmethod
    def Ry(cls, eta=None, unit="rad", **kwargs):
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
    def Rz(cls, eta=None, unit="rad", **kwargs):
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
    def tx(cls, eta=None, **kwargs):
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
    def ty(cls, eta=None, **kwargs):
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

        # return cls(SE3.Ty, axis='ty', eta=eta)

    @classmethod
    def tz(cls, eta=None, **kwargs):
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


class ETS(UserList):
    # listy superpowers
    # this is essentially just a container for a list of ET instances

    def __init__(self, arg):
        super().__init__()
        if isinstance(arg, list):
            if not all([isinstance(a, ET) for a in arg]):
                raise ValueError("bad arg")
            self.data = arg

        self.__fknm = [et.fknm for et in self.data]
        # print(self.__fknm)

        self.inv_mask = np.array([False, False])
        self._m = len(self.data)

    def __str__(self):
        return " * ".join([str(e) for e in self.data])

    def __mul__(self, other):
        if isinstance(other, ET):
            return ETS([*self.data, other])
        elif isinstance(other, ETS):
            return ETS([*self.data, *other.data])

    def __rmul__(self, other):
        return ETS([other, self.data])

    def fkine(self, q, base=None, tool=None):

        return fknm.ETS_fkine(self._m, self.__fknm, q, base, tool)
