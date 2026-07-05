#!/usr/bin/env python3

"""
@author: Jesse Haviland
@author: Peter Corke
"""

from __future__ import annotations
from collections.abc import MutableSequence
from functools import wraps, cached_property
import numpy as np
from numpy.random import uniform
from numpy.linalg import inv, det, cond, svd
from spatialmath import SE3, SE2
from spatialmath.base import (
    getvector,
    issymbol,
    tr2jac,
    verifymatrix,
    tr2jac2,
    t2r,
    rotvelxform,
    simplify,
    getmatrix,
)
from roboticstoolbox.tools.params import rtb_get_param
from roboticstoolbox.robot.IK import IK_GN, IK_LM, IK_NR, IK_QP

from roboticstoolbox.robot.fknm import (
    ETS_init,
    ETS_fkine,
    ETS_jacob0,
    ETS_jacobe,
    ETS_hessian0,
    ETS_hessiane,
    IK_NR_c,
    IK_GN_c,
    IK_LM_c,
)
from copy import deepcopy
from roboticstoolbox.robot.ET import ET, ET2, BaseET
from typing import overload, TypeVar
from typing import Literal as L
from roboticstoolbox.tools.types import ArrayLike, NDArray

T = TypeVar("T", bound="BaseETS")


def _dirties_fknm(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._fknm_stale = True
        return result
    return wrapper


class BaseETS(MutableSequence):
    def __init__(self):
        self._data: list = []
        self._fknm_stale = True
        self._BaseETS__fknm = None

    # ------------------------------------------------------------------
    # MutableSequence abstract methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    @_dirties_fknm
    def __setitem__(self, i, value):
        self._data[i] = value

    @_dirties_fknm
    def __delitem__(self, i):
        del self._data[i]

    @_dirties_fknm
    def insert(self, index: int, value) -> None:
        self._data.insert(index, value)

    def __repr__(self) -> str:
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseETS):
            return self._data == other._data
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # C handle: lazy build on first use after any mutation
    # ------------------------------------------------------------------

    @property
    def _fknm(self):
        if self._fknm_stale:
            self._copy_to_cpp()
        return self._BaseETS__fknm

    def _copy_to_cpp(self):
        self._BaseETS__fknm = ETS_init(
            [et.fknm for et in self._data],
            self.n,
            self.m,
        )
        self._fknm_stale = False

    def __str__(self, q: str | None = None):
        """
        Pretty prints the ETS

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

        :param q: control how joint variables are displayed
        :returns: Pretty printed ETS
        :rtype: str

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz()
            >>> print(e[:2])
            >>> print(e)
            >>> print(e.__str__(""))
            >>> print(e.__str__("θ{0}"))  # numbering from 0
            >>> print(e.__str__("θ{1}"))  # numbering from 1
            >>> # explicit joint indices
            >>> e = ET.Rz(jindex=3) * ET.tx(1) * ET.Rz(jindex=4)
            >>> print(e)
            >>> print(e.__str__("θ{0}"))

        Angular parameters are converted to degrees, except if they
        are symbolic.

        .. runblock:: pycon
        >>> from roboticstoolbox import ET
        >>> from spatialmath.base import symbol
        >>> theta, d = symbol('theta, d')
        >>> e = ET.Rx(theta) * ET.tx(2) * ET.Rx(45, 'deg') * ET.Ry(0.2) * ET.ty(d)
        >>> str(e)

        """

        es = []
        j = 0
        c = 0
        s = None
        unicode = rtb_get_param("unicode")

        # An empty SE3
        if len(self._data) == 0:
            return "SE3()"

        if q is None:
            if len(self.joints()) > 1:
                q = "q{0}"
            else:
                q = "q"

        # For et in the object, display it, data comes from properties
        # which come from the named tuple
        for et in self._data:
            if et.isjoint:
                if q is not None:
                    if et.jindex is None:  # pragma: nocover  this is no longer possible
                        _j = j
                    else:
                        _j = et.jindex
                    qvar = q.format(
                        _j, _j + 1
                    )
                # else:
                #     qvar = ""

                if et.isflip:
                    s = f"{et.axis}(-{qvar})"
                else:
                    s = f"{et.axis}({qvar})"
                j += 1

            elif et.isrotation:
                if issymbol(et.eta):
                    s = f"{et.axis}({et.eta})"
                else:
                    s = f"{et.axis}({et.eta * 180 / np.pi:.4g}°)"

            elif et.istranslation:
                try:
                    s = f"{et.axis}({et.eta:.4g})"
                except TypeError:  # pragma: nocover
                    s = f"{et.axis}({et.eta})"

            elif not et.iselementary:
                s = str(et)
                c += 1

            es.append(s)

        if unicode:
            return " \u2295 ".join(es)
        else:  # pragma: nocover
            return " * ".join(es)

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython

        Print stringified version when variable is displayed in IPython, ie. on
        a line by itself.

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Examples
        --------

        In [1]: e
        Out [1]: R(q0) ⊕ tx(1) ⊕ R(q1) ⊕ tx(1)

        """

        print(self.__str__())  # pragma: nocover

    def joint_idx(self) -> list[int]:
        """
        Get index of joint transforms

        :returns: indices of transforms that are joints
        :rtype: ndarray

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.joint_idx()

        """

        return np.where([e.isjoint for e in self])[0]  # type: ignore

    def joints(self) -> list[ET]:
        """
        Get a list of the variable ETs with this ETS

        :returns: list of ETs that are joints
        :rtype: list[ET]

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.joints()

        """

        return [e for e in self if e.isjoint]

    def jindex_set(self) -> set[int]:  #
        """
        Get set of joint indices

        :returns: set of unique joint indices
        :rtype: set[int]

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(jindex=1) * ET.tx(jindex=2) * ET.Rz(jindex=1) * ET.tx(1)
            >>> e.jointset()

        """

        return set([self[j].jindex for j in self.joint_idx()])  # type: ignore

    @cached_property
    def jindices(self) -> NDArray:
        """
        Get an array of joint indices

        :returns: array of unique joint indices
        :rtype: ndarray

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(jindex=1) * ET.tx(jindex=2) * ET.Rz(jindex=1) * ET.tx(1)
            >>> e.jointset()

        """

        return np.array([j.jindex for j in self.joints()])  # type: ignore

    @property
    def qlim(self):
        r"""
        Get/Set Joint limits

        Limits are extracted from the link objects.  If joints limits are
        not set for:

        - a revolute joint [-𝜋. 𝜋] is returned
        - a prismatic joint an exception is raised

        :param new_qlim: new joint limits to set
        :type new_qlim: ndarray(2,n)
        :returns: array of joint limit values
        :rtype: ndarray(2,n)
        :raises ValueError: unset limits for a prismatic joint

        Examples
        --------

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.qlim

        """

        limits = np.zeros((2, self.n))

        for i, et in enumerate(self.joints()):
            if et.isrotation:
                if et.qlim is None:
                    v = [-np.pi, np.pi]
                else:
                    v = et.qlim
            elif et.istranslation:
                if et.qlim is None:
                    raise ValueError("undefined prismatic joint limit")
                else:
                    v = et.qlim
            else:
                raise ValueError("Undefined Joint Type")  # pragma: nocover
            limits[:, i] = v

        return limits

    @qlim.setter
    def qlim(self, new_qlim: ArrayLike):
        new_qlim = np.array(new_qlim)

        if new_qlim.shape == (2,) and self.n == 1:
            new_qlim = new_qlim.reshape(2, 1)

        if new_qlim.shape != (2, self.n):
            raise ValueError("new_qlim must be of shape (2, n)")

        for j, i in enumerate(self.joint_idx()):
            et = self[i]
            et.qlim = new_qlim[:, j]
            self[i] = et

    @property
    def structure(self) -> str:
        """
        Joint structure string

        A string comprising the characters 'R' or 'P' which indicate the types
        of joints in order from left to right.

        :returns: a string indicating the joint types
        :rtype: str

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.structure

        """

        return "".join(
            ["R" if self._data[i].isrotation else "P" for i in self.joint_idx()]
        )

    @property
    def n(self) -> int:
        """
        Number of joints

        :returns: the number of joints in the ETS
        :rtype: int

        Counts the number of joints in the ETS.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rx() * ET.tx(1) * ET.tz()
            >>> e.n

        See Also
        --------
        :func:`joints`

        """

        return sum(1 for et in self._data if et.isjoint)

    @property
    def m(self) -> int:
        """
        Number of transforms

        :returns: the number of transforms in the ETS
        :rtype: int

        Counts the number of transforms in the ETS.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rx() * ET.tx(1) * ET.tz()
            >>> e.m

        """

        return len(self._data)

    @overload
    def data(self: "ETS") -> list[ET]: ...  # pragma: nocover

    @overload
    def data(self: "ETS2") -> list[ET2]: ...  # pragma: nocover

    @property
    def data(self):
        return self._data

    @data.setter
    @overload
    def data(self: "ETS", new_data: list[ET]): ...  # pragma: nocover

    @data.setter
    @overload
    def data(self: "ETS", new_data: list[ET2]): ...  # pragma: nocover

    @data.setter
    def data(self, new_data):
        self._data = new_data
        self._fknm_stale = True

    @overload
    def split(self: "ETS") -> list["ETS"]: ...  # pragma: nocover

    @overload
    def split(self: "ETS2") -> list["ETS2"]: ...  # pragma: nocover

    def split(self):
        """
        Split ETS into link segments

        :returns: a list of ETS, each one, apart from the last, ends with a variable ET.

        """

        segments = []
        start = 0

        for j, k in enumerate(self.joint_idx()):
            ets_j = self._data[start : k + 1]
            start = k + 1
            segments.append(self.__class__(ets_j))

        tail = self._data[start:]

        if len(tail) > 0:
            segments.append(self.__class__(tail))

        return segments

    def inv(self: T) -> T:
        r"""
        Inverse of ETS

        The inverse of a given ETS.  It is computed as the inverse of the
        individual ETs in the reverse order.

        .. math::

            (\mathbf{E}_0, \mathbf{E}_1 \cdots \mathbf{E}_{n-1} )^{-1} = (\mathbf{E}_{n-1}^{-1}, \mathbf{E}_{n-2}^{-1} \cdots \mathbf{E}_0^{-1}{n-1} )

        :returns: Inverse of the ETS

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(jindex=2) * ET.tx(1) * ET.Rx(jindex=3,flip=True) * ET.tx(1)
            >>> print(e)
            >>> print(e.inv())

        .. rubric:: Notes

        - It is essential to use explicit joint indices to account for
            the reversed order of the transforms.

        """

        return self.__class__([et.inv() for et in reversed(self._data)])  # type: ignore[call-arg]

    @overload
    def __getitem__(self: "BaseETS", i: int) -> BaseET: ...

    @overload
    def __getitem__(self: "ETS", i: int) -> ET: ...

    @overload
    def __getitem__(self: "ETS", i: slice) -> list[ET]: ...

    @overload
    def __getitem__(self: "ETS2", i: int) -> ET2: ...

    @overload
    def __getitem__(self: "ETS2", i: slice) -> list[ET2]: ...

    def __getitem__(self, i):
        """
        Index or slice an ETS

        :param i: the index or slice
        :returns: elementary transform

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e[0]
            >>> e[1]
            >>> e[1:3]

        """
        return self._data[i]  # can be [2] or slice, eg. [3:5]

    def __deepcopy__(self, memo):
        new_data = []

        for data in self:
            new_data.append(deepcopy(data))

        cls = self.__class__
        result = cls(new_data)
        memo[id(self)] = result
        return result

    def plot(self, *args, **kwargs):
        from roboticstoolbox.robot.Robot import Robot, Robot2

        if isinstance(self, ETS):
            robot = Robot(self)
        else:
            robot = Robot2(self)

        robot.plot(*args, **kwargs)

    def teach(self, *args, **kwargs):
        from roboticstoolbox.robot.Robot import Robot, Robot2

        if isinstance(self, ETS):
            robot = Robot(self)
        else:
            robot = Robot2(self)

        robot.teach(*args, **kwargs)

    def random_q(self, i: int = 1) -> NDArray:
        """
        Generate a random valid joint configuration

        :param i: number of configurations to generate
        :returns: random joint configuration
        :rtype: ndarray(n,) or ndarray(i,n)

        Generates a random q vector within the joint limits defined by
        ``self.qlim``.

        Examples
        --------

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.Panda()
            >>> ets = robot.ets()
            >>> q = ets.random_q()
            >>> q

        """

        if i == 1:
            q = np.zeros(self.n)

            for i in range(self.n):
                q[i] = uniform(self.qlim[0, i], self.qlim[1, i])

        else:
            q = np.zeros((i, self.n))

            for j in range(i):
                for i in range(self.n):
                    q[j, i] = uniform(self.qlim[0, i], self.qlim[1, i])

        return q


class ETS(BaseETS):
    """
    This class implements an elementary transform sequence (ETS) for 3D

    An instance can contain an elementary transform (ET) or an elementary
    transform sequence (ETS). It has list-like properties by subclassing
    UserList, which means we can perform indexing, slicing pop, insert, as well
    as using it as an iterator over its values.

    - ``ETS()`` an empty ETS list
    - ``ETS(et)`` an ETS containing a single ET
    - ``ETS([et0, et1, et2])`` an ETS consisting of three ET's

    :param arg: list of ETs or a single ET to initialise the ETS

    Examples
    --------

    .. runblock:: pycon

        >>> from roboticstoolbox import ETS, ET
        >>> e = ET.Rz(0.3) # a single ET, rotation about z
        >>> ets1 = ETS(e)
        >>> len(ets1)
        >>> ets2 = ET.Rz(0.3) * ET.tx(2) # an ETS
        >>> len(ets2)                    # of length 2
        >>> ets2[1]                      # an ET sliced from the ETS

    .. rubric:: References

    - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
      Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
    - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
      Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).


    See Also
    --------
    :func:`rx`
    :func:`ry`
    :func:`rz`
    :func:`tx`
    :func:`ty`
    :func:`tz`

    """

    def __init__(
        self,
        arg: list[ETS | ET] | list[ET] | list[ETS] | ET | ETS | None = None,
    ):
        super().__init__()
        if isinstance(arg, list):
            for item in arg:
                if isinstance(item, ET):
                    self._data.append(deepcopy(item))
                elif isinstance(item, ETS):
                    for ets_item in item:
                        self._data.append(deepcopy(ets_item))
                else:
                    raise TypeError("Invalid arg")
        elif isinstance(arg, ET):
            self._data.append(deepcopy(arg))
        elif isinstance(arg, ETS):
            for ets_item in arg:
                self._data.append(deepcopy(ets_item))
        elif arg is not None:
            raise TypeError("Invalid arg")

        self._auto_jindex = False

        # Check if jindices are set
        joints = self.joints()

        # Number of joints with a jindex
        jindices = 0

        # Number of joints with a sequential jindex (j[2] -> jindex = 2)
        seq_jindex = 0

        # Count them up
        for j, joint in enumerate(joints):
            if joint.jindex is not None:
                jindices += 1
                if joint.jindex == j:
                    seq_jindex += 1

        if (
            jindices == self.n - 1
            and seq_jindex == self.n - 1
            and joints[-1].jindex is None
        ):
            # ets has sequential jindicies, except for the last.
            joints[-1].jindex = self.n - 1
            self._auto_jindex = True

        elif jindices > 0 and not jindices == self.n:
            raise ValueError(
                "You can not have some jindices set for the ET's in arg. It must be all"
                " or none"
            )  # pragma: nocover
        elif jindices == 0 and self.n > 0:
            # Set them ourself
            for j, joint in enumerate(joints):
                joint.jindex = j

            self._auto_jindex = True

    def __mul__(self, other: ET | ETS) -> "ETS":
        if isinstance(other, ET):
            return ETS([*self._data, other])
        else:
            return ETS([*self._data, *other._data])  # pragma: nocover

    def __rmul__(self, other: ET | ETS) -> "ETS":
        return ETS([other, *self._data])  # pragma: nocover

    def __imul__(self, rest: "ETS"):
        return self + rest  # pragma: nocover

    def __add__(self, rest) -> "ETS":
        return self.__mul__(rest)  # pragma: nocover

    def compile(self) -> "ETS":
        """
        Compile an ETS

        :returns: optimised ETS
        :rtype: ETS

        Perform constant folding for faster evaluation.  Consecutive constant
        ETs are compounded, leading to a constant ET which is denoted by
        ``SE3`` when displayed.

        Examples
        --------

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.ETS.Panda()
            >>> ets = robot.ets()
            >>> ets
            >>> ets.compile()

        See Also
        --------
        :func:`isconstant`
        """
        const = None
        ets = ETS()

        for et in self:
            if et.isjoint:
                # a joint
                if const is not None:
                    # flush the constant
                    if not np.array_equal(const, np.eye(4)):
                        ets *= ET.SE3(const)
                    const = None
                ets *= et  # emit the joint ET
            else:
                # not a joint
                if const is None:
                    const = et.A()
                else:
                    const = const @ et.A()

        if const is not None:
            # flush the constant, tool transform
            if not np.array_equal(const, np.eye(4)):
                ets *= ET.SE3(const)
        return ets

    def insert(  # type: ignore[override]
        self,
        index: int,
        value: ET | ETS,
    ) -> None:
        """
        Insert value

        :param index: position to insert at
        :param value: the elementary transform or sequence to insert

        Inserts an ET or ETS into the ET sequence.  The inserted ET is at position
        ``index``; an ETS is expanded and inserted element by element.

        Examples
        --------

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> f = ET.Ry()
            >>> e.insert(2, f)
            >>> e

        """

        if isinstance(value, ET):
            self._data.insert(index, value)
        elif isinstance(value, ETS):
            for j, et in enumerate(value):
                self._data.insert(index + j, et)
        self._fknm_stale = True

    def fkine(
        self,
        q: ArrayLike,
        base: NDArray | SE3 | None = None,
        tool: NDArray | SE3 | None = None,
        include_base: bool = True,
    ) -> SE3:
        """
        Forward kinematics

        :param q: Joint coordinates
        :param base: a base transform applied before the ETS
        :param tool: tool transform, optional
        :param include_base: set to True if the base transform should be considered
        :returns: the transformation matrix representing the pose of the end-effector
        :rtype: SE3

        ``T = ets.fkine(q)`` evaluates forward kinematics for the ets at
        joint configuration ``q``.

        **Trajectory operation**:
        If ``q`` has multiple rows (mxn), it is considered a trajectory and the
        result is an ``SE3`` instance with ``m`` values.

        Examples
        --------

        The following example makes a ``panda`` robot object, gets the ets, and
        solves for the forward kinematics at the listed configuration.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])

        .. rubric:: Notes

        - A tool transform, if provided, is incorporated into the result.
        - Works from the end-effector link to the base

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """

        ret = SE3.Empty()
        fk = self.eval(q, base, tool, include_base)

        if fk.dtype == "O":
            # symbolic
            fk = np.array(simplify(fk))

        if fk.ndim == 3:
            for T in fk:
                ret.append(SE3(T, check=False))  # type: ignore
        else:
            ret = SE3(fk, check=False)

        return ret

    def eval(
        self,
        q: ArrayLike,
        base: NDArray | SE3 | None = None,
        tool: NDArray | SE3 | None = None,
        include_base: bool = True,
    ) -> NDArray:
        """
        Forward kinematics (returns raw ndarray)

        :param q: Joint coordinates
        :param base: a base transform applied before the ETS
        :param tool: tool transform, optional
        :param include_base: set to True if the base transform should be considered
        :returns: the transformation matrix representing the pose of the end-effector
        :rtype: ndarray(4,4) or ndarray(m,4,4)

        ``T = ets.eval(q)`` evaluates forward kinematics for the ets at
        joint configuration ``q``.

        **Trajectory operation**:
        If ``q`` has multiple rows (mxn), it is considered a trajectory and the
        result is a 3d array with ``m`` planes.

        Examples
        --------

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> panda = rtb.models.Panda().ets()
            >>> panda.eval([0, -0.3, 0, -2.2, 0, 2, 0.7854])

        .. rubric:: Notes

        - A tool transform, if provided, is incorporated into the result.
        - Works from the end-effector link to the base

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """

        return ETS_fkine(self._fknm, q, base, tool, include_base, _data=self.data)

    def jacob0(
        self,
        q: ArrayLike,
        tool: NDArray | SE3 | None = None,
    ) -> NDArray:
        r"""
        Manipulator geometric Jacobian in the base frame

        :param q: Joint coordinate vector
        :param tool: a static tool transformation matrix to apply to the end of ETS
        :returns: Manipulator Jacobian in the base frame
        :rtype: ndarray(6,n)

        ``robot.jacob0(q)`` is the manipulator Jacobian matrix which maps
        joint velocity to end-effector spatial velocity expressed in the
        base frame.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        Examples
        --------

        The following example makes a ``Puma560`` robot object, and solves for the
        base-frame Jacobian at the zero joint angle configuration

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.Puma560().ets()
        >>> puma.jacob0([0, 0, 0, 0, 0, 0])

        .. rubric:: Notes

        - This is the geometric Jacobian as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a
            velocity twist as per the text by Lynch & Park.

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """

        return ETS_jacob0(self._fknm, q, tool, _data=self.data, _n=self.n)

    def jacobe(
        self,
        q: ArrayLike,
        tool: NDArray | SE3 | None = None,
    ) -> NDArray:
        r"""
        Manipulator geometric Jacobian in the end-effector frame

        :param q: Joint coordinate vector
        :param tool: a static tool transformation matrix to apply to the end of ETS
        :returns: Manipulator Jacobian in the end-effector frame
        :rtype: ndarray(6,n)

        ``robot.jacobe(q)`` is the manipulator Jacobian matrix which maps
        joint velocity to end-effector spatial velocity expressed in the
        end-effector frame.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        Examples
        --------

        The following example makes a ``Puma560`` robot object, and solves for the
        end-effector frame Jacobian at the zero joint angle configuration

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.Puma560().ets()
        >>> puma.jacobe([0, 0, 0, 0, 0, 0])

        .. rubric:: Notes

        - This is the geometric Jacobian as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a
            velocity twist as per the text by Lynch & Park.

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """

        return ETS_jacobe(self._fknm, q, tool, _data=self.data, _n=self.n)

    def hessian0(
        self,
        q: ArrayLike | None = None,
        J0: NDArray | None = None,
        tool: NDArray | SE3 | None = None,
    ) -> NDArray:
        r"""
        Manipulator Hessian in the base frame

        :param q: joint angles (optional if J0 supplied)
        :param J0: the manipulator Jacobian in the base frame (optional if q supplied)
        :param tool: a static tool transformation matrix to apply to the end of ETS
        :returns: The manipulator Hessian in the base frame
        :rtype: ndarray(n,6,n)

        This method computes the manipulator Hessian in the base frame.  If
        we take the time derivative of the differential kinematic relationship

        .. math::

            \nu    &= \mat{J}(\vec{q}) \dvec{q} \\
            \alpha &= \dmat{J} \dvec{q} + \mat{J} \ddvec{q}

        where

        .. math::

            \dmat{J} = \mat{H} \dvec{q}

        and :math:`\mat{H} \in \mathbb{R}^{6\times n \times n}` is the
        Hessian tensor.

        The elements of the Hessian are

        .. math::

            \mat{H}_{i,j,k} =  \frac{d^2 u_i}{d q_j d q_k}

        where :math:`u = \{t_x, t_y, t_z, r_x, r_y, r_z\}` are the elements
        of the spatial velocity vector.

        Similarly, we can write

        .. math::

            \mat{J}_{i,j} = \frac{d u_i}{d q_j}

        Examples
        --------

        The following example makes a ``Panda`` robot object, and solves for the
        base frame Hessian at the given joint angle configuration

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> panda.hessian0([0, -0.3, 0, -2.2, 0, 2, 0.7854])

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        """

        return ETS_hessian0(self._fknm, q, J0, tool, _data=self.data, _n=self.n)

    def hessiane(
        self,
        q: ArrayLike | None = None,
        Je: NDArray | None = None,
        tool: NDArray | SE3 | None = None,
    ) -> NDArray:
        r"""
        Manipulator Hessian in the end-effector frame

        :param q: joint angles (optional if Je supplied)
        :param Je: the manipulator Jacobian in the end-effector frame (optional if q supplied)
        :param tool: a static tool transformation matrix to apply to the end of ETS
        :returns: The manipulator Hessian in the end-effector frame
        :rtype: ndarray(n,6,n)

        This method computes the manipulator Hessian in the end-effector frame.  If
        we take the time derivative of the differential kinematic relationship

        .. math::

            \nu    &= \mat{J}(\vec{q}) \dvec{q} \\
            \alpha &= \dmat{J} \dvec{q} + \mat{J} \ddvec{q}

        where

        .. math::

            \dmat{J} = \mat{H} \dvec{q}

        and :math:`\mat{H} \in \mathbb{R}^{6\times n \times n}` is the
        Hessian tensor.

        The elements of the Hessian are

        .. math::

            \mat{H}_{i,j,k} =  \frac{d^2 u_i}{d q_j d q_k}

        where :math:`u = \{t_x, t_y, t_z, r_x, r_y, r_z\}` are the elements
        of the spatial velocity vector.

        Similarly, we can write

        .. math::

            \mat{J}_{i,j} = \frac{d u_i}{d q_j}

        Examples
        --------

        The following example makes a ``Panda`` robot object, and solves for the
        end-effector frame Hessian at the given joint angle configuration

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> panda.hessiane([0, -0.3, 0, -2.2, 0, 2, 0.7854])

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        """

        return ETS_hessiane(self._fknm, q, Je, tool, _data=self.data, _n=self.n)

    def jacob0_analytical(
        self,
        q: ArrayLike,
        representation: str = "rpy/xyz",
        tool: NDArray | SE3 | None = None,
    ):
        r"""
        Manipulator analytical Jacobian in the base frame

        :param q: joint coordinate vector
        :param representation: angular representation
        :param tool: a static tool transformation matrix to apply to the end of ETS
        :returns: Manipulator Jacobian in the base frame
        :rtype: ndarray(6,n)

        ``robot.jacob0_analytical(q)`` is the manipulator Jacobian matrix which maps
        joint velocity to end-effector spatial velocity expressed in the base frame.
        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        |``representation``   |       Rotational representation     |
        |---------------------|-------------------------------------|
        |``'rpy/xyz'``        |   RPY angular rates in XYZ order    |
        |``'rpy/zyx'``        |   RPY angular rates in XYZ order    |
        |``'eul'``            |   Euler angular rates in ZYZ order  |
        |``'exp'``            |   exponential coordinate rates      |

        Examples
        --------

        Makes a robot object and computes the analytic Jacobian for the given
        joint configuration

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.ETS.Puma560().ets()
        >>> puma.jacob0_analytical([0, 0, 0, 0, 0, 0])

        """

        T = self.eval(q, tool=tool)
        J = self.jacob0(q, tool=tool)
        gamma = t2r(T)[:3, :3]
        A = rotvelxform(gamma, full=True, inverse=True, representation=representation)
        return A @ J

    def jacobm(self, q: ArrayLike) -> NDArray:
        r"""
        The manipulability Jacobian

        :param q: joint angles/configuration of the robot
        :returns: The manipulability Jacobian
        :rtype: ndarray(n,1)

        This measure relates the rate of change of the manipulability to the
        joint velocities of the robot.
        Yoshikawa's manipulability measure

        .. math::

            m(\vec{q}) = \sqrt{\mat{J}(\vec{q}) \mat{J}(\vec{q})^T}

        This method returns its Jacobian with respect to configuration

        .. math::

            \frac{\partial m(\vec{q})}{\partial \vec{q}}

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        """

        J = self.jacob0(q)
        H = self.hessian0(q)

        manipulability = self.manipulability(q)

        # J = J[axes, :]
        # H = H[:, axes, :]

        b = inv(J @ J.T)
        Jm = np.zeros((self.n, 1))

        for i in range(self.n):
            c = J @ H[i, :, :].T
            Jm[i, 0] = manipulability * (c.flatten("F")).T @ b.flatten("F")

        return Jm

    def manipulability(
        self,
        q,
        method: L["yoshikawa", "minsingular", "invcondition"] = "yoshikawa",
        axes: L["all", "trans", "rot"] | list[bool] = "all",
    ):
        """
        Manipulability measure

        :param q: joint coordinates (trajectory as matrix(m,n))
        :param method: method to use, ``"yoshikawa"`` (default), ``"invcondition"``, or ``"minsingular"``
        :param axes: task space axes to consider: ``"all"`` [default], ``"trans"``, or ``"rot"``
        :returns: the manipulability metric
        :rtype: float | ndarray(m)

        ``manipulability(q)`` is the scalar manipulability index
        for the ets at the joint configuration ``q``.  It indicates
        dexterity, that is, how well conditioned the ets is for motion
        with respect to the 6 degrees of Cartesian motion.  The value is
        zero if the ets is at a singularity.

        Various measures are supported:

        | Measure           |       Description                               |
        |-------------------|-------------------------------------------------|
        | ``"yoshikawa"``   | Volume of the velocity ellipsoid, *distance*    |
        |                   | from singularity [Yoshikawa85]_                 |
        | ``"invcondition"``| Inverse condition number of Jacobian, isotropy  |
        |                   | of the velocity ellipsoid [Klein87]_            |
        | ``"minsingular"`` | Minimum singular value of the Jacobian,         |
        |                   | *distance*  from singularity [Klein87]_         |

        **Trajectory operation**:

        If ``q`` is a matrix (m,n) then the result (m,) is a vector of
        manipulability indices for each joint configuration specified by a row
        of ``q``.

        .. rubric:: Notes

        - Invokes the ``jacob0`` method of the robot if ``J`` is not passed
        - The "all" option includes rotational and translational
          dexterity, but this involves adding different units. It can be
          more useful to look at the translational and rotational
          manipulability separately.
        - Examples in the RVC book (1st edition) can be replicated by
          using the "all" option
        - Asada's measure requires inertial a robot model with inertial
          parameters.

        .. rubric:: References

        .. [Yoshikawa85] Manipulability of Robotic Mechanisms. Yoshikawa T.,
                The International Journal of Robotics Research.
                1985;4(2):3-9. doi:10.1177/027836498500400201
        .. [Klein87] Dexterity Measures for the Design and Control of
                Kinematically Redundant Manipulators. Klein CA, Blaho BE.
                The International Journal of Robotics Research.
                1987;6(2):72-83. doi:10.1177/027836498700600206
        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 7.


        .. versionchanged:: 1.0.4
            Removed 'both' option for axes, added a custom list option.

        """

        axes_list: list[bool] = []

        if isinstance(axes, list):
            axes_list = axes
        elif axes == "all":
            axes_list = [True, True, True, True, True, True]
        elif axes.startswith("trans"):
            axes_list = [True, True, True, False, False, False]
        elif axes.startswith("rot"):
            axes_list = [False, False, False, True, True, True]
        else:
            raise ValueError("axes must be all, trans, rot or both")

        def yoshikawa(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            if J.shape[0] == J.shape[1]:
                # simplified case for square matrix
                return abs(det(J))
            else:
                m2 = det(J @ J.T)
                return np.sqrt(abs(m2))

        def condition(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            return 1 / cond(J)

        def minsingular(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            s = svd(J, compute_uv=False)
            return s[-1]  # return last/smallest singular value of J

        # choose the handler function
        if method == "yoshikawa":
            mfunc = yoshikawa
        elif method == "invcondition":
            mfunc = condition
        elif method == "minsingular":
            mfunc = minsingular
        else:
            raise ValueError("Invalid method chosen")

        # Otherwise use the q vector/matrix
        q = np.array(getmatrix(q, (None, self.n)))
        w = np.zeros(q.shape[0])

        for k, qk in enumerate(q):
            Jk = self.jacob0(qk)
            w[k] = mfunc(self, Jk, qk, axes_list)

        if len(w) == 1:
            return w[0]
        else:
            return w

    def partial_fkine0(self, q: ArrayLike, n: int) -> NDArray:
        r"""
        Manipulator Forward Kinematics nth Partial Derivative

        :param q: joint angles/configuration of the robot
        :param n: the order of derivative (must be >= 3)
        :returns: The nth partial derivative of the forward kinematics

        This method computes the nth derivative of the forward kinematics where ``n`` is
        greater than or equal to 3. This is an extension of the differential kinematics
        where the Jacobian is the first partial derivative and the Hessian is the
        second.

        Examples
        --------

        The following example makes a ``Panda`` robot object, and solves for the
        base-effector frame 4th derivative of the forward kinematics at the given
        joint angle configuration

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> panda.partial_fkine0([0, -0.3, 0, -2.2, 0, 2, 0.7854], n=4)

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        """

        # Calculate the Jacobian and Hessian
        J = self.jacob0(q)
        H = self.hessian0(q)

        # A list of derivatives, starting with the jacobian and hessian
        dT = [J, H]

        # The tensor dimensions of the latest derivative
        # Set to the current size of the Hessian
        size = [self.n, 6, self.n]

        # An array which keeps track of the index of the partial derivative
        # we are calculating
        # It stores the indices in the order: "j, k, l. m, n, o, ..."
        # where count is extended to match oder of the partial derivative
        count = np.array([0, 0])

        # The order of derivative for which we are calculating
        # The Hessian is the 2nd-order so we start with c = 2
        c = 2

        def add_indices(indices, c):
            total = len(indices * 2)
            new_indices = []

            for i in range(total):
                j = i // 2
                new_indices.append([])
                new_indices[i].append(indices[j][0].copy())
                new_indices[i].append(indices[j][1].copy())

                if i % 2 == 0:
                    # if even number
                    new_indices[i][0].append(c)
                else:
                    # if odd number
                    new_indices[i][1].append(c)

            return new_indices

        def add_pdi(pdi):
            total = len(pdi * 2)
            new_pdi = []

            for i in range(total):
                j = i // 2
                new_pdi.append([])
                new_pdi[i].append(pdi[j][0])
                new_pdi[i].append(pdi[j][1])

                # if even number
                if i % 2 == 0:
                    new_pdi[i][0] += 1
                # if odd number
                else:
                    new_pdi[i][1] += 1

            return new_pdi

        # these are the indices used for the hessian
        indices = [[[1], [0]]]

        # The partial derivative indices (pdi)
        # the are the pd indices used in the cross products
        pdi = [[0, 0]]

        # The length of dT correspods to the number of derivatives we have calculated
        while len(dT) != n:
            # Add to the start of the tensor size list
            size.insert(0, self.n)

            # Add an axis to the count array
            count = np.concatenate(([0], count))

            # This variables corresponds to indices within the previous
            # partial derivatives
            # to be cross prodded
            # The order is: "[j, k, l, m, n, o, ...]"
            # Although, our partial derivatives have the order:
            # pd[..., o, n, m, l, k, cartesian DoF, j]
            # For example, consider the Hessian Tensor H[n, 6, n],
            # the index H[k, :, j]. This corrsponds
            # to the second partial derivative of the kinematics of joint j with
            # respect to joint k.
            indices = add_indices(indices, c)

            # This variable corresponds to the indices in Td which corresponds to the
            # partial derivatives we need to use
            pdi = add_pdi(pdi)

            c += 1

            # Allocate our new partial derivative tensor
            pd = np.zeros(size)

            # We need to loop n^c times
            # There are n^c columns to calculate
            for _ in range(self.n**c):
                # Allocate the rotation and translation components
                rot = np.zeros(3)
                trn = np.zeros(3)

                # This loop calculates a single column ([trn, rot])
                # of the tensor for dT(x)
                for j in range(len(indices)):
                    pdr0 = dT[pdi[j][0]]
                    pdr1 = dT[pdi[j][1]]

                    idx0 = count[indices[j][0]]
                    idx1 = count[indices[j][1]]

                    # This is a list of indices selecting the slices of the
                    # previous tensor
                    idx0_slices = np.flip(idx0[1:])
                    idx1_slices = np.flip(idx1[1:])

                    # This index selecting the column within the 2d slice of the
                    # previous tensor
                    idx0_n = idx0[0]
                    idx1_n = idx1[0]

                    # Use our indices to select the rotational column from pdr0 and pdr1
                    col0_rot = pdr0[(*idx0_slices, slice(3, 6), idx0_n)]
                    col1_rot = pdr1[(*idx1_slices, slice(3, 6), idx1_n)]

                    # Use our indices to select the translational column from pdr1
                    col1_trn = pdr1[(*idx1_slices, slice(0, 3), idx1_n)]

                    # Perform the cross product as described in the maths above
                    rot += np.cross(col0_rot, col1_rot)
                    trn += np.cross(col0_rot, col1_trn)

                pd[(*np.flip(count[1:]), slice(0, 3), count[0])] = trn
                pd[(*np.flip(count[1:]), slice(3, 6), count[0])] = rot

                count[0] += 1
                for j in range(len(count)):
                    if count[j] == self.n:
                        count[j] = 0
                        if j != len(count) - 1:
                            count[j + 1] += 1

            dT.append(pd)

        return dT[-1]

    def ik_LM(
        self,
        Tep: NDArray | SE3,
        q0: NDArray | None = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: NDArray | None = None,
        joint_limits: bool = True,
        k: float = 1.0,
        method: L["chan", "wampler", "sugihara"] = "chan",
    ) -> tuple[NDArray, int, int, int, float]:
        r"""
        Fast Levenberg-Marquardt numerical inverse kinematics solver

        :param Tep: the desired end-effector pose
        :param q0: the initial joint coordinate vector
        :param ilimit: maximum iterations allowed per search
        :param slimit: maximum search attempts before failure
        :param tol: maximum allowed residual error E
        :param mask: a 6-vector weighting Cartesian DoF error priority
        :param joint_limits: reject solutions with joint limit violations
        :param k: gain value for the damping matrix Wn
        :param method: one of ``"chan"`` (default), ``"sugihara"`` or ``"wampler"``
        :returns: tuple (q, success, iterations, searches, residual)
        :rtype: tuple

        A method which provides functionality to perform numerical inverse kinematics (IK)
        using the Levenberg-Marquardt method. This is a fast solver implemented in C++.

        See the :ref:`Inverse Kinematics Docs Page <IK>` for more details and for a
        **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

        The operation is defined by the choice of the ``method`` kwarg.

        The step is deined as

        .. math::

            \vec{q}_{k+1}
            &=
            \vec{q}_k +
            \left(
                \mat{A}_k
            \right)^{-1}
            \bf{g}_k \\
            %
            \mat{A}_k
            &=
            {\mat{J}(\vec{q}_k)}^\top
            \mat{W}_e \
            {\mat{J}(\vec{q}_k)}
            +
            \mat{W}_n

        where :math:`\mat{W}_n = \text{diag}(\vec{w_n})(\vec{w_n} \in \mathbb{R}^n_{>0})` is a
        diagonal damping matrix. The damping matrix ensures that :math:`\mat{A}_k` is
        non-singular and positive definite. The performance of the LM method largely depends
        on the choice of :math:`\mat{W}_n`.

        *Chan's Method*

        Chan proposed

        .. math::

            \mat{W}_n
            =
            λ E_k \mat{1}_n

        where λ is a constant which reportedly does not have much influence on performance.
        Use the kwarg `k` to adjust the weighting term λ.

        *Sugihara's Method*

        Sugihara proposed

        .. math::

            \mat{W}_n
            =
            E_k \mat{1}_n + \text{diag}(\hat{\vec{w}}_n)

        where :math:`\hat{\vec{w}}_n \in \mathbb{R}^n`, :math:`\hat{w}_{n_i} = l^2 \sim 0.01 l^2`,
        and :math:`l` is the length of a typical link within the manipulator. We provide the
        variable `k` as a kwarg to adjust the value of :math:`w_n`.

        *Wampler's Method*

        Wampler proposed :math:`\vec{w_n}` to be a constant. This is set through the `k` kwarg.

        Examples
        --------

        The following example gets the ``ets`` of a ``panda`` robot object, makes a goal
        pose ``Tep``, and then solves for the joint coordinates which result in the pose
        ``Tep`` using the `ikine_LM` method.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> panda.ikine_LM(Tep)

        .. rubric:: Notes

        The value for the ``k`` kwarg will depend on the ``method`` chosen and the arm you are
        using. Use the following as a rough guide ``chan, k = 1.0 - 0.01``,
        ``wampler, k = 0.01 - 0.0001``, and ``sugihara, k = 0.1 - 0.0001``

        When using this method, the initial joint coordinates :math:`q_0`, should correspond
        to a non-singular manipulator pose, since it uses the manipulator Jacobian.

        This class supports null-space motion to assist with maximising manipulability and
        avoiding joint limits. These are enabled by setting kq and km to non-zero values.

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        .. seealso:: :meth:`ik_NR` :meth:`ik_GN`

        .. versionchanged:: 1.0.4
            Merged the Levenberg-Marquardt IK solvers into the ik_LM method

        """

        return IK_LM_c(
            self._fknm, Tep, q0, ilimit, slimit, tol, joint_limits, mask, k, method
        )

    def ik_NR(
        self,
        Tep: NDArray | SE3,
        q0: NDArray | None = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: NDArray | None = None,
        joint_limits: bool = True,
        pinv: int = True,
        pinv_damping: float = 0.0,
    ) -> tuple[NDArray, int, int, int, float]:
        r"""
        Fast numerical inverse kinematics using Newton-Raphson optimisation

        :param Tep: the desired end-effector pose or pose trajectory
        :param q0: initial joint configuration (random valid configuration if not supplied)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param mask: a 6-vector weighting end-effector error priority (XYZ translation, XYZ rotation)
        :param joint_limits: reject solutions with invalid joint configurations
        :param pinv: use the pseudo-inverse instead of the normal matrix inverse
        :param pinv_damping: damping factor for the pseudo-inverse
        :returns: tuple (q, success, iterations, searches, residual)
        :rtype: tuple

        ``sol = ets.ik_NR(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom. This
        is a fast solver implemented in C++.

        See the :ref:`Inverse Kinematics Docs Page <IK>` for more details and for a
        **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

        When using this method with redundant robots (>6 DoF), ``pinv`` must be set to ``True``.

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        Each iteration uses the Newton-Raphson optimisation method

        .. math::

            \vec{q}_{k+1} = \vec{q}_k + {^0\mat{J}(\vec{q}_k)}^{-1} \vec{e}_k

        Examples
        --------

        The following example gets the ``ets`` of a ``panda`` robot object, makes a goal
        pose ``Tep``, and then solves for the joint coordinates which result in the pose
        ``Tep`` using the `ik_NR` method.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> panda.ik_NR(Tep)

        .. rubric:: Notes

        When using this method, the initial joint coordinates :math:`q_0`, should correspond
        to a non-singular manipulator pose, since it uses the manipulator Jacobian.

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        .. seealso:: :meth:`ik_LM` :meth:`ik_GN`

        """

        return IK_NR_c(
            self._fknm,
            Tep,
            q0,
            ilimit,
            slimit,
            tol,
            joint_limits,
            mask,
            pinv,
            pinv_damping,
        )

    def ik_GN(
        self,
        Tep: NDArray | SE3,
        q0: NDArray | None = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: NDArray | None = None,
        joint_limits: bool = True,
        pinv: int = True,
        pinv_damping: float = 0.0,
    ) -> tuple[NDArray, int, int, int, float]:
        r"""
        Fast numerical inverse kinematics by Gauss-Newton optimisation

        :param Tep: the desired end-effector pose or pose trajectory
        :param q0: initial joint configuration (random valid configuration if not supplied)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param mask: a 6-vector weighting end-effector error priority (XYZ translation, XYZ rotation)
        :param joint_limits: reject solutions with invalid joint configurations
        :param pinv: use the pseudo-inverse instead of the normal matrix inverse
        :param pinv_damping: damping factor for the pseudo-inverse
        :returns: tuple (q, success, iterations, searches, residual)
        :rtype: tuple

        ``sol = ets.ik_GN(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom. This
        is a fast solver implemented in C++.

        See the :ref:`Inverse Kinematics Docs Page <IK>` for more details and for a
        **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

        When using this method with redundant robots (>6 DoF), ``pinv`` must be set to ``True``.

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        Each iteration uses the Gauss-Newton optimisation method

        .. math::

            \vec{q}_{k+1} &= \vec{q}_k +
            \left(
            {\mat{J}(\vec{q}_k)}^\top
            \mat{W}_e \
            {\mat{J}(\vec{q}_k)}
            \right)^{-1}
            \bf{g}_k \\
            \bf{g}_k &=
            {\mat{J}(\vec{q}_k)}^\top
            \mat{W}_e
            \vec{e}_k

        where :math:`\mat{J} = {^0\mat{J}}` is the base-frame manipulator Jacobian. If
        :math:`\mat{J}(\vec{q}_k)` is non-singular, and :math:`\mat{W}_e = \mat{1}_n`, then
        the above provides the pseudoinverse solution. However, if :math:`\mat{J}(\vec{q}_k)`
        is singular, the above can not be computed and the GN solution is infeasible.

        Examples
        --------

        The following example gets the ``ets`` of a ``panda`` robot object, makes a goal
        pose ``Tep``, and then solves for the joint coordinates which result in the pose
        ``Tep`` using the `ikine_GN` method.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> panda.ik_GN(Tep)

        .. rubric:: Notes

        When using this method, the initial joint coordinates :math:`q_0`, should correspond
        to a non-singular manipulator pose, since it uses the manipulator Jacobian.

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        .. seealso:: :meth:`ik_LM` :meth:`ik_NR`

        """

        return IK_GN_c(
            self._fknm,
            Tep,
            q0,
            ilimit,
            slimit,
            tol,
            joint_limits,
            mask,
            pinv,
            pinv_damping,
        )

    def ikine_LM(
        self,
        Tep: NDArray | SE3,
        q0: ArrayLike | None = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: ArrayLike | None = None,
        joint_limits: bool = True,
        seed: int | None = None,
        k: float = 1.0,
        method: L["chan", "wampler", "sugihara"] = "chan",
        kq: float = 0.0,
        km: float = 0.0,
        ps: float = 0.0,
        pi: NDArray | float = 0.3,
        **kwargs,
    ):
        r"""
        Levenberg-Marquardt numerical inverse kinematics solver

        :param Tep: the desired end-effector pose
        :param q0: the initial joint coordinate vector
        :param ilimit: maximum iterations allowed per search
        :param slimit: maximum search attempts before failure
        :param tol: maximum allowed residual error E
        :param mask: a 6-vector weighting Cartesian DoF error priority
        :param joint_limits: reject solutions with joint limit violations
        :param seed: seed for the RNG used to generate random joint configurations
        :param k: gain value for the damping matrix Wn
        :param method: one of ``"chan"`` (default), ``"sugihara"`` or ``"wampler"``
        :param kq: gain for joint limit avoidance (0.0 disables)
        :param km: gain for manipulability maximisation (0.0 disables)
        :param ps: minimum joint approach distance to limit (radians or metres)
        :param pi: null-space influence distance (radians or metres)
        :returns: IK solution

        A method which provides functionality to perform numerical inverse kinematics (IK)
        using the Levenberg-Marquardt method.

        See the :ref:`Inverse Kinematics Docs Page <IK>` for more details and for a
        **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

        The operation is defined by the choice of the ``method`` kwarg.

        The step is defined as

        .. math::

            \vec{q}_{k+1} 
            &= 
            \vec{q}_k +
            \left(
                \mat{A}_k
            \right)^{-1}
            \bf{g}_k \\
            %
            \mat{A}_k
            &=
            {\mat{J}(\vec{q}_k)}^\top
            \mat{W}_e \
            {\mat{J}(\vec{q}_k)}
            +
            \mat{W}_n

        where :math:`\mat{W}_n = \text{diag}(\vec{w_n})(\vec{w_n} \in \mathbb{R}^n_{>0})` is a
        diagonal damping matrix. The damping matrix ensures that :math:`\mat{A}_k` is
        non-singular and positive definite. The performance of the LM method largely depends
        on the choice of :math:`\mat{W}_n`.

        *Chan's Method*

        Chan proposed

        .. math::

            \mat{W}_n
            =
            λ E_k \mat{1}_n

        where λ is a constant which reportedly does not have much influence on performance.
        Use the kwarg `k` to adjust the weighting term λ.

        *Sugihara's Method*

        Sugihara proposed

        .. math::

            \mat{W}_n
            =
            E_k \mat{1}_n + \text{diag}(\hat{\vec{w}}_n)

        where :math:`\hat{\vec{w}}_n \in \mathbb{R}^n`, :math:`\hat{w}_{n_i} = l^2 \sim 0.01 l^2`,
        and :math:`l` is the length of a typical link within the manipulator. We provide the
        variable `k` as a kwarg to adjust the value of :math:`w_n`.

        *Wampler's Method*

        Wampler proposed :math:`\vec{w_n}` to be a constant. This is set through the `k` kwarg.

        Examples
        --------

        The following example gets the ``ets`` of a ``panda`` robot object, makes a goal
        pose ``Tep``, and then solves for the joint coordinates which result in the pose
        ``Tep`` using the `ikine_LM` method.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> panda.ikine_LM(Tep)

        .. rubric:: Notes

        The value for the ``k`` kwarg will depend on the ``method`` chosen and the arm you are
        using. Use the following as a rough guide ``chan, k = 1.0 - 0.01``,
        ``wampler, k = 0.01 - 0.0001``, and ``sugihara, k = 0.1 - 0.0001``

        When using this method, the initial joint coordinates :math:`q_0`, should correspond
        to a non-singular manipulator pose, since it uses the manipulator Jacobian.

        This class supports null-space motion to assist with maximising manipulability and
        avoiding joint limits. These are enabled by setting kq and km to non-zero values.

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        .. seealso:: :meth:`ikine_NR` :meth:`ikine_GN` :meth:`ikine_QP`

        .. versionchanged:: 1.0.4
            Added the Levenberg-Marquardt IK solver method on the `ETS` class

        """

        solver = IK_LM(
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            joint_limits=joint_limits,
            mask=mask,
            seed=seed,
            k=k,
            method=method,
            kq=kq,
            km=km,
            ps=ps,
            pi=pi,
            **kwargs,
        )

        # if isinstance(Tep, SE3):
        #     Tep = Tep.A

        return solver.solve(ets=self, Tep=Tep, q0=q0)

    def ikine_NR(
        self,
        Tep: NDArray | SE3,
        q0: ArrayLike | None = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: ArrayLike | None = None,
        joint_limits: bool = True,
        seed: int | None = None,
        pinv: bool = False,
        kq: float = 0.0,
        km: float = 0.0,
        ps: float = 0.0,
        pi: NDArray | float = 0.3,
        **kwargs,
    ):
        r"""
        Newton-Raphson numerical inverse kinematics solver

        :param Tep: the desired end-effector pose
        :param q0: the initial joint coordinate vector
        :param ilimit: maximum iterations allowed per search
        :param slimit: maximum search attempts before failure
        :param tol: maximum allowed residual error E
        :param mask: a 6-vector weighting Cartesian DoF error priority
        :param joint_limits: reject solutions with joint limit violations
        :param seed: seed for the RNG used to generate random joint configurations
        :param pinv: use the pseudo-inverse in the step method instead of the normal inverse
        :param kq: gain for joint limit avoidance (0.0 disables)
        :param km: gain for manipulability maximisation (0.0 disables)
        :param ps: minimum joint approach distance to limit (radians or metres)
        :param pi: null-space influence distance (radians or metres)
        :returns: IK solution

        A method which provides functionality to perform numerical inverse kinematics (IK)
        using the Newton-Raphson method.

        See the :ref:`Inverse Kinematics Docs Page <IK>` for more details and for a
        **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

        When using this method with redundant robots (>6 DoF), ``pinv`` must be set to ``True``.

        Each iteration uses the Newton-Raphson optimisation method

        .. math::

            \vec{q}_{k+1} = \vec{q}_k + {^0\mat{J}(\vec{q}_k)}^{-1} \vec{e}_k

        Examples
        --------

        The following example gets the ``ets`` of a ``panda`` robot object, makes a goal
        pose ``Tep``, and then solves for the joint coordinates which result in the pose
        ``Tep`` using the `ikine_NR` method.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> panda.ikine_NR(Tep)

        .. rubric:: Notes

        When using this method, the initial joint coordinates :math:`q_0`, should correspond
        to a non-singular manipulator pose, since it uses the manipulator Jacobian.

        This class supports null-space motion to assist with maximising manipulability and
        avoiding joint limits. These are enabled by setting kq and km to non-zero values.

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        .. seealso:: :meth:`ikine_LM` :meth:`ikine_GN` :meth:`ikine_QP`

        .. versionchanged:: 1.0.4
            Added the Newton-Raphson IK solver method on the `ETS` class

        """

        solver = IK_NR(
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            joint_limits=joint_limits,
            mask=mask,
            seed=seed,
            pinv=pinv,
            kq=kq,
            km=km,
            ps=ps,
            pi=pi,
            **kwargs,
        )

        # if isinstance(Tep, SE3):
        #     Tep = Tep.A

        return solver.solve(ets=self, Tep=Tep, q0=q0)

    def ikine_GN(
        self,
        Tep: NDArray | SE3,
        q0: ArrayLike | None = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: ArrayLike | None = None,
        joint_limits: bool = True,
        seed: int | None = None,
        pinv: bool = False,
        kq: float = 0.0,
        km: float = 0.0,
        ps: float = 0.0,
        pi: NDArray | float = 0.3,
        **kwargs,
    ):
        r"""
        Gauss-Newton numerical inverse kinematics solver

        :param Tep: the desired end-effector pose
        :param q0: the initial joint coordinate vector
        :param ilimit: maximum iterations allowed per search
        :param slimit: maximum search attempts before failure
        :param tol: maximum allowed residual error E
        :param mask: a 6-vector weighting Cartesian DoF error priority
        :param joint_limits: reject solutions with joint limit violations
        :param seed: seed for the RNG used to generate random joint configurations
        :param pinv: use the pseudo-inverse in the step method instead of the normal inverse
        :param kq: gain for joint limit avoidance (0.0 disables)
        :param km: gain for manipulability maximisation (0.0 disables)
        :param ps: minimum joint approach distance to limit (radians or metres)
        :param pi: null-space influence distance (radians or metres)
        :returns: IK solution

        A method which provides functionality to perform numerical inverse kinematics (IK)
        using the Gauss-Newton method.

        See the :ref:`Inverse Kinematics Docs Page <IK>` for more details and for a
        **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

        When using this method with redundant robots (>6 DoF), ``pinv`` must be set to ``True``.

        Each iteration uses the Gauss-Newton optimisation method

        .. math::

            \vec{q}_{k+1} &= \vec{q}_k +
            \left(
            {\mat{J}(\vec{q}_k)}^\top
            \mat{W}_e \
            {\mat{J}(\vec{q}_k)}
            \right)^{-1}
            \bf{g}_k \\
            \bf{g}_k &=
            {\mat{J}(\vec{q}_k)}^\top
            \mat{W}_e
            \vec{e}_k

        where :math:`\mat{J} = {^0\mat{J}}` is the base-frame manipulator Jacobian. If
        :math:`\mat{J}(\vec{q}_k)` is non-singular, and :math:`\mat{W}_e = \mat{1}_n`, then
        the above provides the pseudoinverse solution. However, if :math:`\mat{J}(\vec{q}_k)`
        is singular, the above can not be computed and the GN solution is infeasible.

        Examples
        --------

        The following example gets the ``ets`` of a ``panda`` robot object, makes a goal
        pose ``Tep``, and then solves for the joint coordinates which result in the pose
        ``Tep`` using the `ikine_GN` method.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> panda.ikine_GN(Tep)

        .. rubric:: Notes

        When using this method, the initial joint coordinates :math:`q_0`, should correspond
        to a non-singular manipulator pose, since it uses the manipulator Jacobian.

        This class supports null-space motion to assist with maximising manipulability and
        avoiding joint limits. These are enabled by setting kq and km to non-zero values.

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        .. seealso:: :meth:`ikine_LM` :meth:`ikine_NR` :meth:`ikine_QP`

        .. versionchanged:: 1.0.4
            Added the Gauss-Newton IK solver method on the `ETS` class

        """

        solver = IK_GN(
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            joint_limits=joint_limits,
            mask=mask,
            seed=seed,
            pinv=pinv,
            kq=kq,
            km=km,
            ps=ps,
            pi=pi,
            **kwargs,
        )

        # if isinstance(Tep, SE3):
        #     Tep = Tep.A

        return solver.solve(ets=self, Tep=Tep, q0=q0)

    def ikine_QP(
        self,
        Tep: NDArray | SE3,
        q0: ArrayLike | None = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: ArrayLike | None = None,
        joint_limits: bool = True,
        seed: int | None = None,
        kj=1.0,
        ks=1.0,
        kq: float = 0.0,
        km: float = 0.0,
        ps: float = 0.0,
        pi: NDArray | float = 0.3,
        **kwargs,
    ):
        r"""
        Quadratic programming numerical inverse kinematics solver

        :param Tep: the desired end-effector pose
        :param q0: the initial joint coordinate vector
        :param ilimit: maximum iterations allowed per search
        :param slimit: maximum search attempts before failure
        :param tol: maximum allowed residual error E
        :param mask: a 6-vector weighting Cartesian DoF error priority
        :param joint_limits: reject solutions with joint limit violations
        :param seed: seed for the RNG used to generate random joint configurations
        :param kj: gain for joint velocity norm minimisation
        :param ks: gain adjusting the cost of slack (intentional error)
        :param kq: gain for joint limit avoidance (0.0 disables)
        :param km: gain for manipulability maximisation (0.0 disables)
        :param ps: minimum joint approach distance to limit (radians or metres)
        :param pi: null-space influence distance (radians or metres)
        :returns: IK solution
        :raises ImportError: if the package ``qpsolvers`` is not installed

        A method that provides functionality to perform numerical inverse kinematics
        (IK) using a quadratic programming approach.

        See the :ref:`Inverse Kinematics Docs Page <IK>` for more details and for a
        **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

        Each iteration uses the following approach

        .. math::

            \vec{q}_{k+1} = \vec{q}_{k} + \dot{\vec{q}}.

        where the QP is defined as

        .. math::

            \min_x \quad f_o(\vec{x}) &= \frac{1}{2} \vec{x}^\top \mathcal{Q} \vec{x}+ \mathcal{C}^\top \vec{x}, \\
            \text{subject to} \quad \mathcal{J} \vec{x} &= \vec{\nu},  \\
            \mathcal{A} \vec{x} &\leq \mathcal{B},  \\
            \vec{x}^- &\leq \vec{x} \leq \vec{x}^+

        with

        .. math::

            \vec{x} &=
            \begin{pmatrix}
                \dvec{q} \\ \vec{\delta}
            \end{pmatrix} \in \mathbb{R}^{(n+6)}  \\
            \mathcal{Q} &=
            \begin{pmatrix}
                \lambda_q \mat{1}_{n} & \mathbf{0}_{6 \times 6} \\ \mathbf{0}_{n \times n} & \lambda_\delta \mat{1}_{6}
            \end{pmatrix} \in \mathbb{R}^{(n+6) \times (n+6)} \\
            \mathcal{J} &=
            \begin{pmatrix}
                \mat{J}(\vec{q}) & \mat{1}_{6}
            \end{pmatrix} \in \mathbb{R}^{6 \times (n+6)} \\
            \mathcal{C} &=
            \begin{pmatrix}
                \mat{J}_m \\ \bf{0}_{6 \times 1}
            \end{pmatrix} \in \mathbb{R}^{(n + 6)} \\
            \mathcal{A} &=
            \begin{pmatrix}
                \mat{1}_{n \times n + 6} \\
            \end{pmatrix} \in \mathbb{R}^{(l + n) \times (n + 6)} \\
            \mathcal{B} &=
            \eta
            \begin{pmatrix}
                \frac{\rho_0 - \rho_s}
                        {\rho_i - \rho_s} \\
                \vdots \\
                \frac{\rho_n - \rho_s}
                        {\rho_i - \rho_s}
            \end{pmatrix} \in \mathbb{R}^{n} \\
            \vec{x}^{-, +} &=
            \begin{pmatrix}
                \dvec{q}^{-, +} \\
                \vec{\delta}^{-, +}
            \end{pmatrix} \in \mathbb{R}^{(n+6)},

        where :math:`\vec{\delta} \in \mathbb{R}^6` is the slack vector,
        :math:`\lambda_\delta \in \mathbb{R}^+` is a gain term which adjusts the
        cost of the norm of the slack vector in the optimiser,
        :math:`\dvec{q}^{-,+}` are the minimum and maximum joint velocities, and
        :math:`\dvec{\delta}^{-,+}` are the minimum and maximum slack velocities.

        Examples
        --------

        The following example gets the ``ets`` of a ``panda`` robot object, makes a goal
        pose ``Tep``, and then solves for the joint coordinates which result in the pose
        ``Tep`` using the `ikine_QP` method.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda().ets()
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> panda.ikine_QP(Tep)

        .. rubric:: Notes

        When using this method, the initial joint coordinates :math:`q_0`, should correspond
        to a non-singular manipulator pose, since it uses the manipulator Jacobian.

        This class supports null-space motion to assist with maximising manipulability and
        avoiding joint limits. These are enabled by setting kq and km to non-zero values.

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        .. seealso:: :meth:`ikine_LM` :meth:`ikine_NR` :meth:`ikine_GN`

        .. versionchanged:: 1.0.4
            Added the Quadratic Programming IK solver method on the `ETS` class

        """

        solver = IK_QP(
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            joint_limits=joint_limits,
            mask=mask,
            seed=seed,
            kj=kj,
            ks=ks,
            kq=kq,
            km=km,
            ps=ps,
            pi=pi,
            **kwargs,
        )

        # if isinstance(Tep, SE3):
        #     Tep = Tep.A

        return solver.solve(ets=self, Tep=Tep, q0=q0)


class ETS2(BaseETS):
    """
    This class implements an elementary transform sequence (ETS) for 2D

    :param arg: Function to compute ET value

    An instance can contain an elementary transform (ET) or an elementary
    transform sequence (ETS). It has list-like properties by subclassing
    UserList, which means we can perform indexing, slicing pop, insert, as well
    as using it as an iterator over its values.

    - ``ETS()`` an empty ETS list
    - ``ET2.XY(η)`` is a constant elementary transform
    - ``ET2.XY(η, 'deg')`` as above but the angle is expressed in degrees
    - ``ET2.XY()`` is a joint variable, the value is left free until evaluation
      time
    - ``ET2.XY(j=J)`` as above but the joint index is explicitly given, this
      might correspond to the joint number of a multi-joint robot.
    - ``ET2.XY(flip=True)`` as above but the joint moves in the opposite sense

    where ``XY`` is one of ``R``, ``tx``, ``ty``.

    Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS2 as ET2
            >>> e = ET2.R(0.3)  # a single ET, rotation about z
            >>> len(e)
            >>> e = ET2.R(0.3) * ET2.tx(2)  # an ETS
            >>> len(e)                      # of length 2
            >>> e[1]                        # an ET sliced from the ETS

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    :seealso: :func:`r`, :func:`tx`, :func:`ty`
    """

    def __init__(
        self,
        arg: list[ETS2 | ET2] | list[ET2] | list[ETS2] | ET2 | ETS2 | None = None,
    ):
        super().__init__()
        if isinstance(arg, list):
            for item in arg:
                if isinstance(item, ET2):
                    self._data.append(deepcopy(item))
                elif isinstance(item, ETS2):
                    for ets_item in item:
                        self._data.append(deepcopy(ets_item))
                else:
                    raise TypeError("bad arg")
        elif isinstance(arg, ET2):
            self._data.append(deepcopy(arg))
        elif isinstance(arg, ETS2):
            for ets_item in arg:
                self._data.append(deepcopy(ets_item))
        elif arg is not None:
            raise TypeError("bad arg")
        self._ndims = 2
        self._auto_jindex = False

        # Check if jindices are set
        joints = self.joints()

        # Number of joints with a jindex
        jindices = 0

        # Number of joints with a sequential jindex (j[2] -> jindex = 2)
        seq_jindex = 0

        # Count them up
        for j, joint in enumerate(joints):
            if joint.jindex is not None:
                jindices += 1
                if joint.jindex == j:
                    seq_jindex += 1

        if (
            jindices == self.n - 1
            and seq_jindex == self.n - 1
            and joints[-1].jindex is None
        ):
            # ets has sequential jindicies, except for the last.
            joints[-1].jindex = self.n - 1
            self._auto_jindex = True
        elif jindices > 0 and not jindices == self.n:
            raise ValueError(
                "You can not have some jindices set for the ET's in arg. It must be all"
                " or none"
            )  # pragma: nocover
        elif jindices == 0 and self.n > 0:
            # Set them ourself
            for j, joint in enumerate(joints):
                joint.jindex = j
            self._auto_jindex = True

    def __mul__(self, other: ET2 | ETS2) -> "ETS2":
        if isinstance(other, ET2):
            return ETS2([*self._data, other])
        else:
            return ETS2([*self._data, *other._data])  # pragma: nocover

    def __rmul__(self, other: ET2 | ETS2) -> "ETS2":
        return ETS2([other, *self._data])  # pragma: nocover

    def __imul__(self, rest: "ETS2"):
        return self + rest  # pragma: nocover

    def __add__(self, rest) -> "ETS2":
        return self.__mul__(rest)  # pragma: nocover

    def compile(self) -> "ETS2":
        """
        Compile an ETS2

        :return: optimised ETS2

        Perform constant folding for faster evaluation.  Consecutive constant
        ETs are compounded, leading to a constant ET which is denoted by
        ``SE3`` when displayed.

        :seealso: :func:`isconstant`
        """
        const = None
        ets = ETS2()

        for et in self:
            if et.isjoint:
                # a joint
                if const is not None:
                    # flush the constant
                    if not np.array_equal(const, np.eye(3)):
                        ets *= ET2.SE2(const)
                    const = None
                ets *= et  # emit the joint ET
            else:
                # not a joint
                if const is None:
                    const = et.A()
                else:
                    const = const @ et.A()

        if const is not None:
            # flush the constant, tool transform
            if not np.array_equal(const, np.eye(3)):
                ets *= ET2.SE2(const)
        return ets

    def insert(  # type: ignore[override]
        self,
        i: int,
        arg: ET2 | ETS2,
    ) -> None:
        """
        Insert value

        :param i: position to insert at
        :param arg: the elementary transform or sequence to insert

        Inserts an ET or ETS into the ET sequence.  The inserted value is at position
        ``i``.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET2
            >>> e = ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1)
            >>> f = ET2.R()
            >>> e.insert(2, f)
            >>> e
        """

        if isinstance(arg, ET2):
            self._data.insert(i, arg)
        elif isinstance(arg, ETS2):
            for j, et in enumerate(arg):
                self._data.insert(i + j, et)
        self._fknm_stale = True

    def fkine(
        self,
        q: ArrayLike,
        base: NDArray | SE2 | None = None,
        tool: NDArray | SE2 | None = None,
        include_base: bool = True,
    ) -> SE2:
        """
        Forward kinematics

        :param q: joint coordinates
        :param base: base transform, optional
        :param tool: tool transform, optional
        :returns: transformation matrix representing the end-effector pose
        :rtype: SE2

        ``T = ets.fkine(q)`` evaluates forward kinematics for the robot at
        joint configuration ``q``.

        **Trajectory operation**: If ``q`` has multiple rows (mxn), it is considered a
        trajectory and the result is an ``SE2`` instance with ``m`` values.

        .. note::

            - The robot's base tool transform, if set, is incorporated into the result.
            - A tool transform, if provided, is incorporated into the result.
            - Works from the end-effector link to the base

        .. rubric:: References

        - Kinematic Derivatives using the Elementary Transform Sequence, J. Haviland and P. Corke
        """

        ret = SE2.Empty()
        fk = self.eval(q, base, tool, include_base)

        if fk.dtype == "O":
            # symbolic
            fk = np.array(simplify(fk))

        if fk.ndim == 3:
            for T in fk:
                ret.append(SE2(T, check=False))  # type: ignore
        else:
            ret = SE2(fk, check=False)

        return ret

    def eval(
        self,
        q: ArrayLike,
        base: NDArray | SE2 | None = None,
        tool: NDArray | SE2 | None = None,
        include_base: bool = True,
    ) -> NDArray:
        """
        Forward kinematics (returns raw ndarray)

        :param q: joint coordinates
        :param base: base transform, optional
        :param tool: tool transform, optional
        :returns: transformation matrix representing the end-effector pose
        :rtype: ndarray(3,3) or ndarray(m,3,3)

        ``T = ets.eval(q)`` evaluates forward kinematics for the robot at
        joint configuration ``q``, returning the raw ndarray.

        **Trajectory operation**: If ``q`` has multiple rows (mxn), it is considered a
        trajectory and the result is an ndarray of shape (m,3,3).

        .. note::

            - The robot's base tool transform, if set, is incorporated into the result.
            - A tool transform, if provided, is incorporated into the result.
            - Works from the end-effector link to the base

        .. rubric:: References

        - Kinematic Derivatives using the Elementary Transform Sequence, J. Haviland and P. Corke
        """

        q = getmatrix(q, (None, None))
        l, _ = q.shape  # type: ignore
        end = self[-1]

        if base is None:
            bases = None
        elif isinstance(base, SE2):
            bases = np.array(base.A)
        elif np.array_equal(base, np.eye(3)):  # pragma: nocover
            bases = None
        else:  # pragma: nocover
            bases = base

        if tool is None:
            tools = None
        elif isinstance(tool, SE2):
            tools = np.array(tool.A)
        elif np.array_equal(tool, np.eye(3)):  # pragma: nocover
            tools = None
        else:  # pragma: nocover
            tools = tool

        if l > 1:
            T = np.zeros((l, 3, 3), dtype=object)
        else:
            T = np.zeros((3, 3), dtype=object)

        for k, qk in enumerate(q):  # type: ignore
            link = end  # start with last link

            jindex = 0 if link.jindex is None and link.isjoint else link.jindex
            Tk = link.A(qk[jindex])

            if tools is not None:
                Tk = Tk @ tools

            # add remaining links, back toward the base
            for i in range(self.m - 2, -1, -1):
                link = self._data[i]

                jindex = 0 if link.jindex is None and link.isjoint else link.jindex
                A = link.A(qk[jindex])

                if A is not None:
                    Tk = A @ Tk

            # add base transform if it is set
            if include_base is True and bases is not None:
                Tk = bases @ Tk

            # append
            if l > 1:
                T[k, :, :] = Tk
                # ret.append(SE2(Tk, check=False))  # type: ignore
            else:
                T = Tk
                # ret = SE2(Tk, check=False)

        return T

    def jacob0(
        self,
        q: ArrayLike,
    ) -> NDArray:
        # very inefficient implementation, just put a 1 in last row
        # if its a rotation joint
        q = getvector(q)

        j = 0
        J = np.zeros((3, self.n))
        etjoints = self.joint_idx()

        if not np.all(np.array([self[i].jindex for i in etjoints])):
            # not all joints have a jindex it is required, set them
            for j in range(self.n):
                i = etjoints[j]
                self[i].jindex = j

        for j in range(self.n):
            i = etjoints[j]

            if self[i].jindex is not None:
                jindex = self[i].jindex
            else:
                jindex = 0  # pragma: nocover

            # jindex = 0 if self[i].jindex is None else self[i].jindex

            axis = self[i].axis
            if axis == "R":
                dTdq = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ self[i].A(
                    q[jindex]  # type: ignore
                )
            elif axis == "tx":
                dTdq = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
            elif axis == "ty":
                dTdq = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
            else:  # pragma: nocover
                raise TypeError("Invalid axes")

            E0 = ETS2(self[:i])
            if len(E0) > 0:
                dTdq = E0.fkine(q).A @ dTdq

            Ef = ETS2(self[i + 1 :])
            if len(Ef) > 0:
                dTdq = dTdq @ Ef.fkine(q).A

            T = self.fkine(q).A
            dRdt = dTdq[:2, :2] @ T[:2, :2].T

            J[:2, j] = dTdq[:2, 2]
            J[2, j] = dRdt[1, 0]

        return J

    def jacobe(
        self,
        q: ArrayLike,
    ):
        r"""
        Jacobian in end-effector frame

        :param q: joint coordinates
        :returns: Jacobian matrix
        :rtype: ndarray(3,n)

        ``jacobe(q)`` is the manipulator Jacobian matrix which maps joint
        velocity to end-effector spatial velocity.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, \omega)^T`
        is related to joint velocity by :math:`{}^{e}\nu = {}^{e}\mathbf{J}_0(q) \dot{q}`.

        :seealso: :func:`jacob0`, :func:`hessian0`
        """

        T = self.fkine(q, include_base=False).A
        return tr2jac2(T.T) @ self.jacob0(q)
