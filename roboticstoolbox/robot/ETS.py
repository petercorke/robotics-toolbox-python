#!/usr/bin/env python3

"""
@author: Jesse Haviland
@author: Peter Corke
"""

from collections import UserList
from numpy import (
    pi,
    where,
    all,
    ndarray,
    zeros,
    array,
    eye,
    array_equal,
    sqrt,
    min,
    max,
    where,
    cross,
    flip,
    concatenate,
)
from numpy.random import uniform
from numpy.linalg import inv, det, cond, pinv, matrix_rank, svd, eig
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
)
from roboticstoolbox import rtb_get_param

from collections import UserList
from spatialmath.base import issymbol, getmatrix
from fknm import (
    ETS_init,
    ETS_fkine,
    ETS_jacob0,
    ETS_jacobe,
    ETS_hessian0,
    ETS_hessiane,
    IK_NR,
    IK_GN,
    IK_LM_Chan,
    IK_LM_Wampler,
    IK_LM_Sugihara,
)
from copy import deepcopy
from roboticstoolbox import rtb_get_param
from roboticstoolbox.robot.ET import ET, ET2
from spatialmath.base import getvector
from spatialmath import SE3
from typing import Union, overload, List, Set, Tuple
from sys import version_info

ArrayLike = Union[list, ndarray, tuple, set]

py_ver = version_info

if version_info >= (3, 9):
    from functools import cached_property

    c_property = cached_property
else:
    c_property = property


class BaseETS(UserList):
    def __init__(self, *args):
        super().__init__(*args)

    def _update_internals(self):
        self._m = len(self.data)
        self._n = len([True for et in self.data if et.isjoint])
        self._fknm = ETS_init(
            [et.fknm for et in self.data],
            self._n,
            self._m,
        )
        # self._fknm = [et.fknm for et in self.data]

    def __str__(self, q: Union[str, None] = None):
        """
        Pretty prints the ETS

        :param q: control how joint variables are displayed
        :type q: ArrayLike
        :return: Pretty printed ETS

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
            >>> e = ET.Rz(jindex=3) * ET.tx(1) * ET.Rz(jindex=4)
            >>> print(e)
            >>> print(e.__str__("θ{0}"))

        .. note:: Angular parameters are converted to degrees, except if they
            are symbolic.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> from spatialmath.base import symbol
            >>> theta, d = symbol('theta, d')
            >>> e = ET.Rx(theta) * ET.tx(2) * ET.Rx(45, 'deg') * ET.Ry(0.2) * ET.ty(d)
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
        for et in self.data:

            if et.isjoint:
                if q is not None:
                    if et.jindex is None:
                        _j = j
                    else:
                        _j = et.jindex
                    qvar = q.format(  # lgtm [py/str-format/surplus-argument]  # noqa
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
                    s = f"{et.axis}({et.eta * 180 / pi:.4g}°)"

            elif et.istranslation:
                try:
                    s = f"{et.axis}({et.eta:.4g})"
                except TypeError:
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

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print stringified version when variable is displayed in IPython, ie. on
        a line by itself.

        Example::

            [In [1]: e
            Out [1]: R(q0) ⊕ tx(1) ⊕ R(q1) ⊕ tx(1)
        """
        print(self.__str__())

    def joint_idx(self) -> List[int]:
        """
        Get index of joint transforms

        :return: indices of transforms that are joints

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.joint_idx()

        """
        return where([e.isjoint for e in self])[0]

    def joints(self) -> List[ET]:
        """
        Get a list of the variable ETs with this ETS

        :return: list of ETs that are joints

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.joints()

        """
        return [e for e in self if e.isjoint]

    def jindex_set(self) -> Set[int]:  #
        """
        Get set of joint indices

        :return: set of unique joint indices

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(jindex=1) * ET.tx(jindex=2) * ET.Rz(jindex=1) * ET.tx(1)
            >>> e.jointset()
        """
        return set([self[j].jindex for j in self.joint_idx()])  # type: ignore

    @c_property
    def jindices(self) -> ndarray:
        """
        Get an array of joint indices

        :return: array of unique joint indices

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(jindex=1) * ET.tx(jindex=2) * ET.Rz(jindex=1) * ET.tx(1)
            >>> e.jointset()
        """
        return array([j.jindex for j in self.joints()])  # type: ignore

    @c_property
    def qlim(self):
        r"""
        Joint limits

        :return: Array of joint limit values
        :rtype: ndarray(2,n)
        :exception ValueError: unset limits for a prismatic joint

        Limits are extracted from the link objects.  If joints limits are
        not set for:

            - a revolute joint [-𝜋. 𝜋] is returned
            - a prismatic joint an exception is raised

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.qlim
        """
        limits = zeros((2, self.n))

        for i, et in enumerate(self.joints()):
            if et.isrotation:
                if et.qlim is None:
                    v = [-pi, pi]
                else:
                    v = et.qlim
            elif et.istranslation:
                if et.qlim is None:
                    raise ValueError("undefined prismatic joint limit")
                else:
                    v = et.qlim
            else:
                raise ValueError("Undefined Joint Type")
            limits[:, i] = v

        return limits

    @property
    def structure(self) -> str:
        """
        Joint structure string

        :return: A string indicating the joint types

        A string comprising the characters 'R' or 'P' which indicate the types
        of joints in order from left to right.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.tz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e.structure

        """
        return "".join(
            ["R" if self.data[i].isrotation else "P" for i in self.joint_idx()]
        )

    @property
    def n(self) -> int:
        """
        Number of joints

        :return: the number of joints in the ETS

        Counts the number of joints in the ETS.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rx() * ET.tx(1) * ET.tz()
            >>> e.n

        :seealso: :func:`joints`
        """

        return self._n

    @property
    def m(self) -> int:
        """
        Number of transforms

        :return: the number of transforms in the ETS

        Counts the number of transforms in the ETS.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rx() * ET.tx(1) * ET.tz()
            >>> e.m

        """

        return self._m

    @overload
    def data(self: "ETS") -> List[ET]:  # pragma: nocover
        ...

    @overload
    def data(self: "ETS2") -> List[ET2]:  # pragma: nocover
        ...

    @property
    def data(self):
        return self._data

    @data.setter
    @overload
    def data(self: "ETS", new_data: List[ET]):  # pragma: nocover
        ...

    @data.setter
    @overload
    def data(self: "ETS", new_data: List[ET2]):  # pragma: nocover
        ...

    @data.setter
    def data(self, new_data):
        self._data = new_data

    @overload
    def pop(self: "ETS", i: int = -1) -> ET:  # pragma: nocover
        ...

    @overload
    def pop(self: "ETS2", i: int = -1) -> ET2:  # pragma: nocover
        ...

    def pop(self, i=-1):
        """
        Pop value

        :param i: item in the list to pop, default is last
        :return: the popped value
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
        self._update_internals()
        return item

    @overload
    def split(self: "ETS") -> List["ETS"]:  # pragma: nocover
        ...

    @overload
    def split(self: "ETS2") -> List["ETS2"]:  # pragma: nocover
        ...

    def split(self):
        """
        Split ETS into link segments

        Returns a list of ETS, each one, apart from the last,
        ends with a variable ET.
        """
        segments = []
        start = 0

        for j, k in enumerate(self.joint_idx()):
            ets_j = self.data[start : k + 1]
            start = k + 1
            segments.append(ets_j)

        tail = self.data[start:]

        if isinstance(tail, list):
            tail_len = len(tail)
        elif tail is not None:  # pragma: nocover
            tail_len = 1
        else:  # pragma: nocover
            tail_len = 0

        if tail_len > 0:
            segments.append(tail)

        return segments

    @overload
    def inv(self: "ETS") -> "ETS":  # pragma: nocover
        ...

    @overload
    def inv(self: "ETS2") -> "ETS2":  # pragma: nocover
        ...

    def inv(self):
        r"""
        Inverse of ETS

        :return: Inverse of the ETS

        The inverse of a given ETS.  It is computed as the inverse of the
        individual ETs in the reverse order.

        .. math::

            (\mathbf{E}_0, \mathbf{E}_1 \cdots \mathbf{E}_{n-1} )^{-1} = (\mathbf{E}_{n-1}^{-1}, \mathbf{E}_{n-2}^{-1} \cdots \mathbf{E}_0^{-1}{n-1} )

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz(jindex=2) * ET.tx(1) * ET.Rx(jindex=3,flip=True) * ET.tx(1)
            >>> print(e)
            >>> print(e.inv())

        .. note:: It is essential to use explicit joint indices to account for
            the reversed order of the transforms.
        """  # noqa

        return self.__class__([et.inv() for et in reversed(self.data)])

    @overload
    def __getitem__(self: "ETS", i: int) -> ET:  # pragma: nocover
        ...

    @overload
    def __getitem__(self: "ETS", i: slice) -> List[ET]:  # pragma: nocover
        ...

    @overload
    def __getitem__(self: "ETS2", i: int) -> ET2:  # pragma: nocover
        ...

    @overload
    def __getitem__(self: "ETS2", i: slice) -> List[ET2]:  # pragma: nocover
        ...

    def __getitem__(self, i):
        """
        Index or slice an ETS

        :param i: the index or slince
        :return: Elementary transform

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> e[0]
            >>> e[1]
            >>> e[1:3]

        """
        return self.data[i]  # can be [2] or slice, eg. [3:5]

    def __deepcopy__(self, memo):

        new_data = []

        for data in self:
            new_data.append(deepcopy(data))

        cls = self.__class__
        result = cls(new_data)
        memo[id(self)] = result
        return result

    def plot(self, *args, **kwargs):
        from roboticstoolbox.robot.ERobot import ERobot, ERobot2

        if isinstance(self, ETS):
            robot = ERobot(self)
        else:
            robot = ERobot2(self)

        robot.plot(*args, **kwargs)

    def teach(self, *args, **kwargs):
        from roboticstoolbox.robot.ERobot import ERobot, ERobot2

        if isinstance(self, ETS):
            robot = ERobot(self)
        else:
            robot = ERobot2(self)

        robot.teach(*args, **kwargs)

    def random_q(self, i: int = 1) -> ndarray:
        """
        Generate a random valid joint configuration

        :param i: number of configurations to generate

        Generates a random q vector within the joint limits defined by
        `self.qlim`.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.Panda()
            >>> ets = robot.ets()
            >>> q = ets.random_q()
            >>> q

        """

        if i == 1:
            q = zeros(self.n)

            for i in range(self.n):
                q[i] = uniform(self.qlim[0, i], self.qlim[1, i])

        else:
            q = zeros((i, self.n))

            for j in range(i):
                for i in range(self.n):
                    q[j, i] = uniform(self.qlim[0, i], self.qlim[1, i])

        return q


class ETS(BaseETS):
    """
    This class implements an elementary transform sequence (ETS) for 3D

    :param arg: Function to compute ET value

    An instance can contain an elementary transform (ET) or an elementary
    transform sequence (ETS). It has list-like properties by subclassing
    UserList, which means we can perform indexing, slicing pop, insert, as well
    as using it as an iterator over its values.

    - ``ETS()`` an empty ETS list
    - ``ETS(et)`` an ETS containing a single ET
    - ``ETS([et0, et1, et2])`` an ETS consisting of three ET's

    Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS, ET
            >>> e = ET.Rz(0.3) # a single ET, rotation about z
            >>> ets1 = ETS(e)
            >>> len(ets1)
            >>> ets2 = ET.Rz(0.3) * ET.tx(2) # an ETS
            >>> len(ets2)                    # of length 2
            >>> ets2[1]                      # an ET sliced from the ETS

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    :seealso: :func:`rx`, :func:`ry`, :func:`rz`, :func:`tx`,
        :func:`ty`, :func:`tz`
    """

    def __init__(
        self,
        arg: Union[
            List[Union["ETS", ET]], List[ET], List["ETS"], ET, "ETS", None
        ] = None,
    ):
        super().__init__()
        if isinstance(arg, list):
            for item in arg:
                if isinstance(item, ET):
                    self.data.append(deepcopy(item))
                elif isinstance(item, ETS):
                    for ets_item in item:
                        self.data.append(deepcopy(ets_item))
                else:
                    raise TypeError("Invalid arg")
        elif isinstance(arg, ET):
            self.data.append(deepcopy(arg))
        elif isinstance(arg, ETS):
            for ets_item in arg:
                self.data.append(deepcopy(ets_item))
        elif arg is None:
            self.data = []
        else:
            raise TypeError("Invalid arg")

        super()._update_internals()

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
                "You can not have some jindices set for the ET's in arg. It must be all or none"
            )
        elif jindices == 0 and self.n > 0:
            # Set them ourself
            for j, joint in enumerate(joints):
                joint.jindex = j

            self._auto_jindex = True

    def __mul__(self, other: Union["ET", "ETS"]) -> "ETS":
        if isinstance(other, ET):
            return ETS([*self.data, other])
        else:
            return ETS([*self.data, *other.data])  # pragma: nocover

    def __rmul__(self, other: Union["ET", "ETS"]) -> "ETS":
        return ETS([other, *self.data])  # pragma: nocover

    def __imul__(self, rest: "ETS"):
        return self + rest  # pragma: nocover

    def __add__(self, rest) -> "ETS":
        return self.__mul__(rest)  # pragma: nocover

    def compile(self) -> "ETS":
        """
        Compile an ETS

        :return: optimised ETS

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
                    if not array_equal(const, eye(4)):
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
            if not array_equal(const, eye(4)):
                ets *= ET.SE3(const)
        return ets

    def insert(
        self,
        arg: Union[ET, "ETS"],
        i: int = -1,
    ) -> None:
        """
        Insert value

        :param i: insert an ET or ETS into the ETS, default is at the end
        :param arg: the elementary transform or sequence to insert

        Inserts an ET or ETS into the ET sequence.  The inserted value is at position
        ``i``.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET
            >>> e = ET.Rz() * ET.tx(1) * ET.Rz() * ET.tx(1)
            >>> f = ET.Ry()
            >>> e.insert(f, 2)
            >>> e
        """

        if isinstance(arg, ET):
            if i == -1:
                self.data.append(arg)
            else:
                self.data.insert(i, arg)
        elif isinstance(arg, ETS):
            if i == -1:
                for et in arg:
                    self.data.append(et)
            else:
                for j, et in enumerate(arg):
                    self.data.insert(i + j, et)
        self._update_internals()

    def fkine(
        self,
        q: ArrayLike,
        base: Union[ndarray, SE3, None] = None,
        tool: Union[ndarray, SE3, None] = None,
        include_base: bool = True,
    ) -> SE3:
        """
        Forward kinematics

        :param q: Joint coordinates
        :type q: ArrayLike
        :param base: base transform, optional
        :param tool: tool transform, optional

        :return: The transformation matrix representing the pose of the
            end-effector

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

        ret = SE3.Empty()
        fk = self.eval(q, base, tool, include_base)

        if fk.dtype == "O":
            # symbolic
            fk = array(simplify(fk))

        if fk.ndim == 3:
            for T in fk:
                ret.append(SE3(T, check=False))  # type: ignore
        else:
            ret = SE3(fk, check=False)

        return ret

    def eval(
        self,
        q: ArrayLike,
        base: Union[ndarray, SE3, None] = None,
        tool: Union[ndarray, SE3, None] = None,
        include_base: bool = True,
    ) -> ndarray:
        """
        Forward kinematics

        :param q: Joint coordinates
        :type q: ArrayLike
        :param base: base transform, optional
        :param tool: tool transform, optional

        :return: The transformation matrix representing the pose of the
            end-effector

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

        try:
            return ETS_fkine(self._fknm, q, base, tool, include_base)
        except BaseException:
            pass

        q = getmatrix(q, (None, None))
        l, _ = q.shape  # type: ignore
        end = self.data[-1]

        if isinstance(tool, SE3):
            tool = array(tool.A)

        if isinstance(base, SE3):
            base = array(base.A)

        if base is None:
            bases = None
        elif array_equal(base, eye(3)):  # pragma: nocover
            bases = None
        else:  # pragma: nocover
            bases = base

        if tool is None:
            tools = None
        elif array_equal(tool, eye(3)):  # pragma: nocover
            tools = None
        else:  # pragma: nocover
            tools = tool

        if l > 1:
            T = zeros((l, 4, 4), dtype=object)
        else:
            T = zeros((4, 4), dtype=object)

        # Tk = None

        for k, qk in enumerate(q):  # type: ignore
            link = end  # start with last link

            jindex = 0 if link.jindex is None and link.isjoint else link.jindex

            Tk = link.A(qk[jindex])

            if tools is not None:
                Tk = Tk @ tools

            # add remaining links, back toward the base
            for i in range(self.m - 2, -1, -1):
                link = self.data[i]

                jindex = 0 if link.jindex is None and link.isjoint else link.jindex
                A = link.A(qk[jindex])

                if A is not None:
                    Tk = A @ Tk

            # add base transform if it is set
            if include_base == True and bases is not None:
                Tk = bases @ Tk

            # append
            if l > 1:
                T[k, :, :] = Tk
            else:
                T = Tk

        return T

    def jacob0(
        self,
        q: ArrayLike,
        tool: Union[ndarray, SE3, None] = None,
    ) -> ndarray:
        r"""
        Jacobian in base frame
        :param q: Joint coordinate vector
        :type q: ArrayLike
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None
        :return J: Manipulator Jacobian in the base frame
        ``jacob0(q)`` is the ETS Jacobian matrix which maps joint
        velocity to spatial velocity in the {0} frame.
        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x,
        \omega_y, \omega_z)^T` is related to joint velocity by
        :math:`{}^{e}\nu = {}^{e}\mathbf{J}_0(q) \dot{q}`.
        If ``ets.eval(q)`` is already computed it can be passed in as ``T`` to
        reduce computation time.
        An ETS represents the relative pose from the {0} frame to the end frame
        {e}. This is the composition of mAny relative poses, some constant and
        some functions of the joint variables, which we can write as
        :math:`\mathbf{E}(q)`.
        .. math::
            {}^0 T_e = \mathbf{E}(q) \in \mbox{SE}(3)
        The temporal derivative of this is the spatial
        velocity :math:`\nu` which is a 6-vector is related to the rate of
        change of joint coordinates by the Jacobian matrix.
        .. math::
           {}^0 \nu = {}^0 \mathbf{J}(q) \dot{q} \in \mathbb{R}^6
        This velocity can be expressed relative to the {0} frame or the {e}
        frame.
        :references:
            - `Kinematic Derivatives using the Elementary Transform Sequence, J. Haviland and P. Corke <https://arxiv.org/abs/2010.08696>`_
        :seealso: :func:`jacobe`, :func:`hessian0`
        """  # noqa

        # Use c extension
        try:
            return ETS_jacob0(self._fknm, q, tool)
        except TypeError:
            pass

        # Otherwise use Python
        if tool is None:
            tools = eye(4)
        elif isinstance(tool, SE3):
            tools = array(tool.A)
        else:  # pragma: nocover
            tools = eye(4)

        q = getvector(q, None)

        T = self.eval(q, include_base=False) @ tools

        U = eye(4)
        j = 0
        J = zeros((6, self.n), dtype="object")
        zero = array([0, 0, 0])
        end = self.data[-1]

        for link in self.data:
            jindex = 0 if link.jindex is None and link.isjoint else link.jindex

            if link.isjoint:
                U = U @ link.A(q[jindex])  # type: ignore

                if link == end:
                    U = U @ tools

                Tu = SE3(U, check=False).inv().A @ T
                n = U[:3, 0]
                o = U[:3, 1]
                a = U[:3, 2]
                x = Tu[0, 3]
                y = Tu[1, 3]
                z = Tu[2, 3]

                if link.axis == "Rz":
                    J[:3, j] = (o * x) - (n * y)
                    J[3:, j] = a

                elif link.axis == "Ry":
                    J[:3, j] = (n * z) - (a * x)
                    J[3:, j] = o

                elif link.axis == "Rx":
                    J[:3, j] = (a * y) - (o * z)
                    J[3:, j] = n

                elif link.axis == "tx":
                    J[:3, j] = n
                    J[3:, j] = zero

                elif link.axis == "ty":
                    J[:3, j] = o
                    J[3:, j] = zero

                elif link.axis == "tz":
                    J[:3, j] = a
                    J[3:, j] = zero

                j += 1
            else:
                A = link.A()
                if A is not None:
                    U = U @ A

        return J

    def jacobe(
        self,
        q: ArrayLike,
        tool: Union[ndarray, SE3, None] = None,
    ) -> ndarray:
        r"""
        Manipulator geometric Jacobian in the end-effector frame

        :param q: Joint coordinate vector
        :type q: ArrayLike
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None

        :return J: Manipulator Jacobian in the end-effector frame

        - ``ets.jacobe(q)`` is the manipulator Jacobian matrix which maps
          joint  velocity to end-effector spatial velocity expressed in the
          end-effector frame.
        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        .. warning:: This is the **geometric Jacobian** as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a
            velocity twist as per the text by Lynch & Park.
        """  # noqa

        # Use c extension
        try:
            return ETS_jacobe(self._fknm, q, tool)
        except TypeError:
            pass

        T = self.eval(q, tool=tool, include_base=False)
        return tr2jac(T.T) @ self.jacob0(q, tool=tool)

    def hessian0(
        self,
        q: Union[ArrayLike, None] = None,
        J0: Union[ndarray, None] = None,
        tool: Union[ndarray, SE3, None] = None,
    ) -> ndarray:
        r"""
        Manipulator Hessian

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the world-coordinate frame. This
        function calulcates this based on the ETS of the robot.
        
        One of J0 or q
        is required. Supply J0 if already calculated to save computation time

        :param q: The joint angles/configuration of the robot.
        :type q: ArrayLike
        :param J0: The manipulator Jacobian in the 0 frame
        :param tool: a static tool transformation matrix to apply to the
            end frame, defaults to None

        :return: The manipulator Hessian in 0 frame

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
        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        # Use c extension
        try:
            return ETS_hessian0(self._fknm, q, J0, tool)
        except TypeError:
            pass

        def cross(a, b):
            x = a[1] * b[2] - a[2] * b[1]
            y = a[2] * b[0] - a[0] * b[2]
            z = a[0] * b[1] - a[1] * b[0]
            return array([x, y, z])

        n = self.n

        if J0 is None:
            q = getvector(q, None)
            J0 = self.jacob0(q, tool=tool)
        else:
            verifymatrix(J0, (6, self.n))

        H = zeros((n, 6, n))

        for j in range(n):
            for i in range(j, n):

                H[j, :3, i] = cross(J0[3:, j], J0[:3, i])
                H[j, 3:, i] = cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[i, :3, j] = H[j, :3, i]

        return H

    def hessiane(
        self,
        q: Union[ArrayLike, None] = None,
        Je: Union[ndarray, None] = None,
        tool: Union[ndarray, SE3, None] = None,
    ) -> ndarray:
        r"""
        Manipulator Hessian

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the world-coordinate frame. This
        function calulcates this based on the ETS of the robot.
        
        One of Je or q
        is required. Supply Je if already calculated to save computation time

        :param q: The joint angles/configuration of the robot.
        :type q: ArrayLike
        :param Je: The manipulator Jacobian in the ee frame
        :param tool: a static tool transformation matrix to apply to the
            end frame, defaults to None

        :return: The manipulator Hessian in ee frame

        This method computes the manipulator Hessian in the ee frame.  If
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
        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        # Use c extension
        try:
            return ETS_hessiane(self._fknm, q, Je, tool)
        except TypeError:
            pass

        def cross(a, b):
            x = a[1] * b[2] - a[2] * b[1]
            y = a[2] * b[0] - a[0] * b[2]
            z = a[0] * b[1] - a[1] * b[0]
            return array([x, y, z])

        n = self.n

        if Je is None:
            q = getvector(q, None)
            Je = self.jacobe(q, tool=tool)
        else:
            verifymatrix(Je, (6, self.n))

        H = zeros((n, 6, n))

        for j in range(n):
            for i in range(j, n):

                H[j, :3, i] = cross(Je[3:, j], Je[:3, i])
                H[j, 3:, i] = cross(Je[3:, j], Je[3:, i])

                if i != j:
                    H[i, :3, j] = H[j, :3, i]

        return H

    def jacob0_analytical(
        self,
        q: ArrayLike,
        representation: str = "rpy/xyz",
        tool: Union[ndarray, SE3, None] = None,
    ):
        r"""
        Manipulator analytical Jacobian in the base frame

        :param q: Joint coordinate vector
        :type q: ArrayLike
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None
        :param representation: describes the rotational representation

        :return J: Manipulator Jacobian in the base frame

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        ==================   ==================================
        ``representation``          Rotational representation
        ==================   ==================================
        ``'rpy/xyz'``        RPY angular rates in XYZ order
        ``'rpy/zyx'``        RPY angular rates in XYZ order
        ``'eul'``            Euler angular rates in ZYZ order
        ``'exp'``            exponential coordinate rates
        ==================   ==================================

        Example:
        .. runblock:: pycon
            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.ETS.Puma560()
            >>> puma.jacob0_analytical([0, 0, 0, 0, 0, 0])

        """  # noqa

        T = self.eval(q, tool=tool)
        J = self.jacob0(q, tool=tool)
        A = rotvelxform(t2r(T), full=True, inverse=True, representation=representation)
        return A @ J

    def jacobm(self, q: ArrayLike) -> ndarray:
        r"""
        Calculates the manipulability Jacobian. This measure relates the rate
        of change of the manipulability to the joint velocities of the robot.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).

        :return: The manipulability Jacobian
        :rtype: float ndarray(n)

        Yoshikawa's manipulability measure

        .. math::

            m(\vec{q}) = \sqrt{\mat{J}(\vec{q}) \mat{J}(\vec{q})^T}

        This method returns its Jacobian with respect to configuration

        .. math::

            \frac{\partial m(\vec{q})}{\partial \vec{q}}

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        J = self.jacob0(q)
        H = self.hessian0(q)

        manipulability = self.manipulability(q)

        # J = J[axes, :]
        # H = H[:, axes, :]

        b = inv(J @ J.T)
        Jm = zeros((self.n, 1))

        for i in range(self.n):
            c = J @ H[i, :, :].T
            Jm[i, 0] = manipulability * (c.flatten("F")).T @ b.flatten("F")

        return Jm

    def manipulability(self, q, method="yoshikawa"):
        """
        Manipulability measure

        :param q: Joint coordinates, one of J or q required
        :type q: ndarray(n), or ndarray(m,n)
        :param J: Jacobian in world frame if already computed, one of J or
            q required
        :type J: ndarray(6,n)
        :param method: method to use, "yoshikawa" (default), "condition",
            "minsingular"  or "asada"
        :type method: str
        :param axes: Task space axes to consider: "all" [default],
            "trans", "rot" or "both"
        :type axes: str
        :param kwargs: extra arguments to pass to ``jacob0``
        :return: manipulability
        :rtype: float or ndarray(m)

        - ``manipulability(q)`` is the scalar manipulability index
          for the robot at the joint configuration ``q``.  It indicates
          dexterity, that is, how well conditioned the robot is for motion
          with respect to the 6 degrees of Cartesian motion.  The values is
          zero if the robot is at a singularity.

        Various measures are supported:

        +-------------------+-------------------------------------------------+
        | Measure           |       Description                               |
        +-------------------+-------------------------------------------------+
        | ``"yoshikawa"``   | Volume of the velocity ellipsoid, *distance*    |
        |                   | from singularity [Yoshikawa85]_                 |
        +-------------------+-------------------------------------------------+
        | ``"invcondition"``| Inverse condition number of Jacobian, isotropy  |
        |                   | of the velocity ellipsoid [Klein87]_            |
        +-------------------+-------------------------------------------------+
        | ``"minsingular"`` | Minimum singular value of the Jacobian,         |
        |                   | *distance*  from singularity [Klein87]_         |
        +-------------------+-------------------------------------------------+
        | ``"asada"``       | Isotropy of the task-space acceleration         |
        |                   | ellipsoid which is a function of the Cartesian  |
        |                   | inertia matrix which depends on the inertial    |
        |                   | parameters [Asada83]_                           |
        +-------------------+-------------------------------------------------+

        **Trajectory operation**:

        If ``q`` is a matrix (m,n) then the result (m,) is a vector of
        manipulability indices for each joint configuration specified by a row
        of ``q``.

        .. note::

            - Invokes the ``jacob0`` method of the robot if ``J`` is not passed
            - The "all" option includes rotational and translational
              dexterity, but this involves adding different units. It can be
              more useful to look at the translational and rotational
              manipulability separately.
            - Examples in the RVC book (1st edition) can be replicated by
              using the "all" option
            - Asada's measure requires inertial a robot model with inertial
              parameters.

        :references:

        .. [Yoshikawa85] Manipulability of Robotic Mechanisms. Yoshikawa T.,
                The International Journal of Robotics Research.
                1985;4(2):3-9. doi:10.1177/027836498500400201
        .. [Asada83] A geometrical representation of manipulator dynamics and
                its application to arm design, H. Asada,
                Journal of Dynamic Systems, Measurement, and Control,
                vol. 105, p. 131, 1983.
        .. [Klein87] Dexterity Measures for the Design and Control of
                Kinematically Redundant Manipulators. Klein CA, Blaho BE.
                The International Journal of Robotics Research.
                1987;6(2):72-83. doi:10.1177/027836498700600206

        - Robotics, Vision & Control, Chap 8, P. Corke, Springer 2011.

        """

        def yoshikawa(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            if J.shape[0] == J.shape[1]:
                # simplified case for square matrix
                return abs(det(J))
            else:
                m2 = det(J @ J.T)
                return sqrt(abs(m2))

        def condition(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            return 1 / cond(J)

        def minsingular(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            s = svd(J, compute_uv=False)
            return s[-1]  # return last/smallest singular value of J

        def asada(robot, J, q, axes, **kwargs):
            # dof = np.sum(axes)
            if matrix_rank(J) < 6:
                return 0
            Ji = pinv(J)
            Mx = Ji.T @ robot.inertia(q) @ Ji
            d = where(axes)[0]
            Mx = Mx[d]
            Mx = Mx[:, d.tolist()]
            e, _ = eig(Mx)
            return min(e) / max(e)

        # choose the handler function
        if method == "yoshikawa":
            mfunc = yoshikawa
        elif method == "invcondition":
            mfunc = condition
        elif method == "minsingular":
            mfunc = minsingular
        elif method == "asada":
            mfunc = asada
        else:
            raise ValueError("Invalid method chosen")

        # q = getmatrix(q, (None, self.n))
        # w = zeros(q.shape[0])
        axes = [True, True, True, True, True, True]

        # for k, qk in enumerate(q):
        J = self.jacob0(q)
        w = mfunc(self, J, q, axes)

        # if len(w) == 1:
        #     return w[0]
        # else:
        return w

    def partial_fkine0(self, q: ArrayLike, n: int) -> ndarray:
        r"""
        Manipulator Forward Kinematics nth Partial Derivative

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the ee frame. This
        function calulcates this based on the ETS of the robot. One of Je or q
        is required. Supply Je if already calculated to save computation time

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: ArrayLike
        :param end: the final link/Gripper which the Hessian represents
        :param start: the first link which the Hessian represents
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None

        :return: The nth Partial Derivative of the forward kinematics

        :references:
            - Kinematic Derivatives using the Elementary Transform
                Sequence, J. Haviland and P. Corke
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
        count = array([0, 0])

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
            count = concatenate(([0], count))

            # This variables corresponds to indices within the previous partial derivatives
            # to be cross prodded
            # The order is: "[j, k, l, m, n, o, ...]"
            # Although, our partial derivatives have the order: pd[..., o, n, m, l, k, cartesian DoF, j]
            # For example, consider the Hessian Tensor H[n, 6, n], the index H[k, :, j]. This corrsponds
            # to the second partial derivative of the kinematics of joint j with respect to joint k.
            indices = add_indices(indices, c)

            # This variable corresponds to the indices in Td which corresponds to the
            # partial derivatives we need to use
            pdi = add_pdi(pdi)

            c += 1

            # Allocate our new partial derivative tensor
            pd = zeros(size)

            # We need to loop n^c times
            # There are n^c columns to calculate
            for _ in range(self.n**c):

                # Allocate the rotation and translation components
                rot = zeros(3)
                trn = zeros(3)

                # This loop calculates a single column ([trn, rot]) of the tensor for dT(x)
                for j in range(len(indices)):
                    pdr0 = dT[pdi[j][0]]
                    pdr1 = dT[pdi[j][1]]

                    idx0 = count[indices[j][0]]
                    idx1 = count[indices[j][1]]

                    # This is a list of indices selecting the slices of the previous tensor
                    idx0_slices = flip(idx0[1:])
                    idx1_slices = flip(idx1[1:])

                    # This index selecting the column within the 2d slice of the previous tensor
                    idx0_n = idx0[0]
                    idx1_n = idx1[0]

                    # Use our indices to select the rotational column from pdr0 and pdr1
                    col0_rot = pdr0[(*idx0_slices, slice(3, 6), idx0_n)]
                    col1_rot = pdr1[(*idx1_slices, slice(3, 6), idx1_n)]

                    # Use our indices to select the translational column from pdr1
                    col1_trn = pdr1[(*idx1_slices, slice(0, 3), idx1_n)]

                    # Perform the cross product as described in the maths above
                    rot += cross(col0_rot, col1_rot)
                    trn += cross(col0_rot, col1_trn)

                pd[(*flip(count[1:]), slice(0, 3), count[0])] = trn
                pd[(*flip(count[1:]), slice(3, 6), count[0])] = rot

                count[0] += 1
                for j in range(len(count)):
                    if count[j] == self.n:
                        count[j] = 0
                        if j != len(count) - 1:
                            count[j + 1] += 1

            dT.append(pd)

        return dT[-1]

    def ik_lm_chan(
        self,
        Tep: Union[ndarray, SE3],
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        λ: float = 1.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Chan's Method)

        :param Tep: The desired end-effector pose or pose trajectory
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return IK_LM_Chan(self._fknm, Tep, q0, ilimit, slimit, tol, reject_jl, we, λ)

    def ik_lm_wampler(
        self,
        Tep: Union[ndarray, SE3],
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        λ: float = 1.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Chan's Method)

        :param Tep: The desired end-effector pose or pose trajectory
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return IK_LM_Wampler(self._fknm, Tep, q0, ilimit, slimit, tol, reject_jl, we, λ)

    def ik_lm_sugihara(
        self,
        Tep: Union[ndarray, SE3],
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        λ: float = 1.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Chan's Method)

        :param Tep: The desired end-effector pose or pose trajectory
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return IK_LM_Sugihara(
            self._fknm, Tep, q0, ilimit, slimit, tol, reject_jl, we, λ
        )

    def ik_nr(
        self,
        Tep: Union[ndarray, SE3],
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        use_pinv: int = True,
        pinv_damping: float = 0.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Chan's Method)

        :param Tep: The desired end-effector pose or pose trajectory
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return IK_NR(
            self._fknm,
            Tep,
            q0,
            ilimit,
            slimit,
            tol,
            reject_jl,
            we,
            use_pinv,
            pinv_damping,
        )

    def ik_gn(
        self,
        Tep: Union[ndarray, SE3],
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        use_pinv: int = True,
        pinv_damping: float = 0.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Chan's Method)

        :param Tep: The desired end-effector pose or pose trajectory
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return IK_GN(
            self._fknm,
            Tep,
            q0,
            ilimit,
            slimit,
            tol,
            reject_jl,
            we,
            use_pinv,
            pinv_damping,
        )


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
        arg: Union[
            List[Union["ETS2", ET2]], List[ET2], List["ETS2"], ET2, "ETS2", None
        ] = None,
    ):
        super().__init__()
        if isinstance(arg, list):
            for item in arg:
                if isinstance(item, ET2):
                    self.data.append(deepcopy(item))
                elif isinstance(item, ETS2):
                    for ets_item in item:
                        self.data.append(deepcopy(ets_item))
                else:
                    raise TypeError("bad arg")
        elif isinstance(arg, ET2):
            self.data.append(deepcopy(arg))
        elif isinstance(arg, ETS2):
            for ets_item in arg:
                self.data.append(deepcopy(ets_item))
        elif arg is None:
            self.data = []
        else:
            raise TypeError("Invalid arg")

        self._update_internals()
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
                "You can not have some jindices set for the ET's in arg. It must be all or none"
            )
        elif jindices == 0 and self.n > 0:
            # Set them ourself
            for j, joint in enumerate(joints):
                joint.jindex = j
            self._auto_jindex = True

    def __mul__(self, other: Union[ET2, "ETS2"]) -> "ETS2":
        if isinstance(other, ET2):
            return ETS2([*self.data, other])
        else:
            return ETS2([*self.data, *other.data])  # pragma: nocover

    def __rmul__(self, other: Union[ET2, "ETS2"]) -> "ETS2":
        return ETS2([other, self.data])  # pragma: nocover

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
                    if not array_equal(const, eye(3)):
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
            if not array_equal(const, eye(3)):
                ets *= ET2.SE2(const)
        return ets

    def insert(
        self,
        arg: Union[ET2, "ETS2"],
        i: int = -1,
    ) -> None:
        """
        Insert value

        :param i: insert an ET or ETS into the ETS, default is at the end
        :param arg: the elementary transform or sequence to insert

        Inserts an ET or ETS into the ET sequence.  The inserted value is at position
        ``i``.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET2
            >>> e = ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1)
            >>> f = ET2.R()
            >>> e.insert(f, 2)
            >>> e
        """

        if isinstance(arg, ET2):
            if i == -1:
                self.data.append(arg)
            else:
                self.data.insert(i, arg)
        elif isinstance(arg, ETS2):
            if i == -1:
                for et in arg:
                    self.data.append(et)
            else:
                for j, et in enumerate(arg):
                    self.data.insert(i + j, et)
        self._update_internals()

    def fkine(
        self,
        q: ArrayLike,
        base: Union[ndarray, SE2, None] = None,
        tool: Union[ndarray, SE2, None] = None,
        include_base: bool = True,
    ) -> SE2:
        """
        Forward kinematics
        :param q: Joint coordinates
        :type q: ArrayLike
        :param base: base transform, optional
        :param tool: tool transform, optional

        :return: The transformation matrix representing the pose of the
            end-effector

        - ``T = ets.fkine(q)`` evaluates forward kinematics for the robot at
          joint configuration ``q``.
        **Trajectory operation**:
        If ``q`` has multiple rows (mxn), it is considered a trajectory and the
        result is an ``SE2`` instance with ``m`` values.
        .. note::
            - The robot's base tool transform, if set, is incorporated
              into the result.
            - A tool transform, if provided, is incorporated into the result.
            - Works from the end-effector link to the base
        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        ret = SE2.Empty()
        fk = self.eval(q, base, tool, include_base)

        if fk.dtype == "O":
            # symbolic
            fk = array(simplify(fk))

        if fk.ndim == 3:
            for T in fk:
                ret.append(SE2(T, check=False))  # type: ignore
        else:
            ret = SE2(fk, check=False)

        return ret

    def eval(
        self,
        q: ArrayLike,
        base: Union[ndarray, SE2, None] = None,
        tool: Union[ndarray, SE2, None] = None,
        include_base: bool = True,
    ) -> ndarray:
        """
        Forward kinematics
        :param q: Joint coordinates
        :type q: ArrayLike
        :param base: base transform, optional
        :param tool: tool transform, optional

        :return: The transformation matrix representing the pose of the
            end-effector

        - ``T = ets.fkine(q)`` evaluates forward kinematics for the robot at
          joint configuration ``q``.
        **Trajectory operation**:
        If ``q`` has multiple rows (mxn), it is considered a trajectory and the
        result is an ``SE2`` instance with ``m`` values.
        .. note::
            - The robot's base tool transform, if set, is incorporated
              into the result.
            - A tool transform, if provided, is incorporated into the result.
            - Works from the end-effector link to the base
        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        q = getmatrix(q, (None, None))
        l, _ = q.shape  # type: ignore
        end = self[-1]

        if base is None:
            bases = None
        elif isinstance(base, SE2):
            bases = array(base.A)
        elif array_equal(base, eye(3)):  # pragma: nocover
            bases = None
        else:  # pragma: nocover
            bases = base

        if tool is None:
            tools = None
        elif isinstance(tool, SE2):
            tools = array(tool.A)
        elif array_equal(tool, eye(3)):  # pragma: nocover
            tools = None
        else:  # pragma: nocover
            tools = tool

        if l > 1:
            T = zeros((l, 3, 3), dtype=object)
        else:
            T = zeros((3, 3), dtype=object)

        for k, qk in enumerate(q):  # type: ignore
            link = end  # start with last link

            jindex = 0 if link.jindex is None and link.isjoint else link.jindex
            Tk = link.A(qk[jindex])

            if tools is not None:
                Tk = Tk @ tools

            # add remaining links, back toward the base
            for i in range(self.m - 2, -1, -1):
                link = self.data[i]

                jindex = 0 if link.jindex is None and link.isjoint else link.jindex
                A = link.A(qk[jindex])

                if A is not None:
                    Tk = A @ Tk

            # add base transform if it is set
            if include_base == True and bases is not None:
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
    ):

        # very inefficient implementation, just put a 1 in last row
        # if its a rotation joint
        q = getvector(q)

        j = 0
        J = zeros((3, self.n))
        etjoints = self.joint_idx()

        if not all(array([self[i].jindex for i in etjoints])):
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
                dTdq = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ self[i].A(
                    q[jindex]  # type: ignore
                )
            elif axis == "tx":
                dTdq = array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
            elif axis == "ty":
                dTdq = array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
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
        Jacobian in base frame

        :param q: joint coordinates
        :type q: ArrayLike
        :return: Jacobian matrix

        ``jacobe(q)`` is the manipulator Jacobian matrix which maps joint
        velocity to end-effector spatial velocity.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{e}\nu = {}^{e}\mathbf{J}_0(q) \dot{q}`.

        :seealso: :func:`jacob`, :func:`hessian0`
        """  # noqa

        T = self.fkine(q, include_base=False).A
        return tr2jac2(T.T) @ self.jacob0(q)


# if __name__ == "__main__":

#     from roboticstoolbox import models

#     ur5 = models.URDF.UR5()

#     ur5.fkine(ur5.qz)
#     ur5.jacob0(ur5.qz)
#     ur5.jacob0_analytic(ur5.qz)
