#!/usr/bin/env python3
"""
@author: Jesse Haviland
@author: Peter Corke
"""
from collections import UserList
from types import SimpleNamespace
import copy
from abc import ABC, abstractclassmethod, abstractmethod
from attr import attributes
import numpy as np
from spatialmath import SE3, SE2
from spatialmath.base import (
    getvector,
    getunit,
    trotx,
    troty,
    trotz,
    issymbol,
    tr2jac,
    transl2,
    trot2,
    removesmall,
    trinv,
    trinv2,
    verifymatrix,
    iseye,
    tr2jac2,
)
from roboticstoolbox import rtb_get_param

from collections import UserList
import numpy as np
from spatialmath.base import trotx, troty, trotz, issymbol, getmatrix, tr2rpy
import fknm
import sympy
from copy import deepcopy
from roboticstoolbox import rtb_get_param
from roboticstoolbox.robot.ET import ET, ET2
from spatialmath.base import getvector
from spatialmath import SE3
from typing import Type, Union, overload

from numpy.typing import ArrayLike, NDArray


class BaseETS(UserList):
    def __init__(self, *args):
        super().__init__(*args)

    def _update_internals(self):
        self._fknm = [et.fknm for et in self.data]
        self._m = len(self.data)
        self._n = len([True for et in self.data if et.isjoint])

    def __str__(self, q: str = None):
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
        for et in self.data:

            if et.isjoint:
                if q is not None:
                    if et.jindex is None:
                        _j = j
                    else:
                        _j = et.jindex
                    qvar = q.format(
                        _j, _j + 1
                    )  # lgtm [py/str-format/surplus-argument]  # noqa
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
                s = f"{et.axis}({et.eta:.4g})"

            elif not et.iselementary:
                s = str(et)
                c += 1

            es.append(s)

        if unicode:
            return " \u2295 ".join(es)
        else:  # pragma: nocover
            return " * ".join(es)

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

    @property
    def m(self) -> int:
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

    @overload
    def data(self: "ETS") -> list[ET]:  # pragma: nocover
        ...

    @overload
    def data(self: "ETS2") -> list[ET2]:  # pragma: nocover
        ...

    @property
    def data(self):
        return self._data

    @data.setter
    @overload
    def data(self: "ETS", new_data: list[ET]):  # pragma: nocover
        ...

    @data.setter
    @overload
    def data(self: "ETS", new_data: list[ET2]):  # pragma: nocover
        ...

    @data.setter
    def data(self, new_data):
        self._data = new_data

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
        self._update_internals()
        return item

    @overload
    def split(self: "ETS") -> list["ETS"]:  # pragma: nocover
        ...

    @overload
    def split(self: "ETS2") -> list["ETS2"]:  # pragma: nocover
        ...

    def split(self):
        """
        Split ETS into link segments

        Returns a list of ETS, each one, apart from the last,
        ends with a variable ET.
        """
        segments = []
        start = 0

        for j, k in enumerate(self.joints()):
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

        return self.__class__([et.inv() for et in reversed(self.data)])

    @overload
    def __getitem__(self: "ETS", i: int) -> ET:  # pragma: nocover
        ...

    @overload
    def __getitem__(self: "ETS", i: slice) -> list[ET]:  # pragma: nocover
        ...

    @overload
    def __getitem__(self: "ETS2", i: int) -> ET2:  # pragma: nocover
        ...

    @overload
    def __getitem__(self: "ETS2", i: slice) -> list[ET2]:  # pragma: nocover
        ...

    def __getitem__(self, i):
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


class ETS(BaseETS):
    """
    This class implements an elementary transform sequence (ETS) for 3D

    :param arg: Function to compute ET value
    :type arg: callable

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
            >>> e[1]                         # an ET sliced from the ETS

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    :seealso: :func:`rx`, :func:`ry`, :func:`rz`, :func:`tx`,
        :func:`ty`, :func:`tz`
    """

    def __init__(
        self,
        arg: Union[
            list[Union["ETS", ET]], list[ET], list["ETS"], ET, "ETS", None
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

        try:
            return fknm.ETS_fkine(self._m, self._fknm, q, base, tool, include_base)
        except:
            pass

        q = getmatrix(q, (None, None))
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
            T = np.zeros((l, 4, 4), dtype=object)
        else:
            T = np.zeros((4, 4), dtype=object)
        Tk = np.eye(4)

        for k, qk in enumerate(q):  # type: ignore
            link = end  # start with last link

            jindex = 0 if link.jindex is None and link.isjoint else link.jindex
            A = link.T(qk[jindex])

            if A is None:
                Tk = tool  # pragma: nocover
            else:
                Tk = A @ tool

            # add remaining links, back toward the base
            for i in range(self.m - 2, -1, -1):
                link = self.data[i]

                jindex = 0 if link.jindex is None and link.isjoint else link.jindex
                A = link.T(qk[jindex])

                if A is not None:
                    Tk = A @ Tk

            # add base transform if it is set
            if include_base == True:
                Tk = base @ Tk

            # append
            if l > 1:
                T[k, :, :] = Tk
            else:
                T = Tk

        return T

    def jacob0(
        self,
        q: ArrayLike,
        tool: Union[NDArray[np.float64], SE3, None] = None,
    ) -> NDArray[np.float64]:
        r"""
        Jacobian in base frame

        :param q: Joint coordinate vector
        :type q: ndarray(n)
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None
        :type tool: SE3, optional
        :return J: Manipulator Jacobian in the base frame
        :rtype: ndarray(6,n)

        ``jacob0(q)`` is the ETS Jacobian matrix which maps joint
        velocity to spatial velocity in the {0} frame.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x,
        \omega_y, \omega_z)^T` is related to joint velocity by
        :math:`{}^{e}\nu = {}^{e}\mathbf{J}_0(q) \dot{q}`.

        If ``ets.eval(q)`` is already computed it can be passed in as ``T`` to
        reduce computation time.

        An ETS represents the relative pose from the {0} frame to the end frame
        {e}. This is the composition of many relative poses, some constant and
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
            return fknm.ETS_jacob0(self._m, self._n, self._fknm, q, tool)
        except TypeError:
            pass

        # Otherwise use Python
        if tool is None:
            tool = np.eye(4)
        elif isinstance(tool, SE3):
            tool = np.array(tool.A)

        q = getvector(q, None)

        T = self.fkine(q, include_base=False) @ tool

        U = np.eye(4)
        j = 0
        J = np.zeros((6, self.n), dtype="object")
        zero = np.array([0, 0, 0])
        end = self.data[-1]

        for link in self.data:
            jindex = 0 if link.jindex is None and link.isjoint else link.jindex

            if link.isjoint:
                U = U @ link.T(q[jindex])  # type: ignore

                if link == end:
                    U = U @ tool

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
                A = link.T()
                if A is not None:
                    U = U @ A

        return J

    def jacobe(
        self,
        q: ArrayLike,
        tool: Union[NDArray[np.float64], SE3, None] = None,
    ) -> NDArray[np.float64]:
        r"""
        Manipulator geometric Jacobian in the end-effector frame

        :param q: Joint coordinate vector
        :type q: ndarray(n)

        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None
        :type tool: SE3, optional

        :return J: Manipulator Jacobian in the end-effector frame
        :rtype: ndarray(6,n)

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
            return fknm.ETS_jacobe(self._m, self._n, self._fknm, q, tool)
        except TypeError:
            pass

        T = self.fkine(q, tool=tool, include_base=False)
        return tr2jac(T.T) @ self.jacob0(q, tool=tool)

    def hessian0(
        self,
        q: Union[ArrayLike, None] = None,
        J0: Union[NDArray[np.float64], None] = None,
        tool: Union[NDArray[np.float64], SE3, None] = None,
    ) -> NDArray[np.float64]:
        r"""
        Manipulator Hessian

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the world-coordinate frame. This
        function calulcates this based on the ETS of the robot.
        
        One of J0 or q
        is required. Supply J0 if already calculated to save computation time

        :param q: The joint angles/configuration of the robot.
        :type q: float ndarray(n)
        :param J0: The manipulator Jacobian in the 0 frame
        :type J0: float ndarray(6,n)

        :return: The manipulator Hessian in 0 frame
        :rtype: float ndarray(6,n,n)

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
            return fknm.ETS_hessian0(self._m, self._n, self._fknm, q, J0, tool)
        except TypeError:
            pass

        def cross(a, b):
            x = a[1] * b[2] - a[2] * b[1]
            y = a[2] * b[0] - a[0] * b[2]
            z = a[0] * b[1] - a[1] * b[0]
            return np.array([x, y, z])

        n = self.n

        if J0 is None:
            q = getvector(q, None)
            J0 = self.jacob0(q, tool=tool)
        else:
            verifymatrix(J0, (6, self.n))

        H = np.zeros((n, 6, n))

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
        Je: Union[NDArray[np.float64], None] = None,
        tool: Union[NDArray[np.float64], SE3, None] = None,
    ) -> NDArray[np.float64]:
        r"""
        Manipulator Hessian

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the world-coordinate frame. This
        function calulcates this based on the ETS of the robot.
        
        One of J0 or q
        is required. Supply J0 if already calculated to save computation time

        :param q: The joint angles/configuration of the robot.
        :type q: float ndarray(n)
        :param J0: The manipulator Jacobian in the ee frame
        :type J0: float ndarray(6,n)

        :return: The manipulator Hessian in ee frame
        :rtype: float ndarray(6,n,n)

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
            return fknm.ETS_hessiane(self._m, self._n, self._fknm, q, Je, tool)
        except TypeError:
            pass

        # def cross(a, b):
        #     x = a[1] * b[2] - a[2] * b[1]
        #     y = a[2] * b[0] - a[0] * b[2]
        #     z = a[0] * b[1] - a[1] * b[0]
        #     return np.array([x, y, z])

        # n = self.n

        # if Je is None:
        #     q = getvector(q, None)
        #     Je = self.jacobe(q, tool=tool)
        # else:
        #     verifymatrix(Je, (6, self.n))

        # H = np.zeros((n, 6, n))

        # for j in range(n):
        #     for i in range(j, n):

        #         H[j, :3, i] = cross(Je[3:, j], Je[:3, i])
        #         H[j, 3:, i] = cross(Je[3:, j], Je[3:, i])

        #         if i != j:
        #             H[i, :3, j] = H[j, :3, i]

        # return H

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

    def insert(
        self,
        arg: Union[ET, "ETS"],
        i: int = -1,
    ) -> None:
        """
        Insert value

        :param i: insert an ET or ETS into the ETS, default is at the end
        :type i: int
        :param arg: the elementary transform or sequence to insert
        :type arg: ET

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


class ETS2(BaseETS):
    """
    This class implements an elementary transform sequence (ETS) for 2D

    :param arg: Function to compute ET value
    :type arg: callable

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

            >>> from roboticstoolbox import ETS2 as ETS
            >>> e = ETS.r(0.3)  # a single ET, rotation about z
            >>> len(e)
            >>> e = ETS.r(0.3) * ETS.tx(2)  # an ETS
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
            list[Union["ETS2", ET2]], list[ET2], list["ETS2"], ET2, "ETS2", None
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
        :rtype: ETS2

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
                    if not np.all(const == np.eye(3)):
                        ets *= ET2.SE2(const)
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
            if not np.all(const == np.eye(3)):
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
        :type i: int
        :param arg: the elementary transform or sequence to insert
        :type arg: ET

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
        base: Union[NDArray[np.float64], SE2, None] = None,
        tool: Union[NDArray[np.float64], SE2, None] = None,
        include_base: bool = True,
    ) -> NDArray[np.float64]:
        """
        Forward kinematics
        :param q: Joint coordinates
        :type q: ndarray(n) or ndarray(m,n)
        :param base: base transform, optional
        :type base: ndarray(3,3) or SE2
        :param tool: tool transform, optional
        :type tool: ndarray(3,3) or SE2
        :return: The transformation matrix representing the pose of the
            end-effector
        :rtype: SE2 instance
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
            base = np.eye(3)
        elif isinstance(base, SE2):
            base = np.array(base.A)

        if tool is None:
            tool = np.eye(3)
        elif isinstance(tool, SE2):
            tool = np.array(tool.A)

        if l > 1:
            T = np.zeros((l, 3, 3), dtype=object)
        else:
            T = np.zeros((3, 3), dtype=object)
        Tk = np.eye(3)

        for k, qk in enumerate(q):  # type: ignore
            link = end  # start with last link

            jindex = 0 if link.jindex is None and link.isjoint else link.jindex
            A = link.T(qk[jindex])

            if A is None:
                Tk = tool  # pragma: nocover
            else:
                Tk = A @ tool

            # add remaining links, back toward the base
            for i in range(self.m - 2, -1, -1):
                link = self.data[i]

                jindex = 0 if link.jindex is None and link.isjoint else link.jindex
                A = link.T(qk[jindex])

                if A is not None:
                    Tk = A @ Tk

            # add base transform if it is set
            if include_base == True:
                Tk = base @ Tk

            # append
            if l > 1:
                T[k, :, :] = Tk
            else:
                T = Tk

        return T

    def jacob0(
        self,
        q: ArrayLike,
    ):

        # very inefficient implementation, just put a 1 in last row
        # if its a rotation joint
        q = getvector(q)

        j = 0
        J = np.zeros((3, self.n))
        etjoints = self.joints()

        if not all([self[i].jindex for i in etjoints]):
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
                dTdq = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ self[i].T(
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
                dTdq = E0.fkine(q) @ dTdq

            Ef = ETS2(self[i + 1 :])
            if len(Ef) > 0:
                dTdq = dTdq @ Ef.fkine(q)

            T = self.fkine(q)
            dRdt = dTdq[:2, :2] @ T[:2, :2].T
            J[:, j] = np.r_[dTdq[:2, 2].T, dRdt[1, 0]]

        return J

    def jacobe(
        self,
        q: ArrayLike,
    ):
        r"""
        Jacobian in base frame

        :param q: joint coordinates
        :type q: array_like
        :return: Jacobian matrix
        :rtype: ndarray(6,n)

        ``jacobe(q)`` is the manipulator Jacobian matrix which maps joint
        velocity to end-effector spatial velocity.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{e}\nu = {}^{e}\mathbf{J}_0(q) \dot{q}`.

        :seealso: :func:`jacob`, :func:`hessian0`
        """  # noqa

        T = self.fkine(q, include_base=False)
        return tr2jac2(T.T) @ self.jacob0(q)


#     @property
#     def s(self):
#         if self.axis[1] == 'x':
#             if self.axis[0] == 'R':
#                 return np.r_[0, 0, 0, 1, 0, 0]
#             else:
#                 return np.r_[1, 0, 0, 0, 0, 0]
#         elif self.axis[1] == 'y':
#             if self.axis[0] == 'R':
#                 return np.r_[0, 0, 0, 0, 1, 0]
#             else:
#                 return np.r_[0, 1, 0, 0, 0, 0]
#         else:
#             if self.axis[0] == 'R':
#                 return np.r_[0, 0, 0, 0, 0, 1]
#             else:
#                 return np.r_[0, 0, 1, 0, 0, 0]


#     def hessian0(self, q=None, J0=None):
#         r"""
#         Hessian in base frame

#         :param q: joint coordinates
#         :type q: array_like
#         :param J0: Jacobian in {0} frame
#         :type J0: ndarray(6,n)
#         :return: Hessian matrix
#         :rtype: ndarray(6,n,n)

#         This method calculcates the Hessisan of the ETS. One of ``J0`` or
#         ``q`` is required. If ``J0`` is already calculated for the joint
#         coordinates ``q`` it can be passed in to to save computation time

#         An ETS represents the relative pose from the {0} frame to the end frame
#         {e}. This is the composition of many relative poses, some constant and
#         some functions of the joint variables, which we can write as
#         :math:`\mathbf{E}(q)`.

#         .. math::

#             {}^0 T_e = \mathbf{E}(q) \in \mbox{SE}(3)

#         The temporal derivative of this is the spatial
#         velocity :math:`\nu` which is a 6-vector is related to the rate of
#         change of joint coordinates by the Jacobian matrix.

#         .. math::

#             {}^0 \nu = {}^0 \mathbf{J}(q) \dot{q} \in \mathbb{R}^6

#         This velocity can be expressed relative to the {0} frame or the {e}
#         frame.

#         The temporal derivative of spatial velocity is spatial acceleration,
#         which again can be expressed with respect to the {0} or {e} frames

#         .. math::

#             {}^0 \dot{\nu} = \mathbf{J}(q) \ddot{q} + \dot{\mathbf{J}}(q) \dot{q} \in \mathbb{R}^6 \\
#                       &= \mathbf{J}(q) \ddot{q} + \dot{q}^T \mathbf{H}(q) \dot{q}

#         The manipulator Hessian tensor :math:`H` maps joint velocity to
#         end-effector spatial acceleration, expressed in the {0} coordinate
#         frame.

#         :references:
#             - `Kinematic Derivatives using the Elementary Transform Sequence, J. Haviland and P. Corke <https://arxiv.org/abs/2010.08696>`_

#         :seealso: :func:`jacob0`
#         """  # noqa

#         n = self.n

#         if J0 is None:
#             if q is None:
#                 q = np.copy(self.q)
#             else:
#                 q = getvector(q, n)

#             J0 = self.jacob0(q)
#         else:
#             verifymatrix(J0, (6, n))

#         H = np.zeros((6, n, n))

#         for j in range(n):
#             for i in range(j, n):

#                 H[:3, i, j] = np.cross(J0[3:, j], J0[:3, i])
#                 H[3:, i, j] = np.cross(J0[3:, j], J0[3:, i])

#                 if i != j:
#                     H[:3, j, i] = H[:3, i, j]

#         return H


#     def _inverse(self, T):
#         return trinv2(T)
