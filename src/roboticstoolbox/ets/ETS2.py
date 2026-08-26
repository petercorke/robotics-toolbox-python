#!/usr/bin/env python3

"""
@author: Jesse Haviland
@author: Peter Corke
"""

from __future__ import annotations
from functools import cached_property
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

from roboticstoolbox.ets.fknm import (
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
from roboticstoolbox.ets.ET2 import ET2
from roboticstoolbox.ets._ET import BaseET
from roboticstoolbox.ets._ETS import BaseETS, T, _dirties_fknm
from typing import overload, TypeVar
from typing import Literal as L
from roboticstoolbox.tools.types import ArrayLike, NDArray


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

        q = self._resolve_q(q, allow_trajectory=True)
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
        j = 0
        J = np.zeros((3, self.n))
        etjoints = self.joint_idx()

        if not np.all(np.array([self[i].jindex for i in etjoints])):
            # not all joints have a jindex it is required, set them
            for j in range(self.n):
                i = etjoints[j]
                self[i].jindex = j
            self.__dict__.pop("jindices", None)  # invalidate cached_property

        q = self._resolve_q(q)

        for j in range(self.n):
            i = etjoints[j]

            if self[i].jindex is not None:
                jindex = self[i].jindex
            else:
                jindex = 0  # pragma: nocover

            # jindex = 0 if self[i].jindex is None else self[i].jindex

            axis = self[i].kind
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
