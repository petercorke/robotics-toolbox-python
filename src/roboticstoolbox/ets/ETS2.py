#!/usr/bin/env python3

"""
@author: Jesse Haviland
@author: Peter Corke
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from spatialmath import SE2
from spatialmath.base import (
    simplify,
    tr2jac2,
)

from roboticstoolbox.ets._ETS import BaseETS
from roboticstoolbox.ets.ET2 import ET2
from roboticstoolbox.robot.IK import IKSolution, normalise_q, within_limits
from roboticstoolbox.tools.types import ArrayLike, NDArray


class ETS2(BaseETS[ET2]):
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
    - ``ET2.XY(jindex=J)`` as above but the joint index is explicitly given, this
      might correspond to the joint number of a multi-joint robot.
    - ``ET2.XY(flip=True)`` as above but the joint moves in the opposite sense

    where ``XY`` is one of ``R``, ``tx``, ``ty``.

    Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET2
            >>> e = ET2.R(0.3) * ET2.tx(2)  # an ETS2
            >>> len(e)                      # of length 2
            >>> e[1]                        # an ET2 indexed from the ETS2

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    .. seealso:: ``r``, ``tx``, ``ty``
    """

    def __init__(
        self, arg: list[ETS2 | ET2] | list[ET2] | list[ETS2] | ET2 | ETS2 | None = None
    ) -> None:
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

    def __mul__(self, other: ET2 | ETS2) -> ETS2:
        if isinstance(other, ET2):
            return ETS2([*self._data, other])
        else:
            return ETS2([*self._data, *other._data])  # pragma: nocover

    def __rmul__(self, other: ET2 | ETS2) -> ETS2:
        return ETS2([other, *self._data])  # pragma: nocover

    def __imul__(self, rest: ET2 | ETS2) -> ETS2:
        return self + rest  # pragma: nocover

    def __add__(self, rest: ET2 | ETS2) -> ETS2:
        return self.__mul__(rest)  # pragma: nocover

    def fixjoint(self, j: int, value: float, unit: str = "rad") -> ETS2:
        """Return an independent ETS2 with the specified joint held constant.

        j is the zero-based position among joint ETs, independent of their
        jindex values. Remaining joints retain explicit global jindices;
        automatically indexed joints are renumbered to fill the gap. Joint
        limits and flip flags are preserved. The original is not modified.

        :param j: index of the joint to fix
        :param value: fixed joint coordinate, in radians/degrees or metres
        :param unit: angular unit for rotational joints, "rad" [default] or "deg"
        :returns: ETS2 with one fewer joint
        :raises ValueError: for an invalid joint position or angular unit

        Fixing a flipped revolute joint preserves the other joint as a variable:

        .. runblock:: pycon

            >>> from roboticstoolbox import ET2
            >>> ets = ET2.R(flip=True) * ET2.tx(1) * ET2.R()
            >>> fixed = ets.fixjoint(0, 90, unit="deg")
            >>> fixed.n, ets.n
            (1, 2)

        .. versionadded:: 1.5.0
        """
        # bool subclasses int, but must not select joint 0 or 1.
        if (
            isinstance(j, bool)
            or not isinstance(j, (int, np.integer))
            or not 0 <= j < self.n
        ):
            raise ValueError(f"joint index {j} out of range for {self.n} joints")
        if unit not in ("rad", "deg"):
            raise ValueError("unit must be 'rad' or 'deg'")

        parts = []
        joint_index = 0
        for et in self:
            if not et.isjoint:
                parts.append(et)
                continue

            constructor = getattr(ET2, et.kind)
            if joint_index == j:
                fixed = -value if et.isflip else value
                if et.isrotation:
                    new = constructor(fixed, unit=unit)
                else:
                    if unit != "rad":
                        raise ValueError("unit='deg' is only valid for rotational joints")
                    new = constructor(fixed)
                if et.offset != 0:
                    # the joint transforms by +-q + offset
                    new.param = new.param + et.offset
            else:
                if self._auto_jindex:
                    new = constructor(flip=et.isflip, qlim=et.qlim)
                else:
                    new = constructor(flip=et.isflip, qlim=et.qlim, jindex=et.jindex)
                if et.offset != 0:
                    new.offset = et.offset
            parts.append(new)
            joint_index += 1

        return ETS2(parts)

    def compile(self) -> ETS2:
        """
        Compile an ETS2

        :return: optimised ETS2

        Perform constant folding for faster evaluation.  Consecutive constant
        ETs are compounded, leading to a constant ET which is denoted by
        ``SE3`` when displayed.

        .. seealso:: ``isconstant``
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

    def insert(
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
        q: ArrayLike | None = None,
        base: NDArray | SE2 | None = None,
        tool: NDArray | SE2 | None = None,
        include_base: bool = True,
    ) -> SE2:
        """
        Forward kinematics

        :param q: joint coordinates, optional for a sequence with no joints
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
                ret.append(SE2(T, check=False))
        else:
            ret = SE2(fk, check=False)

        return ret

    def eval(
        self,
        q: ArrayLike | None = None,
        base: NDArray | SE2 | None = None,
        tool: NDArray | SE2 | None = None,
        include_base: bool = True,
    ) -> NDArray:
        """
        Forward kinematics (returns raw ndarray)

        :param q: joint coordinates, optional for a sequence with no joints
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

        if q is None:
            q = []

        q = self._resolve_q(q, allow_trajectory=True)
        l, _ = q.shape
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

        for k, qk in enumerate(q):
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
                # ret.append(SE2(Tk, check=False))
            else:
                T = Tk
                # ret = SE2(Tk, check=False)

        return T

    def jacob0(
        self,
        q: ArrayLike,
    ) -> NDArray:
        """
        Manipulator Jacobian in the base frame

        :param q: the joint coordinates
        :returns: the Jacobian, rows are :math:`(v_x, v_y, \\omega)`, one column
            per joint

        .. note:: This is a simple, not efficient, implementation.
        """
        # very inefficient implementation, just put a 1 in last row
        # if its a rotation joint
        j = 0
        J = np.zeros((3, self.n))
        etjoints = self.joint_idx()

        if any(self[i].jindex is None for i in etjoints):
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
                    q[jindex]
                )
            elif axis == "tx":
                dTdq = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
            elif axis == "ty":
                dTdq = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
            else:  # pragma: nocover
                raise TypeError("Invalid axes")

            if self[i].isflip:
                dTdq = -dTdq

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

    def jacobe(self, q: ArrayLike) -> Any:
        r"""
        Jacobian in end-effector frame

        :param q: joint coordinates
        :returns: Jacobian matrix
        :rtype: ndarray(3,n)

        ``jacobe(q)`` is the manipulator Jacobian matrix which maps joint
        velocity to end-effector spatial velocity.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, \omega)^T`
        is related to joint velocity by :math:`{}^{e}\nu = {}^{e}\mathbf{J}_0(q) \dot{q}`.

        .. seealso:: :func:`jacob0`, ``hessian0``
        """

        T = self.fkine(q, include_base=False).A
        return tr2jac2(T.T) @ self.jacob0(q)

    def ikine_LM(
        self,
        Tep: NDArray | SE2,
        q0: ArrayLike | None = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: ArrayLike | None = None,
        joint_limits: bool = True,
        seed: int | None = None,
        k: float = 1.0,
    ) -> IKSolution:
        r"""
        Levenberg-Marquardt numerical inverse kinematics solver

        :param Tep: the desired end-effector pose, an ``SE2`` or a 3x3 array, or a
            pose trajectory, an ``SE2`` with several values or an (N, 3, 3) array
        :param q0: the initial joint coordinates, a vector (n,), or a matrix (m, n)
            whose rows are the starting points of the first m searches (any
            further searches start from random coordinates within the joint
            limits)
        :param ilimit: maximum iterations allowed per search
        :param slimit: maximum search attempts before failure
        :param tol: maximum allowed residual error E
        :param mask: a 3-vector weighting the (x, y, θ) error priority, for
            example ``[1, 1, 0]`` to solve for position only
        :param joint_limits: reject solutions with joint limit violations
        :param seed: seed for the RNG used to generate random joint configurations
        :param k: gain λ of Chan's damping matrix, see below
        :returns: the solution, see :class:`~roboticstoolbox.robot.IK.IKSolution`
        :raises ValueError: ``Tep`` is not a 3x3 pose or an (N, 3, 3) trajectory,
            ``mask`` is not a 3-vector, ``q0`` has the wrong number of columns, or
            a joint has an infinite limit so no random configuration can be drawn

        ``sol = ets.ikine_LM(Tep)`` are the joint coordinates (n) corresponding to
        the planar end-effector pose ``Tep``. The solver is the planar counterpart
        of :meth:`ETS.ikine_LM`: the pose error is the 3-vector
        :math:`\vec{e} = (\Delta x, \Delta y, \Delta\theta)` in the world frame,
        the residual is :math:`E = \tfrac{1}{2} \vec{e}^\top \mat{W}_e \vec{e}`
        with :math:`\mat{W}_e = \text{diag}(\text{mask})`, and each step is

        .. math::

            \vec{q}_{k+1} = \vec{q}_k +
            \left( \mat{J}^\top \mat{W}_e \mat{J} + \lambda E_k \mat{1}_n \right)^{-1}
            \mat{J}^\top \mat{W}_e \vec{e}

        where :math:`\mat{J}` is :meth:`jacob0`. Because `E` is quadratic, `tol`
        bounds the pose error only to roughly :math:`\sqrt{2 \cdot \text{tol}}`;
        tighten it if a specific linear-scale accuracy is needed.

        Each search starts from a row of ``q0``, or a random configuration drawn
        within the joint limits, and runs for at most ``ilimit`` iterations; up
        to ``slimit`` searches are made before the solver gives up. A solution
        that lies outside the joint limits is rejected when ``joint_limits`` is
        True, and the search continues.

        **Trajectory operation**: for a trajectory of N poses the solver is run
        once per pose, each from the same starting points, and the returned
        ``q`` has shape (N, n). ``success`` is True only if every pose was
        solved, ``iterations`` and ``searches`` are summed, ``residual`` is the
        largest over the poses and ``reason`` is that of the last failed pose.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> from roboticstoolbox.ets.ET2 import R, tx
            >>> ets = R() * tx(1) * R() * tx(1)
            >>> Tep = ets.fkine([0.3, 0.7])
            >>> sol = ets.ikine_LM(Tep, seed=0)
            >>> sol.success
            >>> ets.fkine(sol.q)

        .. rubric:: References

        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        .. seealso:: :meth:`fkine` :meth:`jacob0` :meth:`ETS.ikine_LM`

        .. versionadded:: 1.5.0
        """
        if mask is None:
            We = np.eye(3)
        else:
            m = np.asarray(mask, dtype=float).flatten()
            if m.shape != (3,):
                raise ValueError("mask must be a 3-vector (x, y, θ)")
            We = np.diag(m)

        traj = False
        targets: NDArray

        if isinstance(Tep, SE2):
            if len(Tep) > 1:
                traj = True
                targets = np.array([T.A for T in Tep], dtype=float)
            else:
                targets = np.asarray(Tep.A, dtype=float)[np.newaxis]
        else:
            Tep = np.asarray(Tep, dtype=float)
            if Tep.ndim == 3 and Tep.shape[1:] == (3, 3):
                traj = True
                targets = Tep
            elif Tep.shape == (3, 3):
                targets = Tep[np.newaxis]
            else:
                raise ValueError(
                    "Tep must be an SE2, a 3x3 array or an (N, 3, 3) array"
                )

        n = self.n
        qlim = self.qlim

        if not np.all(np.isfinite(qlim)):
            bad = np.flatnonzero(~np.all(np.isfinite(qlim), axis=0))
            raise ValueError(
                f"Joint limit(s) for joint index(es) {bad.tolist()} are not "
                f"finite (qlim={qlim[:, bad].tolist()}) -- can't generate a "
                "random configuration within an infinite/undefined range."
            )

        rng = np.random.default_rng(seed)
        starts = rng.uniform(qlim[0], qlim[1], size=(slimit, n))

        if q0 is not None:
            q0a = np.asarray(q0, dtype=float)
            if q0a.ndim == 1:
                q0a = q0a[np.newaxis]
            if q0a.ndim != 2 or q0a.shape[1] != n:
                raise ValueError(f"q0 must have {n} columns, one per joint")
            rows = min(q0a.shape[0], slimit)
            starts[:rows] = q0a[:rows]

        revolute = np.array([joint.isrotation for joint in self.joints()])

        def solve_one(T: NDArray) -> IKSolution:
            theta_d = np.arctan2(T[1, 0], T[0, 0])

            def error(q: NDArray) -> tuple[NDArray, float]:
                Te = self.eval(q)
                dtheta = theta_d - np.arctan2(Te[1, 0], Te[0, 0])
                e = np.array(
                    [
                        T[0, 2] - Te[0, 2],
                        T[1, 2] - Te[1, 2],
                        np.arctan2(np.sin(dtheta), np.cos(dtheta)),
                    ]
                )
                return e, float(0.5 * e @ We @ e)

            total_i = 0
            found_with_limits = False
            linalg_error = 0
            E = np.inf
            q = starts[0]

            for search in range(slimit):
                q = starts[search].copy()
                i = 0

                # Check the starting point before taking a step, which can be
                # singular even though the requested pose is already reached.
                e, E = error(q)

                while True:
                    if E < tol:
                        q = normalise_q(q, qlim, revolute)

                        if joint_limits and not within_limits(q, qlim):
                            found_with_limits = True
                            break

                        return IKSolution(
                            q=q,
                            success=True,
                            iterations=total_i + i,
                            searches=search + 1,
                            residual=E,
                            reason="Success",
                        )

                    if i >= ilimit:
                        break

                    i += 1

                    J = self.jacob0(q)
                    A = J.T @ We @ J + k * E * np.eye(n)
                    g = J.T @ We @ e

                    try:
                        q = q + np.linalg.solve(A, g)
                    except np.linalg.LinAlgError:
                        linalg_error += 1
                        break

                    e, E = error(q)

                total_i += i

            reason = "iteration and search limit reached"

            if linalg_error:
                reason += f", {linalg_error} numpy.LinAlgError encountered"

            if found_with_limits:
                reason += ", solution found but violates joint limits"

            return IKSolution(
                q=q,
                success=False,
                iterations=total_i,
                searches=slimit,
                residual=E,
                reason=reason,
            )

        if not traj:
            return solve_one(targets[0])

        q = np.empty((targets.shape[0], n))
        success = True
        iterations = 0
        searches = 0
        residual = 0.0
        reason = ""

        for i, T in enumerate(targets):
            sol = solve_one(T)
            q[i] = sol.q
            if not sol.success:
                success = False
                reason = sol.reason
            iterations += sol.iterations
            searches += sol.searches
            residual = max(residual, sol.residual)

        return IKSolution(
            q=q,
            success=success,
            iterations=iterations,
            searches=searches,
            residual=residual,
            reason=reason,
        )
