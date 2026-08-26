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
from roboticstoolbox.ets.ET import ET
from roboticstoolbox.ets._ET import BaseET
from roboticstoolbox.ets._ETS import BaseETS, T, _dirties_fknm
from typing import overload, TypeVar
from typing import Literal as L
from roboticstoolbox.tools.types import ArrayLike, NDArray


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

        :param q: Joint coordinates -- either *global* (length
            ``max(self.jindices) + 1``, addressed by each joint's own
            ``jindex``) or *compact* (length ``self.n``, positionally
            ordered to match :meth:`joints`). These only differ when this
            ETS is one branch of a larger, branched robot -- e.g. it's what
            ``ikine_LM``/``ik_LM`` return when solving for just this ETS.
            For a whole, unbranched robot the two coincide.
        :param base: a base transform applied before the ETS
        :param tool: tool transform, optional
        :param include_base: set to True if the base transform should be considered
        :raises ValueError: if ``q``'s length matches neither interpretation
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

        q = self._resolve_q(q, allow_trajectory=True)
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

        q = self._resolve_q(q)
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

        q = self._resolve_q(q)
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

        if q is not None:
            q = self._resolve_q(q)
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

        if q is not None:
            q = self._resolve_q(q)
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

    @staticmethod
    def _template_consume(elements: list["BaseET"], template: list[str]) -> int:
        """
        Greedily match a run of ETs against an ordered template of ``kind``
        strings, where each template slot is optional but present slots must
        occur in the given order.

        :param elements: ETs to match, in the order they occur in the ETS
        :param template: ordered ``kind`` strings the elements may occupy
        :returns: number of leading ``elements`` consumed before the first
            one that fits no remaining template slot
        """
        slot = 0
        consumed = 0
        for et in elements:
            if et.kind in template[slot:]:
                slot = template.index(et.kind, slot) + 1
                consumed += 1
            else:
                break
        return consumed

    def _split_convention(self, template: list[str], joint_slots: set[int], exposed: str):
        """
        Split according to a DH-like convention.

        A segment is a 4-slot ordered ``template`` (e.g. Rz, tz, tx, Rx for
        DH); each slot is optional except that exactly one of the two
        ``joint_slots`` must be occupied by the segment's joint. Content
        between two joints must fully account for the gap in template
        order, else ``ValueError``. The end named by ``exposed`` (``"head"``
        or ``"tail"``) is reported separately and unvalidated; the other end
        is folded into the boundary segment.
        """
        idx = list(self.joint_idx())
        if len(idx) == 0:
            raise ValueError("ETS has no joints")

        def joint_slot(et: "ET") -> int:
            if et.kind not in template or template.index(et.kind) not in joint_slots:
                raise ValueError(
                    "ETS is not a valid DH/MDH parameterisation: "
                    f"{et} is not a permitted joint for this convention"
                )
            return template.index(et.kind)

        slots = [joint_slot(self[k]) for k in idx]
        start = [0] * len(idx)
        end = [0] * len(idx)

        if exposed == "head":
            gap = list(reversed(self[0 : idx[0]]))
            sub = list(reversed(template[0 : slots[0]]))
            consumed = self._template_consume(gap, sub)
            start[0] = idx[0] - consumed
            head = self[0 : start[0]]
        else:
            head = self[0:0]

        for i, k in enumerate(idx):
            hi_bound = idx[i + 1] if i + 1 < len(idx) else len(self)
            consumed = self._template_consume(
                list(self[k + 1 : hi_bound]), template[slots[i] + 1 :]
            )
            end[i] = k + 1 + consumed

            if i + 1 < len(idx):
                gap = self[end[i] : idx[i + 1]]
                gap_consumed = self._template_consume(
                    list(gap), template[0 : slots[i + 1]]
                )
                if gap_consumed != len(gap):
                    raise ValueError(
                        "ETS is not a valid DH/MDH parameterisation: "
                        f"unexpected term at index {end[i] + gap_consumed}"
                    )
                start[i + 1] = end[i]

        if exposed == "tail":
            tail = self[end[-1] :]
        else:
            end[-1] = len(self)
            tail = self[0:0]

        segments = [self.__class__(self[start[i] : end[i]]) for i in range(len(idx))]
        return segments, head, tail

    def split(self, method: str = "last") -> list["ETS"]:
        r"""
        Split ETS into link segments

        :param method: one of ``"first"``, ``"last"`` (default), ``"dh"``, or ``"mdh"``.
        :returns: ``[base, *segments, gripper]`` -- a list of length
            ``n_joints + 2``. ``base``/``gripper`` are empty ETS when the
            method has no concept of one (e.g. ``"dh"`` never populates
            ``gripper``, ``"mdh"`` never populates ``base``).

        Split an ETS into segments representing links, plus a base and gripper segment.
        Unpack with ``base, *segments, gripper = ets.split(method)``.

        The behaviour depends on the ``method`` argument:

        * ``"first"``: each link segment begins with a joint and continues upto, but not
          including the next joint. Any constant ET before the first joint are part of
          the base. There are no gripper ET, they are included in the last link segment.
        * ``"last"``: each link segment ends with a joint and include all ET after the
          previous joint. Any constant ET after the last joint are part of the gripper.
          There are no base ET, they are included in the first link segment.
        * ``"dh"``: similar to ``"first"`` but each segment is validated against the
          Denavit-Hartenberg convention: an ordered, 4-slot template
          Rz($θ_j$) tz($d_j$) tx($a_j$) Rx($α_j$), exactly one of Rz/tz being the
          segment's joint and the rest optional (but present slots must occur in
          this order). Content between two joints that cannot be accounted for by
          the template raises ``ValueError``. The base (content before the first
          joint's template slots) is unvalidated and reported separately; trailing
          content past the last joint's template is folded into the last segment.
        * ``"mdh"``: similar to ``"last"`` but validated against the modified
          Denavit-Hartenberg convention, template tx($a_{j-1}$) Rx($α_{j-1}$)
          Rz($θ_j$) tz($d_j$), same rules as ``"dh"`` mirrored: the gripper
          (content past the last joint's template slots) is unvalidated and
          reported separately; leading content before the first joint's template
          is folded into the first segment.

        .. runblock:: pycon
            >>> from roboticstoolbox.ets.ETS import *
            >>> e = tz(1) * Rx("q1") * tx(2) * Ry("q2") * ty(3) * Rz("q3") * tz(4)
            >>> base, *segments, gripper = e.split("first")
            >>> print("|".join(str(_) for _ in segments), f"+ base={str(base)}, gripper={str(gripper)}")
            >>> base, *segments, gripper = e.split("last")
            >>> print("|".join(str(_) for _ in segments), f"+ base={str(base)}, gripper={str(gripper)}")
            >>> e = tz(1) * Rz("q1") * tx(2) * Rx(90, 'deg') * Rz("q2") * tz(4) * Rz(180, 'deg') * tz("q3") * Rz(270, 'deg') * tz(4)
            >>> base, *segments, gripper = e.split("dh")
            >>> print("|".join(str(_) for _ in segments), f"+ base={str(base)}, gripper={str(gripper)}")
            >>> base, *segments, gripper = e.split("mdh")
            >>> print("|".join(str(_) for _ in segments), f"+ base={str(base)}, gripper={str(gripper)}")
        """

        match method.lower():
            case "dh":
                segments, head, tail = self._split_convention(
                    ["Rz", "tz", "tx", "Rx"], {0, 1}, exposed="head"
                )
            case "mdh":
                segments, head, tail = self._split_convention(
                    ["tx", "Rx", "Rz", "tz"], {2, 3}, exposed="tail"
                )
            case _:
                return super().split(method)  # type: ignore

        return [head] + segments + [tail]  # type: ignore

