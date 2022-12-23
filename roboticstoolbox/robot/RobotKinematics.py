"""
@author: Jesse Haviland
"""

from roboticstoolbox.robot.RobotProto import KinematicsProtocol
from roboticstoolbox.tools.types import ArrayLike, NDArray
from roboticstoolbox.robot.Link import Link
from roboticstoolbox.robot.Gripper import Gripper
from spatialmath import SE3
from typing import Union, Tuple, overload
from typing_extensions import Literal as L


class RobotKinematicsMixin:
    """
    The Robot Kinematics Mixin class

    This class contains kinematic methods for the ``robot`` class. All
    methods contained within this class have a full implementation within the
    ``ETS`` class and are simply passed through to the ``ETS`` class.

    """

    # --------------------------------------------------------------------- #
    # --------- Kinematic Methods ----------------------------------------- #
    # --------------------------------------------------------------------- #

    def fkine(
        self: KinematicsProtocol,
        q: ArrayLike,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        tool: Union[NDArray, SE3, None] = None,
        include_base: bool = True,
    ) -> SE3:
        """
        Forward kinematics

        ``T = robot.fkine(q)`` evaluates forward kinematics for the robot at
        joint configuration ``q``.

        **Trajectory operation**:
        If ``q`` has multiple rows (mxn), it is considered a trajectory and the
        result is an ``SE3`` instance with ``m`` values.

        Attributes
        ----------
        q
            Joint coordinates
        end
            end-effector or gripper to compute forward kinematics to
        start
            the link to compute forward kinematics from
        tool
            tool transform, optional

        Returns
        -------
            The transformation matrix representing the pose of the
            end-effector

        Examples
        --------
        The following example makes a ``panda`` robot object, and solves for the
        forward kinematics at the listed configuration.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda()
        >>> panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])

        Notes
        -----
        - For a robot with a single end-effector there is no need to
            specify ``end``
        - For a robot with multiple end-effectors, the ``end`` must
            be specified.
        - The robot's base tool transform, if set, is incorporated
            into the result.
        - A tool transform, if provided, is incorporated into the result.
        - Works from the end-effector link to the base

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """  # noqa

        return SE3(
            self.ets(start, end).fkine(
                q, base=self._T, tool=tool, include_base=include_base
            ),
            check=False,
        )

    def jacob0(
        self: KinematicsProtocol,
        q: ArrayLike,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        r"""
        Manipulator geometric Jacobian in the ``start`` frame

        ``robot.jacobo(q)`` is the manipulator Jacobian matrix which maps
        joint  velocity to end-effector spatial velocity expressed in the
        base frame.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        Parameters
        ----------
        q
            Joint coordinate vector
        end
            the particular link or gripper whose velocity the Jacobian
            describes, defaults to the end-effector if only one is present
        start
            the link considered as the base frame, defaults to the robots's base frame
        tool
            a static tool transformation matrix to apply to the
            end of end, defaults to None

        Returns
        -------
        J0
            Manipulator Jacobian in the ``start`` frame

        Examples
        --------
        The following example makes a ``Puma560`` robot object, and solves for the
        base-frame Jacobian at the zero joint angle configuration

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.Puma560()
        >>> puma.jacob0([0, 0, 0, 0, 0, 0])

        Notes
        -----
        - This is the geometric Jacobian as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a
            velocity twist as per the text by Lynch & Park.

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """  # noqa

        return self.ets(start, end).jacob0(q, tool=tool)

    def jacobe(
        self: KinematicsProtocol,
        q: ArrayLike,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        r"""
        Manipulator geometric Jacobian in the end-effector frame

        ``robot.jacobe(q)`` is the manipulator Jacobian matrix which maps
        joint  velocity to end-effector spatial velocity expressed in the
        ``end`` frame.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        Parameters
        ----------
        q
            Joint coordinate vector
        end
            the particular link or gripper whose velocity the Jacobian
            describes, defaults to the end-effector if only one is present
        start
            the link considered as the base frame, defaults to the robots's base frame
        tool
            a static tool transformation matrix to apply to the
            end of end, defaults to None

        Returns
        -------
        Je
            Manipulator Jacobian in the ``end`` frame

        Examples
        --------
        The following example makes a ``Puma560`` robot object, and solves for the
        end-effector frame Jacobian at the zero joint angle configuration

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.Puma560()
        >>> puma.jacobe([0, 0, 0, 0, 0, 0])

        Notes
        -----
        - This is the geometric Jacobian as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a
            velocity twist as per the text by Lynch & Park.

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """  # noqa

        return self.ets(start, end).jacobe(q, tool=tool)

    @overload
    def hessian0(
        self: KinematicsProtocol,
        q: ArrayLike = ...,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        J0: None = None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        ...

    @overload
    def hessian0(
        self: KinematicsProtocol,
        q: None = None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        J0: NDArray = ...,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        ...

    def hessian0(
        self: KinematicsProtocol,
        q=None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        J0=None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        r"""
        Manipulator Hessian

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the ``start`` frame. This
        function calulcates this based on the ETS of the robot. One of J0 or q
        is required. Supply J0 if already calculated to save computation time

        Parameters
        ----------
        q
            The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        end
            the final link/Gripper which the Hessian represents
        start
            the first link which the Hessian represents
        J0
            The manipulator Jacobian in the ``start`` frame
        tool
            a static tool transformation matrix to apply to the
            end of end, defaults to None

        Returns
        -------
        h0
            The manipulator Hessian in the ``start`` frame

        Synopsis
        --------
        This method computes the manipulator Hessian in the ``start`` frame.  If
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
        >>> panda = rtb.models.Panda()
        >>> panda.hessian0([0, -0.3, 0, -2.2, 0, 2, 0.7854])

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        """  # noqa

        return self.ets(start, end).hessian0(q, J0=J0, tool=tool)

    @overload
    def hessiane(
        self: KinematicsProtocol,
        q: ArrayLike = ...,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        Je: None = None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        ...

    @overload
    def hessiane(
        self: KinematicsProtocol,
        q: None = None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        Je: NDArray = ...,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        ...

    def hessiane(
        self: KinematicsProtocol,
        q=None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        Je=None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        r"""
        Manipulator Hessian

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the ``end`` coordinate frame. This
        function calulcates this based on the ETS of the robot. One of J0 or q
        is required. Supply J0 if already calculated to save computation time

        Parameters
        ----------
        q
            The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        end
            the final link/Gripper which the Hessian represents
        start
            the first link which the Hessian represents
        J0
            The manipulator Jacobian in the ``end`` frame
        tool
            a static tool transformation matrix to apply to the
            end of end, defaults to None

        Returns
        -------
        he
            The manipulator Hessian in ``end`` frame

        Synopsis
        --------
        This method computes the manipulator Hessian in the ``end`` frame.  If
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
        >>> panda = rtb.models.Panda()
        >>> panda.hessiane([0, -0.3, 0, -2.2, 0, 2, 0.7854])

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        """  # noqa

        return self.ets(start, end).hessiane(q, Je=Je, tool=tool)

    def partial_fkine0(
        self: KinematicsProtocol,
        q: ArrayLike,
        n: int = 3,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
    ):
        r"""
        Manipulator Forward Kinematics nth Partial Derivative

        This method computes the nth derivative of the forward kinematics where ``n`` is
        greater than or equal to 3. This is an extension of the differential kinematics
        where the Jacobian is the first partial derivative and the Hessian is the
        second.

        Parameters
        ----------
        q
            The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        end
            the final link/Gripper which the Hessian represents
        start
            the first link which the Hessian represents
        tool
            a static tool transformation matrix to apply to the
            end of end, defaults to None

        Returns
        -------
        A
            The nth Partial Derivative of the forward kinematics

        Examples
        --------
        The following example makes a ``Panda`` robot object, and solves for the
        base-effector frame 4th defivative of the forward kinematics at the given
        joint angle configuration

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda()
        >>> panda.partial_fkine0([0, -0.3, 0, -2.2, 0, 2, 0.7854], n=4)

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        """  # noqa

        return self.ets(start, end).partial_fkine0(q, n=n)

    def jacob0_analytical(
        self: KinematicsProtocol,
        q: ArrayLike,
        representation: L["rpy/xyz", "rpy/zyx", "eul", "exp"] = "rpy/xyz",
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        tool: Union[NDArray, SE3, None] = None,
    ):
        r"""
        Manipulator analytical Jacobian in the ``start`` frame

        ``robot.jacob0_analytical(q)`` is the manipulator Jacobian matrix which maps
        joint  velocity to end-effector spatial velocity expressed in the
        ``start`` frame.

        Parameters
        ----------
        q
            Joint coordinate vector
        representation
            angular representation
        end
            the particular link or gripper whose velocity the Jacobian
            describes, defaults to the base link
        start
            the link considered as the end-effector, defaults to the robots's end-effector
        tool
            a static tool transformation matrix to apply to the
            end of end, defaults to None

        Returns
        -------
        jacob0
            Manipulator Jacobian in the ``start`` frame

        Synopsis
        --------
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
        >>> puma = rtb.models.ETS.Puma560()
        >>> puma.jacob0_analytical([0, 0, 0, 0, 0, 0])

        """  # noqa

        return self.ets(start, end).jacob0_analytical(
            q, tool=tool, representation=representation
        )

    # --------------------------------------------------------------------- #
    # --------- IK Methods ------------------------------------------------ #
    # --------------------------------------------------------------------- #

    def ik_lm_chan(
        self: KinematicsProtocol,
        Tep: Union[NDArray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[NDArray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[NDArray, None] = None,
        λ: float = 1.0,
    ) -> Tuple[NDArray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Chan's Method)

        Parameters
        ----------
        Tep
            The desired end-effector pose or pose trajectory
        end
            the particular link or gripper to compute the pose of
        start
            the link considered as the base frame, defaults to the robots's base frame
        q0
            initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        ilimit
            maximum number of iterations per search
        slimit
            maximum number of search attempts
        tol
            final error tolerance
        reject_jl
            constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        we
            a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        λ
            value of lambda for the damping matrix Wn

        Returns
        -------
        sol
            inverse kinematic solution

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

        Notes
        -----
        - See `Toolbox kinematics wiki page
            <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
        - Implements a Levenberg-Marquadt variable-damping solver.
        - The tolerance is computed on the norm of the error between
            current and desired tool pose.  This norm is computed from
            distances and angles without any kind of weighting.
        - The inverse kinematic solution is generally not unique, and
            depends on the initial guess ``q0``.

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        See Also
        --------
            TODO
        """  # noqa

        return self.ets(start, end).ik_lm_chan(
            Tep, q0, ilimit, slimit, tol, reject_jl, we, λ
        )

    def ik_lm_wampler(
        self: KinematicsProtocol,
        Tep: Union[NDArray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[NDArray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[NDArray, None] = None,
        λ: float = 1.0,
    ) -> Tuple[NDArray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Wamplers's Method)

        Parameters
        ----------
        Tep
            The desired end-effector pose or pose trajectory
        end
            the particular link or gripper to compute the pose of
        start
            the link considered as the base frame, defaults to the robots's base frame
        q0
            initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        ilimit
            maximum number of iterations per search
        slimit
            maximum number of search attempts
        tol
            final error tolerance
        reject_jl
            constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        we
            a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        λ
            value of lambda for the damping matrix Wn

        Returns
        -------
        sol
            inverse kinematic solution

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



        Notes
        -----
        - See `Toolbox kinematics wiki page
            <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
        - Implements a Levenberg-Marquadt variable-damping solver.
        - The tolerance is computed on the norm of the error between
            current and desired tool pose.  This norm is computed from
            distances and angles without any kind of weighting.
        - The inverse kinematic solution is generally not unique, and
            depends on the initial guess ``q0``.

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        See Also
        --------
            TODO
        """  # noqa

        return self.ets(start, end).ik_lm_wampler(
            Tep, q0, ilimit, slimit, tol, reject_jl, we, λ
        )

    def ik_lm_sugihara(
        self: KinematicsProtocol,
        Tep: Union[NDArray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[NDArray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[NDArray, None] = None,
        λ: float = 1.0,
    ) -> Tuple[NDArray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Sugihara's Method)

        Parameters
        ----------
        Tep
            The desired end-effector pose or pose trajectory
        end
            the particular link or gripper to compute the pose of
        start
            the link considered as the base frame, defaults to the robots's base frame
        q0
            initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        ilimit
            maximum number of iterations per search
        slimit
            maximum number of search attempts
        tol
            final error tolerance
        reject_jl
            constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        we
            a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        λ
            value of lambda for the damping matrix Wn

        Returns
        -------
        sol
            inverse kinematic solution

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



        Notes
        -----
        - See `Toolbox kinematics wiki page
            <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
        - Implements a Levenberg-Marquadt variable-damping solver.
        - The tolerance is computed on the norm of the error between
            current and desired tool pose.  This norm is computed from
            distances and angles without any kind of weighting.
        - The inverse kinematic solution is generally not unique, and
            depends on the initial guess ``q0``.

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        See Also
        --------
            TODO
        """  # noqa

        return self.ets(start, end).ik_lm_sugihara(
            Tep, q0, ilimit, slimit, tol, reject_jl, we, λ
        )

    def ik_nr(
        self: KinematicsProtocol,
        Tep: Union[NDArray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[NDArray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[NDArray, None] = None,
        use_pinv: int = True,
        pinv_damping: float = 0.0,
    ) -> Tuple[NDArray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Newton-Raphson Method)

        Parameters
        ----------
        Tep
            The desired end-effector pose or pose trajectory
        end
            the particular link or gripper to compute the pose of
        start
            the link considered as the base frame, defaults to the robots's base frame
        q0
            initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        ilimit
            maximum number of iterations per search
        slimit
            maximum number of search attempts
        tol
            final error tolerance
        reject_jl
            constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        we
            a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        λ
            value of lambda for the damping matrix Wn

        Returns
        -------
        sol
            inverse kinematic solution

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



        Notes
        -----
        - See `Toolbox kinematics wiki page
            <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
        - Implements a Levenberg-Marquadt variable-damping solver.
        - The tolerance is computed on the norm of the error between
            current and desired tool pose.  This norm is computed from
            distances and angles without any kind of weighting.
        - The inverse kinematic solution is generally not unique, and
            depends on the initial guess ``q0``.

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        See Also
        --------
            TODO
        """  # noqa

        return self.ets(start, end).ik_nr(
            Tep, q0, ilimit, slimit, tol, reject_jl, we, use_pinv, pinv_damping
        )

    def ik_gn(
        self: KinematicsProtocol,
        Tep: Union[NDArray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[NDArray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[NDArray, None] = None,
        use_pinv: int = True,
        pinv_damping: float = 0.0,
    ) -> Tuple[NDArray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Gauss-NewtonMethod)

        Parameters
        ----------
        Tep
            The desired end-effector pose or pose trajectory
        end
            the particular link or gripper to compute the pose of
        start
            the link considered as the base frame, defaults to the robots's base frame
        q0
            initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        ilimit
            maximum number of iterations per search
        slimit
            maximum number of search attempts
        tol
            final error tolerance
        reject_jl
            constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        we
            a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        λ
            value of lambda for the damping matrix Wn

        Returns
        -------
        sol
            inverse kinematic solution

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



        Notes
        -----
        - See `Toolbox kinematics wiki page
            <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
        - Implements a Levenberg-Marquadt variable-damping solver.
        - The tolerance is computed on the norm of the error between
            current and desired tool pose.  This norm is computed from
            distances and angles without any kind of weighting.
        - The inverse kinematic solution is generally not unique, and
            depends on the initial guess ``q0``.

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        See Also
        --------
            TODO
        """  # noqa

        return self.ets(start, end).ik_gn(
            Tep, q0, ilimit, slimit, tol, reject_jl, we, use_pinv, pinv_damping
        )

    def ikine_LM(
        self: KinematicsProtocol,
        Tep: Union[NDArray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[ArrayLike, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: Union[ArrayLike, None] = None,
        joint_limits: bool = True,
        seed: Union[int, None] = None,
        k: float = 1.0,
        method="chan",
        kq: float = 0.0,
        km: float = 0.0,
        ps: float = 0.0,
        pi: Union[NDArray, float] = 0.3,
        **kwargs,
    ):
        r"""
        Levenberg-Marquadt Numerical Inverse Kinematics Solver

        A method which provides functionality to perform numerical inverse kinematics (IK)
        using the Levemberg-Marquadt method.

        See the :ref:`Inverse Kinematics Docs Page <IK>` for more details and for a
        **tutorial** on numerical IK, see `here <https://bit.ly/3ak5GDi>`_.

        Parameters
        ----------
        Parameters
        ----------
        Tep
            The desired end-effector pose
        end
            the link considered as the end-effector
        start
            the link considered as the base frame, defaults to the robots's base frame
        q0
            The initial joint coordinate vector
        ilimit
            How many iterations are allowed within a search before a new search
            is started
        slimit
            How many searches are allowed before being deemed unsuccessful
        tol
            Maximum allowed residual error E
        mask
            A 6 vector which assigns weights to Cartesian degrees-of-freedom
            error priority
        joint_limits
            Reject solutions with joint limit violations
        seed
            A seed for the private RNG used to generate random joint coordinate
            vectors
        k
            Sets the gain value for the damping matrix Wn in the next iteration. See
            synopsis
        method
            One of "chan", "sugihara" or "wampler". Defines which method is used
            to calculate the damping matrix Wn in the ``step`` method
        kq
            The gain for joint limit avoidance. Setting to 0.0 will remove this
            completely from the solution
        km
            The gain for maximisation. Setting to 0.0 will remove this completely
            from the solution
        ps
            The minimum angle/distance (in radians or metres) in which the joint is
            allowed to approach to its limit
        pi
            The influence angle/distance (in radians or metres) in null space motion
            becomes active

        Synopsis
        --------
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
        The following example makes a ``panda`` robot object, makes a goal
        pose ``Tep``, and then solves for the joint coordinates which result in the pose
        ``Tep`` using the `ikine_LM` method.

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.Panda()
        >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        >>> panda.ikine_LM(Tep)

        Notes
        -----
        The value for the ``k`` kwarg will depend on the ``method`` chosen and the arm you are
        using. Use the following as a rough guide ``chan, k = 1.0 - 0.01``,
        ``wampler, k = 0.01 - 0.0001``, and ``sugihara, k = 0.1 - 0.0001``

        When using the this method, the initial joint coordinates :math:`q_0`, should correspond
        to a non-singular manipulator pose, since it uses the manipulator Jacobian.

        This class supports null-space motion to assist with maximising manipulability and
        avoiding joint limits. These are enabled by setting kq and km to non-zero values.

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part II:
          Acceleration and Advanced Applications." arXiv preprint arXiv:2207.01794 (2022).

        See Also
        --------
        :py:class:`~roboticstoolbox.robot.IK.IK_LM`
            An IK Solver class which implements the Levemberg Marquadt optimisation technique
        ikine_NR
            Implements the :py:class:`~roboticstoolbox.robot.IK.IK_NR` class as a method within the :py:class:`Robot` class
        ikine_GN
            Implements the :py:class:`~roboticstoolbox.robot.IK.IK_GN` class as a method within the :py:class:`Robot` class
        ikine_QP
            Implements the :py:class:`~roboticstoolbox.robot.IK.IK_QP` class as a method within the :py:class:`Robot` class


        .. versionchanged:: 1.0.3
            Added the Levemberg-Marquadt IK solver method on the `Robot` class

        """  # noqa

        return self.ets(start, end).ikine_LM(
            Tep=Tep,
            q0=q0,
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
