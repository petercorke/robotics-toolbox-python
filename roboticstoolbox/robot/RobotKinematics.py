"""
@author: Jesse Haviland
"""

from roboticstoolbox.robot.RobotProto import KinematicsProtocol
from roboticstoolbox.tools.types import ArrayLike, NDArray
from roboticstoolbox.robot.Link import Link
from roboticstoolbox.robot.Gripper import Gripper
from spatialmath import SE3
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    TypeVar,
    Union,
    Dict,
    Tuple,
    overload,
    Literal as L,
)


class RobotKinematicsMixin:

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
        """

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
        """

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
        """

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
        """

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
        """

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

        """

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
