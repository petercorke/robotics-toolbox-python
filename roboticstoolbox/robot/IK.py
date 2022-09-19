#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Tuple, Union
import roboticstoolbox as rtb
from dataclasses import dataclass
from spatialmath import SE3

ArrayLike = Union[list, np.ndarray, tuple, set]

# from roboticstoolbox.tools.null import null
# import roboticstoolbox as rtb
# from spatialmath import base
# from spatialmath import SE3
# import scipy.optimize as opt
# import math

# from roboticstoolbox.tools.p_servo import p_servo

# iksol = namedtuple("IKsolution", "q, success, reason, iterations, residual",
#     defaults=(None, False, None, None, None)) # Py >= 3.7 only
# iksol = namedtuple("IKsolution", "q, success, reason, iterations, residual, jl_valid")


# try:
#     import qpsolvers as qp

#     _qp = True
# except ImportError:  # pragma nocover
#     _qp = False


# ===================================================================== #


@dataclass
class IKSolution:
    """
    A dataclass for representing an IK solution

    :param q: The joint coordinates of the solution (ndarray). Note that these
        will not be valid if failed to find a solution
    :param success: True if a valid solution was found
    :param iterations: How many iterations were performed
    :param searches: How many searches were performed
    :param residual: The final error value from the cost function
    :param reason: The reason the IK problem failed if applicable
    """

    q: np.ndarray
    success: bool
    iterations: int
    searches: int
    residual: float
    reason: str

    def __str__(self):
        if self.success:
            return f"IKSolution: q={np.round(self.q, 4)}, success=True, iterations={self.iterations}, searches={self.searches}, residual={self.residual}"
        else:
            return f"IKSolution: q={np.round(self.q, 4)}, success=False, reason={self.reason}, iterations={self.iterations}, searches={self.searches}, residual={np.round(self.residual, 4)}"


class IKSolver(ABC):
    """
    An abstract super class which provides basic functionality to perform numerical inverse
    kinematics (IK). Superclasses can inherit this class and implement the solve method.
    """

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: Union[ArrayLike, None] = None,
        joint_limits: bool = True,
    ):
        """
        name: The name of the IK algorithm
        ilimit: How many iterations are allowed within a search before a new search is started
        slimit: How many searches are allowed before being deemed unsuccessful
        tol: Maximum allowed residual error E
        mask: A 6 vector which assigns weights to Cartesian degrees-of-freedom
        problems: Total number of IK problems within the experiment
        joint_limits: Reject solutions with joint limit violations

        λΣ: The gain for joint limit avoidance. Setting to 0.0 will remove this completely from the solution
        λm: The gain for maximisation. Setting to 0.0 will remove this completely from the solution
        ps: The minimum angle/distance (in radians or metres) in which the joint is allowed to approach to its limit
        pi: The influence angle/distance (in radians or metres) in null space motion becomes active
        """

        # Solver parameters
        self.name = name
        self.slimit = slimit
        self.ilimit = ilimit
        self.tol = tol

        if mask is None:
            mask = np.ones(6)

        self.We = np.diag(mask)
        self.joint_limits = joint_limits

    def solve(
        self, ets: "rtb.ETS", Tep: np.ndarray, q0: Union[ArrayLike, None]
    ) -> IKSolution:
        """
        This method will attempt to solve the IK problem and obtain joint coordinates
        which result the the end-effector pose Tep.

        :return: An IKSolution dataclass:
            :param q: The joint coordinates of the solution (ndarray). Note that these
                will not be valid if failed to find a solution
            :param success: True if a valid solution was found
            :param iterations: How many iterations were performed
            :param searches: How many searches were performed
            :param residual: The final error value from the cost function
            :param jl_valid: True if q is inbounds of the robots joint limits
            :param reason: The reason the IK problem failed if applicable
        """

        if q0 is None:
            q0 = ets.random_q(self.slimit)
        elif not isinstance(q0, np.ndarray):
            q0 = np.array(q0)

        if q0.ndim == 1:
            q0_new = ets.random_q(self.slimit)
            q0_new[0] = q0
            q0 = q0_new

        # Iteration count
        i = 0
        total_i = 0

        # Initialise variables
        E = 0.0
        q = q0[0]
        jl_valid = False

        for search in range(self.slimit):
            q = q0[search].copy()

            while i <= self.ilimit:
                i += 1

                # Attempt a step
                try:
                    E, q = self.step(ets, Tep, q)

                except np.linalg.LinAlgError:
                    # Abandon search and try again
                    break

                # Check if we have arrived
                if E < self.tol:

                    # Wrap q to be within +- 180 deg
                    # If your robot has larger than 180 deg range on a joint
                    # this line should be modified in incorporate the extra range
                    q = (q + np.pi) % (2 * np.pi) - np.pi

                    # Check if we have violated joint limits
                    jl_valid = self.check_jl(ets, q)

                    if not jl_valid and self.joint_limits:
                        # Abandon search and try again
                        break
                    else:
                        return IKSolution(
                            q=q,
                            success=True,
                            iterations=total_i + i,
                            searches=search + 1,
                            residual=E,
                            reason="Success",
                        )

            total_i += i
            i = 0

        # If we make it here, then we have failed
        reason = "ilimit and slimit reached"

        if E < self.tol:
            reason += ", solution found but violates joint limits"
        else:
            reason += ", no solution found"

        return IKSolution(
            q=q,
            success=False,
            iterations=total_i,
            searches=self.slimit,
            residual=E,
            reason=reason,
        )

    def error(self, Te: np.ndarray, Tep: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates the engle axis error between current end-effector pose Te and
        the desired end-effector pose Tep. Also calulates the quadratic error E
        which is weighted by the diagonal matrix We.

        Returns a tuple:
        e: angle-axis error (ndarray in R^6)
        E: The quadratic error weighted by We
        """
        e = rtb.angle_axis(Te, Tep)
        E = 0.5 * e @ self.We @ e

        return e, E

    def check_jl(self, ets: "rtb.ETS", q: np.ndarray):
        """
        Checks if the joints are within their respective limits

        Returns a True if joints within feasible limits otherwise False
        """

        # Loop through the joints in the ETS
        for i in range(ets.n):

            # Get the corresponding joint limits
            ql0 = ets.qlim[0, i]
            ql1 = ets.qlim[1, i]

            # Check if q exceeds the limits
            if q[i] < ql0 or q[i] > ql1:
                return False

        # If we make it here, all the joints are fine
        return True

    @abstractmethod
    def step(
        self, ets: "rtb.ETS", Tep: np.ndarray, q: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Superclasses will implement this method to perform a step of the implemented
        IK algorithm

        :return: A tuple containing
            :param E: The new error value
            :param q: The new joint coordinate vector

        """
        pass


def null_Σ(ets: "rtb.ETS", q: np.ndarray, ps: float, pi: Optional[np.ndarray]):
    """
    Formulates a relationship between joint limits and the joint velocity.
    When this is projected into the null-space of the differential kinematics
    to attempt to avoid exceeding joint limits

    q: The joint coordinates of the robot
    ps: The minimum angle/distance (in radians or metres) in which the joint is
        allowed to approach to its limit
    pi: The influence angle/distance (in radians or metres) in which the velocity
        damper becomes active

    returns: Σ
    """

    # If pi wasn't supplied, set it to some default value
    if pi is None:
        pi = 0.3 * np.ones(ets.n)

    # Add cost to going in the direction of joint limits, if they are within
    # the influence distance
    Σ = np.zeros((ets.n, 1))

    for i in range(ets.n):
        qi = q[i]
        ql0 = ets.qlim[0, i]
        ql1 = ets.qlim[1, i]

        if qi - ql0 <= pi[i]:
            Σ[i, 0] = -np.power(((qi - ql0) - pi[i]), 2) / np.power((ps - pi[i]), 2)
        if ql1 - qi <= pi[i]:
            Σ[i, 0] = np.power(((ql1 - qi) - pi[i]), 2) / np.power((ps - pi[i]), 2)

    return -Σ


def calc_qnull(
    ets: "rtb.ETS",
    q: np.ndarray,
    J: np.ndarray,
    λΣ: float,
    λm: float,
    ps: float,
    pi: Optional[np.ndarray],
):
    """
    Calculates the desired null-space motion according to the gains λΣ and λm.
    This is a helper method that is used within the `step` method of an IK solver

    Returns qnull: the desired null-space motion
    """

    qnull_grad = np.zeros(ets.n)
    qnull = np.zeros(ets.n)

    # Add the joint limit avoidance if the gain is above 0
    if λΣ > 0:
        Σ = null_Σ(ets, q, ps, pi)
        qnull_grad += (1.0 / λΣ * Σ).flatten()

    # Add the manipulability maximisation if the gain is above 0
    if λm > 0:
        Jm = ets.jacobm(q)
        qnull_grad += (1.0 / λm * Jm).flatten()

    # Calculate the null-space motion
    if λΣ > 0 or λΣ > 0:
        null_space = np.eye(ets.n) - np.linalg.pinv(J) @ J
        qnull = null_space @ qnull_grad

    return qnull.flatten()


class IK_NR(IKSolver):
    def __init__(
        self,
        pinv=False,
        λΣ: float = 0.0,
        λm: float = 0.0,
        ps: float = 0.1,
        pi: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pinv = pinv
        self.λΣ = λΣ
        self.λm = λm
        self.ps = ps
        self.pi = pi

        self.name = f"NR (pinv={pinv})"

        if self.λΣ > 0.0:
            self.name += " Σ"

        if self.λm > 0.0:
            self.name += " Jm"

    def step(self, ets: "rtb.ETS", Tep: np.ndarray, q: np.ndarray):
        Te = ets.eval(q)
        e, E = self.error(Te, Tep)

        J = ets.jacob0(q)

        # Null-space motion
        qnull = calc_qnull(ets, q, J, self.λΣ, self.λm, self.ps, self.pi)

        if self.pinv:
            q += np.linalg.pinv(J) @ e + qnull
        else:
            q += np.linalg.inv(J) @ e + qnull

        return E, q


class IK_LM(IKSolver):
    def __init__(
        self,
        k=1.0,
        method="chan",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if method.lower().startswith("sugi"):
            self.method = 1
            method_name = "Sugihara"
        elif method.lower().startswith("wamp"):
            self.method = 2
            method_name = "Wampler"
        else:
            self.method = 0
            method_name = "Chan"

        self.k = k

        self.name = f"LM ({method_name} λ={k})"

    def step(self, ets: "rtb.ETS", Tep: np.ndarray, q: np.ndarray):
        Te = ets.eval(q)
        e, E = self.error(Te, Tep)

        if self.method == 1:
            # Sugihara's method
            Wn = E * np.eye(ets.n) + self.k * np.eye(ets.n)
        elif self.method == 2:
            # Wampler's method
            Wn = self.k * np.eye(ets.n)
        else:
            # Chan's method
            Wn = self.k * E * np.eye(ets.n)

        J = ets.jacob0(q)
        g = J.T @ self.We @ e

        q += np.linalg.inv(J.T @ self.We @ J + Wn) @ g

        return E, q


# class IKMixin:
#     def ikine_NR(
#         self: Union["rtb.ETS", "IKMixin"],
#         Tep: Union[np.ndarray, SE3],
#         q0: Union[np.ndarray, None] = None,
#         ilimit: int = 30,
#         slimit: int = 100,
#         tol: float = 1e-6,
#         joint_limits: bool = True,
#         we: Union[np.ndarray, None] = None,
#         λΣ: float = 0.0,
#         λm: float = 0.0,
#         ps: float = 0.1,
#         pi: Optional[np.ndarray] = None,
#     ):
#         solver = NR(
#             ilimit=ilimit,
#             slimit=slimit,
#             tol=tol,
#             joint_limits=joint_limits,
#             we=we,
#             λΣ=λΣ,
#             λm=λm,
#             ps=ps,
#             pi=pi,
#         )

#         if isinstance(Tep, SE3):
#             Tep = Tep.A

#         return solver.solve(ets=self, Tep=Tep, q0=q0)

#     def ikine_LM(
#         self: Union["rtb.ETS", "IKMixin"],
#         Tep: Union[np.ndarray, SE3],
#         q0: Union[np.ndarray, None] = None,
#         ilimit: int = 30,
#         slimit: int = 100,
#         tol: float = 1e-6,
#         joint_limits: bool = True,
#         mask: Union[np.ndarray, None] = None,
#     ):
#         solver = NR(
#             ilimit=ilimit,
#             slimit=slimit,
#             tol=tol,
#             joint_limits=joint_limits,
#             mask=mask,
#         )

#         if isinstance(Tep, SE3):
#             Tep = Tep.A

#         return solver.solve(ets=self, Tep=Tep, q0=q0)


# class IKMixin:
#     def ikine_mmc(self, T, q0=None):

#         if not _qp:
#             raise ImportError(
#                 "the package qpsolvers is required for this function. \nInstall using 'pip install qpsolvers'"
#             )

#         arrived = False

#         n = self.n
#         q = self.q
#         dt = 0.05

#         e_prev = 100000
#         q_last = q
#         gain = 1000.0

#         while not arrived:

#             Te = self.fkine(q)
#             eTep = Te.inv() * T
#             e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

#             if e < e_prev:
#                 # good update
#                 # gain = gain * 2
#                 # dt = dt * 2
#                 # print('Up')
#                 pass
#             else:
#                 # bad update
#                 # gain = gain / 2
#                 dt = dt / 2
#                 q = q_last
#                 # print('Down')

#             e_prev = e
#             q_last = q
#             # print(gain)
#             # print(self.manipulability(q))
#             # print(e)

#             v, arrived = p_servo(Te, T, gain=gain, threshold=0.000001)

#             # Gain term (lambda) for control minimisation
#             Y = 0.01

#             # Quadratic component of objective function
#             Q = np.eye(n + 6)

#             # Joint velocity component of Q
#             Q[:n, :n] *= Y

#             # Slack component of Q
#             Q[n:, n:] = (1 / e) * np.eye(6)

#             # The equality contraints
#             Aeq = np.c_[self.jacobe(q), np.eye(6)]
#             beq = v.reshape((6,))

#             # The inequality constraints for joint limit avoidance
#             Ain = np.zeros((n + 6, n + 6))
#             bin = np.zeros(n + 6)

#             # The minimum angle (in radians) in which the joint is allowed to
#             # approach to its limit
#             ps = 0.05

#             # The influence angle (in radians) in which the velocity damper
#             # becomes active
#             pi = 0.9

#             Ain[:n, :n], bin[:n] = self.joint_velocity_damper(ps, pi, n)
#             c = np.r_[-self.jacobm(q).reshape((n,)), np.zeros(6)]
#             # lb = -np.r_[self.qdlim[:n], 10 * np.ones(6)]
#             # ub = np.r_[self.qdlim[:n], 10 * np.ones(6)]

#             # Solve for the joint velocities dq
#             qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq)

#             for i in range(self.n):
#                 q[i] += qd[i] * (dt)

#         return q

#     # --------------------------------------------------------------------- #

#     def ikine_LM(
#         self,
#         T,
#         q0=None,
#         mask=None,
#         ilimit=500,
#         rlimit=100,
#         tol=1e-10,
#         L=0.1,
#         Lmin=0,
#         search=False,
#         slimit=100,
#         transpose=None,
#         end=None,
#     ):
#         """
#         Numerical inverse kinematics by Levenberg-Marquadt optimization
#         (Robot superclass)

#         :param T: The desired end-effector pose or pose trajectory
#         :type T: SE3
#         :param q0: initial joint configuration (default all zeros)
#         :type q0: ndarray(n)
#         :param mask: mask vector that correspond to translation in X, Y and Z
#             and rotation about X, Y and Z respectively.
#         :type mask: ndarray(6)
#         :param ilimit: maximum number of iterations (default 500)
#         :type ilimit: int
#         :param rlimit: maximum number of consecutive step rejections
#             (default 100)
#         :type rlimit: int
#         :param tol: final error tolerance (default 1e-10)
#         :type tol: float
#         :param L: initial value of lambda
#         :type L: float (default 0.1)
#         :param Lmin: minimum allowable value of lambda
#         :type Lmin: float (default 0)
#         :param search: search over all configurations
#         :type search: bool
#         :param slimit: maximum number of search attempts
#         :type slimit: int (default 100)
#         :param transpose: use Jacobian transpose with step size A, rather
#             than Levenberg-Marquadt
#         :type transpose: float
#         :return: inverse kinematic solution
#         :rtype: named tuple

#         ``sol = robot.ikine_LM(T)`` are the joint coordinates (n) corresponding
#         to the robot end-effector pose ``T`` which is an ``SE3`` object. This
#         method can be used for robots with any number of degrees of freedom.
#         The return value ``sol`` is a named tuple with elements:

#         ============    ==========  ===============================================
#         Element         Type        Description
#         ============    ==========  ===============================================
#         ``q``           ndarray(n)  joint coordinates in units of radians or metres
#         ``success``     bool        whether a solution was found
#         ``reason``      str         reason for the failure
#         ``iterations``  int         number of iterations
#         ``residual``    float       final value of cost function
#         ============    ==========  ===============================================

#         If ``success=False`` the ``q`` values will be valid numbers, but the
#         solution will be in error.  The amount of error is indicated by
#         the ``residual``.

#         **Trajectory operation**:

#         If ``len(T) = m > 1`` it is considered to be a trajectory, then the
#         result is a named tuple whose elements are

#         ============    ============   ===============================================
#         Element         Type           Description
#         ============    ============   ===============================================
#         ``q``           ndarray(m,n)   joint coordinates in units of radians or metres
#         ``success``     bool(m)        whether a solution was found
#         ``reason``      list of str    reason for the failure
#         ``iterations``  ndarray(m)     number of iterations
#         ``residual``    ndarray(m)     final value of cost function
#         ============    ============   ===============================================

#         The initial estimate of q for each time step is taken as the solution
#         from the previous time step.

#         **Underactuated robots:**

#         For the case where the manipulator has fewer than 6 DOF the
#         solution space has more dimensions than can be spanned by the
#         manipulator joint coordinates.

#         In this case we specify the ``mask`` option where the ``mask`` vector
#         (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
#         will be ignored in reaching a solution.  The mask vector has six
#         elements that correspond to translation in X, Y and Z, and rotation
#         about X, Y and Z respectively. The value should be 0 (for ignore)
#         or 1. The number of non-zero elements must equal the number of
#         manipulator DOF.

#         For example when using a 3 DOF manipulator tool orientation might
#         be unimportant, in which case use the option ``mask=[1 1 1 0 0 0]``.

#         **Global search**:

#         ``sol = robot.ikine_LM(T, search=True)`` as above but peforms a
#         brute-force search with initial conditions chosen randomly from the
#         entire configuration space.  If a numerical solution is found from that
#         initial condition, it is returned, otherwise another initial condition
#         is chosen.

#         .. note::

#             - See `Toolbox kinematics wiki page
#                 <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
#             - Implements a Levenberg-Marquadt variable-damping solver.
#             - The tolerance is computed on the norm of the error between
#               current and desired tool pose.  This norm is computed from
#               distances and angles without any kind of weighting.
#             - The inverse kinematic solution is generally not unique, and
#               depends on the initial guess ``q0``.
#             - The default value of ``q0`` is zero which is a poor choice for
#               most manipulators since it often corresponds to a
#               kinematic singularity.
#             - Such a solution is completely general, though much less
#               efficient than analytic inverse kinematic solutions derived
#               symbolically.
#             - This approach allows a solution to be obtained at a singularity,
#               but the joint angles within the null space are arbitrarily
#               assigned.
#             - Joint offsets, if defined, are accounted for in the solution.
#             - Joint limits are not considered in this solution.
#             - If the search option is used any prismatic joint must have
#               joint limits defined.

#         :references:
#             - Robotics, Vision & Control, P. Corke, Springer 2011,
#               Section 8.4.

#         :seealso: :func:`ikine_LMS`, :func:`ikine_unc`, :func:`ikine_con`,
#             :func:`ikine_min`
#         """  # noqa E501

#         if not isinstance(T, SE3):
#             raise TypeError("argument must be SE3")

#         if isinstance(self, rtb.DHRobot):
#             end = None

#         solutions = []

#         if search:
#             # Randomised search for a starting point
#             # quiet = True

#             qlim = self.qlim
#             qspan = qlim[1] - qlim[0]  # range of joint motion

#             for k in range(slimit):
#                 # choose a random joint coordinate
#                 q0_k = np.random.rand(self.n) * qspan + qlim[0, :]
#                 print("search starts at ", q0_k)

#                 # recurse into the solver
#                 solution = self.ikine_LM(
#                     T[0],
#                     q0_k,
#                     mask,
#                     ilimit,
#                     rlimit,
#                     tol,
#                     L,
#                     Lmin,
#                     False,
#                     slimit,
#                     transpose,
#                 )

#                 if solution.success:
#                     q0 = solution.q
#                     if len(T) == 1:
#                         # we're done
#                         return solution
#                     else:
#                         # more to do on the trajectory
#                         solutions.append(solution)
#                         del T[0]
#             # no solution found, stop now
#             return iksol(None, False, None, None, None)

#         if q0 is None:
#             q0 = np.zeros((self.n,))
#         else:
#             q0 = base.getvector(q0, self.n)

#         if mask is not None:
#             mask = base.getvector(mask, 6)
#             if not self.n >= np.sum(mask):
#                 raise ValueError(
#                     "Number of robot DOF must be >= the number "
#                     "of 1s in the mask matrix"
#                 )
#         else:
#             mask = np.ones(6)
#         W = np.diag(mask)

#         tcount = 0  # Total iteration count
#         rejcount = 0  # Rejected step count
#         nm = 0
#         revolutes = self.revolutejoints

#         q = q0
#         for Tk in T:
#             iterations = 0
#             Li = L  # lambda
#             failure = None
#             while True:
#                 # Update the count and test against iteration limit
#                 iterations += 1

#                 if iterations > ilimit:
#                     failure = f"iteration limit {ilimit} exceeded"
#                     break

#                 e = base.tr2delta(self.fkine(q, end=end).A, Tk.A)

#                 # Are we there yet?
#                 if base.norm(W @ e) < tol:
#                     break

#                 # Compute the Jacobian
#                 J = self.jacobe(q, end=end)

#                 JtJ = J.T @ W @ J

#                 if transpose is not None:
#                     # Do the simple Jacobian transpose with constant gain
#                     dq = transpose * J.T @ e  # lgtm [py/multiple-definition]
#                     q += dq
#                 else:
#                     # Do the damped inverse Gauss-Newton with
#                     # Levenberg-Marquadt
#                     # dq = np.linalg.inv(
#                     #     JtJ + ((Li + Lmin) * np.eye(self.n))
#                     # ) @ J.T @ W @ e
#                     dq = (
#                         np.linalg.inv(JtJ + ((Li + Lmin) * np.diag(np.diag(JtJ))))
#                         @ J.T
#                         @ W
#                         @ e
#                     )
#                     # print(J.T @ W @ e)

#                     # Compute possible new value of
#                     qnew = q + dq

#                     # And figure out the new error
#                     enew = base.tr2delta(self.fkine(qnew, end=end).A, Tk.A)

#                     # Was it a good update?
#                     if np.linalg.norm(W @ enew) < np.linalg.norm(W @ e):
#                         # Step is accepted
#                         q = qnew
#                         e = enew
#                         Li /= 2
#                         rejcount = 0
#                     else:
#                         # Step is rejected, increase the damping and retry
#                         Li *= 2
#                         rejcount += 1
#                         if rejcount > rlimit:
#                             failure = f"rejected-step limit {rlimit} exceeded"
#                             break

#                 # Wrap angles for revolute joints
#                 k = np.logical_and(q > np.pi, revolutes)
#                 q[k] -= 2 * np.pi

#                 k = np.logical_and(q < -np.pi, revolutes)
#                 q[k] += +2 * np.pi

#                 nm = np.linalg.norm(W @ e)
#                 # qs = ", ".join(["{:8.3f}".format(qi) for qi in q])
#                 # print(f"λ={Li:8.2g}, |e|={nm:8.2g}: q={qs}")

#             # LM process finished, for better or worse
#             # failure will be None or an error message
#             solution = iksol(q, failure is None, failure, iterations, nm)
#             solutions.append(solution)

#             tcount += iterations

#         if len(T) == 1:
#             return solutions[0]
#         else:
#             return iksol(
#                 np.vstack([sol.q for sol in solutions]),
#                 np.array([sol.success for sol in solutions]),
#                 [sol.reason for sol in solutions],
#                 np.array([sol.iterations for sol in solutions]),
#                 np.array([sol.residual for sol in solutions]),
#             )

#     # --------------------------------------------------------------------- #

#     def ikine_LMS(
#         self, T, q0=None, mask=None, ilimit=500, tol=1e-10, wN=1e-3, Lmin=0, end=None
#     ):
#         """
#         Numerical inverse kinematics by Levenberg-Marquadt optimization
#         (Robot superclass)

#         :param T: The desired end-effector pose or pose trajectory
#         :type T: SE3
#         :param q0: initial joint configuration (default all zeros)
#         :type q0: ndarray(n)
#         :param mask: mask vector that correspond to translation in X, Y and Z
#             and rotation about X, Y and Z respectively.
#         :type mask: ndarray(6)
#         :param ilimit: maximum number of iterations (default 500)
#         :type ilimit: int
#         :param tol: final error tolerance (default 1e-10)
#         :type tol: float
#         :param ωN: damping coefficient
#         :type ωN: float (default 1e-3)
#         :return: inverse kinematic solution
#         :rtype: named tuple

#         ``sol = robot.ikine_LM(T)`` are the joint coordinates (n) corresponding
#         to the robot end-effector pose ``T`` which is an ``SE3`` object. This
#         method can be used for robots with any number of degrees of freedom.
#         The return value ``sol`` is a named tuple with elements:

#         ============    ==========  ===============================================
#         Element         Type        Description
#         ============    ==========  ===============================================
#         ``q``           ndarray(n)  joint coordinates in units of radians or metres
#         ``success``     bool        whether a solution was found
#         ``reason``      str         reason for the failure
#         ``iterations``  int         number of iterations
#         ``residual``    float       final value of cost function
#         ============    ==========  ===============================================

#         If ``success=False`` the ``q`` values will be valid numbers, but the
#         solution will be in error.  The amount of error is indicated by
#         the ``residual``.

#         **Trajectory operation**:

#         If ``len(T) = m > 1`` it is considered to be a trajectory, then the
#         result is a named tuple whose elements are

#         ============    ============   ===============================================
#         Element         Type           Description
#         ============    ============   ===============================================
#         ``q``           ndarray(m,n)   joint coordinates in units of radians or metres
#         ``success``     bool(m)        whether a solution was found
#         ``reason``      list of str    reason for the failure
#         ``iterations``  ndarray(m)     number of iterations
#         ``residual``    ndarray(m)     final value of cost function
#         ============    ============   ===============================================

#         **Underactuated robots:**

#         For the case where the manipulator has fewer than 6 DOF the
#         solution space has more dimensions than can be spanned by the
#         manipulator joint coordinates.

#         In this case we specify the ``mask`` option where the ``mask`` vector
#         (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
#         will be ignored in reaching a solution.  The mask vector has six
#         elements that correspond to translation in X, Y and Z, and rotation
#         about X, Y and Z respectively. The value should be 0 (for ignore)
#         or 1. The number of non-zero elements should equal the number of
#         manipulator DOF.

#         For example when using a 3 DOF manipulator rotation orientation might
#         be unimportant in which case use the option: mask = [1 1 1 0 0 0].

#         .. note::

#             - See `Toolbox kinematics wiki page
#                 <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
#             - Implements a modified Levenberg-Marquadt variable-damping solver
#               which is quite robust in practice.
#             - Similar to ``ikine_LM`` but uses a different error metric
#             - The tolerance is computed on the norm of the error between
#               current and desired tool pose.  This norm is computed from
#               distances and angles without any kind of weighting.
#             - The inverse kinematic solution is generally not unique, and
#               depends on the initial guess ``q0``.
#             - The default value of ``q0`` is zero which is a poor choice for
#               most manipulators since it often corresponds to a
#               kinematic singularity.
#             - Such a solution is completely general, though much less
#               efficient than analytic inverse kinematic solutions derived
#               symbolically.
#             - This approach allows a solution to be obtained at a singularity,
#               but the joint angles within the null space are arbitrarily
#               assigned.
#             - Joint offsets, if defined, are accounted for in the solution.
#             - Joint limits are not considered in this solution.

#         :references:
#             - "Solvability-Unconcerned Inverse Kinematics by the
#               Levenberg–Marquardt Method", T. Sugihara, IEEE T-RO, 27(5),
#               October 2011, pp. 984-991.

#         :seealso: :func:`ikine_LM`, :func:`ikine_unc`, :func:`ikine_con`,
#             :func:`ikine_min`
#         """  # noqa E501

#         if not isinstance(T, SE3):
#             raise TypeError("argument must be SE3")

#         if isinstance(self, rtb.DHRobot):
#             end = None

#         solutions = []

#         if q0 is None:
#             q0 = np.zeros((self.n,))
#         else:
#             q0 = base.getvector(q0, self.n)

#         if mask is not None:
#             mask = base.getvector(mask, 6)
#             if not self.n >= np.sum(mask):
#                 raise ValueError(
#                     "Number of robot DOF must be >= the number "
#                     "of 1s in the mask matrix"
#                 )
#         else:
#             mask = np.ones(6)
#         W = np.diag(mask)

#         tcount = 0  # Total iteration count
#         revolutes = self.revolutejoints

#         q = q0
#         for Tk in T:
#             iterations = 0
#             failure = None
#             while True:
#                 # Update the count and test against iteration limit
#                 iterations += 1

#                 if iterations > ilimit:
#                     failure = f"iteration limit {ilimit} exceeded"
#                     break

#                 e = _angle_axis(self.fkine(q, end=end).A, Tk.A)

#                 # Are we there yet?
#                 E = 0.5 * e.T @ W @ e
#                 if E < tol:
#                     break

#                 # Compute the Jacobian and projection matrices
#                 J = self.jacob0(q, end=end)
#                 WN = E * np.eye(self.n) + wN * np.eye(self.n)
#                 H = J.T @ W @ J + WN  # n x n
#                 g = J.T @ W @ e  # n x 1

#                 # Compute new value of q
#                 q += np.linalg.inv(H) @ g  # n x 1
#                 # print(np.linalg.norm(np.linalg.inv(H) @ g))
#                 # print(e)
#                 # print(g)
#                 # print(q)
#                 # print(J)

#                 # Wrap angles for revolute joints
#                 k = np.logical_and(q > np.pi, revolutes)
#                 q[k] -= 2 * np.pi

#                 k = np.logical_and(q < -np.pi, revolutes)
#                 q[k] += +2 * np.pi

#                 # qs = ", ".join(["{:8.3f}".format(qi) for qi in q])
#                 # print(f"|e|={E:8.2g}, det(H)={np.linalg.det(H)}: q={qs}")

#             # LM process finished, for better or worse
#             # failure will be None or an error message
#             solution = iksol(q, failure is None, failure, iterations, E)
#             solutions.append(solution)

#             tcount += iterations

#         if len(T) == 1:
#             return solutions[0]
#         else:
#             return iksol(
#                 np.vstack([sol.q for sol in solutions]),
#                 np.array([sol.success for sol in solutions]),
#                 [sol.reason for sol in solutions],
#                 np.array([sol.iterations for sol in solutions]),
#                 np.array([sol.residual for sol in solutions]),
#             )

#     # --------------------------------------------------------------------- #

#     def ikine_min(
#         self,
#         T,
#         q0=None,
#         qlim=False,
#         ilimit=1000,
#         tol=1e-16,
#         method=None,
#         stiffness=0,
#         costfun=None,
#         options={},
#         end=None,
#     ):
#         r"""
#         Inverse kinematics by optimization with joint limits (Robot superclass)

#         :param T: The desired end-effector pose or pose trajectory
#         :type T: SE3
#         :param q0: initial joint configuration (default all zeros)
#         :type q0: ndarray(n)
#         :param qlim: enforce joint limits
#         :type qlim: bool
#         :param ilimit: Iteration limit (default 1000)
#         :type ilimit: int
#         :param tol: Tolerance (default 1e-16)
#         :type tol: tol
#         :param method: minimization method to use
#         :type method: str
#         :param stiffness: Stiffness used to impose a smoothness contraint on
#             joint angles, useful when n is large (default 0)
#         :type stiffness: float
#         :param costfun: User supplied cost term, optional
#         :type costfun: callable
#         :return: inverse kinematic solution
#         :rtype: named tuple

#         ``sol = robot.ikine_min(T)`` are the joint coordinates (n)
#         corresponding to the robot end-effector pose T which is an SE3 object.
#         The return value ``sol`` is a named tuple with elements:

#         ============    ==========  ============================================================
#         Element         Type        Description
#         ============    ==========  ============================================================
#         ``q``           ndarray(n)  joint coordinates in units of radians or metres, or ``None``
#         ``success``     bool        whether a solution was found
#         ``reason``      str         reason for the failure
#         ``iterations``  int         number of iterations
#         ``residual``    float       final value of cost function
#         ============    ==========  ============================================================

#         **Minimization method**:

#         By default this method uses:

#         - the Scipy ``SLSQP`` (Sequential Least Squares Programming) minimizer
#           for the case of no joint limits
#         - the Scipy ``trust-constr`` minimizer for the case with joint limits.
#           This gives good results but is very slow.  An alternative is
#           ``L-BFGS-B`` (Broyden–Fletcher–Goldfarb–Shanno) but for redundant
#           robots can sometimes give poor results, pushing against the joint
#           limits when there is no need to.

#         In both case the function to be minimized is the squared norm of a
#         vector :math:`[d,a]` with components respectively the
#         translation error and rotation error in Euler vector form, between the
#         desired pose and the current estimate obtained by inverse kinematics.

#         **Additional cost terms**:

#         This method supports two additional costs:

#         - ``stiffness`` imposes a penalty on joint variation
#           :math:`\sum_{j=1}^N (q_j - q_{j-1})^2` which tends to keep the
#           arm straight
#         - ``costfun`` add a cost given by a user-specified function
#           ``costfun(q)``

#         **Trajectory operation**:

#         If ``len(T) > 1`` it is considered to be a trajectory, and the result
#         is a list of named tuples such that ``sol[k]`` corresponds to
#         ``T[k]``. The initial estimate of q for each time step is taken as the
#         solution from the previous time step.

#         .. note::

#             - See `Toolbox kinematics wiki page
#                 <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
#             - Uses ``SciPy.minimize`` with bounds.
#             - Joint limits are considered in this solution.
#             - Can be used for robots with arbitrary degrees of freedom.
#             - The inverse kinematic solution is generally not unique, and
#               depends on the initial guess ``q0``.
#             - The default value of ``q0`` is zero which is a poor choice for
#               most manipulators since it often corresponds to a
#               kinematic singularity.
#             - Such a solution is completely general, though much less
#               efficient than analytic inverse kinematic solutions derived
#               symbolically.
#             - The objective function (error) is
#               :math:`\sum \left( (\mat{T}^{-1} \cal{K}(\vec{q}) - \mat{1} ) \mat{\Omega} \right)^2`
#               where :math:`\mat{\Omega}` is a diagonal matrix.
#             - Joint offsets, if defined, are accounted for in the solution.

#         .. warning::

#             - The objective function is rather uncommon.
#             - Order of magnitude slower than ``ikine_LM`` or ``ikine_LMS``, it
#               uses a scalar cost-function and does not provide a Jacobian.

#         :author: Bryan Moutrie, for RTB-MATLAB

#         :seealso: :func:`ikine_LM`, :func:`ikine_LMS`, :func:`ikine_unc`,
#             :func:`ikine_min`

#         """  # noqa E501

#         if not isinstance(T, SE3):
#             raise TypeError("argument must be SE3")

#         if isinstance(self, rtb.DHRobot):
#             end = None

#         if q0 is None:
#             q0 = np.zeros((self.n))
#         else:
#             q0 = base.getvector(q0, self.n)

#         solutions = []

#         wr = 1 / self.reach
#         weight = np.r_[wr, wr, wr, 1, 1, 1]

#         optdict = {"maxiter": ilimit}
#         if options is not None and isinstance(options, dict):
#             optdict.update(options)
#         else:
#             raise ValueError("options must be a dict")

#         if qlim:
#             # dealing with joint limits
#             bounds = opt.Bounds(self.qlim[0, :], self.qlim[1, :])

#             if method is None:
#                 method = "trust-constr"
#         else:
#             # no joint limits
#             if method is None:
#                 method = "SLSQP"
#             bounds = None

#         def cost(q, T, weight, costfun, stiffness):
#             # T, weight, costfun, stiffness = args
#             e = _angle_axis(self.fkine(q, end=end).A, T) * weight
#             E = (e**2).sum()

#             if stiffness > 0:
#                 # Enforce a continuity constraint on joints, minimum bend
#                 E += np.sum(np.diff(q) ** 2) * stiffness

#             if costfun is not None:
#                 E += (e**2).sum() + costfun(q)

#             return E

#         for Tk in T:
#             res = opt.minimize(
#                 cost,
#                 q0,
#                 args=(Tk.A, weight, costfun, stiffness),
#                 bounds=bounds,
#                 method=method,
#                 tol=tol,
#                 options=options,
#             )

#             # trust-constr seems to work better than L-BFGS-B which often
#             # runs a joint up against its limit and terminates with position
#             # error.
#             # but 'truts-constr' is 5x slower

#             solution = iksol(res.x, res.success, res.message, res.nit, res.fun)
#             solutions.append(solution)
#             q0 = res.x  # use this solution as initial estimate for next time

#         if len(T) == 1:
#             return solutions[0]
#         else:
#             return solutions

#     # --------------------------------------------------------------------- #

#     def ikine_global(
#         self, T, qlim=False, ilimit=1000, tol=1e-16, method=None, options={}, end=None
#     ):
#         r"""
#         .. warning:: Experimental code for using SciPy global optimizers.

#         Each global optimizer has quite a different call signature, so final
#         design will need a bit of thought.

#         """

#         # basinhopping:
#         # brute: ranges, finish=None
#         # differential_evolution:  bounds, tol
#         # shgo: bounds, options:f_tol
#         # dual_annealing: bounds

#         if not isinstance(T, SE3):
#             raise TypeError("argument must be SE3")

#         if isinstance(self, rtb.DHRobot):
#             end = None

#         solutions = []

#         # wr = 1 / self.reach
#         # weight = np.r_[wr, wr, wr, 1, 1, 1]

#         optdict = {}

#         if method is None:
#             method = "differential-evolution"

#         if method == "brute":
#             # requires a tuple of tuples
#             optdict["ranges"] = tuple([tuple(li.qlim) for li in self])
#         else:
#             optdict["bounds"] = tuple([tuple(li.qlim) for li in self])

#         if method not in [
#             "basinhopping",
#             "brute",
#             "differential_evolution",
#             "shgo",
#             "dual_annealing",
#         ]:
#             raise ValueError("unknown global optimizer requested")

#         global_minimizer = opt.__dict__[method]

#         def cost(q, T, weight):
#             # T, weight, costfun, stiffness = args
#             e = _angle_axis(self.fkine(q, end=end).A, T) * weight
#             return (e**2).sum()

#         for _ in T:
#             res = global_minimizer(cost, **optdict)

#             solution = iksol(res.x, res.success, res.message, res.nit, res.fun)
#             solutions.append(solution)

#             # q0 was not used so I commented it out
#             # q0 = res.x  # use this solution as initial estimate for next time

#         if len(T) == 1:
#             return solutions[0]
#         else:
#             return solutions


# def _angle_axis(T, Td):
#     d = base.transl(Td) - base.transl(T)
#     R = base.t2r(Td) @ base.t2r(T).T
#     li = np.r_[R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]

#     if base.iszerovec(li):
#         # diagonal matrix case
#         if np.trace(R) > 0:
#             # (1,1,1) case
#             a = np.zeros((3,))
#         else:
#             a = np.pi / 2 * (np.diag(R) + 1)
#     else:
#         # non-diagonal matrix case
#         ln = base.norm(li)
#         a = math.atan2(ln, np.trace(R) - 1) * li / ln

#     return np.r_[d, a]


# def _angle_axis_sekiguchi(T, Td):
#     d = base.transl(Td) - base.transl(T)
#     R = base.t2r(Td) @ base.t2r(T).T
#     li = np.r_[R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]

#     if base.iszerovec(li):
#         # diagonal matrix case
#         if np.trace(R) > 0:
#             # (1,1,1) case
#             a = np.zeros((3,))
#         else:
#             # (1, -1, -1), (-1, 1, -1) or (-1, -1, 1) case
#             a = np.pi / 2 * (np.diag(R) + 1)
#             # as per Sekiguchi paper
#             if R[1, 0] > 0 and R[2, 1] > 0 and R[0, 2] > 0:
#                 a = np.pi / np.sqrt(2) * np.sqrt(np.diag(R) + 1)
#             elif R[1, 0] > 0:  # (2)
#                 a = np.pi / np.sqrt(2) * np.sqrt(np.diag(R) @ np.r_[1, 1, -1] + 1)
#             elif R[0, 2] > 0:  # (3)
#                 a = np.pi / np.sqrt(2) * np.sqrt(np.diag(R) @ np.r_[1, -1, 1] + 1)
#             elif R[2, 1] > 0:  # (4)
#                 a = np.pi / np.sqrt(2) * np.sqrt(np.diag(R) @ np.r_[-1, 1, 1] + 1)
#     else:
#         # non-diagonal matrix case
#         ln = base.norm(li)
#         a = math.atan2(ln, np.trace(R) - 1) * li / ln

#     return np.r_[d, a]


# if __name__ == "__main__":  # pragma nocover

#     # import roboticstoolbox as rtb
#     # from spatialmath import SE3

#     # np.set_printoptions(
#     # linewidth=120, formatter={'float': lambda x: f"{x:9.5g}"
#     # if abs(x) > 1e-10 else f"{0:9.5g}"})

#     robot = rtb.models.DH.Panda()

#     T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
#     # sol = robot.ikine_LMS(T)         # solve IK
#     # print(sol)                    # display joint angles

#     # print(T)
#     # print(robot.fkine(sol.q))
#     # robot.plot(sol.q)

#     # sol = robot.ikine_unc(
#     # T, costfun=lambda q: q[1] * 1e-6 if q[1] > 0 else -q[1])
#     # print(sol)
#     # print(robot.fkine(sol.q))
#     # robot.plot(sol.q)

#     sol = robot.ikine_global(T, method="brute")
