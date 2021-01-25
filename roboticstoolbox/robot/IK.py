#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
from collections import namedtuple
from roboticstoolbox.tools.null import null
from spatialmath import base
from spatialmath import SE3, Twist3
import scipy.optimize as opt
import math
import qpsolvers as qp
from roboticstoolbox.tools.p_servo import p_servo
# iksol = namedtuple("IKsolution", "q, success, reason, iterations, residual",
#     defaults=(None, False, None, None, None)) # Py >= 3.7 only
iksol = namedtuple("IKsolution", "q, success, reason, iterations, residual")

# ===================================================================== #


class IKMixin:

    def ikine_mmc(
                self, T,
                q0=None):

        arrived = False

        n = self.n
        q = self.q
        dt = 0.05

        e_prev = 100000
        q_last = q
        gain = 1000.0

        while not arrived:

            Te = self.fkine(q)
            eTep = Te.inv() * T
            e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))

            if e < e_prev:
                # good update
                # gain = gain * 2
                # dt = dt * 2
                # print('Up')
                pass
            else:
                # bad update
                # gain = gain / 2
                dt = dt / 2
                # q = q_last
                # print('Down')

            e_prev = e
            q_last = q
            # print(gain)
            # print(self.manipulability(q))
            # print(e)

            v, arrived = p_servo(Te, T, gain=gain, threshold=0.000001)

            # Gain term (lambda) for control minimisation
            Y = 0.01

            # Quadratic component of objective function
            Q = np.eye(n + 6)

            # Joint velocity component of Q
            Q[:n, :n] *= Y

            # Slack component of Q
            Q[n:, n:] = (1 / e) * np.eye(6)

            # The equality contraints
            Aeq = np.c_[self.jacobe(q), np.eye(6)]
            beq = v.reshape((6,))

            # The inequality constraints for joint limit avoidance
            Ain = np.zeros((n + 6, n + 6))
            bin = np.zeros(n + 6)

            # The minimum angle (in radians) in which the joint is allowed to
            # approach to its limit
            ps = 0.05

            # The influence angle (in radians) in which the velocity damper
            # becomes active
            pi = 0.9

            Ain[:n, :n], bin[:n] = self.joint_velocity_damper(ps, pi, n)
            c = np.r_[-self.jacobm(q).reshape((n,)), np.zeros(6)]
            # lb = -np.r_[self.qdlim[:n], 10 * np.ones(6)]
            # ub = np.r_[self.qdlim[:n], 10 * np.ones(6)]

            # Solve for the joint velocities dq
            qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq)

            for i in range(self.n):
                q[i] += qd[i] * (dt)

        return q

# --------------------------------------------------------------------- #

    def ikine_LM(
            self, T,
            q0=None,
            mask=None,
            ilimit=500,
            rlimit=100,
            tol=1e-10,
            L=0.1,
            Lmin=0,
            search=False,
            slimit=100,
            transpose=None):

        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Robot superclass)

        :param T: The desired end-effector pose or pose trajectory
        :type T: SE3
        :param q0: initial joint configuration (default all zeros)
        :type q0: ndarray(n)
        :param mask: mask vector that correspond to translation in X, Y and Z
            and rotation about X, Y and Z respectively.
        :type mask: ndarray(6)
        :param ilimit: maximum number of iterations (default 500)
        :type ilimit: int
        :param rlimit: maximum number of consecutive step rejections (default 100)
        :type rlimit: int
        :param tol: final error tolerance (default 1e-10)
        :type tol: float
        :param L: initial value of lambda
        :type L: float (default 0.1)
        :param Lmin: minimum allowable value of lambda
        :type Lmin: float (default 0)
        :param search: search over all configurations
        :type search: bool
        :param slimit: maximum number of search attempts
        :type slimit: int (default 100)
        :param transpose: use Jacobian transpose with step size A, rather
            than Levenberg-Marquadt
        :type transpose: float
        :return: inverse kinematic solution
        :rtype: named tuple

        ``sol = robot.ikine_LM(T)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``T`` which is an ``SE3`` object. This method can
        be used for robots with any number of degrees of freedom. The return
        value ``sol`` is a named tuple with elements:

        ============    ==========  ============================================================
        Element         Type        Description
        ============    ==========  ============================================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres, or ``None``
        ``success``     bool        whether a solution was found
        ``reason``      str         reason for the failure
        ``iterations``  int         number of iterations
        ``residual``    float       final value of cost function
        ============    ==========  ============================================================

        **Trajectory operation**:

        If ``len(T) > 1`` it is considered to be a trajectory, and the result is
        a list of named tuples such that ``sol[k]`` corresponds to ``T[k]``. The
        initial estimate of q for each time step is taken as the solution from
        the previous time step.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``mask`` option where the ``mask`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The mask vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value should be 0 (for ignore)
        or 1. The number of non-zero elements must equal the number of
        manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``mask=[1 1 1 0 0 0]``.

        **Global search**:

        ``sol = robot.ikine_LM(T, search=True)`` as above but peforms a
        brute-force search with initial conditions chosen randomly from the
        entire configuration space.  If a numerical solution is found from that
        initial condition, it is returned, otherwise another initial condition is
        chosen.

        .. note::

            - See `Toolbox kinematics wiki page <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-step-size solver.
            - The tolerance is computed on the norm of the error between
              current and desired tool pose.  This norm is computed from
              distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
              depends on the initial guess ``q0``.
            - The default value of ``q0`` is zero which is a poor choice for 
              most manipulators since it often corresponds to a
              kinematic singularity.
            - Such a solution is completely general, though much less
              efficient than analytic inverse kinematic solutions derived
              symbolically.
            - This approach allows a solution to be obtained at a singularity,
              but the joint angles within the null space are arbitrarily
              assigned.
            - Joint offsets, if defined, are accounted for in the solution.
            - Joint limits are not considered in this solution.
            - If the search option is used any prismatic joint must have
              joint limits defined.

        :references:
            - Robotics, Vision & Control, P. Corke, Springer 2011,
              Section 8.4.

        :seealso: :func:`ikine_LMS`, :func:`ikine_unc`, :func:`ikine_con`, :func:`ikine_min`
        """

        if not isinstance(T, SE3):
            T = SE3(T)

        solutions = []

        if search:
            # Randomised search for a starting point
            # quiet = True

            qlim = self.qlim
            qspan = qlim[1] - qlim[0]  # range of joint motion

            for k in range(slimit):
                # choose a random joint coordinate
                q0_k = np.random.rand(self.n) * qspan + qlim[0, :]

                # recurse into the solver
                solution = self.ikine_LM(
                    T[0],
                    q0_k,
                    mask,
                    ilimit,
                    rlimit,
                    tol,
                    L,
                    Lmin,
                    False,
                    slimit,
                    transpose)

                if solution.success:
                    q0 = solution.q
                    if len(T) == 1:
                        # we're done
                        return solution
                    else:
                        # more to do on the trajectory
                        solutions.append(solution)
                        del T[0]
            else:
                # no solution found, stop now
                return iksol(None, False, None, None, None)

        if q0 is None:
            q0 = np.zeros((self.n,))
        else:
            q0 = base.getvector(q0, self.n)

        if mask is not None:
            mask = base.getvector(mask, 6)
            if not self.n >= np.sum(mask):
                raise ValueError('Number of robot DOF must be >= the number '
                                 'of 1s in the mask matrix')
        else:
            mask = np.ones(6)
        W = np.diag(mask)

        tcount = 0    # Total iteration count
        rejcount = 0  # Rejected step count
        nm = 0

        # bool vector indicating revolute joints
        revolutes = np.array([link.isrevolute for link in self])

        q = q0
        for Tk in T:
            iterations = 0
            Li = L  # lambda
            failure = None
            while True:
                # Update the count and test against iteration limit
                iterations += 1

                if iterations > ilimit:
                    failure = f"iteration limit {ilimit} exceeded"
                    break

                e = base.tr2delta(self.fkine(q).A, Tk.A)

                # Are we there yet?
                if base.norm(W @ e) < tol:
                    break

                # Compute the Jacobian
                J = self.jacobe(q)

                JtJ = J.T @ W @ J

                if transpose is not None:
                    # Do the simple Jacobian transpose with constant gain
                    dq = transpose * J.T @ e    # lgtm [py/multiple-definition]
                    q += dq
                else:
                    # Do the damped inverse Gauss-Newton with
                    # Levenberg-Marquadt
                    # dq = np.linalg.inv(
                    #     JtJ + ((Li + Lmin) * np.eye(self.n))
                    # ) @ J.T @ W @ e
                    dq = np.linalg.inv(
                        JtJ + ((Li + Lmin) * np.diag(np.diag(JtJ)))
                    ) @ J.T @ W @ e
                    # print(J.T @ W @ e)

                    # Compute possible new value of
                    qnew = q + dq

                    # And figure out the new error
                    enew = base.tr2delta(self.fkine(qnew).A, Tk.A)

                    # Was it a good update?
                    if np.linalg.norm(W @ enew) < np.linalg.norm(W @ e):
                        # Step is accepted
                        q = qnew
                        e = enew
                        Li /= 2
                        rejcount = 0
                    else:
                        # Step is rejected, increase the damping and retry
                        Li *= 2
                        rejcount += 1
                        if rejcount > rlimit:
                            failure = f"rejected-step limit {rlimit} exceeded"
                            break

                # Wrap angles for revolute joints
                k = np.logical_and(q > np.pi, revolutes)
                q[k] -= 2 * np.pi

                k = np.logical_and(q < -np.pi, revolutes)
                q[k] += + 2 * np.pi

                nm = np.linalg.norm(W @ e)
                qs = ", ".join(["{:8.3f}".format(qi) for qi in q])
                # print(f"λ={Li:8.2g}, |e|={nm:8.2g}: q={qs}")

            # LM process finished, for better or worse
            # failure will be None or an error message
            solution = iksol(q, failure is None, failure, iterations, nm)
            solutions.append(solution)

            tcount += iterations

        if len(T) == 1:
            return solutions[0]
        else:
            return solutions

# --------------------------------------------------------------------- #

    def ikine_LMS(
            self, T,
            q0=None,
            mask=None,
            ilimit=500,
            tol=1e-10,
            wN=1e-3,
            Lmin=0
            ):

        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Robot superclass)

        :param T: The desired end-effector pose or pose trajectory
        :type T: SE3
        :param q0: initial joint configuration (default all zeros)
        :type q0: ndarray(n)
        :param mask: mask vector that correspond to translation in X, Y and Z
            and rotation about X, Y and Z respectively.
        :type mask: ndarray(6)
        :param ilimit: maximum number of iterations (default 500)
        :type ilimit: int 
        :param tol: final error tolerance (default 1e-10)
        :type tol: float 
        :param ωN: damping coefficient
        :type ωN: float (default 1e-3)
        :return: inverse kinematic solution
        :rtype: named tuple

        ``sol = robot.ikine_LM(T)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``T`` which is an ``SE3`` object. This method can
        be used for robots with any number of degrees of freedom. The return
        value ``sol`` is a named tuple with elements:

        ============    ==========  ============================================================
        Element         Type        Description
        ============    ==========  ============================================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres, or ``None``
        ``success``     bool        whether a solution was found
        ``reason``      str         reason for the failure
        ``iterations``  int         number of iterations
        ``residual``    float       final value of cost function
        ============    ==========  ============================================================

        **Trajectory operation**:

        If ``len(T) > 1`` it is considered to be a trajectory, and the result is
        a list of named tuples such that ``sol[k]`` corresponds to ``T[k]``. The
        initial estimate of q for each time step is taken as the solution from
        the previous time step.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``mask`` option where the ``mask`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The mask vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value should be 0 (for ignore)
        or 1. The number of non-zero elements should equal the number of
        manipulator DOF.

        For example when using a 3 DOF manipulator rotation orientation might
        be unimportant in which case use the option: mask = [1 1 1 0 0 0].

        .. note::

            - See `Toolbox kinematics wiki page <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a modified Levenberg-Marquadt variable-step-size solver
              which is quite robust in practice.
            - The tolerance is computed on the norm of the error between
              current and desired tool pose.  This norm is computed from
              distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
              depends on the initial guess ``q0``.
            - The default value of ``q0`` is zero which is a poor choice for 
              most manipulators since it often corresponds to a
              kinematic singularity.
            - Such a solution is completely general, though much less
              efficient than analytic inverse kinematic solutions derived
              symbolically.
            - This approach allows a solution to be obtained at a singularity,
              but the joint angles within the null space are arbitrarily
              assigned.
            - Joint offsets, if defined, are accounted for in the solution.
            - Joint limits are not considered in this solution.

        :references:
            - "Solvability-Unconcerned Inverse Kinematics by the
              Levenberg–Marquardt Method", T. Sugihara, IEEE T-RO, 27(5), 
              October 2011, pp. 984-991.

        :seealso: :func:`ikine_LM`, :func:`ikine_unc`, :func:`ikine_con`, :func:`ikine_min`
        """

        if not isinstance(T, SE3):
            T = SE3(T)

        solutions = []

        if q0 is None:
            q0 = np.zeros((self.n,))
        else:
            q0 = base.getvector(q0, self.n)

        if mask is not None:
            mask = base.getvector(mask, 6)
            if not self.n >= np.sum(mask):
                raise ValueError('Number of robot DOF must be >= the number '
                                'of 1s in the mask matrix')
        else:
            mask = np.ones(6)
        W = np.diag(mask)

        tcount = 0  # Total iteration count

        # bool vector indicating revolute joints
        revolutes = np.array([link.isrevolute for link in self])

        q = q0
        for Tk in T:
            iterations = 0
            failure = None
            while True:
                # Update the count and test against iteration limit
                iterations += 1

                if iterations > ilimit:
                    failure = f"iteration limit {ilimit} exceeded"
                    break

                e = _angle_axis(self.fkine(q).A, Tk.A)

                # Are we there yet?
                E = 0.5 * e.T @ W @ e
                if E < tol:
                    break

                # Compute the Jacobian and projection matrices
                J = self.jacob0(q)
                WN = E * np.eye(self.n) + wN * np.eye(self.n)
                H = J.T @ W @ J + WN  # n x n
                g = J.T @ W @ e       # n x 1

                # Compute new value of q
                q += np.linalg.inv(H) @ g  # n x 1
                # print(np.linalg.norm(np.linalg.inv(H) @ g))
                # print(e)
                # print(g)
                # print(q)
                # print(J)

                # Wrap angles for revolute joints
                k = np.logical_and(q > np.pi, revolutes)
                q[k] -= 2 * np.pi

                k = np.logical_and(q < -np.pi, revolutes)
                q[k] += + 2 * np.pi

                qs = ", ".join(["{:8.3f}".format(qi) for qi in q])
                # print(f"|e|={E:8.2g}, det(H)={np.linalg.det(H)}: q={qs}")

            # LM process finished, for better or worse
            # failure will be None or an error message
            solution = iksol(q, failure is None, failure, iterations, E)
            solutions.append(solution)

            tcount += iterations

        if len(T) == 1:
            return solutions[0]
        else:
            return solutions


# --------------------------------------------------------------------- #

    # def ikine_unc(self, T, q0=None, ilimit=1000, tol=1e-16, stiffness=0, costfun=None):
    #     r"""
    #     Inverse manipulator by optimization without joint limits (Robot
    #     superclass)

    #     :param T: The desired end-effector pose or pose trajectory
    #     :type T: SE3
    #     :param q0: initial joint configuration (default all zeros)
    #     :type q0: ndarray(n)
    #     :param tol: Tolerance (default 1e-16)
    #     :type tol: tol
    #     :param ilimit: Iteration limit (default 1000)
    #     :type ilimit: int
    #     :param stiffness: Stiffness used to impose a smoothness contraint on
    #         joint angles, useful when n is large (default 0)
    #     :type stiffness: float
    #     :param costfun: User supplied cost term, optional
    #     :type costfun: callable
    #     :return: inverse kinematic solution
    #     :rtype: named tuple

    #     ``sol = robot.ikine_unc(T)`` are the joint coordinates (n) corresponding
    #     to the robot end-effector pose T which is an SE3 object.  The
    #     return value ``sol`` is a named tuple with elements:

    #     ============    ==========  ============================================================
    #     Element         Type        Description
    #     ============    ==========  ============================================================
    #     ``q``           ndarray(n)  joint coordinates in units of radians or metres, or ``None``
    #     ``success``     bool        whether a solution was found
    #     ``reason``      str         reason for the failure
    #     ``iterations``  int         number of iterations
    #     ``residual``    float       final value of cost function
    #     ============    ==========  ============================================================

    #     This method the Scipy SLSQP minimizer to minimize the squared norm of a
    #     vector :math:`[d,a]` with components respectively the translation error
    #     and rotation error in Euler vector form, between the desired pose and
    #     the current estimate obtained by inverse kinematics.

    #     **Additional cost terms**:

    #     This method supports two additional costs:

    #     - ``stiffness`` imposes a penalty on joint variation :math:`\sum_{j=1}^N (q_j - q_{j-1})^2`
    #       which tends to keep the arm straight
    #     - ``costfun`` add a cost given by a user-specified function ``costfun(q)``

    #     **Trajectory operation**:

    #     If ``len(T) > 1`` it is considered to be a trajectory, and the result is
    #     a list of named tuples such that ``sol[k]`` corresponds to ``T[k]``. The
    #     initial estimate of q for each time step is taken as the solution from
    #     the previous time step.


    #     .. note::

    #         - Uses ``SciPy.minimize`` SLSQP without bounds.
    #         - Joint limits are not considered in this solution.
    #         - Can be used for robots with arbitrary degrees of freedom.
    #         - The inverse kinematic solution is generally not unique, and
    #           depends on the initial guess ``q0``.
    #         - The default value of ``q0`` is zero which is a poor choice for 
    #           most manipulators since it often corresponds to a
    #           kinematic singularity.
    #         - Such a solution is completely general, though much less
    #           efficient than analytic inverse kinematic solutions derived
    #           symbolically.
    #         - The objective function (error) is 
    #           :math:`\sum \left( (\mat{T}^{-1} \cal{K}(\vec{q}) - \mat{1} ) \mat{\Omega} \right)^2`
    #           where :math:`\mat{\Omega}` is a diagonal matrix.
    #         - Joint offsets, if defined, are accounted for in the solution.

    #     .. warning:: 
        
    #         - The objective function is rather uncommon.
    #         - Order of magnitude slower than ``ikine_LM`` or ``ikine_LMS``, it
    #           uses a scalar cost-function and does not provide a Jacobian.

    #     :seealso: :func:`ikine_LM`, :func:`ikine_LMS`, :func:`ikine_con`, :func:`ikine_min`
    #     """

    #     if not isinstance(T, SE3):
    #         T = SE3(T)

    #     if q0 is None:
    #         q0 = np.zeros((self.n))
    #     else:
    #         q0 = base.getvector(q0, self.n)

    #     solutions = []

    #     wr = 1 / self.reach
    #     weight = np.r_[wr, wr, wr, 1, 1, 1]

    #     def cost(q, T, weight, costfun, stiffness):
    #         # T, weight, costfun, stiffness = args
    #         e = _angle_axis(self.fkine(q).A, T) * weight
    #         E = (e**2).sum()

    #         if stiffness > 0:
    #             # Enforce a continuity constraint on joints, minimum bend
    #             E += np.sum(np.diff(q)**2) * stiffness

    #         if costfun is not None:
    #             E += (e**2).sum() + costfun(q)

    #         return E

    #     for Tk in T:

    #         res = minimize(
    #             cost,
    #             q0,
    #             args=(Tk.A, weight, costfun, stiffness),
    #             tol=tol,
    #             method='SLSQP',
    #             options={'maxiter': ilimit}
    #             )

    #             # final gradient tolerance must be < gtol for success, bump
    #             # this number up a bit
    #             # SLSQP seems to work better than BFGS, L-BFGS-B

    #         solution = iksol(res.x, res.success, res.message, res.nit, res.fun)
    #         solutions.append(solution)
    #         q0 = res.x  # use this solution as initial estimate for next time

    #     if len(T) == 1:
    #         return solutions[0]
    #     else:
    #         return solutions

# --------------------------------------------------------------------- #

    def ikine_min(self, T, q0=None, qlim=False, ilimit=1000, tol=1e-16, method=None, stiffness=0, costfun=None, options={}):
        r"""
        Inverse kinematics by optimization with joint limits (Robot superclass)

        :param T: The desired end-effector pose or pose trajectory
        :type T: SE3
        :param q0: initial joint configuration (default all zeros)
        :type q0: ndarray(n)
        :param qlim: enforce joint limits
        :type qlim: bool
        :param ilimit: Iteration limit (default 1000)
        :type ilimit: int
        :param tol: Tolerance (default 1e-16)
        :type tol: tol
        :param method: minimization method to use
        :type method: str
        :param stiffness: Stiffness used to impose a smoothness contraint on
            joint angles, useful when n is large (default 0)
        :type stiffness: float
        :param costfun: User supplied cost term, optional
        :type costfun: callable
        :return: inverse kinematic solution
        :rtype: named tuple

        ``sol = robot.ikine_min(T)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose T which is an SE3 object.  The
        return value ``sol`` is a named tuple with elements:

        ============    ==========  ============================================================
        Element         Type        Description
        ============    ==========  ============================================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres, or ``None``
        ``success``     bool        whether a solution was found
        ``reason``      str         reason for the failure
        ``iterations``  int         number of iterations
        ``residual``    float       final value of cost function
        ============    ==========  ============================================================

        **Minimization method**:

        By default this method uses:

        - the Scipy ``SLSQP`` (Sequential Least Squares Programming) minimizer
          for the case of no joint limits
        - the Scipy ``trust-constr`` minimizer for the case with joint limits.
          This gives good results but is very slow.  An alternative is
          ``L-BFGS-B`` (Broyden–Fletcher–Goldfarb–Shanno) but for redundant
          robots can sometimes give poor results, pushing against the joint
          limits when there is no need to.
          
        In both case the function to be minimized is the squared norm of a 
        vector :math:`[d,a]` with components respectively the
        translation error and rotation error in Euler vector form, between the
        desired pose and the current estimate obtained by inverse kinematics.

        **Additional cost terms**:

        This method supports two additional costs:

        - ``stiffness`` imposes a penalty on joint variation 
          :math:`\sum_{j=1}^N (q_j - q_{j-1})^2` which tends to keep the 
          arm straight
        - ``costfun`` add a cost given by a user-specified function ``costfun(q)``

        **Trajectory operation**:

        If ``len(T) > 1`` it is considered to be a trajectory, and the result is
        a list of named tuples such that ``sol[k]`` corresponds to ``T[k]``. The
        initial estimate of q for each time step is taken as the solution from
        the previous time step.

        .. note::

            - See `Toolbox kinematics wiki page <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Uses ``SciPy.minimize`` with bounds.
            - Joint limits are considered in this solution.
            - Can be used for robots with arbitrary degrees of freedom.
            - The inverse kinematic solution is generally not unique, and
              depends on the initial guess ``q0``.
            - The default value of ``q0`` is zero which is a poor choice for 
              most manipulators since it often corresponds to a
              kinematic singularity.
            - Such a solution is completely general, though much less
              efficient than analytic inverse kinematic solutions derived
              symbolically.
            - The objective function (error) is 
              :math:`\sum \left( (\mat{T}^{-1} \cal{K}(\vec{q}) - \mat{1} ) \mat{\Omega} \right)^2`
              where :math:`\mat{\Omega}` is a diagonal matrix.
            - Joint offsets, if defined, are accounted for in the solution.

        .. warning:: 
        
            - The objective function is rather uncommon.
            - Order of magnitude slower than ``ikine_LM`` or ``ikine_LMS``, it
              uses a scalar cost-function and does not provide a Jacobian.

        :author: Bryan Moutrie, for RTB-MATLAB

        :seealso: :func:`ikine_LM`, :func:`ikine_LMS`, :func:`ikine_unc`, :func:`ikine_min`

        """

        if not isinstance(T, SE3):
            T = SE3(T)

        if q0 is None:
            q0 = np.zeros((self.n))
        else:
            q0 = base.getvector(q0, self.n)

        solutions = []

        wr = 1 / self.reach
        weight = np.r_[wr, wr, wr, 1, 1, 1]

        optdict = {'maxiter': ilimit}
        if options is not None and isinstance(options, dict):
            optdict.update(options)
        else:
            raise ValueError('options must be a dict')

        if qlim:
            # dealing with joint limits
            bounds = opt.Bounds(self.qlim[0, :], self.qlim[1, :])

            if method is None:
                method='trust-constr'
        else:
            # no joint limits
            if method is None:
                method = 'SLSQP'
            bounds = None

        def cost(q, T, weight, costfun, stiffness):
            # T, weight, costfun, stiffness = args
            e = _angle_axis(self.fkine(q).A, T) * weight
            E = (e**2).sum()

            if stiffness > 0:
                # Enforce a continuity constraint on joints, minimum bend
                E += np.sum(np.diff(q)**2) * stiffness

            if costfun is not None:
                E += (e**2).sum() + costfun(q)

            return E

        for Tk in T:
            res = opt.minimize(
                cost,
                q0, 
                args=(Tk.A, weight, costfun, stiffness),
                bounds=bounds,
                method=method,
                tol=tol,
                options=options
            )

            # trust-constr seems to work better than L-BFGS-B which often
            # runs a joint up against its limit and terminates with position 
            # error.
            # but 'truts-constr' is 5x slower

            solution = iksol(res.x, res.success, res.message, res.nit, res.fun)
            solutions.append(solution)
            q0 = res.x  # use this solution as initial estimate for next time

        if len(T) == 1:
            return solutions[0]
        else:
            return solutions

# --------------------------------------------------------------------- #

    def ikine_global(self, T, q0=None, qlim=False, ilimit=1000, tol=1e-16, method=None, options={}):
        r"""
        .. warning:: Experimental code for using SciPy global optimizers.

        Each global optimizer has quite a different call signature, so final
        design will need a bit of thought.

        """

        # basinhopping:
        # brute: ranges, finish=None
        # differential_evolution:  bounds, tol
        # shgo: bounds, options:f_tol
        # dual_annealing: bounds

        if not isinstance(T, SE3):
            T = SE3(T)

        if q0 is None:
            q0 = np.zeros((self.n))
        else:
            q0 = base.getvector(q0, self.n)

        solutions = []

        wr = 1 / self.reach
        weight = np.r_[wr, wr, wr, 1, 1, 1]

        optdict = {}

        if method is None:
            method='differential-evolution'

        if method == 'brute':
            # requires a tuple of tuples
            optdict['ranges'] = tuple([tuple(l.qlim) for l in self])
        else:
            optdict['bounds'] = tuple([tuple(l.qlim) for l in self])


        if method not in ['basinhopping', 'brute', 'differential_evolution',
                          'shgo', 'dual_annealing']:
            raise ValueError('unknown global optimizer requested')

        global_minimizer = opt.__dict__[method]

        def cost(q, T, weight):
            # T, weight, costfun, stiffness = args
            e = _angle_axis(self.fkine(q).A, T) * weight
            return (e**2).sum()

        for Tk in T:
            res = global_minimizer(
                cost,
                **optdict)
            

            solution = iksol(res.x, res.success, res.message, res.nit, res.fun)
            solutions.append(solution)
            q0 = res.x  # use this solution as initial estimate for next time

        if len(T) == 1:
            return solutions[0]
        else:
            return solutions
            
# --------------------------------------------------------------------- #

    # def ikine_min(self, T, q0=None, pweight=1.0, stiffness=0.0,
    #            qlimits=True, ilimit=1000):
    #     """
    #     Inverse kinematics by optimization with joint limits (Robot superclass)

    #     :param T: The desired end-effector pose or pose trajectory
    #     :type T: SE3
    #     :param q0: initial joint configuration (default all zeros)
    #     :type q0: ndarray(n)
    #     :param pweight: weighting on position error norm compared to rotation
    #         error (default 1)
    #     :type pweight: float
    #     :param stiffness: Stiffness used to impose a smoothness contraint on
    #         joint angles, useful when n is large (default 0)
    #     :type stiffness: float
    #     :param qlimits: Enforce joint limits (default True)
    #     :type qlimits: bool
    #     :param ilimit: Iteration limit (default 1000)
    #     :type ilimit: bool
    #     :return: inverse kinematic solution
    #     :rtype: named tuple

    #     ``sol = robot.ikine_unc(T)`` are the joint coordinates (n) corresponding
    #     to the robot end-effector pose T which is an SE3 object.  The
    #     return value ``sol`` is a named tuple with elements:

    #     ============    ==========  ============================================================
    #     Element         Type        Description
    #     ============    ==========  ============================================================
    #     ``q``           ndarray(n)  joint coordinates in units of radians or metres, or ``None``
    #     ``success``     bool        whether a solution was found
    #     ``reason``      str         reason for the failure
    #     ``iterations``  int         number of iterations
    #     ``residual``    float       final value of cost function
    #     ============    ==========  ============================================================

    #     **Trajectory operation**:

    #     If ``len(T) > 1`` it is considered to be a trajectory, and the result is
    #     a list of named tuples such that ``sol[k]`` corresponds to ``T[k]``. The
    #     initial estimate of q for each time step is taken as the solution from
    #     the previous time step.

    #     .. note::

    #         - PROTOTYPE CODE UNDER DEVELOPMENT, intended to do numerical
    #           inverse kinematics with joint limits
    #         - Uses ``SciPy.minimize`` with/without constraints.
    #         - The inverse kinematic solution is generally not unique, and
    #           depends on the initial guess ``q0``.
    #         - This norm is computed from distances and angles and ``pweight``
    #           can be used to scale the position error norm to be congruent 
    #           with rotation error norm.
    #         - For a highly redundant robot ``stiffness`` can be used to impose 
    #           a smoothness contraint on joint angles, tending toward solutions
    #           with are smooth curves.
    #         - The default value of ``q0`` is zero which is a poor choice for 
    #           most manipulators since it often corresponds to a
    #           kinematic singularity.
    #         - Such a solution is completely general, though much less
    #           efficient than analytic inverse kinematic solutions derived
    #           symbolically.
    #         - This approach allows a solution to obtained at a singularity,
    #           but the joint angles within the null space are arbitrarily
    #           assigned.
    #         - Joint offsets, if defined, are accounted for in the solution.
    #         - Joint limits become explicit bounds if 'qlimits' is set.

    #     .. warning:: 
        
    #         - The objective function is rather uncommon.
    #         - Order of magnitude slower than ``ikine_LM`` or ``ikine_LMS``, it
    #           uses a scalar cost-function and does not provide a Jacobian.

    #     :seealso: :func:`ikine_LM`, :func:`ikine_LMS`, :func:`ikine_unc`, :func:`ikine_con`
    #     """

    #     if not isinstance(T, SE3):
    #         T = SE3(T)

    #     if q0 is None:
    #         q0 = np.zeros((self.n,))
    #     else:
    #         q0 = base.getvector(q0, self.n)

    #     col = 2
    #     solutions = []

    #     # Define the cost function to minimise
    #     def cost(q, *args):
    #         T, pweight, col, stiffness = args
    #         Tq = self.fkine(q).A

    #         # translation error
    #         dT = base.transl(T) - base.transl(Tq)
    #         E = np.linalg.norm(dT) * pweight

    #         # Rotation error
    #         # Find dot product of two columns
    #         dd = np.dot(T[0:3, col], Tq[0:3, col])
    #         E += np.arccos(dd)**2 * 1000

    #         if stiffness > 0:
    #             # Enforce a continuity constraint on joints, minimum bend
    #             E += np.sum(np.diff(q)**2) * stiffness

    #         return E

    #     for Tk in T:

    #         if qlimits:
    #             bounds = Bounds(self.qlim[0, :], self.qlim[1, :])

    #             res = minimize(
    #                 cost,
    #                 q0, 
    #                 args=(Tk.A, pweight, col, stiffness),
    #                 bounds=bounds,
    #                 options={'gtol': 1e-6, 'maxiter': ilimit})
    #         else:
    #             # No joint limits, unconstrained optimization
    #             # final gradient tolerance must be < gtol for success, bump
    #             # this number up a bit
    #             res = minimize(
    #                 cost,
    #                 q0, 
    #                 args=(Tk.A, pweight, col, stiffness),
    #                 options={'gtol': 1e-6, 'maxiter': ilimit})

    #         solution = iksol(res.x, res.success, res.message, res.nit, res.fun)
    #         solutions.append(solution)
    #         q0 = res.x  # use this solution as initial estimate for next time

    #     if len(T) == 1:
    #         return solutions[0]
    #     else:
    #         return solutions


    # def qmincon(self, q=None):
    #     """
    #     Move away from joint limits

    #     :param q: Joint coordinates
    #     :type q: ndarray(n)
    #     :retrun qs: The calculated joint values
    #     :rtype qs: ndarray(n)
    #     :return: Optimisation solved (True) or failed (False)
    #     :rtype: bool
    #     :return: Final value of the objective function
    #     :rtype: float

    #     ``qs, success, err = qmincon(q)`` exploits null space motion and
    #     returns a set of joint angles ``qs`` (n) that result in the same
    #     end-effector pose but are away from the joint coordinate limits.
    #     ``n`` is the number of robot joints. ``success`` is True for
    #     successful optimisation. ``err`` is the scalar final value of
    #     the objective function.

    #     **Trajectory operation**

    #     In all cases if ``q`` is (m,n) it is taken as a pose sequence and
    #     ``qmincon()`` returns the adjusted joint coordinates (m,n)
    #     corresponding to each of the configurations in the sequence.

    #     ``err`` and ``success`` are also (m) and indicate the results of
    #     optimisation for the corresponding trajectory step.

    #     .. note:: Robot must be redundant.

    #     """

    #     def sumsqr(A):
    #         return np.sum(A**2)

    #     def cost(x, ub, lb, qm, N):
    #         return sumsqr(
    #             (2 * (N @ x + qm) - ub - lb) / (ub - lb))

    #     q = getmatrix(q, (None, self.n))

    #     qstar = np.zeros((q.shape[0], self.n))
    #     error = np.zeros(q.shape[0])
    #     success = np.zeros(q.shape[0])

    #     lb = self.qlim[0, :]
    #     ub = self.qlim[1, :]

    #     for k, qk in enumerate(q):

    #         J = self.jacobe(qk)

    #         N = null(J)

    #         x0 = np.zeros(N.shape[1])
    #         A = np.r_[N, -N]
    #         b = np.r_[ub - qk, qk - lb].reshape(A.shape[0],)

    #         con = LinearConstraint(A, -np.inf, b)

    #         res = minimize(
    #             lambda x: cost(x, ub, lb, qk, N),
    #             x0, constraints=con)

    #         qstar[k, :] = qk + N @ res.x
    #         error[k] = res.fun
    #         success[k] = res.success

    #     if q.shape[0] == 1:
    #         return qstar[0, :], success[0], error[0]
    #     else:
    #         return qstar, success, error

def _angle_axis(T, Td):
    d = base.transl(Td) - base.transl(T)
    R = base.t2r(Td) @ base.t2r(T).T
    l = np.r_[R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]
    if base.iszerovec(l):
        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        # non-diagonal matrix case
        ln = base.norm(l)
        a = math.atan2(ln, np.trace(R) - 1) * l / ln
        
    return np.r_[d, a]

def _angle_axis_sekiguchi(T, Td):
    d = base.transl(Td) - base.transl(T)
    R = base.t2r(Td) @ base.t2r(T).T
    l = np.r_[R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]
    if base.iszerovec(l):
        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            # (1, -1, -1), (-1, 1, -1) or (-1, -1, 1) case
            a = np.pi / 2 * (np.diag(R) + 1)
            # as per Sekiguchi paper
            if R[1,0] > 0 and R[2,1] > 0 and R[0,2] > 0:
                a = np.pi / np.sqrt(2) * np.sqrt(n.diag(R) + 1)
            elif R[1,0] > 0: # (2)
                a = np.pi / np.sqrt(2) * np.sqrt(n.diag(R) @ np.r_[1,1,-1] + 1)
            elif R[0,2] > 0: # (3)
                a = np.pi / np.sqrt(2) * np.sqrt(n.diag(R) @ np.r_[1,-1,1] + 1)
            elif R[2,1] > 0: # (4)
                a = np.pi / np.sqrt(2) * np.sqrt(n.diag(R) @ np.r_[-1,1,1] + 1)
    else:
        # non-diagonal matrix case
        ln = base.norm(l)
        a = math.atan2(ln, np.trace(R) - 1) * l / ln

    return np.r_[d, a]


if __name__ == "__main__":  # pragma nocover

    import roboticstoolbox as rtb
    from spatialmath import SE3

    # np.set_printoptions(linewidth=120, formatter={'float': lambda x: f"{x:9.5g}" if abs(x) > 1e-10 else f"{0:9.5g}"})


    robot = rtb.models.DH.Panda()

    T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    # sol = robot.ikine_LMS(T)         # solve IK
    # print(sol)                    # display joint angles

    # print(T)
    # print(robot.fkine(sol.q))
    # robot.plot(sol.q)

    # sol = robot.ikine_unc(T, costfun=lambda q: q[1] * 1e-6 if q[1] > 0 else -q[1])
    # print(sol)
    # print(robot.fkine(sol.q))
    # robot.plot(sol.q)

    sol = robot.ikine_global(T, method='brute')