#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
from collections import namedtuple
from roboticstoolbox.tools.null import null
from spatialmath import base
from spatialmath import SE3, Twist3
from scipy.optimize import minimize, Bounds, LinearConstraint
import math
iksol = namedtuple("IKsolution", "q, success, reason, iterations, residual",
    defaults=(None, False, None, None, None))

# ===================================================================== #
class IKMixin:

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
                return iksol()

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
        rejcount = 0  # Rejected step count
        nm = 0

        # bool vector indicating revolute joints
        revolutes = np.array([link.isrevolute() for link in self])

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
                if np.linalg.norm(W @ e) < tol:
                    # print(iterations)
                    break

                # Compute the Jacobian
                J = self.jacobe(q)

                JtJ = J.T @ W @ J

                if transpose is not None:
                    # Do the simple Jacobian transpose with constant gain
                    dq = transpose * J.T @ e    # lgtm [py/multiple-definition]
                else:
                    # Do the damped inverse Gauss-Newton with
                    # Levenberg-Marquadt
                    dq = np.linalg.inv(
                        JtJ + ((Li + Lmin) * np.eye(self.n))
                    ) @ J.T @ W @ e

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

        def angle_axis(T, Td):
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
            else:
                # non-diagonal matrix case
                ln = base.norm(l)
                a = math.atan2(ln, np.trace(R) - 1) * l / ln

            return np.r_[d, a]

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
        revolutes = np.array([link.isrevolute() for link in self])

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

                e = angle_axis(self.fkine(q).A, Tk.A)

                # Are we there yet?
                E = 0.5 * e.T @ W @ e
                if E < tol:
                    break

                # Compute the Jacobian
                J = self.jacob0(q)

                JtJ = J.T @ W @ J

                # Do the damped inverse Gauss-Newton with
                # Levenberg-Marquadt
                dq = np.linalg.inv(
                    JtJ + (E + wN) * np.eye(self.n)
                ) @ J.T @ W @ e

                # Compute possible new value of
                q += dq

                # Wrap angles for revolute joints
                k = np.logical_and(q > np.pi, revolutes)
                q[k] -= 2 * np.pi

                k = np.logical_and(q < -np.pi, revolutes)
                q[k] += + 2 * np.pi

                nm = np.linalg.norm(W @ e)
                qs = ", ".join(["{:8.3f}".format(qi) for qi in q])
                # print(f"|e|={E:8.2g}: q={qs}")

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

    def ikine_unc(self, T, q0=None, ilimit=1000):
        r"""
        Inverse manipulator by optimization without joint limits (Robot
        superclass)

        :param T: The desired end-effector pose or pose trajectory
        :type T: SE3
        :param q0: initial joint configuration (default all zeros)
        :type q0: ndarray(n)
        :return: inverse kinematic solution
        :rtype: named tuple

        :param T: The desired end-effector pose
        :type T: SE3
        :param ilimit: Iteration limit (default 1000)
        :type ilimit: bool
        :return: inverse kinematic solution
        :rtype: named tuple


        ``sol = robot.ikine_unc(T)`` are the joint coordinates (n) corresponding
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

        **Trajectory operation**:

        If ``len(T) > 1`` it is considered to be a trajectory, and the result is
        a list of named tuples such that ``sol[k]`` corresponds to ``T[k]``. The
        initial estimate of q for each time step is taken as the solution from
        the previous time step.

        .. note::

            - Uses ``SciPy.minimize`` without bounds.
            - Joint limits are not considered in this solution.
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

        :seealso: :func:`ikine_LM`, :func:`ikine_LMS`, :func:`ikine_con`, :func:`ikine_min`
        """

        if not isinstance(T, SE3):
            T = SE3(T)

        if q0 is None:
            q0 = np.zeros((self.n))
        else:
            q0 = base.getvector(q0, self.n)

        solutions = []

        omega = np.diag([1, 1, 1, 3 / self.reach])

        def cost(q, *args):
            T, omega = args
            E = (base.trinv(T) @ self.fkine(q).A - np.eye(4)) @ omega
            return (E**2).sum()  # quicker than np.sum(E**2)

        for Tk in T:

            res = minimize(
                cost,
                q0,
                args=(Tk.A, omega),
                options={'gtol': 1e-6, 'maxiter': ilimit})

            solution = iksol(res.x, res.success, res.message, res.nit, res.fun)
            solutions.append(solution)
            q0 = res.x  # use this solution as initial estimate for next time

        if len(T) == 1:
            return solutions[0]
        else:
            return solutions

# --------------------------------------------------------------------- #

    def ikine_con(self, T, q0=None, **kwargs):
        r"""
        Inverse kinematics by optimization with joint limits (Robot superclass)

        :param T: The desired end-effector pose or pose trajectory
        :type T: SE3
        :param q0: initial joint configuration (default all zeros)
        :type q0: ndarray(n)
        :return: inverse kinematic solution
        :rtype: named tuple


        ``sol = robot.ikine_unc(T)`` are the joint coordinates (n) corresponding
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

        **Trajectory operation**:

        If ``len(T) > 1`` it is considered to be a trajectory, and the result is
        a list of named tuples such that ``sol[k]`` corresponds to ``T[k]``. The
        initial estimate of q for each time step is taken as the solution from
        the previous time step.

        .. note::

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

        omega = np.diag([1, 1, 1, 3 / self.reach])

        def cost(q, *args):
            T, omega = args
            E = (base.trinv(T) @ self.fkine(q).A - np.eye(4)) @ omega
            return (E**2).sum()  # quicker than np.sum(E**2)

        bnds = Bounds(self.qlim[0, :], self.qlim[1, :])

        for Tk in T:
            res = minimize(
                cost,
                q0, 
                args=(Tk.A, omega),
                bounds=bnds, 
                tol=1e-8,
                options={'ftol': 1e-10})

            solution = iksol(res.x, res.success, res.message, res.nit, res.fun)
            solutions.append(solution)
            q0 = res.x  # use this solution as initial estimate for next time

        if len(T) == 1:
            return solutions[0]
        else:
            return solutions

# --------------------------------------------------------------------- #

    def ikine_min(self, T, q0=None, pweight=1.0, stiffness=0.0,
               qlimits=True, ilimit=1000):
        """
        Inverse kinematics by optimization with joint limits (Robot superclass)

        :param T: The desired end-effector pose or pose trajectory
        :type T: SE3
        :param q0: initial joint configuration (default all zeros)
        :type q0: ndarray(n)
        :param pweight: weighting on position error norm compared to rotation
            error (default 1)
        :type pweight: float
        :param stiffness: Stiffness used to impose a smoothness contraint on
            joint angles, useful when n is large (default 0)
        :type stiffness: float
        :param qlimits: Enforce joint limits (default True)
        :type qlimits: bool
        :param ilimit: Iteration limit (default 1000)
        :type ilimit: bool
        :return: inverse kinematic solution
        :rtype: named tuple

        ``sol = robot.ikine_unc(T)`` are the joint coordinates (n) corresponding
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

        **Trajectory operation**:

        If ``len(T) > 1`` it is considered to be a trajectory, and the result is
        a list of named tuples such that ``sol[k]`` corresponds to ``T[k]``. The
        initial estimate of q for each time step is taken as the solution from
        the previous time step.

        .. note::

            - PROTOTYPE CODE UNDER DEVELOPMENT, intended to do numerical
              inverse kinematics with joint limits
            - Uses ``SciPy.minimize`` with/without constraints.
            - The inverse kinematic solution is generally not unique, and
              depends on the initial guess ``q0``.
            - This norm is computed from distances and angles and ``pweight``
              can be used to scale the position error norm to be congruent 
              with rotation error norm.
            - For a highly redundant robot ``stiffness`` can be used to impose 
              a smoothness contraint on joint angles, tending toward solutions
              with are smooth curves.
            - The default value of ``q0`` is zero which is a poor choice for 
              most manipulators since it often corresponds to a
              kinematic singularity.
            - Such a solution is completely general, though much less
              efficient than analytic inverse kinematic solutions derived
              symbolically.
            - This approach allows a solution to obtained at a singularity,
              but the joint angles within the null space are arbitrarily
              assigned.
            - Joint offsets, if defined, are accounted for in the solution.
            - Joint limits become explicit bounds if 'qlimits' is set.

        .. warning:: 
        
            - The objective function is rather uncommon.
            - Order of magnitude slower than ``ikine_LM`` or ``ikine_LMS``, it
              uses a scalar cost-function and does not provide a Jacobian.

        :seealso: :func:`ikine_LM`, :func:`ikine_LMS`, :func:`ikine_unc`, :func:`ikine_con`
        """

        if not isinstance(T, SE3):
            T = SE3(T)

        if q0 is None:
            q0 = np.zeros((self.n,))
        else:
            q0 = base.getvector(q0, self.n)

        col = 2
        solutions = []

        # Define the cost function to minimise
        def cost(q, *args):
            T, pweight, col, stiffness = args
            Tq = self.fkine(q).A

            # translation error
            dT = base.transl(T) - base.transl(Tq)
            E = np.linalg.norm(dT) * pweight

            # Rotation error
            # Find dot product of two columns
            dd = np.dot(T[0:3, col], Tq[0:3, col])
            E += np.arccos(dd)**2 * 1000

            if stiffness > 0:
                # Enforce a continuity constraint on joints, minimum bend
                E += np.sum(np.diff(q)**2) * stiffness

            return E

        for Tk in T:

            if qlimits:
                bounds = Bounds(self.qlim[0, :], self.qlim[1, :])

                res = minimize(
                    cost,
                    q0, 
                    args=(Tk.A, pweight, col, stiffness),
                    bounds=bounds,
                    options={'gtol': 1e-6, 'maxiter': ilimit})
            else:
                # No joint limits, unconstrained optimization
                res = minimize(
                    cost,
                    q0, 
                    args=(Tk.A, pweight, col, stiffness),
                    options={'gtol': 1e-6, 'maxiter': ilimit})

            solution = iksol(res.x, res.success, res.message, res.nit, res.fun)
            solutions.append(solution)
            q0 = res.x  # use this solution as initial estimate for next time

        if len(T) == 1:
            return solutions[0]
        else:
            return solutions


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
