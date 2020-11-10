"""
Rigid-body dynamics functionality of the Toolbox.

Requires access to:

    * ``links`` list of ``Link`` objects, atttribute
    * ``rne()`` the inverse dynamics method

so must be subclassed by ``DHRobot`` class.

:todo: perhaps these should be abstract properties, methods of this calss
"""
from collections import namedtuple
from functools import wraps
import numpy as np
from spatialmath.base import \
    getvector, verifymatrix, isscalar, getmatrix, t2r
from scipy import integrate, interpolate
from spatialmath.base import symbolic as sym
from frne import init, frne, delete


def _check_rne(func):
    @wraps(func)
    def wrapper_check_rne(*args, **kwargs):
        if args[0]._rne_ob is None or args[0]._dynchanged:
            args[0].delete_rne()
            args[0]._init_rne()
        args[0]._rne_changed = False
        return func(*args, **kwargs)
    return wrapper_check_rne


class DHDynamicsMixin:

    def printdyn(self):
        """
        Print dynamic parameters

        Display the kinematic and dynamic parameters to the console in
        reable format
        """
        for j, link in enumerate(self.links):
            print("\nLink {:d}::".format(j), link)
            print(link.dyn(indent=2))

    def delete_rne(self):
        """
        Frees the memory holding the robot object in c if the robot object
        has been initialised in c.
        """
        if self._rne_ob is not None:
            delete(self._rne_ob)
            self._dynchanged = False
            self._rne_ob = None

    def _init_rne(self):
        # Compress link data into a 1D array
        L = np.zeros(24 * self.n)

        for i in range(self.n):
            j = i * 24
            L[j] = self.links[i].alpha
            L[j + 1] = self.links[i].a
            L[j + 2] = self.links[i].theta
            L[j + 3] = self.links[i].d
            L[j + 4] = self.links[i].sigma
            L[j + 5] = self.links[i].offset
            L[j + 6] = self.links[i].m
            L[j + 7:j + 10] = self.links[i].r.flatten()
            L[j + 10:j + 19] = self.links[i].I.flatten()
            L[j + 19] = self.links[i].Jm
            L[j + 20] = self.links[i].G
            L[j + 21] = self.links[i].B
            L[j + 22:j + 24] = self.links[i].Tc.flatten()

        self._rne_ob = init(self.n, self.mdh, L, self.gravity)

    @_check_rne
    def rne(self, q, qd=None, qdd=None, grav=None, fext=None):
        r"""
        Inverse dynamics

        :param q: Joint coordinates
        :type q: ndarray(n)
        :param qd: Joint velocity
        :type qd: ndarray(n)
        :param qdd: The joint accelerations of the robot
        :type qdd: ndarray(n)
        :param grav: Gravity vector to overwrite robots gravity value
        :type grav: ndarray(6)
        :param fext: Specify wrench acting on the end-effector
                     :math:`W=[F_x F_y F_z M_x M_y M_z]`
        :type fext: ndarray(6)

        ``tau = rne(q, qd, qdd, grav, fext)`` is the joint torque required for
        the robot to achieve the specified joint position ``q`` (1xn), velocity
        ``qd`` (1xn) and acceleration ``qdd`` (1xn), where n is the number of
        robot joints. ``fext`` describes the wrench acting on the end-effector

        Trajectory operation:
        If q, qd and qdd (mxn) are matrices with m cols representing a
        trajectory then tau (mxn) is a matrix with cols corresponding to each
        trajectory step.

        .. note::
            - The torque computed contains a contribution due to armature
              inertia and joint friction.
            - If a model has no dynamic parameters set the result is zero.

        :seealso: :func:`rne_python`
        """
        trajn = 1

        try:
            q = getvector(q, self.n, 'row')
            qd = getvector(qd, self.n, 'row')
            qdd = getvector(qdd, self.n, 'row')
        except ValueError:
            trajn = q.shape[0]
            verifymatrix(q, (trajn, self.n))
            verifymatrix(qd, (trajn, self.n))
            verifymatrix(qdd, (trajn, self.n))

        if grav is None:
            grav = self.gravity
        else:
            grav = getvector(grav, 3)

        # The c function doesn't handle base rotation, so we need to hack the
        # gravity vector instead
        grav = self.base.R.T @ grav

        if fext is None:
            fext = np.zeros(6)
        else:
            fext = getvector(fext, 6)

        tau = np.zeros((trajn, self.n))

        for i in range(trajn):
            tau[i, :] = frne(
                self._rne_ob, q[i, :], qd[i, :], qdd[i, :], grav, fext)

        if trajn == 1:
            return tau[0, :]
        else:
            return tau

    def fdyn(
            self, T, q0, torqfun=None, targs=None, qd0=None,
            solver='RK45', sargs=None, dt=None, progress=False):
        """
        Integrate forward dynamics

        :param T: integration time
        :type T: float
        :param q0: initial joint coordinates
        :type q0: array_like
        :param qd0: initial joint velocities, assumed zero if not given
        :type qd0: array_like
        :param torqfun: a function that computes torque as a function of time
        and/or state
        :type torqfun: callable
        :param targs: argumments passed to ``torqfun``
        :type targs: dict
        :type solver: name of scipy solver to use, RK45 is the default
        :param solver: str
        :type sargs: arguments passed to the solver
        :param sargs: dict
        :type dt: time step for results
        :param dt: float
        :param progress: show progress bar, default False
        :type progress: bool

        :return: robot trajectory
        :rtype: namedtuple

        - ``tg = R.fdyn(T, q)`` integrates the dynamics of the robot with zero
          input torques over the time  interval 0 to ``T`` and returns the
          trajectory as a namedtuple with elements:

            - ``t`` the time vector (M,)
            - ``q`` the joint coordinates (M,n)
            - ``qd`` the joint velocities (M,n)

        - ``tg = R.fdyn(T, q, torqfun)`` as above but the torque applied to the
          joints is given by the provided function::

                tau = function(robot, t, q, qd, **args)

        where the inputs are:

            - the robot object
            - current time
            - current joint coordinates (n,)
            - current joint velocity (n,)
            - args, optional keyword arguments can be specified, these are
              passed in from the ``targs`` kewyword argument.

        The function must return a Numpy array (n,) of joint forces/torques.

        Examples:

         #. to apply zero joint torque to the robot without Coulomb
            friction::

                def myfunc(robot, t, q, qd):
                    return np.zeros((robot.n,))

                tg = robot.nofriction().fdyn(5, q0, myfunc)

                plt.figure()
                plt.plot(tg.t, tg.q)
                plt.show()

            We could also use a lambda function::

                tg = robot.nofriction().fdyn(
                    5, q0, lambda r, t, q, qd: np.zeros((r.n,)))

         #. the robot is controlled by a PD controller. We first define a
            function to compute the control which has additional parameters for
            the setpoint and control gains (qstar, P, D)::

                def myfunc(robot, t, q, qd, qstar, P, D):
                    return (qstar - q) * P + qd * D  # P, D are (6,)

                targs = {'qstar': VALUE, 'P': VALUE, 'D': VALUE}
                tg = robot.fdyn(10, q0, myfunc, targs=targs) )

        Many integrators have variable step length which is problematic if we
        want to animate the result.  If ``dt`` is specified then the solver
        results are interpolated in time steps of ``dt``.

        .. note::

            - This function performs poorly with non-linear joint friction,
            such as Coulomb friction.  The R.nofriction() method can be used
            to set this friction to zero.
            - If the function is not specified then zero force/torque is
            applied to the manipulator joints.
            - Interpolation is performed using `ScipY integrate.ode
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`
            - The SciPy RK45 integrator is used by default
            - Interpolation is performed using `SciPy interp1
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`

        :seealso: :func:`DHRobot.accel`, :func:`DHRobot.nofriction`,
            :func:`DHRobot.rne`.
        """

        n = self.n

        if not isscalar(T):
            raise ValueError('T must be a scalar')
        q0 = getvector(q0, n)
        if qd0 is None:
            qd0 = np.zeros((n,))
        else:
            qd0 = getvector(qd0, n)
        if torqfun is not None:
            if not callable(torqfun):
                raise ValueError('torque function must be callable')
        if sargs is None:
            sargs = {}
        if targs is None:
            targs = {}

        # concatenate q and qd into the initial state vector
        x0 = np.r_[q0, qd0]

        # get user specified integrator
        scipy_integrator = integrate.__dict__[solver]

        integrator = scipy_integrator(
            lambda t, y: self._fdyn(t, y, torqfun, targs),
            t0=0.0, y0=x0, t_bound=T, **sargs
            )

        # initialize list of time and states
        tlist = [0]
        xlist = [np.r_[q0, qd0]]

        if progress:
            _printProgressBar(
                0, prefix='Progress:', suffix='complete', length=60)

        while integrator.status == 'running':

            # step the integrator, calls _fdyn multiple times
            integrator.step()

            if integrator.status == 'failed':
                raise RuntimeError('integration completed with failed status ')

            # stash the results
            tlist.append(integrator.t)
            xlist.append(integrator.y)

            # update the progress bar
            if progress:
                _printProgressBar(
                    integrator.t / T, prefix='Progress:', suffix='complete',
                    length=60)

        # cleanup the progress bar
        if progress:
            print('\r' + ' ' * 90 + '\r')

        tarray = np.array(tlist)
        xarray = np.array(xlist)

        if dt is not None:
            # interpolate data to equal time steps of dt
            interp = interpolate.interp1d(tarray, xarray, axis=0)

            tnew = np.arange(0, T, dt)
            xnew = interp(tnew)
            return namedtuple('fdyn', 't q qd')(tnew, xnew[:, :n], xnew[:, n:])
        else:
            return namedtuple('fdyn', 't q qd')(
                tarray, xarray[:, :n], xarray[:, n:])

    def _fdyn(self, t, x, torqfun, targs):
        """
        Private function called by fdyn

        :param t: current time
        :type t: float
        :param x: current state [q, qd]
        :type x: numpy array (2n,)
        :param torqfun: a function that computes torque as a function of time
        and/or state
        :type torqfun: callable
        :param targs: argumments passed to ``torqfun``
        :type targs: dict

        :return: derivative of current state [qd, qdd]
        :rtype: numpy array (2n,)

        Called by ``fdyn`` to evaluate the robot velocity and acceleration for
        forward dynamics.
        """
        n = self.n

        q = x[0:n]
        qd = x[n:]

        # evaluate the torque function if one is given
        if torqfun is None:
            tau = np.zeros((n,))
        else:
            tau = torqfun(self, t, q, qd, **targs)
            if len(tau) != n or not all(np.isreal(tau)):
                raise RuntimeError(
                    'torque function must return vector with N real elements')

        qdd = self.accel(q, qd, tau)

        return np.r_[qd, qdd]

    def accel(self, q, qd, torque):
        r"""
        Compute acceleration due to applied torque

        :param q: Joint coordinates
        :type q: ndarray(n)
        :param qd: Joint velocity
        :type qd: ndarray(n)
        :param torque: Joint torques of the robot
        :type torque:  ndarray(n)
        :return: Joint accelerations of the robot
        :rtype: ndarray(n)

        ``qdd = accel(q, qd, torque)`` calculates a vector (n) of joint
        accelerations that result from applying the actuator force/torque (n)
        to the manipulator in state `q` (n) and `qd` (n), and ``n`` is
        the number of robot joints.

        :math:`\ddot{q} = \mathbf{I}^{-1} \left(\tau - \mathbf{C}(q)\dot{q} - \mathbf{g}(q)\right)`

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.accel(puma.qz, 0.5 * np.ones(6), np.zeros(6))

        **Trajectory operation**

        If `q`, `qd`, torque are matrices (m,n) then ``qdd`` is a matrix (m,n)
        where each row is the acceleration corresponding to the equivalent cols
        of q, qd, torque.



        .. note::
            - Useful for simulation of manipulator dynamics, in
              conjunction with a numerical integration function.
            - Uses the method 1 of Walker and Orin to compute the forward
              dynamics.
            - Featherstone's method is more efficient for robots with large
              numbers of joints.
            - Joint friction is considered.

        :references:
            - Efficient dynamic computer simulation of robotic mechanisms,
              M. W. Walker and D. E. Orin,
              ASME Journa of Dynamic Systems, Measurement and Control, vol.
              104, no. 3, pp. 205-211, 1982.

        """

        trajn = 1

        try:
            q = getvector(q, self.n, 'row')
            qd = getvector(qd, self.n, 'row')
            torque = getvector(torque, self.n, 'row')
        except ValueError:
            trajn = q.shape[0]
            verifymatrix(q, (trajn, self.n))
            verifymatrix(qd, (trajn, self.n))
            verifymatrix(torque, (trajn, self.n))

        qdd = np.zeros((trajn, self.n))

        for i in range(trajn):
            # Compute current manipulator inertia torques resulting from unit
            # acceleration of each joint with no gravity.
            qI = (np.c_[q[i, :]] @ np.ones((1, self.n))).T
            qdI = np.zeros((self.n, self.n))
            qddI = np.eye(self.n)

            M = self.rne(qI, qdI, qddI, grav=[0, 0, 0])

            # Compute gravity and coriolis torque torques resulting from zero
            # acceleration at given velocity & with gravity acting.
            tau = self.rne(q[i, :], qd[i, :], np.zeros((1, self.n)))

            inter = np.expand_dims((torque[i, :] - tau), axis=1)

            qdd[i, :] = np.linalg.solve(M, inter).flatten()

        if trajn == 1:
            return qdd[0, :]
        else:
            return qdd

    def pay(self, W, q=None, J=None, frame=1):
        """
        tau = pay(W, J) Returns the generalised joint force/torques due to a
        payload wrench W applied to the end-effector. Where the manipulator
        Jacobian is J (6xn), and n is the number of robot joints.

        tau = pay(W, q, frame) as above but the Jacobian is calculated at pose
        q in the frame given by frame which is 0 for base frame, 1 for
        end-effector frame.

        Uses the formula tau = J'W, where W is a wrench vector applied at the
        end effector, W = [Fx Fy Fz Mx My Mz]'.

        Trajectory operation:
          In the case q is nxm or J is 6xnxm then tau is nxm where each row
          is the generalised force/torque at the pose given by corresponding
          row of q.

        :param W: A wrench vector applied at the end effector,
            W = [Fx Fy Fz Mx My Mz]
        :type W: ndarray(6)
        :param q: Joint coordinates
        :type q: ndarray(n)
        :param J: The manipulator Jacobian (Optional, if not supplied will
            use the q value).
        :type J: ndarray(6,n)
        :param frame: The frame in which to torques are expressed in when J
            is not supplied. 0 means base frame of the robot, 1 means end-
            effector frame
        :type frame: int

        :return: Joint forces/torques due to w
        :rtype: ndarray(n)

        .. note::
            - Wrench vector and Jacobian must be from the same reference
              frame.
            - Tool transforms are taken into consideration when frame=1.
            - Must have a constant wrench - no trajectory support for this
              yet.

        """

        try:
            W = getvector(W, 6)
            trajn = 0
        except ValueError:
            trajn = W.shape[0]
            verifymatrix(W, (trajn, 6))

        if trajn:
            # A trajectory
            if J is not None:
                # Jacobian supplied
                verifymatrix(J, (trajn, 6, self.n))
            else:
                # Use q instead
                verifymatrix(q, (trajn, self.n))
                J = np.zeros((trajn, 6, self.n))
                for i in range(trajn):
                    if frame:
                        J[i, :, :] = self.jacobe(q[i, :])
                    else:
                        J[i, :, :] = self.jacob0(q[i, :])
        else:
            # Single configuration
            if J is not None:
                # Jacobian supplied
                verifymatrix(J, (6, self.n))
            else:
                # Use q instead
                if q is None:
                    q = np.copy(self.q)
                else:
                    q = getvector(q, self.n)

                if frame:
                    J = self.jacobe(q)
                else:
                    J = self.jacob0(q)

        if trajn == 0:
            tau = -J.T @ W
        else:
            tau = np.zeros((trajn, self.n))

            for i in range(trajn):
                tau[i, :] = -J[i, :, :].T @ W[i, :]

        return tau

    def payload(self, m, p=np.zeros(3)):
        """
        payload(m, p) adds payload mass adds a payload with point mass m at
        position p in the end-effector coordinate frame.

        payload(m) adds payload mass adds a payload with point mass m at
        in the end-effector coordinate frame.

        payload(0) removes added payload.

        :param m: mass (kg)
        :type m: float
        :param p: position in end-effector frame
        :type p: ndarray(3,1)

        """

        p = getvector(p, 3, out='col')
        lastlink = self.links[self.n - 1]

        lastlink.m = m
        lastlink.r = p

    def jointdynamics(self, q, qd=None):
        """
        Transfer function of joint actuator

        :param q: Joint coordinates
        :type q: ndarray(n)
        :param qd: Joint velocity
        :type qd: ndarray(n)
        :return: transfer function denominators
        :rtype: list of 2-tuples

        - ``tf = jointdynamics(qd, q)`` calculates a vector of n
          continuous-time transfer functions that represent the transfer
          function 1/(Js+B) for each joint based on the dynamic parameters
          of the robot and the configuration q (n). n is the number of robot
          joints.

        - ``tf = jointdynamics(q, qd)`` as above but include the linearized
          effects of Coulomb friction when operating at joint velocity QD
          (1xN).
        """

        tf = []
        for j, link in enumerate(self.links):

            # compute inertia for this joint
            zero = np.zeros((self.n))
            qdd = np.zeros((self.n))
            qdd[j] = 1
            M = self.rne(q, zero, qdd, grav=[0, 0, 0])
            J = link.Jm + M[j] / abs(link.G) ** 2

            # compute friction
            B = link.B
            if qd is not None:
                # add linearized Coulomb friction at the operating point
                if qd > 0:
                    B += link.Tc[0] / qd[j]
                elif qd < 0:
                    B += link.Tc[1] / qd[j]
            tf.append((J, B))

        return tf

    def cinertia(self, q=None):
        r"""
        Cartesian manipulator inertia matrix

        :param q: Joint coordinates
        :type q: ndarray(n)

        :return: The inertia matrix
        :rtype: ndarray(6,6)

        ``robot.cinertia(q)`` is the Cartesian (operational space) inertia
        matrix which relates Cartesian force/torque to Cartesian
        acceleration at the joint configuration q.

        :math:`\mathbf{M} = {\mathbf{J}(q)^+}^T \mathbf{I}(q) \mathbf{J}(q)^+

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.cinertia(puma.qz)

        **Trajectory operation**

        If ``q`` is a matrix (m,n), each row is interpretted as a joint state
        vector, and the result is a 3d-matrix (n,n,m) where each plane
        corresponds to the Cartesian inertia for the corresponding
        row of ``q``.

        .. warning:: Assumes that the operational space has 6 DOF.

        :seealso: :func:`inertia`
        """
        q = getmatrix(q, (None, self.n))

        Mt = np.zeros((q.shape[0], 6, 6))

        for k, qk in enumerate(q):
            J = self.jacob0(qk)
            Ji = np.linalg.pinv(J)
            M = self.inertia(qk)
            Mt[k, :, :] = Ji.T @ M @ Ji

        if q.shape[0] == 1:
            return Mt[0, :, :]
        else:
            return Mt

    def inertia(self, q=None):
        """
        DHRobot.INERTIA Manipulator inertia matrix

        :param q: Joint coordinates
        :type q: ndarray(n)

        :return: The inertia matrix
        :rtype: ndarray(n,n)

        ``inertia(q)`` is the symmetric joint inertia matrix (n,n) which
        relates joint torque to joint acceleration for the robot at joint
        configuration q.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.inertia(puma.qz)

        **Trajectory operation**

        If ``q`` is a matrix (m,n), each row is interpretted as a joint state
        vector, and the result is a 3d-matrix (nxnxk) where each plane
        corresponds to the inertia for the corresponding row of q.

        .. note::
            - The diagonal elements ``I[j,j]`` are the inertia seen by joint
              actuator ``j``.
            - The off-diagonal elements ``I[j,k]`` are coupling inertias that
              relate acceleration on joint ``j`` to force/torque on
              joint ``k``.
            - The diagonal terms include the motor inertia reflected through
              the gear ratio.

        :seealso: :func:`cinertia`
        """
        q = getmatrix(q, (None, self.n))

        In = np.zeros((q.shape[0], self.n, self.n))

        for k, qk in enumerate(q):
            In[k, :, :] = self.rne(
                (np.c_[qk] @ np.ones((1, self.n))).T,
                np.zeros((self.n, self.n)),
                np.eye(self.n),
                grav=[0, 0, 0])

        if q.shape[0] == 1:
            return In[0, :, :]
        else:
            return In

    def coriolis(self, q, qd):
        r"""
        Coriolis and centripetal term

        :param q: Joint coordinates
        :type q: ndarray(n)
        :param qd: Joint velocity
        :type qd: ndarray(n)
        :return: Velocity matrix
        :rtype: ndarray(n,n)

        ``coriolis(q, qd)`` calculates the Coriolis/centripetal matrix (n,n)
        for the robot in configuration ``q`` and velocity ``qd``, where ``n``
        is the number of joints.

        The product :math:`\mathbf{C} \dot{q}` is the vector of joint
        force/torque due to velocity coupling. The diagonal elements are due to
        centripetal effects and the off-diagonal elements are due to Coriolis
        effects. This matrix is also known as the velocity coupling matrix,
        since it describes the disturbance forces on any joint due to
        velocity of all other joints.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.coriolis(puma.qz, 0.5 * np.ones((6,)))

        **Trajectory operation**

        If ``q`` and `qd` are matrices (m,n), each row is interpretted as a
        joint configuration, and the result (n,n,m) is a 3d-matrix where
        each plane corresponds to a row of ``q`` and ``qd``.

        .. note::
            - Joint viscous friction is also a joint force proportional to
              velocity but it is eliminated in the computation of this value.
            - Computationally slow, involves :math:`n^2/2` invocations of RNE.
        """

        q = getmatrix(q, (None, self.n))
        qd = getmatrix(qd, (None, self.n))
        if q.shape[0] != qd.shape[0]:
            raise ValueError('q and qd must have the same number of rows')

        # ensure that friction doesn't enter the mix, it's also a velocity
        # dependent force/torque
        r1 = self.nofriction(True, True)

        C = np.zeros((q.shape[0], self.n, self.n))
        Csq = np.zeros((q.shape[0], self.n, self.n))

        # Find the torques that depend on a single finite joint speed,
        # these are due to the squared (centripetal) terms
        # set QD = [1 0 0 ...] then resulting torque is due to qd_1^2
        for k, qk in enumerate(q):
            for i in range(self.n):
                QD = np.zeros(self.n)
                QD[i] = 1
                tau = r1.rne(
                    qk, QD, np.zeros(self.n), grav=[0, 0, 0])
                Csq[k, :, i] = Csq[k, :, i] + tau

        # Find the torques that depend on a pair of finite joint speeds,
        # these are due to the product (Coriolis) terms
        # set QD = [1 1 0 ...] then resulting torque is due to
        # qd_1 qd_2 + qd_1^2 + qd_2^2
        for k, (qk, qdk) in enumerate(zip(q, qd)):
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    # Find a product term  qd_i * qd_j
                    QD = np.zeros(self.n)
                    QD[i] = 1
                    QD[j] = 1
                    tau = r1.rne(qk, QD, np.zeros(self.n), grav=[0, 0, 0])

                    C[k, :, j] = C[k, :, j] + \
                        (tau - Csq[k, :, j] - Csq[k, :, i]) * qdk[i] / 2

                    C[k, :, i] = C[k, :, i] + \
                        (tau - Csq[k, :, j] - Csq[k, :, i]) * qdk[j] / 2

            C[k, :, :] = C[k, :, :] + Csq[k, :, :] @ np.diag(qdk)

        if q.shape[0] == 1:
            return C[0, :, :]
        else:
            return C

    def itorque(self, q, qdd):
        r"""
        Inertia torque

        :param q: Joint coordinates
        :type q: ndarray(n)
        :param qdd: Joint acceleration
        :type qdd: ndarray(n)

        :return: The inertia torque vector
        :rtype: ndarray(n)

        ``itorque(q, qdd)`` is the inertia force/torque vector (n) at
        the specified joint configuration q (n) and acceleration qdd (n), and
        ``n`` is the number of robot joints.
        It is :math:`\mathbf{I}(q) \ddot{q}`.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.itorque(puma.qz, 0.5 * np.ones((6,)))

        **Trajectory operation**

        If ``q`` and ``qdd`` are matrices (m,n), each row is interpretted as a
        joint configuration, and the result is a matrix (m,n) where each row is
        the corresponding joint torques.

        .. note:: If the robot model contains non-zero motor inertia then this
              will be included in the result.

        :seealso: :func:`inertia`
        """

        q = getmatrix(q, (None, self.n))
        qdd = getmatrix(qdd, (None, self.n))
        if q.shape[0] != qdd.shape[0]:
            raise ValueError('q and qdd must have the same number of rows')

        taui = np.zeros((q.shape[0], self.n))

        for k, (qk, qddk) in enumerate(zip(q, qdd)):
            taui[k, :] = self.rne(
                qk, np.zeros(self.n), qddk, grav=[0, 0, 0])

        if q.shape[0] == 1:
            return taui[0, :]
        else:
            return taui

    def gravload(self, q=None, grav=None):
        """
        Compute gravity load

        :param q: Joint coordinates
        :type q: ndarray(n)
        :param grav: The gravity vector (Optional, if not supplied will
            use the stored gravity values).
        :type grav: ndarray(3)

        :return: The generalised joint force/torques due to gravity
        :rtype: ndarray(n)

        ``taug = gravload(q)`` calculates the joint gravity loading (n) for
        the robot in the joint configuration ``q`` and using the default
        gravitational acceleration specified in the DHRobot object.

        ``taug = gravload(q, grav)`` as above except the gravitational
        acceleration is explicitly specified as `grav``.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.gravload(puma.qz)

        **Trajectory operation**

        If q is a matrix (nxm) each column is interpreted as a joint
        configuration vector, and the result is a matrix (nxm) each column
        being the corresponding joint torques.

        """

        trajn = 1

        if q is None:
            q = self.q

        if grav is None:
            grav = getvector(np.copy(self.gravity), 3, 'row')

        try:
            q = getvector(q, self.n, 'row')
            grav = getvector(grav, 3, 'row')
        except ValueError:
            trajn = q.shape[0]
            verifymatrix(q, (trajn, self.n))

        if grav.shape[0] < trajn:
            grav = (grav.T @ np.ones((1, trajn))).T
        verifymatrix(grav, (trajn, 3))

        taug = np.zeros((trajn, self.n))

        for i in range(trajn):
            taug[i, :] = self.rne(
                 q[i, :], np.zeros(self.n), np.zeros(self.n), grav[i, :])

        if trajn == 1:
            return taug[0, :]
        else:
            return taug

    def paycap(self, w, tauR, frame=1, q=None):
        """
        Static payload capacity of a robot

        :param w: The payload wrench
        :type w: ndarray(n)
        :param tauR: Joint torque matrix minimum and maximums
        :type tauR: ndarray(n,2)
        :param frame: The frame in which to torques are expressed in when J
            is not supplied. 'base' means base frame of the robot, 'ee' means
            end-effector frame
        :type frame: str
        :param q: Joint coordinates
        :type q: ndarray(n)

        :return: The maximum permissible payload wrench
        :rtype: ndarray(6)
        :return: Joint index (zero indexed) which hits its
            force/torque limit
        :rtype: int

        ``wmax, joint = paycap(q, w, f, tauR)`` returns the maximum permissible
        payload wrench ``wmax`` (6) applied at the end-effector, and the index
        of the joint (zero indexed) which hits its force/torque limit at that
        wrench. ``q`` (n) is the manipulator pose, ``w`` the payload wrench
        (6), ``f`` the wrench reference frame and tauR (nx2) is a matrix of
        joint forces/torques (first col is maximum, second col minimum).

        **Trajectory operation:**

        In the case q is nxm then wmax is Mx6 and J is Mx1 where the rows are
        the results at the pose given by corresponding row of q.

        .. note::
            - Wrench vector and Jacobian must be from the same reference frame
            - Tool transforms are taken into consideration for frame=1.
        """
        # TODO rewrite
        trajn = 1

        if q is None:
            q = self.q

        try:
            q = getvector(q, self.n, 'row')
            w = getvector(w, 6, 'row')
        except ValueError:
            trajn = q.shape[1]
            verifymatrix(q, (trajn, self.n))
            verifymatrix(w, (trajn, 6))

        verifymatrix(tauR, (self.n, 2))

        wmax = np.zeros((trajn, 6))
        joint = np.zeros(trajn, dtype=np.int)

        for i in range(trajn):
            tauB = self.gravload(q[i, :])

            # tauP = self.rne(
            #     np.zeros(self.n), np.zeros(self.n),
            #     q, grav=[0, 0, 0], fext=w/np.linalg.norm(w))

            tauP = self.pay(
                w[i, :]/np.linalg.norm(w[i, :]), q=q[i, :], frame=frame)

            M = tauP > 0
            m = tauP <= 0

            TAUm = np.ones(self.n)
            TAUM = np.ones(self.n)

            for c in range(self.n):
                TAUM[c] = tauR[c, 0]
                TAUm[c] = tauR[c, 1]

            WM = np.zeros(self.n)
            WM[M] = (TAUM[M] - tauB[M]) / tauP[M]
            WM[m] = (TAUm[m] - tauB[m]) / tauP[m]

            WM[WM == np.NINF] = np.Inf

            wmax[i, :] = WM
            joint[i] = np.argmin(WM)

        if trajn == 1:
            return wmax[0, :], joint[0]
        else:
            return wmax, joint

    def perturb(self, p=0.1):
        """
        Perturb robot parameters

        rp = perturb(p) is a new robot object in which the dynamic parameters
        (link mass and inertia) have been perturbed. The perturbation is
        multiplicative so that values are multiplied by random numbers in the
        interval (1-p) to (1+p). The name string of the perturbed robot is
        prefixed by 'P/'.

        Useful for investigating the robustness of various model-based control
        schemes. For example to vary parameters in the range +/- 10 percent
        is: r2 = puma.perturb(0.1)

        :param p: The percent (+/-) to be perturbed. Default 10%
        :type p: float

        :return: A copy of the robot with dynamic parameters perturbed
        :rtype: DHRobot

        """

        r2 = self.copy()
        r2.name = 'P/' + self.name

        for i in range(self.n):
            s = (2 * np.random.random() - 1) * p + 1
            r2.links[i].m = r2.links[i].m * s

            s = (2 * np.random.random() - 1) * p + 1
            r2.links[i].I = r2.links[i].I * s    # noqa

        return r2

    def rne_python(
            self, Q, QD=None, QDD=None,
            grav=None, fext=None, debug=False, basewrench=False):
        """
        Compute inverse dynamics via recursive Newton-Euler formulation

        :param Q: Joint coordinates
        :param QD: Joint velocity
        :param QDD: Joint acceleration
        :param grav: [description], defaults to None
        :type grav: [type], optional
        :param fext: end-effector wrench, defaults to None
        :type fext: 6-element array-like, optional
        :param debug: print debug information to console, defaults to False
        :type debug: bool, optional
        :param basewrench: compute the base wrench, defaults to False
        :type basewrench: bool, optional
        :raises ValueError: for misshaped inputs
        :return: Joint force/torques
        :rtype: NumPy array

        Recursive Newton-Euler for standard Denavit-Hartenberg notation.

        - ``rne_dh(q, qd, qdd)`` where the arguments have shape (n,) where n is
          the number of robot joints.  The result has shape (n,).
        - ``rne_dh(q, qd, qdd)`` where the arguments have shape (m,n) where n
          is the number of robot joints and where m is the number of steps in
          the joint trajectory.  The result has shape (m,n).
        - ``rne_dh(p)`` where the input is a 1D array ``p`` = [q, qd, qdd] with
          shape (3n,), and the result has shape (n,).
        - ``rne_dh(p)`` where the input is a 2D array ``p`` = [q, qd, qdd] with
          shape (m,3n) and the result has shape (m,n).

        .. note::
            - This is a pure Python implementation and slower than the .rne()
            which is written in C.
            - This version supports symbolic model parameters
            - Verified against MATLAB code

        :seealso: :func:`rne`
        """

        def removesmall(x):
            return x

        n = self.n

        if self.symbolic:
            dtype = 'O'
        else:
            dtype = None

        z0 = np.array([0, 0, 1], dtype=dtype)

        if grav is None:
            grav = self.gravity  # default gravity from the object
        else:
            grav = getvector(grav, 3)

        if fext is None:
            fext = np.zeros((6,), dtype=dtype)
        else:
            fext = getvector(fext, 6)

        if debug:
            print('grav', grav)
            print('fext', fext)

        # unpack the joint coordinates and derivatives
        if Q is not None and QD is None and QDD is None:
            # single argument case
            Q = getmatrix(Q, (None, self.n * 3))
            q = Q[:, 0:n]
            qd = Q[:, n:2 * n]
            qdd = Q[:, 2 * n:]

        else:
            # 3 argument case
            q = getmatrix(Q, (None, self.n))
            qd = getmatrix(QD, (None, self.n))
            qdd = getmatrix(QDD, (None, self.n))

        nk = q.shape[0]

        tau = np.zeros((nk, n), dtype=dtype)
        if basewrench:
            wbase = np.zeros((nk, n), dtype=dtype)

        for k in range(nk):
            # take the k'th row of data
            q_k = q[k, :]
            qd_k = qd[k, :]
            qdd_k = qdd[k, :]

            if debug:
                print('q_k', q_k)
                print('qd_k', qd_k)
                print('qdd_k', qdd_k)
                print()

            # joint vector quantities stored columwise in matrix
            #  m suffix for matrix
            Fm = np.zeros((3, n), dtype=dtype)
            Nm = np.zeros((3, n), dtype=dtype)
            # if robot.issym
            #     pstarm = sym([]);
            # else
            #     pstarm = [];
            pstarm = np.zeros((3, n), dtype=dtype)
            Rm = []

            # rotate base velocity and acceleration into L1 frame
            Rb = t2r(self.base.A).T
            # base has zero angular velocity
            w = Rb @ np.zeros((3,), dtype=dtype)
            # base has zero angular acceleration
            wd = Rb @ np.zeros((3,), dtype=dtype)
            vd = Rb @ grav

            # ----------------  initialize some variables ----------------- #

            for j in range(n):
                link = self.links[j]

                # compute the link rotation matrix
                if link.sigma == 0:
                    # revolute axis
                    Tj = link.A(q_k[j]).A
                    d = link.d
                else:
                    # prismatic
                    Tj = link.A(link.theta).A
                    d = q_k[j]

                # compute pstar:
                #   O_{j-1} to O_j in {j}, negative inverse of link xform
                alpha = link.alpha
                if self.mdh:
                    pstar = np.r_[
                        link.a, -d * sym.sin(alpha), d * sym.cos(alpha)]
                    if j == 0:
                        if self._base:
                            Tj = self._base.A @ Tj
                            pstar = self._base.A @ pstar
                else:
                    pstar = np.r_[
                        link.a, d * sym.sin(alpha), d * sym.cos(alpha)]

                # stash them for later
                Rm.append(t2r(Tj))
                pstarm[:, j] = pstar

            # -----------------  the forward recursion -------------------- #

            for j, link in enumerate(self.links):

                Rt = Rm[j].T    # transpose!!
                pstar = pstarm[:, j]
                r = link.r

                # statement order is important here

                if self.mdh:
                    if link.isrevolute():
                        # revolute axis
                        w_ = Rt @ w + z0 * qd_k[j]
                        wd_ = Rt @ wd \
                            + z0 * qdd_k[j] \
                            + _cross(Rt @ w, z0 * qd_k[j])
                        vd_ = Rt @ _cross(wd, pstar) \
                            + _cross(w, _cross(w, pstar)) \
                            + vd
                    else:
                        # prismatic axis
                        w_ = Rt @ w
                        wd_ = Rt @ wd
                        vd_ = Rt @ (
                            _cross(wd, pstar)
                            + _cross(w, _cross(w, pstar))
                            + vd
                            ) \
                            + 2 * _cross(Rt @ w, z0 * qd_k[j]) \
                            + z0 * qdd_k[j]
                    # trailing underscore means new value, update here
                    w = w_
                    wd = wd_
                    vd = vd_
                else:
                    if link.isrevolute():
                        # revolute axis
                        wd = Rt @ (
                            wd + z0 * qdd_k[j]
                            + _cross(w, z0 * qd_k[j]))
                        w = Rt @ (w + z0 * qd_k[j])
                        vd = _cross(wd, pstar) \
                            + _cross(w, _cross(w, pstar)) \
                            + Rt @ vd
                    else:
                        # prismatic axis
                        w = Rt @ w
                        wd = Rt @ wd
                        vd = Rt @  (z0 * qdd_k[j] + vd) \
                            + _cross(wd, pstar) \
                            + 2 * _cross(w, Rt @ z0 * qd_k[j]) \
                            + _cross(w, _cross(w, pstar))

                vhat = _cross(wd, r) \
                    + _cross(w, _cross(w, r)) \
                    + vd
                Fm[:, j] = link.m * vhat
                Nm[:, j] = link.I @ wd + _cross(w, link.I @ w)

                if debug:
                    print('w:     ', removesmall(w))
                    print('wd:    ', removesmall(wd))
                    print('vd:    ', removesmall(vd))
                    print('vdbar: ', removesmall(vhat))
                    print()

            if debug:
                print('Fm\n', Fm)
                print('Nm\n', Nm)

            # -----------------  the backward recursion -------------------- #

            f = fext[:3]      # force/moments on end of arm
            nn = fext[3:]

            for j in range(n - 1, -1, -1):
                link = self.links[j]
                r = link.r

                #
                # order of these statements is important, since both
                # nn and f are functions of previous f.
                #
                if self.mdh:
                    if j == (n - 1):
                        R = np.eye(3, dtype=dtype)
                        pstar = np.zeros((3,), dtype=dtype)
                    else:
                        R = Rm[j + 1]
                        pstar = pstarm[:, j + 1]

                    f_ = R @ f + Fm[:, j]
                    nn_ = R @ nn \
                        + _cross(pstar, R @ f) \
                        + _cross(pstar, Fm[:, j]) \
                        + Nm[:, j]
                    f = f_
                    nn = nn_

                else:
                    pstar = pstarm[:, j]
                    if j == (n - 1):
                        R = np.eye(3, dtype=dtype)
                    else:
                        R = Rm[j + 1]

                    nn = R @ (nn + _cross(R.T @ pstar, f)) \
                        + _cross(pstar + r, Fm[:, j]) \
                        + Nm[:, j]
                    f = R @ f + Fm[:, j]

                if debug:
                    print('f: ', removesmall(f))
                    print('n: ', removesmall(nn))

                R = Rm[j]
                if self.mdh:
                    if link.isrevolute():
                        # revolute axis
                        t = nn @ z0
                    else:
                        # prismatic
                        t = f @ z0
                else:
                    if link.isrevolute():
                        # revolute axis
                        t = nn @ (R.T @ z0)
                    else:
                        # prismatic
                        t = f @ (R.T @ z0)

                # add joint inertia and friction
                #  no Coulomb friction if model is symbolic
                tau[k, j] = t \
                    + link.G ** 2 * link.Jm * qdd_k[j] \
                    - link.friction(qd_k[j], coulomb=not self.symbolic)
                if debug:
                    print(
                        f'j={j:}, G={link.G:}, Jm={link.Jm:}, friction={link.friction(qd_k[j], coulomb=False):}')  # noqa
                    print()

            # compute the base wrench and save it
            if basewrench:
                R = Rm[0]
                nn = R @ nn
                f = R @ f
                wbase[k, :] = np.r_[f, nn]

        # if self.symbolic:
        #     # simplify symbolic expressions
        #     print(
        #       'start symbolic simplification, this might take a while...')
        #     # from sympy import trigsimp

        #     # tau = trigsimp(tau)
        #     # consider using multiprocessing to spread over cores
        #     #  https://stackoverflow.com/questions/33844085/using-multiprocessing-with-sympy  # noqa
        #     print('done')
        #     if tau.shape[0] == 1:
        #         return tau.reshape(self.n)
        #     else:
        #         return tau

        if tau.shape[0] == 1:
            return tau.flatten()
        else:
            return tau


def _printProgressBar(
        fraction, prefix='', suffix='', decimals=1,
        length=50, fill='█', printEnd="\r"):

    percent = ("{0:." + str(decimals) + "f}").format(fraction * 100)
    filledLength = int(length * fraction)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)


def _cross(a, b):
    return np.r_[
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]]


if __name__ == "__main__":   # pragma nocover

    import roboticstoolbox as rtb
    # from spatialmath.base import symbolic as sym

    puma = rtb.models.DH.Puma560()

    # for j, link in enumerate(puma):
    #     print(f'joint {j:}::')
    #     print(link.dyn(indent=4))
    #     print()

    # tau = puma.rne_dh(puma.qz, puma.qz, puma.qz)
    # print(tau)
    # tau = puma.rne_dh(np.r_[puma.qz, puma.qz, puma.qz])
    # print(tau)
    # tau = puma.rne_dh([0,0,0,0,0,0],  [0,0,0,0,0,0],  [0,0,0,0,0,0])
    # print(tau)
    # tau = puma.rne_dh([0,0,0,0,0,0,  0,0,0,0,0,0,  0,0,0,0,0,0])
    # print(tau)

    # puma = rtb.models.DH.Puma560(symbolic=True)
    # print(puma)
    # g = sym.symbol('g')
    # puma.gravity = [0, 0, g]
    # q = sym.symbol('q_:6')
    # qd = sym.symbol('qd_:6')
    # qdd = sym.symbol('qdd_:6')

    # tau = puma.rne_dh(q, qd, qdd, debug=False)

    # print(tau[0].coeff(qdd[0]))
    # print(tau[0].expand().coeff(qdd[0]))

    q = puma.qz
    # qd = puma.qz
    # qdd = puma.qz
    ones = np.ones((6,))
    qd = ones
    qdd = ones

    print(puma.rne(q, qd, qdd))
    print(puma.rne_python(q, qd, qdd, debug=False))

    print(puma.gravity)
    print([link.isrevolute() for link in puma])

    # NOT CONVINCED WE NEED THIS, AND IT'S ORPHAN CODE
    # def gravjac(self, q, grav=None):
    #     """
    #     Compute gravity load and Jacobian

    #     :param q: The joint configuration of the robot
    #     :type q: ndarray(n)
    #     :param grav: The gravity vector (Optional, if not supplied will
    #         use the stored gravity values).
    #     :type grav: ndarray(3,)

    #     :return tau: The generalised joint force/torques due to gravity
    #     :rtype tau: ndarray(n,)

    #     ``tauB = gravjac(q, grav)`` calculates the generalised joint force/
    #     torques due to gravity and the Jacobian

    #     Trajectory operation:
    #     If q is nxm where n is the number of robot joints then a
    #     trajectory is assumed where each row of q corresponds to a robot
    #     configuration. tau (nxm) is the generalised joint torque, each row
    #     corresponding to an input pose, and jacob0 (6xnxm) where each
    #     plane is a Jacobian corresponding to an input pose.

    #     .. note::
    #         - The gravity vector is defined by the SerialLink property if not
    #           explicitly given.
    #         - Does not use inverse dynamics function RNE.
    #         - Faster than computing gravity and Jacobian separately.

    #     Written by Bryan Moutrie

    #     :seealso: :func:`gravload`
    #     """

    #     # TODO use np.cross instead
    #     def _cross3(self, a, b):
    #         c = np.zeros(3)
    #         c[2] = a[0] * b[1] - a[1] * b[0]
    #         c[0] = a[1] * b[2] - a[2] * b[1]
    #         c[1] = a[2] * b[0] - a[0] * b[2]
    #         return c

    #     def makeJ(O, A, e, r):
    #         J[3:6,:] = A
    #         for j in range(r):
    #             if r[j]:
    #                 J[0:3,j] = cross3(A(:,j),e-O(:,j));
    #             else:
    #                 J[:,j] = J[[4 5 6 1 2 3],j]; %J(1:3,:) = 0;

    #     if grav is None:
    #         grav = np.copy(self.gravity)
    #     else:
    #         grav = getvector(grav, 3)

    #     try:
    #         if q is not None:
    #             q = getvector(q, self.n, 'col')
    #         else:
    #             q = np.copy(self.q)
    #             q = getvector(q, self.n, 'col')

    #         poses = 1
    #     except ValueError:
    #         poses = q.shape[1]
    #         verifymatrix(q, (self.n, poses))

    #     if not self.mdh:
    #         baseAxis = self.base.a
    #         baseOrigin = self.base.t

    #     tauB = np.zeros((self.n, poses))

    #     # Forces
    #     force = np.zeros((3, self.n))

    #     for joint in range(self.n):
    #         force[:, joint] = np.squeeze(self.links[joint].m * grav)

    #     # Centre of masses (local frames)
    #     r = np.zeros((4, self.n))
    #     for joint in range(self.n):
    #         r[:, joint] = np.r_[np.squeeze(self.links[joint].r), 1]

    #     for pose in range(poses):
    #         com_arr = np.zeros((3, self.n))

    #         T = self.fkine_all(q[:, pose])

    #         jointOrigins = np.zeros((3, self.n))
    #         jointAxes = np.zeros((3, self.n))
    #         for i in range(self.n):
    #             jointOrigins[:, i] = T[i].t
    #             jointAxes[:, i] = T[i].a

    #         if not self.mdh:
    #             jointOrigins = np.c_[
    #                 baseOrigin, jointOrigins[:, :-1]
    #             ]
    #             jointAxes = np.c_[
    #                 baseAxis, jointAxes[:, :-1]
    #             ]

    #         # Backwards recursion
    #         for joint in range(self.n - 1, -1, -1):
    #             # C.o.M. in world frame, homog
    #             com = T[joint].A @ r[:, joint]

    #             # Add it to the distal others
    #             com_arr[:, joint] = com[0:3]

    #             t = np.zeros(3)

    #             # for all links distal to it
    #             for link in range(joint, self.n):
    #                 if not self.links[joint].sigma:
    #                     # Revolute joint
    #                     d = com_arr[:, link] - jointOrigins[:, joint]
    #                     t = t + self._cross3(d, force[:, link])
    #                     # Though r x F would give the applied torque
    #                     # and not the reaction torque, the gravity
    #                     # vector is nominally in the positive z
    #                     # direction, not negative, hence the force is
    #                     # the reaction force
    #                 else:
    #                     # Prismatic joint
    #                     # Force on prismatic joint
    #                     t = t + force[:, link]

    #             tauB[joint, pose] = t.T @ jointAxes[:, joint]

    #     if poses == 1:
    #         return tauB[:, 0]
    #     else:
    #         return tauB
