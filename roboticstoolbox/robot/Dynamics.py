"""
Rigid-body dynamics functionality of the Toolbox.

Requires access to:

    * ``links`` list of ``Link`` objects, atttribute
    * ``rne()`` the inverse dynamics method

so must be subclassed by ``DHRobot`` class.

:todo: perhaps these should be abstract properties, methods of this calss
"""
from collections import namedtuple
from typing import Any, Callable, Dict, Union
import numpy as np
from spatialmath.base import getvector, verifymatrix, isscalar, getmatrix, t2r, rot2jac
from scipy import integrate, interpolate
from spatialmath.base import symbolic as sym
from roboticstoolbox import rtb_get_param
from roboticstoolbox.robot.RobotProto import RobotProto

from roboticstoolbox.tools.types import ArrayLike, NDArray
from typing_extensions import Self
import roboticstoolbox as rtb

from ansitable import ANSITable, Column
import warnings


class DynamicsMixin:

    # --------------------------------------------------------------------- #
    def dynamics(self: RobotProto):
        """
        Pretty print the dynamic parameters (Robot superclass)

        The dynamic parameters (inertial and friction) are printed in a table,
        with one row per link.

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot.links[2].dyntable()
        >>> robot.dyntable()

        """
        unicode = rtb_get_param("unicode")
        table = ANSITable(
            Column("j", colalign=">", headalign="^"),
            Column("m", colalign="<", headalign="^"),
            Column("r", colalign="<", headalign="^"),
            Column("I", colalign="<", headalign="^"),
            Column("Jm", colalign="<", headalign="^"),
            Column("B", colalign="<", headalign="^"),
            Column("Tc", colalign="<", headalign="^"),
            Column("G", colalign="<", headalign="^"),
            border="thin" if unicode else "ascii",
        )

        for j, link in enumerate(self.links):
            table.row(link.name, *link._dyn2list())
        table.print()

    def dynamics_list(self: RobotProto):
        """
        Print dynamic parameters (Robot superclass)

        Display the kinematic and dynamic parameters to the console in
        reable format

        """
        for j, link in enumerate(self.links):
            print("\nLink {:d}::".format(j), link)
            print(link.dyn(indent=2))

    # --------------------------------------------------------------------- #

    def friction(self: RobotProto, qd: NDArray) -> NDArray:
        r"""
        Manipulator joint friction (Robot superclass)

        ``robot.friction(qd)`` is a vector of joint friction
        forces/torques for the robot moving with joint velocities ``qd``.

        The friction model includes:

        - Viscous friction which is a linear function of velocity.
        - Coulomb friction which is proportional to sign(qd).

        .. math::

            \tau_j = G^2 B \dot{q}_j + |G_j| \left\{ \begin{array}{ll}
                \tau_{C,j}^+ & \mbox{if $\dot{q}_j > 0$} \\
                \tau_{C,j}^- & \mbox{if $\dot{q}_j < 0$} \end{array} \right.

        Parameters
        ----------
        qd
            The joint velocities of the robot

        Returns
        -------
        friction
            The joint friction forces/torques for the robot

        Notes
        -----
        - The friction value should be added to the motor output torque to
            determine the nett torque. It has a negative value when qd > 0.
        - The returned friction value is referred to the output of the
            gearbox.
        - The friction parameters in the Link object are referred to the
            motor.
        - Motor viscous friction is scaled up by :math:`G^2`.
        - Motor Coulomb friction is scaled up by math:`G`.
        - The appropriate Coulomb friction value to use in the
            non-symmetric case depends on the sign of the joint velocity,
            not the motor velocity.
        - Coulomb friction is zero for zero joint velocity, stiction is
            not modeled.
        - The absolute value of the gear ratio is used.  Negative gear
            ratios are tricky: the Puma560 robot has negative gear ratio for
            joints 1 and 3.
        - The absolute value of the gear ratio is used. Negative gear
            ratios are tricky: the Puma560 has negative gear ratio for
            joints 1 and 3.

        See Also
        --------
        :func:`Robot.nofriction`
        :func:`Link.friction`

        """

        qd = np.array(getvector(qd, self.n))
        tau = np.zeros(self.n)

        for i in range(self.n):
            tau[i] = self.links[i].friction(qd[i])

        return tau

    # --------------------------------------------------------------------- #

    def nofriction(self: RobotProto, coulomb: bool = True, viscous: bool = False):
        """
        Remove manipulator joint friction

        ``nofriction()`` copies the robot and returns
        a robot with the same link parameters except the Coulomb and/or viscous
        friction parameter are set to zero.

        Parameters
        ----------
        coulomb
            set the Coulomb friction to 0
        viscous
            set the viscous friction to 0

        Returns
        -------
        robot
            A copy of the robot with dynamic parameters perturbed

        See Also
        --------
        :func:`Robot.friction`
        :func:`Link.nofriction`

        """

        # shallow copy the robot object
        if isinstance(self, rtb.DHRobot):
            self.delete_rne()  # remove the inherited C pointers

        nf = self.copy()
        nf.name = "NF/" + self.name

        # add the modified links (copies)
        nf._links = [link.nofriction(coulomb, viscous) for link in self.links]

        return nf

    def fdyn(
        self: RobotProto,
        T: float,
        q0: ArrayLike,
        Q: Union[Callable[[Any, float, NDArray, NDArray], NDArray], None] = None,
        Q_args: Dict = {},
        qd0: Union[ArrayLike, None] = None,
        solver: str = "RK45",
        solver_args: Dict = {},
        dt: Union[float, None] = None,
        progress: bool = False,
    ):
        """
        Integrate forward dynamics


        ``tg = R.fdyn(T, q)`` integrates the dynamics of the robot with zero
        input torques over the time  interval 0 to ``T`` and returns the
        trajectory as a namedtuple with elements:

        - ``t`` the time vector (M,)
        - ``q`` the joint coordinates (M,n)
        - ``qd`` the joint velocities (M,n)

        ``tg = R.fdyn(T, q, torqfun)`` as above but the torque applied to the
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

        Parameters
        ----------
        T
            integration time
        q0
            initial joint coordinates
        qd0
            initial joint velocities, assumed zero if not given
        Q
            a function that computes generalized joint force as a function of
            time and/or state
        Q_args
            positional arguments passed to ``torque``
        solver
            str
        solver_args
            dict
        dt
            float
        progress
            show progress bar, default False

        Returns
        -------
        trajectory
            robot trajectory

        Examples
        --------

        To apply zero joint torque to the robot without Coulomb
        friction:

        >>> def myfunc(robot, t, q, qd):
        >>>     return np.zeros((robot.n,))

        >>> tg = robot.nofriction().fdyn(5, q0, myfunc)

        >>> plt.figure()
        >>> plt.plot(tg.t, tg.q)
        >>> plt.show()

        We could also use a lambda function::

        >>> tg = robot.nofriction().fdyn(
        >>>     5, q0, lambda r, t, q, qd: np.zeros((r.n,)))

        The robot is controlled by a PD controller. We first define a
        function to compute the control which has additional parameters for
        the setpoint and control gains (qstar, P, D)::

        >>> def myfunc(robot, t, q, qd, qstar, P, D):
        >>>     return (qstar - q) * P + qd * D  # P, D are (6,)

        >>> tg = robot.fdyn(10, q0, myfunc, torque_args=(qstar, P, D)) )

        Many integrators have variable step length which is problematic if we
        want to animate the result.  If ``dt`` is specified then the solver
        results are interpolated in time steps of ``dt``.

        Notes
        -----
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

        See Also
        --------
        :func:`DHRobot.accel`
        :func:`DHRobot.nofriction`,
        :func:`DHRobot.rne`.

        """

        n = self.n

        if not isscalar(T):
            raise ValueError("T must be a scalar")
        q0 = getvector(q0, n)
        if qd0 is None:
            qd0 = np.zeros((n,))
        else:
            qd0 = getvector(qd0, n)
        if Q is not None:
            if not callable(Q):
                raise ValueError("generalized joint torque function must be callable")

        # concatenate q and qd into the initial state vector
        x0 = np.r_[q0, qd0]

        # get user specified integrator
        scipy_integrator = integrate.__dict__[solver]

        integrator = scipy_integrator(
            lambda t, y: self._fdyn(t, y, Q, Q_args),
            t0=0.0,
            y0=x0,
            t_bound=T,
            **solver_args,
        )

        # initialize list of time and states
        tlist = [0]
        xlist = [np.r_[q0, qd0]]

        if progress:
            _printProgressBar(0, prefix="Progress:", suffix="complete", length=60)

        while integrator.status == "running":

            # step the integrator, calls _fdyn multiple times
            integrator.step()

            if integrator.status == "failed":
                raise RuntimeError("integration completed with failed status ")

            # stash the results
            tlist.append(integrator.t)
            xlist.append(integrator.y)

            # update the progress bar
            if progress:
                _printProgressBar(
                    integrator.t / T, prefix="Progress:", suffix="complete", length=60
                )

        # cleanup the progress bar
        if progress:
            print("\r" + " " * 90 + "\r")

        tarray = np.array(tlist)
        xarray = np.array(xlist)

        if dt is not None:
            # interpolate data to equal time steps of dt
            interp = interpolate.interp1d(tarray, xarray, axis=0)

            tnew = np.arange(0, T, dt)
            xnew = interp(tnew)
            return namedtuple("fdyn", "t q qd")(tnew, xnew[:, :n], xnew[:, n:])
        else:
            return namedtuple("fdyn", "t q qd")(tarray, xarray[:, :n], xarray[:, n:])

    def _fdyn(
        self: RobotProto,
        t: float,
        x: NDArray,
        Qfunc: Callable[[Any, float, NDArray, NDArray], NDArray],
        Qargs: Dict,
    ):
        """
        Private function called by fdyn

        Called by ``fdyn`` to evaluate the robot velocity and acceleration for
        forward dynamics.

        Parameters
        ----------
        t
            current time
        x
            current state [q, qd]
        Qfunc
            a function that computes torque as a function of time
            and/or state
        Qargs : dict
            argumments passed to ``Qfunc``

        Returns
        -------
        fdyn
            derivative of current state [qd, qdd]

        """
        n = self.n

        q = x[0:n]
        qd = x[n:]

        # evaluate the torque function if one is given
        if Qfunc is None:
            tau = np.zeros((n,))
        else:
            tau = Qfunc(self, t, q, qd, **Qargs)
            if len(tau) != n or not all(np.isreal(tau)):
                raise RuntimeError(
                    "torque function must return vector with N real elements"
                )

        qdd = self.accel(q, qd, tau)

        return np.r_[qd, qdd]

    def accel(self: RobotProto, q, qd, torque, gravity=None):
        r"""
        Compute acceleration due to applied torque

        ``qdd = accel(q, qd, torque)`` calculates a vector (n) of joint
        accelerations that result from applying the actuator force/torque (n)
        to the manipulator in state `q` (n) and `qd` (n), and ``n`` is
        the number of robot joints.

        .. math::

            \ddot{q} = \mathbf{M}^{-1} \left(\tau - \mathbf{C}(q)\dot{q} - \mathbf{g}(q)\right)

        **Trajectory operation**

        If `q`, `qd`, torque are matrices (m,n) then ``qdd`` is a matrix (m,n)
        where each row is the acceleration corresponding to the equivalent rows
        of q, qd, torque.

        Parameters
        ----------
        q
            Joint coordinates
        qd
            Joint velocity
        torque
            Joint torques of the robot
        gravity
            Gravitational acceleration (Optional, if not supplied will
            use the ``gravity`` attribute of self).

        Returns
        -------
        ndarray(n)

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.accel(puma.qz, 0.5 * np.ones(6), np.zeros(6))

        Notes
        -----
        - Useful for simulation of manipulator dynamics, in
            conjunction with a numerical integration function.
        - Uses the method 1 of Walker and Orin to compute the forward
            dynamics.
        - Featherstone's method is more efficient for robots with large
            numbers of joints.
        - Joint friction is considered.

        References
        ----------
        - Efficient dynamic computer simulation of robotic mechanisms,
            M. W. Walker and D. E. Orin,
            ASME Journa of Dynamic Systems, Measurement and Control, vol.
            104, no. 3, pp. 205-211, 1982.

        """  # noqa

        q = getmatrix(q, (None, self.n))
        qd = getmatrix(qd, (None, self.n))
        torque = getmatrix(torque, (None, self.n))

        qdd = np.zeros((q.shape[0], self.n))

        for k, (qk, qdk, tauk) in enumerate(zip(q, qd, torque)):
            # Compute current manipulator inertia torques resulting from unit
            # acceleration of each joint with no gravity.
            qI = (np.c_[qk] @ np.ones((1, self.n))).T
            qdI = np.zeros((self.n, self.n))
            qddI = np.eye(self.n)

            M = self.rne(qI, qdI, qddI, gravity=[0, 0, 0])

            # Compute gravity and coriolis torque torques resulting from zero
            # acceleration at given velocity & with gravity acting.
            tau = self.rne(qk, qdk, np.zeros((1, self.n)), gravity=gravity)

            # solve is faster than inv() which is faster than pinv()
            qdd[k, :] = np.linalg.solve(M, tauk - tau)

        if q.shape[0] == 1:
            return qdd[0, :]
        else:
            return qdd

    def pay(
        self: RobotProto,
        W: ArrayLike,
        q: Union[NDArray, None] = None,
        J: Union[NDArray, None] = None,
        frame: int = 1,
    ):
        """
        Generalised joint force/torque due to a payload wrench

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

        Parameters
        ----------
        W
            A wrench vector applied at the end effector,
            W = [Fx Fy Fz Mx My Mz]
        q
            Joint coordinates
        J
            The manipulator Jacobian (Optional, if not supplied will
            use the q value).
        frame
            The frame in which to torques are expressed in when J
            is not supplied. 0 means base frame of the robot, 1 means end-
            effector frame

        Returns
        -------
        t
            Joint forces/torques due to w

        Notes
        -----
        - Wrench vector and Jacobian must be from the same reference
            frame.
        - Tool transforms are taken into consideration when frame=1.
        - Must have a constant wrench - no trajectory support for this
            yet.

        """

        try:
            W = np.array(getvector(W, 6))
            trajn = 0
        except ValueError:
            if isinstance(W, NDArray):
                trajn = W.shape[0]
                verifymatrix(W, (trajn, 6))
            else:
                raise ValueError("W is invalid")

        if trajn:
            # A trajectory
            if J is not None:
                # Jacobian supplied
                verifymatrix(J, (trajn, 6, self.n))
            elif q is not None:
                # Use q instead
                verifymatrix(q, (trajn, self.n))
                J = np.zeros((trajn, 6, self.n))
                for i in range(trajn):
                    if frame:
                        J[i, :, :] = self.jacobe(q[i, :])
                    else:
                        J[i, :, :] = self.jacob0(q[i, :])
            else:
                raise ValueError("q of J is needed for trajectory")
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

    def payload(self: RobotProto, m: float, p=np.zeros(3)):
        """
        Add a payload to the end-effector

        payload(m, p) adds payload mass adds a payload with point mass m at
        position p in the end-effector coordinate frame.

        payload(m) adds payload mass adds a payload with point mass m at
        in the end-effector coordinate frame.

        payload(0) removes added payload.

        Parameters
        ----------
        m
            mass (kg)
        p
            position in end-effector frame

        """

        p = getvector(p, 3, out="col")
        lastlink = self.links[self.n - 1]

        lastlink.m = m
        lastlink.r = p

    def jointdynamics(self: RobotProto, q, qd=None):
        """
        Transfer function of joint actuator

        ``tf = jointdynamics(qd, q)`` calculates a vector of n
        continuous-time transfer functions that represent the transfer
        function 1/(Js+B) for each joint based on the dynamic parameters
        of the robot and the configuration q (n). n is the number of robot
        joints.  The result is a list of tuples (J, B) for each joint.

        ``tf = jointdynamics(q, qd)`` as above but include the linearized
        effects of Coulomb friction when operating at joint velocity QD
        (1xN).

        Parameters
        ----------
        q
            Joint coordinates
        qd
            Joint velocity

        Returns
        -------
        list of 2-tuples
            transfer function denominators

        """

        tf = []
        for j, link in enumerate(self.links):

            # compute inertia for this joint
            zero = np.zeros((self.n))
            qdd = np.zeros((self.n))
            qdd[j] = 1
            M = self.rne(q, zero, qdd, gravity=[0, 0, 0])
            J = link.Jm + M[j] / abs(link.G) ** 2

            # compute friction
            B = link.B
            if qd is not None:
                # add linearized Coulomb friction at the operating point
                if qd > 0:
                    B += link.Tc[0] / qd[j]
                elif qd < 0:
                    B += link.Tc[1] / qd[j]
            tf.append(((1,), (J, B)))

        return tf

    def cinertia(self: RobotProto, q):
        """
        Deprecated, use ``inertia_x``

        """
        warnings.warn("cinertia is deprecated, use inertia_x", DeprecationWarning)

    def inertia(self: RobotProto, q: NDArray) -> NDArray:
        """Manipulator inertia matrix
        ``inertia(q)`` is the symmetric joint inertia matrix (n,n) which
        relates joint torque to joint acceleration for the robot at joint
        configuration q.

        **Trajectory operation**

        If ``q`` is a matrix (m,n), each row is interpretted as a joint state
        vector, and the result is a 3d-matrix (nxnxk) where each plane
        corresponds to the inertia for the corresponding row of q.

        Parameters
        ----------
        q
            Joint coordinates

        Returns
        -------
        inertia
            The inertia matrix

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.inertia(puma.qz)

        Notes
        -----
        - The diagonal elements ``M[j,j]`` are the inertia seen by joint
            actuator ``j``.
        - The off-diagonal elements ``M[j,k]`` are coupling inertias that
            relate acceleration on joint ``j`` to force/torque on
            joint ``k``.
        - The diagonal terms include the motor inertia reflected through
            the gear ratio.

        See Also
        --------
        :func:`cinertia`

        """
        q = getmatrix(q, (None, self.n))

        In = np.zeros((q.shape[0], self.n, self.n))

        for k, qk in enumerate(q):
            In[k, :, :] = self.rne(
                (np.c_[qk] @ np.ones((1, self.n))).T,
                np.zeros((self.n, self.n)),
                np.eye(self.n),
                gravity=[0, 0, 0],
            )

        if q.shape[0] == 1:
            return In[0, :, :]
        else:
            return In

    def coriolis(self: RobotProto, q, qd):
        r"""
        Coriolis and centripetal term

        ``coriolis(q, qd)`` calculates the Coriolis/centripetal matrix (n,n)
        for the robot in configuration ``q`` and velocity ``qd``, where ``n``
        is the number of joints.

        The product :math:`\mathbf{C} \dot{q}` is the vector of joint
        force/torque due to velocity coupling. The diagonal elements are due to
        centripetal effects and the off-diagonal elements are due to Coriolis
        effects. This matrix is also known as the velocity coupling matrix,
        since it describes the disturbance forces on any joint due to
        velocity of all other joints.

        **Trajectory operation**

        If ``q`` and `qd` are matrices (m,n), each row is interpretted as a
        joint configuration, and the result (n,n,m) is a 3d-matrix where
        each plane corresponds to a row of ``q`` and ``qd``.

        Parameters
        ----------
        q
            Joint coordinates
        qd
            Joint velocity

        Returns
        -------
        coriolis
            Velocity matrix

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.coriolis(puma.qz, 0.5 * np.ones((6,)))

        Notes
        -----
        - Joint viscous friction is also a joint force proportional to
            velocity but it is eliminated in the computation of this value.
        - Computationally slow, involves :math:`n^2/2` invocations of RNE.

        """

        q = getmatrix(q, (None, self.n))
        qd = getmatrix(qd, (None, self.n))
        if q.shape[0] != qd.shape[0]:
            raise ValueError("q and qd must have the same number of rows")

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
                tau = r1.rne(qk, QD, np.zeros(self.n), gravity=[0, 0, 0])
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
                    tau = r1.rne(qk, QD, np.zeros(self.n), gravity=[0, 0, 0])

                    C[k, :, j] = (
                        C[k, :, j] + (tau - Csq[k, :, j] - Csq[k, :, i]) * qdk[i] / 2
                    )

                    C[k, :, i] = (
                        C[k, :, i] + (tau - Csq[k, :, j] - Csq[k, :, i]) * qdk[j] / 2
                    )

            C[k, :, :] = C[k, :, :] + Csq[k, :, :] @ np.diag(qdk)

        if q.shape[0] == 1:
            return C[0, :, :]
        else:
            return C

    def gravload(
        self: RobotProto,
        q: Union[ArrayLike, None] = None,
        gravity: Union[ArrayLike, None] = None,
    ):
        """
        Compute gravity load

        ``robot.gravload(q)`` calculates the joint gravity loading (n) for
        the robot in the joint configuration ``q`` and using the default
        gravitational acceleration specified in the DHRobot object.

        ``robot.gravload(q, gravity=g)`` as above except the gravitational
        acceleration is explicitly specified as ``g``.

        **Trajectory operation**

        If q is a matrix (nxm) each column is interpreted as a joint
        configuration vector, and the result is a matrix (nxm) each column
        being the corresponding joint torques.

        Parameters
        ----------
        q
            Joint coordinates
        gravity : ndarray(3)
            Gravitational acceleration (Optional, if not supplied will
            use the stored gravity values).

        Returns
        -------
        gravload
            The generalised joint force/torques due to gravity

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.gravload(puma.qz)

        """

        q = getmatrix(q, (None, self.n))

        if gravity is None:
            gravity = self.gravity
        else:
            gravity = getvector(gravity, 3)

        taug = np.zeros((q.shape[0], self.n))
        z = np.zeros(self.n)

        for k, qk in enumerate(q):
            taug[k, :] = self.rne(qk, z, z, gravity=gravity)

        if q.shape[0] == 1:
            return taug[0, :]
        else:
            return taug

    def inertia_x(
        self: RobotProto, q=None, pinv=False, representation="rpy/xyz", Ji=None
    ):
        r"""
        Operational space inertia matrix

        ``robot.inertia_x(q)`` is the operational space (Cartesian) inertia
        matrix which relates Cartesian force/torque to Cartesian
        acceleration at the joint configuration q.

        .. math::

            \mathbf{M}_x = \mathbf{J}(q)^{-T} \mathbf{M}(q) \mathbf{J}(q)^{-1}

        The transformation to operational space requires an analytical, rather
        than geometric, Jacobian. ``analytical`` can be one of:

            =============  ========================================
            Value          Rotational representation
            =============  ========================================
            ``'rpy/xyz'``  RPY angular rates in XYZ order (default)
            ``'rpy/zyx'``  RPY angular rates in XYZ order
            ``'eul'``      Euler angular rates in ZYZ order
            ``'exp'``      exponential coordinate rates
            =============  ========================================

        **Trajectory operation**

        If ``q`` is a matrix (m,n), each row is interpretted as a joint state
        vector, and the result is a 3d-matrix (m,n,n) where each plane
        corresponds to the Cartesian inertia for the corresponding
        row of ``q``.

        Parameters
        ----------
        q
            Joint coordinates
        pinv
            use pseudo inverse rather than inverse (Default value = False)
        analytical
            the type of analytical Jacobian to use, default is
            'rpy/xyz'
        representation
            (Default value = "rpy/xyz")
        Ji
            The inverse analytical Jacobian (base-frame)

        Returns
        -------
        inertia_x
            The operational space inertia matrix

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.inertia_x(puma.qz)

        Notes
        -----
        - If the robot is not 6 DOF the ``pinv`` option is set True.
        - ``pinv()`` is around 5x slower than ``inv()``

        .. warning:: Assumes that the operational space has 6 DOF.

        See Also
        --------
        :func:`inertia`

        """

        q = getmatrix(q, (None, self.n))
        if q.shape[1] != 6:
            pinv = True

        if q.shape[0] == 1:
            # single q case
            if Ji is None:
                Ja = self.jacob0_analytical(q[0, :], representation)
                if pinv:
                    Ji = np.linalg.pinv(Ja)
                else:
                    Ji = np.linalg.inv(Ja)
            M = self.inertia(q[0, :])
            return Ji.T @ M @ Ji

        else:
            # trajectory case
            Mt = np.zeros((q.shape[0], 6, 6))

            for k, qk in enumerate(q):
                Ja = self.jacob0_analytical(qk, representation)
                if pinv:
                    Ji = np.linalg.pinv(Ja)
                else:
                    Ji = np.linalg.inv(Ja)
                M = self.inertia(qk)
                Mt[k, :, :] = Ji.T @ M @ Ji

            return Mt

    def coriolis_x(
        self: RobotProto,
        q,
        qd,
        pinv=False,
        representation="rpy/xyz",
        J=None,
        Ji=None,
        Jd=None,
        C=None,
        Mx=None,
    ):
        r"""
        Operational space Coriolis and centripetal term

        ``coriolis_x(q, qd)`` is the Coriolis/centripetal matrix (m,m)
        in operational space for the robot in configuration ``q`` and velocity
        ``qd``, where ``n`` is the number of joints.

        .. math::

            \mathbf{C}_x = \mathbf{J}(q)^{-T} \left(
                \mathbf{C}(q) - \mathbf{M}_x(q) \mathbf{J})(q)
                \right) \mathbf{J}(q)^{-1}

        The product :math:`\mathbf{C} \dot{x}` is the operational space wrench
        due to joint velocity coupling. This matrix is also known as the
        velocity coupling matrix, since it describes the disturbance forces on
        any joint due to velocity of all other joints.

        The transformation to operational space requires an analytical, rather
        than geometric, Jacobian. ``analytical`` can be one of:

            =============  ========================================
            Value          Rotational representation
            =============  ========================================
            ``'rpy/xyz'``  RPY angular rates in XYZ order (default)
            ``'rpy/zyx'``  RPY angular rates in XYZ order
            ``'eul'``      Euler angular rates in ZYZ order
            ``'exp'``      exponential coordinate rates
            =============  ========================================

        **Trajectory operation**

        If ``q`` and `qd` are matrices (m,n), each row is interpretted as a
        joint configuration, and the result (n,n,m) is a 3d-matrix where
        each plane corresponds to a row of ``q`` and ``qd``.

        Parameters
        ----------
        q
            Joint coordinates
        qd
            Joint velocity
        pinv
            use pseudo inverse rather than inverse (Default value = False)
        analytical
            the type of analytical Jacobian to use, default is
            'rpy/xyz'
        representation
             (Default value = "rpy/xyz")
        J

        Ji

        Jd

        C

        Mx


        Returns
        -------
        ndarray(6,6) or ndarray(m,6,6)
            Operational space velocity matrix

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.coriolis_x(puma.qz, 0.5 * np.ones((6,)))

        Notes
        -----
        - Joint viscous friction is also a joint force proportional to
            velocity but it is eliminated in the computation of this value.
        - Computationally slow, involves :math:`n^2/2` invocations of RNE.
        - If the robot is not 6 DOF the ``pinv`` option is set True.
        - ``pinv()`` is around 5x slower than ``inv()``

        .. warning:: Assumes that the operational space has 6 DOF.

        See Also
        --------
        :func:`coriolis`
        :func:`inertia_x`
        :func:`hessian0`

        """

        q = getmatrix(q, (None, self.n))
        qd = getmatrix(qd, (None, self.n))
        n = q.shape[1]
        if n != 6:
            pinv = True

        if q.shape[0] == 1:
            # single q case
            if Ji is None:
                Ja = self.jacob0_analytical(q[0, :], representation)
                if pinv:
                    Ji = np.linalg.pinv(Ja)
                else:
                    Ji = np.linalg.inv(Ja)
            if C is None:
                C = self.coriolis(q[0, :], qd[0, :])
            if Mx is None:
                Mx = self.inertia_x(q[0, :], Ji=Ji)
            if Jd is None:
                Jd = self.jacob0_dot(q[0, :], qd[0, :], J0=Ja)
            return Ji.T @ (C - Mx @ Jd) @ Ji
        else:
            # trajectory case
            Ct = np.zeros((q.shape[0], 6, 6))

            for k, (qk, qdk) in enumerate(zip(q, qd)):

                if Ji is None:
                    Ja = self.jacob0_analytical(q[0, :], representation)
                    if pinv:
                        Ji = np.linalg.pinv(Ja)
                    else:
                        Ji = np.linalg.inv(Ja)

                C = self.coriolis(qk, qdk)
                Mx = self.inertia_x(qk, Ji=Ji)
                Jd = self.jacob0_dot(qk, qdk, J0=J)

                Ct[k, :, :] = Ji.T @ (C - Mx @ Jd) @ Ji

            return Ct

    def gravload_x(
        self: RobotProto,
        q=None,
        gravity=None,
        pinv=False,
        representation="rpy/xyz",
        Ji=None,
    ):
        r"""
        Operational space gravity load

        ``robot.gravload_x(q)`` calculates the gravity wrench for
        the robot in the joint configuration ``q`` and using the default
        gravitational acceleration specified in the robot object.

        ``robot.gravload_x(q, gravity=g)`` as above except the gravitational
        acceleration is explicitly specified as ``g``.

        .. math::

            \mathbf{G}_x = \mathbf{J}(q)^{-T} \mathbf{G}(q)

        The transformation to operational space requires an analytical, rather
        than geometric, Jacobian. ``analytical`` can be one of:

            =============  ========================================
            Value          Rotational representation
            =============  ========================================
            ``'rpy/xyz'``  RPY angular rates in XYZ order (default)
            ``'rpy/zyx'``  RPY angular rates in XYZ order
            ``'eul'``      Euler angular rates in ZYZ order
            ``'exp'``      exponential coordinate rates
            =============  ========================================

        **Trajectory operation**

        If q is a matrix (nxm) each column is interpreted as a joint
        configuration vector, and the result is a matrix (nxm) each column
        being the corresponding joint torques.

        Parameters
        ----------
        q
            Joint coordinates
        gravity
            Gravitational acceleration (Optional, if not supplied will
            use the ``gravity`` attribute of self).
        pinv
            use pseudo inverse rather than inverse (Default value = False)
        analytical
            the type of analytical Jacobian to use, default is
            'rpy/xyz'
        representation :
             (Default value = "rpy/xyz")
        Ji :


        Returns
        -------
        gravload
            The operational space gravity wrench

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.gravload_x(puma.qz)

        Notes
        -----
        - If the robot is not 6 DOF the ``pinv`` option is set True.
        - ``pinv()`` is around 5x slower than ``inv()``

        .. warning:: Assumes that the operational space has 6 DOF.

        See Also
        --------
        :func:`gravload`

        """

        q = getmatrix(q, (None, self.n))
        if q.shape[1] != 6:
            pinv = True

        # if gravity is None:
        #     gravity = self.gravity
        # else:
        #     gravity = getvector(gravity, 3)

        if q.shape[0] == 1:
            # single q case
            if Ji is None:
                Ja = self.jacob0_analytical(q[0, :], representation=representation)
                if pinv:
                    Ji = np.linalg.pinv(Ja)
                else:
                    Ji = np.linalg.inv(Ja)
            G = self.gravload(q[0, :])
            return Ji.T @ G

        else:
            # trajectory case
            taug = np.zeros((q.shape[0], self.n))
            # z = np.zeros(self.n)

            for k, qk in enumerate(q):
                Ja = self.jacob0_analytical(qk, representation=representation)
                G = self.gravload(qk)
                if pinv:
                    Ji = np.linalg.pinv(Ja)
                else:
                    Ji = np.linalg.inv(Ja)

                taug[k, :] = Ji.T @ G

            return taug

    def accel_x(
        self: RobotProto,
        q,
        xd,
        wrench,
        gravity=None,
        pinv=False,
        representation="rpy/xyz",
    ):
        r"""
        Operational space acceleration due to applied wrench

        ``xdd = accel_x(q, qd, wrench)`` is the operational space acceleration
        due to ``wrench`` applied to the end-effector of a robot in joint
        configuration ``q`` and joint velocity ``qd``.

        .. math::

            \ddot{x} = \mathbf{J}(q) \mathbf{M}(q)^{-1} \left(
                \mathbf{J}(q)^T w - \mathbf{C}(q)\dot{q} - \mathbf{g}(q)
                \right)

        **Trajectory operation**

        If `q`, `qd`, torque are matrices (m,n) then ``qdd`` is a matrix (m,n)
        where each row is the acceleration corresponding to the equivalent rows
        of q, qd, wrench.

        Parameters
        ----------
        q
            Joint coordinates
        qd
            Joint velocity
        wrench
            Wrench applied to the end-effector
        gravity
            Gravitational acceleration (Optional, if not supplied will
            use the ``gravity`` attribute of self).
        pinv
            use pseudo inverse rather than inverse
        analytical
            the type of analytical Jacobian to use, default is
            'rpy/xyz'
        xd :
        representation :
            (Default value = "rpy/xyz")

        Returns
        -------
        accel
            Operational space accelerations of the end-effector

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.accel_x(puma.qz, 0.5 * np.ones(6), np.zeros(6))

        Notes
        -----
        - Useful for simulation of manipulator dynamics, in
            conjunction with a numerical integration function.
        - Uses the method 1 of Walker and Orin to compute the forward
            dynamics.
        - Featherstone's method is more efficient for robots with large
            numbers of joints.
        - Joint friction is considered.

        See Also
        --------
        :func:`accel`

        """  # noqa

        q = getmatrix(q, (None, self.n))
        xd = getmatrix(xd, (None, 6))
        w = getmatrix(wrench, (None, 6))
        if q.shape[1] != 6:
            pinv = True

        xdd = np.zeros((q.shape[0], self.n))

        for k, (qk, xdk, wk) in enumerate(zip(q, xd, w)):

            Ja = self.jacob0_analytical(qk, representation=representation)
            if pinv:
                Ji = np.linalg.pinv(Ja)
            else:
                Ji = np.linalg.inv(Ja)

            # Compute current manipulator inertia tensor
            # shortcut from torques resulting from unit
            # acceleration of each joint with zero gravity and zero velocity
            qI = (np.c_[qk] @ np.ones((1, self.n))).T
            qdI = np.zeros((self.n, self.n))
            qddI = np.eye(self.n)
            M = self.rne(qI, qdI, qddI, gravity=[0, 0, 0])

            # Compute gravity and coriolis torque torques resulting from zero
            # acceleration at given velocity & with gravity acting.
            tau_rne = self.rne(qk, Ji @ xdk, np.zeros((1, self.n)), gravity=gravity)

            # solve is faster than inv() which is faster than pinv()
            #   tau_rne = C(q,qd) + G(q)
            #   qdd = M^-1 (tau - C(q,qd) - G(q))
            qdd = np.linalg.solve(M, Ja.T @ wk - tau_rne)

            # xd = Ja qd
            # xdd = Jad qd + Ja qdd
            #
            # Ja = T J
            # Jad = Td J + T Jd
            # assume Td = 0, not sure how valid that is

            # need Jacobian dot
            qdk = Ji @ xdk
            Jd = self.jacob0_dot(qk, qdk, J0=Ja)

            xdd[k, :] = T @ (Jd @ qdk + J @ qdd)

        if q.shape[0] == 1:
            return xdd[0, :]
        else:
            return xdd

    def itorque(self: RobotProto, q, qdd):
        r"""
        Inertia torque

        ``itorque(q, qdd)`` is the inertia force/torque vector (n) at
        the specified joint configuration q (n) and acceleration qdd (n), and
        ``n`` is the number of robot joints.
        It is :math:`\mathbf{I}(q) \ddot{q}`.

        **Trajectory operation**

        If ``q`` and ``qdd`` are matrices (m,n), each row is interpretted as a
        joint configuration, and the result is a matrix (m,n) where each row is
        the corresponding joint torques.

        Parameters
        ----------
        q
            Joint coordinates
        qdd
            Joint acceleration

        Returns
        -------
        itorque
            The inertia torque vector

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.itorque(puma.qz, 0.5 * np.ones((6,)))

        Notes
        -----
        - If the robot model contains non-zero motor inertia then this
            will be included in the result.

        See Also
        --------
        :func:`inertia`

        """

        q = getmatrix(q, (None, self.n))
        qdd = getmatrix(qdd, (None, self.n))
        if q.shape[0] != qdd.shape[0]:
            raise ValueError("q and qdd must have the same number of rows")

        taui = np.zeros((q.shape[0], self.n))

        for k, (qk, qddk) in enumerate(zip(q, qdd)):
            taui[k, :] = self.rne(qk, np.zeros(self.n), qddk, gravity=[0, 0, 0])

        if q.shape[0] == 1:
            return taui[0, :]
        else:
            return taui

    def paycap(
        self: RobotProto,
        w: NDArray,
        tauR: NDArray,
        frame: int = 1,
        q: Union[ArrayLike, None] = None,
    ):
        """
        Static payload capacity of a robot

        ``wmax, joint = paycap(q, w, f, tauR)`` returns the maximum permissible
        payload wrench ``wmax`` (6) applied at the end-effector, and the index
        of the joint (zero indexed) which hits its force/torque limit at that
        wrench. ``q`` (n) is the manipulator pose, ``w`` the payload wrench
        (6), ``f`` the wrench reference frame and tauR (nx2) is a matrix of
        joint forces/torques (first col is maximum, second col minimum).

        **Trajectory operation:**

        In the case q is nxm then wmax is Mx6 and J is Mx1 where the rows are
        the results at the pose given by corresponding row of q.

        Parameters
        ----------
        w
            The payload wrench
        tauR
            Joint torque matrix minimum and maximums
        frame
            The frame in which to torques are expressed in when J
            is not supplied. 'base' means base frame of the robot, 'ee' means
            end-effector frame
        q
            Joint coordinates

        Returns
        -------
        ndarray(6)
            The maximum permissible payload wrench

        Notes
        -----
        - Wrench vector and Jacobian must be from the same reference frame
        - Tool transforms are taken into consideration for frame=1.

        """

        # TODO rewrite
        trajn = 1

        if q is None:
            q = self.q
        else:
            q = np.array(q)

        try:
            q = np.array(getvector(q, self.n, "row"))
            w = np.array(getvector(w, 6, "row"))
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

            tauP = self.pay(w[i, :] / np.linalg.norm(w[i, :]), q=q[i, :], frame=frame)

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

    def perturb(self: RobotProto, p=0.1):
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

        Parameters
        ----------
        p
            The percent (+/-) to be perturbed. Default 10%

        Returns
        -------
        DHRobot
            A copy of the robot with dynamic parameters perturbed

        """

        r2 = self.copy()
        r2.name = "P/" + self.name

        for i in range(self.n):
            s = (2 * np.random.random() - 1) * p + 1
            r2.links[i].m = r2.links[i].m * s

            s = (2 * np.random.random() - 1) * p + 1
            r2.links[i].I = r2.links[i].I * s  # noqa

        return r2


def _printProgressBar(
    fraction, prefix="", suffix="", decimals=1, length=50, fill="█", printEnd="\r"
):

    percent = ("{0:." + str(decimals) + "f}").format(fraction * 100)
    filledLength = int(length * fraction)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)


if __name__ == "__main__":  # pragma nocover

    import roboticstoolbox as rtb

    puma = rtb.models.DH.Puma560()
