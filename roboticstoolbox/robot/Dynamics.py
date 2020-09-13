"""
Rigid-body dynamics functionality of the Toolbox.

Requires access to:

    * ``links`` list of ``Link`` objects, atttribute
    * ``rne()`` the inverse dynamics method

so must be subclassed by ``SerialLink`` class.

:todo: perhaps these should be abstract properties, methods of this calss
"""

import numpy as np
import roboticstoolbox as rp
from spatialmath.base.argcheck import \
    getvector, verifymatrix


class Dynamics:

    def accel(self, q, qd, torque, ):
        """
        Compute acceleration due to applied torque

        :param q: The joint angles/configuration of the robot
        :type q: float ndarray(n)
        :param qd: The joint velocities of the robot
        :type qd: float ndarray(n)
        :param torque: The joint torques of the robot
        :type torque: float ndarray(n)

        ``qdd = accel(q, qd, torque)`` calculates a vector (n) of joint
        accelerations that result from applying the actuator force/torque (n)
        to the manipulator in state q (n) and qd (n), and n is the number of
        robot joints.

        If q, qd, torque are matrices (nxk) then qdd is a matrix (nxk) where
        each row is the acceleration corresponding to the equivalent cols of
        q, qd, torque.

        :return qdd: The joint accelerations of the robot
        :rtype qdd: float ndarray(n)

        :notes:
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
            q = getvector(q, self.n, 'col')
            qd = getvector(qd, self.n, 'col')
            torque = getvector(torque, self.n, 'col')
        except ValueError:
            trajn = q.shape[1]
            verifymatrix(q, (self.n, trajn))
            verifymatrix(qd, (self.n, trajn))
            verifymatrix(torque, (self.n, trajn))

        qdd = np.zeros((self.n, trajn))

        for i in range(trajn):
            # Compute current manipulator inertia torques resulting from unit
            # acceleration of each joint with no gravity.
            qI = np.c_[q[:, i]] @ np.ones((1, self.n))
            qdI = np.zeros((self.n, self.n))
            qddI = np.eye(self.n)

            m = self.rne(qI, qdI, qddI, grav=[0, 0, 0])

            # Compute gravity and coriolis torque torques resulting from zero
            # acceleration at given velocity & with gravity acting.
            tau = self.rne(q[:, i], qd[:, i], np.zeros((1, self.n)))

            inter = np.expand_dims((torque[:, i] - tau), axis=1)

            qdd[:, i] = (np.linalg.inv(m) @ inter).flatten()

        if trajn == 1:
            return qdd[:, 0]
        else:
            return qdd

    def nofriction(self, coulomb=True, viscous=False):
        """
        NFrobot = nofriction(coulomb, viscous) copies the robot and returns
        a robot with the same parameters except, the Coulomb and/or viscous
        friction parameter set to zero

        NFrobot = nofriction(coulomb, viscous) copies the robot and returns
        a robot with the same parameters except the Coulomb friction parameter
        is set to zero

        :param coulomb: if True, will set the coulomb friction to 0
        :type coulomb: bool

        :return: A copy of the robot with dynamic parameters perturbed
        :rtype: SerialLink

        """

        L = []

        for i in range(self.n):
            L.append(self.links[i].nofriction(coulomb, viscous))

        return rp.SerialLink(
            L,
            name='NF' + self.name,
            manufacturer=self.manuf,
            base=self.base,
            tool=self.tool,
            gravity=self.gravity)

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
        :type W: float ndarray(6)
        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param J: The manipulator Jacobian (Optional, if not supplied will
            use the q value).
        :type J: float ndarray(6,n)
        :param frame: The frame in which to torques are expressed in when J
            is not supplied. 0 means base frame of the robot, 1 means end-
            effector frame
        :type frame: int

        :return tau: The joint forces/torques due to w
        :rtype tau: float ndarray(n)

        :notes:
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
            trajn = W.shape[1]
            verifymatrix(W, (6, trajn))

        if trajn:
            # A trajectory
            if J is not None:
                # Jacobian supplied
                verifymatrix(J, (6, self.n, trajn))
            else:
                # Use q instead
                verifymatrix(q, (self.n, trajn))
                J = np.zeros((6, self.n, trajn))
                for i in range(trajn):
                    if frame:
                        J[:, :, i] = self.jacobe(q[:, i])
                    else:
                        J[:, :, i] = self.jacob0(q[:, i])
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
            tau = np.zeros((self.n, trajn))

            for i in range(trajn):
                tau[:, i] = -J[:, :, i].T @ W[:, i]

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
        :type p: float ndarray(3,1)

        """

        p = getvector(p, 3, out='col')
        lastlink = self.links[self.n - 1]

        lastlink.m = m
        lastlink.r = p

    def jointdynamics(self, q, qd=None):
        """
        Transfer function of joint actuator

        :param q: The joint angles/configuration of the robot
        :type q: float ndarray(n)
        :param qd: The joint velocities of the robot
        :type qd: float ndarray(n)
        :return: transfer function denominators
        :rtype: list of 2-tuples

        - ``tf = jointdynamics(qd, q)`` calculates a vector of n continuous-time
          transfer functions that represent the transfer function
          1/(Js+B) for each joint based on the dynamic parameters of the robot
          and the configuration q (n). n is the number of robot joints.

        - ``tf = jointdynamics(q, qd)`` as above but include the linearized effects
          of Coulomb friction when operating at joint velocity QD (1xN).
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

    def friction(self, qd):
        """
        tau = friction(qd) calculates the vector of joint friction
        forces/torques for the robot moving with joint velocities qd.

        The friction model includes:

        - Viscous friction which is a linear function of velocity.
        - Coulomb friction which is proportional to sign(qd).

        :param qd: The joint velocities of the robot
        :type qd: float ndarray(n)

        :return: The joint friction forces.torques for the robot
        :rtype: float ndarray(n,)

        :notes:
            - The friction value should be added to the motor output torque,
              it has a negative value when qd>0.
            - The returned friction value is referred to the output of the
              gearbox.
            - The friction parameters in the Link object are referred to the
              motor.
            - Motor viscous friction is scaled up by G^2.
            - Motor Coulomb friction is scaled up by G.
            - The appropriate Coulomb friction value to use in the
              non-symmetric case depends on the sign of the joint velocity,
              not the motor velocity.
            - The absolute value of the gear ratio is used. Negative gear
              ratios are tricky: the Puma560 has negative gear ratio for
              joints 1 and 3.

        """

        qd = getvector(qd, self.n)
        tau = np.zeros(self.n)

        for i in range(self.n):
            tau[i] = self.links[i].friction(qd[i])

        return tau

    def cinertia(self, q=None):
        """
        M = cinertia(q) is the nxn Cartesian (operational space) inertia
        matrix which relates Cartesian force/torque to Cartesian
        acceleration at the joint configuration q.

        M = cinertia() as above except uses the stored q value of the robot
        object.

        If q is a matrix (nxk), each row is interpretted as a joint state
        vector, and the result is a 3d-matrix (nxnxk) where each plane
        corresponds to the cinertia for the corresponding row of q.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :return M: The inertia matrix
        :rtype M: float ndarray(n,n)

        """

        trajn = 1

        if q is None:
            q = self.q

        try:
            q = getvector(q, self.n, 'col')
        except ValueError:
            trajn = q.shape[1]
            verifymatrix(q, (self.n, trajn))

        Mt = np.zeros((self.n, self.n, trajn))

        for i in range(trajn):
            J = self.jacob0(q[:, i])
            Ji = np.linalg.pinv(J)
            M = self.inertia(q[:, i])
            Mt[:, :, i] = Ji.T @ M @ Ji

        if trajn == 1:
            return Mt[:, :, 0]
        else:
            return Mt

    def inertia(self, q=None):
        """
        SerialLink.INERTIA Manipulator inertia matrix

        I = inertia(q) is the symmetric joint inertia matrix (nxn) which
        relates joint torque to joint acceleration for the robot at joint
        configuration q.

        I = inertia() as above except uses the stored q value of the robot
        object.

        If q is a matrix (nxk), each row is interpretted as a joint state
        vector, and the result is a 3d-matrix (nxnxk) where each plane
        corresponds to the inertia for the corresponding row of q.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :return I: The inertia matrix
        :rtype I: float ndarray(n,n)

        :notes:
            - The diagonal elements I(J,J) are the inertia seen by joint
              actuator J.
            - The off-diagonal elements I(J,K) are coupling inertias that
              relate acceleration on joint J to force/torque on joint K.
            - The diagonal terms include the motor inertia reflected through
              the gear ratio.

        """

        trajn = 1

        try:
            q = getvector(q, self.n, 'col')
        except ValueError:
            trajn = q.shape[1]
            verifymatrix(q, (self.n, trajn))

        In = np.zeros((self.n, self.n, trajn))

        for i in range(trajn):
            In[:, :, i] = self.rne(
                np.c_[q[:, i]] @ np.ones((1, self.n)),
                np.zeros((self.n, self.n)),
                np.eye(self.n),
                grav=[0, 0, 0])

        if trajn == 1:
            return In[:, :, 0]
        else:
            return In

    def coriolis(self, q, qd):
        """
        Coriolis and centripetal term

        ``C = coriolis(q, qd)`` calculates the Coriolis/centripetal matrix (nxn)
        for the robot in configuration q and velocity qd, where n is the
        number of joints. The product c*qd is the vector of joint
        force/torque due to velocity coupling. The diagonal elements are due
        to centripetal effects and the off-diagonal elements are due to
        Coriolis effects. This matrix is also known as the velocity coupling
        matrix, since it describes the disturbance forces on any joint due to
        velocity of all other joints.

        If q and qd are matrices (nxk), each row is interpretted as a
        joint state vector, and the result (nxnxk) is a 3d-matrix where
        each plane corresponds to a row of q and qd.

        :notes:
            - Joint viscous friction is also a joint force proportional to
              velocity but it is eliminated in the computation of this value.
            - Computationally slow, involves n^2/2 invocations of RNE.

        """

        trajn = 1

        try:
            q = getvector(q, self.n, 'col')
            qd = getvector(qd, self.n, 'col')
        except ValueError:
            trajn = q.shape[1]
            verifymatrix(q, (self.n, trajn))
            verifymatrix(qd, (self.n, trajn))

        r1 = self.nofriction(True, True)

        C = np.zeros((self.n, self.n, trajn))
        Csq = np.zeros((self.n, self.n, trajn))

        # Find the torques that depend on a single finite joint speed,
        # these are due to the squared (centripetal) terms
        # set QD = [1 0 0 ...] then resulting torque is due to qd_1^2
        for j in range(trajn):
            for i in range(self.n):
                QD = np.zeros(self.n)
                QD[i] = 1
                tau = r1.rne(
                    q[:, j], QD, np.zeros(self.n), grav=[0, 0, 0])
                Csq[:, i, j] = Csq[:, i, j] + tau

        # Find the torques that depend on a pair of finite joint speeds,
        # these are due to the product (Coridolis) terms
        # set QD = [1 1 0 ...] then resulting torque is due to
        # qd_1 qd_2 + qd_1^2 + qd_2^2
        for k in range(trajn):
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    # Find a product term  qd_i * qd_j
                    QD = np.zeros(self.n)
                    QD[i] = 1
                    QD[j] = 1
                    tau = r1.rne(q[:, k], QD, np.zeros(self.n), grav=[0, 0, 0])

                    C[:, j, k] = C[:, j, k] + \
                        (tau - Csq[:, j, k] - Csq[:, i, k]) * qd[i, k] / 2

                    C[:, i, k] = C[:, i, k] + \
                        (tau - Csq[:, j, k] - Csq[:, i, k]) * qd[j, k] / 2

            C[:, :, k] = C[:, :, k] + Csq[:, :, k] @ np.diag(qd[:, k])

        if trajn == 1:
            return C[:, :, 0]
        else:
            return C

    def itorque(self, q, qdd):
        """
        Inertia torque

        :param qdd: The joint accelerations of the robot
        :type qdd: float ndarray(n)
        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :return taui: The inertia torque vector
        :rtype taui: float ndarray(n)

        ``tauI = itorque(q, qdd)`` is the inertia force/torque vector (n) at the
        specified joint configuration q (n) and acceleration qdd (n), and n
        is the number of robot joints. taui = inertia(q) * qdd.

        If q and qdd are matrices (nxk), each row is interpretted as a joint
        state vector, and the result is a matrix (nxk) where each row is the
        corresponding joint torques.

        :notes:
            - If the robot model contains non-zero motor inertia then this
              will included in the result.

        """

        trajn = 1
        
        try:
            q = getvector(q, self.n, 'col')
            qdd = getvector(qdd, self.n, 'col')
        except ValueError:
            trajn = q.shape[1]
            verifymatrix(q, (self.n, trajn))
            verifymatrix(qdd, (self.n, trajn))

        taui = np.zeros((self.n, trajn))

        for i in range(trajn):
            taui[:, i] = self.rne(
                q[:, i], np.zeros(self.n), qdd[:, i], grav=[0, 0, 0])

        if trajn == 1:
            return taui[:, 0]
        else:
            return taui

    # NOT CONVINCED WE NEED THIS, AND IT'S ORPHAN CODE
    # def gravjac(self, q, grav=None):
    #     """
    #     Compute gravity load and Jacobian

    #     :param q: The joint angles/configuration of the robot 
    #     :type q: float ndarray(n)
    #     :param grav: The gravity vector (Optional, if not supplied will
    #         use the stored gravity values).
    #     :type grav: float ndarray(3,)

    #     :return tau: The generalised joint force/torques due to gravity
    #     :rtype tau: float ndarray(n,)

    #     ``tauB = gravjac(q, grav)`` calculates the generalised joint force/torques
    #     due to gravity and the Jacobian

    #     Trajectory operation:
    #     If q is nxm where n is the number of robot joints then a
    #     trajectory is assumed where each row of q corresponds to a robot
    #     configuration. tau (nxm) is the generalised joint torque, each row
    #     corresponding to an input pose, and jacob0 (6xnxm) where each
    #     plane is a Jacobian corresponding to an input pose.

    #     :notes:
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


    def gravload(self, q=None, grav=None):
        """
        Compute gravity load

        :param q: The joint angles/configuration of the robot
        :type q: float ndarray(n)
        :param grav: The gravity vector (Optional, if not supplied will
            use the stored gravity values).
        :type grav: float ndarray(3)

        :return taug: The generalised joint force/torques due to gravity
        :rtype taug: float ndarray(n)

        ``taug = gravload(q)`` calculates the joint gravity loading (n) for
        the robot in the joint configuration ``q`` and using the default
        gravitational acceleration specified in the SerialLink object.

        ``taug = gravload(q, grav)`` as above except the gravitational acceleration
        is explicitly specified as `grav``.

        If q is a matrix (nxm) each column is interpreted as a joint
        configuration vector, and the result is a matrix (nxm) each column
        being the corresponding joint torques.

        """

        trajn = 1

        if q is None:
            q = self.q

        if grav is None:
            grav = getvector(np.copy(self.gravity), 3, 'col')

        try:
            q = getvector(q, self.n, 'col')
            grav = getvector(grav, 3, 'col')
        except ValueError:
            trajn = q.shape[1]
            verifymatrix(q, (self.n, trajn))

        if grav.shape[1] < trajn:
            grav = grav @ np.ones((1, trajn))
        verifymatrix(grav, (3, trajn))

        taug = np.zeros((self.n, trajn))

        for i in range(trajn):
            taug[:, i] = self.rne(
                 q[:, i], np.zeros(self.n), np.zeros(self.n),grav[:, i])

        if trajn == 1:
            return taug[:, 0]
        else:
            return taug

    def paycap(self, w, tauR, frame=1, q=None):
        """
        Static payload capacity of a robot

        :param w: The payload wrench
        :type w: float ndarray(n)
        :param tauR: Joint torque matrix minimum and maximums
        :type tauR: float ndarray(n,2)
        :param frame: The frame in which to torques are expressed in when J
            is not supplied. 'base' means base frame of the robot, 'ee' means end-
            effector frame
        :type frame: str
        :param q: The joint angles/configuration of the robot.
        :type q: float ndarray(n)

        :return wmax: The maximum permissible payload wrench
        :rtype wmax: float ndarray(6)
        :return joint: The joint index (zero indexed) which hits its
            force/torque limit
        :rtype joint: int

        ``wmax, joint = paycap(q, w, f, tauR)`` returns the maximum permissible
        payload wrench ``wmax`` (6) applied at the end-effector, and the index
        of the joint (zero indexed) which hits its force/torque limit at that
        wrench. ``q`` (n) is the manipulator pose, ``w`` the payload wrench (6), ``f`` the
        wrench reference frame and tauR (nx2) is a matrix of joint
        forces/torques (first col is maximum, second col minimum).

        Trajectory operation:
        In the case q is nxm then wmax is Mx6 and J is Mx1 where the rows are
        the results at the pose given by corresponding row of q.


        :notes:
            - Wrench vector and Jacobian must be from the same reference frame
            - Tool transforms are taken into consideration for frame=1.
        """

        trajn = 1

        if q is None:
            q = self.q

        try:
            q = getvector(q, self.n, 'col')
            w = getvector(w, 6, 'col')
        except ValueError:
            trajn = q.shape[1]
            verifymatrix(q, (self.n, trajn))
            verifymatrix(w, (6, trajn))

        verifymatrix(tauR, (self.n, 2))

        wmax = np.zeros((6, trajn))
        joint = np.zeros(trajn, dtype=np.int)

        for i in range(trajn):
            tauB = self.gravload(q[:, i])

            # tauP = self.rne(
            #     np.zeros(self.n), np.zeros(self.n),
            #     q, grav=[0, 0, 0], fext=w/np.linalg.norm(w))

            tauP = self.pay(
                w[:, i]/np.linalg.norm(w[:, i]), q=q[:, i], frame=frame)

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

            wmax[:, i] = WM
            joint[i] = np.argmin(WM)

        if trajn == 1:
            return wmax[:, 0], joint[0]
        else:
            return wmax, joint

    def perterb(self, p=0.1):
        '''
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
        :rtype: SerialLink

        '''

        r2 = self._copy()
        r2.name = 'P/' + self.name

        for i in range(self.n):
            s = (2 * np.random.random() - 1) * p + 1
            r2.links[i].m = r2.links[i].m * s

            s = (2 * np.random.random() - 1) * p + 1
            r2.links[i].I = r2.links[i].I * s    # noqa

        return r2
