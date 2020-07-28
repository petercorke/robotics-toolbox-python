#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

import numpy as np
import spatialmath as sp
from spatialmath import SE3
from spatialmath.base.argcheck import getvector, verifymatrix
# from roboticstoolbox.robot.ET import ET


class ETS(object):
    """
    The Elementary Transform Sequence (ETS). A superclass which represents the
    kinematics of a serial-link manipulator

    :param et_list: List of elementary transforms which represent the robot
        kinematics
    :type et_list: list of etb.robot.et
    :param q_idx: List of indexes within the ets_list which correspond to
        joints
    :type q_idx: list of int
    :param name: Name of the robot
    :type name: str, optional
    :param manufacturer: Manufacturer of the robot
    :type manufacturer: str, optional
    :param base: Location of the base is the world frame
    :type base: SE3, optional
    :param tool: Offset of the flange of the robot to the end-effector
    :type tool: SE3, optional

    :references: 
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke
    """

    def __init__(
            self,
            et_list,
            q_idx,
            name='noname',
            manufacturer='',
            base=SE3(),
            tool=SE3(),
            gravity=np.array([0, 0, 9.81])):

        # self._name = name
        # self._manuf = manufacturer
        self._ets = et_list
        self._q_idx = q_idx
        # self._base = base
        # self._tool = tool
        # self._T = np.eye(4)

        super(ETS, self).__init__()

        # Number of transforms in the ETS
        self._M = len(self._ets)

        # Number of joints in the robot
        self._n = len(self._q_idx)

        # Current joint angles of the robot
        self._q = np.zeros((self._n,))

        self.name = name
        self.manuf = manufacturer
        self.base = base
        self.tool = tool
        self.gravity = gravity

        # Current joint angles of the robot
        self.q = np.zeros(self.n)

    # @classmethod
    # def dh_to_ets(cls, robot):
    #     """
    #     Converts a robot modelled with standard or modified DH parameters to an
    #     ETS representation

    #     :param robot: The robot model to be converted
    #     :type robot: SerialLink
    #     :return: List of returned :class:`bluepy.btle.Characteristic` objects
    #     :rtype: ets class
    #     """
    #     ets = []
    #     q_idx = []
    #     M = 0

    #     for j in range(robot.n):
    #         L = robot.links[j]

    #         # Method for modified DH parameters
    #         if robot.mdh:

    #             # Append Tx(a)
    #             if L.a != 0:
    #                 ets.append(ET.Ttx(L.a))
    #                 M += 1

    #             # Append Rx(alpha)
    #             if L.alpha != 0:
    #                 ets.append(ET.TRx(L.alpha))
    #                 M += 1

    #             if L.is_revolute:
    #                 # Append Tz(d)
    #                 if L.d != 0:
    #                     ets.append(ET.Ttz(L.d))
    #                     M += 1

    #                 # Append Rz(q)
    #                 ets.append(ET.TRz(joint=j+1))
    #                 q_idx.append(M)
    #                 M += 1

    #             else:
    #                 # Append Tz(q)
    #                 ets.append(ET.Ttz(joint=j+1))
    #                 q_idx.append(M)
    #                 M += 1

    #                 # Append Rz(theta)
    #                 if L.theta != 0:
    #                     ets.append(ET.TRz(L.alpha))
    #                     M += 1

    #     return cls(
    #         ets,
    #         q_idx,
    #         robot.name,
    #         robot.manuf,
    #         robot.base,
    #         robot.tool)

    @property
    def q(self):
        return self._q

    @property
    def ets(self):
        return self._ets

    @property
    def name(self):
        return self._name

    @property
    def manuf(self):
        return self._manuf

    @property
    def base(self):
        return self._base

    @property
    def tool(self):
        return self._tool

    @property
    def n(self):
        return self._n

    @property
    def M(self):
        return self._M

    @property
    def ets(self):
        return self._ets

    @property
    def q_idx(self):
        return self._q_idx

    @property
    def gravity(self):
        return self._gravity

    @name.setter
    def name(self, name_new):
        self._name = name_new

    @manuf.setter
    def manuf(self, manuf_new):
        self._manuf = manuf_new

    @gravity.setter
    def gravity(self, gravity_new):
        self._gravity = getvector(gravity_new, 3, 'col')

    @q.setter
    def q(self, q_new):
        q_new = getvector(q_new, self.n)
        self._q = q_new

    @base.setter
    def base(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._base = T

    @tool.setter
    def tool(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._tool = T

    def fkine(self, q=None):
        '''
        Evaluates the forward kinematics of a robot based on its ETS and
        joint angles q.

        :param q: The joint coordinates of the robot
        :type q: float np.ndarray(n,)
        :return: The transformation matrix representing the pose of the
            end-effector
        :rtype: float np.ndarray(4,4)

        References: Kinematic Derivatives using the Elementary Transform
            Sequence, J. Haviland and P. Corke
        '''

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        j = 0
        trans = SE3()

        for i in range(self.M):
            if self.ets[i]._type == 1:
                T = self.ets[i].T(q[j])
                j += 1
            else:
                T = self.ets[i].T()

            trans = trans * T

        trans = trans * self.tool

        return trans

    def jacob0(self, q=None):
        """
        The manipulator Jacobian matrix maps joint velocity to end-effector
        spatial velocity, expressed in the world-coordinate frame.

        :param q: The joint coordinates of the robot
        :type q: float np.ndarray(n,)
        :return: The manipulator Jacobian in 0 frame
        :rtype: float np.ndarray(6,n)

        References: Kinematic Derivatives using the Elementary Transform
            Sequence, J. Haviland and P. Corke
        """

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        T = self.fkine(q)
        U = np.eye(4)
        j = 0
        J = np.zeros((6, self.n))

        for i in range(self.M):

            if i != self.q_idx[j]:
                U = U @ self.ets[i].T()
            else:
                if self.ets[i]._axis == 'Rz':
                    U = U @ self.ets[i].T(q[j])
                    Tu = np.linalg.inv(U) @ T

                    n = U[:3, 0]
                    o = U[:3, 1]
                    a = U[:3, 2]
                    y = Tu[1, 3]
                    x = Tu[0, 3]

                    J[:3, j] = (o * x) - (n * y)
                    J[3:, j] = a

                    j += 1
                elif self.ets[i]._axis == 'tx':
                    U = U @ self.ets[i].T(q[j])
                    n = U[:3, 0]

                    J[:3, j] = n
                    J[3:, j] = np.array([0, 0, 0])

                    j += 1
                elif self.ets[i]._axis == 'ty':
                    U = U @ self.ets[i].T(q[j])
                    o = U[:3, 1]

                    J[:3, j] = o
                    J[3:, j] = np.array([0, 0, 0])

                    j += 1
                elif self.ets[i]._axis == 'tz':
                    U = U @ self.ets[i].T(q[j])
                    a = U[:3, 2]

                    J[:3, j] = a
                    J[3:, j] = np.array([0, 0, 0])

                    j += 1

        return J

    def hessian0(self, q=None, J0=None):
        """
        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the world-coordinate frame. This
        function calulcates this based on the ETS of the robot. One of J0 or q
        is required. Supply J0 if already calculated to save computation time

        :param q: The joint coordinates of the robot
        :type q: float np.ndarray(n,)
        :param J0: The manipulator Jacobian in the 0 frame
        :type J0: float np.ndarray(6,n)
        :return: The manipulator Hessian in 0 frame
        :rtype: float np.ndarray(6,n,n)

        References: Kinematic Derivatives using the Elementary Transform
            Sequence, J. Haviland and P. Corke
        """

        if J0 is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J0 = self.jacob0(q)
        else:
            verifymatrix(J0, (6, self.n))

        H = np.zeros((6, self.n, self.n))

        for j in range(self.n):
            for i in range(j, self.n):

                H[:3, i, j] = np.cross(J0[3:, j], J0[:3, i])
                H[3:, i, j] = np.cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[:3, j, i] = H[:3, i, j]

        return H

    def manipulability(self, q=None, J=None):
        """
        Calculates the manipulability index (scalar) robot at the joint
        configuration q. It indicates dexterity, that is, how isotropic the
        robot's motion is with respect to the 6 degrees of Cartesian motion.
        The measure is high when the manipulator is capable of equal motion
        in all directions and low when the manipulator is close to a
        singularity. One of J or q is required. Supply J if already
        calculated to save computation time

        :param q: The joint coordinates of the robot
        :type q: float np.ndarray(n,)
        :param J: The manipulator Jacobian in any frame
        :type J: float np.ndarray(6,n)
        :return: The manipulability index
        :rtype: float

        References: Analysis and control of robot manipulators with redundancy,
        T. Yoshikawa,
        Robotics Research: The First International Symposium (M. Brady and
        R. Paul, eds.), pp. 735-747, The MIT press, 1984.
        """

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q)
        else:
            verifymatrix(J, (6, self.n))

        return np.sqrt(np.linalg.det(J @ np.transpose(J)))

    def jacobm(self, q=None, J=None, H=None):
        """
        Calculates the manipulability Jacobian. This measure relates the rate
        of change of the manipulability to the joint velocities of the robot.
        One of J or q is required. Supply J and H if already calculated to
        save computation time

        :param q: The joint coordinates of the robot
        :type q: float np.ndarray(n,)
        :param J: The manipulator Jacobian in any frame
        :type J: float np.ndarray(6,n)
        :param H: The manipulator Hessian in any frame
        :type H: float np.ndarray(6,n,n)
        :return: The manipulability Jacobian
        :rtype: float np.ndarray(n,1)

        References: Maximising Manipulability in Resolved-Rate Motion Control,
            J. Haviland and P. Corke
        """

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q)
        else:
            verifymatrix(J, (6, self.n))

        if H is None:
            H = self.hessian0(J0=J)
        else:
            verifymatrix(H, (6, self.n, self.n))

        manipulability = self.manipulability(J=J)
        b = np.linalg.inv(J @ np.transpose(J))
        Jm = np.zeros((self.n, 1))

        for i in range(self.n):
            c = J @ np.transpose(H[:, :, i])
            Jm[i, 0] = manipulability * \
                np.transpose(c.flatten('F')) @ b.flatten('F')

        return Jm

    def __str__(self):
        """
        Pretty prints the ETS Model of the robot. Will output angles in degrees

        :return: Pretty print of the robot model
        :rtype: str
        """
        axes = ''

        for i in range(self._n):
            axes += self.ets[self.q_idx[i]].axis

        rpy = sp.base.tr2rpy(self.tool, unit='deg')

        for i in range(3):
            if rpy[i] == 0:
                rpy[i] = 0

        model = '\n%s (%s): %d axis, %s, ETS\n'\
            'Elementary Transform Sequence:\n'\
            '%s\n'\
            'tool:  t = (%g, %g, %g),  RPY/xyz = (%g, %g, %g) deg' % (
                self.name, self.manuf, self.n, axes,
                self.ets,
                self.tool[0, 3], self.tool[1, 3],
                self.tool[2, 3], rpy[0], rpy[1], rpy[2]
            )

        return model

    """
    The spatial velocity Jacobian which relates the velocity in base
    frame to velocity in the end-effector frame.

    Parameters
    ----------
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot
    Returns
    -------
    J : float np.ndarray(6,n)
        The velocity Jacobian in ee frame
    Examples
    --------
    >>> J = panda.jacobev(np.array([1,1,1,1,1,1,1]))
    >>> J = panda.Jev

    See Also
    --------
    ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame
    ropy.robot.m : Calculates the manipulability index of the robot
    ropy.robot.Jm : Calculates the manipiulability Jacobian
    ropy.robot.fkine : Calculates the forward kinematics of a robot
    """
    def jacobev(self, q):
        r = self.fkine(q)[0:3, 0:3]
        r = np.linalg.inv(r)

        Jv = np.zeros((6, 6))
        Jv[:3, :3] = r
        Jv[3:, 3:] = r

        return Jv

    def jacobe(self, q=None):

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        J0 = self.jacob0(q)
        Je = self.jacobev(q) @ J0
        return Je

    def jacob0v(self, q):
        r = self.fkine(q)[0:3, 0:3]

        Jv = np.zeros((6, 6))
        Jv[:3, :3] = r
        Jv[3:, 3:] = r

        return Jv
