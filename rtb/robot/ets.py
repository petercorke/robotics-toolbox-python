#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

import numpy as np


class ets(object):
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
    :type base: float np.ndarray(4,4), optional
    :param tool: Offset of the flange of the robot to the end-effector
    :type tool: float np.ndarray(4,4), optional

    References: Kinematic Derivatives using the Elementary Transform Sequence,
        J. Haviland and P. Corke
    """

    def __init__(
            self,
            et_list,
            q_idx,
            name='noname',
            manufacturer='',
            base=np.eye(4, 4),
            tool=np.eye(4, 4)
            ):

        self._name = name
        self._manuf = manufacturer
        self._ets = et_list
        self._q_idx = q_idx
        self._base = base
        self._tool = tool
        self._T = np.eye(4)

        super(ets, self).__init__()

        # Number of transforms in the ETS
        self._M = len(self._ets)

        # Number of joints in the robot
        self._n = len(self._q_idx)

    @classmethod
    def dh_to_ets(cls, robot):
        """
        Converts a robot modelled with standard or modified DH parameters to an
        ETS representation

        :param robot: The robot model to be converted
        :type robot: SerialLink
        :return: List of returned :class:`bluepy.btle.Characteristic` objects
        :rtype: ets class
        """
        ets = []
        q_idx = []
        M = 0

        for j in range(robot.n):
            L = robot.links[j]

            # Method for modified DH parameters
            if robot.mdh:

                # Append Tx(a)
                if L.a != 0:
                    ets.append(et(et.Ttx, L.a))
                    M += 1

                # Append Rx(alpha)
                if L.alpha != 0:
                    ets.append(et(et.TRx, L.alpha))
                    M += 1

                if L.is_revolute:
                    # Append Tz(d)
                    if L.d != 0:
                        ets.append(et(et.Ttz, L.d))
                        M += 1

                    # Append Rz(q)
                    ets.append(et(et.TRz, i=j+1))
                    q_idx.append(M)
                    M += 1

                else:
                    # Append Tz(q)
                    ets.append(et(et.Ttz, i=j+1))
                    q_idx.append(M)
                    M += 1

                    # Append Rz(theta)
                    if L.theta != 0:
                        ets.append(et(et.TRz, L.alpha))
                        M += 1

        return cls(
            ets,
            q_idx,
            robot.name,
            robot.manuf,
            robot.base,
            robot.tool)

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

    def fkine(self, q):
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

        # if not isinstance(q, np.ndarray):
        #     raise TypeError('q array must be a numpy ndarray.')
        # if q.shape != (self._n,):
        #     raise ValueError('q must be a 1 dim (n,) array')

        j = 0
        trans = np.eye(4)

        for i in range(self.M):
            if self._ets[i]._type == 1:
                T = self._ets[i].T(q[j])
                j += 1
            else:
                T = self._ets[i].T()

            trans = trans @ T

        return trans

    def __str__(self):
        """
        Pretty prints the ETS Model of the robot. Will output angles in degrees

        :return: Pretty print of the robot model
        :rtype: str
        """
        axes = ''

        for i in range(self._n):
            axes += self.ets[self.q_idx[i]].axis

        model = '\n%s (%s): %d axis, %s, ETS\n'\
            'Elementary Transform Sequence:\n'\
            '%s\n'\
            'tool:  t = (%g, %g, %g),  RPY/xyz = (%g, %g, %g) deg' % (
                self.name, self.manuf, self.n, axes,
                self.ets,
                self.tool[0, 3], self.tool[1, 3], self.tool[2, 3], 0, 0, 0
            )

        return model


class et(object):
    """This class implements a single elementary transform (ET)

    :param axis_func: The function which calculated the values of the ET.
    :type axis_func: static et.T__ function
    :param eta: The coordinate of the ET. If not supplied the ET corresponds
        to a variable ET which is a joint
    :type eta: float, optional
    :param i: If this ET corresponds to a joint, i corresponds to the joint
        number within the robot
    :type i: int, optional
    :param axis: The axis in which the ET is oriented. One of 'Rx', 'Ry',
    'Rz', 'tx', 'ty', 'tz'.
    :type axis_s: str

    References: Kinematic Derivatives using the Elementary Transform Sequence,
        J. Haviland and P. Corke
    """
    def __init__(self, axis_func, eta=None, i=None):

        super(et, self).__init__()
        self.STATIC = 0
        self.VARIABLE = 1

        self._eta = eta
        self._axis_func = axis_func

        if axis_func == et.TRx:
            self._axis = 'Rx'
        elif axis_func == et.TRy:
            self._axis = 'Ry'
        elif axis_func == et.TRz:
            self._axis = 'Rz'
        elif axis_func == et.Ttx:
            self._axis = 'tx'
        elif axis_func == et.Tty:
            self._axis = 'ty'
        elif axis_func == et.Ttz:
            self._axis = 'tz'
        else:
            raise TypeError(
                'axis_func array must be an ET function, one of: et.TRx, '
                'et.TRy, et.TRz, et.Ttx, et.Tty, or et.Ttz.')

        if self.eta is not None:
            self._type = self.STATIC
            self._T = axis_func(eta)
        else:
            self._type = self.VARIABLE
            self._i = i

        if self._type is self.STATIC and self.axis[0] == 'R':
            self._eta_deg = self.eta * (180/np.pi)

    @property
    def eta(self):
        return self._eta

    @property
    def eta_deg(self):
        return self._eta_deg

    @property
    def axis_func(self):
        return self._eta

    @property
    def axis_func(self):
        return self._axis_func

    @property
    def axis(self):
        return self._axis

    @property
    def i(self):
        return self._i

    def T(self, q=None):
        """
        Calculates the transformation matrix of the ET

        :param q: Is used if this ET is variable (a joint)
        :type q: float (radians), required for variable ET's
        :return: The transformation matrix of the ET
        :rtype: float np.ndarray(4,4)
        """
        if self._type is self.STATIC:
            return self._T
        else:
            return self.axis_func(q)

    def __str__(self):
        """
        Pretty prints the ET. Will output angles in degrees

        :return: The transformation matrix of the ET
        :rtype: str
        """
        if self._type is self.STATIC:
            if self.axis[0] == 'R':
                return '%s(%g)' % (self.axis, self.eta_deg)
            else:
                return '%s(%g)' % (self.axis, self.eta)
        else:
            return '%s(q%d)' % (self.axis, self.i)

    def __repr__(self):
        return str(self)

    @staticmethod
    def TRx(q):
        """
        An elementary transform (ET). A pure rotation of q about the x-axis

        :param q: The amount of rotation about the x-axis
        :type q: float (radians)
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """
        return np.array([
            [1, 0,          0,         0],
            [0, np.cos(q), -np.sin(q), 0],
            [0, np.sin(q),  np.cos(q), 0],
            [0, 0,          0,         1]
        ])

    @staticmethod
    def TRy(q):
        """
        An elementary transform (ET). A pure rotation of q about the y-axis

        :param q: The amount of rotation about the y-axis
        :type q: float (radians)
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """
        return np.array([
            [np.cos(q),  0, np.sin(q), 0],
            [0,          1, 0,         0],
            [-np.sin(q), 0, np.cos(q), 0],
            [0,          0, 0,         1]
        ])

    @staticmethod
    def TRz(q):
        """
        An elementary transform (ET). A pure rotation of q about the z-axis

        :param q: The amount of rotation about the z-axis
        :type q: float (radians)
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """
        return np.array([
            [np.cos(q), -np.sin(q), 0, 0],
            [np.sin(q),  np.cos(q), 0, 0],
            [0,          0,         1, 0],
            [0,          0,         0, 1]
        ])

    @staticmethod
    def Ttx(q):
        """
        An elementary transform (ET). A pure translation of q along the x-axis

        :param q: The amount of translation along the x-axis
        :type q: float (metres)
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """
        return np.array([
            [1, 0, 0, q],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def Tty(q):
        """
        An elementary transform (ET). A pure translation of q along the x-axis

        :param q: The amount of translation along the x-axis
        :type q: float (metres)
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, q],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def Ttz(q):
        """
        An elementary transform (ET). A pure translation of q along the x-axis

        :param q: The amount of translation along the x-axis
        :type q: float (metres)
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, q],
            [0, 0, 0, 1]
        ])
