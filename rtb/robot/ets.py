#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

import numpy as np
# import time


class ets(object):
    """
    The Elementary Transform Sequence (ETS). A superclass which represents the kinematics of a serial-link manipulator

    :param et_list: List of elementary transforms which represent the robot kinematics
    :type et_list: list of etb.robot.et
    :param q_idx: List of indexes within the ets_list which correspond to joints
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
        :type robot: SerialLink, required
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
                    ets.append(et(et.Ttx, 'tx', L.a))
                    M += 1

                # Append Rx(alpha)
                if L.alpha != 0:
                    ets.append(et(et.TRx, 'Rx', L.alpha))
                    M += 1

                if L.is_revolute:
                    # Append Tz(d)
                    if L.d != 0:
                        ets.append(et(et.Ttz, 'tz', L.d))
                        M += 1

                    # Append Rz(q)
                    ets.append(et(et.TRz, 'Rz', i=j+1))
                    q_idx.append(M)
                    M += 1

                else:
                    # Append Tz(q)
                    ets.append(et(et.Ttz, 'tz', i=j+1))
                    q_idx.append(M)
                    M += 1

                    # Append Rz(theta)
                    if L.theta != 0:
                        ets.append(et(et.TRz, 'Rz', L.alpha))
                        M += 1

        return cls(
            ets,
            q_idx,
            robot.name,
            robot.manuf,
            robot.base,
            robot.tool)


class et(object):
    """This class implements a single elementary transform (ET)

    :param axis: The function which calculated the values of the ET.
    :type axis: function
    :param axis_s: The axis in which the ET is oriented. One of 'Rx', 'Ry',
        'Rz', 'tx', 'ty', 'tz'.
    :type axis_s: str
    :param eta: The coordinate of the ET. If not supplied the ET corresponds
        to a variable ET which is a joint
    :type eta: float, optional
    :param i: If this ET corresponds to a joint, i corresponds to the joint
        number within the robot
    :type i: int, optional

    References: Kinematic Derivatives using the Elementary Transform Sequence,
        J. Haviland and P. Corke
    """
    def __init__(self, axis, axis_s, eta=None, i=None):

        super(et, self).__init__()
        self.STATIC = 0
        self.VARIABLE = 1

        self._eta = eta
        self._axis = axis
        self._axis_s = axis_s

        if self._eta is not None:
            self._type = self.STATIC
            self._T = axis(eta)
        else:
            self._type = self.VARIABLE
            self._i = i

        if self._type is self.STATIC and self._axis_s[0] == 'R':
            self._eta_deg = self._eta * (180/np.pi)

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
            return self._axis(q)

    def __str__(self):
        """
        Pretty prints the ET. Will output angles in degrees

        :return: The transformation matrix of the ET
        :rtype: str
        """
        if self._type is self.STATIC:
            if self._axis_s[0] == 'R':
                return '%s(%g)' % (self._axis_s, self._eta_deg)
            else:
                return '%s(%g)' % (self._axis_s, self._eta)
        else:
            return '%s(q%d)' % (self._axis_s, self._i)

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
