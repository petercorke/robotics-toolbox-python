#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

import numpy as np
import spatialmath as sm


class ET(object):
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
    def __init__(self, axis_func, axis, eta=None, joint=None):

        super(ET, self).__init__()
        self.STATIC = 0
        self.VARIABLE = 1

        self._eta = eta
        self._axis_func = axis_func
        self._axis = axis

        if self.eta is not None:
            self._type = self.STATIC
            self._T = axis_func(eta)
        else:
            self._type = self.VARIABLE
            self._j = joint

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
        return self._axis_func

    @property
    def axis(self):
        return self._axis

    @property
    def j(self):
        return self._j

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
            return '%s(q%d)' % (self.axis, self.j)

    def __repr__(self):
        return str(self)

    @staticmethod
    def _check_args(eta, joint):
        if eta is None and joint is None:
            raise ValueError(
                'One of eta (the elementary transform parameter), '
                'or joint (the joint number) must be supplied')

    @classmethod
    def TRx(cls, eta=None, joint=None):
        """
        An elementary transform (ET). A pure rotation of eta about the x-axis

        :param eta: The amount of rotation about the x-axis
        :type eta: float (radians)
        :param joint: The joint number within the robot
        :type joint: int
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """

        cls._check_args(eta, joint)

        def axis_func(eta):
            return np.array([
                [1, 0,          0,         0],
                [0, np.cos(eta), -np.sin(eta), 0],
                [0, np.sin(eta),  np.cos(eta), 0],
                [0, 0,          0,         1]
            ])

        return cls(axis_func, axis='Rx', eta=eta, joint=joint)

    @classmethod
    def TRy(cls, eta=None, joint=None):

        cls._check_args(eta, joint)

        def axis_func(eta):
            return np.array([
                [np.cos(eta),  0, np.sin(eta), 0],
                [0,          1, 0,         0],
                [-np.sin(eta), 0, np.cos(eta), 0],
                [0,          0, 0,         1]
            ])

        return cls(axis_func, axis='Ry', eta=eta, joint=joint)

    @classmethod
    def TRz(cls, eta=None, joint=None):

        cls._check_args(eta, joint)

        def axis_func(eta):
            return np.array([
                [np.cos(eta), -np.sin(eta), 0, 0],
                [np.sin(eta),  np.cos(eta), 0, 0],
                [0,          0,         1, 0],
                [0,          0,         0, 1]
            ])

        return cls(axis_func, axis='Rz', eta=eta, joint=joint)

    # @staticmethod
    # def TRx(q):
    #     """
    #     An elementary transform (ET). A pure rotation of q about the x-axis

    #     :param q: The amount of rotation about the x-axis
    #     :type q: float (radians)
    #     :return: The transformation matrix which is in SE(3)
    #     :rtype: float np.ndarray(4,4)
    #     """
    #     return np.array([
    #         [1, 0,          0,         0],
    #         [0, np.cos(q), -np.sin(q), 0],
    #         [0, np.sin(q),  np.cos(q), 0],
    #         [0, 0,          0,         1]
    #     ])

    # @staticmethod
    # def TRy(q):
    #     """
    #     An elementary transform (ET). A pure rotation of q about the y-axis

    #     :param q: The amount of rotation about the y-axis
    #     :type q: float (radians)
    #     :return: The transformation matrix which is in SE(3)
    #     :rtype: float np.ndarray(4,4)
    #     """
    #     return np.array([
    #         [np.cos(q),  0, np.sin(q), 0],
    #         [0,          1, 0,         0],
    #         [-np.sin(q), 0, np.cos(q), 0],
    #         [0,          0, 0,         1]
    #     ])

    # @staticmethod
    # def TRz(q):
    #     """
    #     An elementary transform (ET). A pure rotation of q about the z-axis

    #     :param q: The amount of rotation about the z-axis
    #     :type q: float (radians)
    #     :return: The transformation matrix which is in SE(3)
    #     :rtype: float np.ndarray(4,4)
    #     """
    #     return np.array([
    #         [np.cos(q), -np.sin(q), 0, 0],
    #         [np.sin(q),  np.cos(q), 0, 0],
    #         [0,          0,         1, 0],
    #         [0,          0,         0, 1]
    #     ])

    @classmethod
    def Ttx(cls, eta=None, joint=None):
        """
        An elementary transform (ET). A pure translation of q along the x-axis

        :param q: The amount of translation along the x-axis
        :type q: float (metres)
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """
        cls._check_args(eta, joint)

        def axis_func(eta):
            return np.array([
                [1, 0, 0, eta],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        return cls(axis_func, axis='tx', eta=eta, joint=joint)

    @classmethod
    def Tty(cls, eta=None, joint=None):
        """
        An elementary transform (ET). A pure translation of q along the x-axis

        :param q: The amount of translation along the x-axis
        :type q: float (metres)
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """
        cls._check_args(eta, joint)

        def axis_func(eta):
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, eta],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        return cls(axis_func, axis='ty', eta=eta, joint=joint)

    @classmethod
    def Ttz(cls, eta=None, joint=None):
        """
        An elementary transform (ET). A pure translation of q along the x-axis

        :param q: The amount of translation along the x-axis
        :type q: float (metres)
        :return: The transformation matrix which is in SE(3)
        :rtype: float np.ndarray(4,4)
        """
        cls._check_args(eta, joint)

        def axis_func(eta):
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, eta],
                [0, 0, 0, 1]
            ])

        return cls(axis_func, axis='tz', eta=eta, joint=joint)
