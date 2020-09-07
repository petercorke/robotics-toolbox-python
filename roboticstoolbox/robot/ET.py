#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy as np
from spatialmath import SE3


class ET(object):
    """
    This class implements a single elementary transform (ET)

    :param axis_func: The function which calculates the transform of the ET.
    :type axis_func: static et.T__ function
    :param eta: The coordinate of the ET. If not supplied the ET corresponds
        to a variable ET which is a joint
    :type eta: float, optional
    :param joint: If this ET corresponds to a joint, this corresponds to the
        joint number within the robot
    :type joint: int, optional

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
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
            self._jtype = self.STATIC
            self._T = axis_func(eta)
        else:
            self._jtype = self.VARIABLE
            self.j = joint

        if self._jtype is self.STATIC and self.axis[0] == 'R':
            self._eta_deg = self.eta * (180 / np.pi)

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

    @property
    def jtype(self):
        return self._jtype

    @j.setter
    def j(self, j_new):
        self._j = j_new

    def T(self, q=None):
        """
        Calculates the transformation matrix of the ET

        :param q: Is used if this ET is variable (a joint)
        :type q: float (radians), required for variable ET's
        :return: The transformation matrix of the ET
        :rtype: SE3

        """
        if self.jtype is self.STATIC:
            return self._T
        else:
            return self.axis_func(q)

    def __str__(self):
        """
        Pretty prints the ET. Will output angles in degrees

        :return: The transformation matrix of the ET
        :rtype: str

        """
        if self.jtype is self.STATIC:
            if self.axis[0] == 'R':
                return '%s(%g)' % (self.axis, self.eta_deg)
            else:
                return '%s(%g)' % (self.axis, self.eta)
        else:
            return '%s(q%d)' % (self.axis, self.j)

    def __repr__(self):
        return str(self)

    # @staticmethod
    # def _check_args(eta, joint):
    #     if eta is None and joint is None:
    #         raise ValueError(
    #             'One of eta (the elementary transform parameter), '
    #             'or joint (the joint number) must be supplied')

    @classmethod
    def TRx(cls, eta=None):
        """
        An elementary transform (ET). A pure rotation of eta about the x-axis.

        L = TRx(eta) will instantiate an ET object which represents a pure
        rotation about the x-axis by amount eta.

        L = TRx() as above except this ET representation a variable
        rotation, i.e. a joint

        :param eta: The amount of rotation about the x-axis
        :type eta: float (radians)
        :param joint: The joint number within the robot
        :type joint: int
        :return: An ET object
        :rtype: ET

        """

        def axis_func(eta):
            return SE3(np.array([
                [1, 0, 0, 0],
                [0, np.cos(eta), -np.sin(eta), 0],
                [0, np.sin(eta), np.cos(eta), 0],
                [0, 0, 0, 1]
            ]))

        return cls(axis_func, axis='Rx', eta=eta)

    @classmethod
    def TRy(cls, eta=None):
        """
        An elementary transform (ET). A pure rotation of eta about the y-axis.

        L = TRy(eta) will instantiate an ET object which represents a pure
        rotation about the y-axis by amount eta.

        L = TRy() as above except this ET representation a variable
        rotation, i.e. a joint

        :param eta: The amount of rotation about the y-axis
        :type eta: float (radians)
        :param joint: The joint number within the robot
        :type joint: int
        :return: An ET object
        :rtype: ET

        """

        def axis_func(eta):
            return SE3(np.array([
                [np.cos(eta), 0, np.sin(eta), 0],
                [0, 1, 0, 0],
                [-np.sin(eta), 0, np.cos(eta), 0],
                [0, 0, 0, 1]
            ]))

        return cls(axis_func, axis='Ry', eta=eta)

    @classmethod
    def TRz(cls, eta=None):
        """
        An elementary transform (ET). A pure rotation of eta about the z-axis.

        L = TRz(eta) will instantiate an ET object which represents a pure
        rotation about the z-axis by amount eta.

        L = TRz() as above except this ET representation a variable
        rotation, i.e. a joint

        :param eta: The amount of rotation about the z-axis
        :type eta: float (radians)
        :return: An ET object
        :rtype: ET

        """

        def axis_func(eta):
            return SE3(np.array([
                [np.cos(eta), -np.sin(eta), 0, 0],
                [np.sin(eta), np.cos(eta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]))

        return cls(axis_func, axis='Rz', eta=eta)

    @classmethod
    def Ttx(cls, eta=None):
        """
        An elementary transform (ET). A pure translation of eta along the
        x-axis

        L = Ttx(eta) will instantiate an ET object which represents a pure
        translation along the x-axis by amount eta.

        L = Ttx() as above except this ET representation a variable
        translation, i.e. a joint

        :param eta: The amount of translation along the x-axis
        :type eta: float (metres)
        :return: An ET object
        :rtype: ET

        """

        def axis_func(eta):
            return SE3(np.array([
                [1, 0, 0, eta],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]))

        return cls(axis_func, axis='tx', eta=eta)

    @classmethod
    def Tty(cls, eta=None):
        """
        An elementary transform (ET). A pure translation of eta along the
        x-axis

        L = Tty(eta) will instantiate an ET object which represents a pure
        translation along the y-axis by amount eta.

        L = Tty() as above except this ET representation a variable
        translation, i.e. a joint

        :param eta: The amount of translation along the x-axis
        :type eta: float (metres)
        :return: An ET object
        :rtype: ET

        """

        def axis_func(eta):
            return SE3(np.array([
                [1, 0, 0, 0],
                [0, 1, 0, eta],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]))

        return cls(axis_func, axis='ty', eta=eta)

    @classmethod
    def Ttz(cls, eta=None):
        """
        An elementary transform (ET). A pure translation of eta along the
        z-axis

        L = Ttz(eta) will instantiate an ET object which represents a pure
        translation along the z-axis by amount eta.

        L = Ttz() as above except this ET representation a variable
        translation, i.e. a joint

        :param eta: The amount of translation along the x-axis
        :type eta: float (metres)
        :return: An ET object
        :rtype: ET

        """

        def axis_func(eta):
            return SE3(np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, eta],
                [0, 0, 0, 1]
            ]))

        return cls(axis_func, axis='tz', eta=eta)
