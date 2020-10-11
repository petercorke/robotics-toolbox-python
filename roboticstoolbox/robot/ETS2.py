#!/usr/bin/env python3
"""
@author: Jesse Haviland
@author: Peter Corke
"""
from collections import UserList, namedtuple
import numpy as np
from spatialmath import SE2
from spatialmath.base import getvector


class ET(UserList):
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

    # TODO should probably prefix with __
    STATIC = 0
    VARIABLE = 1
    et = namedtuple('ET', 'eta axis_func axis jtype T')

    def __init__(self, axis_func=None, axis=None, eta=None):

        super().__init__()  # init UserList superclass

        if axis_func is None and axis is None and eta is None:
            # ET()
            # create instance with no values
            self.data = []
            return

        if eta is not None:
            # constant value specified
            jtype = self.STATIC
            T = axis_func(eta)
        else:
            # it's a variable joint
            jtype = self.VARIABLE
            T = None

        # Save all the params in a named tuple
        e = self.et(eta, axis_func, axis, jtype, T)

        # And make it the only value of this instance
        self.data = [e]

    @property
    def eta(self):
        return self.data[0].eta

    @property
    def axis_func(self):
        return self.data[0].axis_func

    @property
    def axis(self):
        return self.data[0].axis

    @property
    def jtype(self):
        return self.data[0].jtype

    def T(self, q=None):
        """
        Calculates the transformation matrix of the ET

        :param q: Is used if this ET is variable (a joint)
        :type q: float (radians), required for variable ET's
        :return: The transformation matrix of the ET
        :rtype: SE2

        """
        if self.jtype is self.STATIC:
            return self.data[0].T
        else:
            return self.axis_func(q)

    def joints(self):
        """
        Get index of joint transforms

        :return: indices of transforms that are joints
        :rtype: list
        """
        return np.where([e.jtype == self.VARIABLE for e in self])[0]

    def eval(self, q):
        """
        Evaluate an ETS with joint coordinate substitution

        :param q: joint coordinates
        :type q: array-like
        :return: product of transformations
        :rtype: SE2

        Effectively the forward-kinematics of the ET sequence.  Compounds the
        transforms left to right, and substitutes in joint coordinates as 
        needed from consecutive elements of ``q``.
        """
        T = SE2()

        q = getvector(q, out='sequence')
        for e in self:
            if e.jtype == self.VARIABLE:
                T *= e.T(q.pop(0))
            else:
                T *= e.T()
        return T

    def __str__(self):
        """
        Pretty prints the ET. Will output angles in degrees

        :return: The transformation matrix of the ET
        :rtype: str

        """
        es = []
        joint = 0

        # For et in the object, display it, data comes from properties
        # which come from the named tuple
        for et in self:

            if et.jtype is self.STATIC:
                if et.axis[0] == 'R':
                    s = '%s(%g)' % (et.axis, et.eta * 180 / np.pi)
                else:
                    s = '%s(%g)' % (et.axis, et.eta)
            else:
                s = '%s(q%d)' % (et.axis, joint)
                joint += 1
            es.append(s)

        return " * ".join(es)

    # redefine * operator to concatenate the internal lists
    def __mul__(self, rest):
        prod = ET()
        prod.data = self.data + rest.data
        return prod

    def __rmul__(self, rest):
        prod = ET()
        prod.data = self.data + rest
        return prod

    # redefine so that indexing returns an ET type
    def __getitem__(self, i):
        item = ET()
        data = self.data[i]  # can be [2] or slice, eg. [3:5]
        # ensure that data is always a list
        if isinstance(data, list):
            item.data = data
        else:
            item.data = [data]
        return item

    def __repr__(self):
        return str(self)

    @classmethod
    def r(cls, eta=None):
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
        return cls(lambda theta: SE2(theta), axis='R', eta=eta)

    @classmethod
    def tx(cls, eta=None):
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
        return cls(lambda x: SE2(x, 0), axis='tx', eta=eta)

    @classmethod
    def ty(cls, eta=None):
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
        return cls(lambda y: SE2(0, y), axis='ty', eta=eta)

if __name__ == "__main__":
    from math import pi

    e = ET.tx(1) * ET.r() * ET.ty(2) * ET.r(pi / 2)
    print(e.joints())
    print(e.eval([0]))

