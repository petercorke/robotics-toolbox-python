#!/usr/bin/env python3
"""
@author: Jesse Haviland
@author: Peter Corke
"""
from collections import UserList, namedtuple
import numpy as np
from spatialmath import SE2
from spatialmath.base import getvector, getunit

# TODO, should this class be called ET2?  Typically would be used exclusively
# to ET
# TODO factor out mostly common code with ETS


class ETS(UserList):
    """
    This class implements an elementary transform sequence (ETS) for 2D

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

    :seealso: :func:`r`, :func:`tx`, :func:`ty`
    """

    et = namedtuple('ETS2', 'eta axis_func axis joint T')

    def __init__(self, axis_func=None, axis=None, eta=None, unit='rad'):

        super().__init__()  # init UserList superclass

        if axis_func is None and axis is None and eta is None:
            # ET()
            # create instance with no values
            self.data = []
            return

        if eta is not None:
            # constant value specified
            joint = False
            eta = getunit(eta, unit)
            T = axis_func(eta)
        else:
            # it's a variable joint
            joint = True
            if unit != 'rad':
                raise ValueError('can only use radians for a variable transform')
            T = None

        # Save all the params in a named tuple
        e = self.et(eta, axis_func, axis, joint, T)

        # And make it the only value of this instance
        self.data = [e]

    @property
    def eta(self):
        """
        The transform constant

        :return: The constant η if set
        :rtype: float or None

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox.robot.ETS2 import ETS
            >>> e = ETS.tx(1)
            >>> e.eta
            >>> e = ETS.r(90, 'deg')
            >>> e.eta
            >>> e = ETS.ty()
            >>> e.eta

        .. note:: If the value was given in degrees it will be converted to
            radians.
        """
        return self.data[0].eta

    @property
    def axis_func(self):
        return self.data[0].axis_func

    @property
    def axis(self):
        """
        The transform type and axis

        :return: The transform type and axis
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox.robot.ETS2 import ETS
            >>> e = ETS.tx(1)
            >>> e.axis
            >>> e = ETS.r(90, 'deg')
            >>> e.axis
        """
        return self.data[0].axis

    @property
    def isjoint(self):
        """
        Test if ET is a joint

        :return: True if a joint
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.isjoint
            >>> e = ETS.tx()
            >>> e.isjoint
        """
        return self.data[0].joint

    @property
    def isrevolute(self):
        """
        Test if ET is a rotation

        :return: True if a rotation
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.isrevolute
            >>> e = ETS.rx()
            >>> e.isrevolute
        """
        return self.axis[0] == 'R'

    @property
    def isprismatic(self):
        """
        Test if ET is a translation

        :return: True if a translation
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.isprismatic
            >>> e = ETS.rx()
            >>> e.isprismatic
        """
        return self.axis[0] == 'P'

    def T(self, q=None):
        """
        Evaluate an elementary transformation

        :param q: Is used if this ET is variable (a joint)
        :type q: float (radians), required for variable ET's
        :return: The transformation matrix of the ET
        :rtype: SE2 instance

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox.robot.ETS2 import ETS
            >>> e = ETS.tx(1)
            >>> e.T()
            >>> e = ETS.tx()
            >>> e.T(2)
        """
        if self.isjoint:
            return self.data[0].T
        else:
            return self.axis_func(q)

    def joints(self):
        """
        Get index of joint transforms

        :return: indices of transforms that are joints
        :rtype: list of int

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox.robot.ETS2 import ETS
            >>> e = ETS.r() * ETS.tx(1) * ETS.r() * ETS.tx(1)
            >>> e.joints()
        """
        return np.where([e.isjoint for e in self])[0]

    @property
    def config(self):
        """
        Joint configuration string

        :return: A string indicating the joint types
        :rtype: str

        A string comprising the characters 'R' or 'P' which indicate the types
        of joints in order from left to right.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox.robot.ETS2 import ETS
            >>> e = ETS.ty() * ETS.tx(1) * ETS.r() * ETS.tx(1)
            >>> e.config
        """
        return ''.join(['R' if self.isrevolute else 'P' for i in self.joints()])


    def eval(self, q):
        """
        Evaluate an ETS with joint coordinate substitution

        :param q: joint coordinates
        :type q: array-like
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :return: product of transformations
        :rtype: SE2 instance

        Effectively the forward-kinematics of the ET sequence.  Compounds the
        transforms left to right, and substitutes in joint coordinates as
        needed from consecutive elements of ``q``.

        Example:

        .. runblock:: pycon
            :linenos:

            >>> from roboticstoolbox.robot.ETS2 import ETS
            >>> e = ETS.r() * ETS.tx(1) * ETS.r() * ETS.tx(1)
            >>> print(e)
            >>> len(e)
            >>> e[1:3]
            >>> e.eval([0, 0])
            >>> e.eval([90, -90], 'deg')
        """
        T = SE2()

        q = getvector(q, out='sequence')
        for e in self:
            if e.isjoint:
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

            if et.isjoint:
                s = '%s(q%d)' % (et.axis, joint)
                joint += 1
            else:
                if et.isrevolute:
                    s = '%s(%g)' % (et.axis, et.eta * 180 / np.pi)
                else:
                    s = '%s(%g)' % (et.axis, et.eta)

            es.append(s)

        return " * ".join(es)

    # redefine * operator to concatenate the internal lists
    def __mul__(self, rest):
        """
        Overloaded ``*`` operator

        :return: [description]
        :rtype: [type]

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox.robot.ETS2 import ETS
            >>> e1 = ETS.r()
            >>> len(e1)
            >>> e2= ETS.tx(2)
            >>> len(e2)
            >>> e = e1 * e2
            >>> len(e)

        .. note:: The ``*`` operator implies composition, but actually the 
            result is a new ETS instance that contains the concatenation of
            the left and right operands in an internal list. In this example
            we see the length of the product is 2.
        """
        prod = ETS()
        prod.data = self.data + rest.data
        return prod

    def __rmul__(self, rest):
        prod = ETS()
        prod.data = self.data + rest
        return prod

    # redefine so that indexing returns an ET type
    def __getitem__(self, i):
        """
        Index or slice an ETS

        :param i: the index or slince
        :type i: int or slice
        :return: Elementary transform (sequence)
        :rtype: ETS

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox.robot.ETS2 import ETS
            >>> e = ETS.r() * ETS.tx(1) * ETS.r() * ETS.tx(1)
            >>> e[0]
            >>> e[1]
            >>> e[1:3]

        """
        item = ETS()
        data = self.data[i]  # can be [2] or slice, eg. [3:5]
        # ensure that data is always a list
        if isinstance(data, list):
            item.data = data
        else:
            item.data = [data]
        return item

    def pop(self, i=-1):
        """
        Pop value

        :param i: item in the list to pop, default is last
        :type i: int
        :return: the popped value
        :rtype: instance of same type
        :raises IndexError: if there are no values to pop

        Removes a value from the value list and returns it.  The original
        instance is modified. 

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
            >>> tail = e.pop()
            >>> tail
            >>> e
        """
        item = ETS()
        item.data = [super().pop(i)]
        return item
                
    def __repr__(self):
        return str(self)

    @classmethod
    def r(cls, eta=None, unit='rad'):
        """
        Pure rotation

        :param η: rotation angle
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.r(η)`` is an elementary rotation by a constant angle η
        - ``ETS.r()`` is an elementary rotation by a variable angle, i.e. a
          revolute robot joint

        .. note:: In the 2D case this is rotation around the normal to the
            xy-plane.

        :seealso: :func:`ETS`
        """
        return cls(lambda theta: SE2(theta), axis='R', eta=eta, unit=unit)

    @classmethod
    def tx(cls, eta=None):
        """
        Pure translation along the x-axis

        :param η: translation distance along the z-axis
        :type η: float
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.tx(η)`` is an elementary translation along the x-axis by a 
          distance constant η
        - ``ETS.tx()`` is an elementary translation along the x-axis by a
          variable distance, i.e. a prismatic robot joint

        :seealso: :func:`ETS`
        """
        return cls(lambda x: SE2(x, 0), axis='tx', eta=eta)

    @classmethod
    def ty(cls, eta=None):
        """
        Pure translation along the y-axis

        :param η: translation distance along the y-axis
        :type η: float
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.tx(η)`` is an elementary translation along the y-axis by a 
          distance constant η
        - ``ETS.tx()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint

        :seealso: :func:`ETS`
        """
        return cls(lambda y: SE2(0, y), axis='ty', eta=eta)


if __name__ == "__main__":
    from math import pi

    e = ET.tx(1) * ET.r() * ET.ty(2) * ET.r(pi / 2)
    print(e.joints())
    print(e.eval([0]))
