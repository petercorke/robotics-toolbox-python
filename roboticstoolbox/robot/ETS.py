#!/usr/bin/env python3
"""
@author: Jesse Haviland
@author: Peter Corke
"""
from collections import UserList, namedtuple
import numpy as np
from spatialmath import SE3
from spatialmath.base import getvector, getunit, trotx, troty, trotz


class ETS(UserList):
    """
    This class implements an elementary transform sequence (ETS) for 3D

    :param axis_func: The function which calculates the transform of the ET.
    :type axis_func: static et.T__ function
    :param η: The coordinate of the ET. If not supplied the ET corresponds
        to a variable ET which is a joint
    :type eta: float, optional
    :param joint: If this ET corresponds to a joint, this corresponds to the
        joint number within the robot
    :type joint: int, optional


    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    :seealso: :func:`rx`, :func:`ry`, :func:`rz`, :func:`tx`, :func:`ty`, :func:`tz`
    """

    # _STATIC = 0
    # _VARIABLE = 1
    ets = namedtuple('ETS3', 'eta axis_func axis joint T')

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
            if unit != 'rad':
                raise ValueError('can only use radians for a variable transform')
            joint = True
            T = None

        # Save all the params in a named tuple
        e = self.ets(eta, axis_func, axis, joint, T)

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

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.eta
            >>> e = ETS.rx(90, 'deg')
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

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.axis
            >>> e = ETS.rx(90, 'deg')
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

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
            >>> e.config

        """
        return ''.join(['R' if self.isrevolute else 'P' for i in self.joints()])

    def T(self, q=None):
        """
        Evaluate an elementary transformation

        :param q: Is used if this ET is variable (a joint)
        :type q: float (radians), required for variable ET's
        :return: The transformation matrix of the ET
        :rtype: SE3 instance

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.tx(1)
            >>> e.T()
            >>> e = ETS.tx()
            >>> e.T(2)

        """
        if self.isjoint:
            return self.axis_func(q)
        else:
            return self.data[0].T  

    def joints(self):
        """
        Get index of joint transforms

        :return: indices of transforms that are joints
        :rtype: list of int

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
            >>> e.joints()

        """
        return np.where([e.isjoint for e in self])[0]

    def eval(self, q=None, unit='rad'):
        """
        Evaluate an ETS with joint coordinate substitution

        :param q: joint coordinates
        :type q: array-like
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :return: product of transformations
        :rtype: SE3 instance

        Effectively the forward-kinematics of the ET sequence.  Compounds the
        transforms left to right, and substitutes in joint coordinates as
        needed from consecutive elements of ``q``.

        Example:

        .. runblock:: pycon
            :linenos:

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
            >>> print(e)
            >>> len(e)
            >>> e[1:3]
            >>> e.eval([0, 0])
            >>> e.eval([90, -90], 'deg')
        """
        T = np.eye(4)
        if q is not None:
            q = getvector(q, out='sequence')
        for e in self:
            if e.isjoint:
                qj = q.pop(0)
                if e.isrevolute and unit == 'deg':
                    qj *= np.pi / 180.0
                T = T @ e.T(qj)
            else:
                T = T @ e.T()
        return SE3(T)

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

            >>> from roboticstoolbox import ETS
            >>> e1 = ETS.rz()
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

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
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

    def __repr__(self):
        return str(self)

    @classmethod
    def rx(cls, eta=None, unit='rad'):
        """
        Pure rotation about the x-axis

        :param η: rotation about the x-axis
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.rz(η)`` is an elementary rotation about the x-axis by a constant
          angle η
        - ``ETS.rz()`` is an elementary rotation about the x-axis by a variable
          angle, i.e. a revolute robot joint

        :seealso: :func:`ETS`
        """
        return cls(trotx, axis='Rx', eta=eta, unit=unit)

    @classmethod
    def ry(cls, eta=None, unit='rad'):
        """
        Pure rotation about the y-axis

        :param η: rotation about the y-axis
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.ry(η)`` is an elementary rotation about the y-axis by a constant
          angle η
        - ``ETS.ry()`` is an elementary rotation about the y-axis by a variable
          angle, i.e. a revolute robot joint

        :seealso: :func:`ETS`

        """
        return cls(troty, axis='Ry', eta=eta, unit=unit)

    @classmethod
    def rz(cls, eta=None, unit='rad'):
        """
        Pure rotation about the z-axis

        :param η: rotation about the z-axis
        :type η: float
        :param unit: angular unit, "rad" [default] or "deg"
        :type unit: str
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.rz(η)`` is an elementary rotation about the z-axis by a constant
          angle η
        - ``ETS.rz()`` is an elementary rotation about the z-axis by a variable
          angle, i.e. a revolute robot joint

        :seealso: :func:`ETS`
        """
        return cls(trotz, axis='Rz', eta=eta, unit=unit)

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

        # this method is 3x faster than using lambda x: transl(x, 0, 0)
        def axis_func(eta):
            return np.array([
                [1, 0, 0, eta],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        return cls(axis_func, axis='tx', eta=eta)

    @classmethod
    def ty(cls, eta=None):
        """
        Pure translation along the y-axis

        :param η: translation distance along the y-axis
        :type η: float
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.ty(η)`` is an elementary translation along the y-axis by a 
          distance constant η
        - ``ETS.ty()`` is an elementary translation along the y-axis by a
          variable distance, i.e. a prismatic robot joint

        :seealso: :func:`ETS`

        """
        def axis_func(eta):
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, eta],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        return cls(axis_func, axis='ty', eta=eta)

        # return cls(SE3.Ty, axis='ty', eta=eta)

    @classmethod
    def tz(cls, eta=None):
        """
        Pure translation along the z-axis

        :param η: translation distance along the z-axis
        :type η: float
        :return: An elementary transform
        :rtype: ETS instance

        - ``ETS.tz(η)`` is an elementary translation along the z-axis by a 
          distance constant η
        - ``ETS.tz()`` is an elementary translation along the z-axis by a
          variable distance, i.e. a prismatic robot joint

        :seealso: :func:`ETS`
        """
        def axis_func(eta):
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, eta],
                [0, 0, 0, 1]
            ])

        return cls(axis_func, axis='tz', eta=eta)


if __name__ == "__main__":

    print(ETS.rx(0.2))
    print(ETS.rx(45, 'deg'))
    print(ETS.tz(0.75))
    e = ETS.rx(45, 'deg') * ETS.tz(0.75)
    print(e)
    print(e.eval())

    from roboticstoolbox import ETS
    e = ETS.rz() * ETS.tx(1) * ETS.rz() * ETS.tx(1)
    print(e.eval([0, 0]))
    print(e.eval([90, -90], 'deg'))
    
    # l1 = 0.672
    # l2 = -0.2337
    # l3 = 0.4318
    # l4 = 0.0203
    # l5 = 0.0837
    # l6 = 0.4318

    # e = ETS.tz(l1) * ETS.rz() * ETS.ry() * ETS.ty(l2) * ETS.tz(l3) * ETS.ry() \
    #     * ETS.tx(l4) * ETS.ty(l5) * ETS.tz(l6) * ETS.rz() * ETS.ry() * ETS.rz()
    # print(e.joints())
    # print(e.config)
    # print(e.eval(np.zeros((6,))))
