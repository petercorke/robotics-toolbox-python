#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

# import numpy as np
from spatialmath import SE3
# from spatialmath.base.argcheck import getvector, verifymatrix, isscalar
import roboticstoolbox as rp
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.Link import Link


class ELink(Link):
    """
    ETS link class

    :param ets: kinematic - The elementary transforms which make up the link
    :type ets: ETS

    :param qlim: joint variable limits [min max]
    :type qlim: float ndarray(2)

    :param m: dynamic - link mass
    :type m: float
    :param r: dynamic - position of COM with respect to link frame
    :type r:  SE3
    :param I: dynamic - inertia of link with respect to COM
    :type I: float ndarray(3,3)
    :param Jm: dynamic - motor inertia
    :type Jm: float
    :param B: dynamic - motor viscous friction
    :type B: float
    :param Tc: dynamic - motor Coulomb friction (1x2 or 2x1)
    :type Tc: float ndarray(2)
    :param G: dynamic - gear ratio
    :type G: float

    The ELink object holds all information related to a robot link and can form
    a serial-connected chain or a rigid-body tree.

    It inherits from the Link class which provides common functionality such
    as joint and link such as kinematics parameters,
    .

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    :seealso: :class:`Link`, :class:`DHLink`
    """

    def __init__(
            self,
            ets=ETS(),
            v=None,
            parent=None,
            **kwargs):

        # process common options
        super(ELink, self).__init__(**kwargs)

        # check we have an ETS
        if isinstance(ets, ETS):
            self._ets = ets
        else:
            raise TypeError(
                'The ets argument must be of type ETS')

        if v is None and len(ets) > 0 and ets[-1].isjoint:
            v = ets.pop()

        # TODO simplify this logic, can be ELink class or None
        if isinstance(parent, list):
            raise TypeError(
                'Only one parent link can be present')
        elif not isinstance(parent, ELink) and parent is not None:
            raise TypeError(
                'Parent must be of type ELink')

        self._parent = parent
        self._child = []
        self._joint_name = None
        self._jindex = None

        # Initialise the static transform representing the constant
        # component of the ETS
        self._init_Ts()

        # Check the variable joint
        if v is None:
            self._joint = False
        elif not isinstance(v, ETS):
            raise TypeError('v must be of type ETS')
        elif not v[0].isjoint:
            raise ValueError('v must be a variable ETS')
        elif len(v) > 1:
            raise ValueError(
                "An elementary link can only have one joint variable")
        else:
            self._joint = True

        self._v = v

    def _init_Ts(self):
        # Number of transforms in the ETS excluding the joint variable
        self._M = len(self._ets)

        # Initialise joints
        if isinstance(self._ets, ETS):
            self._Ts = SE3()
            for i in range(self.M):
                if self._ets[i].isjoint:
                    raise ValueError('The transforms in ets must be constant')

                if not isinstance(self._ets[i].T(), SE3):
                    self._Ts *= SE3(self._ets[i].T())
                else:
                    self._Ts *= self._ets[i].T()

        elif isinstance(self._ets, SE3):
            self._Ts = self._ets

    def __repr__(self):
        name = self.__class__.__name__
        s = "ets=" + str(self.ets())
        if self.parent is not None:
            s += ", parent=" + str(self.parent.name)
        args = [s] + super()._params()
        return name + "(" + ", ".join(args) + ")"

    def __str__(self):
        """
        Pretty prints the ETS Model of the link. Will output angles in degrees

        :return: Pretty print of the robot link
        :rtype: str
        """
        # name = self.__class__.__name__
        if self.parent is None:
            parent = ""
        else:
            parent = f" [{self.parent.name}]"
        return f"name[{self.name}({parent}): {self.ets()}] "

    @property
    def v(self):
        return self._v

    @property
    def Ts(self):
        return self._Ts

    @property
    def collision(self):
        return self._collision

    @property
    def geometry(self):
        return self._geometry

    @property
    def isjoint(self):
        return self._v is not None

    @property
    def jindex(self):
        return self._jindex

    @jindex.setter
    def jindex(self, j):
        self._jindex = j

    # @property
    # def ets(self):
    #     return self._ets

    # @property
    # def parent_name(self):
    #     return self._parent_name

    # @property
    # def child_name(self):
    #     return self._child_name

    @property
    def parent(self):
        return self._parent

    @property
    def child(self):
        return self._child

    @property
    def M(self):
        return self._M

    @collision.setter
    def collision(self, coll):
        new_coll = []

        if isinstance(coll, list):
            for gi in coll:
                if isinstance(gi, rp.Shape):
                    new_coll.append(gi)
                else:
                    raise TypeError('Collision must be of Shape class')
        elif isinstance(coll, rp.Shape):
            new_coll.append(coll)
        else:
            raise TypeError('Geometry must be of Shape class or list of Shape')

        self._collision = new_coll

    @geometry.setter
    def geometry(self, geom):
        new_geom = []

        if isinstance(geom, list):
            for gi in geom:
                if isinstance(gi, rp.Shape):
                    new_geom.append(gi)
                else:
                    raise TypeError('Geometry must be of Shape class')
        elif isinstance(geom, rp.Shape):
            new_geom.append(geom)
        else:
            raise TypeError('Geometry must be of Shape class or list of Shape')

        self._geometry = new_geom

    # @r.setter
    # def r(self, T):
    #     if not isinstance(T, SE3):
    #         T = SE3(T)
    #     self._r = T

    def A(self, q=None, fast=False):
        """
        Link transform matrix

        T = A(q) is the link homogeneous transformation matrix (4x4)
        corresponding to the link variable q

        :param q: Joint coordinate (radians or metres). Not required for links
            with no variable
        :type q: float
        :return T: link homogeneous transformation matrix
        :rtype T: SE3

        """

        # j = 0
        # tr = SE3()

        if self.isjoint and q is None:
            raise ValueError("q is required for variable joints")

        # for k in range(self.M):
        #     if self.ets[k].jtype == self.ets[k].VARIABLE:
        #         T = self.ets[k].T(q)
        #         j += 1
        #     else:
        #         T = self.ets[k].T()

        #     tr = tr * T

        if self.v is not None:
            if fast:
                return self.Ts.A @ self.v.T(q)
            else:
                return SE3(self.Ts.A @ self.v.T(q), check=False)
        else:
            if fast:
                return self.Ts.A
            else:
                return self.Ts

    def ets(self):
        if self.v is None:
            return self._ets
        else:
            return self._ets * self.v

    def closest_point(self, shape, inf_dist=1.0):
        '''
        closest_point(shape, inf_dist) returns the minimum euclidean
        distance between this link and shape, provided it is less than
        inf_dist. It will also return the points on self and shape in the
        world frame which connect the line of length distance between the
        shapes. If the distance is negative then the shapes are collided.

        :param shape: The shape to compare distance to
        :type shape: Shape
        :param inf_dist: The minimum distance within which to consider
            the shape
        :type inf_dist: float
        :returns: d, p1, p2 where d is the distance between the shapes,
            p1 and p2 are the points in the world frame on the respective
            shapes
        :rtype: float, SE3, SE3
        '''

        d = 10000
        p1 = None,
        p2 = None

        for col in self.collision:
            td, tp1, tp2 = col.closest_point(shape, inf_dist)

            if td is not None and td < d:
                d = td
                p1 = tp1
                p2 = tp2

        if d == 10000:
            d = None

        return d, p1, p2

    def collided(self, shape):
        '''
        collided(shape) checks if this link and shape have collided

        :param shape: The shape to compare distance to
        :type shape: Shape
        :returns: True if shapes have collided
        :rtype: bool
        '''

        for col in self.collision:
            if col.collided(shape):
                return True

        return False
