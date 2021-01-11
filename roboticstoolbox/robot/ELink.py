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
            jindex=None,
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
            if jindex is not None:
                v.jindex = jindex
            elif jindex is None and v.jindex is not None:
                jindex = v.jindex

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
        self._jindex = jindex

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
        if self.name is None:
            s = f"ets={self.ets()}"
        else:
            s = f"{self.name}, ets={self.ets()}"
        if self.parent is not None:
            s += f", parent={self.parent.name}"
        args = [s] + super()._params()
        return name + "(" + ", ".join(args) + ")"

    def __str__(self):
        """
        Pretty prints the ETS Model of the link. Will output angles in degrees

        :return: Pretty print of the robot link
        :rtype: str
        """
        name = self.__class__.__name__

        if self.name is None:
            return f"{name}[{self.ets()}] "
        else:
            if self.parent is None:
                parent = ""
            else:
                parent = f" [{self.parent.name}]"
            return f"{name}[{self.name}({parent}): {self.ets()}] "

    @property
    def v(self):
        """
        Variable part of link ETS

        :return: joint variable transform
        :rtype: ETS instance

        The ETS for each ELink comprises a constant part (possible the
        identity) followed by an optional joint variable transform.
        This property returns the latter.

        .. runblock:: pycon

            >>> from roboticstoolbox import ELink, ETS
            >>> link = ELink( ETS.tz(0.333) * ETS.rx(90, 'deg') * ETS.rz() )
            >>> print(link.v)
        """
        return self._v

    @property
    def Ts(self):
        """
        Constant part of link ETS

        :return: constant part of link transform
        :rtype: SE3 instance

        The ETS for each ELink comprises a constant part (possible the
        identity) followed by an optional joint variable transform.
        This property returns the constant part.  If no constant part
        is given, this returns an identity matrix.

        .. runblock:: pycon

            >>> from roboticstoolbox import ELink, ETS
            >>> link = ELink( ETS.tz(0.333) * ETS.rx(90, 'deg') * ETS.rz() )
            >>> link.Ts
            >>> link = ELink( ETS.rz() )
            >>> link.Ts
        """
        return self._Ts

    @property
    def isjoint(self):
        """
        Test if link has joint

        :return: test if link has a joint
        :rtype: bool

        The ETS for each ELink comprises a constant part (possible the
        identity) followed by an optional joint variable transform.
        This property returns the whether the

        .. runblock:: pycon

            >>> from roboticstoolbox import models
            >>> robot = models.URDF.Panda()
            >>> robot[1].isjoint  # link with joint
            >>> robot[8].isjoint  # static link
        """
        return self._v is not None

    @property
    def jindex(self):
        """
        Get/set joint index

        - ``link.jindex`` is the joint index
            :return: joint index
            :rtype: int
        - ``link.jindex = ...`` checks and sets the joint index

        For a serial-link manipulator the joints are numbered starting at zero
        and increasing sequentially toward the end-effector.  For branched
        mechanisms this is not so straightforward.

        The link's ``jindex`` property specifies the index of its joint
        variable within a vector of joint coordinates.

        .. note:: ``jindex`` values must be a sequence of integers starting
            at zero.
        """
        return self._jindex

    @jindex.setter
    def jindex(self, j):
        self._jindex = j

    # def isrevolute(self):
    #     """
    #     Checks if the joint is of revolute type

    #     :return: Ture if is revolute
    #     :rtype: bool
    #     """
    #     return self.v.isrevolute

    @property
    def isprismatic(self):
        """
        Checks if the joint is of prismatic type

        :return: True if is prismatic
        :rtype: bool
        """
        return self.isjoint and self.v.isprismatic

    @property
    def isrevolute(self):
        """
        Checks if the joint is of revolute type

        :return: True if is revolute
        :rtype: bool
        """
        return self.isjoint and self.v.isrevolute

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
        """
        Parent link

        :return: Link's parent
        :rtype: ELink instance

        This is a reference to

        .. runblock:: pycon

            >>> from roboticstoolbox import models
            >>> robot = models.URDF.Panda()
            >>> robot[0].parent  # base link has no parent
            >>> robot[1].parent  # second link's parent
        """
        return self._parent

    @property
    def child(self):
        return self._child

    @property
    def M(self):
        return self._M

    @property
    def geometry(self):
        """
        Get/set joint visual geometry

        - ``link.geometry`` is the list of the visual geometries which
            represent the shape of the link
            :return: the visual geometries
            :rtype: list of Shape
        - ``link.geometry = ...`` checks and sets the geometry
        - ``link.geometry.append(...)`` add geometry
        """
        return self._geometry

    @property
    def collision(self):
        """
        Get/set joint collision geometry

        - ``link.collision`` is the list of the collision geometries which
            represent the collidable shape of the link.
            :return: the collision geometries
            :rtype: list of Shape
        - ``link.collision = ...`` checks and sets the collision geometry
        - ``link.collision.append(...)`` add collision geometry

        The collision geometries are what is used to check for collisions.
        """
        return self._collision

    # @r.setter
    # def r(self, T):
    #     if not isinstance(T, SE3):
    #         T = SE3(T)
    #     self._r = T

    def A(self, q=None, fast=False):
        """
        Link transform matrix

        :param q: Joint coordinate (radians or metres). Not required for links
            with no variable
        :type q: float
        :param fast: return NumPy array instead of ``SE3``
        :type param: bool
        :return T: link frame transformation matrix
        :rtype T: SE3 or ndarray(4,4)

        ``LINK.A(q)`` is an SE(3) matrix that describes the rigid-body 
          transformation from the previous to the current link frame to
          the next, which depends on the joint coordinate ``q``.

        """

        if self.isjoint:
            # a variable joint
            if q is None:
                raise ValueError("q is required for variable joints")
            T = self.Ts.A @ self.v.T(q)
        else:
            # a fixed joint
            T = self.Ts.A

        if fast:
            return T
        else:
            return SE3(T, check=False)

    def ets(self):
        if self.v is None:
            return self._ets
        else:
            return self._ets * self.v

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
