#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

import numpy as np
from spatialmath import SE3
from spatialmath.base.argcheck import getvector, verifymatrix, isscalar
import ropy as rp


class ELink(object):
    """
    A link superclass for all link types. A Link object holds all information
    related to a robot joint and link such as kinematics parameters,
    rigid-body inertial parameters, motor and transmission parameters.

    :param ETS: kinematic - The elementary transforms which make up the link
    :type ETS: list (ET)

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

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    """

    def __init__(
            self,
            ET_list=[],
            name='',
            parent=None,
            qlim=np.zeros(2),
            m=0.0,
            r=None,
            I=np.zeros((3, 3)),  # noqa
            Jm=0.0,
            B=0.0,
            Tc=np.zeros(2),
            G=1.0,
            geometry=[],
            collision=[]):

        super(ELink, self).__init__()

        self.STATIC = 0
        self.VARIABLE = 1

        self._ets = ET_list
        self._q_idx = []

        self._name = name

        if isinstance(parent, ELink):
            parent = [parent]
        elif parent is None:
            parent = []
        elif not isinstance(parent, list):
            raise TypeError('The parent link must be of type ELink or list of Elink')

        self._parent = parent
        self._child = []

        # Number of transforms in the ETS
        self._M = len(self._ets)

        # Initialise joints
        for i in range(self.M):
            if ET_list[i].jtype is not ET_list[i].STATIC:
                ET_list[i].j = len(self._q_idx)
                self._q_idx.append(i)

        if len(self._q_idx) > 1:
            raise ValueError(
                "An elementary link can only have one joint variable")
        elif len(self._q_idx) == 0:
            self._jtype = self.STATIC
            self._q_idx = None
        else:
            self._jtype = self.VARIABLE
            self._q_idx = self._q_idx[0]

        self.qlim = qlim
        self.geometry = geometry
        self.collision = collision

        # Dynamic Parameters
        self.m = m
        self.r = r
        self.I = I  # noqa
        self.Jm = Jm
        self.B = B
        self.Tc = Tc
        self.G = G

    @property
    def collision(self):
        return self._collision

    @property
    def geometry(self):
        return self._geometry

    @property
    def jtype(self):
        return self._jtype

    @property
    def ets(self):
        return self._ets

    @property
    def name(self):
        return self._name

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

    @property
    def qlim(self):
        return self._qlim

    @property
    def m(self):
        return self._m

    @property
    def r(self):
        return self._r

    @property
    def I(self):  # noqa
        return self._I

    @property
    def Jm(self):
        return self._Jm

    @property
    def B(self):
        return self._B

    @property
    def Tc(self):
        return self._Tc

    @property
    def G(self):
        return self._G

    @property
    def q_idx(self):
        return self._q_idx

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

    @qlim.setter
    def qlim(self, qlim_new):
        self._qlim = getvector(qlim_new, 2)

    @m.setter
    def m(self, m_new):
        self._m = m_new

    @r.setter
    def r(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._r = T

    @I.setter
    def I(self, I_new):  # noqa
        # Try for Inertia Matrix
        try:
            verifymatrix(I_new, (3, 3))
        except (ValueError, TypeError):

            # Try for the moments and products of inertia
            # [Ixx Iyy Izz Ixy Iyz Ixz]
            try:
                Ia = getvector(I_new, 6)
                I_new = np.array([
                    [Ia[0], Ia[3], Ia[5]],
                    [Ia[3], Ia[1], Ia[4]],
                    [Ia[5], Ia[4], Ia[2]]
                ])
            except ValueError:

                # Try for the moments of inertia
                # [Ixx Iyy Izz]
                Ia = getvector(I_new, 3)
                I_new = np.diag(Ia)

        self._I = I_new

    @Jm.setter
    def Jm(self, Jm_new):
        self._Jm = Jm_new

    @B.setter
    def B(self, B_new):
        if isscalar(B_new):
            self._B = B_new
        else:
            raise TypeError("B must be a scalar")

    @Tc.setter
    def Tc(self, Tc_new):

        try:
            # sets Coulomb friction parameters to [F -F], for a symmetric
            # Coulomb friction model.
            Tc = getvector(Tc_new, 1)
            Tc_new = np.array([Tc[0], -Tc[0]])
        except ValueError:
            # [FP FM] sets Coulomb friction to [FP FM], for an asymmetric
            # Coulomb friction model. FP>0 and FM<0.  FP is applied for a
            # positive joint velocity and FM for a negative joint
            # velocity.
            Tc_new = getvector(Tc_new, 2)

        self._Tc = Tc_new

    @G.setter
    def G(self, G_new):
        self._G = G_new

    def __str__(self):
        """
        Pretty prints the ETS Model of the link. Will output angles in degrees

        :return: Pretty print of the robot link
        :rtype: str
        """
        return str(self._ets)

    def _copy(self):
        # Copy the Link
        link = ELink(  # noqa
            ET_list=self.ets,
            qlim=self.qlim,
            m=self.m,
            r=self.r,
            I=self.I,
            Jm=self.Jm,
            B=self.B,
            Tc=self.Tc,
            G=self.G)

        return link

    def dyn(self):
        """
        Show inertial properties of link

        s = dyn() returns a string representation the inertial properties of
        the link object in a multi-line format. The properties shown are mass,
        centre of mass, inertia, friction, gear ratio and motor properties.

        :return s: The string representation of the link dynamics
        :rtype s: string
        """

        s = "m     =  {:.2f} \n" \
            "r     =  {:.2f} {:.2f} {:.2f} \n" \
            "        | {:.2f} {:.2f} {:.2f} | \n" \
            "I     = | {:.2f} {:.2f} {:.2f} | \n" \
            "        | {:.2f} {:.2f} {:.2f} | \n" \
            "Jm    =  {:.2f} \n" \
            "B     =  {:.2f} \n" \
            "Tc    =  {:.2f}(+) {:.2f}(-) \n" \
            "G     =  {:.2f} \n" \
            "qlim  =  {:.2f} to {:.2f}".format(
                self.m,
                self.r.t[0], self.r.t[1], self.r.t[2],
                self.I[0, 0], self.I[0, 1], self.I[0, 2],
                self.I[1, 0], self.I[1, 1], self.I[1, 2],
                self.I[2, 0], self.I[2, 1], self.I[2, 2],
                self.Jm,
                self.B,
                self.Tc[0], self.Tc[1],
                self.G,
                self.qlim[0], self.qlim[1]
            )

        return s

    def A(self, q=None):
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

        j = 0
        tr = SE3()

        if self.q_idx is not None and q is None:
            raise ValueError("q is required for variable joints")

        for k in range(self.M):
            if self.ets[k].jtype == self.ets[k].VARIABLE:
                T = self.ets[k].T(q)
                j += 1
            else:
                T = self.ets[k].T()

            tr = tr * T

        return tr

    def islimit(self, q):
        """
        Checks if the joint is exceeding a joint limit

        :return: True if joint is exceeded
        :rtype: bool

        """

        if q < self.qlim[0] or q > self.qlim[1]:
            return True
        else:
            return False

    def nofriction(self, coulomb=True, viscous=False):
        """
        l2 = nofriction(coulomb, viscous) copies the link and returns a link
        with the same parameters except, the Coulomb and/or viscous friction
        parameter to zero.

        l2 = nofriction() as above except the the Coulomb parameter is set to
        zero.

        :param coulomb: if True, will set the coulomb friction to 0
        :type coulomb: bool
        :param viscous: if True, will set the viscous friction to 0
        :type viscous: bool
        """

        # Copy the Link
        link = self._copy()

        if viscous:
            link.B = 0.0

        if coulomb:
            link.Tc = [0.0, 0.0]

        return link

    def friction(self, qd):
        """
        tau = friction(qd) Calculates the joint friction force/torque (n)
        for joint velocity qd (n). The friction model includes:

        - Viscous friction which is a linear function of velocity.
        - Coulomb friction which is proportional to sign(qd).

        :param qd: The joint velocity
        :type qd: float

        :return tau: the friction force/torque
        :rtype tau: float

        :notes:
            - The friction value should be added to the motor output torque,
              it has a negative value when qd > 0.
            - The returned friction value is referred to the output of the
              gearbox.
            - The friction parameters in the Link object are referred to the
              motor.
            - Motor viscous friction is scaled up by G^2.
            - Motor Coulomb friction is scaled up by G.
            - The appropriate Coulomb friction value to use in the
              non-symmetric case depends on the sign of the joint velocity,
              not the motor velocity.
            - The absolute value of the gear ratio is used.  Negative gear
              ratios are tricky: the Puma560 has negative gear ratio for
              joints 1 and 3.

        """

        tau = self.B * np.abs(self.G) * qd

        if qd > 0:
            tau += self.Tc[0]
        elif qd < 0:
            tau += self.Tc[1]

        # Scale up by gear ratio
        tau = -np.abs(self.G) * tau

        return tau
