import copy
from abc import ABC
import numpy as np
from functools import wraps
from spatialmath.base import getvector, isscalar, isvector, ismatrix
from ansitable import ANSITable, Column


def _listen_dyn(func):
    """
    Decorator for property setters

    Use this decorator for any property setter that updates a parameter that
    affects the result of inverse dynamics.  This allows the C version of the
    parameters only having to be updated when they change, rather than on
    every call.  This decorator signals the change by invoking the
    ``dynchanged()`` method of the robot that owns the link.

    Example::

        @m.setter
        @_listen_dyn
        def m(self, m_new):
            self._m = m_new

    """
    @wraps(func)
    def wrapper_listen_dyn(*args):
        if args[0]._robot is not None:
            args[0]._robot.dynchanged()
        return func(*args)
    return wrapper_listen_dyn


class Link(ABC):
    """
    Link superclass

    :param name: name of the link
    :type name: str

    :param qlim: joint variable limits [min, max]
    :type qlim: float ndarray(1,2)
    :param flip: joint moves in opposite direction
    :type flip: bool

    :param m: dynamic - link mass
    :type m: float
    :param r: dynamic - position of COM with respect to link frame
    :type r:  float ndarray(3)
    :param I: dynamic - inertia of link with respect to COM
    :type I: ndarray

    :param Jm: dynamic - motor inertia
    :type Jm: float
    :param B: dynamic - motor viscous friction
    :type B: float, or ndarray(2,)
    :param Tc: dynamic - motor Coulomb friction [Tc⁺, Tc⁻]
    :type Tc: ndarray(2,)
    :param G: dynamic - gear ratio
    :type G: float

    An abstract link superclass for all link types.

    .. inheritance-diagram:: roboticstoolbox.RevoluteDH
        roboticstoolbox.PrismaticDH roboticstoolbox.RevoluteMDH
        roboticstoolbox.PrismaticMDH roboticstoolbox.ELink
        :top-classes: roboticstoolbox.robot.Link
        :parts: 2

    It holds metadata related to:

    - a robot link, such as rigid-body inertial parameters defined in the link
      frame, and link name
    - a robot joint, that connects this link to its parent, such as joint
      limits, direction of motion, motor and transmission parameters.

    .. note::
        - For a more sophisticated actuator model use the ``actuator``
          attribute which is not initialized or used by this Toolbox.
        - There is no ability to name a joint as supported by URDF

    """

    def __init__(
            self,
            name=None,
            qlim=None,
            flip=False,
            m=0.0,
            r=np.zeros((3,)),
            I=np.zeros((3, 3)),  # noqa
            Jm=0.0,
            B=0.0,
            Tc=np.zeros((2,)),
            G=0.0,
            mesh=None,
            geometry=[],
            collision=[],
            **kwargs):

        self._robot = None  # reference to owning robot

        self._name = name

        # Joint angle parameters
        self.flip = flip
        self.qlim = qlim

        # Link geometry
        self.geometry = geometry
        self.collision = collision

        # TODO this is a leftover from VPython development, should be
        # using geometry instead
        self.mesh = mesh

        # Link dynamic Parameters
        self.m = m
        self.r = r
        self.I = I  # noqa

        # Motor dynamic parameters
        self.Jm = Jm
        self.B = B
        self.Tc = Tc
        self.G = G
        self.actuator = None  # reference to more advanced actuator model

    def copy(self):
        """
        Copy of link object

        :return: Shallow copy of link object
        :rtype: Link

        ``link.copy()`` is a new Link subclass instance with a copy of all
        the parameters.
        """
        new = copy.copy(self)
        for k, v in self.__dict__.items():
            if k.startswith('_') and isinstance(v, np.ndarray):
                setattr(new, k, np.copy(v))
        return new

    def _copy(self):
        raise DeprecationWarning('Use copy method of Link class')

    def dyn(self, indent=0):
        """
        Inertial properties of link as a string

        :param indent: indent each line by this many spaces
        :type indent: int
        :return: The string representation of the link dynamics
        :rtype: string

        ``link.dyn()`` is a string representation the inertial properties of
        the link object in a multi-line format. The properties shown are mass,
        centre of mass, inertia, friction, gear ratio and motor properties.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> print(robot.links[2])        # kinematic parameters
            >>> print(robot.links[2].dyn())  # dynamic parameters

        :seealso: :func:`~dyntable`
        """

        s = "m     =  {:8.2g} \n" \
            "r     =  {:8.2g} {:8.2g} {:8.2g} \n" \
            "        | {:8.2g} {:8.2g} {:8.2g} | \n" \
            "I     = | {:8.2g} {:8.2g} {:8.2g} | \n" \
            "        | {:8.2g} {:8.2g} {:8.2g} | \n" \
            "Jm    =  {:8.2g} \n" \
            "B     =  {:8.2g} \n" \
            "Tc    =  {:8.2g}(+) {:8.2g}(-) \n" \
            "G     =  {:8.2g} \n" \
            "qlim  =  {:8.2g} to {:8.2g}".format(
                self.m,
                self.r[0], self.r[1], self.r[2],
                self.I[0, 0], self.I[0, 1], self.I[0, 2],
                self.I[1, 0], self.I[1, 1], self.I[1, 2],
                self.I[2, 0], self.I[2, 1], self.I[2, 2],
                self.Jm,
                self.B,
                self.Tc[0], self.Tc[1],
                self.G,
                self.qlim[0], self.qlim[1]
            )

        if indent > 0:
            # insert indentations into the string
            # TODO there is probably a tidier way to integrate this step with
            # above
            sp = ' ' * indent
            s = sp + s.replace('\n', '\n' + sp)

        return s

    def _dyn2list(self, fmt="{: .3g}"):
        """
        Inertial properties of link as a string

        :param fmt: conversion format for each number
        :type fmt: str
        :return: The string representation of the link dynamics
        :rtype: string

        ``link.)_dyn2list()`` returns a list of pretty-printed inertial
        properties of the link The properties included are mass, centre of
        mass, inertia, friction, gear ratio and motor properties.

        :seealso: :func:`~dyn`
        """
        ANSITable(
            Column("Parameter", headalign="^"),
            Column("Value", headalign="^", colalign="<"),
            border="thin")

        def format(l, fmt, val):  # noqa
            if isinstance(val, np.ndarray):
                s = ', '.join([fmt.format(v) for v in val])
            else:
                s = fmt.format(val)
            l.append(s)

        dyn = []
        format(dyn, fmt, self.m)
        format(dyn, fmt, self.r)
        I = self.I.flatten()  # noqa
        format(dyn, fmt, np.r_[[I[k] for k in [0, 4, 8, 1, 2, 5]]])
        format(dyn, fmt, self.Jm)
        format(dyn, fmt, self.B)
        format(dyn, fmt, self.Tc)
        format(dyn, fmt, self.G)

        return dyn

    def _format(self, l, name, ignorevalue=0, indices=None):  # noqa  # pragma nocover
        v = getattr(self, name)
        s = None
        if v is None:
            return
        if isscalar(v) and v != ignorevalue:
            s = f"{name}={v}"
        elif isinstance(v, np.ndarray):
            if np.linalg.norm(v, ord=np.inf) > 0:
                if indices is not None:
                    flat = v.flatten()
                    v = np.r_[[flat[k] for k in indices]]
                s = f"{name}=[" + ", ".join([str(x) for x in v]) + "]"
        if s is not None:
            l.append(s)

    def _params(self):  # pragma nocover
        l = []  # noqa
        self._format(l, "name")
        self._format(l, "flip", False)
        self._format(l, "qlim")
        self._format(l, "m")
        self._format(l, "r")
        self._format(l, "I", indices=[0, 4, 8, 1, 2, 5])
        self._format(l, "Jm")
        self._format(l, "B")
        self._format(l, "Tc")
        self._format(l, "G")

        return l

    def islimit(self, q):
        """
        Checks if joint exceeds limit

        :param q: joint coordinate
        :type q: float
        :return: True if joint is exceeded
        :rtype: bool

        ``link.islimit(q)`` is True if ``q`` exceeds the joint limits defined
        by ``link``.

        .. note:: If no limits are set always return False.

        :seealso: :func:`qlim`
        """
        if self.qlim is None:
            return False
        else:
            return q < self.qlim[0] or q > self.qlim[1]

    def nofriction(self, coulomb=True, viscous=False):
        """
        Clone link without friction

        :param coulomb: if True, will set the Coulomb friction to 0
        :type coulomb: bool
        :param viscous: if True, will set the viscous friction to 0
        :type viscous: bool

        ``link.nofriction()`` is a copy of the link instance with the same
        parameters except, the Coulomb and/or viscous friction parameters are
        set to zero.

        .. note:: For simulation it can be useful to remove Couloumb friction
            which can cause problems for numerical integration.

        """

        # Copy the Link
        link = self.copy()

        if viscous:
            link.B = 0.0

        if coulomb:
            link.Tc = [0.0, 0.0]

        return link

    def friction(self, qd, coulomb=True):
        r"""
        Compute joint friction

        :param qd: The joint velocity
        :type qd: float
        :param coulomb: include Coulomb friction
        :type coloumb: bool, default True
        :return tau: the friction force/torque
        :rtype tau: float

        ``friction(qd)`` is the joint friction force/torque
        for joint velocity ``qd``. The friction model includes:

        - Viscous friction which is a linear function of velocity.
        - Coulomb friction which is proportional to sign(qd).

        .. math::

            \tau = G^2 B \dot{q} + |G| \left\{ \begin{array}{ll}
                \tau_C^+ & \mbox{if $\dot{q} > 0$} \\
                \tau_C^- & \mbox{if $\dot{q} < 0$} \end{array} \right.

        .. note::

            - The friction value should be added to the motor output torque to
              determine the nett torque. It has a negative value when qd > 0.
            - The returned friction value is referred to the output of the
              gearbox.
            - The friction parameters in the Link object are referred to the
              motor.
            - Motor viscous friction is scaled up by :math:`G^2`.
            - Motor Coulomb friction is scaled up by math:`G`.
            - The appropriate Coulomb friction value to use in the
              non-symmetric case depends on the sign of the joint velocity,
              not the motor velocity.
            - Coulomb friction is zero for zero joint velocity, stiction is
              not modeled.
            - The absolute value of the gear ratio is used.  Negative gear
              ratios are tricky: the Puma560 robot has negative gear ratio for
              joints 1 and 3.

        """

        tau = self.B * np.abs(self.G) * qd

        if coulomb:
            if qd > 0:
                tau += self.Tc[0]
            elif qd < 0:
                tau += self.Tc[1]

        # Scale up by gear ratio
        tau = -np.abs(self.G) * tau

        return tau

# -------------------------------------------------------------------------- #

    @property
    def name(self):
        """
        Get/set link name

        - ``link.name`` is the link name

        :return: link name
        :rtype: str

        - ``link.name = ...`` checks and sets the link name
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

# -------------------------------------------------------------------------- #

    @property
    def qlim(self):
        """
        Get/set joint limits

        - ``link.qlim`` is the joint limits

        :return: joint limits
        :rtype: ndarray(2,) or Nine

        - ``link.qlim = ...`` checks and sets the joint limits

        .. note::
            - The limits are not widely enforced within the toolbox.
            - If no joint limits are specified the value is ``None``

        :seealso: :func:`~islimit`
        """
        return self._qlim

    @qlim.setter
    def qlim(self, qlim_new):
        if qlim_new is None:
            self._qlim = None
        else:
            self._qlim = getvector(qlim_new, 2)

# -------------------------------------------------------------------------- #

    @property
    def flip(self):
        """
        Get/set joint flip

        - ``link.flip`` is the joint flip status

        :return: joint flip
        :rtype: bool

        - ``link.flip = ...`` checks and sets the joint flip status

        Joint flip defines the direction of motion of the joint.

        ``flip = False`` is conventional motion direction:

            - revolute motion is a positive rotation about the z-axis
            - prismatic motion is a positive translation along the z-axis

        ``flip = True`` is the opposite motion direction:

            - revolute motion is a negative rotation about the z-axis
            - prismatic motion is a negative translation along the z-axis

        """
        return self._flip

    @flip.setter
    def flip(self, flip_new):
        self._flip = flip_new

# -------------------------------------------------------------------------- #

    @property
    def m(self):
        """
        Get/set link mass

        - ``link.m`` is the link mass

        :return: link mass
        :rtype: float

        - ``link.m = ...`` checks and sets the link mass

        """
        return self._m

    @m.setter
    @_listen_dyn
    def m(self, m_new):
        self._m = m_new

# -------------------------------------------------------------------------- #

    @property
    def r(self):
        """
        Get/set link centre of mass

        - ``link.r`` is the link centre of mass

        :return: link centre of mass
        :rtype: ndarray(3,)

        - ``link.r = ...`` checks and sets the link centre of mass

        The link centre of mass is a 3-vector defined with respect to the link
        frame.
        """
        return self._r

    @r.setter
    @_listen_dyn
    def r(self, r_new):
        self._r = getvector(r_new, 3)

# -------------------------------------------------------------------------- #

    @property
    def I(self):    # noqa
        r"""
        Get/set link inertia

        - ``link.I`` is the link inertia

        :return: link inertia
        :rtype: ndarray(3,3)

        - ``link.I = ...`` checks and sets the link inertia

        Link inertia is a symmetric 3x3 matrix describing the inertia with
        respect to a frame with its origin at the centre of mass, and with
        axes parallel to those of the link frame.

        The inertia matrix is

        :math:`\begin{bmatrix} I_{xx} & I_{xy} & I_{xz} \\ I_{xy} & I_{yy} & I_{yz} \\I_{xz} & I_{yz} & I_{zz} \end{bmatrix}`


        and can be specified as either:

        - a 3 ⨉ 3 symmetric matrix
        - a 3-vector :math:`(I_{xx}, I_{yy}, I_{zz})`
        - a 6-vector :math:`(I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{yz}, I_{xz})`

        .. note:: Referred to the link side of the gearbox.
        """  # noqa
        return self._I

    @I.setter
    @_listen_dyn
    def I(self, I_new):  # noqa

        if ismatrix(I_new, (3, 3)):
            # 3x3 matrix passed
            if np.any(np.abs(I_new - I_new.T) > 1e-8):
                raise ValueError('3x3 matrix is not symmetric')

        elif isvector(I_new, 9):
            # 3x3 matrix passed as a 1d vector
            I_new = I_new.reshape(3, 3)
            if np.any(np.abs(I_new - I_new.T) > 1e-8):
                raise ValueError('3x3 matrix is not symmetric')

        elif isvector(I_new, 6):
            # 6-vector passed, moments and products of inertia,
            # [Ixx Iyy Izz Ixy Iyz Ixz]
            I_new = np.array([
                [I_new[0], I_new[3], I_new[5]],
                [I_new[3], I_new[1], I_new[4]],
                [I_new[5], I_new[4], I_new[2]]
            ])

        elif isvector(I_new, 3):
            # 3-vector passed, moments of inertia [Ixx Iyy Izz]
            I_new = np.diag(I_new)

        else:
            raise ValueError('invalid shape passed: must be (3,3), (6,), (3,)')

        self._I = I_new

# -------------------------------------------------------------------------- #

    @property
    def Jm(self):
        """
        Get/set motor inertia

        - ``link.Jm`` is the motor inertia

        :return: motor inertia
        :rtype: float

        - ``link.Jm = ...`` checks and sets the motor inertia

        .. note:: Referred to the motor side of the gearbox.
        """
        return self._Jm

    @Jm.setter
    @_listen_dyn
    def Jm(self, Jm_new):
        self._Jm = Jm_new

# -------------------------------------------------------------------------- #

    @property
    def B(self):
        """
        Get/set motor viscous friction

        - ``link.B`` is the motor viscous friction

        :return: motor viscous friction
        :rtype: float

        - ``link.B = ...`` checks and sets the motor viscous friction

        .. note::
            - Referred to the motor side of the gearbox.
            - Viscous friction is the same for positive and negative motion.
        """
        return self._B

    @B.setter
    @_listen_dyn
    def B(self, B_new):
        if isscalar(B_new):
            self._B = B_new
        else:
            raise TypeError("B must be a scalar")

# -------------------------------------------------------------------------- #

    @property
    def Tc(self):
        r"""
        Get/set motor Coulomb friction

        - ``link.Tc`` is the motor Coulomb friction

        :return: motor Coulomb friction
        :rtype: ndarray(2)

        - ``link.Tc = ...`` checks and sets the motor Coulomb friction. If a
          scalar is given the value is set to [T, -T], if a 2-vector it is
          assumed to be in the order [Tc⁺, Tc⁻]

        Coulomb friction is a non-linear friction effect defined by two
        parameters such that

        .. math::

            \tau = \left\{ \begin{array}{ll}
                \tau_C^+ & \mbox{if $\dot{q} > 0$} \\
                \tau_C^- & \mbox{if $\dot{q} < 0$} \end{array} \right.

        .. note::
            -  Referred to the motor side of the gearbox.
            - :math:`\tau_C^+` must be :math:`> 0`, and :math:`\tau_C^-` must
              be :math:`< 0`.
        """
        return self._Tc

    @Tc.setter
    @_listen_dyn
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

# -------------------------------------------------------------------------- #

    @property
    def G(self):
        """
        Get/set gear ratio

        - ``link.G`` is the transmission gear ratio

        :return: gear ratio
        :rtype: float

        - ``link.G = ...`` checks and sets the gear ratio

        .. note::
            - The ratio of motor motion : link motion
            - The gear ratio can be negative, see also the ``flip`` attribute.

        :seealso: :func:`flip`
        """
        return self._G

    @G.setter
    @_listen_dyn
    def G(self, G_new):
        self._G = G_new

# -------------------------------------------------------------------------- #

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


if __name__ == "__main__":  # pragma nocover

    import roboticstoolbox as rtb
    from spatialmath.base import sym

    ET = rtb.ETS

    d, a, theta = sym.symbol('d, a, theta')

    e = ET.tx(d) * ET.ty(a) * ET.rz(theta)
    print(e)
    link = rtb.ELink()
