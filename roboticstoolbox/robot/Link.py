import copy
import numpy as np
from functools import wraps
from spatialmath.base.argcheck import getvector, \
    isscalar, isvector, ismatrix


def _listen_dyn(func):
    @wraps(func)
    def wrapper_listen_dyn(*args):
        if args[0]._robot is not None:
            args[0]._robot.dynchanged()
        return func(*args)
    return wrapper_listen_dyn


class Link:

    def __init__(
            self,
            name='',
            offset=0.0,
            qlim=np.zeros(2),
            flip=False,
            m=0.0,
            r=np.zeros((3,)),
            I=np.zeros((3, 3)),  # noqa
            Jm=0.0,
            B=0.0,
            Tc=np.zeros((2,)),
            G=1.0,
            mesh=None,
            **kwargs):

        self._robot = None  # reference to owning robot

        self._name = name

        self.offset = offset
        self.flip = flip
        self.qlim = qlim

        # TODO fix the path
        self.mesh = mesh

        # Dynamic Parameters
        self.m = m
        self.r = r
        self.I = I  # noqa
        self.Jm = Jm
        self.B = B
        self.Tc = Tc
        self.G = G

    def copy(self):
        """
        Copy of link object

        :return: Shallow copy of link object
        :rtype: Link
        """
        return copy.copy(self)

    def _copy(self):
        raise DeprecationWarning('Use copy method of Link class')

    def dyn(self, indent=0):
        """
        Show inertial properties of link

        s = dyn() returns a string representation the inertial properties of
        the link object in a multi-line format. The properties shown are mass,
        centre of mass, inertia, friction, gear ratio and motor properties.

        :param indent: indent each line by this many spaces
        :type indent: int
        :return s: The string representation of the link dynamics
        :rtype s: string
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

    def islimit(self, q):
        """
        Checks if the joint is exceeding a joint limit

        :return: True if joint is exceeded
        :rtype: bool

        :seealso: :func:`qlim`
        """

        if q < self.qlim[0] or q > self.qlim[1]:
            return True
        else:
            return False

    def nofriction(self, coulomb=True, viscous=False):
        """
        ``l2 = nofriction(coulomb, viscous)`` copies the link and returns a
        link with the same parameters except, the Coulomb and/or viscous
        friction parameter to zero.

        ``l2 = nofriction()`` as above except the the Coulomb parameter is set
        to zero.

        :param coulomb: if True, will set the Coulomb friction to 0
        :type coulomb: bool
        :param viscous: if True, will set the viscous friction to 0
        :type viscous: bool
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

        .. notes::

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
    def offset(self):
        """
        Get/set joint variable offset

        - ``link.offset`` is the joint variable offset
            :return: joint variable offset
            :rtype: float
        - ``link.offset = ...`` checks and sets the joint variable offset

        The offset is added to the joint angle before forward kinematics, and
        subtracted after inverse kinematics.  It is used to define the joint
        configuration for zero joint coordinates.

        """
        return self._offset

    @offset.setter
    def offset(self, offset_new):
        self._offset = offset_new

# -------------------------------------------------------------------------- #

    @property
    def qlim(self):
        """
        Get/set joint limits

        - ``link.qlim`` is the joint limits
            :return: joint limits
            :rtype: ndarray(2,)
        - ``link.a = ...`` checks and sets the joint limits

        .. note:: The limits are not widely enforced within the toolbox.

        :seealso: :func:`~islimit`
        """
        return self._qlim

    @qlim.setter
    def qlim(self, qlim_new):
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

        .. math::

            \begin{bmatrix} I_{xx} & I_{xy} & I_{xz} \\
                            I_{xy} & I_{yy} & I_{yz} \\
                            I_{xz} & I_{yz} & I_{zz} \end{bmatrix}

        and can be specified as either:

        - a :math:`3 \times 3` symmetric matrix
        - a 3-vector :math:`(I_{xx}, I_{yy}, I_{zz})`
        - a 6-vector :math:`(I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{yz}, I_{xz})`
        """
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

        .. note:: Viscous friction is the same for positive and negative
            motion.
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

        .. note:: :math:`\tau_C^+` must be :math:`> 0`, and :math:`\tau_C^-`
            must be :math:`< 0`.
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

        .. note:: The gear ratio can be negative.

        """
        return self._G

    @G.setter
    @_listen_dyn
    def G(self, G_new):
        self._G = G_new
