#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

# import numpy as np
# from spatialmath import SE3
import roboticstoolbox as rp
from roboticstoolbox.robot.Link import Link, _dirties_frne, _copy_shapes
from roboticstoolbox.ets.ETS import ETS
from roboticstoolbox.ets.ET import ET
from spatialmath import SE3
from functools import wraps
from numpy import ndarray, cos, sin, array
from copy import deepcopy
from roboticstoolbox.tools.types import ArrayLike


def _check_rne(func):
    """
    @_check_rne decorator

    Decorator applied to any method to calls to C RNE code.  Works in
    conjunction with::

        @_dirties_frne
        def dyn_param_setter(self, value):

    which marks the dynamic parameters as having changed using the robot's
    ``.dynchanged()`` method.

    If this is the case, then the parameters are re-serialized prior to
    invoking inverse dynamics.

    :seealso: :func:`Link._listen_dyn`
    """

    @wraps(func)
    def wrapper_check_rne(*args, **kwargs):
        if args[0]._frne is None or args[0]._frne_stale:
            args[0]._copy_to_cpp()
        return func(*args, **kwargs)

    return wrapper_check_rne


# --------------------------------------------------------------#

try:  # pragma: no cover
    # print('Using SymPy')
    import sympy as sym

    def _issymbol(x):  # type: ignore
        return isinstance(x, sym.Expr)

except ImportError:

    def _issymbol(x):  # pylint: disable=unused-argument
        return False


def _cos(theta) -> float:
    if _issymbol(theta):
        return sym.cos(theta)  # type: ignore
    else:
        return cos(theta)


def _sin(theta) -> float:
    if _issymbol(theta):
        return sym.sin(theta)  # type: ignore
    else:
        return sin(theta)


# --------------------------------------------------------------#


class DHLink(Link):
    """
    A link superclass for all robots defined using Denavit-Hartenberg notation.
    A Link object holds all information related to a robot joint and link such
    as kinematics parameters, rigid-body inertial parameters, motor and
    transmission parameters.

    :param theta: kinematic: joint angle
    :param d: kinematic - link offset
    :param alpha: kinematic - link twist
    :param a: kinematic - link length
    :param sigma: kinematic - 0 if revolute, 1 if prismatic
    :param mdh: kinematic - False if standard D&H, True for modified D&H
    :param offset: kinematic - joint variable offset
    :param qlim: joint variable limits [min, max]
    :type qlim: ndarray(2,)
    :param flip: joint moves in opposite direction

    :param m: dynamic - link mass
    :type m: float
    :param r: dynamic - position of COM with respect to link frame
    :type r: ndarray(3,)
    :param I: dynamic - inertia of link with respect to COM
    :type I: ndarray
    :param Jm: dynamic - motor inertia
    :type Jm: float
    :param B: dynamic - motor viscous friction: B=B⁺=B⁻, [B⁺, B⁻]
    :type B: float, or ndarray(2,)
    :param Tc: dynamic - motor Coulomb friction [Tc⁺, Tc⁻]
    :type Tc: ndarray(2,)
    :param G: dynamic - gear ratio
    :type G: float

    :references:
        - Robotics, Vision & Control, P. Corke, Springer 2023, Chap 7.

    """

    def __init__(
        self,
        d: float = 0.0,
        alpha: float = 0.0,
        theta: float = 0.0,
        a: float = 0.0,
        sigma: int = 0,
        mdh: bool = False,
        offset: float = 0,
        flip: bool = False,
        qlim: ArrayLike | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        ets = self._to_ets(sigma, theta, d, alpha, a, offset, flip, mdh)
        self._ets = ets

        # Set the variable et
        for et in ets:
            if et.isjoint:
                self._v = et
                break

        # Set the qlim if provided now we have an ETS
        if qlim is not None and self.v:
            self.v.qlim = qlim

        # DH Kinematic parameters
        self.sigma = sigma
        self.theta = theta
        self.d = d
        self.alpha = alpha
        self.a = a
        self.mdh = mdh
        self.offset = offset
        self.id = None
        self.mesh = None
        self.number = None

    def _to_ets(self, sigma: int, theta: float, d: float, alpha: float, a: float, offset: float, flip: bool, mdh: bool) -> ETS:
        ets = ETS()

        isrevolute = False if sigma else True

        # MDH format: a alpha theta d
        if mdh:
            if a != 0:
                ets *= ET.tx(a)
            if alpha != 0:
                ets *= ET.Rx(alpha)

            if isrevolute:
                if offset != 0:
                    ets *= ET.Rz(offset)

                # Swapped d and theta: will make no difference to the transform
                # But makes ets end with variable which is needed
                if d != 0:
                    ets *= ET.tz(d)

                ets *= ET.Rz(flip=flip)  # joint
            else:
                if theta != 0:
                    ets *= ET.Rz(theta)

                if offset != 0:
                    ets *= ET.tz(offset)

                ets *= ET.tz(flip=flip)  # joint
        else:
            # DH format: theta d a alpha
            if isrevolute:
                if offset != 0:
                    ets *= ET.Rz(offset)
                ets *= ET.Rz(flip=flip)

                if d != 0:
                    ets *= ET.tz(d)
            else:
                if theta != 0:
                    ets *= ET.Rz(theta)

                if offset != 0:
                    ets *= ET.tz(offset)
                ets *= ET.tz(flip=flip)

            if a != 0:
                ets *= ET.tx(a)
            if alpha != 0:
                ets *= ET.Rx(alpha)

        return ets

    @property
    def isjoint(self) -> bool:
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
        return True

    # @classmethod
    # def StandardDH(cls, links: List["StandardDH"]) -> List["DHLink"]:
    #     """
    #     Takes a list of standard DH links and converts to a list
    #     of modified DH parameters

    #     """

    #     new_links = []
    #     ets = ETS()

    #     # Make an ets of the whole Robot
    #     for link in links:
    #         ets += link.ets

    #     # Split ETS with variables at the end
    #     segs = ets.split()

    #     # Construct MDH Links
    #     for (seg, link) in zip(segs, links):
    #         flip = True if seg[-1].isflip else False
    #         isrevolute = True if seg[-1].axis[0] == "R" else False

    #         a = 0.0
    #         alpha = 0.0
    #         theta = 0.0
    #         d = 0.0
    #         offset = 0.0

    #         # Find MDH Parameters
    #         for et in seg:
    #             if et.axis == "tx":
    #                 a = et.eta
    #             elif et.axis == "Rx":
    #                 alpha = et.eta
    #             elif et.axis == "Rz" and not et.isjoint:
    #                 offset = et.eta
    #             elif et.axis == "Rz" and not isrevolute:
    #                 theta = et.eta
    #             elif et.axis == "tz" and isrevolute:
    #                 d = et.eta

    #         # Make the link
    #         if isrevolute:
    #             new_link = RevoluteMDH(
    #                 d=d,
    #                 a=a,
    #                 alpha=alpha,
    #                 offset=offset,
    #                 qlim=link.qlim,
    #                 flip=flip,
    #                 name=link.name,
    #                 m=link.m,
    #                 r=link.r,
    #                 I=link.I,
    #                 Jm=link.Jm,
    #                 B=link.B,
    #                 Tc=link.Tc,
    #                 G=link.G,
    #                 geometry=link.geometry,
    #                 collision=link.collision,
    #             )
    #         else:
    #             new_link = PrismaticMDH(
    #                 theta=theta,
    #                 a=a,
    #                 alpha=alpha,
    #                 offset=offset,
    #                 qlim=link.qlim,
    #                 flip=flip,
    #                 name=link.name,
    #                 m=link.m,
    #                 r=link.r,
    #                 I=link.I,
    #                 Jm=link.Jm,
    #                 B=link.B,
    #                 Tc=link.Tc,
    #                 G=link.G,
    #                 geometry=link.geometry,
    #                 collision=link.collision,
    #             )

    #         new_links.append(new_link)

    #     return new_links

    def __add__(self, L):
        if isinstance(L, DHLink):
            return rp.DHRobot([self, L])

        elif isinstance(L, rp.DHRobot):
            nlinks: list[DHLink] = [self]

            # TODO - Should I do a deep copy here a physically copy the Links
            # and not just the references?
            # Copy Link references to new list
            for i in range(L.n):
                nlinks.append(L.links[i])

            return rp.DHRobot(
                nlinks,
                name=L.name,
                manufacturer=L.manufacturer,
                base=L.base,
                tool=L.tool,
                gravity=L.gravity,
            )

        else:
            raise TypeError("Cannot add a Link with a non Link object")

    def __str__(self):

        s = ""

        if self.offset == 0:
            offset = ""
        else:
            offset = f" + {self.offset}"
        if self.id is None:
            qvar = "q"
        else:
            qvar = f"q{self.id}"
        cls = self.__class__.__name__
        if self.isrevolute:
            s = f"{cls}:   θ={qvar}{offset},  d={self.d},  a={self.a},  ⍺={self.alpha}"
        elif self.isprismatic:
            s = (
                f"{cls}:  θ={self.theta},  d={qvar}{offset}, "
                f" a={self.a},  "
                f"⍺={self.alpha}"
            )
        return s

    def __repr__(self):
        name = self.__class__.__name__
        args = []
        if self.isrevolute:
            self._format_param(args, "d")
        else:
            self._format_param(args, "theta", "θ")
        self._format_param(args, "a")
        self._format_param(args, "alpha", "⍺")
        args.extend(super()._params())
        return name + "(" + ", ".join(args) + ")"

    def __deepcopy__(self, memo):
        kwargs = {
            "name": deepcopy(self.name),
            "joint_name": deepcopy(self._joint_name),
            "m": deepcopy(self.m),
            "r": deepcopy(self.r),
            "I": deepcopy(self.I),
            "Jm": deepcopy(self.Jm),
            "B": deepcopy(self.B),
            "Tc": deepcopy(self.Tc),
            "G": deepcopy(self.G),
            "qlim": deepcopy(self.qlim),
            "geometry": _copy_shapes(self._geometry, self.name),
            "collision": _copy_shapes(self._collision, self.name),
            "d": deepcopy(self.d),
            "alpha": deepcopy(self.alpha),
            "theta": deepcopy(self.theta),
            "a": deepcopy(self.a),
            "sigma": deepcopy(self.sigma),
            "mdh": deepcopy(self.mdh),
            "offset": deepcopy(self.offset),
            "flip": deepcopy(self.isflip),
        }

        cls = self.__class__

        if "Revolute" in str(cls):
            del kwargs["theta"]
            del kwargs["sigma"]
            del kwargs["mdh"]
        elif "Prismatic" in str(cls):
            del kwargs["d"]
            del kwargs["sigma"]
            del kwargs["mdh"]

        result = cls(**kwargs)
        result._robot = self.robot
        result.sigma = self.sigma
        result.number = self.number

        memo[id(self)] = result
        return result

    # -------------------------------------------------------------------------- #

    @property
    def theta(self):
        """
        Get/set joint angle.

        :returns: joint angle
        :rtype: float

        - ``link.theta`` is the joint angle
        - ``link.theta = ...`` checks and sets the joint angle
        """
        return self._theta

    @theta.setter
    @_dirties_frne
    def theta(self, theta_new):
        if not self.sigma and theta_new != 0.0:
            raise ValueError("theta is not valid for revolute joints")
        else:
            self._theta = theta_new

    # -------------------------------------------------------------------------- #

    @property
    def d(self):
        """
        Get/set link offset.

        :returns: link offset
        :rtype: float

        - ``link.d`` is the link offset
        - ``link.d = ...`` checks and sets the link offset
        """
        return self._d

    @d.setter
    @_dirties_frne
    def d(self, d_new):
        if self.sigma and d_new != 0.0:
            raise ValueError("f is not valid for prismatic joints")
        else:
            self._d = d_new

    # -------------------------------------------------------------------------- #

    @property
    def a(self):
        """
        Get/set link length.

        :returns: link length
        :rtype: float

        - ``link.a`` is the link length
        - ``link.a = ...`` checks and sets the link length
        """
        return self._a

    @a.setter
    @_dirties_frne
    def a(self, a_new):
        self._a = a_new

    # -------------------------------------------------------------------------- #

    @property
    def alpha(self):
        """
        Get/set link twist.

        :returns: link twist
        :rtype: float

        - ``link.alpha`` is the link twist
        - ``link.alpha = ...`` checks and sets the link twist
        """
        return self._alpha

    @alpha.setter
    @_dirties_frne
    def alpha(self, alpha_new):
        self._alpha = alpha_new

    # -------------------------------------------------------------------------- #

    @property
    def sigma(self):
        """
        Get/set joint type.

        :returns: joint type
        :rtype: int
        :seealso: :func:`isrevolute`, :func:`isprismatic`

        - ``link.sigma`` is the joint type
        - ``link.sigma = ...`` checks and sets the joint type

        The joint type is 0 for a revolute joint, and 1 for a prismatic joint.
        """
        return self._sigma

    @sigma.setter
    @_dirties_frne
    def sigma(self, sigma_new):
        self._sigma = sigma_new

    # -------------------------------------------------------------------------- #

    @property
    def mdh(self):
        """
        Get/set kinematic convention.

        :returns: kinematic convention
        :rtype: bool

        - ``link.mdh`` is the kinematic convention
        - ``link.mdh = ...`` checks and sets the kinematic convention

        True for modified Denavit-Hartenberg notation (eg. Craig's textbook),
        False for standard Denavit-Hartenberg notation (eg. Siciliano, Spong, Paul).
        """
        return self._mdh

    @mdh.setter
    @_dirties_frne
    def mdh(self, mdh_new):
        self._mdh = int(mdh_new)

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

    def A(self, q: float) -> SE3:
        r"""
        Link transform matrix

        :param q: joint coordinate
        :returns: SE(3) link homogeneous transformation
        :rtype: SE3

        ``A(q)`` is an ``SE3`` instance representing the SE(3) homogeneous
        transformation matrix corresponding to the link's joint variable ``q``
        which is either the Denavit-Hartenberg parameter :math:`\theta_j`
        (revolute) or :math:`d_j` (prismatic).

        This is the relative pose of the current link frame with respect to the
        previous link frame.

        For details of the computation see the documentation for the
        subclasses, click on the right side of the class boxes below.

        .. inheritance-diagram:: roboticstoolbox.RevoluteDH
            roboticstoolbox.PrismaticDH roboticstoolbox.RevoluteMDH
            roboticstoolbox.PrismaticMDH
            :top-classes: roboticstoolbox.robot.DHLink.DHLink
            :parts: 2

        .. note::

            - For a revolute joint the ``theta`` parameter of the link is
              ignored, and ``q`` used instead.
            - For a prismatic joint the ``d`` parameter of the link is ignored,
              and ``q`` used instead.
            - The joint ``offset`` parameter is added to ``q`` before
              computation of the transformation matrix.
            - The computation is different for standard and modified
              Denavit-Hartenberg parameters.

        :seealso: :class:`RevoluteDH`, :class:`PrismaticDH`,
            :class:`RevoluteMDH`, :class:`PrismaticMDH`
        """

        sa = _sin(self.alpha)
        ca = _cos(self.alpha)

        if self.ets[-1].isflip:
            q = -q + self.offset
        else:
            q = q + self.offset

        if self.sigma == 0:
            # revolute
            st = _sin(q)
            ct = _cos(q)
            d = self.d
        else:
            # prismatic
            st = _sin(self.theta)
            ct = _cos(self.theta)
            d = q

        if self.mdh == 0:
            # standard DH
            T = array(
                [
                    [ct, -st * ca, st * sa, self.a * ct],
                    [st, ct * ca, -ct * sa, self.a * st],
                    [0, sa, ca, d],
                    [0, 0, 0, 1],
                ]
            )
        else:
            # modified DH
            T = array(
                [
                    [ct, -st, 0, self.a],
                    [st * ca, ct * ca, -sa, -sa * d],
                    [st * sa, ct * sa, ca, ca * d],
                    [0, 0, 0, 1],
                ]
            )

        return SE3(T, check=False)

    @property
    def isrevolute(self):
        """
        Checks if the joint is of revolute type

        :return: True if revolute
        :rtype: bool

        :seealso: :func:`sigma`
        """

        if not self.sigma:
            return True
        else:
            return False

    @property
    def isprismatic(self):
        """
        Checks if the joint is of prismatic type

        :return: True if prismatic
        :rtype: bool

        :seealso: :func:`sigma`
        """

        if self.sigma:
            return True
        else:
            return False


# -------------------------------------------------------------------------- #


class RevoluteDH(DHLink):
    r"""
    Class for revolute links using standard DH convention.

    :param d: kinematic - link offset
    :param alpha: kinematic - link twist
    :param a: kinematic - link length
    :param offset: kinematic - joint variable offset
    :param qlim: joint variable limits [min, max]
    :type qlim: ndarray(2,)
    :param flip: joint moves in opposite direction

    :param m: dynamic - link mass
    :type m: float
    :param r: dynamic - position of COM with respect to link frame
    :type r: ndarray(3,)
    :param I: dynamic - inertia of link with respect to COM
    :type I: ndarray
    :param Jm: dynamic - motor inertia
    :type Jm: float
    :param B: dynamic - motor viscous friction: B=B⁺=B⁻, [B⁺, B⁻]
    :type B: float, or ndarray(2,)
    :param Tc: dynamic - motor Coulomb friction [Tc⁺, Tc⁻]
    :type Tc: ndarray(2,)
    :param G: dynamic - gear ratio
    :type G: float

    A subclass of the :class:`DHLink` class for a revolute joint that holds all
    information related to a robot link such as kinematics parameters,
    rigid-body inertial parameters, motor and transmission parameters.
    The link transform is
    :math:`\underbrace{\mathbf{T}_{rz}(q_i)}_{\mbox{variable}} \cdot \mathbf{T}_{tz}(d_i) \cdot \mathbf{T}_{tx}(a_i) \cdot \mathbf{T}_{rx}(\alpha_i)`
    where :math:`q_i` is the joint variable.

    :references:
        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 7.

    :seealso: :func:`PrismaticDH`, :func:`DHLink`, :func:`RevoluteMDH`
    """

    def __init__(
        self, d: float = 0.0, a: float = 0.0, alpha: float = 0.0, offset: float = 0.0,
        qlim: ArrayLike | None = None, flip: bool = False, **kwargs
    ):

        theta = 0.0
        sigma = 0
        mdh = False

        super().__init__(
            d=d,
            alpha=alpha,
            theta=theta,
            a=a,
            sigma=sigma,
            mdh=mdh,
            offset=offset,
            qlim=qlim,
            flip=flip,
            **kwargs,
        )


class PrismaticDH(DHLink):
    r"""
    Class for prismatic link using standard DH convention.

    :param theta: kinematic: joint angle
    :param alpha: kinematic - link twist
    :param a: kinematic - link length
    :param offset: kinematic - joint variable offset
    :param qlim: joint variable limits [min, max]
    :type qlim: ndarray(2,)
    :param flip: joint moves in opposite direction

    :param m: dynamic - link mass
    :type m: float
    :param r: dynamic - position of COM with respect to link frame
    :type r: ndarray(3,)
    :param I: dynamic - inertia of link with respect to COM
    :type I: ndarray
    :param Jm: dynamic - motor inertia
    :type Jm: float
    :param B: dynamic - motor viscous friction: B=B⁺=B⁻, [B⁺, B⁻]
    :type B: float, or ndarray(2,)
    :param Tc: dynamic - motor Coulomb friction [Tc⁺, Tc⁻]
    :type Tc: ndarray(2,)
    :param G: dynamic - gear ratio
    :type G: float

    A subclass of the DHLink class for a prismatic joint that holds all
    information related to a robot link such as kinematics parameters,
    rigid-body inertial parameters, motor and transmission parameters.
    The link transform is
    :math:`\mathbf{T}_{rz}(\theta_i) \cdot \underbrace{\mathbf{T}_{tz}(q_i)}_{\mbox{variable}} \cdot \mathbf{T}_{tx}(a_i) \cdot \mathbf{T}_{rx}(\alpha_i)`
    where :math:`q_i` is the joint variable.

    :references:
        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 7.

    :seealso: :func:`RevoluteDH`, :func:`DHLink`, :func:`PrismaticMDH`
    """

    def __init__(
        self, theta: float = 0.0, a: float = 0.0, alpha: float = 0.0,
        offset: float = 0.0, qlim: ArrayLike | None = None, flip: bool = False, **kwargs
    ):

        d = 0.0
        sigma = 1
        mdh = False

        super().__init__(
            theta=theta,
            d=d,
            a=a,
            alpha=alpha,
            sigma=sigma,
            mdh=mdh,
            offset=offset,
            qlim=qlim,
            flip=flip,
            **kwargs,
        )


class RevoluteMDH(DHLink):
    r"""
    Class for revolute links using modified DH convention.

    :param d: kinematic - link offset
    :param alpha: kinematic - link twist
    :param a: kinematic - link length
    :param offset: kinematic - joint variable offset
    :param qlim: joint variable limits [min, max]
    :type qlim: ndarray(2,)
    :param flip: joint moves in opposite direction

    :param m: dynamic - link mass
    :type m: float
    :param r: dynamic - position of COM with respect to link frame
    :type r: ndarray(3,)
    :param I: dynamic - inertia of link with respect to COM
    :type I: ndarray
    :param Jm: dynamic - motor inertia
    :type Jm: float
    :param B: dynamic - motor viscous friction: B=B⁺=B⁻, [B⁺, B⁻]
    :type B: float, or ndarray(2,)
    :param Tc: dynamic - motor Coulomb friction [Tc⁺, Tc⁻]
    :type Tc: ndarray(2,)
    :param G: dynamic - gear ratio
    :type G: float

    A subclass of the DHLink class for a revolute joint that holds all
    information related to a robot link such as kinematics parameters,
    rigid-body inertial parameters, motor and transmission parameters.

    The link transform is

    :math:`\mathbf{T}_{tx}(a_{i-1}) \cdot \mathbf{T}_{rx}(\alpha_{i-1}) \cdot \underbrace{\mathbf{T}_{rz}(q_i)}_{\mbox{variable}} \cdot \mathbf{T}_{tz}(d_i)`

    where :math:`q_i` is the joint variable.

    :references:
        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 7.

    :seealso: :func:`PrismaticMDH`, :func:`DHLink`, :func:`RevoluteDH`
    """

    def __init__(
        self, d: float = 0.0, a: float = 0.0, alpha: float = 0.0, offset: float = 0.0,
        qlim: ArrayLike | None = None, flip: bool = False, **kwargs
    ):

        theta = 0.0
        sigma = 0
        mdh = True

        super().__init__(
            d=d,
            alpha=alpha,
            theta=theta,
            a=a,
            sigma=sigma,
            mdh=mdh,
            offset=offset,
            qlim=qlim,
            flip=flip,
            **kwargs,
        )


class PrismaticMDH(DHLink):
    r"""
    Class for prismatic link using modified DH convention.

    :param theta: kinematic: joint angle
    :param alpha: kinematic - link twist
    :param a: kinematic - link length
    :param offset: kinematic - joint variable offset
    :param qlim: joint variable limits [min, max]
    :type qlim: ndarray(2,)
    :param flip: joint moves in opposite direction

    :param m: dynamic - link mass
    :type m: float
    :param r: dynamic - position of COM with respect to link frame
    :type r: ndarray(3,)
    :param I: dynamic - inertia of link with respect to COM
    :type I: ndarray
    :param Jm: dynamic - motor inertia
    :type Jm: float
    :param B: dynamic - motor viscous friction: B=B⁺=B⁻, [B⁺, B⁻]
    :type B: float, or ndarray(2,)
    :param Tc: dynamic - motor Coulomb friction [Tc⁺, Tc⁻]
    :type Tc: ndarray(2,)
    :param G: dynamic - gear ratio
    :type G: float

    A subclass of the DHLink class for a prismatic joint that holds all
    information related to a robot link such as kinematics parameters,
    rigid-body inertial parameters, motor and transmission parameters.

    The link transform is

    :math:`\mathbf{T}_{tx}(a_{i-1}) \cdot \mathbf{T}_{rx}(\alpha_{i-1}) \cdot \mathbf{T}_{rz}(\theta_i) \cdot \underbrace{\mathbf{T}_{tz}(q_i)}_{\mbox{variable}}`

    where :math:`q_i` is the joint variable.

    :references:
        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 7.

    :seealso: :func:`RevoluteMDH`, :func:`DHLink`, :func:`PrismaticDH`
    """

    def __init__(
        self, theta: float = 0.0, a: float = 0.0, alpha: float = 0.0,
        offset: float = 0.0, qlim: ArrayLike | None = None, flip: bool = False, **kwargs
    ):

        d = 0.0
        sigma = 1
        mdh = True

        super().__init__(
            theta=theta,
            d=d,
            a=a,
            alpha=alpha,
            sigma=sigma,
            mdh=mdh,
            offset=offset,
            qlim=qlim,
            flip=flip,
            **kwargs,
        )
