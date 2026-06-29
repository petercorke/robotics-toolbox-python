from __future__ import annotations

from copy import deepcopy
from abc import ABC
from typing_extensions import Self

# from multiprocessing.sharedctypes import Value
import numpy as np
from functools import wraps
from spatialmath.base import getvector, isscalar, isvector, ismatrix
from spatialmath import SE3, SE2
from ansitable import ANSITable, Column
from spatialgeometry import Shape, SceneNode, SceneGroup
from typing import overload

import roboticstoolbox as rtb
from roboticstoolbox.robot.ETS import ETS, ETS2
from roboticstoolbox.robot.ET import ET, ET2
from warnings import warn

from roboticstoolbox.tools.types import ArrayLike, NDArray

# A generic type variable representing any subclass of BaseETS
# ETSType = TypeVar("ETSType", bound=BaseETS)
# ETType = TypeVar("ETType", bound=BaseET)


def _copy_shapes(shapes, link_name):
    """Copy a list of coal/geometry shape objects, warning and skipping any that
    do not support deepcopy (coal CollisionObject instances cannot be pickled)."""
    import warnings
    from copy import deepcopy

    result = []
    warned = False
    for shape in shapes:
        try:
            result.append(deepcopy(shape))
        except RuntimeError:
            if not warned:
                warnings.warn(
                    f"Collision/geometry shape on link '{link_name}' could not be "
                    "copied (coal objects do not support deepcopy). "
                    "Shapes will be absent from the copied link. "
                    "Reload the robot from URDF if collision checking is needed.",
                    UserWarning,
                    stacklevel=5,
                )
                warned = True
    return result


def _listen_dyn(func):
    """
    @_listen_dyn

    Decorator for property setters

    Use this decorator for any property setter that updates a parameter that
    affects the result of inverse dynamics.  This allows the C version of the
    parameters only having to be updated when they change, rather than on
    every call.  This decorator signals the change by:

    - invoking the ``.dynchanged()`` method of the robot that owns the link.
      This assumes that the Link object is owned by a robot, this happens
      when the Link object is passed to a robot constructor.
    - setting the ``._hasdynamics`` attribute of the Link

    Example::

        @m.setter
        @_listen_dyn
        def m(self, m_new):
            self._m = m_new

    :seealso: :func:`DHLink._dyn_changed`
    """

    @wraps(func)
    def wrapper_listen_dyn(*args):
        if args[0]._robot is not None:
            args[0]._robot.dynchanged()
        args[0]._hasdynamics = True
        return func(*args)

    return wrapper_listen_dyn


class BaseLink(SceneNode, ABC):
    """
    An abstract link superclass for all link types.

    :param ets: kinematic - The elementary transforms which make up the link
    :param name: name of the link
    :param parent: a reference to the parent link in the kinematic chain
    :param joint_name: the name of the joint variable
    :param m: dynamic - link mass
    :param r: dynamic - position of COM with respect to link frame
    :param I: dynamic - inertia of link with respect to COM
    :param Jm: dynamic - motor inertia
    :param B: dynamic - motor viscous friction
    :param Tc: dynamic - motor Coulomb friction [Tc⁺, Tc⁻]
    :param G: dynamic - gear ratio
    :param qlim: joint variable limits [min, max]
    :param geometry: the visual geometry which represents the link. This is used
        to display the link in Swift
    :param collision: the collision geometry which represents the link in collision
        checkers

    It holds metadata related to:

    - a robot link, such as rigid-body inertial parameters defined in the link
      frame, and link name
    - a robot joint, that connects this link to its parent, such as joint
      limits, direction of motion, motor and transmission parameters.

    .. inheritance-diagram:: roboticstoolbox.RevoluteDH
        roboticstoolbox.PrismaticDH roboticstoolbox.RevoluteMDH
        roboticstoolbox.PrismaticMDH roboticstoolbox.Link
        :top-classes: roboticstoolbox.robot.Link
        :parts: 2

    .. rubric:: Notes

    - For a more sophisticated actuator model use the ``actuator``
        attribute which is not initialized or used by this Toolbox.
    - There is no ability to name a joint as supported by URDF

    """

    def __init__(
        self,
        ets: ETS | ETS2 | ET | ET2 = ETS(),
        name=None,
        parent: Self | str | None = None,
        joint_name: str | None = None,
        m: float | None = None,
        r: ArrayLike | None = None,
        I: ArrayLike | None = None,
        Jm: float | None = None,
        B: float | None = None,
        Tc: ArrayLike | None = None,
        G: float | None = None,
        qlim: ArrayLike | None = None,
        qdlim: float | None = None,
        tlim: float | None = None,
        geometry: list[Shape] = [],
        collision: list[Shape] = [],
        **kwargs,
    ):
        # Initialise the scene node
        super().__init__()

        # Reference to parent robot
        self._robot = None

        # Set name of link and joint()
        if name is None:
            self._name = ""
        else:
            self._name = name

        # Link geometry
        self._geometry = SceneGroup(scene_children=geometry)
        self._scene_children.append(self._geometry)

        # Collision Geometry
        self._collision = SceneGroup(scene_children=collision)
        self._scene_children.append(self._collision)

        # Link dynamic Parameters
        def dynpar(self, name, value, default):
            if value is None:
                value = default
                setattr(self, name, value)
                return 0
            else:
                setattr(self, name, value)
                return 1

        dynchange = 0

        # link inertial parameters
        dynchange += dynpar(self, "m", m, 0.0)
        dynchange += dynpar(self, "r", r, np.zeros((3,)))
        dynchange += dynpar(self, "I", I, np.zeros((3, 3)))

        # Motor inertial and frictional parameters
        dynchange += dynpar(self, "Jm", Jm, 0.0)
        dynchange += dynpar(self, "B", B, 0.0)
        dynchange += dynpar(self, "Tc", Tc, np.zeros((2,)))
        dynchange += dynpar(self, "G", G, 0.0)

        # reference to more advanced actuator model
        self.actuator = None
        self._hasdynamics = dynchange > 0

        # Check ETS argument
        if isinstance(ets, ET):
            ets = ETS(ets)
        elif isinstance(ets, ET2):
            ets = ETS2(ets)
        elif not isinstance(ets, (ETS, ETS2)):
            print(ets)
            raise TypeError("The ets argument must be of type ETS or ET")

        self.ets = ets

        # Check parent argument
        if parent is not None:
            if isinstance(parent, str):
                self.parent = None
                self._parent_name = parent
            elif isinstance(parent, BaseLink):
                self.parent = parent
                self._parent_name = None

            else:
                raise TypeError("parent must be BaseLink subclass")
        else:
            self._parent = None
            self._parent_name = None

        self._joint_name = joint_name
        self._children = []

        self.number = 0

        ## Set the limits if provided

        # joint limits are attached to the variable part of the ETS
        if qlim is not None and self.v:
            self.v.qlim = qlim

        # set qdlim and tlim on the link
        self._qdlim = qdlim
        self._tlim = tlim

    # -------------------------------------------------------------------------- #

    def _init_Ts(self):
        # Compute the leading, constant, part of the ETS

        if isinstance(self, Link2):
            T = None
        else:
            T = None

        for et in self._ets:
            # constant transforms only
            if et.isjoint:
                break
            else:
                if T is None:
                    T = et.A()
                else:
                    T = T @ et.A()

        self._Ts = T

    @property
    def Ts(self) -> NDArray | None:
        """
        Constant part of link ETS

        :returns: constant part of link transform
        :rtype: ndarray(4,4) or None

        The ETS for each Link comprises a constant part (possible the
        identity) followed by an optional joint variable transform.
        This property returns the constant part.  If no constant part
        is given, this returns an identity matrix.

        Examples
        --------
        .. runblock:: pycon
        >>> from roboticstoolbox import Link, ET
        >>> link = Link( ET.tz(0.333) * ET.Rx(90, 'deg') * ET.Rz() )
        >>> link.Ts
        >>> link = Link( ET.Rz() )
        >>> link.Ts

        """
        return self._Ts

    @overload
    def ets(self: "Link") -> ETS: ...  # pragma: nocover

    @overload
    def ets(self: "Link2") -> ETS2: ...  # pragma: nocover

    @property
    def ets(self) -> ETS:
        """
        Get/set link ets

        - ``link.ets`` is the link ets
        - ``link.ets = ...`` checks and sets the link ets

        :param ets: the new link ets
        :returns: the current link ets

        """

        return self._ets

    @ets.setter
    @overload
    def ets(self: "Link", new_ets: ETS): ...  # pragma: nocover

    @ets.setter
    @overload
    def ets(self: "Link2", new_ets: ETS2): ...  # pragma: nocover

    @ets.setter
    def ets(self, new_ets):
        if new_ets.n > 1:
            raise ValueError("An elementary link can only have one joint variable")

        if new_ets.n == 1 and not new_ets[-1].isjoint:
            raise ValueError("Variable link must be at the end of the ETS")

        self._ets = new_ets
        self._init_Ts()

        if self._ets.n:
            self._v = self._ets[-1]
            self._isjoint = True
        else:
            self._v = None
            self._isjoint = False

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        if len(self.ets) > 0:
            s += repr(self.ets) + ", "
        s += ", ".join(self._params())
        s += ")"
        return s

    def __str__(self) -> str:
        """
        Pretty prints the ETS Model of the link

        :returns: pretty print of the robot link

        Will output angles in degrees
        """

        s = self.__class__.__name__ + "("
        if self.name is not None:
            s += f'"{self.name}"'

        ets = self.ets
        if len(ets) > 0:
            s += f", {ets}"
        # if self.name is None:
        #     return f"{name}[{self.ets}] "
        # else:
        #     if self.parent is None:
        #         parent = ""
        #     elif isinstance(self.parent, str):
        #         parent = f" [{self.parent}]"
        #     else:
        #         parent = f" [{self.parent.name}]"
        params = self._params(name=False)
        if len(params) > 0:
            s += ", "  # pragma: nocover
        s += ", ".join(params)
        s += ")"
        return s

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        Print colorized output when variable is displayed in IPython, ie. on a line by
        itself.
        """
        # see
        # https://ipython.org/ipython-doc/stable/api/generated/IPython.lib.pretty.html

        p.text(str(self))  # pragma: nocover

    # -------------------------------------------------------------------------- #

    def copy(self: Self) -> Self:
        """
        Copy of link object

        :returns: copy of link object

        ``link.copy()`` is a new Link subclass instance with a copy of all
        the parameters.
        """

        return deepcopy(self)

    def _copy(self):
        raise DeprecationWarning("Use copy method of Link class")

    def __deepcopy__(self, memo):
        ets = deepcopy(self.ets)
        name = deepcopy(self.name)
        parent = self.parent
        joint_name = deepcopy(self._joint_name)
        m = deepcopy(self.m)
        r = deepcopy(self.r)
        I = deepcopy(self.I)
        Jm = deepcopy(self.Jm)
        B = deepcopy(self.B)
        Tc = deepcopy(self.Tc)
        G = deepcopy(self.G)
        qlim = deepcopy(self.qlim)
        qdlim = deepcopy(self.qdlim)
        tlim = deepcopy(self.tlim)
        geometry = _copy_shapes(self._geometry, self.name)
        collision = _copy_shapes(self._collision, self.name)

        cls = self.__class__
        result = cls(
            ets=ets,
            name=name,
            parent=parent,
            joint_name=joint_name,
            m=m,
            r=r,
            I=I,
            Jm=Jm,
            B=B,
            Tc=Tc,
            G=G,
            qlim=qlim,
            qdlim=qdlim,
            tlim=tlim,
            geometry=geometry,  # type: ignore
            collision=collision,  # type: ignore
        )

        if self._children:
            result._children = self._children.copy()

        result._robot = self.robot

        memo[id(self)] = result
        return result

    # -------------------------------------------------------------------------- #

    @overload
    def v(self: "Link") -> ET | None: ...  # pragma: nocover

    @overload
    def v(self: "Link2") -> ET2 | None: ...  # pragma: nocover

    @property
    def v(self):
        """
        Variable part of link ETS

        :returns: joint variable transform

        The ETS for each Link comprises a constant part (possible the
        identity) followed by an optional joint variable transform.
        This property returns the latter.

        Examples
        --------
        .. runblock:: pycon
        >>> from roboticstoolbox import Link, ET, ETS
        >>> link = Link( ET.tz(0.333) * ET.Rx(90, 'deg') * ET.Rz() )
        >>> print(link.v)

        """
        return self._v

    # -------------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        """
        Get/set link name

        - ``link.name`` is the link name
        - ``link.name = ...`` checks and sets the link name

        :returns: link name

        """
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    # -------------------------------------------------------------------------- #

    @property
    def robot(self) -> rtb.BaseRobot | None:
        """
        Get forward reference to the robot which owns this link

        - ``link.robot`` is the robot reference
        - ``link.robot = ...`` checks and sets the robot reference

        :returns: The robot object

        """
        return self._robot

    @robot.setter
    def robot(self, robot_ref: "rtb.BaseRobot"):
        """
        Set the forward reference to the robot which owns this link
        """
        self._robot = robot_ref

    # -------------------------------------------------------------------------- #

    @property
    def qlim(self) -> NDArray | None:
        """
        Get/set joint limits

        - ``link.qlim`` is the joint limits
        - ``link.qlim = ...`` checks and sets the joint limits

        :returns: joint limits
        :rtype: ndarray(2,) or None

        .. rubric:: Notes

        - The limits are not widely enforced within the toolbox.
        - If no joint limits are specified the value is ``None``

        :seealso: :func:`~islimit`
        """

        if self.v:
            return self.v.qlim
        else:
            return None

    @qlim.setter
    def qlim(self, qlim_new: ArrayLike):
        if self.v:
            self.ets.qlim = qlim_new
        else:
            raise ValueError("Can not set qlim on a static joint")

    @property
    def qdlim(self) -> float | None:
        """
        Get/set joint velocity limits

        - ``link.qdlim`` is the joint velocity limits
        - ``link.qdlim = ...`` checks and sets the joint velocity limits

        :returns: joint velocity limits

        .. rubric:: Notes

        - The limits are not widely enforced within the toolbox.
        - If no joint velocity limits are specified the value is ``None``

        :seealso: :func:`~islimit`
        """

        return self._qdlim
    
    @qdlim.setter
    def qdlim(self, qdlim_new: float):
        self._qdlim = qdlim_new

    @property
    def tlim(self) -> float | None:
        """
        Get/set joint torque limits

        - ``link.tlim`` is the joint torque/force limits
        - ``link.tlim = ...`` checks and sets the joint torque limits

        :returns: joint torque limits

        .. rubric:: Notes

        - The limits are not widely enforced within the toolbox.
        - If no joint torque limits are specified the value is ``None``

        :seealso: :func:`~islimit`
        """

        return self._tlim   
    
    @tlim.setter
    def tlim(self, tlim_new: float):
        self._tlim = tlim_new

    @property
    def hasdynamics(self) -> bool:
        """
        Link has dynamic parameters (Link superclass)

        :returns: Link has dynamic parameters

        Link has some assigned (non-default) dynamic parameters.  These could
        have been assigned:

        - at constructor time, eg. ``m=1.2``
        - by invoking a setter method, eg. ``link.m = 1.2``

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot[1].hasdynamics

        """
        return self._hasdynamics

    # -------------------------------------------------------------------------- #

    @property
    def isflip(self) -> bool:
        """
        Get/set joint flip

        :returns: joint flip

        - ``link.flip`` is the joint flip status
        - ``link.flip = ...`` checks and sets the joint flip status

        Joint flip defines the direction of motion of the joint.

        ``flip = False`` is conventional motion direction:

            - revolute motion is a positive rotation about the z-axis
            - prismatic motion is a positive translation along the z-axis

        ``flip = True`` is the opposite motion direction:

            - revolute motion is a negative rotation about the z-axis
            - prismatic motion is a negative translation along the z-axis
        """

        return self.v.isflip if self.v else False

    # -------------------------------------------------------------------------- #

    @property
    def m(self) -> float:
        """
        Get/set link mass

        - ``link.m`` is the link mass
        - ``link.m = ...`` checks and sets the link mass

        :returns: link mass

        """

        return self._m

    @m.setter
    @_listen_dyn
    def m(self, m_new: float):
        self._m = m_new

    # -------------------------------------------------------------------------- #

    @property
    def r(self) -> NDArray:
        """
        Get/set link centre of mass

        :returns: link centre of mass
        :rtype: ndarray(3,)

        The link centre of mass is a 3-vector defined with respect to the link
        frame.

        - ``link.r`` is the link centre of mass
        - ``link.r = ...`` checks and sets the link centre of mass
        """

        return self._r  # type: ignore

    @r.setter
    @_listen_dyn
    def r(self, r_new: ArrayLike):
        self._r = getvector(r_new, 3)

    # -------------------------------------------------------------------------- #

    @property
    def I(self) -> NDArray:
        r"""
        Get/set link inertia

        :returns: link inertia
        :rtype: ndarray(3,3)

        Link inertia is a symmetric 3x3 matrix describing the inertia with
        respect to a frame with its origin at the centre of mass, and with
        axes parallel to those of the link frame.

        - ``link.I`` is the link inertia
        - ``link.I = ...`` checks and sets the link inertia

        The inertia matrix is

        :math:`\begin{bmatrix} I_{xx} & I_{xy} & I_{xz} \\ I_{xy} & I_{yy} & I_{yz} \\I_{xz} & I_{yz} & I_{zz} \end{bmatrix}`

        and can be specified as either:

        - a 3 ⨉ 3 symmetric matrix
        - a 3-vector :math:`(I_{xx}, I_{yy}, I_{zz})`
        - a 6-vector :math:`(I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{yz}, I_{xz})`

        .. rubric:: Notes

        - Referred to the link side of the gearbox.

        """

        return self._I  # type: ignore

    @I.setter
    @_listen_dyn
    def I(self, I_new: ArrayLike):
        if ismatrix(I_new, (3, 3)):
            # 3x3 matrix passed
            if np.any(np.abs(I_new - I_new.T) > 1e-8):  # type: ignore
                raise ValueError("3x3 matrix is not symmetric")

        elif isvector(I_new, 9):
            # 3x3 matrix passed as a 1d vector
            I_new = I_new.reshape(3, 3)  # type: ignore
            if np.any(np.abs(I_new - I_new.T) > 1e-8):  # type: ignore
                raise ValueError("3x3 matrix is not symmetric")

        elif isvector(I_new, 6):
            # 6-vector passed, moments and products of inertia,
            # [Ixx Iyy Izz Ixy Iyz Ixz]
            I_new = np.array(
                [
                    [I_new[0], I_new[3], I_new[5]],  # type: ignore
                    [I_new[3], I_new[1], I_new[4]],  # type: ignore
                    [I_new[5], I_new[4], I_new[2]],  # type: ignore
                ]
            )

        elif isvector(I_new, 3):
            # 3-vector passed, moments of inertia [Ixx Iyy Izz]
            I_new = np.diag(I_new)  # type: ignore

        else:
            raise ValueError("invalid shape passed: must be (3,3), (6,), (3,)")

        self._I = I_new

    # -------------------------------------------------------------------------- #

    @property
    def Jm(self) -> float:
        """
        Get/set motor inertia

        - ``link.Jm`` is the motor inertia
        - ``link.Jm = ...`` checks and sets the motor inertia

        :returns: motor inertia

        .. rubric:: Notes

        - Referred to the motor side of the gearbox.

        """

        return self._Jm

    @Jm.setter
    @_listen_dyn
    def Jm(self, Jm_new: float):
        self._Jm = Jm_new

    # -------------------------------------------------------------------------- #

    @property
    def B(self) -> float:
        """
        Get/set motor viscous friction

        - ``link.B`` is the motor viscous friction
        - ``link.B = ...`` checks and sets the motor viscous friction

        :returns: motor viscous friction

        .. rubric:: Notes

        - Referred to the motor side of the gearbox.
        - Viscous friction is the same for positive and negative motion.

        """
        return self._B

    @B.setter
    @_listen_dyn
    def B(self, B_new: float):
        if isscalar(B_new):
            self._B = B_new
        else:
            raise TypeError("B must be a scalar")

    # -------------------------------------------------------------------------- #

    @property
    def Tc(self) -> NDArray:
        r"""
        Get/set motor Coulomb friction

        :returns: motor Coulomb friction
        :rtype: ndarray(2,)

        - ``link.Tc`` is the motor Coulomb friction
        - ``link.Tc = ...`` checks and sets the motor Coulomb friction. If a
          scalar is given the value is set to [T, -T], if a 2-vector it is
          assumed to be in the order [Tc⁺, Tc⁻]

        Coulomb friction is a non-linear friction effect defined by two
        parameters such that

        .. math::

            \tau = \left\{ \begin{array}{ll}
                \tau_C^+ & \mbox{if $\dot{q} > 0$} \\
                \tau_C^- & \mbox{if $\dot{q} < 0$} \end{array} \right.

        .. rubric:: Notes

        -  Referred to the motor side of the gearbox.
        - :math:`\tau_C^+` must be :math:`> 0`, and :math:`\tau_C^-` must
            be :math:`< 0`.

        """

        return self._Tc

    @Tc.setter
    @_listen_dyn
    def Tc(self, Tc_new: ArrayLike):
        try:
            # sets Coulomb friction parameters to [F -F], for a symmetric
            # Coulomb friction model.
            Tc = getvector(Tc_new, 1)
            Tc_new = np.array([Tc[0], -Tc[0]])  # type: ignore
        except ValueError:
            # [FP FM] sets Coulomb friction to [FP FM], for an asymmetric
            # Coulomb friction model. FP>0 and FM<0.  FP is applied for a
            # positive joint velocity and FM for a negative joint
            # velocity.
            Tc_new = np.array(getvector(Tc_new, 2))

        self._Tc = Tc_new

    # -------------------------------------------------------------------------- #

    @property
    def G(self) -> float:
        """
        Get/set gear ratio

        - ``link.G`` is the transmission gear ratio
        - ``link.G = ...`` checks and sets the gear ratio

        :returns: gear ratio

        .. rubric:: Notes

        - The ratio of motor motion : link motion
        - The gear ratio can be negative, see also the ``flip`` attribute.

        :seealso: :func:`flip`
        """

        return self._G

    @G.setter
    @_listen_dyn
    def G(self, G_new: float):
        self._G = G_new

    # -------------------------------------------------------------------------- #

    @property
    def geometry(self) -> SceneGroup:
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
    def collision(self) -> SceneGroup:
        """
        Get/set joint collision geometry

        The collision geometries are what is used to check for collisions.

        - ``link.collision`` is the list of the collision geometries which
            represent the collidable shape of the link.
            :return: the collision geometries
            :rtype: list of Shape
        - ``link.collision = ...`` checks and sets the collision geometry
        - ``link.collision.append(...)`` add collision geometry

        """

        return self._collision

    @collision.setter
    def collision(self, coll: SceneGroup | list[Shape] | Shape):
        if isinstance(coll, list):
            self.collision.scene_children = coll  # type: ignore
        elif isinstance(coll, Shape):
            self.collision.scene_children.append(coll)
        elif isinstance(coll, SceneGroup):
            self._collision = coll

    @geometry.setter
    def geometry(self, geom: SceneGroup | list[Shape] | Shape):
        if isinstance(geom, list):
            self.geometry.scene_children = geom  # type: ignore
        elif isinstance(geom, Shape):
            self.geometry.scene_children.append(geom)
        elif isinstance(geom, SceneGroup):
            self._geometry = geom

    # -------------------------------------------------------------------------- #

    @property
    def isjoint(self) -> bool:
        """
        Test if link has joint

        :returns: test if link has a joint

        The ETS for each Link comprises a constant part (possible the
        identity) followed by an optional joint variable transform.
        This property returns the whether the Link contains the
        variable transform.

        Examples
        --------
        .. runblock:: pycon
        >>> from roboticstoolbox import models
        >>> robot = models.URDF.Panda()
        >>> robot[1].isjoint  # link with joint
        >>> robot[8].isjoint  # static link

        """

        # return self.v.isjoint if self.v else False
        return self._isjoint

    @property
    def jindex(self) -> int | None:
        """
        Get/set joint index

        - ``link.jindex`` is the joint index
        - ``link.jindex = ...`` checks and sets the joint index

        :returns: joint index

        For a serial-link manipulator the joints are numbered starting at zero
        and increasing sequentially toward the end-effector.  For branched
        mechanisms this is not so straightforward.
        The link's ``jindex`` property specifies the index of its joint
        variable within a vector of joint coordinates.

        .. rubric:: Notes

        - ``jindex`` values must be a sequence of integers starting
            at zero.

        """
        return None if not self.v else self.v._jindex

    @jindex.setter
    def jindex(self, j: int):
        if self.v:
            self.v.jindex = j
            self.ets._auto_jindex = False

    @property
    def isprismatic(self) -> bool:
        """
        Checks if the joint is of prismatic type

        :returns: True if is prismatic
        """
        return self.v.istranslation if self.v else False

    @property
    def isrevolute(self) -> bool:
        """
        Checks if the joint is of revolute type

        :returns: True if is revolute
        """

        return self.v.isrotation if self.v else False

    @property
    def parent(self) -> Self | None:
        """
        Parent link

        :returns: Link's parent

        This is a reference to the links parent in the kinematic
        chain

        Examples
        --------
        .. runblock:: pycon
        >>> from roboticstoolbox import models
        >>> robot = models.URDF.Panda()
        >>> robot[0].parent  # base link has no parent
        >>> robot[1].parent  # second link's parent

        """

        return self._parent

    @parent.setter
    def parent(self, parent: Self | None):
        self._parent = parent

    @property
    def parent_name(self) -> str | None:
        """
        Parent link name

        :returns: Link's parent name

        """

        if isinstance(self.parent, BaseLink):
            return self.parent.name
        else:
            return self._parent_name

    @property
    def children(self) -> list[Link] | None:
        """
        List of child links

        :returns: child links

        The list will be empty for a end-effector link
        """

        return self._children

    @property
    def nchildren(self) -> int:
        """
        Number of child links

        :returns: number of child links

        Will be zero for an end-effector link
        """
        return len(self._children)

    def closest_point(
        self, shape: Shape, inf_dist: float = 1.0, skip: bool = False
    ) -> tuple[int | None, NDArray | None, NDArray | None]:
        """
        Finds the closest point to a shape

        :param shape: The shape to compare distance to
        :param inf_dist: The minimum distance within which to consider the shape
        :param skip: Skip setting all shape transforms
        :returns: (d, p1, p2) where d is the distance between the shapes,
            p1 is the point in the world frame on the link [x, y, z], and
            p2 is the point in the world frame on the shape [x, y, z].

        closest_point(shape, inf_dist) returns the minimum euclidean
        distance between this link and shape, provided it is less than
        inf_dist. It will also return the points on self and shape in the
        world frame which connect the line of length distance between the
        shapes. If the distance is negative then the shapes are collided.
        """

        if not skip:
            self.robot._update_link_tf(self.robot.q)  # type: ignore
            self._propogate_scene_tree()
            shape._propogate_scene_tree()

        d = 10000
        p1 = None
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

    def iscollided(self, shape: Shape, skip: bool = False) -> bool:
        """
        Checks for collision with a shape

        :param shape: The shape to compare distance to
        :param skip: Skip setting all shape transforms
        :returns: True if shapes have collided

        ``iscollided(shape)`` checks if this link and shape have collided
        """

        if not skip:
            self.robot._update_link_tf(self.robot.q)  # type: ignore
            self._propogate_scene_tree()
            shape._propogate_scene_tree()

        for col in self.collision:
            if col.iscollided(shape):
                return True

        return False

    def collided(self, shape: Shape, skip: bool = False):
        """
        Checks for collision with a shape

        :param shape: The shape to compare distance to
        :param skip: Skip setting all shape transforms
        :returns: True if shapes have collided

        ``iscollided(shape)`` checks if this link and shape have collided
        """

        warn("base kwarg is deprecated, use pose instead", FutureWarning)
        return self.iscollided(shape=shape, skip=skip)

    def dyn(self, indent=0):
        """
        Inertial properties of link as a string

        :param indent: indent each line by this many spaces

        ``link.dyn()`` is a string representation the inertial properties of
        the link object in a multi-line format. The properties shown are mass,
        centre of mass, inertia, friction, gear ratio and motor properties.

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> print(robot.links[2])        # kinematic parameters
        >>> print(robot.links[2].dyn())  # dynamic parameters

        :seealso: :func:`~dyntable`
        """

        qlim = [0, 0] if self.qlim is None else self.qlim

        s = (
            "m     =  {:8.2g} \n"
            "r     =  {:8.2g} {:8.2g} {:8.2g} \n"
            "        | {:8.2g} {:8.2g} {:8.2g} | \n"
            "I     = | {:8.2g} {:8.2g} {:8.2g} | \n"
            "        | {:8.2g} {:8.2g} {:8.2g} | \n"
            "Jm    =  {:8.2g} \n"
            "B     =  {:8.2g} \n"
            "Tc    =  {:8.2g}(+) {:8.2g}(-) \n"
            "G     =  {:8.2g} \n"
            "qlim  =  {:8.2g} to {:8.2g}".format(
                self.m,
                self.r[0],
                self.r[1],
                self.r[2],
                self.I[0, 0],
                self.I[0, 1],
                self.I[0, 2],
                self.I[1, 0],
                self.I[1, 1],
                self.I[1, 2],
                self.I[2, 0],
                self.I[2, 1],
                self.I[2, 2],
                self.Jm,
                self.B,
                self.Tc[0],
                self.Tc[1],
                self.G,
                qlim[0],
                qlim[1],
            )
        )

        if indent > 0:
            # insert indentations into the string
            # TODO there is probably a tidier way to integrate this step with
            # above
            sp = " " * indent
            s = sp + s.replace("\n", "\n" + sp)

        return s

    def _dyn2list(self, fmt="{: .3g}"):
        """
        Inertial properties of link as a string

        :param fmt: conversion format for each number
        :returns: The string representation of the link dynamics

        ``link._dyn2list()`` returns a list of pretty-printed inertial
        properties of the link The properties included are mass, centre of
        mass, inertia, friction, gear ratio and motor properties.

        :seealso: :func:`~dyn`
        """

        ANSITable(
            Column("Parameter", headalign="^"),
            Column("Value", headalign="^", colalign="<"),
            border="thin",
        )

        def format(l, fmt, val):
            if isinstance(val, np.ndarray):
                try:
                    s = ", ".join([fmt.format(v) for v in val])
                except TypeError:  # pragma: nocover
                    # handle symbolic case
                    s = ", ".join([str(v) for v in val])
            else:
                try:
                    s = fmt.format(val)
                except TypeError:  # pragma: nocover
                    # handle symbolic case
                    s = str(val)
            l.append(s)

        dyn = []
        format(dyn, fmt, self.m)
        format(dyn, fmt, self.r)
        I = self.I.flatten()
        format(dyn, fmt, np.r_[[I[k] for k in [0, 4, 8, 1, 5, 2]]])
        format(dyn, fmt, self.Jm)
        format(dyn, fmt, self.B)
        format(dyn, fmt, self.Tc)
        format(dyn, fmt, self.G)

        return dyn

    def _format_param(
        self,
        l,
        name,
        symbol=None,
        ignorevalue=None,
        indices=None,
    ):  # noqa  # pragma nocover
        # if value == ignorevalue then don't display it

        v = getattr(self, name)
        s = None
        if v is None:
            return
        if isinstance(v, str):
            s = f'{name} = "{v}"'
        elif isscalar(v) and v != ignorevalue:
            if symbol is not None:
                s = f"{symbol}={v:.3g}"
            else:  # pragma: nocover
                try:
                    s = f"{name}={v:.3g}"
                except TypeError:
                    s = f"{name}={v}"
        elif isinstance(v, np.ndarray):
            # if np.linalg.norm(v, ord=np.inf) > 0:
            #     if indices is not None:
            #         flat = v.flatten()
            #         v = np.r_[[flat[k] for k in indices]]
            #     s = f"{name}=[" + ", ".join([f"{x:.3g}" for x in v]) + "]"
            if indices is not None:
                v = v.ravel()[indices]
            s = f"{name}=" + np.array2string(
                v,
                separator=", ",
                suppress_small=True,
                formatter={"float": lambda x: f"{x:.3g}"},
            )
        if s is not None:
            l.append(s)

    def _params(self, name: bool = True):  # pragma nocover

        l = []
        if name:
            self._format_param(l, "name")
        if self.parent_name is not None:
            l.append('parent="' + self.parent_name + '"')
        elif isinstance(self.parent, BaseLink):
            l.append('parent="' + self.parent.name + '"')
        self._format_param(l, "parent")
        self._format_param(l, "isflip", ignorevalue=False)
        self._format_param(l, "qlim")
        if self._hasdynamics:
            self._format_param(l, "m")
            self._format_param(l, "r")
            self._format_param(l, "I", indices=[0, 4, 8, 1, 2, 5])
            self._format_param(l, "Jm")
            self._format_param(l, "B")
            self._format_param(l, "Tc")
            self._format_param(l, "G")

        return l

    def islimit(self, q: float):
        """
        Checks if joint exceeds limit

        :param q: joint coordinate
        :returns: True if joint is exceeded

        ``link.islimit(q)`` is True if ``q`` exceeds the joint limits defined
        by ``link``.

        .. rubric:: Notes

        - If no limits are set always return False.

        :seealso: :func:`qlim`
        """

        if self.qlim is None:
            return False
        else:
            return q < self.qlim[0] or q > self.qlim[1]

    def nofriction(self, coulomb: bool = True, viscous: bool = False):
        """
        Clone link without friction

        :param coulomb: if True, will set the Coulomb friction to 0
        :param viscous: if True, will set the viscous friction to 0

        ``link.nofriction()`` is a copy of the link instance with the same
        parameters except, the Coulomb and/or viscous friction parameters are
        set to zero.

        .. rubric:: Notes

        - For simulation it can be useful to remove Couloumb friction
            which can cause problems for numerical integration.

        """

        # Copy the Link
        link = self.copy()

        if viscous:
            link.B = 0.0

        if coulomb:
            link.Tc = [0.0, 0.0]

        return link

    def friction(self, qd: float, coulomb: bool = True):
        r"""
        Compute joint friction

        :param qd: The joint velocity
        :param coulomb: include Coulomb friction
        :returns: the friction force/torque

        ``friction(qd)`` is the joint friction force/torque
        for joint velocity ``qd``. The friction model includes:

        - Viscous friction which is a linear function of velocity.
        - Coulomb friction which is proportional to sign(qd).

        .. math::

            \tau = G^2 B \dot{q} + |G| \left\{ \begin{array}{ll}
                \tau_C^+ & \mbox{if $\dot{q} > 0$} \\
                \tau_C^- & \mbox{if $\dot{q} < 0$} \end{array} \right.

        .. rubric:: Notes

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


class Link(BaseLink):
    """
    ETS link class

    :param ets: kinematic - The elementary transforms which make up the link
    :param jindex: the joint variable index
    :param name: name of the link
    :param parent: a reference to the parent link in the kinematic chain
    :param joint_name: the name of the joint variable
    :param m: dynamic - link mass
    :param r: dynamic - position of COM with respect to link frame
    :param I: dynamic - inertia of link with respect to COM
    :param Jm: dynamic - motor inertia
    :param B: dynamic - motor viscous friction
    :param Tc: dynamic - motor Coulomb friction [Tc⁺, Tc⁻]
    :param G: dynamic - gear ratio
    :param qlim: joint variable limits [min, max]
    :param geometry: the visual geometry which represents the link. This is used
        to display the link in Swift
    :param collision: the collision geometry which represents the link in collision
        checkers

    The Link object holds all information related to a robot link and can form
    a serial-connected chain or a rigid-body tree.
    It inherits from the Link class which provides common functionality such
    as joint and link such as kinematics parameters,
    The transform to the next link is given as an ETS with the joint
    variable, if present, as the last term.  This is preprocessed and
    the object stores:

    - ``Ts`` the constant part as a NumPy array, or None
    - ``v`` a pointer to an ETS object representing the joint variable, or None

    :seealso: :class:`Link2`, :class:`DHLink`
    """

    def __init__(
        self, ets: ETS | ET = ETS(), jindex: int | None = None, **kwargs
    ):
        # process common options
        super().__init__(ets=ets, **kwargs)

        # check we have an ETS
        if not isinstance(self._ets, ETS):  # pragma: nocover
            raise TypeError("The ets argument must be of type ETS")

        # Set the jindex
        if len(self._ets) > 0 and self._ets[-1].isjoint:
            if jindex is not None:
                self._ets[-1].jindex = jindex
                self._ets._auto_jindex = False

    def A(self, q: float = 0.0) -> SE3:
        """
        Link transform matrix

        :param q: Joint coordinate (radians or metres). Not required for links
            with no variable
        :returns: link frame transformation matrix

        ``link.A(q)`` is an SE(3) matrix that describes the rigid-body
        transformation from the previous to the current link frame to
        the next, which depends on the joint coordinate ``q``.
        """
        if self.isjoint:
            if self._Ts is not None:
                return SE3(self._Ts @ self._ets[-1].A(q), check=False)
            else:
                return SE3(self._ets[-1].A(q), check=False)

        elif self._Ts is not None:
            return SE3(self._Ts, check=False)
        else:
            return SE3()


class Link2(BaseLink):
    def __init__(self, ets: ETS2 = ETS2(), jindex: int | None = None, **kwargs):
        # process common options
        super().__init__(ets=ets, **kwargs)

        # check we have an ETS
        if not isinstance(self._ets, ETS2):  # pragma: nocover
            raise TypeError("The self._ets argument must be of type ETS2")

        # Set the jindex
        if len(self._ets) > 0 and self._ets[-1].isjoint:
            if jindex is not None:
                self._ets[-1].jindex = jindex  # pragma: nocover

    def A(self, q: float = 0.0) -> SE2:
        """
        Link transform matrix

        :param q: Joint coordinate (radians or metres). Not required for links
            with no variable
        :returns: link frame transformation matrix

        ``link.A(q)`` is an SE(2) matrix that describes the rigid-body
        transformation from the previous to the current link frame to
        the next, which depends on the joint coordinate ``q``.
        """

        if self.isjoint:
            if self._Ts is not None:
                return SE2(self._Ts @ self._ets[-1].A(q), check=False)
            else:
                return SE2(self._ets[-1].A(q), check=False)

        elif self._Ts is not None:
            return SE2(self._Ts, check=False)
        else:
            return SE2()
