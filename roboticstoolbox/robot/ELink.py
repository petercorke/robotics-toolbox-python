#!/usr/bin/env python

"""
@author: Jesse Haviland
"""

from spatialgeometry import Shape
from roboticstoolbox.robot.ETS import ETS, ETS2
from roboticstoolbox.robot.ET import ET, ET2
from roboticstoolbox.robot.Link import Link
from numpy import ndarray, eye
import fknm
from typing import Union, overload


class BaseELink(Link):
    def __init__(
        self,
        ets: Union[ETS, ETS2] = ETS(),
        name: str = None,
        parent: Union["BaseELink", str, None] = None,
        joint_name: str = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # check we have an ETS
        if not isinstance(ets, (ETS, ETS2)):
            raise TypeError("The ets argument must be of type ETS")

        if ets.n > 1:
            raise ValueError("An elementary link can only have one joint variable")

        if ets.n == 1 and not ets[-1].isjoint:
            raise ValueError("Variable link must be at the end of the ETS")

        self._name = name
        self._ets = ets

        if parent is not None:
            if isinstance(parent, BaseELink):
                self._parent = parent
                self._parent_name = None
            elif isinstance(parent, str):
                self._parent = None
                self._parent_name = parent
            else:
                raise TypeError("parent must be BaseELink subclass")
        else:
            self._parent = None
            self._parent_name = None

        self._joint_name = joint_name
        self._children = []
        self._init_Ts()

    def __repr__(self):
        name = self.__class__.__name__
        if self.name is None:
            s = f"ets={self.ets}"
        else:
            s = f"{self.name}, ets={self.ets}"
        if self.parent is None:
            parent = ""
        elif isinstance(self.parent, str):
            parent = f" [{self.parent}]"
        else:
            parent = f" [{self.parent.name}]"
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
            return f"{name}[{self.ets}] "
        else:
            if self.parent is None:
                parent = ""
            elif isinstance(self.parent, str):
                parent = f" [{self.parent}]"
            else:
                parent = f" [{self.parent.name}]"
            return f"{name}[{self.name}({parent}): {self.ets}] "

    def copy(self, parent=None):
        new = super().copy()

        new._geometry = [shape.copy() for shape in self._geometry]
        new._collision = [shape.copy() for shape in self._collision]

        # invalidate references to parent, child
        new._parent = parent
        new._children = []
        return new

    def _init_Ts(self):
        # Compute the leading, constant, part of the ETS

        # Ts can not be equal to None otherwise things seem
        # to break everywhere, so initialise Ts np be identity

        if isinstance(self, ELink2):
            T = eye(3)
        else:
            T = eye(4)

        for et in self._ets:
            # constant transforms only
            if et.isjoint:
                break
            else:
                T = T @ et.T()

        self._Ts = T

    @overload
    def v(self: "ELink") -> Union["ET", None]:
        ...

    @overload
    def v(self: "ELink2") -> Union["ET2", None]:
        ...

    @property
    def v(self):
        """
        Variable part of link ETS
        :return: joint variable transform
        :rtype: ET instance
        The ETS for each ELink comprises a constant part (possible the
        identity) followed by an optional joint variable transform.
        This property returns the latter.
        .. runblock:: pycon
            >>> from roboticstoolbox import ELink, ETS
            >>> link = ELink( ET.tz(0.333) * ET.Rx(90, 'deg') * ETS.Rz() )
            >>> print(link.v)
        """
        if self.isjoint:
            return self._ets[-1]
        else:
            return None

    # @v.setter
    # def v(self, new):
    #     if not isinstance(new, ETS) and new is not None:
    #         raise TypeError("v must be an ETS object")

    #     self._v = new
    #     self._update_fknm()

    @property
    def Ts(self) -> ndarray:
        """
        Constant part of link ETS
        :return: constant part of link transform
        :rtype: SE3 instance
        The ETS for each ELink comprises a constant part (possible the
        identity) followed by an optional joint variable transform.
        This property returns the constant part.  If no constant part
        is given, this returns an identity matrix.
        .. runblock:: pycon
            >>> from roboticstoolbox import ELink, ET
            >>> link = ELink( ET.tz(0.333) * ET.Rx(90, 'deg') * ET.Rz() )
            >>> link.Ts
            >>> link = ELink( ET.Rz() )
            >>> link.Ts
        """
        return self._Ts

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
        return len(self._ets) > 0 and self._ets[-1].isjoint

    @property
    def jindex(self) -> Union[None, int]:
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
        return None if not self.isjoint else self._ets[-1]._jindex

    @jindex.setter
    def jindex(self, j: int):
        self._ets[-1].jindex = j
        # try:
        #     self._update_fknm()
        # except AttributeError:
        #     # ELink2 doesnt have this
        #     pass

    @property
    def isprismatic(self) -> bool:
        """
        Checks if the joint is of prismatic type
        :return: True if is prismatic
        :rtype: bool
        """
        return self.isjoint and self._ets[-1].istranslation

    @property
    def isrevolute(self) -> bool:
        """
        Checks if the joint is of revolute type
        :return: True if is revolute
        :rtype: bool
        """
        return self.isjoint and self._ets[-1].isrotation

    @property
    def ets(self):
        return self._ets

    @overload
    def parent(self: "ELink") -> Union["ELink", None]:
        ...

    @overload
    def parent(self: "ELink2") -> Union["ELink2", None]:
        ...

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
    def parent_name(self) -> Union[str, None]:
        """
        Parent link name

        :return: Link's parent name
        """
        if self._parent is not None:
            return self._parent.name
        else:
            return self._parent_name

    @property
    def children(self) -> Union[list["ELink"], None]:
        """
        List of child links
        :return: child links
        :rtype: list of ``ELink`` instances
        The list will be empty for a end-effector link
        """
        return self._children

    @property
    def nchildren(self) -> int:
        """
        Number of child links
        :return: number of child links
        :rtype: int
        Will be zero for an end-effector link
        """
        return len(self._children)

    @property
    def geometry(self) -> list[Shape]:
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
    def collision(self) -> list[Shape]:
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

    @collision.setter
    def collision(self, coll: Union[Shape, list[Shape]]):
        new_coll = []

        if isinstance(coll, list):
            for gi in coll:
                if isinstance(gi, Shape):
                    new_coll.append(gi)
                else:
                    raise TypeError("Collision must be of Shape class")
        elif isinstance(coll, Shape):
            new_coll.append(coll)
        else:
            raise TypeError("Geometry must be of Shape class or list of Shape")

        self._collision = new_coll

    @geometry.setter
    def geometry(self, geom: Union[Shape, list[Shape]]):
        new_geom = []

        if isinstance(geom, list):
            for gi in geom:
                if isinstance(gi, Shape):
                    new_geom.append(gi)
                else:
                    raise TypeError("Geometry must be of Shape class")
        elif isinstance(geom, Shape):
            new_geom.append(geom)
        else:
            raise TypeError("Geometry must be of Shape class or list of Shape")

        self._geometry = new_geom


class ELink(BaseELink):
    """
    ETS link class

    :param ets: kinematic - The elementary transforms which make up the link
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
    The transform to the next link is given as an ETS with the joint
    variable, if present, as the last term.  This is preprocessed and
    the object stores:
        * ``Ts`` the constant part as a NumPy array, or None
        * ``v`` a pointer to an ETS object representing the joint variable.
          or None

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke

    :seealso: :class:`Link`, :class:`DHLink`
    """

    def __init__(self, ets: ETS = ETS(), jindex: Union[None, int] = None, **kwargs):

        # process common options
        super().__init__(ets=ets, **kwargs)

        # check we have an ETS
        if not isinstance(ets, ETS):
            raise TypeError("The ets argument must be of type ETS2")

        self._ets = ets

        # Set the jindex
        if len(ets) > 0 and ets[-1].isjoint:
            if jindex is not None:
                ets[-1].jindex = jindex

        # Private variable, can be written to but never replaced!
        # The c will adjust the inside of this array with a reference
        # to this specific array. If replaced --> segfault
        self._fk = eye(4)
        self._init_fknm()

    # def copy(self, parent=None):
    #     new = super().copy(parent)
    #     new._init_fknm()
    #     return new

    def _get_fknm(self):
        isflip = False
        axis = 0
        jindex = 0

        if self.isjoint:
            isflip = self.ets[-1].isflip
            jindex = self.jindex

            if jindex is None:
                jindex = 0

            if self.ets[-1].axis == "Rx":
                axis = 0
            elif self.ets[-1].axis == "Ry":
                axis = 1
            elif self.ets[-1].axis == "Rz":
                axis = 2
            elif self.ets[-1].axis == "tx":
                axis = 3
            elif self.ets[-1].axis == "ty":
                axis = 4
            elif self.ets[-1].axis == "tz":
                axis = 5

        parent = None
        # if self.parent is None:
        #     parent = None
        # else:
        #     parent = self.parent._fknm

        shape_base = []
        shape_wT = []
        shape_sT = []
        shape_sq = []

        for shap in self.geometry:
            shape_base.append(shap._base)
            shape_wT.append(shap._wT)
            shape_sT.append(shap._sT)
            shape_sq.append(shap._sq)

        for shap in self.collision:
            shape_base.append(shap._base)
            shape_wT.append(shap._wT)
            shape_sT.append(shap._sT)
            shape_sq.append(shap._sq)

        return isflip, axis, jindex, parent, shape_base, shape_wT, shape_sT, shape_sq

    def _init_fknm(self):
        if isinstance(self.parent, str):
            # Initialise later
            return

        (
            isflip,
            axis,
            jindex,
            parent,
            shape_base,
            shape_wT,
            shape_sT,
            shape_sq,
        ) = self._get_fknm()

        self._fknm = fknm.link_init(
            self.isjoint,
            isflip,
            axis,
            jindex,
            len(shape_base),
            self._Ts,
            self._fk,
            shape_base,
            shape_wT,
            shape_sT,
            shape_sq,
            parent,
        )

    def _update_fknm(self):

        # Check if not initialized yet
        try:
            if self._fknm is None:
                self._init_fknm()
                return
        except AttributeError:
            return

        (
            isflip,
            axis,
            jindex,
            parent,
            shape_base,
            shape_wT,
            shape_sT,
            shape_sq,
        ) = self._get_fknm()

        fknm.link_update(
            self._fknm,
            self.isjoint,
            isflip,
            axis,
            jindex,
            len(shape_base),
            self._Ts,
            self._fk,
            shape_base,
            shape_wT,
            shape_sT,
            shape_sq,
            parent,
        )

    @property
    def ets(self: "ELink") -> "ETS":
        """
        Link transform in ETS form

        :return: elementary transform sequence for link transform
        :rtype: ETS or ETS2 instance

        The sequence:
            - has at least one element
            - may include zero or more constant transforms
            - no more than one variable transform, which if present will
              be last in the sequence
        """
        return self._ets

    # @property
    # def geometry(self):
    #     """
    #     Get/set joint visual geometry
    #     - ``link.geometry`` is the list of the visual geometries which
    #         represent the shape of the link
    #         :return: the visual geometries
    #         :rtype: list of Shape
    #     - ``link.geometry = ...`` checks and sets the geometry
    #     - ``link.geometry.append(...)`` add geometry
    #     """
    #     return self._geometry

    # @property
    # def collision(self):
    #     """
    #     Get/set joint collision geometry
    #     - ``link.collision`` is the list of the collision geometries which
    #         represent the collidable shape of the link.
    #         :return: the collision geometries
    #         :rtype: list of Shape
    #     - ``link.collision = ...`` checks and sets the collision geometry
    #     - ``link.collision.append(...)`` add collision geometry
    #     The collision geometries are what is used to check for collisions.
    #     """
    #     return self._collision

    # @collision.setter
    # def collision(self, coll):
    #     # Different from BaseELink due to self._update_fknm() required

    #     new_coll = []

    #     if isinstance(coll, list):
    #         for gi in coll:
    #             if isinstance(gi, Shape):
    #                 new_coll.append(gi)
    #             else:
    #                 raise TypeError("Collision must be of Shape class")
    #     elif isinstance(coll, Shape):
    #         new_coll.append(coll)
    #     else:
    #         raise TypeError("Geometry must be of Shape class or list of Shape")

    #     self._collision = new_coll
    #     # self._update_fknm()

    # @geometry.setter
    # def geometry(self, geom):
    #     # Different from BaseELink due to self._update_fknm() required
    #     new_geom = []

    #     if isinstance(geom, list):
    #         for gi in geom:
    #             if isinstance(gi, Shape):
    #                 new_geom.append(gi)
    #             else:
    #                 raise TypeError("Geometry must be of Shape class")
    #     elif isinstance(geom, Shape):
    #         new_geom.append(geom)
    #     else:
    #         raise TypeError("Geometry must be of Shape class or list of Shape")

    #     self._geometry = new_geom
    #     # self._update_fknm()

    # @property
    # def fk(self):
    #     """
    #     The forward kinemtics up to and including this link
    #     This value can be accessed after calling fkine_all(q)
    #     from the robot object.
    #     """

    #     return SE3(self._fk, check=False)

    def T(self, q: float = 0.0) -> ndarray:
        """
        Link transform matrix
        :param q: Joint coordinate (radians or metres). Not required for links
            with no variable

        :return T: link frame transformation matrix

        ``LINK.A(q)`` is an SE(3) matrix that describes the rigid-body
          transformation from the previous to the current link frame to
          the next, which depends on the joint coordinate ``q``.
        """
        if self.isjoint:
            return self._Ts @ self._ets[-1].T(q)
        else:
            return self._Ts


class ELink2(BaseELink):
    def __init__(self, ets: ETS2 = ETS2(), jindex: int = None, **kwargs):

        # process common options
        super().__init__(**kwargs)

        # check we have an ETS
        if not isinstance(ets, ETS2):
            raise TypeError("The ets argument must be of type ETS2")

        self._ets = ets

        # Set the jindex
        if len(ets) > 0 and ets[-1].isjoint:
            if jindex is not None:
                ets[-1].jindex = jindex

    @property
    def ets(self: "ELink2") -> "ETS2":
        """
        Link transform in ETS form

        :return: elementary transform sequence for link transform
        :rtype: ETS or ETS2 instance

        The sequence:
            - has at least one element
            - may include zero or more constant transforms
            - no more than one variable transform, which if present will
              be last in the sequence
        """
        return self._ets

    def T(self, q: float = 0.0) -> ndarray:
        """
        Link transform matrix

        :param q: Joint coordinate (radians or metres). Not required for links
            with no variable

        :return T: link frame transformation matrix

        ``LINK.A(q)`` is an SE(3) matrix that describes the rigid-body
          transformation from the previous to the current link frame to
          the next, which depends on the joint coordinate ``q``.

        If ``fast`` is True return a NumPy array, either SE(2) or SE(3).
        A value of None means that it is the identity matrix.
        """

        if self.isjoint:
            return self._Ts @ self._ets[-1].T(q)
        else:
            return self._Ts
