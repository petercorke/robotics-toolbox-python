#!/usr/bin/env python

"""
@author: Jesse Haviland
@author: Peter Corke
"""

from __future__ import annotations

# import sys
import inspect
import io
from abc import ABC
from copy import deepcopy
from functools import lru_cache
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    TypeVar,
    cast,
)
from warnings import warn

import numpy as np
from ansitable import ANSITable, Column
from spatialgeometry import SceneNode
from spatialmath import SE3
from spatialmath.base.argcheck import (
    getmatrix,
    getunit,
    getvector,
)
from typing_extensions import Literal as L

import roboticstoolbox as rtb
from roboticstoolbox.backends.Connector import Connector
from roboticstoolbox.ets.ET import ET
from roboticstoolbox.ets.ETS import ETS
from roboticstoolbox.ets.fknm import Robot_link_T
from roboticstoolbox.robot.Dynamics import DynamicsMixin
from roboticstoolbox.robot.Gripper import Gripper
from roboticstoolbox.robot.Link import BaseLink, Link, _dyn_version
from roboticstoolbox.robot.RobotPlottingMPL import RobotPlottingMPLMixin
from roboticstoolbox.tools.params import rtb_get_param
from roboticstoolbox.tools.types import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.typing import ColorType as Color  # pragma nocover
else:
    Color = None

# _default_backend = None

# A generic type variable representing any subclass of BaseLink
LinkType = TypeVar("LinkType", bound=BaseLink)

# Link attributes scanned for symbolic (e.g. SymPy) content at build time --
# DH kinematic parameters plus dynamics parameters; getattr(..., None) below
# skips whichever of these don't exist on a given Link subclass (e.g. a
# non-DH Link has no a/alpha/theta/d).
_SYMBOLIC_LINK_ATTRS = (
    "a", "alpha", "theta", "d", "offset",
    "m", "r", "I", "Jm", "G", "B", "Tc",
)


def _is_symbolic(value: Any) -> bool:
    """True if value holds non-numeric (e.g. SymPy) content -- same
    object-dtype test used by roboticstoolbox.ets.fknm._is_symbolic."""
    if value is None:
        return False
    return np.asarray(value).dtype == object


# the modules of the robot classes themselves, as distinct from models built from them
_FRAMEWORK_MODULES = ("roboticstoolbox.robot.", "roboticstoolbox.models.URDF.URDFRobot")


class BaseRobot(SceneNode, DynamicsMixin, RobotPlottingMPLMixin, ABC, Generic[LinkType]):
    # set when the links are processed in the constructor
    _base_link: LinkType

    #: default graphical backend for every robot of this class, used when the
    #: robot has not been given its own, see :meth:`set_default_backend`
    _class_default_backend: Any = None

    def __init__(
        self,
        links: list[LinkType],
        gripper_links: LinkType | list[LinkType] | None = None,
        name: str = "",
        manufacturer: str = "",
        comment: str = "",
        base: NDArray | SE3 | None = None,
        tool: NDArray | SE3 | None = None,
        gravity: ArrayLike = [0, 0, -9.81],
        keywords: list[str] | tuple[str, ...] = [],
        symbolic: bool = False,
        configs: dict[str, NDArray] | None = None,
        check_jindex: bool = True,
    ) -> None:
        # Initialise the scene node
        SceneNode.__init__(self)

        # Lets sort out links now
        self._linkdict: dict[str, LinkType] = {}

        # Sort links and set self.link, self.n, self.base_link,
        # self.ee_links
        self._sort_links(links, gripper_links, check_jindex)

        # Fix number of links for gripper links
        self._nlinks = len(links)

        for gripper in self.grippers:
            self._nlinks += len(gripper.links)

        # Set the pose of the robot in the world frame
        # in the scenenode object to a numpy array
        if isinstance(base, SE3):
            if len(base) != 1:
                raise ValueError("base must be a single SE3, not a batch")
            self._T = cast(NDArray, base.A)  # SE3.A typed list|ndarray; single pose always ndarray
        elif isinstance(base, np.ndarray):
            self._T = base

        # Set the robot tool transform
        if isinstance(tool, SE3):
            if len(tool) != 1:
                raise ValueError("tool must be a single SE3, not a batch")
            self._tool = cast(NDArray, tool.A)  # SE3.A typed list|ndarray; single pose always ndarray
        elif isinstance(tool, np.ndarray):
            self._tool = tool
        else:
            self._tool = np.eye(4)

        # Set the keywords
        if keywords is not None and not isinstance(keywords, (tuple, list)):
            raise TypeError("keywords must be a list or tuple")
        else:
            self._keywords = list(keywords)

        # Gravity is in the negative-z direction.
        self.gravity = np.array(gravity)

        # Basic arguments
        self.name = name
        self.manufacturer = manufacturer
        self.comment = comment
        self._symbolic = symbolic
        self._reach: float | None = None
        self._dyn_cache: tuple[int, bool] | None = None
        self._hasgeometry = False
        self._hascollision = False
        self._urdf_string = ""
        self._urdf_filepath = ""

        # Time to checkout the links for geometry information
        auto_symbolic = False
        for link in self.links:
            # Add link back to robot object
            link._robot = self

            if len(link.geometry) > 0:
                self._hasgeometry = True
            if link.collision:
                self._hascollision = True

            # Detect symbolic (e.g. SymPy) model parameters regardless of
            # whether the caller remembered to pass symbolic=True -- see
            # rne.md issue 3/4: forgetting the flag previously left
            # self.symbolic False even with genuinely symbolic link
            # parameters, which broke rne_python()'s own float64
            # allocation, not just the C dispatch. symbolic= is kept as an
            # override (OR'd in below), not the source of truth.
            if not auto_symbolic:
                for attr in _SYMBOLIC_LINK_ATTRS:
                    if _is_symbolic(getattr(link, attr, None)):
                        auto_symbolic = True
                        break

        self._symbolic = self._symbolic or auto_symbolic

        # Current joint configuration, velocity, acceleration
        self.q = np.zeros(self.n)
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)

        # The current control mode of the robot
        self.control_mode = "v"

        # Set up named configuration property
        if configs is None:
            configs = dict()
        self._configs = configs

        # A flag for watching dynamics properties
        self._frne_stale = False

        # Set up qlim
        qlim = np.zeros((2, self.n))
        j = 0

        for i in range(len(self.links)):
            if self.links[i].isjoint:
                qlim[:, j] = self.links[i].qlim
                j += 1
        self._qlim = qlim

        self._valid_qlim = False
        for i in range(self.n):
            if any(qlim[:, i] != 0) and not any(np.isnan(qlim[:, i])):
                self._valid_qlim = True

        # SceneNode, set a reference to the first link
        self.scene_children = [self.links[0]]

        self._default_backend = None
        self._active_plot_env: Any = None

    # --------------------------------------------------------------------- #
    # --------- Private Methods ------------------------------------------- #
    # --------------------------------------------------------------------- #

    def _sort_links(
        self,
        links: list[LinkType],
        gripper_links: LinkType | list[LinkType] | None,
        check_jindex: bool,
    ) -> None:
        """
        This method does several things for setting up the links of a robot

        - Gives each link a unique name if it doesn't have one
        - Assigns each link a parent if it doesn't have one
        - Finds and sets the base link
        - Finds and sets the ee links
        - sets the jindices
        - sets n
        - sets links

        """

        # The ordered links
        orlinks: list[LinkType] = []

        # The end-effector links
        self._ee_links: list[LinkType] = []

        # Check all the incoming Link objects
        n: int = 0

        # Make sure each link has a name
        # ------------------------------
        for k, link in enumerate(links):
            if not isinstance(link, BaseLink):
                raise TypeError("links should all be Link subclass")

            # If link has no name, give it one
            if link.name is None or link.name == "":
                link.name = f"link-{k}"

            link.number = k + 1

            # Put it in the link dictionary, check for duplicates
            if link.name in self._linkdict:
                raise ValueError(f"link name {link.name} is not unique")

            self._linkdict[link.name] = link

            if link.isjoint:
                n += 1

        # Resolve parents given by name, within the context of
        # this set of links
        # ----------------------------------------------------
        for link in links:
            if link.parent is None and link.parent_name is not None:
                link.parent = self._linkdict[link.parent_name]

        if all([link.parent is None for link in links]):
            # No parent links were given, assume they are sequential
            for i in range(len(links) - 1):
                # li = links[i]
                links[i + 1].parent = links[i]

        # Set the base link
        # -----------------
        for link in links:
            # Is this a base link?

            if isinstance(link.parent, BaseLink):
                # Update children of this link's parent
                link.parent._children.append(link)
            else:
                try:
                    if self._base_link is not None:
                        raise ValueError("Multiple base links")
                except AttributeError:
                    pass

                self._base_link = link

        if not hasattr(self, "_base_link"):
            raise ValueError(
                "Invalid link configuration provided, must have a base link"
            )

        # Scene node, set links between the links
        # ---------------------------------------
        for link in links:
            if isinstance(link.parent, BaseLink):
                link.scene_parent = link.parent

        # Set up the gripper, make a list containing the root of all
        # grippers
        # ----------------------------------------------------------
        if gripper_links is None:
            gripper_links = []

        if not isinstance(gripper_links, list):
            gripper_links = [gripper_links]

        # An empty list to hold all grippers
        self._grippers = []

        # Make a gripper object for each gripper
        for link in gripper_links:
            g_links = self.dfs_links(link)

            # Remove gripper links from the robot
            for g_link in g_links:
                # print(g_link)
                links.remove(g_link)

            # Save the gripper object
            self._grippers.append(Gripper(g_links, name=link.name))

        # Subtract the n of the grippers from the n of the robot
        for gripper in self._grippers:
            n -= gripper.n

        # Set the ee links
        # ----------------
        ee_links: list[LinkType] = []

        if len(gripper_links) == 0:
            for link in links:
                # Is this a leaf node? and do we not have any grippers
                if link.children is None or len(link.children) == 0:
                    # No children, must be an end-effector
                    ee_links.append(link)
        else:
            for link in gripper_links:
                # Use the passed in value
                if link.parent is not None:
                    ee_links.append(link.parent)

        self._ee_links = ee_links

        # Assign the joint indices and sort the links
        # -------------------------------------------
        if all([link.jindex is None or link.ets._auto_jindex for link in links]):
            # No joints have an index
            jindex = [0]  # "mutable integer" hack

            def visit_link(link: Any, jindex: Any) -> None:
                # if it's a joint, assign it a jindex and increment it
                if link.isjoint and link in links:
                    link.jindex = jindex[0]
                    jindex[0] += 1

                if link in links:
                    orlinks.append(link)

            # visit all links in DFS order
            self.dfs_links(self.base_link, lambda link: visit_link(link, jindex))

        elif all(
            [
                link.jindex is not None and not link.ets._auto_jindex
                for link in links
                if link.isjoint
            ]
        ):
            # Jindex set on all, check they are unique and contiguous
            if check_jindex:
                jset = set(range(n))
                jindices = {link.jindex for link in links if link.isjoint}
                for link in links:
                    if link.isjoint and link.jindex not in jset:
                        hint = ""
                        if jindices == set(range(1, n + 1)):
                            hint = (
                                f"; joint indices count from zero, but these are"
                                f" 1 to {n}.  Joint names that count from one,"
                                " such as 'A1', need a trailing underscore,"
                                " 'A1_', to be numbered from zero"
                            )
                        raise ValueError(
                            f"joint index {link.jindex} was repeated or out of"
                            f" range{hint}"
                        )
                    jset -= set([link.jindex])
                if len(jset) > 0:  # pragma nocover  # is impossible
                    raise ValueError(f"joints {jset} were not assigned")
            orlinks = links
        else:
            # must be a mixture of Links with/without jindex
            raise ValueError("all links must have a jindex, or none have a jindex")

        # Set n
        # -----
        self._n = n

        # Set links
        # ---------
        self._links = orlinks

    def dynchanged(self, what: str | None = None) -> None:
        """
        Dynamic parameters have changed

        Called from a property setter to inform the robot that the cache of
        dynamic parameters is invalid.

        .. seealso:: :func:`roboticstoolbox.Link._dirties_frne`

        """

        self._frne_stale = True

    # --------------------------------------------------------------------- #
    # --------- Magic Methods --------------------------------------------- #
    # --------------------------------------------------------------------- #

    def __iter__(self) -> Any:
        self._iter = 0
        return self

    def __next__(self) -> LinkType:
        if self._iter < len(self.links):
            link = self[self._iter]
            self._iter += 1
            return link
        else:
            raise StopIteration

    def __getitem__(self, i: int | str) -> LinkType:
        """
        Get link

        This also supports iterating over each link in the robot object,
        from the base to the tool.

        :param i: link number or name
        :returns: i'th link or named link

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> print(robot[1]) # print the 2nd link
            >>> print([link.a for link in robot])  # print all the a_j values

        .. rubric:: Notes

        ``Robot`` supports link lookup by name,
            eg. ``robot['link1']``

        """

        if isinstance(i, int):
            return self._links[i]
        else:
            return self._linkdict[i]

    def __str__(self) -> str:
        """
        Pretty prints the ETS Model of the robot.

        :returns: Pretty print of the robot model

        .. rubric:: Notes

        - Constant links are shown in blue.
        - End-effector links are prefixed with an @
        - Angles in degrees
        - The robot base frame is denoted as ``BASE`` and is equal to the
            robot's ``base`` attribute.

        """

        unicode = rtb_get_param("unicode")
        border = "thin" if unicode else "ascii"

        table = ANSITable(
            Column("link", headalign="^", colalign=">"),
            Column("link", headalign="^", colalign="<"),
            Column("joint", headalign="^", colalign=">"),
            Column("parent", headalign="^", colalign="<"),
            Column("ETS: parent to link", headalign="^", colalign="<"),
            border=border,
        )

        for k, link in enumerate(self.links):
            color = "" if link.isjoint else "<<blue>>"
            ee = "@" if link in self.ee_links else ""
            ets = link.ets
            if link.parent is None:
                parent_name = "BASE"
            else:
                parent_name = link.parent.name
            s = ets.__str__(f"q{link.jindex}")
            # if len(s) > 0:
            #     op = " \u2295 " if unicode else " * "  # \oplus
            #     s = op + s

            if link.isjoint:
                jname = link.jindex
            else:
                jname = ""  # type: ignore[assignment]
            table.row(
                # link.jindex,
                k,
                color + ee + link.name,
                jname,
                parent_name,
                f"{s}",
            )

        s = f"{self._class_label()}: {self.name}"
        if self.manufacturer is not None and len(self.manufacturer) > 0:
            s += f" (by {self.manufacturer})"
        s += f", {self.n} joints ({self.structure})"
        if len(self.grippers) > 0:
            s += (
                f", {len(self.grippers)} gripper{'s' if len(self.grippers) > 1 else ''}"
            )
        if self.nbranches > 1:
            s += f", {self.nbranches} branches"
        if self._hasdynamics:
            s += ", dynamics"
        if any([len(link.geometry) > 0 for link in self.links]):
            s += ", geometry"
        if any([len(link.collision) > 0 for link in self.links]):
            s += ", collision"
        s += "\n"

        s += str(table)
        s += self.configurations_str(border=border)

        return s

    def __repr__(self) -> str:
        return str(self)

    def _class_label(self) -> str:
        """
        The class of this robot, for the first line of its display

        ``Robot``, ``Robot2``, ``DHRobot`` or ``URDFRobot`` for an instance of the class
        itself, and ``X.Y`` for an instance of a subclass ``Y`` of one of them, for
        example ``DHRobot.Puma560`` or ``URDFRobot.Panda``: the kind of robot, then the
        model.
        """
        cls = type(self)
        if cls.__module__.startswith(_FRAMEWORK_MODULES):
            return cls.__name__
        return f"{cls.__mro__[1].__name__}.{cls.__name__}"

    # --------------------------------------------------------------------- #
    # --------- Properties ------------------------------------------------ #
    # --------------------------------------------------------------------- #

    @property
    def links(self) -> list[LinkType]:
        """
        Robot links

        :returns: A list of link objects

        .. rubric:: Notes

        It is probably more concise to index the robot object rather
        than the list of links, ie. the following are equivalent:
        - ``robot.links[i]``
        - ``robot[i]``

        """

        return self._links

    @property
    def link_dict(self) -> dict[str, LinkType]:
        return self._linkdict

    @property
    def grippers(self) -> list[Gripper]:
        """
        Grippers attached to the robot

        :returns: A list of grippers
        """

        return self._grippers

    @property
    def base_link(self) -> LinkType:
        """
        Get the robot base link

        - ``robot.base_link`` is the robot base link

        :returns: the first link in the robot tree
        """

        return self._base_link

    @property
    def ee_links(self) -> list[LinkType]:
        return self._ee_links

    @property
    def n(self) -> int:
        """
        Number of joints

        :returns: Number of joints

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.n

        .. seealso:: :func:`nlinks` :func:`nbranches`

        """

        return self._n

    @property
    def nlinks(self) -> Any:
        """
        Number of links

        The returned number is the total of both variable joints and
        static links

        :returns: Number of links

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.nlinks

        .. seealso:: :func:`n` :func:`nbranches`

        """

        return self._nlinks

    @property
    def nbranches(self) -> int:
        """
        Number of branches

        Number of branches in this robot.  Computed as the number of links with
        zero children

        :returns: number of branches in the robot's kinematic tree

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.ETS.Panda()
            >>> robot.nbranches

        .. seealso:: :func:`n` :func:`nlinks`
        """

        return sum([link.nchildren == 0 for link in self.links]) + len(self.grippers)

    # --------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        """
        Get/set robot name

        - ``robot.name`` is the robot name
        - ``robot.name = ...`` checks and sets the robot name

        :param name: the new robot name
        :returns: the current robot name
        """
        return self._name

    @name.setter
    def name(self, name_new: str) -> None:
        self._name = name_new

    @property
    def comment(self) -> str:
        """
        Get/set robot comment

        - ``robot.comment`` is the robot comment
        - ``robot.comment = ...`` checks and sets the robot comment

        :param name: the new robot comment
        :returns: robot comment
        """
        return self._comment

    @comment.setter
    def comment(self, comment_new: str) -> None:
        self._comment = comment_new

    @property
    def manufacturer(self) -> Any:
        """
        Get/set robot manufacturer's name

        - ``robot.manufacturer`` is the robot manufacturer's name
        - ``robot.manufacturer = ...`` checks and sets the manufacturer's name

        :returns: robot manufacturer's name
        """
        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, manufacturer_new: Any) -> None:
        self._manufacturer = manufacturer_new

    @property
    def configs(self) -> dict[str, NDArray]:
        """
        Named joint configurations

        :returns: dictionary of joint configurations, such as ``qr`` and ``qz``,
            indexed by name
        """
        return self._configs

    @property
    def keywords(self) -> list[str]:
        """
        Keywords that describe the robot

        :returns: the keywords, which the model catalog can filter on
        """
        return self._keywords

    @property
    def symbolic(self) -> bool:
        return self._symbolic

    @property
    def hasdynamics(self) -> bool:
        """
        Robot has dynamic parameters

        :returns: True if at least one link has a non-zero mass or motor inertia

        These are the parameters that the inertial dynamics need.  It is false for
        a model of the kinematics only, such as the default ``rtb.models.Panda``,
        and the dynamics methods ``gravload``, ``inertia``, ``coriolis``,
        ``accel``, ``fdyn``, ``itorque`` and ``rne`` raise ``ValueError`` for
        such a model, rather than return zeros or a confusing error.  It is *any*
        link and not all, since the first link of the DH Puma560 has a mass of
        zero.

        The answer is cached, and recomputed only when the mass or motor inertia
        of a link has been changed, so it costs nothing to ask.

        .. versionchanged:: 1.5.0
            This was true if any dynamic parameter had been assigned, even to
            zero, and was not reset when the parameters were set back to zero.
            A robot made from DH links with no masses reported True.  The
            dynamics methods now raise ``ValueError`` if it is False, where they
            returned zeros or raised ``LinAlgError``.

        .. warning:: **Behaviour change.**  Scripts that call the dynamics
            methods on a kinematics-only model, and accept the zeros, now get a
            ``ValueError``.

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.hasdynamics

        """

        version = _dyn_version()
        cache = self._dyn_cache
        if cache is None or cache[0] != version:
            # the mass or motor inertia of some link has changed since the last time
            cache = (version, any(link.hasdynamics for link in self.links))
            self._dyn_cache = cache
        return cache[1]

    @property
    def _hasdynamics(self) -> bool:
        # the name that the display and the catalog use
        return self.hasdynamics

    @property
    def hasgeometry(self) -> Any:
        """
        Robot has geometry model

        At least one link has associated mesh to describe its shape.

        :returns: Robot has geometry model

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.hasgeometry

        .. seealso:: :func:`hascollision`

        """

        return self._hasgeometry

    @property
    def hascollision(self) -> Any:
        """
        Robot has collision model

        :returns: True if at least one link has associated collision geometry.

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.hascollision

        .. seealso:: :func:`hasgeometry`

        """

        return self._hascollision

    @property
    def default_backend(self) -> Any:
        """
        Get default graphical backend

        - ``robot.default_backend`` Get the default graphical backend, used when
            no explicit backend is passed to ``Robot.plot``.  This is the value set
            for the robot, otherwise the value set for all robots by
            :meth:`set_default_backend`, otherwise ``None``.
        - ``robot.default_backend = ...`` Set the default graphical backend, used when
            no explicit backend is passed to ``Robot.plot``.  The default set here will
            be overridden if the particular ``Robot`` subclass cannot support it.

        :returns: backend name
        """
        if self._default_backend is not None:
            return self._default_backend
        return self._class_default_backend

    @default_backend.setter
    def default_backend(self, be: Any) -> None:
        self._default_backend = be

    @classmethod
    def set_default_backend(cls, be: Any) -> None:
        """
        Set the default graphical backend for all robots

        :param be: backend name, for example ``"swift"``, or ``None`` to clear

        ``Robot.set_default_backend("swift")`` sets the backend used by
        :meth:`plot` for every robot, existing or created later, that has not
        been given one of its own by ``robot.default_backend = ...``.  The robot's
        own value takes precedence.

        .. seealso:: :attr:`default_backend`
        """
        cls._class_default_backend = be

    @property
    def gravity(self) -> NDArray:
        """
        Get/set default gravitational acceleration

        - ``robot.gravity`` is the default gravitational acceleration
        - ``robot.gravity = ...`` checks and sets default gravitational
            acceleration

        :param gravity: gravitational acceleration in the world frame,
            downwards gravitational force is equivalent to robot base
            acceleration upwards (positive)
        :returns: gravitational acceleration

        """

        return self._gravity

    @gravity.setter
    def gravity(self, gravity_new: ArrayLike) -> None:
        self._gravity = np.array(getvector(gravity_new, 3))
        self.dynchanged()

    # --------------------------------------------------------------------- #

    @property
    def q(self) -> NDArray:
        """
        Get/set robot joint configuration

        - ``robot.q`` is the robot joint configuration
        - ``robot.q = ...`` checks and sets the joint configuration

        :param q: the new robot joint configuration
        :returns: robot joint configuration
        """

        return self._q

    @q.setter
    def q(self, q_new: ArrayLike) -> None:
        self._q = np.array(getvector(q_new, self.n))

    @property
    def qd(self) -> NDArray:
        """
        Get/set robot joint velocity

        - ``robot.qd`` is the robot joint velocity
        - ``robot.qd = ...`` checks and sets the joint velocity

        :returns: robot joint velocity
        """

        return self._qd

    @qd.setter
    def qd(self, qd_new: ArrayLike) -> None:
        self._qd = np.array(getvector(qd_new, self.n))

    @property
    def qdd(self) -> NDArray:
        """
        Get/set robot joint acceleration

        - ``robot.qdd`` is the robot joint acceleration
        - ``robot.qdd = ...`` checks and sets the robot joint acceleration

        :returns: robot joint acceleration
        """
        return self._qdd

    @qdd.setter
    def qdd(self, qdd_new: ArrayLike) -> None:
        self._qdd = np.array(getvector(qdd_new, self.n))

    @property
    def qlim(self) -> NDArray:
        r"""
        Joint limits

        Limits are extracted from the link objects.  If joints limits are
        not set for:

        - a revolute joint [-𝜋. 𝜋] is returned
        - a prismatic joint an exception is raised

        :param qlim: An array of joints limits (2, n)
        :raises ValueError: unset limits for a prismatic joint
        :returns: Array of joint limit values

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.qlim

        """

        limits = np.zeros((2, self.n))
        j = 0
        v: ArrayLike

        for link in self.links:
            if link.isrevolute:
                # a symbolic model holds SymPy numbers, which np.isnan rejects
                qlim = None if link.qlim is None else np.asarray(link.qlim, dtype=float)
                if qlim is None or np.any(np.isnan(qlim)):
                    v = [-np.pi, np.pi]
                else:
                    v = qlim
            elif link.isprismatic:
                if link.qlim is None:
                    raise ValueError("Undefined prismatic joint limit")
                else:
                    v = link.qlim
            else:
                # Fixed link
                continue  # pragma nocover

            limits[:, j] = v
            j += 1

        return limits

    @qlim.setter
    def qlim(self, new_qlim: ArrayLike) -> None:
        new_qlim = np.array(new_qlim)

        if new_qlim.shape != (2, self.n):
            raise ValueError("new_qlim must be of shape (2, n)")

        j = 0
        for link in self.links:
            if link.isjoint:
                link.qlim = new_qlim[:, j]
                j += 1

    @property
    def structure(self) -> str:
        """
        Return the joint structure string

        A string with one letter per joint: ``R`` for a revolute
        joint, and ``P`` for a prismatic joint.

        :returns: joint configuration string

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.structure
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.structure

        .. rubric:: Notes

        Fixed joints, that maintain a constant link relative pose,
        are not included.
        ``len(self.structure) == self.n``.

        """

        structure = []

        for link in self.links:
            if link.isrevolute:
                structure.append("R")
            elif link.isprismatic:
                structure.append("P")

        return "".join(structure)

    @property
    def prismaticjoints(self) -> list[bool]:
        """
        Prismatic joints as bool array

        :returns: array of joint type, True if prismatic

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.prismaticjoints
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.prismaticjoints

        .. rubric:: Notes

        Fixed joints, that maintain a constant link relative pose,
        are not included.

        .. seealso:: :meth:`~roboticstoolbox.robot.Link.Link.isprismatic` :func:`revolutejoints`

        """

        return [link.isprismatic for link in self.links if link.isjoint]

    @property
    def revolutejoints(self) -> list[bool]:
        """
        Revolute joints as bool array

        :returns: array of joint type, True if revolute

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.revolutejoints
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.revolutejoints

        .. rubric:: Notes

        Fixed joints, that maintain a constant link relative pose,
        are not included.

        .. seealso:: :meth:`~roboticstoolbox.robot.Link.Link.isrevolute` :func:`prismaticjoints`

        """

        return [link.isrevolute for link in self.links if link.isjoint]

    @property
    def control_mode(self) -> str:
        """
        Get/set robot control mode

        - ``robot.control_type`` is the robot control mode
        - ``robot.control_type = ...`` checks and sets the robot control mode

        :param control_mode: the new robot control mode
        :returns: the current robot control mode
        """

        return self._control_mode

    @control_mode.setter
    def control_mode(self, cn: str) -> None:
        if cn == "p" or cn == "v" or cn == "a":
            self._control_mode = cn
        else:
            raise ValueError("Control type must be one of 'p', 'v', or 'a'")

    # --------------------------------------------------------------------- #

    @property
    def urdf_string(self) -> str:
        return self._urdf_string

    @property
    def urdf_filepath(self) -> str:
        return self._urdf_filepath

    # --------------------------------------------------------------------- #

    @property
    def tool(self) -> SE3:
        """
        Get/set robot tool transform

        - ``robot.tool`` is the robot tool transform as an SE3 object
        - ``robot._tool`` is the robot tool transform as a numpy array
        - ``robot.tool = ...`` checks and sets the robot tool transform

        :param tool: the new robot tool transform (as an SE(3))
        :returns: robot tool transform
        """
        return SE3(self._tool, check=False)

    @tool.setter
    def tool(self, T: SE3 | NDArray) -> None:
        if isinstance(T, SE3):
            if len(T) != 1:
                raise ValueError("tool must be a single SE3, not a batch")
            self._tool = cast(NDArray, T.A)  # SE3.A typed list|ndarray; single pose always ndarray
        else:
            self._tool = T

    @property
    def base(self) -> SE3:
        """
        Get/set robot base transform

        - ``robot.base`` is the robot base transform
        - ``robot.base = ...`` checks and sets the robot base transform

        :param base: the new robot base transform
        :returns: the current robot base transform
        """

        # return a copy, otherwise somebody with
        # reference to the base can change it

        # This now returns the Scene Node transform
        # self._T is a copy of SceneNode.__T
        return SE3(self._T, check=False)

    @base.setter
    def base(self, T: NDArray | SE3) -> None:
        if isinstance(self, rtb.Robot):
            # All 3D robots
            # Set the SceneNode T
            if isinstance(T, SE3):
                if len(T) != 1:
                    raise ValueError("base must be a single SE3, not a batch")
                self._T = cast(NDArray, T.A)  # SE3.A typed list|ndarray; single pose always ndarray
            else:
                self._T = T

    # --------------------------------------------------------------------- #

    @lru_cache(maxsize=32)
    def get_path(
        self,
        end: Gripper | LinkType | str | None = None,
        start: Gripper | LinkType | str | None = None,
    ) -> tuple[list[LinkType], int, SE3]:
        """
        Find a path from start to end

        :param end: end-effector or gripper to compute forward kinematics to
        :param start: name or reference to a base link, defaults to None
        :raises ValueError: link not known or ambiguous
        :returns: tuple of the link path, the number of links in that path,
            and the tool transform present after the end link
        """

        def search(
            start: Any, end: Any, explored: set[LinkType | Link], path: list[LinkType | Link]
        ) -> list[LinkType | Link] | None:
            link = self._getlink(start, self.base_link)
            end = self._getlink(end, self.ee_links[0])

            toplevel = len(path) == 0
            explored.add(link)

            if link == end:
                return path

            # unlike regular DFS, the neighbours of the node are its children
            # and its parent.

            # visit child nodes below start
            if toplevel:
                path = [link]

            if link.children is not None:
                for child in link.children:
                    if child not in explored:
                        path.append(child)
                        p = search(child, end, explored, path)
                        if p is not None:
                            return p

            # We didn't find the node below, keep going up a level, and recursing
            # down again
            if toplevel:
                path = []

            if link.parent is not None:
                parent = link.parent  # go up one level toward the root
                if parent not in explored:
                    if len(path) == 0:
                        p = search(parent, end, explored, [link])
                    else:
                        path.append(link)
                        p = search(parent, end, explored, path)

                    if p is not None and len(p) > 0:
                        return p

            return None

        end, start, tool = self._get_limit_links(end=end, start=start)

        path = search(start, end, set(), [])

        if path is None or len(path) == 0:
            raise ValueError("No path found")  # pragma nocover
        elif path[-1] != end:
            path.append(end)

        if tool is None:
            tool = SE3()

        return path, len(path), tool

    @lru_cache(maxsize=32)
    def _getlink(
        self,
        link: LinkType | Gripper | str | None,
        default: LinkType | Gripper | str | None = None,
    ) -> LinkType | Link:
        """
        Validate reference to Link

        ``robot._getlink(link)`` is a validated reference to a Link within
        the ERobot ``robot``.  If ``link`` is:

        -  an ``Link`` reference it is validated as belonging to
          ``robot``.
        - a string, then it looked up in the robot's link name dictionary, and
          a Link reference returned.

        :param link: link
        :raises ValueError: link does not belong to this ERobot
        :raises TypeError: bad argument
        :returns: link reference
        """

        if link is None:
            link = default

        if isinstance(link, str):
            if link in self.link_dict:
                return self.link_dict[link]

            raise ValueError(f"no link named {link}")

        elif isinstance(link, BaseLink):
            if link in self.links:
                return link
            else:
                for gripper in self.grippers:
                    if link in gripper.links:
                        return link

                raise ValueError("link not in robot links")
        elif isinstance(link, Gripper):
            for gripper in self.grippers:
                if link is gripper:
                    return gripper.links[0]

            raise ValueError("Gripper not in robot")
        else:
            raise TypeError("unknown argument")

    def _find_ets(self, start: Any, end: Any, explored: Any, path: Any) -> ETS | None:
        """
        Privade method which will recursively find the ETS of a path
        see ets()
        """

        link = self._getlink(start, self.base_link)
        end = self._getlink(end, self.ee_links[0])

        toplevel = path is None
        explored.add(link)

        if link == end:
            return path

        # unlike regular DFS, the neighbours of the node are its children
        # and its parent.

        # visit child nodes below start
        if toplevel:
            path = link.ets

        if link.children is not None:
            for child in link.children:
                if child not in explored:
                    p = self._find_ets(child, end, explored, path * child.ets)
                    if p is not None:
                        return p

        # we didn't find the node below, keep going up a level, and recursing
        # down again
        if toplevel:
            path = None
        if link.parent is not None:
            parent = link.parent  # go up one level toward the root
            if parent not in explored:
                if path is None:
                    p = self._find_ets(parent, end, explored, link.ets.inv())
                else:
                    p = self._find_ets(parent, end, explored, path * link.ets.inv())
                if p is not None:
                    return p

        return None

    def _gripper_ets(self, gripper: Gripper) -> ETS:
        """
        Privade method which will find the ETS of a gripper

        """

        return ETS(ET.SE3(gripper.tool))

    @lru_cache(maxsize=32)
    def _get_limit_links(
        self,
        end: Gripper | LinkType | str | None = None,
        start: Gripper | LinkType | str | None = None,
    ) -> tuple[LinkType, LinkType, SE3 | None]:
        """
        Get and validate an end-effector, and a base link

        Helper method to find or validate an end-effector and base link

        :param end: end-effector or gripper, defaults to None
        :param start: name or reference to a base link, defaults to None
        :raises ValueError: a link is not known or is ambiguous
        :raises TypeError: an unsupported link argument is supplied
        :returns: tuple of end-effector link, start link, and gripper tool
            transform (or None if no gripper transform applies)
        """

        tool = None
        if end is None:
            if len(self.grippers) > 1:
                end_ret = self.grippers[0].links[0]
                tool = self.grippers[0].tool
                if len(self.grippers) > 1:
                    # Warn user: more than one gripper
                    print("More than one gripper present, using robot.grippers[0]")
            elif len(self.grippers) == 1:
                end_ret = self.grippers[0].links[0]
                tool = self.grippers[0].tool

            # no grippers, use ee link if just one
            elif len(self.ee_links) > 1:
                end_ret = self.ee_links[0]
                if len(self.ee_links) > 1:
                    # Warn user: more than one EE
                    print("More than one end-effector present, using robot.ee_links[0]")
            else:
                end_ret = self.ee_links[0]

        else:
            # Check if end corresponds to gripper
            for gripper in self.grippers:
                if end == gripper or end == gripper.name:
                    tool = gripper.tool
                    # end_ret = gripper.links[0]

            # otherwise check for end in the links
            end_ret = self._getlink(end)

        if start is None:
            start_ret = self.base_link

            # Cache result
            self._cache_start = start
        else:
            # start effector is specified
            start_ret = self._getlink(start)

        # because Gripper returns Link not LinkType
        return end_ret, start_ret, tool

    @lru_cache(maxsize=32)
    def ets(
        self,
        start: LinkType | Gripper | str | None = None,
        end: LinkType | Gripper | str | None = None,
    ) -> ETS:
        """
        Robot to ETS

        ``robot.ets()`` is an ETS representing the kinematics from base to
        end-effector.

        ``robot.ets(end=link)`` is an ETS representing the kinematics from
        base to the link ``link`` specified as a Link reference or a name.

        ``robot.ets(start=l1, end=l2)`` is an ETS representing the kinematics
        from link ``l1`` to link ``l2``.

        :param start: start link of the path, defaults to the base link
        :param end: end link of the path, defaults to the end-effector
        :raises ValueError: a link does not belong to this ERobot
        :raises TypeError: a bad link argument
        :returns: elementary transform sequence

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> panda = rtb.models.ETS.Panda()
            >>> panda.ets()

        """

        # ets to stand and end in case of grippers
        ets_init = None
        ets_end = None

        if isinstance(start, Gripper):
            ets_init = self._gripper_ets(start).inv()
            link = start.links[0].parent
            if link is None:  # pragma nocover
                raise ValueError("Invalid robot link configuration")
        else:
            link = self._getlink(start, self.base_link)

        if end is None:
            if len(self.grippers) > 1:
                end_link = self.grippers[0].links[0]
                ets_end = self._gripper_ets(self.grippers[0])
                print("multiple grippers present, ambiguous, using self.grippers[0]")
            elif len(self.grippers) == 1:
                end_link = self.grippers[0].links[0]
                ets_end = self._gripper_ets(self.grippers[0])
            elif len(self.ee_links) > 1:
                end_link = self._getlink(end, self.ee_links[0])
                print(
                    "multiple end-effectors present, ambiguous, using self.ee_links[0]"
                )
            else:
                end_link = self._getlink(end, self.ee_links[0])
        else:
            if isinstance(end, Gripper):
                ets_end = self._gripper_ets(end)
                end_link = end.links[0].parent
                if end_link is None:  # pragma nocover
                    raise ValueError("Invalid robot link configuration")
            else:
                end_link = self._getlink(end, self.ee_links[0])

        explored: set[Any] = set()

        if link is end_link:
            ets = link.ets
        else:
            ets = self._find_ets(link, end_link, explored, path=None)

        if ets is None:
            raise ValueError(
                "Could not find the requested ETS in this robot"
            )  # pragma nocover

        if ets_init is not None:
            ets = ets_init * ets

        if ets_end is not None:
            ets = ets * ets_end

        return ets

    def copy(self) -> Any:
        """
        Make a copy of the robot

        :returns: a deep copy, which shares no state with the original
        """
        return deepcopy(self)

    def __deepcopy__(self, memo: Any) -> Any:
        links = []

        # if isinstance(self, rtb.DHRobot):
        #     cls = rtb.DHRobot
        cls: type[BaseRobot[Any]]
        if isinstance(self, rtb.Robot2):
            cls = rtb.Robot2
        else:
            cls = rtb.Robot

        for link in self.links:
            links.append(deepcopy(link))

        name = deepcopy(self.name)
        manufacturer = deepcopy(self.manufacturer)
        comment = deepcopy(self.comment)
        base = deepcopy(self.base)
        tool = deepcopy(self.tool)
        gravity = deepcopy(self.gravity)
        keywords = deepcopy(self.keywords)
        symbolic = deepcopy(self.symbolic)
        configs = deepcopy(self.configs)

        result = cls(
            links,
            name=name,
            manufacturer=manufacturer,
            comment=comment,
            base=base,
            tool=tool,
            gravity=gravity,
            keywords=keywords,
            symbolic=symbolic,
            configs=configs,
        )

        # if a configuration was an attribute of original robot, make it an
        # attribute of the deep copy
        for config in configs:
            if hasattr(self, config):
                setattr(result, config, configs[config])

        memo[id(self)] = result
        return result

    # --------------------------------------------------------------------- #

    def todegrees(self, q: Any) -> NDArray:
        """
        Convert joint angles to degrees

        :param q: joint configuration as a vector or matrix (one row per pose)
        :returns: joint coordinates with revolute values converted to degrees
            and prismatic values unchanged, preserving a matrix input's rows

        A vector is converted elementwise. If ``q`` is a matrix, with one
        column per joint, the conversion is performed columnwise.

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> from math import pi
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.todegrees([pi/4, pi/8, 2, -pi/4, pi/6, pi/3])

        """

        q = getmatrix(q, (None, self.n))

        for j, revolute in enumerate(self.revolutejoints):
            if revolute:
                q[:, j] *= 180.0 / np.pi

        if q.shape[0] == 1:
            return q[0]
        else:
            return q

    def toradians(self, q: Any) -> NDArray:
        """
        Convert joint angles to radians

        ``robot.toradians(q)`` converts joint coordinates ``q`` to radians
        taking into account whether elements of ``q`` correspond to revolute
        or prismatic joints, ie. prismatic joint values are not converted.

        If ``q`` is a matrix, with one column per joint, the conversion is
        performed columnwise.

        :param q: The joint configuration of the robot
        :returns: a vector of joint coordinates in radians and metres

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.toradians([10, 20, 2, 30, 40, 50])

        """

        q = getmatrix(q, (None, self.n))

        for j, revolute in enumerate(self.revolutejoints):
            if revolute:
                q[:, j] *= np.pi / 180.0

        if q.shape[0] == 1:
            return q[0]
        else:
            return q

    def isrevolute(self, j: Any) -> bool:
        """
        Check if joint is revolute

        :returns: True if revolute

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.revolutejoints
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.isrevolute(1)

        .. seealso:: :meth:`~roboticstoolbox.robot.Link.Link.isrevolute` :func:`revolutejoints`

        """
        return self.revolutejoints[j]

    def isprismatic(self, j: Any) -> bool:
        """
        Check if joint is prismatic

        :returns: True if prismatic

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.prismaticjoints
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.isprismatic(1)

        .. seealso:: :meth:`~roboticstoolbox.robot.Link.Link.isprismatic` :func:`prismaticjoints`

        """

        return self.prismaticjoints[j]

    # --------------------------------------------------------------------- #

    def dfs_links(
        self,
        start: LinkType,
        func: Callable[[LinkType | None], Any] | None = None,
    ) -> list[LinkType]:
        """
        A link search method

        Visit all links from start in depth-first order and will apply
        func to each visited link

        :param start: The link to start at
        :param func: An optional function to apply to each link as it is found
        :returns: A list of links
        """

        visited = []

        def vis_children(link: Any) -> None:
            visited.append(link)
            if func is not None:
                func(link)

            for li in link.children:
                if li not in visited:
                    vis_children(li)

        vis_children(start)

        return visited

    def addconfiguration_attr(self, name: str, q: ArrayLike, unit: str = "rad") -> None:
        """
        Add a named joint configuration as an attribute

        :param name: Name of the joint configuration
        :param q: Joint configuration

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.addconfiguration_attr("mypos", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            >>> robot.mypos
            >>> robot.configs["mypos"]

        .. rubric:: Notes

        - Used in robot model init method to store the ``qr`` configuration
        - Dynamically adding attributes to objects can cause issues with
            Python type checking.
        - Configuration is also added to the robot instance's dictionary of
            named configurations.

        .. seealso:: :meth:`addconfiguration`

        """

        v = getunit(q, unit, dim=self.n)
        self._configs[name] = v
        setattr(self, name, v)

    def addconfiguration(self, name: str, q: ArrayLike) -> None:
        """
        Add a named joint configuration

        Add a named configuration to the robot instance's dictionary of named
        configurations.

        :param name: Name of the joint configuration
        :param q: Joint configuration

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.addconfiguration_attr("mypos", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            >>> robot.configs["mypos"]

        .. seealso:: :meth:`addconfiguration`

        """

        self._configs[name] = np.array(getvector(q, self.n))

    def configurations_str(self, border: str = "thin") -> Any:
        deg = 180 / np.pi

        # TODO: factor this out of DHRobot
        def angle(theta: Any, fmt: Any = None) -> Any:
            if fmt is not None:
                try:
                    return fmt.format(theta * deg) + "\u00b0"
                except TypeError:  # pragma nocover
                    pass

            return str(theta * deg) + "\u00b0"  # pragma nocover

        # show named configurations
        if len(self._configs) > 0:
            table = ANSITable(
                Column("name", colalign=">"),
                *[
                    Column(f"q{j:d}", colalign="<", headalign="<")
                    for j in range(self.n)
                ],
                border=border,
            )

            for name, q in self._configs.items():
                qlist = []
                for j, c in enumerate(self.structure):
                    if c == "P":
                        qlist.append(f"{q[j]: .3g}")
                    else:
                        qlist.append(angle(q[j], "{: .3g}"))
                table.row(name, *qlist)

            return "\n" + str(table)
        else:  # pragma nocover
            return ""

    def random_q(self) -> Any:
        """
        Return a random joint configuration

        The value for each joint is uniform randomly distributed  between the
        limits set for the robot.

        .. rubric:: Notes

        The joint limit for all joints must be set.

        :returns: Random joint configuration :rtype: ndarray(n)

        .. seealso:: :meth:`~roboticstoolbox.robot.Robot.Robot.qlim` :meth:`~roboticstoolbox.robot.Link.Link.qlim`

        """

        qlim = self.qlim
        if np.any(np.isnan(qlim)):
            raise ValueError("some joint limits not defined")  # pragma nocover
        return np.random.uniform(low=qlim[0, :], high=qlim[1, :], size=(self.n,))

    def hierarchy(self) -> None:
        """
        Pretty print the robot link hierarchy

        :returns: Pretty print of the robot model

        .. rubric:: Examples

        Makes a robot and prints the hierarchy

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Panda()
        >>> robot.hierarchy()

        """

        def recurse(link: Any, indent: int = 0) -> None:
            print(" " * indent * 2, link.name)
            if link.children is not None:
                for child in link.children:
                    recurse(child, indent + 1)

        recurse(self.base_link)

    def segments(self) -> list[list[LinkType | None]]:
        """
        Segments of branched robot

        For a single-chain robot with structure::

            L1 - L2 - L3

        the return is ``[[None, L1, L2, L3]]``

        For a robot with structure::

            L1 - L2 +-  L3 - L4
                    +- L5 - L6

        the return is ``[[None, L1, L2], [L2, L3, L4], [L2, L5, L6]]``

        :returns: Segment list

        .. rubric:: Notes

        - the length of the list is the number of segments in the robot
        - the first segment always starts with ``None`` which represents
            the base transform (since there is no base link)
        - the last link of one segment is also the first link of subsequent
            segments
        """

        def recurse(link: Link) -> Any:
            segs = [link.parent]
            while True:
                segs.append(link)
                children = link.children or []
                if len(children) == 0:
                    return [segs]
                elif len(children) == 1:
                    link = children[0]
                    continue
                elif len(children) > 1:
                    segs = [segs]  # type: ignore[list-item]

                    for child in children:
                        segs.extend(recurse(child))

                    return segs

        return recurse(self.links[0])

    # --------------------------------------------------------------------- #
    # Scene Graph section
    # --------------------------------------------------------------------- #

    def _update_link_tf(self, q: ArrayLike | None = None) -> None:
        """
        This private method updates the local transform of each link within
        this robot according to q (or self.q if q is none)
        """

        @lru_cache(maxsize=2)
        def get_link_ets() -> Any:
            return [link.ets._fknm for link in self.links]

        @lru_cache(maxsize=2)
        def get_link_scene_node() -> Any:
            return [link._T_reference for link in self.links]

        Robot_link_T(get_link_ets(), get_link_scene_node(), self._q, q)

        for gripper in self.grippers:
            gripper._update_link_tf()

    # --------------------------------------------------------------------- #
    # --------- PyPlot Methods -------------------------------------------- #
    # --------------------------------------------------------------------- #

    def _get_graphical_backend(
        self,
        backend: L["swift", "pyplot", "pyplot2"] | None = None,
    ) -> Connector:
        import sys

        from roboticstoolbox.backends import load_backend

        # Resolve which backend name to use
        if backend is not None:
            using_backend = backend.lower()
        else:
            # In JupyterLite/Pyodide, only pyplot/pyplot2 are available
            if sys.platform == "emscripten":
                if isinstance(self, rtb.Robot2):
                    using_backend = "pyplot2"
                else:
                    using_backend = "pyplot"
            else:
                # Infer from robot type and robot-level default override
                using_backend = self.default_backend
                if using_backend is None:
                    if isinstance(self, rtb.DHRobot):
                        using_backend = "pyplot"
                    elif isinstance(self, rtb.Robot2):
                        using_backend = "pyplot2"
                    elif isinstance(self, rtb.Robot):
                        using_backend = "swift" if self.hasgeometry else "pyplot"
                    else:
                        using_backend = "pyplot"

        # swift is optional; fall back to pyplot with a helpful message
        if using_backend == "swift":  # pragma nocover
            if isinstance(self, rtb.DHRobot):
                raise NotImplementedError(
                    "Plotting in Swift is not implemented for DHRobots yet"
                )
            try:
                return load_backend("swift")
            except ModuleNotFoundError:
                if backend is not None:
                    # user explicitly asked for swift
                    print(
                        "Swift is not installed. "
                        "Install it with: pip install swift-sim"
                    )
                using_backend = "pyplot"

        return load_backend(using_backend)

    def plot(
        self,
        q: ArrayLike,
        backend: L["swift", "pyplot", "pyplot2"] | None = None,
        block: bool = False,
        dt: float = 0.05,
        limits: ArrayLike | None = None,
        vellipse: bool = False,
        fellipse: bool = False,
        fig: str | None = None,
        movie: str | None = None,
        loop: bool = False,
        notebook: bool | None = None,
        ax: Any = None,
        env: Connector | None = None,
        view: str | None = None,
        **kwargs: Any,
    ) -> Connector:
        """
        Graphical display and animation

        ``robot.plot(q, 'pyplot')`` displays a graphical view of a robot
        based on the kinematic model and the joint configuration ``q``.
        This is a stick figure polyline which joins the origins of the
        link coordinate frames. The plot will autoscale with an aspect
        ratio of 1.

        If ``q`` (m,n) representing a joint-space trajectory it will create an
        animation with a pause of ``dt`` seconds between each frame.

        :param q: The joint configuration of the robot.
        :param backend: The graphical backend to use, currently 'swift' and 'pyplot' are implemented. Defaults to 'swift' of a ``Robot`` and 'pyplot` for a ``DHRobot``
        :param block: ``True`` waits until the window is closed; the PyPlot backend has
            already shown the figure, so ``False`` and ``None`` return at once,
            defaults to False
        :param dt: if q is a trajectory, this describes the delay in seconds between frames
        :param limits: Custom view limits for the plot. If not supplied will autoscale, [x1, x2, y1, y2, z1, z2] (this option is for 'pyplot' only)
        :param vellipse: (Plot Option) Plot the velocity ellipse at the end-effector (this option is for 'pyplot' only)
        :param fellipse: (Plot Option) Plot the force ellipse at the end-effector (this option is for 'pyplot' only)
        :param fig: (Plot Option) The figure label to plot in (this option is for 'pyplot' only)
        :param ax: (Plot Option) Existing Matplotlib 3D axes to draw the scene on, instead of making a figure (this option is for 'pyplot' only)
        :param env: The scene returned by an earlier ``plot``, to add this robot to, instead of making a new scene; see *Several configurations* below.  The backend of the scene is used, so ``backend`` is ignored, and the options that are set when a scene is made (``limits``, ``fig``, ``ax``, ``render_mode``, ``inline_*``, ``notebook`` and ``browser``) cannot be given
        :param color: (Plot Option) The color of the robot's line, any Matplotlib color (this option is for 'pyplot' only)
        :param view: (Plot Option) ``"xy"``, ``"xz"`` or ``"yz"`` to start with a flat view of that plane, the first axis to the right and the second up, or ``None`` (default) for the usual three-dimensional view.  The view is not locked, the plot can be rotated as usual.  With ``env`` it changes the view of that scene (this option is for 'pyplot' only)
        :param movie: (Plot Option) The filename to save the movie to (this option is for 'pyplot' only)
        :param loop: (Plot Option) Loop the movie (this option is for 'pyplot' only)
        :param notebook: (Plot Option, Swift only) ``True`` shows Swift in a frame of the current notebook cell, ``False`` opens it in a browser window, and ``None`` (default) chooses the frame in a notebook, unless a ``browser`` is named, and a window otherwise, so the same call works in a script and in a notebook.  The browser showing the notebook must be on the machine running the kernel, since the frame points at ``localhost``
        :param browser: (Plot Option, Swift only) the browser for the window, or ``None`` (default) for the system's.  This is a name known to Python's ``webbrowser`` module, which depends on the operating system, for example ``"chrome"``, ``"firefox"`` or ``"safari"`` on macOS and ``"google-chrome"`` or ``"chromium"`` on Linux, or a command line with ``%s`` where the URL goes, for any other browser; ``webbrowser.get(name)`` tells you whether a name works.  Naming a browser means a window, not the notebook frame.  ``browser="notebook"`` is deprecated, use ``notebook=True``.  See :py:meth:`swift.Swift.Swift.launch`
        :param realtime: (Plot Option) Pace the steps of a trajectory in real time, one each ``dt``, defaults to True for a trajectory (this option is for 'swift' only)
        :param jointaxes: (Plot Option) Plot an arrow indicating the axes in which the joint revolves around(revolute joint) or translates along (prosmatic joint) (this option is for 'pyplot' only)
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame at the location of the end-effector. Uses three arrows, red, green and blue to indicate the x, y, and z-axes. (this option is for 'pyplot' only)
        :param shadow: (Plot Option) Plot a shadow of the robot in the x-y plane. (this option is for 'pyplot' only)
        :param name: (Plot Option) Plot the name of the robot near its base (this option is for 'pyplot' only)
        :param render_mode: (Plot Option) Rendering mode for matplotlib backends: ``'window'``, ``'notebook-widget'``, or ``'notebook-inline'``. If omitted, a mode is chosen from the environment, so the same call works for any ``%matplotlib`` setting: for the interactive ``ipympl`` backend a single pose is a widget that can be rotated, but a trajectory uses ``'notebook-inline'``, since a widget cannot update while the cell is running.
        :param inline_every_n: (Plot Option) In notebook-inline mode, push one rendered frame every N simulation steps. Larger N reduces output load.
        :param inline_format: (Plot Option) In notebook-inline mode, frame format: ``'svg'`` (default) or ``'png'``.
        :param inline_dpi: (Plot Option) DPI for PNG inline frames only; ignored when ``inline_format='svg'``.
        :returns: A reference to the environment object which controls the figure

        .. note::

            Python's ``webbrowser`` module cannot tell whether a browser is installed, so
            a browser that cannot be opened is not reported as such.  On macOS the names
            ``"chrome"``, ``"firefox"`` and ``"safari"`` are always accepted, whether or
            not the app is installed.  If it is not, no window opens and there is no
            error: Swift waits about 10 seconds for the page to connect and then reports
            ``Could not connect to the Swift simulator``.  If you see that, check that
            the browser named by ``browser`` is installed, or leave it as ``None`` to use
            the system's default browser.  See :py:meth:`swift.Swift.Swift.launch`.

        .. rubric:: Several configurations

        ``plot`` returns the scene it made, and passing that as ``env`` to another
        ``plot`` adds a robot to it instead of making a new scene.  The robot is the
        same model, and each plot of it holds the pose it was given, so the robot is
        shown in two configurations at once, here the initial and the final.  Give each
        a ``color``, and turn off what would be drawn twice.

        .. plot::
            :include-source:

            import roboticstoolbox as rtb

            puma = rtb.models.DH.Puma560()
            env = puma.plot(puma.qr, color="gray", block=False)
            puma.plot(puma.qn, env=env, color="red", name=False, block=False)

        The scene can be given a pose to draw directly, ``handle = env.add(puma, q=q)``,
        and ``handle.q = q`` moves only that drawing.  A robot that is added without a
        ``q`` follows ``puma.q``, as it always has.  A trajectory can be plotted on top
        of an earlier plot in the same way; the earlier plot stays where it was.  In a
        notebook, an animation needs the scene to have been made with
        ``render_mode="notebook-inline"``.

        .. rubric:: Notes

        - By default this method will block until the figure is dismissed.
            To avoid this set ``block=False``.
        - For PyPlot, the polyline joins the origins of the link frames,
            but for some Denavit-Hartenberg models those frames may not
            actually be on the robot, ie. the lines to not necessarily
            represent the links of the robot.

        .. seealso:: :func:`teach`

        """

        # a scene to add to, or a new one
        adding = env is not None
        if env is None:
            env = self._get_graphical_backend(backend)
        elif ax is not None:
            raise ValueError("ax and env cannot both be given: the scene has axes")

        if movie is not None:
            from roboticstoolbox.backends.PyPlot import PyPlot

        launch_kwargs: dict[str, Any] = {}
        for key in (
            "render_mode",
            "inline_every_n",
            "inline_format",
            "inline_dpi",
            "browser",
            "realtime",
        ):
            if key in kwargs:
                launch_kwargs[key] = kwargs.pop(key)

        if view is not None:
            from roboticstoolbox.tools.plot import plane_view

            plane_view(view)  # a bad name raises before a scene is made
            if "view" not in inspect.signature(env.launch).parameters:
                raise ValueError(f"the {type(env).__name__} backend cannot set a view")
            if not adding:
                launch_kwargs["view"] = view

        if adding:
            # these are fixed when a scene is made
            fixed = [k for k in launch_kwargs if k != "realtime"]
            if limits is not None:
                fixed.append("limits")
            if fig is not None:
                fixed.append("fig")
            if notebook is not None:
                fixed.append("notebook")
            if fixed:
                raise ValueError(
                    f"{', '.join(fixed)} cannot be used with env, they are set when "
                    "the scene is made"
                )
        if ax is not None:
            if "ax" not in inspect.signature(env.launch).parameters:
                raise ValueError(f"the {type(env).__name__} backend cannot plot on ax")
            launch_kwargs["ax"] = ax

        q = np.array(getmatrix(q, (None, self.n)))
        self.q = q[0, :]

        # a trajectory is an animation, which the PyPlot backends render differently
        # in a notebook (the other backends have no render mode)
        if q.shape[0] > 1 and hasattr(env, "render_mode") and not adding:
            launch_kwargs.setdefault("animation", True)

        # Swift is a web page, served by this process.  In a notebook it is shown in a
        # frame of the cell, not in a browser window of its own.  An animation is paced
        # in real time, one step to each dt: unpaced, Swift draws at most 60 frames
        # per second and skips the steps that come sooner, so the whole trajectory runs
        # in a few milliseconds and only its first frame is ever drawn.
        is_swift = type(env).__name__ == "Swift"
        if is_swift and not adding:
            from roboticstoolbox.backends.PyPlot.PyPlot import _isnotebook

            if launch_kwargs.get("browser") == "notebook":
                warn(
                    'plot(browser="notebook") is deprecated, use plot(notebook=True)',
                    FutureWarning,
                    stacklevel=2,
                )
                del launch_kwargs["browser"]
                notebook = True
            if notebook is None:
                notebook = launch_kwargs.get("browser") is None and _isnotebook()
            if notebook:
                if launch_kwargs.get("browser") is not None:
                    raise ValueError(
                        f"browser={launch_kwargs['browser']!r} cannot be used with "
                        "notebook=True: the page is shown in the cell, not in a window"
                    )
                # Swift 2.1 has notebook=, before that it was browser="notebook"
                if "notebook" in inspect.signature(env.launch).parameters:
                    launch_kwargs["notebook"] = True
                else:
                    launch_kwargs["browser"] = "notebook"
            launch_kwargs.setdefault("realtime", q.shape[0] > 1)

        # Add the self to the figure in readonly mode
        if adding:
            pass  # the scene was launched by the plot that made it
        elif q.shape[0] == 1:
            env.launch(
                name=self.name + " Plot", limits=limits, fig=fig, **launch_kwargs
            )
        else:
            env.launch(
                name=self.name + " Trajectory Plot",
                limits=limits,
                fig=fig,
                **launch_kwargs,
            )

        if adding and view is not None:
            from roboticstoolbox.tools.plot import set_plane_view

            set_plane_view(env.ax, view)  # type: ignore[attr-defined]

        # The drawing of this robot holds the pose it is drawn in, if the backend lets
        # it, so the same robot can be added to a scene again in another pose.
        if "q" in inspect.signature(env.add).parameters:
            kwargs["q"] = q[0]
        handle = env.add(self, readonly=True, **kwargs)
        self._active_plot_env = env

        if vellipse:
            vell = self.vellipse(q[0], centre="ee", add=False)
            env.add(vell)
        else:
            vell = None

        if fellipse:
            fell = self.fellipse(q[0], centre="ee", add=False)
            env.add(fell)
        else:
            fell = None

        # List of images saved from each plot
        images = []

        if movie is not None:  # pragma: nocover
            loop = False

        while True:
            for qk in q:
                if hasattr(handle, "q"):
                    # The drawing keeps the pose of what it shows, as Swift does in the
                    # handle that add() returned, and the robot is only a model.
                    # Swift deprecates driving it through robot.q.
                    handle.q = qk
                else:
                    self.q = qk
                if vell is not None:
                    vell.q = qk
                if fell is not None:
                    fell.q = qk
                env.step(dt)

                if movie is not None and isinstance(env, PyPlot):  # pragma nocover
                    images.append(env.getframe())

            if movie is not None:  # pragma nocover
                # save it as an animated GIF
                images[0].save(
                    movie,
                    save_all=True,
                    append_images=images[1:],
                    optimize=False,
                    duration=dt,
                    loop=0,
                )
            if not loop:
                break

        self.q = q[-1]

        # Keep the plot open
        if block:  # pragma nocover
            env.hold()

        return env

    def teach(
        self,
        q: ArrayLike | None,
        block: bool = True,
        limits: ArrayLike | None = None,
        vellipse: bool = False,
        fellipse: bool = False,
        backend: L["swift", "pyplot", "pyplot2"] | None = None,
        view: str | None = None,
    ) -> Connector:
        """
        Graphical teach pendant

        ``robot.teach(q)`` opens a graphical view (PyPlot or Swift --
        see ``backend``) which allows the user to "drive" a graphical
        robot using a graphical slider panel. The robot's initial joint
        configuration is ``q``.

        ``robot.teach()`` as above except the robot's stored value of ``q``
        is used.

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :param block: ``True`` waits until the window is closed; the PyPlot backend has
            already shown the figure, so ``False`` and ``None`` return at once,
            defaults to True
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2] (this option is for
            'pyplot'/'pyplot2' only)
        :param vellipse: (Plot Option) Plot the velocity ellipse at the
            end-effector (this option is for 'pyplot' only)
        :param fellipse: (Plot Option) Plot the force ellipse at the
            end-effector (this option is for 'pyplot' only)
        :param backend: The graphical backend to use -- 'swift', 'pyplot',
            or 'pyplot2'. Defaults to whatever :meth:`plot` would pick for
            this robot (see its own ``backend`` for the resolution rule).
        :param view: ``"xy"``, ``"xz"`` or ``"yz"`` to start with a flat view of
            that plane, the first axis to the right and the second up, or ``None``
            (default) for the usual three-dimensional view.  The view is not
            locked, the plot can be rotated as usual (this option is for 'pyplot'
            only)

        :returns: A reference to the environment object which controls
            the figure/view

        .. rubric:: Notes

        - Program execution is blocked until the teach window is
          dismissed.  If ``block=False`` the method is non-blocking but
          you need to poll the window manager (PyPlot) or keep calling
          ``env.step()`` yourself (Swift) to ensure the window/panel
          remains responsive.
        - Once ``teach()`` returns (``block=True``), ``robot.q`` holds
          the final pose the sliders were left at -- true for every
          backend, even though Swift's own live joint state during the
          session lives on the returned environment's handle
          (``env.swift_objects[0].q``), not ``robot.q`` itself, to keep
          the robot model plain and shareable while teaching is
          in progress (see swift's ``AssemblyHandle``).
        - The slider limits are derived from the joint limit properties.
          If not set then:

          - For revolute joints they are assumed to be [-pi, +pi]
          - For prismatic joint they are assumed unknown and an error
            occurs.

        """

        if q is None:
            q = np.zeros((self.n,))
        else:
            q = getvector(q, self.n)

        # Make an empty 3D figure
        env = self._get_graphical_backend(backend)

        if not getattr(env, "supports_teach", True):  # pragma: nocover
            raise TypeError(
                f"teach() is not supported by the {type(env).__name__} backend"
            )

        # Add the self to the figure in readonly mode
        launch_kwargs: dict[str, Any] = {}
        if view is not None:
            from roboticstoolbox.tools.plot import plane_view

            plane_view(view)  # a bad name raises before a scene is made
            if "view" not in inspect.signature(env.launch).parameters:
                raise ValueError(f"the {type(env).__name__} backend cannot set a view")
            launch_kwargs["view"] = view
        env.launch(name="Teach " + self.name, limits=limits, **launch_kwargs)
        handle = env.add(
            self,
            readonly=True,
            # jointaxes=jointaxes,
            # jointlabels=jointlabels,
            # eeframe=eeframe,
            # shadow=shadow,
            # name=name,
        )
        self._active_plot_env = env

        # True if _add_teach_panel already fully handled block=True itself
        # (Swift's own env.run() loop -- see its docstring) -- PyPlot/
        # PyPlot2 never do (their own env.hold() below is what actually
        # enters matplotlib's blocking GUI mainloop), so their
        # _add_teach_panel implicitly returns None here, same as always.
        already_blocked = env._add_teach_panel(self, q, handle, block)

        if vellipse:
            vell = self.vellipse(q, centre="ee", add=False)
            # the ellipsoid shown by the teach pendant, read back by the
            # backend; not part of the backend interface
            setattr(env, "_teach_vellipse", vell)
            env.add(vell)

        if fellipse:
            fell = self.fellipse(q, centre="ee", add=False)
            setattr(env, "_teach_fellipse", fell)
            env.add(fell)

        # Keep the plot open -- skipped if the backend already blocked
        # itself above: calling this a second time isn't just wasteful,
        # it can hang outright for Swift (env.hold()'s disconnect-poll
        # never expires in headless mode, and even non-headless there's
        # a race where close() doesn't synchronously guarantee the
        # websocket's been dropped from socket.USERS before returning).
        if block and not already_blocked:  # pragma nocover
            env.hold()

        return env

    # --------------------------------------------------------------------- #

    # --------------------------------------------------------------------- #
    # --------- Utility Methods ------------------------------------------- #
    # --------------------------------------------------------------------- #

    def showgraph(self, display_graph: bool = True, **kwargs: Any) -> str | None:
        """
        Display a link transform graph in browser

        ``robot.showgraph()`` displays a graph of the robot's link frames
        and the ETS between them.  It uses GraphViz dot.

        The nodes are:
            - Base is shown as a grey square. This is the world frame origin,
              but can be changed using the ``base`` attribute of the robot.
            - Link frames are indicated by circles
            - ETS transforms are indicated by rounded boxes

        The edges are:
            - an arrow if ``jtype`` is False or the joint is fixed
            - an arrow with a round head if ``jtype`` is True and the joint is
              revolute
            - an arrow with a box head if ``jtype`` is True and the joint is
              prismatic

        Edge labels or nodes in blue have a fixed transformation to the
        preceding link.

        :param display_graph: Open the graph in a browser if True. Otherwise will return the file path
        :param etsbox: Put the link ETS in a box, otherwise an edge label
        :param jtype: Arrowhead to node indicates revolute or prismatic type
        :param static: Show static joints in blue and bold

        .. rubric:: Examples

        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.URDF.Panda()
        >>> panda.showgraph()

        .. image:: ../figs/panda-graph.svg
            :width: 600

        .. seealso:: :func:`dotfile` :func:`graph`

        """

        # Lazy import
        import subprocess
        import tempfile
        import webbrowser

        # create the temporary dotfile
        dotfile = tempfile.TemporaryFile(mode="w")
        self.dotfile(dotfile, **kwargs)

        # rewind the dot file, create PDF file in the filesystem, run dot
        dotfile.seek(0)
        pdffile = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        subprocess.run("dot -Tpdf", shell=True, stdin=dotfile, stdout=pdffile)

        # open the PDF file in browser (hopefully portable), then cleanup
        if display_graph:  # pragma nocover
            webbrowser.open(f"file://{pdffile.name}")
        else:
            return pdffile.name

        return None

    def dotfile(
        self,
        filename: str | IO[str],
        etsbox: bool = False,
        ets: L["full", "brief"] = "full",
        jtype: bool = False,
        static: bool = True,
    ) -> None:
        """
        Write a link transform graph as a GraphViz dot file

        The file can be processed using dot::

            % dot -Tpng -o out.png dotfile.dot

        The nodes are:

        - Base is shown as a grey square.  This is the world frame origin,
          but can be changed using the ``base`` attribute of the robot.
        - Link frames are indicated by circles
        - ETS transforms are indicated by rounded boxes

        The edges are:

        - an arrow if ``jtype`` is False or the joint is fixed
        - an arrow with a round head if ``jtype`` is True and the joint is
          revolute
        - an arrow with a box head if ``jtype`` is True and the joint is
          prismatic

        Edge labels or nodes in blue have a fixed transformation to the
        preceding link.

        .. rubric:: Notes

        If ``filename`` is a file object then the file will *not*
        be closed after the GraphViz model is written.

        :param file: Name of file to write to
        :param etsbox: Put the link ETS in a box, otherwise an edge label
        :param ets: Display the full ets with "full" or a brief version with "brief"
        :param jtype: Arrowhead to node indicates revolute or prismatic type
        :param static: Show static joints in blue and bold

        .. seealso:: :func:`showgraph` :func:`graph`

        """

        file: IO[str]
        if isinstance(filename, str):
            file = open(filename, "w", encoding="utf-8")
        else:
            file = filename

        header = r"""digraph G {
graph [rankdir=LR];
"""

        def draw_edge(link: Any, etsbox: Any, jtype: Any, static: Any) -> None:
            # draw the edge
            if jtype:
                if link.isprismatic:
                    edge_options = 'arrowhead="box", arrowtail="inv", dir="both"'
                elif link.isrevolute:
                    edge_options = 'arrowhead="dot", arrowtail="inv", dir="both"'
                else:
                    edge_options = 'arrowhead="normal"'
            else:
                edge_options = 'arrowhead="normal"'

            if link.parent is None:
                parent = "BASE"
            else:
                parent = link.parent.name

            if etsbox:
                # put the ets fragment in a box
                if not link.isjoint and static:
                    node_options = ', fontcolor="blue"'
                else:
                    node_options = ""

                try:
                    file.write(
                        '  {}_ets [shape=box, style=rounded, label="{}"{}];\n'.format(
                            link.name,
                            link.ets.__str__(q=f"q{link.jindex}"),
                            node_options,
                        )
                    )
                except UnicodeEncodeError:  # pragma nocover
                    file.write(
                        '  {}_ets [shape=box, style=rounded, label="{}"{}];\n'.format(
                            link.name,
                            link.ets.__str__(q=f"q{link.jindex}")
                            .encode("ascii", "ignore")
                            .decode("ascii"),
                            node_options,
                        )
                    )

                file.write(f"  {parent} -> {link.name}_ets;\n")
                file.write(
                    f"  {link.name}_ets -> {link.name} [{edge_options}];\n"
                )
            else:
                # put the ets fragment as an edge label
                if not link.isjoint and static:
                    edge_options += ', fontcolor="blue"'
                if ets == "full":
                    estr = link.ets.__str__(q=f"q{link.jindex}")
                elif ets == "brief":
                    if link.jindex is None:
                        estr = ""
                    else:
                        estr = f"...q{link.jindex}"
                else:
                    return
                try:
                    file.write(
                        f'  {parent} -> {link.name} [label="{estr}", {edge_options}];\n'
                    )
                except UnicodeEncodeError:  # pragma nocover
                    file.write(
                        '  {} -> {} [label="{}", {}];\n'.format(
                            parent,
                            link.name,
                            estr.encode("ascii", "ignore").decode("ascii"),
                            edge_options,
                        )
                    )

        file.write(header)

        # add the base link
        file.write("  BASE [shape=square, style=filled, fillcolor=gray]\n")

        # add the links
        for link in self:
            # draw the link frame node (circle) or ee node (doublecircle)
            if link in self.ee_links:
                # end-effector
                node_options = 'shape="doublecircle", color="blue", fontcolor="blue"'
            else:
                node_options = 'shape="circle"'

            file.write(f"  {link.name} [{node_options}];\n")

            draw_edge(link, etsbox, jtype, static)

        for gripper in self.grippers:
            for link in gripper.links:
                file.write(f"  {link.name} [shape=cds];\n")
                draw_edge(link, etsbox, jtype, static)

        file.write("}\n")

        if isinstance(filename, str):
            file.close()

    def graph(
        self,
        format: L["dot", "mermaid", "mermaid_fenced"] = "dot",
        filename: str | IO[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Render a link transform graph as text

        :param format: graph output format: ``"dot"``, ``"mermaid"`` or
            ``"mermaid_fenced"``, defaults to ``"dot"``
        :param filename: destination path or open text stream, defaults to
            None
        :param kwargs: for ``format="dot"``, forwarded to :meth:`dotfile`
            (``etsbox``, ``ets``, ``jtype``, ``static``); ignored otherwise
        :return: rendered graph text

        Creates graph text representing the robot's link transform graph --
        the same structure :meth:`dotfile`/:meth:`showgraph` render, in a
        choice of formats. If ``filename`` is provided, the rendered text
        is also written to it.

        ``"dot"`` produces the same GraphViz output as :meth:`dotfile`,
        just returned as a string rather than requiring a real file --
        ``etsbox``/``ets``/``jtype``/``static`` behave exactly as they do
        there.

        ``"mermaid"``/``"mermaid_fenced"`` render a simplified `Mermaid
        <https://mermaid.js.org>`_ flowchart: one node per link (a double
        circle for end-effectors, a hexagon for gripper links), edges
        labelled with the joint type (``R``/``P``) and joint index for
        actuated joints. Mermaid has no equivalent of DOT's per-edge
        arrowhead/colour styling, so ``jtype``/``static``/``etsbox`` only
        affect ``"dot"`` output. ``"mermaid_fenced"`` wraps the same text
        in a fenced ``mermaid`` code block, ready to paste into Markdown
        or render directly in a Jupyter cell::

            from IPython.display import Markdown, display
            display(Markdown(robot.graph(format="mermaid_fenced")))

        .. seealso:: :meth:`dotfile` :meth:`showgraph`
        """
        graph_format = format.lower().replace("-", "_")

        if graph_format == "dot":
            stream = io.StringIO()
            self.dotfile(stream, **kwargs)
            text = stream.getvalue()

        elif graph_format in ("mermaid", "mermaid_fenced"):
            lines = ["flowchart LR"]
            node_ids: dict[Any, str] = {}

            def node_id(link: Any) -> str:
                return node_ids.setdefault(link, f"n{len(node_ids)}")

            lines.append('    BASE["BASE"]')

            for link in self:
                nid = node_id(link)
                if link in self.ee_links:
                    lines.append(f'    {nid}(("{link.name}"))')
                else:
                    lines.append(f'    {nid}["{link.name}"]')

            for gripper in self.grippers:
                for link in gripper.links:
                    lines.append(f'    {node_id(link)}{{{{"{link.name}"}}}}')

            def edge_label(link: Any) -> str:
                if not link.isjoint:
                    return ""
                kind = "P" if link.isprismatic else "R"
                return f"|{kind}{link.jindex}|"

            for link in self:
                parent = "BASE" if link.parent is None else node_id(link.parent)
                lines.append(f"    {parent} -->{edge_label(link)} {node_id(link)}")

            for gripper in self.grippers:
                for link in gripper.links:
                    parent = (
                        "BASE" if link.parent is None else node_id(link.parent)
                    )
                    lines.append(
                        f"    {parent} -->{edge_label(link)} {node_id(link)}"
                    )

            mermaid_text = "\n".join(lines) + "\n"
            if graph_format == "mermaid_fenced":
                text = f"```mermaid\n{mermaid_text}```\n"
            else:
                text = mermaid_text

        else:
            raise ValueError(
                f"unsupported graph format {format!r}, expected 'dot', "
                "'mermaid' or 'mermaid_fenced'"
            )

        if filename is not None:
            if isinstance(filename, str):
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                filename.write(text)

        return text
