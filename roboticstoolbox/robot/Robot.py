# import sys
from os.path import splitext
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath.base.argcheck import (
    isvector,
    getvector,
    getmatrix,
    getunit,
    verifymatrix,
)
import spatialmath.base as smb
from ansitable import ANSITable, Column
from roboticstoolbox.backends.PyPlot import PyPlot
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
from roboticstoolbox.robot.Dynamics import DynamicsMixin
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.tools.types import ArrayLike, NDArray

from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    TypeVar,
    Union,
    Dict,
    Tuple,
    overload,
    Literal as L,
)

from roboticstoolbox.robot.RobotKinematics import RobotKinematicsMixin

# from typing_extensions import Protocol

from spatialgeometry import Shape
from fknm import Robot_link_T
from functools import lru_cache
from spatialgeometry import SceneNode
from roboticstoolbox.robot.Link import BaseLink, Link
from roboticstoolbox.tools import xacro
from roboticstoolbox.tools import URDF
from roboticstoolbox.robot.Gripper import Gripper
from roboticstoolbox.tools.data import rtb_path_to_datafile
from roboticstoolbox.tools.params import rtb_get_param
from warnings import warn
from pathlib import PurePosixPath, Path

if TYPE_CHECKING:
    from matplotlib.cm import Color  # pragma: nocover
else:
    Color = None

try:
    from matplotlib import colors
    from matplotlib import cm

    _mpl = True
except ImportError:  # pragma nocover
    cm = str
    pass

_default_backend = None

# A generic type variable representing any subclass of BaseLink
LinkType = TypeVar("LinkType", bound=BaseLink)


class BaseRobot(SceneNode, DynamicsMixin, ABC, Generic[LinkType]):
    def __init__(
        self,
        links: List[LinkType],
        gripper_links: Union[LinkType, List[LinkType], None] = None,
        name: str = "",
        manufacturer: str = "",
        comment: str = "",
        base: Union[NDArray, SE3, None] = None,
        tool: Union[NDArray, SE3, None] = None,
        gravity: ArrayLike = [0, 0, -9.81],
        keywords: Union[List[str], Tuple[str]] = [],
        symbolic: bool = False,
        configs: Union[Dict[str, NDArray], None] = None,
        check_jindex: bool = True,
    ):

        # Initialise the scene node
        SceneNode.__init__(self)

        # Lets sort out links now
        self._linkdict: Dict[str, LinkType] = {}

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
            self._T = base.A
        elif isinstance(base, np.ndarray):
            self._T = base

        # Set the robot tool transform
        if isinstance(tool, SE3):
            self._tool = tool.A
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
        self._reach = None
        self._hasdynamics = False
        self._hasgeometry = False
        self._hascollision = False
        self._urdf_string = ""
        self._urdf_filepath = ""

        # Time to checkout the links for geometry information
        for link in self.links:

            # Add link back to robot object
            link._robot = self

            if link.hasdynamics:
                self._hasdynamics = True
            if link.geometry:
                self._hasgeometry = []
            if link.collision:
                self._hascollision = True

            if isinstance(link, Link):
                if len(link.geometry) > 0:
                    self._hasgeometry = True

        # Current joint configuraiton, velocity, acceleration
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
        self._dynchanged = False

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
        self.scene_children = [self.links[0]]  # type: ignore

    # --------------------------------------------------------------------- #
    # --------- Private Methods ------------------------------------------- #
    # --------------------------------------------------------------------- #

    def _sort_links(
        self,
        links: List[LinkType],
        gripper_links: Union[LinkType, List[LinkType], None],
        check_jindex: bool,
    ):
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
        orlinks: List[LinkType] = []

        # The end-effector links
        self._ee_links: List[LinkType] = []

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

            # link.number = k + 1

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
                li = links[i]
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
        ee_links: List[LinkType] = []

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

            def visit_link(link, jindex):
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
                for link in links:
                    if link.isjoint and link.jindex not in jset:
                        raise ValueError(
                            f"joint index {link.jindex} was " "repeated or out of range"
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

    def dynchanged(self, what: Union[str, None] = None):
        """
        Dynamic parameters have changed

        Called from a property setter to inform the robot that the cache of
        dynamic parameters is invalid.

        See Also
        --------
        :func:`roboticstoolbox.Link._listen_dyn`

        """

        self._dynchanged = True
        if what != "gravity":
            self._hasdynamics = True

    # --------------------------------------------------------------------- #
    # --------- Magic Methods --------------------------------------------- #
    # --------------------------------------------------------------------- #

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self) -> LinkType:
        if self._iter < len(self.links):
            link = self[self._iter]
            self._iter += 1
            return link
        else:
            raise StopIteration

    def __getitem__(self, i: Union[int, str]) -> LinkType:
        """
        Get link

        This also supports iterating over each link in the robot object,
        from the base to the tool.

        Parameters
        ----------
        i
            link number or name

        Returns
        -------
        link
            i'th link or named link

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> print(robot[1]) # print the 2nd link
        >>> print([link.a for link in robot])  # print all the a_j values

        Notes
        -----
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

        Returns
        -------
        str
            Pretty print of the robot model

        Notes
        -----
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
                jname = ""
            table.row(
                # link.jindex,
                k,
                color + ee + link.name,
                jname,
                parent_name,
                f"{s}",
            )

        classname = "ERobot"

        s = f"{classname}: {self.name}"
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

    # --------------------------------------------------------------------- #
    # --------- Properties ------------------------------------------------ #
    # --------------------------------------------------------------------- #

    @property
    def links(self) -> List[LinkType]:
        """
        Robot links

        Returns
        -------
        links
            A list of link objects

        Notes
        -----
        It is probably more concise to index the robot object rather
        than the list of links, ie. the following are equivalent:
        - ``robot.links[i]``
        - ``robot[i]``

        """

        return self._links

    @property
    def link_dict(self) -> Dict[str, LinkType]:
        return self._linkdict

    @property
    def grippers(self) -> List[Gripper]:
        """
        Grippers attached to the robot

        Returns
        -------
        grippers
            A list of grippers

        """

        return self._grippers

    @property
    def base_link(self) -> LinkType:
        """
        Get the robot base link

        - ``robot.base_link`` is the robot base link

        Returns
        -------
        base_link
            the first link in the robot tree

        """

        return self._base_link

    @property
    def ee_links(self) -> List[LinkType]:
        return self._ee_links

    @property
    def n(self) -> int:
        """
        Number of joints

        Returns
        -------
        n
            Number of joints

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot.n

        See Also
        --------
        :func:`nlinks`
        :func:`nbranches`

        """

        return self._n

    @property
    def nlinks(self):
        """
        Number of links

        The returned number is the total of both variable joints and
        static links

        Returns
        -------
        nlinks
            Number of links

        Examples
        --------

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot.nlinks

        See Also
        --------
        :func:`n`
        :func:`nbranches`

        """

        return self._nlinks

    @property
    def nbranches(self) -> int:
        """
        Number of branches

        Number of branches in this robot.  Computed as the number of links with
        zero children

        Returns
        -------
        nbranches
            number of branches in the robot's kinematic tree

        Examples
        --------

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.ETS.Panda()
        >>> robot.nbranches

        See Also
        --------
        :func:`n`
        :func:`nlinks`
        """

        return sum([link.nchildren == 0 for link in self.links]) + len(self.grippers)

    # --------------------------------------------------------------------- #

    @property
    def name(self):
        """
        Get/set robot name

        - ``robot.name`` is the robot name
        - ``robot.name = ...`` checks and sets the robot name

        Parameters
        ----------
        name
            the new robot name

        Returns
        -------
        name
            the current robot name

        """
        return self._name

    @name.setter
    def name(self, name_new):
        self._name = name_new

    @property
    def comment(self):
        """
        Get/set robot comment

        - ``robot.comment`` is the robot comment
        - ``robot.comment = ...`` checks and sets the robot comment

        Parameters
        ----------
        name
            the new robot comment

        Returns
        -------
        comment
            robot comment

        """
        return self._comment

    @comment.setter
    def comment(self, comment_new):
        self._comment = comment_new

    @property
    def manufacturer(self):
        """
        Get/set robot manufacturer's name

        - ``robot.manufacturer`` is the robot manufacturer's name
        - ``robot.manufacturer = ...`` checks and sets the manufacturer's name

        Returns
        -------
        manufacturer
            robot manufacturer's name

        """
        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, manufacturer_new):
        self._manufacturer = manufacturer_new

    @property
    def configs(self) -> Dict[str, NDArray]:
        return self._configs

    @property
    def keywords(self) -> List[str]:
        return self._keywords

    @property
    def symbolic(self) -> bool:
        return self._symbolic

    @property
    def hasdynamics(self):
        """
        Robot has dynamic parameters

        Returns
        -------
        hasdynamics
            Robot has dynamic parameters

        At least one link has associated dynamic parameters.

        Examples
        --------

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot.hasdynamics:

        """

        return self._hasdynamics

    @property
    def hasgeometry(self):
        """
        Robot has geometry model

        At least one link has associated mesh to describe its shape.

        Returns
        -------
        hasgeometry
            Robot has geometry model

        Examples
        --------

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot.hasgeometry

        See Also
        --------
        :func:`hascollision`

        """

        return self._hasgeometry

    @property
    def hascollision(self):
        """
        Robot has collision model

        Returns
        -------
        hascollision
            Robot has collision model

        At least one link has associated collision model.

        Examples
        --------

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot.hascollision

        See Also
        --------
        :func:`hasgeometry`

        """

        return self._hascollision

    @property
    def default_backend(self):
        """
        Get default graphical backend

        - ``robot.default_backend`` Get the default graphical backend, used when
            no explicit backend is passed to ``Robot.plot``.
        - ``robot.default_backend = ...`` Set the default graphical backend, used when
            no explicit backend is passed to ``Robot.plot``.  The default set here will
            be overridden if the particular ``Robot`` subclass cannot support it.

        Returns
        -------
        default_backend
            backend name


        """
        return _default_backend

    @default_backend.setter
    def default_backend(self, be):
        _default_backend = be

    @property
    def gravity(self) -> NDArray:
        """
        Get/set default gravitational acceleration (Robot superclass)

        - ``robot.name`` is the default gravitational acceleration
        - ``robot.name = ...`` checks and sets default gravitational
            acceleration


        Parameters
        ----------
        gravity
            the new gravitational acceleration for this robot

        Returns
        -------
        gravity
            gravitational acceleration

        Notes
        -----
        If the z-axis is upward, out of the Earth, this should be
        a positive number.

        """

        return self._gravity

    @gravity.setter
    def gravity(self, gravity_new: ArrayLike):
        self._gravity = np.array(getvector(gravity_new, 3))
        self.dynchanged()

    # --------------------------------------------------------------------- #

    @property
    def q(self) -> NDArray:
        """
        Get/set robot joint configuration

        - ``robot.q`` is the robot joint configuration
        - ``robot.q = ...`` checks and sets the joint configuration

        Parameters
        ----------
        q
            the new robot joint configuration

        Returns
        -------
        q
            robot joint configuration

        """

        return self._q

    @q.setter
    def q(self, q_new: ArrayLike):
        self._q = np.array(getvector(q_new, self.n))

    @property
    def qd(self) -> NDArray:
        """
        Get/set robot joint velocity

        - ``robot.qd`` is the robot joint velocity
        - ``robot.qd = ...`` checks and sets the joint velocity

        Returns
        -------
        qd
            robot joint velocity

        """

        return self._qd

    @qd.setter
    def qd(self, qd_new: ArrayLike):
        self._qd = np.array(getvector(qd_new, self.n))

    @property
    def qdd(self) -> NDArray:
        """
        Get/set robot joint acceleration

        - ``robot.qdd`` is the robot joint acceleration
        - ``robot.qdd = ...`` checks and sets the robot joint acceleration

        Returns
        -------
        qdd
            robot joint acceleration


        """
        return self._qdd

    @qdd.setter
    def qdd(self, qdd_new: ArrayLike):
        self._qdd = np.array(getvector(qdd_new, self.n))

    @property
    def qlim(self) -> NDArray:
        r"""
        Joint limits

        Limits are extracted from the link objects.  If joints limits are
        not set for:

        - a revolute joint [-𝜋. 𝜋] is returned
        - a prismatic joint an exception is raised

        Attributes
        ----------
        qlim
            An array of joints limits (2, n)

        Raises
        ------
        ValueError
            unset limits for a prismatic joint

        Returns
        -------
        qlim
            Array of joint limit values

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot.qlim

        """

        limits = np.zeros((2, self.n))
        j = 0

        for link in self.links:

            if link.isrevolute:
                if link.qlim is None or np.any(np.isnan(link.qlim)):
                    v = [-np.pi, np.pi]
                else:
                    v = link.qlim
            elif link.isprismatic:
                if link.qlim is None:
                    raise ValueError("Undefined prismatic joint limit")
                else:
                    v = link.qlim
            else:
                # Fixed link
                continue  # pragma: nocover

            limits[:, j] = v
            j += 1

        return limits

    @property
    def structure(self) -> str:
        """
        Return the joint structure string

        A string with one letter per joint: ``R`` for a revolute
        joint, and ``P`` for a prismatic joint.

        Returns
        -------
        structure
            joint configuration string

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.structure
        >>> stanford = rtb.models.DH.Stanford()
        >>> stanford.structure

        Notes
        -----
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
    def prismaticjoints(self) -> List[bool]:
        """
        Revolute joints as bool array

        Returns
        -------
        prismaticjoints
            array of joint type, True if prismatic

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.prismaticjoints()
        >>> stanford = rtb.models.DH.Stanford()
        >>> stanford.prismaticjoints()

        Notes
        -----
        Fixed joints, that maintain a constant link relative pose,
        are not included.

        See Also
        --------
        :func:`Link.isprismatic`
        :func:`revolutejoints`

        """

        return [link.isprismatic for link in self.links if link.isjoint]

    @property
    def revolutejoints(self) -> List[bool]:
        """
        Revolute joints as bool array

        Returns
        -------
        revolutejoints
            array of joint type, True if revolute

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.revolutejoints()
        >>> stanford = rtb.models.DH.Stanford()
        >>> stanford.revolutejoints()

        Notes
        -----
        Fixed joints, that maintain a constant link relative pose,
        are not included.

        See Also
        --------
        :func:`Link.isrevolute`
        :func:`prismaticjoints`

        """

        return [link.isrevolute for link in self.links if link.isjoint]

    @property
    def control_mode(self) -> str:
        """
        Get/set robot control mode

        - ``robot.control_type`` is the robot control mode
        - ``robot.control_type = ...`` checks and sets the robot control mode

        Parameters
        ----------
        control_mode
            the new robot control mode

        Returns
        -------
        control_mode
            the current robot control mode

        """

        return self._control_mode

    @control_mode.setter
    def control_mode(self, cn: str):
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

        Parameters
        ----------
        tool
            the new robot tool transform (as an SE(3))

        Returns
        -------
        tool
            robot tool transform



        """
        return SE3(self._tool, check=False)

    @tool.setter
    def tool(self, T: Union[SE3, NDArray]):
        if isinstance(T, SE3):
            self._tool = T.A
        else:
            self._tool = T

    @property
    def base(self) -> SE3:
        """
        Get/set robot base transform

        - ``robot.base`` is the robot base transform
        - ``robot.base = ...`` checks and sets the robot base transform

        Parameters
        ----------
        base
            the new robot base transform

        Returns
        -------
        base
            the current robot base transform

        """

        # return a copy, otherwise somebody with
        # reference to the base can change it

        # This now returns the Scene Node transform
        # self._T is a copy of SceneNode.__T
        return SE3(self._T, check=False)

    @base.setter
    def base(self, T: Union[NDArray, SE3]):

        if isinstance(self, rtb.Robot):
            # All 3D robots
            # Set the SceneNode T
            if isinstance(T, SE3):
                self._T = T.A
            else:
                self._T = T

        else:
            raise ValueError("base must be set to None (no tool), SE2, or SE3")

    # --------------------------------------------------------------------- #

    def _getlink(
        self,
        link: Union[LinkType, Gripper, str, None],
        default: Union[LinkType, Gripper, str, None] = None,
    ) -> Union[LinkType, Link]:
        """
        Validate reference to Link

        ``robot._getlink(link)`` is a validated reference to a Link within
        the ERobot ``robot``.  If ``link`` is:

        -  an ``Link`` reference it is validated as belonging to
          ``robot``.
        - a string, then it looked up in the robot's link name dictionary, and
          a Link reference returned.

        Parameters
        ----------
        link
            link

        Raises
        ------
        ValueError
            link does not belong to this ERobot
        TypeError
            bad argument

        Returns
        -------
        link
            link reference

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

    def _find_ets(self, start, end, explored, path) -> Union[ETS, None]:
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

    def _gripper_ets(self, gripper: Gripper) -> ETS:
        """
        Privade method which will find the ETS of a gripper

        """

        return ETS(ET.SE3(gripper.tool))

    @lru_cache(maxsize=32)
    def ets(
        self,
        start: Union[LinkType, Gripper, str, None] = None,
        end: Union[LinkType, Gripper, str, None] = None,
    ) -> ETS:
        """
        Robot to ETS

        - ``robot.ets()`` is an ETS representing the kinematics from base to
          end-effector.
        - ``robot.ets(end=link)`` is an ETS representing the kinematics from
          base to the link ``link`` specified as a Link reference or a name.
        - ``robot.ets(start=l1, end=l2)`` is an ETS representing the kinematics
          from link ``l1`` to link ``l2``.

        Parameters
        ----------
        :param start: start of path, defaults to ``base_link``
        :param end: end of path, defaults to end-effector

        Raises
        ------
        ValueError
            a link does not belong to this ERobot
        TypeError
            a bad link argument

        Returns
        -------
        ets
            elementary transform sequence

        Examples
        --------
        .. runblock:: pycon
            >>> import roboticstoolbox as rtb
            >>> panda = rtb.models.ETS.Panda()
            >>> panda.ets()

        """

        # ets to stand and end incase of grippers
        ets_init = None
        ets_end = None

        if isinstance(start, Gripper):
            ets_init = self._gripper_ets(start).inv()
            link = start.links[0].parent
            if link is None:  # pragma: nocover
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
            elif len(self.grippers) > 1:
                end_link = self._getlink(end, self.ee_links[0])
                print(
                    "multiple end-effectors present, ambiguous, using self.ee_links[0]"
                )
            else:
                end_link = self._getlink(end, self.ee_links[0])
        else:
            if isinstance(end, Gripper):
                ets_end = self._gripper_ets(end)
                end_link = end.links[0].parent  # type: ignore
                if end_link is None:  # pragma: nocover
                    raise ValueError("Invalid robot link configuration")
            else:
                end_link = self._getlink(end, self.ee_links[0])

        explored = set()

        if link is end_link:
            ets = link.ets
        else:
            ets = self._find_ets(link, end_link, explored, path=None)

        if ets is None:
            raise ValueError("Could not find the requested ETS in this robot")

        if ets_init is not None:
            ets = ets_init * ets

        if ets_end is not None:
            ets = ets * ets_end

        return ets

    def copy(self):
        return deepcopy(self)

    def __deepcopy__(self, memo):

        links = []

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

        cls = self.__class__
        result = cls(
            links=links,
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

    def todegrees(self, q) -> NDArray:
        """
        Convert joint angles to degrees

        Parameters
        ----------
        q
            The joint configuration of the robot

        Returns
        -------
        q
            a vector of joint coordinates in degrees and metres

        ``robot.todegrees(q)`` converts joint coordinates ``q`` to degrees
        taking into account whether elements of ``q`` correspond to revolute
        or prismatic joints, ie. prismatic joint values are not converted.

        If ``q`` is a matrix, with one column per joint, the conversion is
        performed columnwise.

        Examples
        --------
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

    def toradians(self, q) -> NDArray:
        """
        Convert joint angles to radians

        ``robot.toradians(q)`` converts joint coordinates ``q`` to radians
        taking into account whether elements of ``q`` correspond to revolute
        or prismatic joints, ie. prismatic joint values are not converted.

        If ``q`` is a matrix, with one column per joint, the conversion is
        performed columnwise.

        Parameters
        ----------
        q
            The joint configuration of the robot


        Returns
        -------
        q
            a vector of joint coordinates in radians and metres


        Examples
        --------
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

    def isrevolute(self, j) -> bool:
        """
        Check if joint is revolute

        Returns
        -------
        j
            True if revolute

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.revolutejoints()
        >>> stanford = rtb.models.DH.Stanford()
        >>> stanford.isrevolute(1)

        See Also
        --------
        :func:`Link.isrevolute`
        :func:`revolutejoints`

        """
        return self.revolutejoints[j]

    def isprismatic(self, j) -> bool:
        """
        Check if joint is prismatic

        Returns
        -------
        j
            True if prismatic

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> puma = rtb.models.DH.Puma560()
        >>> puma.prismaticjoints()
        >>> stanford = rtb.models.DH.Stanford()
        >>> stanford.isprismatic(1)

        See Also
        --------
        :func:`Link.isprismatic`
        :func:`prismaticjoints`

        """

        return self.prismaticjoints[j]

    # --------------------------------------------------------------------- #

    def dfs_links(
        self,
        start: LinkType,
        func: Union[None, Callable[[LinkType], Any]] = None,
    ):
        """
        A link search method

        Visit all links from start in depth-first order and will apply
        func to each visited link

        Parameters
        ----------
        start
            The link to start at
        func
            An optional function to apply to each link as it is found

        Returns
        -------
        links
            A list of links

        """

        visited = []

        def vis_children(link):
            visited.append(link)
            if func is not None:
                func(link)

            for li in link.children:
                if li not in visited:
                    vis_children(li)

        vis_children(start)

        return visited

    def addconfiguration_attr(self, name: str, q: ArrayLike, unit: str = "rad"):
        """
        Add a named joint configuration as an attribute

        Parameters
        ----------
        name
            Name of the joint configuration
        q
            Joint configuration

        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot.addconfiguration_attr("mypos", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        >>> robot.mypos
        >>> robot.configs["mypos"]

        Notes
        -----
        - Used in robot model init method to store the ``qr`` configuration
        - Dynamically adding attributes to objects can cause issues with
            Python type checking.
        - Configuration is also added to the robot instance's dictionary of
            named configurations.

        See Also
        --------
        :meth:`addconfiguration`

        """

        v = getvector(q, self.n)
        v = getunit(v, unit)
        v = np.array(v)
        self._configs[name] = v
        setattr(self, name, v)

    def addconfiguration(self, name: str, q: NDArray):
        """
        Add a named joint configuration

        Add a named configuration to the robot instance's dictionary of named
        configurations.

        Parameters
        ----------
        name
            Name of the joint configuration
        q
            Joint configuration



        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> robot.addconfiguration_attr("mypos", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        >>> robot.configs["mypos"]

        See Also
        --------
        :meth:`addconfiguration`

        """

        self._configs[name] = q

    def configurations_str(self, border="thin"):
        deg = 180 / np.pi

        # TODO: factor this out of DHRobot
        def angle(theta, fmt=None):

            if fmt is not None:
                try:
                    return fmt.format(theta * deg) + "\u00b0"
                except TypeError:
                    pass

            # pragma nocover
            return str(theta * deg) + "\u00b0"

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

    def random_q(self):
        """
        Return a random joint configuration

        The value for each joint is uniform randomly distributed  between the
        limits set for the robot.

        Note
        ----
        The joint limit for all joints must be set.

        Returns
        -------
        q
            Random joint configuration :rtype: ndarray(n)

        See Also
        --------
        :func:`Robot.qlim`
        :func:`Link.qlim`

        """

        qlim = self.qlim
        if np.any(np.isnan(qlim)):
            raise ValueError("some joint limits not defined")
        return np.random.uniform(low=qlim[0, :], high=qlim[1, :], size=(self.n,))

    def linkcolormap(self, linkcolors: Union[List[Any], str] = "viridis"):
        """
        Create a colormap for robot joints

        - ``cm = robot.linkcolormap()`` is an n-element colormap that gives a
          unique color for every link.  The RGBA colors for link ``j`` are
          ``cm(j)``.
        - ``cm = robot.linkcolormap(cmap)`` as above but ``cmap`` is the name
          of a valid matplotlib colormap.  The default, example above, is the
          ``viridis`` colormap.
        - ``cm = robot.linkcolormap(list of colors)`` as above but a
          colormap is created from a list of n color names given as strings,
          tuples or hexstrings.

        Parameters
        ----------
        linkcolors
            list of colors or colormap, defaults to "viridis"

        Returns
        -------
        color map
            the color map


        Examples
        --------
        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> cm = robot.linkcolormap("inferno")
        >>> print(cm(range(6))) # cm(i) is 3rd color in colormap
        >>> cm = robot.linkcolormap(
        >>>     ['red', 'g', (0,0.5,0), '#0f8040', 'yellow', 'cyan'])
        >>> print(cm(range(6)))

        Notes
        -----
        - Colormaps have 4-elements: red, green, blue, alpha (RGBA)
        - Names of supported colors and colormaps are defined in the
          matplotlib documentation.
            - `Specifying colors
            <https://matplotlib.org/3.1.0/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py>`_
            - `Colormaps
            <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py>`_

        """

        if isinstance(linkcolors, list) and len(linkcolors) == self.n:
            # provided a list of color names
            return colors.ListedColormap(linkcolors)  # type: ignore
        else:
            # assume it is a colormap name
            return cm.get_cmap(linkcolors, 6)  # type: ignore

    def hierarchy(self) -> None:
        """
        Pretty print the robot link hierachy

        Returns
        -------
        Pretty print of the robot model

        Examples
        --------
        Makes a robot and prints the heirachy

        .. runblock:: pycon
        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Panda()
        >>> robot.hierarchy()

        """

        def recurse(link, indent=0):
            print(" " * indent * 2, link.name)
            for child in link.child:
                recurse(child, indent + 1)

        recurse(self.base_link)

    def segments(self) -> List[List[Union[LinkType, None]]]:
        """
        Segments of branched robot

        For a single-chain robot with structure::

            L1 - L2 - L3

        the return is ``[[None, L1, L2, L3]]``

        For a robot with structure::

            L1 - L2 +-  L3 - L4
                    +- L5 - L6

        the return is ``[[None, L1, L2], [L2, L3, L4], [L2, L5, L6]]``

        Returns
        -------
        segments
            Segment list

        Notes
        -----
        - the length of the list is the number of segments in the robot
        - the first segment always starts with ``None`` which represents
            the base transform (since there is no base link)
        - the last link of one segment is also the first link of subsequent
            segments
        """

        def recurse(link: Link):

            segs = [link.parent]
            while True:
                segs.append(link)
                if link.nchildren == 0:
                    return [segs]
                elif link.nchildren == 1:
                    link = link.children[0]  # type: ignore
                    continue
                elif link.nchildren > 1:
                    segs = [segs]

                    for child in link.children:  # type: ignore
                        segs.extend(recurse(child))

                    return segs

        return recurse(self.links[0])  # type: ignore

    # --------------------------------------------------------------------- #
    # Scene Graph section
    # --------------------------------------------------------------------- #

    def _update_link_tf(self, q: Union[ArrayLike, None] = None):
        """
        This private method updates the local transform of each link within
        this robot according to q (or self.q if q is none)
        """

        @lru_cache(maxsize=2)
        def get_link_ets():
            return [link.ets._fknm for link in self.links]

        @lru_cache(maxsize=2)
        def get_link_scene_node():
            return [link._T_reference for link in self.links]

        Robot_link_T(get_link_ets(), get_link_scene_node(), self._q, q)

        [gripper._update_link_tf() for gripper in self.grippers]

    # --------------------------------------------------------------------- #


class Robot(BaseRobot[Link], RobotKinematicsMixin):

    _color = True

    def __init__(
        self,
        arg: Union[List[Link], ETS],
        gripper_links: Union[Link, List[Link], None] = None,
        name: str = "",
        manufacturer: str = "",
        comment: str = "",
        base: Union[NDArray, SE3, None] = None,
        tool: Union[NDArray, SE3, None] = None,
        gravity: ArrayLike = [0, 0, -9.81],
        keywords: Union[List[str], Tuple[str]] = [],
        symbolic: bool = False,
        configs: Union[Dict[str, NDArray], None] = None,
        check_jindex: bool = True,
        urdf_string: Union[str, None] = None,
        urdf_filepath: Union[Path, PurePosixPath, None] = None,
    ):

        # Process links
        if isinstance(arg, ETS):
            # We're passed an ETS string
            links = []
            # chop it up into segments, a link frame after every joint
            parent = None
            for j, ets_j in enumerate(arg.split()):
                elink = Link(ETS(ets_j), parent=parent, name=f"link{j:d}")
                if (
                    elink.qlim is None
                    and elink.v is not None
                    and elink.v.qlim is not None
                ):
                    elink.qlim = elink.v.qlim
                parent = elink
                links.append(elink)
        elif smb.islistof(arg, Link):
            links = arg
        else:
            raise TypeError("arg was invalid, must be List[Link], ETS, or Robot")

        # Initialise Base Robot object
        super().__init__(
            links=links,
            gripper_links=gripper_links,
            name=name,
            manufacturer=manufacturer,
            comment=comment,
            base=base,
            tool=tool,
            gravity=gravity,
            keywords=keywords,
            symbolic=symbolic,
            configs=configs,
            check_jindex=check_jindex,
        )

        self.links

    # --------------------------------------------------------------------- #
    # --------- Swift Methods --------------------------------------------- #
    # --------------------------------------------------------------------- #

    def _to_dict(self, robot_alpha=1.0, collision_alpha=0.0):

        ob = []

        for link in self.links:

            if robot_alpha > 0:
                for gi in link.geometry:
                    gi.set_alpha(robot_alpha)
                    ob.append(gi.to_dict())
            if collision_alpha > 0:
                for gi in link.collision:
                    gi.set_alpha(collision_alpha)
                    ob.append(gi.to_dict())

        # Do the grippers now
        for gripper in self.grippers:
            for link in gripper.links:

                if robot_alpha > 0:
                    for gi in link.geometry:
                        gi.set_alpha(robot_alpha)
                        ob.append(gi.to_dict())
                if collision_alpha > 0:
                    for gi in link.collision:
                        gi.set_alpha(collision_alpha)
                        ob.append(gi.to_dict())

        # for o in ob:
        #     print(o)

        return ob

    def _fk_dict(self, robot_alpha=1.0, collision_alpha=0.0):
        ob = []

        # Do the robot
        for link in self.links:

            if robot_alpha > 0:
                for gi in link.geometry:
                    ob.append(gi.fk_dict())
            if collision_alpha > 0:
                for gi in link.collision:
                    ob.append(gi.fk_dict())

        # Do the grippers now
        for gripper in self.grippers:
            for link in gripper.links:
                if robot_alpha > 0:
                    for gi in link.geometry:
                        ob.append(gi.fk_dict())
                if collision_alpha > 0:
                    for gi in link.collision:
                        ob.append(gi.fk_dict())

        return ob

    # --------------------------------------------------------------------- #
    # --------- URDF Methods ---------------------------------------------- #
    # --------------------------------------------------------------------- #

    @staticmethod
    def URDF_read(
        file_path, tld=None, xacro_tld=None
    ) -> Tuple[List[Link], str, str, Union[Path, PurePosixPath]]:
        """
        Read a URDF file as Links

        File should be specified relative to ``RTBDATA/URDF/xacro``

        Parameters
        ----------
        file_path
            File path relative to the xacro folder
        tld
            A custom top-level directory which holds the xacro data,
            defaults to None
        xacro_tld
            A custom top-level within the xacro data,
            defaults to None

        Returns
        -------
        links
            a list of links
        name
            the name of the robot
        urdf
            a string representing the URDF
        file_path
            a path to the original file

        Notes
        -----
        If ``tld`` is not supplied, filepath pointing to xacro data should
        be directly under ``RTBDATA/URDF/xacro`` OR under ``./xacro`` relative
        to the model file calling this method. If ``tld`` is supplied, then
        ```file_path``` needs to be relative to ``tld``

        """

        # Get the path to the class that defines the robot
        if tld is None:
            base_path = rtb_path_to_datafile("xacro")
        else:
            base_path = PurePosixPath(tld)

        # Add on relative path to get to the URDF or xacro file
        # base_path = PurePath(classpath).parent.parent / 'URDF' / 'xacro'
        file_path = base_path / PurePosixPath(file_path)
        _, ext = splitext(file_path)

        if ext == ".xacro":
            # it's a xacro file, preprocess it
            if xacro_tld is not None:
                xacro_tld = base_path / PurePosixPath(xacro_tld)
            urdf_string = xacro.main(file_path, xacro_tld)
            try:
                urdf = URDF.loadstr(urdf_string, file_path, base_path)
            except BaseException as e:
                print("error parsing URDF file", file_path)
                raise e
        else:  # pragma nocover
            urdf_string = open(file_path).read()
            urdf = URDF.loadstr(urdf_string, file_path, base_path)

        if not isinstance(urdf_string, str):
            raise ValueError("Parsing failed, did not get valid URDF string back")

        return urdf.elinks, urdf.name, urdf_string, file_path

    @classmethod
    def URDF(cls, file_path: str, gripper: Union[int, str, None] = None):
        """
        Construct a Robot object from URDF file

        Parameters
        ----------
        file_path
            the path to the URDF
        gripper
            index or name of the gripper link(s)

        Returns
        -------
        If ``gripper`` is specified, links from that link outward are removed
        from the rigid-body tree and folded into a ``Gripper`` object.

        """

        links, name, _, _ = Robot.URDF_read(file_path)

        gripperLink: Union[Link, None] = None

        if gripper is not None:
            if isinstance(gripper, int):
                gripperLink = links[gripper]
            elif isinstance(gripper, str):
                for link in links:
                    if link.name == gripper:
                        gripperLink = link
                        break
                else:
                    raise ValueError(f"no link named {gripper}")
            else:
                raise TypeError("bad argument passed as gripper")

        links, name, urdf_string, urdf_filepath = Robot.URDF_read(file_path)

        return cls(
            links,
            name=name,
            gripper_links=gripperLink,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

    # --------------------------------------------------------------------- #
    # --------- Utility Methods ------------------------------------------- #
    # --------------------------------------------------------------------- #

    def showgraph(self, **kwargs):
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
            - an arrow if `jtype` is False or the joint is fixed
            - an arrow with a round head if `jtype` is True and the joint is
              revolute
            - an arrow with a box head if `jtype` is True and the joint is
              prismatic

        Edge labels or nodes in blue have a fixed transformation to the
        preceding link.

        Parameters
        ----------
        etsbox
            Put the link ETS in a box, otherwise an edge label
        jtype
            Arrowhead to node indicates revolute or prismatic type
        static
            Show static joints in blue and bold

        Examples
        --------
        >>> import roboticstoolbox as rtb
        >>> panda = rtb.models.URDF.Panda()
        >>> panda.showgraph()

        .. image:: ../figs/panda-graph.svg
            :width: 600

        See Also
        --------
        :func:`dotfile`

        """

        # Lazy import
        import tempfile
        import subprocess
        import webbrowser

        # create the temporary dotfile
        dotfile = tempfile.TemporaryFile(mode="w")
        self.dotfile(dotfile, **kwargs)

        # rewind the dot file, create PDF file in the filesystem, run dot
        dotfile.seek(0)
        pdffile = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        subprocess.run("dot -Tpdf", shell=True, stdin=dotfile, stdout=pdffile)

        # open the PDF file in browser (hopefully portable), then cleanup
        webbrowser.open(f"file://{pdffile.name}")

    def dotfile(
        self,
        filename: Union[str, IO[str]],
        etsbox: bool = False,
        ets: L["full", "brief"] = "full",
        jtype: bool = False,
        static: bool = True,
    ):
        """
        Write a link transform graph as a GraphViz dot file

        The file can be processed using dot:
            % dot -Tpng -o out.png dotfile.dot

        The nodes are:
            - Base is shown as a grey square.  This is the world frame origin,
              but can be changed using the ``base`` attribute of the robot.
            - Link frames are indicated by circles
            - ETS transforms are indicated by rounded boxes

        The edges are:
            - an arrow if `jtype` is False or the joint is fixed
            - an arrow with a round head if `jtype` is True and the joint is
              revolute
            - an arrow with a box head if `jtype` is True and the joint is
              prismatic

        Edge labels or nodes in blue have a fixed transformation to the
        preceding link.

        Note
        ----
        If ``filename`` is a file object then the file will *not*
            be closed after the GraphViz model is written.

        Parameters
        ----------
        file
            Name of file to write to
        etsbox
            Put the link ETS in a box, otherwise an edge label
        ets
            Display the full ets with "full" or a brief version with "brief"
        jtype
            Arrowhead to node indicates revolute or prismatic type
        static
            Show static joints in blue and bold

        See Also
        --------
        :func:`showgraph`

        """

        if isinstance(filename, str):
            file = open(filename, "w")
        else:
            file = filename

        header = r"""digraph G {
graph [rankdir=LR];
"""

        def draw_edge(link, etsbox, jtype, static):
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
                file.write(
                    "  {}_ets [shape=box, style=rounded, "
                    'label="{}"{}];\n'.format(
                        link.name, link.ets.__str__(q=f"q{link.jindex}"), node_options
                    )
                )
                file.write("  {} -> {}_ets;\n".format(parent, link.name))
                file.write(
                    "  {}_ets -> {} [{}];\n".format(link.name, link.name, edge_options)
                )
            else:
                # put the ets fragment as an edge label
                if not link.isjoint and static:
                    edge_options += 'fontcolor="blue"'
                if ets == "full":
                    estr = link.ets.__str__(q=f"q{link.jindex}")
                elif ets == "brief":
                    if link.jindex is None:
                        estr = ""
                    else:
                        estr = f"...q{link.jindex}"
                else:
                    return
                file.write(
                    '  {} -> {} [label="{}", {}];\n'.format(
                        parent,
                        link.name,
                        estr,
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

            file.write("  {} [{}];\n".format(link.name, node_options))

            draw_edge(link, etsbox, jtype, static)

        for gripper in self.grippers:
            for link in gripper.links:
                file.write("  {} [shape=cds];\n".format(link.name))
                draw_edge(link, etsbox, jtype, static)

        file.write("}\n")

        if isinstance(filename, str):
            file.close()  # noqa

    # --------------------------------------------------------------------- #
    # --------- Kinematic Methods ----------------------------------------- #
    # --------------------------------------------------------------------- #

    @property
    def reach(self) -> float:
        r"""
        Reach of the robot

        A conservative estimate of the reach of the robot. It is computed as
        the sum of the translational ETs that define the link transform.

        Note
        ----
        Computed on the first access. If kinematic parameters
        subsequently change this will not be reflected.

        Returns
        -------
        reach
            Maximum reach of the robot

        Notes
        -----
        - Probably an overestimate of reach
        - Used by numerical inverse kinematics to scale translational
          error.
        - For a prismatic joint, uses ``qlim`` if it is set

        """

        # TODO
        # This should be a start, end method and compute the reach based on the
        # given ets. Then use an lru_cache to speed up return

        if self._reach is None:
            d_all = []
            for link in self.ee_links:
                d = 0
                while True:
                    for et in link.ets:
                        if et.istranslation:
                            if et.isjoint:
                                # the length of a prismatic joint depends on the
                                # joint limits.  They might be set in the ET
                                # or in the Link depending on how the robot
                                # was constructed
                                if link.qlim is not None:
                                    d += max(link.qlim)
                                elif et.qlim is not None:
                                    d += max(et.qlim)
                            else:
                                d += abs(et.eta)
                    link = link.parent
                    if link is None or isinstance(link, str):
                        d_all.append(d)
                        break

            self._reach = max(d_all)
        return self._reach

    def fkine_all(self, q: ArrayLike) -> SE3:
        """
        Compute the pose of every link frame

        ``T = robot.fkine_all(q)`` is  an SE3 instance with ``robot.nlinks +
        1`` values:

        - ``T[0]`` is the base transform
        - ``T[i]`` is the pose of link whose ``number`` is ``i``

        Parameters
        ----------
        q
            The joint configuration

        Returns
        -------
        fkine_all
            Pose of all links

        References
        ----------
        - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
          Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

        """
        q = getvector(q)

        Tbase = SE3(self.base)  # add base, also sets the type

        linkframes = Tbase.__class__.Alloc(self.nlinks + 1)
        linkframes[0] = Tbase

        def recurse(Tall, Tparent, q, link):
            # if joint??
            T = Tparent
            while True:
                T *= SE3(link.A(q[link.jindex]))

                Tall[link.number] = T

                if link.nchildren == 0:
                    # no children
                    return
                elif link.nchildren == 1:
                    # one child
                    if link in self.ee_links:
                        # this link is an end-effector, go no further
                        return
                    link = link.children[0]
                    continue
                else:
                    # multiple children
                    for child in link.children:
                        recurse(Tall, T, q, child)
                    return

        recurse(linkframes, Tbase, q, self.links[0])

        return linkframes

    @overload
    def manipulability(
        self,
        q: ArrayLike = ...,
        J: None = None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        method: L["yoshikawa", "asada", "minsingular", "invcondition"] = "yoshikawa",
        axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
        **kwargs,
    ) -> Union[float, NDArray]:
        ...

    @overload
    def manipulability(
        self,
        q: None = None,
        J: NDArray = ...,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        method: L["yoshikawa", "asada", "minsingular", "invcondition"] = "yoshikawa",
        axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
        **kwargs,
    ) -> Union[float, NDArray]:
        ...

    def manipulability(
        self,
        q=None,
        J=None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        method: L["yoshikawa", "asada", "minsingular", "invcondition"] = "yoshikawa",
        axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
        **kwargs,
    ):
        """
        Manipulability measure

        ``manipulability(q)`` is the scalar manipulability index
        for the robot at the joint configuration ``q``.  It indicates
        dexterity, that is, how well conditioned the robot is for motion
        with respect to the 6 degrees of Cartesian motion.  The values is
        zero if the robot is at a singularity.

        Parameters
        ----------
        q
            Joint coordinates, one of J or q required
        J
            Jacobian in base frame if already computed, one of J or
            q required
        method
            method to use, "yoshikawa" (default), "invcondition",
            "minsingular"  or "asada"
        axes
            Task space axes to consider: "all" [default],
            "trans", or "rot"

        Returns
        -------
        manipulability

        Synopsis
        --------

        Various measures are supported:

        | Measure           |       Description                               |
        |-------------------|-------------------------------------------------|
        | ``"yoshikawa"``   | Volume of the velocity ellipsoid, *distance*    |
        |                   | from singularity [Yoshikawa85]_                 |
        | ``"invcondition"``| Inverse condition number of Jacobian, isotropy  |
        |                   | of the velocity ellipsoid [Klein87]_            |
        | ``"minsingular"`` | Minimum singular value of the Jacobian,         |
        |                   | *distance*  from singularity [Klein87]_         |
        | ``"asada"``       | Isotropy of the task-space acceleration         |
        |                   | ellipsoid which is a function of the Cartesian  |
        |                   | inertia matrix which depends on the inertial    |
        |                   | parameters [Asada83]_                           |

        **Trajectory operation**:

        If ``q`` is a matrix (m,n) then the result (m,) is a vector of
        manipulability indices for each joint configuration specified by a row
        of ``q``.

        Notes
        -----
        - Invokes the ``jacob0`` method of the robot if ``J`` is not passed
        - The "all" option includes rotational and translational
            dexterity, but this involves adding different units. It can be
            more useful to look at the translational and rotational
            manipulability separately.
        - Examples in the RVC book (1st edition) can be replicated by
            using the "all" option
        - Asada's measure requires inertial a robot model with inertial
            parameters.

        References
        ----------
        .. [Yoshikawa85] Manipulability of Robotic Mechanisms. Yoshikawa T.,
                The International Journal of Robotics Research.
                1985;4(2):3-9. doi:10.1177/027836498500400201
        .. [Asada83] A geometrical representation of manipulator dynamics and
                its application to arm design, H. Asada,
                Journal of Dynamic Systems, Measurement, and Control,
                vol. 105, p. 131, 1983.
        .. [Klein87] Dexterity Measures for the Design and Control of
                Kinematically Redundant Manipulators. Klein CA, Blaho BE.
                The International Journal of Robotics Research.
                1987;6(2):72-83. doi:10.1177/027836498700600206
        - Robotics, Vision & Control, Chap 8, P. Corke, Springer 2011.


        .. versionchanged:: 1.0.3
            Removed 'both' option for axes, added a custom list option.

        """

        ets = self.ets(end, start)

        axes_list: List[bool] = []

        if isinstance(axes, list):
            axes_list = axes
        elif axes == "all":
            axes_list = [True, True, True, True, True, True]
        elif axes.startswith("trans"):
            axes_list = [True, True, True, False, False, False]
        elif axes.startswith("rot"):
            axes_list = [False, False, False, True, True, True]
        else:
            raise ValueError("axes must be all, trans, rot or both")

        def yoshikawa(robot, J, q, axes_list):
            J = J[axes_list, :]
            if J.shape[0] == J.shape[1]:
                # simplified case for square matrix
                return abs(np.linalg.det(J))
            else:
                m2 = np.linalg.det(J @ J.T)
                return np.sqrt(abs(m2))

        def condition(robot, J, q, axes_list):
            J = J[axes_list, :]

            # return 1/cond(J)
            return 1 / np.linalg.cond(J)

        def minsingular(robot, J, q, axes_list):
            J = J[axes_list, :]
            s = np.linalg.svd(J, compute_uv=False)

            # return last/smallest singular value of J
            return s[-1]

        def asada(robot, J, q, axes_list):
            # dof = np.sum(axes_list)
            if np.linalg.matrix_rank(J) < 6:
                return 0
            Ji = np.linalg.pinv(J)
            Mx = Ji.T @ robot.inertia(q) @ Ji
            d = np.where(axes_list)[0]
            Mx = Mx[d]
            Mx = Mx[:, d.tolist()]
            e, _ = np.linalg.eig(Mx)
            return np.min(e) / np.max(e)

        # choose the handler function
        if method.lower().startswith("yoshi"):
            mfunc = yoshikawa
        elif method.lower().startswith("invc"):
            mfunc = condition
        elif method.lower().startswith("mins"):
            mfunc = minsingular
        elif method.lower().startswith("asa"):
            mfunc = asada
        else:
            raise ValueError("Invalid method chosen")

        # Calculate manipulability based on supplied Jacobian
        if J is not None:
            w = [mfunc(self, J, q, axes_list)]

        # Otherwise use the q vector/matrix
        else:
            q = np.array(getmatrix(q, (None, self.n)))
            w = np.zeros(q.shape[0])

            for k, qk in enumerate(q):
                Jk = ets.jacob0(qk)
                w[k] = mfunc(self, Jk, qk, axes_list)

        if len(w) == 1:
            return w[0]
        else:
            return w

    def jtraj(
        self,
        T1: Union[NDArray, SE3],
        T2: Union[NDArray, SE3],
        t: Union[NDArray, int],
        **kwargs,
    ):
        """
        Joint-space trajectory between SE(3) poses

        The initial and final poses are mapped to joint space using inverse
        kinematics:

        - if the object has an analytic solution ``ikine_a`` that will be used,
        - otherwise the general numerical algorithm ``ikine_lm`` will be used.

        ``traj = obot.jtraj(T1, T2, t)`` is a trajectory object whose
        attribute ``traj.q`` is a row-wise joint-space trajectory.

        Parameters
        ----------
        T1
            initial end-effector pose
        T2
            final end-effector pose
        t
            time vector or number of steps
        kwargs
            arguments passed to the IK solver

        Returns
        -------
        trajectory

        """

        if hasattr(self, "ikine_a"):
            ik = self.ikine_a  # type: ignore
        else:
            ik = self.ikine_LM

        q1 = ik(T1, **kwargs)
        q2 = ik(T2, **kwargs)

        return rtb.jtraj(q1.q, q2.q, t)

    @overload
    def jacob0_dot(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        J0: None = None,
        representation: Union[L["rpy/xyz", "rpy/zyx", "eul", "exp"], None] = None,
    ) -> NDArray:
        ...

    @overload
    def jacob0_dot(
        self,
        q: None,
        qd: ArrayLike,
        J0: NDArray = ...,
        representation: Union[L["rpy/xyz", "rpy/zyx", "eul", "exp"], None] = None,
    ) -> NDArray:
        ...

    def jacob0_dot(
        self,
        q,
        qd: ArrayLike,
        J0=None,
        representation: Union[L["rpy/xyz", "rpy/zyx", "eul", "exp"], None] = None,
    ):
        r"""
        Derivative of Jacobian

        ``robot.jacob_dot(q, qd)`` computes the rate of change of the
        Jacobian elements

        .. math::

            \dmat{J} = \frac{d \mat{J}}{d \vec{q}} \frac{d \vec{q}}{dt}

        where the first term is the rank-3 Hessian.

        Parameters
        ----------
        q
        The joint configuration of the robot
        qd
            The joint velocity of the robot
        J0
            Jacobian in {0} frame
        representation
            angular representation

        Returns
        -------
        jdot
            The derivative of the manipulator Jacobian

        Synopsis
        --------

        If ``J0`` is already calculated for the joint
        coordinates ``q`` it can be passed in to to save computation time.

        It is computed as the mode-3 product of the Hessian tensor and the
        velocity vector.

        The derivative of an analytical Jacobian can be obtained by setting
        ``representation`` as

        |``representation``   |       Rotational representation     |
        |---------------------|-------------------------------------|
        |``'rpy/xyz'``        |   RPY angular rates in XYZ order    |
        |``'rpy/zyx'``        |   RPY angular rates in XYZ order    |
        |``'eul'``            |   Euler angular rates in ZYZ order  |
        |``'exp'``            |   exponential coordinate rates      |


        References
        ----------
        - Kinematic Derivatives using the Elementary Transform
            Sequence, J. Haviland and P. Corke

        See Also
        --------
        :func:`jacob0`
        :func:`hessian0`

        """

        qd = np.array(qd)

        if representation is None:

            if J0 is None:
                J0 = self.jacob0(q)
            H = self.hessian0(q, J0=J0)

        else:
            # # determine analytic rotation
            # T = self.fkine(q).A
            # gamma = smb.r2x(smb.t2r(T), representation=representation)

            # # get transformation angular velocity to analytic velocity
            # Ai = smb.rotvelxform(
            #     gamma, representation=representation, inverse=True, full=True
            # )

            # # get analytic rate from joint rates
            # omega = J0[3:, :] @ qd
            # gamma_dot = Ai[3:, 3:] @ omega
            # Ai_dot = smb.rotvelxform_inv_dot(gamma, gamma_dot, full=True)
            # Ai_dot = sp.linalg.block_diag(np.zeros((3, 3)), Ai_dot)

            # Jd = Ai_dot @ J0 + Ai @ Jd

            # not actually sure this can be written in closed form

            H = smb.numhess(
                lambda q: self.jacob0_analytical(q, representation=representation), q
            )

            # Jd = Ai @ Jd

            # return Jd

        return np.tensordot(H, qd, (0, 0))

    # @overload
    # def jacobm(
    #     self,
    #     q: ArrayLike = ...,
    #     J: None = None,
    #     H: None = None,
    #     end: Union[str, Link, Gripper, None] = None,
    #     start: Union[str, Link, Gripper, None] = None,
    #     axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
    # ):
    #     ...

    # @overload
    # def jacobm(
    #     self,
    #     q: None = None,
    #     J: NDArray = ...,
    #     H: NDArray = ...,
    #     end: Union[str, Link, Gripper, None] = None,
    #     start: Union[str, Link, Gripper, None] = None,
    #     axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
    # ):
    #     ...

    # def jacobm(
    #     self,
    #     q=None,
    #     J=None,
    #     H=None,
    #     end: Union[str, Link, Gripper, None] = None,
    #     start: Union[str, Link, Gripper, None] = None,
    #     axes: Union[L["all", "trans", "rot"], List[bool]] = "all",
    # ):
    #     r"""
    #     The manipulability Jacobian

    #     This measure relates the rate of change of the manipulability to the
    #     joint velocities of the robot. One of J or q is required. Supply J
    #     and H if already calculated to save computation time

    #     Parameters
    #     ----------
    #     q
    #         The joint angles/configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     J
    #         The manipulator Jacobian in any frame
    #     H
    #         The manipulator Hessian in any frame
    #     end
    #         the final link or Gripper which the Hessian represents
    #     start
    #         the first link which the Hessian represents

    #     Returns
    #     -------
    #     jacobm
    #         The manipulability Jacobian

    #     Synopsis
    #     --------
    #     Yoshikawa's manipulability measure

    #     .. math::

    #         m(\vec{q}) = \sqrt{\mat{J}(\vec{q}) \mat{J}(\vec{q})^T}

    #     This method returns its Jacobian with respect to configuration

    #     .. math::

    #         \frac{\partial m(\vec{q})}{\partial \vec{q}}

    #     References
    #     ----------
    #     - J. Haviland, and P. Corke. "Manipulator Differential Kinematics Part I:
    #       Kinematics, Velocity, and Applications." arXiv preprint arXiv:2207.01796 (2022).

    #     """

    #     end, start, _ = self._get_limit_links(end, start)

    #     if axes == "all":
    #         axes = [True, True, True, True, True, True]
    #     elif axes.startswith("trans"):
    #         axes = [True, True, True, False, False, False]
    #     elif axes.startswith("rot"):
    #         axes = [False, False, False, True, True, True]
    #     else:
    #         raise ValueError("axes must be all, trans or rot")

    #     if J is None:
    #         if q is None:
    #             q = np.copy(self.q)
    #         else:
    #             q = getvector(q, self.n)

    #         J = self.jacob0(q, start=start, end=end)
    #     else:
    #         verifymatrix(J, (6, self.n))

    #     n = J.shape[1]

    #     if H is None:
    #         H = self.hessian0(J0=J, start=start, end=end)
    #     else:
    #         verifymatrix(H, (6, self.n, self.n))

    #     manipulability = self.manipulability(q, J=J, start=start, end=end, axes=axes)

    #     J = J[axes, :]
    #     H = H[:, axes, :]

    #     b = np.linalg.inv(J @ np.transpose(J))
    #     Jm = np.zeros((n, 1))

    #     for i in range(n):
    #         c = J @ np.transpose(H[i, :, :])
    #         Jm[i, 0] = manipulability * np.transpose(c.flatten("F")) @ b.flatten("F")

    #     return Jm

    # --------------------------------------------------------------------- #

    # --------------------------------------------------------------------- #

    # # TODO probably should be a static method
    # def _get_graphical_backend(self, backend=None):

    #     default = self.default_backend

    #     # figure out the right default
    #     if backend is None:
    #         if isinstance(self, rtb.DHRobot):
    #             default = "pyplot"
    #         elif isinstance(self, rtb.ERobot2):
    #             default = "pyplot2"
    #         elif isinstance(self, rtb.ERobot):
    #             if self.hasgeometry:
    #                 default = "swift"
    #             else:
    #                 default = "pyplot"

    #     if backend is not None:
    #         backend = backend.lower()

    #     # find the right backend, modules are imported here on an as needs
    #     # basis
    #     if backend == "swift" or default == "swift":  # pragma nocover
    #         # swift was requested, is it installed?
    #         if isinstance(self, rtb.DHRobot):
    #             raise NotImplementedError(
    #                 "Plotting in Swift is not implemented for DHRobots yet"
    #             )
    #         try:
    #             # yes, use it
    #             from roboticstoolbox.backends.swift import Swift

    #             env = Swift()
    #             return env
    #         except ModuleNotFoundError:
    #             if backend == "swift":
    #                 print("Swift is not installed, " "install it using pip or conda")
    #             backend = "pyplot"

    #     elif backend == "vpython" or default == "vpython":  # pragma nocover
    #         # vpython was requested, is it installed?
    #         if not isinstance(self, rtb.DHRobot):
    #             raise NotImplementedError(
    #                 "Plotting in VPython is only implemented for DHRobots"
    #             )
    #         try:
    #             # yes, use it
    #             from roboticstoolbox.backends.VPython import VPython

    #             env = VPython()
    #             return env
    #         except ModuleNotFoundError:
    #             if backend == "vpython":
    #                 print("VPython is not installed, " "install it using pip or conda")
    #             backend = "pyplot"
    #     if backend is None:
    #         backend = default

    #     if backend == "pyplot":
    #         from roboticstoolbox.backends.PyPlot import PyPlot

    #         env = PyPlot()

    #     elif backend == "pyplot2":
    #         from roboticstoolbox.backends.PyPlot import PyPlot2

    #         env = PyPlot2()

    #     else:
    #         raise ValueError("unknown backend", backend)

    #     return env

    # def plot(
    #     self,
    #     q,
    #     backend=None,
    #     block=False,
    #     dt=0.050,
    #     limits=None,
    #     vellipse=False,
    #     fellipse=False,
    #     fig=None,
    #     movie=None,
    #     loop=False,
    #     **kwargs,
    # ):
    #     """
    #     Graphical display and animation

    #     :param q: The joint configuration of the robot.
    #     :type q: float ndarray(n)
    #     :param backend: The graphical backend to use, currently 'swift'
    #         and 'pyplot' are implemented. Defaults to 'swift' of an ``ERobot``
    #         and 'pyplot` for a ``DHRobot``
    #     :type backend: string
    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param dt: if q is a trajectory, this describes the delay in
    #         seconds between frames
    #     :type dt: float
    #     :param limits: Custom view limits for the plot. If not supplied will
    #         autoscale, [x1, x2, y1, y2, z1, z2]
    #         (this option is for 'pyplot' only)
    #     :type limits: ndarray(6)
    #     :param vellipse: (Plot Option) Plot the velocity ellipse at the
    #         end-effector (this option is for 'pyplot' only)
    #     :type vellipse: bool
    #     :param vellipse: (Plot Option) Plot the force ellipse at the
    #         end-effector (this option is for 'pyplot' only)
    #     :type vellipse: bool
    #     :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
    #         which the joint revolves around(revolute joint) or translates
    #         along (prosmatic joint) (this option is for 'pyplot' only)
    #     :type jointaxes: bool
    #     :param eeframe: (Plot Option) Plot the end-effector coordinate frame
    #         at the location of the end-effector. Uses three arrows, red,
    #         green and blue to indicate the x, y, and z-axes.
    #         (this option is for 'pyplot' only)
    #     :type eeframe: bool
    #     :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
    #         plane. (this option is for 'pyplot' only)
    #     :type shadow: bool
    #     :param name: (Plot Option) Plot the name of the robot near its base
    #         (this option is for 'pyplot' only)
    #     :type name: bool
    #     :param movie: name of file in which to save an animated GIF
    #         (this option is for 'pyplot' only)
    #     :type movie: str

    #     :return: A reference to the environment object which controls the
    #         figure
    #     :rtype: Swift or PyPlot

    #     - ``robot.plot(q, 'pyplot')`` displays a graphical view of a robot
    #       based on the kinematic model and the joint configuration ``q``.
    #       This is a stick figure polyline which joins the origins of the
    #       link coordinate frames. The plot will autoscale with an aspect
    #       ratio of 1.

    #     If ``q`` (m,n) representing a joint-space trajectory it will create an
    #     animation with a pause of ``dt`` seconds between each frame.

    #     .. note::
    #         - By default this method will block until the figure is dismissed.
    #           To avoid this set ``block=False``.
    #         - For PyPlot, the polyline joins the origins of the link frames,
    #           but for some Denavit-Hartenberg models those frames may not
    #           actually be on the robot, ie. the lines to not neccessarily
    #           represent the links of the robot.

    #     :seealso: :func:`teach`
    #     """

    #     env = None

    #     env = self._get_graphical_backend(backend)

    #     q = getmatrix(q, (None, self.n))
    #     self.q = q[0, :]

    #     # Add the self to the figure in readonly mode
    #     # Add the self to the figure in readonly mode
    #     if q.shape[0] == 1:
    #         env.launch(self.name + " Plot", limits=limits, fig=fig)
    #     else:
    #         env.launch(self.name + " Trajectory Plot", limits=limits, fig=fig)

    #     env.add(self, readonly=True, **kwargs)

    #     if vellipse:
    #         vell = self.vellipse(centre="ee")
    #         env.add(vell)

    #     if fellipse:
    #         fell = self.fellipse(centre="ee")
    #         env.add(fell)

    #     # Stop lint error
    #     images = []  # list of images saved from each plot

    #     if movie is not None:
    #         loop = False

    #     while True:
    #         for qk in q:
    #             self.q = qk
    #             if vellipse:
    #                 vell.q = qk
    #             if fellipse:
    #                 fell.q = qk
    #             env.step(dt)

    #             if movie is not None:  # pragma nocover
    #                 images.append(env.getframe())

    #         if movie is not None:  # pragma nocover
    #             # save it as an animated GIF
    #             images[0].save(
    #                 movie,
    #                 save_all=True,
    #                 append_images=images[1:],
    #                 optimize=False,
    #                 duration=dt,
    #                 loop=0,
    #             )
    #         if not loop:
    #             break

    #     # Keep the plot open
    #     if block:  # pragma: no cover
    #         env.hold()

    #     return env

    # --------------------------------------------------------------------- #

    # def fellipse(self, q=None, opt="trans", unit="rad", centre=[0, 0, 0]):
    #     """
    #     Create a force ellipsoid object for plotting with PyPlot

    #     :param q: The joint configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param opt: 'trans' or 'rot' will plot either the translational or
    #         rotational force ellipsoid
    #     :type opt: string
    #     :param centre:
    #     :type centre: list or str('ee')

    #     :return: An EllipsePlot object
    #     :rtype: EllipsePlot

    #     - ``robot.fellipse(q)`` creates a force ellipsoid for the robot at
    #       pose ``q``. The ellipsoid is centered at the origin.

    #     - ``robot.fellipse()`` as above except the joint configuration is that
    #       stored in the robot object.

    #     .. note::
    #         - By default the ellipsoid related to translational motion is
    #           drawn.  Use ``opt='rot'`` to draw the rotational velocity
    #           ellipsoid.
    #         - By default the ellipsoid is drawn at the origin.  The option
    #           ``centre`` allows its origin to set to set to the specified
    #           3-vector, or the string "ee" ensures it is drawn at the
    #           end-effector position.

    #     """
    #     if isinstance(self, rtb.ERobot):  # pragma nocover
    #         raise NotImplementedError("ERobot fellipse not implemented yet")

    #     q = getunit(q, unit)
    #     ell = EllipsePlot(self, q, "f", opt, centre=centre)
    #     return ell

    # def vellipse(self, q=None, opt="trans", unit="rad", centre=[0, 0, 0], scale=0.1):
    #     """
    #     Create a velocity ellipsoid object for plotting with PyPlot

    #     :param q: The joint configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param opt: 'trans' or 'rot' will plot either the translational or
    #         rotational velocity ellipsoid
    #     :type opt: string
    #     :param centre:
    #     :type centre: list or str('ee')

    #     :return: An EllipsePlot object
    #     :rtype: EllipsePlot

    #     - ``robot.vellipse(q)`` creates a force ellipsoid for the robot at
    #       pose ``q``. The ellipsoid is centered at the origin.

    #     - ``robot.vellipse()`` as above except the joint configuration is that
    #       stored in the robot object.

    #     .. note::
    #         - By default the ellipsoid related to translational motion is
    #           drawn.  Use ``opt='rot'`` to draw the rotational velocity
    #           ellipsoid.
    #         - By default the ellipsoid is drawn at the origin.  The option
    #           ``centre`` allows its origin to set to set to the specified
    #           3-vector, or the string "ee" ensures it is drawn at the
    #           end-effector position.
    #     """
    #     if isinstance(self, rtb.ERobot):  # pragma nocover
    #         raise NotImplementedError("ERobot vellipse not implemented yet")

    #     q = getunit(q, unit)
    #     ell = EllipsePlot(self, q, "v", opt, centre=centre, scale=scale)
    #     return ell

    # def plot_ellipse(
    #     self,
    #     ellipse,
    #     block=True,
    #     limits=None,
    #     jointaxes=True,
    #     eeframe=True,
    #     shadow=True,
    #     name=True,
    # ):
    #     """
    #     Plot the an ellipsoid

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param ellipse: the ellipsoid to plot
    #     :type ellipse: EllipsePlot
    #     :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
    #         which the joint revolves around(revolute joint) or translates
    #         along (prosmatic joint)
    #     :type jointaxes: bool
    #     :param eeframe: (Plot Option) Plot the end-effector coordinate frame
    #         at the location of the end-effector. Uses three arrows, red,
    #         green and blue to indicate the x, y, and z-axes.
    #     :type eeframe: bool
    #     :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
    #         plane
    #     :type shadow: bool
    #     :param name: (Plot Option) Plot the name of the robot near its base
    #     :type name: bool

    #     :return: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     - ``robot.plot_ellipse(ellipsoid)`` displays the ellipsoid.

    #     .. note::
    #         - By default the ellipsoid is drawn at the origin.  The option
    #           ``centre`` allows its origin to set to set to the specified
    #           3-vector, or the string "ee" ensures it is drawn at the
    #           end-effector position.
    #     """

    #     if not isinstance(ellipse, EllipsePlot):  # pragma nocover
    #         raise TypeError(
    #             "ellipse must be of type " "roboticstoolbox.backend.PyPlot.EllipsePlot"
    #         )

    #     env = PyPlot()

    #     # Add the robot to the figure in readonly mode
    #     env.launch(ellipse.robot.name + " " + ellipse.name, limits=limits)

    #     env.add(ellipse, jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

    #     # Keep the plot open
    #     if block:  # pragma: no cover
    #         env.hold()

    #     return env

    # def plot_fellipse(
    #     self,
    #     q=None,
    #     block=True,
    #     fellipse=None,
    #     limits=None,
    #     opt="trans",
    #     centre=[0, 0, 0],
    #     jointaxes=True,
    #     eeframe=True,
    #     shadow=True,
    #     name=True,
    # ):
    #     """
    #     Plot the force ellipsoid for manipulator

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param q: The joint configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param fellipse: the vellocity ellipsoid to plot
    #     :type fellipse: EllipsePlot
    #     :param limits: Custom view limits for the plot. If not supplied will
    #         autoscale, [x1, x2, y1, y2, z1, z2]
    #     :type limits: ndarray(6)
    #     :param opt: 'trans' or 'rot' will plot either the translational or
    #         rotational force ellipsoid
    #     :type opt: string
    #     :param centre: The coordinates to plot the fellipse [x, y, z] or "ee"
    #         to plot at the end-effector location
    #     :type centre: array_like or str
    #     :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
    #         which the joint revolves around(revolute joint) or translates
    #         along (prosmatic joint)
    #     :type jointaxes: bool
    #     :param eeframe: (Plot Option) Plot the end-effector coordinate frame
    #         at the location of the end-effector. Uses three arrows, red,
    #         green and blue to indicate the x, y, and z-axes.
    #     :type eeframe: bool
    #     :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
    #         plane
    #     :type shadow: bool
    #     :param name: (Plot Option) Plot the name of the robot near its base
    #     :type name: bool

    #     :return: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     - ``robot.plot_fellipse(q)`` displays the velocity ellipsoid for the
    #       robot at pose ``q``. The plot will autoscale with an aspect ratio
    #       of 1.

    #     - ``plot_fellipse()`` as above except the robot is plotted with joint
    #       coordinates stored in the robot object.

    #     - ``robot.plot_fellipse(vellipse)`` specifies a custon ellipse to plot.

    #     .. note::
    #         - By default the ellipsoid related to translational motion is
    #           drawn.  Use ``opt='rot'`` to draw the rotational velocity
    #           ellipsoid.
    #         - By default the ellipsoid is drawn at the origin.  The option
    #           ``centre`` allows its origin to set to set to the specified
    #           3-vector, or the string "ee" ensures it is drawn at the
    #           end-effector position.
    #     """

    #     if isinstance(self, rtb.ERobot):  # pragma nocover
    #         raise NotImplementedError(
    #             "Ellipse Plotting of ERobot's not implemented yet"
    #         )

    #     if q is not None:
    #         self.q = q

    #     if fellipse is None:
    #         fellipse = self.fellipse(q=q, opt=opt, centre=centre)

    #     return self.plot_ellipse(
    #         fellipse,
    #         block=block,
    #         limits=limits,
    #         jointaxes=jointaxes,
    #         eeframe=eeframe,
    #         shadow=shadow,
    #         name=name,
    #     )

    # def plot_vellipse(
    #     self,
    #     q=None,
    #     block=True,
    #     vellipse=None,
    #     limits=None,
    #     opt="trans",
    #     centre=[0, 0, 0],
    #     jointaxes=True,
    #     eeframe=True,
    #     shadow=True,
    #     name=True,
    # ):
    #     """
    #     Plot the velocity ellipsoid for manipulator

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param q: The joint configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param vellipse: the vellocity ellipsoid to plot
    #     :type vellipse: EllipsePlot
    #     :param limits: Custom view limits for the plot. If not supplied will
    #         autoscale, [x1, x2, y1, y2, z1, z2]
    #     :type limits: ndarray(6)
    #     :param opt: 'trans' or 'rot' will plot either the translational or
    #         rotational velocity ellipsoid
    #     :type opt: string
    #     :param centre: The coordinates to plot the vellipse [x, y, z] or "ee"
    #         to plot at the end-effector location
    #     :type centre: array_like or str
    #     :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
    #         which the joint revolves around(revolute joint) or translates
    #         along (prosmatic joint)
    #     :type jointaxes: bool
    #     :param eeframe: (Plot Option) Plot the end-effector coordinate frame
    #         at the location of the end-effector. Uses three arrows, red,
    #         green and blue to indicate the x, y, and z-axes.
    #     :type eeframe: bool
    #     :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
    #         plane
    #     :type shadow: bool
    #     :param name: (Plot Option) Plot the name of the robot near its base
    #     :type name: bool

    #     :return: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     - ``robot.plot_vellipse(q)`` displays the velocity ellipsoid for the
    #       robot at pose ``q``. The plot will autoscale with an aspect ratio
    #       of 1.

    #     - ``plot_vellipse()`` as above except the robot is plotted with joint
    #       coordinates stored in the robot object.

    #     - ``robot.plot_vellipse(vellipse)`` specifies a custon ellipse to plot.

    #     .. note::
    #         - By default the ellipsoid related to translational motion is
    #           drawn.  Use ``opt='rot'`` to draw the rotational velocity
    #           ellipsoid.
    #         - By default the ellipsoid is drawn at the origin.  The option
    #           ``centre`` allows its origin to set to set to the specified
    #           3-vector, or the string "ee" ensures it is drawn at the
    #           end-effector position.
    #     """

    #     if isinstance(self, rtb.ERobot):  # pragma nocover
    #         raise NotImplementedError(
    #             "Ellipse Plotting of ERobot's not implemented yet"
    #         )

    #     if q is not None:
    #         self.q = q

    #     if vellipse is None:
    #         vellipse = self.vellipse(q=q, opt=opt, centre=centre)

    #     return self.plot_ellipse(
    #         vellipse,
    #         block=block,
    #         limits=limits,
    #         jointaxes=jointaxes,
    #         eeframe=eeframe,
    #         shadow=shadow,
    #         name=name,
    #     )

    # --------------------------------------------------------------------- #

    # def teach(
    #     self,
    #     q=None,
    #     block=True,
    #     order="xyz",
    #     limits=None,
    #     jointaxes=True,
    #     jointlabels=False,
    #     vellipse=False,
    #     fellipse=False,
    #     eeframe=True,
    #     shadow=True,
    #     name=True,
    #     backend=None,
    # ):
    #     """
    #     Graphical teach pendant

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param q: The joint configuration of the robot (Optional,
    #               if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param limits: Custom view limits for the plot. If not supplied will
    #                    autoscale, [x1, x2, y1, y2, z1, z2]
    #     :type limits: ndarray(6)
    #     :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
    #                       which the joint revolves around(revolute joint) or
    #                       translates along (prismatic joint)
    #     :type jointaxes: bool
    #     :param eeframe: (Plot Option) Plot the end-effector coordinate frame
    #         at the location of the end-effector. Uses three arrows, red,
    #         green and blue to indicate the x, y, and z-axes.
    #     :type eeframe: bool
    #     :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
    #         plane
    #     :type shadow: bool
    #     :param name: (Plot Option) Plot the name of the robot near its base
    #     :type name: bool

    #     :return: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     - ``robot.teach(q)`` creates a matplotlib plot which allows the user to
    #       "drive" a graphical robot using a graphical slider panel. The robot's
    #       inital joint configuration is ``q``. The plot will autoscale with an
    #       aspect ratio of 1.

    #     - ``robot.teach()`` as above except the robot's stored value of ``q``
    #         is used.

    #     .. note::
    #         - Program execution is blocked until the teach window is
    #           dismissed.  If ``block=False`` the method is non-blocking but
    #           you need to poll the window manager to ensure that the window
    #           remains responsive.
    #         - The slider limits are derived from the joint limit properties.
    #           If not set then:
    #             - For revolute joints they are assumed to be [-pi, +pi]
    #             - For prismatic joint they are assumed unknown and an error
    #               occurs.
    #     """

    #     if q is None:
    #         q = np.zeros((self.n,))
    #     else:
    #         q = getvector(q, self.n)

    #     # Make an empty 3D figure
    #     env = self._get_graphical_backend(backend)

    #     # Add the self to the figure in readonly mode
    #     env.launch("Teach " + self.name, limits=limits)
    #     env.add(
    #         self,
    #         readonly=True,
    #         # jointaxes=jointaxes,
    #         # jointlabels=jointlabels,
    #         # eeframe=eeframe,
    #         # shadow=shadow,
    #         # name=name,
    #     )

    #     env._add_teach_panel(self, q)

    #     if vellipse:
    #         vell = self.vellipse(centre="ee", scale=0.5)
    #         env.add(vell)

    #     if fellipse:
    #         fell = self.fellipse(centre="ee")
    #         env.add(fell)

    #     # Keep the plot open
    #     if block:  # pragma: no cover
    #         env.hold()

    #     return env

    # --------------------------------------------------------------------- #

    # def closest_point(
    #     self, q: ArrayLike, shape: Shape, inf_dist: float = 1.0, skip: bool = False
    # ) -> Tuple[Union[int, None], Union[NDArray, None], Union[NDArray, None],]:
    #     """
    #     closest_point(shape, inf_dist) returns the minimum euclidean
    #     distance between this robot and shape, provided it is less than
    #     inf_dist. It will also return the points on self and shape in the
    #     world frame which connect the line of length distance between the
    #     shapes. If the distance is negative then the shapes are collided.

    #     :param shape: The shape to compare distance to
    #     :param inf_dist: The minimum distance within which to consider
    #         the shape
    #     :param skip: Skip setting all shape transforms based on q, use this
    #         option if using this method in conjuction with Swift to save time

    #     :returns: d, p1, p2 where d is the distance between the shapes,
    #         p1 and p2 are the points in the world frame on the respective
    #         shapes. The points returned are [x, y, z].
    #     """

    #     if not skip:
    #         self._update_link_tf(q)
    #         self._propogate_scene_tree()
    #         shape._propogate_scene_tree()

    #     d = 10000
    #     p1 = None
    #     p2 = None

    #     for link in self.links:
    #         td, tp1, tp2 = link.closest_point(shape, inf_dist, skip=True)

    #         if td is not None and td < d:
    #             d = td
    #             p1 = tp1
    #             p2 = tp2

    #     if d == 10000:
    #         d = None

    #     return d, p1, p2

    # def iscollided(self, q, shape, skip=False):
    #     """
    #     collided(shape) checks if this robot and shape have collided
    #     :param shape: The shape to compare distance to
    #     :type shape: Shape
    #     :param skip: Skip setting all shape transforms based on q, use this
    #         option if using this method in conjuction with Swift to save time
    #     :type skip: boolean
    #     :returns: True if shapes have collided
    #     :rtype: bool
    #     """

    #     if not skip:
    #         self._update_link_tf(q)
    #         self._propogate_scene_tree()
    #         shape._propogate_scene_tree()

    #     for link in self.links:
    #         if link.iscollided(shape, skip=True):
    #             return True

    #     if isinstance(self, rtb.ERobot):
    #         for gripper in self.grippers:
    #             for link in gripper.links:
    #                 if link.iscollided(shape, skip=True):
    #                     return True

    #     return False

    # def collided(self, q, shape, skip=False):
    #     """
    #     collided(shape) checks if this robot and shape have collided
    #     :param shape: The shape to compare distance to
    #     :type shape: Shape
    #     :param skip: Skip setting all shape transforms based on q, use this
    #         option if using this method in conjuction with Swift to save time
    #     :type skip: boolean
    #     :returns: True if shapes have collided
    #     :rtype: bool
    #     """
    #     warn("method collided is deprecated, use iscollided instead", FutureWarning)
    #     return self.iscollided(q, shape, skip=skip)

    # def joint_velocity_damper(self, ps=0.05, pi=0.1, n=None, gain=1.0):
    #     """
    #     Formulates an inequality contraint which, when optimised for will
    #     make it impossible for the robot to run into joint limits. Requires
    #     the joint limits of the robot to be specified. See examples/mmc.py
    #     for use case

    #     :param ps: The minimum angle (in radians) in which the joint is
    #         allowed to approach to its limit
    #     :type ps: float
    #     :param pi: The influence angle (in radians) in which the velocity
    #         damper becomes active
    #     :type pi: float
    #     :param n: The number of joints to consider. Defaults to all joints
    #     :type n: int
    #     :param gain: The gain for the velocity damper
    #     :type gain: float

    #     :returns: Ain, Bin as the inequality contraints for an optisator
    #     :rtype: ndarray(6), ndarray(6)
    #     """

    #     if n is None:
    #         n = self.n

    #     Ain = np.zeros((n, n))
    #     Bin = np.zeros(n)

    #     for i in range(n):
    #         if self.q[i] - self.qlim[0, i] <= pi:
    #             Bin[i] = -gain * (((self.qlim[0, i] - self.q[i]) + ps) / (pi - ps))
    #             Ain[i, i] = -1
    #         if self.qlim[1, i] - self.q[i] <= pi:
    #             Bin[i] = gain * ((self.qlim[1, i] - self.q[i]) - ps) / (pi - ps)
    #             Ain[i, i] = 1

    #     return Ain, Bin
