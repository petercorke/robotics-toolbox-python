# import sys
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
from typing import Any, Callable, List, Set, Union, Dict, Tuple
from spatialgeometry import Shape
from fknm import Robot_link_T
from functools import lru_cache
from spatialgeometry import SceneNode
from roboticstoolbox.robot.Link import BaseLink, Link

# from numpy import all, eye, isin
from roboticstoolbox.robot.Gripper import Gripper
from numpy import ndarray
from warnings import warn

# import scipy as sp

try:
    from matplotlib import colors
    from matplotlib import cm

    _mpl = True
except ImportError:  # pragma nocover
    pass

_default_backend = None

ArrayLike = Union[list, np.ndarray, tuple, set]


class BaseRobot(SceneNode, ABC):
    def __init__(
        self,
        links: List[BaseLink],
        gripper_links: Union[List[BaseLink], None] = None,
        name: str = "",
        manufacturer: str = "",
        comment: str = "",
        base=Union[np.ndarray, SE3, None],
        tool=Union[np.ndarray, SE3, None],
        gravity: ArrayLike = [0, 0, -9.81],
        keywords: Union[List[str], Tuple[str]] = [],
        symbolic: bool = False,
        configs: Union[Dict[str, ndarray], None] = None,
        check_jindex: bool = True,
    ):

        # Initialise the scene node
        SceneNode.__init__(self)

        # Lets sort out links now
        self._linkdict = {}

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
            self._keywords = keywords

        # Gravity is in the negative-z direction.
        self._gravity = np.array(gravity)

        # Basic arguments
        self.name = name
        self.manufacturer = manufacturer
        self._comment = comment
        self._symbolic = symbolic
        self._reach = None
        self._hasdynamics = False
        self._hasgeometry = False
        self._hascollision = False

        # Time to checkout the links for geometry information
        for link in self.links:
            if not isinstance(link, BaseLink):
                raise TypeError("links should all be Link subclass")

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

    def _sort_links(
        self,
        links: List[BaseLink],
        gripper_links: Union[List[BaseLink], None],
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

        """

        # The ordered links
        orlinks = []

        # The end-effector links
        self._ee_links = []

        # Check all the incoming Link objects
        n = 0

        # Make sure each link has a name
        # ------------------------------
        for k, link in enumerate(links):

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
                link._parent = self._linkdict[link.parent_name]

        if all([link.parent is None for link in links]):

            # No parent links were given, assume they are sequential
            for i in range(len(links) - 1):
                links[i + 1]._parent = links[i]

        # Set the base link
        # -----------------
        for link in links:
            # Is this a base link?
            if link._parent is None:
                try:
                    if self._base_link is not None:
                        raise ValueError("Multiple base links")
                except AttributeError:
                    pass

                self._base_link = link
            else:
                # No, update children of this link's parent
                link._parent._children.append(link)

        if self.base_link is None:  # Pragma: nocover
            raise ValueError(
                "Invalid link configuration provided, must have a base link"
            )

        # Scene node, set links between the links
        # ---------------------------------------
        for link in links:
            if link.parent is not None:
                link.scene_parent = link.parent

        # Set up the gripper, make a list containing the root of all
        # grippers
        # ----------------------------------------------------------
        if gripper_links is not None:
            if isinstance(gripper_links, Link):
                gripper_links = [gripper_links]
        else:
            gripper_links = []

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
        self.ee_links = []

        if len(gripper_links) == 0:
            for link in links:
                # Is this a leaf node? and do we not have any grippers
                if link.children is None or len(link.children) == 0:
                    # No children, must be an end-effector
                    self.ee_links.append(link)
        else:
            for link in gripper_links:
                # Use the passed in value
                self.ee_links.append(link.parent)

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

        elif all([link.jindex is not None for link in links if link.isjoint]):
            # Jindex set on all, check they are unique and contiguous
            if check_jindex:
                jset = set(range(self._n))
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

    # --------------------------------------------------------------------- #
    # --------- Properties ------------------------------------------------ #
    # --------------------------------------------------------------------- #

    @property
    def links(self) -> List[BaseLink]:
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
    def base_link(self) -> BaseLink:
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
    def n(self):
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

    # --------------------------------------------------------------------- #

    @property
    def name(self):
        """
        Get/set robot name

        - ``robot.name`` is the robot name
        - ``robot.name = ...`` checks and sets therobot name

        Returns
        -------
        name
            robot name

        """
        return self._name

    @name.setter
    def name(self, name_new):
        self._name = name_new

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

    # --------------------------------------------------------------------- #

    @property
    def q(self) -> ndarray:
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
    def qd(self) -> ndarray:
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
    def qdd(self) -> ndarray:
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
    def qlim(self) -> ndarray:
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
                continue

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
    def tool(self, T: Union[SE3, np.ndarray]):
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
    def base(self, T: Union[np.ndarray, SE3]):

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

    def todegrees(self, q) -> ndarray:
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

    def toradians(self, q) -> ndarray:
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
        self, start: BaseLink, func: Union[None, Callable[[BaseLink], Any]] = None
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

    # --------------------------------------------------------------------- #


class Robot(SceneNode, ABC, DynamicsMixin):

    _color = True

    def __init__(
        self,
        links,
        name="noname",
        manufacturer="",
        comment="",
        base=SE3(),
        tool=SE3(),
        gravity=None,
        keywords=(),
        symbolic=False,
        configs=None,
    ):

        pass

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

    def __repr__(self):
        return str(self)

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter < len(self.links):
            link = self[self._iter]
            self._iter += 1
            return link
        else:
            raise StopIteration

    def __getitem__(self, i):
        """
        Get link (Robot superclass)

        :param i: link number or name
        :type i: int or str
        :return: i'th link or named link
        :rtype: Link subclass

        This also supports iterating over each link in the robot object,
        from the base to the tool.

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> print(robot[1]) # print the 2nd link
            >>> print([link.a for link in robot])  # print all the a_j values

        .. note:: ``ERobot`` supports link lookup by name,
            eg. ``robot['link1']``
        """
        return self._links[i]

    def dynchanged(self, what=None):
        """
        Dynamic parameters have changed (Robot superclass)

        Called from a property setter to inform the robot that the cache of
        dynamic parameters is invalid.

        :seealso: :func:`roboticstoolbox.Link._listen_dyn`
        """
        self._dynchanged = True
        if what != "gravity":
            self._hasdynamics = True

    def _getq(self, q=None):
        """
        Get joint coordinates (Robot superclass)

        :param q: passed value, defaults to None
        :type q: array_like, optional
        :return: passed or value from robot state
        :rtype: ndarray(n,)
        """
        if q is None:
            return self.q
        elif isvector(q, self.n):
            return getvector(q, self.n)
        else:
            return getmatrix(q, (None, self.n))

    @property
    def configs(self) -> Dict[str, np.ndarray]:
        return self._configs

    # @abstractproperty
    # def nbranches(self):
    #     """
    #     Number of branches (Robot superclass)

    #     :return: Number of branches
    #     :rtype: int

    #     Example:

    #     .. runblock:: pycon

    #         >>> import roboticstoolbox as rtb
    #         >>> robot = rtb.models.DH.Puma560()
    #         >>> robot.nbranches

    #     :seealso: :func:`n`, :func:`nlinks`
    #     """
    #     return self._n

    # @property
    # def qrandom(self):
    #     """
    #     Return a random joint configuration

    #     :return: Random joint configuration :rtype: ndarray(n)

    #     The value for each joint is uniform randomly distributed  between the
    #     limits set for the robot.

    #     .. note:: The joint limit for all joints must be set.

    #     :seealso: :func:`Robot.qlim`, :func:`Link.qlim`
    #     """
    #     qlim = self.qlim
    #     if np.any(np.isnan(qlim)):
    #         raise ValueError("some joint limits not defined")
    #     return np.random.uniform(low=qlim[0, :], high=qlim[1, :], size=(self.n,))

    def addconfiguration_attr(self, name: str, q: ArrayLike, unit: str = "rad"):
        """
        Add a named joint configuration as an attribute (Robot superclass)

        :param name: Name of the joint configuration
        :param q: Joint configuration
        :type q: Arraylike

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.addconfiguration_attr("mypos", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            >>> robot.mypos
            >>> robot.configs["mypos"]

        .. note::
            - Used in robot model init method to store the ``qr`` configuration
            - Dynamically adding attributes to objects can cause issues with
              Python type checking.
            - Configuration is also added to the robot instance's dictionary of
              named configurations.

        :seealso: :meth:`addconfiguration`
        """
        v = getvector(q, self.n)
        v = getunit(v, unit)
        v = np.array(v)
        self._configs[name] = v
        setattr(self, name, v)

    def addconfiguration(self, name: str, q: np.ndarray):
        """
        Add a named joint configuration (Robot superclass)

        :param name: Name of the joint configuration
        :type name: str
        :param q: Joint configuration
        :type q: ndarray(n) or list

        Add a named configuration to the robot instance's dictionary of named
        configurations.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.addconfiguration_attr("mypos", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            >>> robot.configs["mypos"]

        :seealso: :meth:`addconfiguration`
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

    # TODO not very efficient
    # TODO keep a mapping from joint to link

    def linkcolormap(self, linkcolors="viridis"):
        """
        Create a colormap for robot joints

        :param linkcolors: list of colors or colormap, defaults to "viridis"
        :type linkcolors: list or str, optional
        :return: color map
        :rtype: matplotlib.colors.ListedColormap

        - ``cm = robot.linkcolormap()`` is an n-element colormap that gives a
          unique color for every link.  The RGBA colors for link ``j`` are
          ``cm(j)``.
        - ``cm = robot.linkcolormap(cmap)`` as above but ``cmap`` is the name
          of a valid matplotlib colormap.  The default, example above, is the
          ``viridis`` colormap.
        - ``cm = robot.linkcolormap(list of colors)`` as above but a
          colormap is created from a list of n color names given as strings,
          tuples or hexstrings.

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> cm = robot.linkcolormap("inferno")
            >>> print(cm(range(6))) # cm(i) is 3rd color in colormap
            >>> cm = robot.linkcolormap(
            >>>     ['red', 'g', (0,0.5,0), '#0f8040', 'yellow', 'cyan'])
            >>> print(cm(range(6)))

        .. note::

            - Colormaps have 4-elements: red, green, blue, alpha (RGBA)
            - Names of supported colors and colormaps are defined in the
              matplotlib documentation.

                - `Specifying colors
                <https://matplotlib.org/3.1.0/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py>`_
                - `Colormaps
                <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py>`_
        """  # noqa

        if isinstance(linkcolors, list) and len(linkcolors) == self.n:
            # provided a list of color names
            return colors.ListedColormap(linkcolors)
        else:
            # assume it is a colormap name
            return cm.get_cmap(linkcolors, 6)

    def jtraj(self, T1, T2, t, **kwargs):
        """
        Joint-space trajectory between SE(3) poses

        :param T1: initial end-effector pose
        :type T1: SE3 instance
        :param T2: final end-effector pose
        :type T2: SE3 instance
        :param t: time vector or number of steps
        :type t: ndarray(m) or int
        :param kwargs: arguments passed to the IK solver
        :return: trajectory
        :rtype: Trajectory instance

        ``traj = obot.jtraj(T1, T2, t)`` is a trajectory object whose
        attribute ``traj.q`` is a row-wise joint-space trajectory.

        The initial and final poses are mapped to joint space using inverse
        kinematics:

        - if the object has an analytic solution ``ikine_a`` that will be used,
        - otherwise the general numerical algorithm ``ikine_min`` will be used.


        """

        if hasattr(self, "ikine_a"):
            ik = self.ikine_a
        else:
            ik = self.ikine_min

        q1 = ik(T1, **kwargs)
        q2 = ik(T2, **kwargs)

        return rtb.jtraj(q1.q, q2.q, t)

    def manipulability(self, q=None, J=None, method="yoshikawa", axes="all", **kwargs):
        """
        Manipulability measure

        :param q: Joint coordinates, one of J or q required
        :type q: ndarray(n), or ndarray(m,n)
        :param J: Jacobian in world frame if already computed, one of J or
            q required
        :type J: ndarray(6,n)
        :param method: method to use, "yoshikawa" (default), "condition",
            "minsingular"  or "asada"
        :type method: str
        :param axes: Task space axes to consider: "all" [default],
            "trans", "rot" or "both"
        :type axes: str
        :param kwargs: extra arguments to pass to ``jacob0``
        :return: manipulability
        :rtype: float or ndarray(m)

        - ``manipulability(q)`` is the scalar manipulability index
          for the robot at the joint configuration ``q``.  It indicates
          dexterity, that is, how well conditioned the robot is for motion
          with respect to the 6 degrees of Cartesian motion.  The values is
          zero if the robot is at a singularity.

        Various measures are supported:

        +-------------------+-------------------------------------------------+
        | Measure           |       Description                               |
        +-------------------+-------------------------------------------------+
        | ``"yoshikawa"``   | Volume of the velocity ellipsoid, *distance*    |
        |                   | from singularity [Yoshikawa85]_                 |
        +-------------------+-------------------------------------------------+
        | ``"invcondition"``| Inverse condition number of Jacobian, isotropy  |
        |                   | of the velocity ellipsoid [Klein87]_            |
        +-------------------+-------------------------------------------------+
        | ``"minsingular"`` | Minimum singular value of the Jacobian,         |
        |                   | *distance*  from singularity [Klein87]_         |
        +-------------------+-------------------------------------------------+
        | ``"asada"``       | Isotropy of the task-space acceleration         |
        |                   | ellipsoid which is a function of the Cartesian  |
        |                   | inertia matrix which depends on the inertial    |
        |                   | parameters [Asada83]_                           |
        +-------------------+-------------------------------------------------+

        **Trajectory operation**:

        If ``q`` is a matrix (m,n) then the result (m,) is a vector of
        manipulability indices for each joint configuration specified by a row
        of ``q``.

        .. note::

            - Invokes the ``jacob0`` method of the robot if ``J`` is not passed
            - The "all" option includes rotational and translational
              dexterity, but this involves adding different units. It can be
              more useful to look at the translational and rotational
              manipulability separately.
            - Examples in the RVC book (1st edition) can be replicated by
              using the "all" option
            - Asada's measure requires inertial a robot model with inertial
              parameters.

        :references:

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

        """
        if isinstance(axes, list) and len(axes) == 6:
            pass
        elif axes == "all":
            axes = [True, True, True, True, True, True]
        elif axes.startswith("trans"):
            axes = [True, True, True, False, False, False]
        elif axes.startswith("rot"):
            axes = [False, False, False, True, True, True]
        elif axes == "both":
            return (
                self.manipulability(q, J, method, axes="trans", **kwargs),
                self.manipulability(q, J, method, axes="rot", **kwargs),
            )
        else:
            raise ValueError("axes must be all, trans, rot or both")

        def yoshikawa(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            if J.shape[0] == J.shape[1]:
                # simplified case for square matrix
                return abs(np.linalg.det(J))
            else:
                m2 = np.linalg.det(J @ J.T)
                return np.sqrt(abs(m2))

        def condition(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            return 1 / np.linalg.cond(J)  # return 1/cond(J)

        def minsingular(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            s = np.linalg.svd(J, compute_uv=False)
            return s[-1]  # return last/smallest singular value of J

        def asada(robot, J, q, axes, **kwargs):
            # dof = np.sum(axes)
            if np.linalg.matrix_rank(J) < 6:
                return 0
            Ji = np.linalg.pinv(J)
            Mx = Ji.T @ robot.inertia(q) @ Ji
            d = np.where(axes)[0]
            Mx = Mx[d]
            Mx = Mx[:, d.tolist()]
            e, _ = np.linalg.eig(Mx)
            return np.min(e) / np.max(e)

        # choose the handler function
        if method == "yoshikawa":
            mfunc = yoshikawa
        elif method == "invcondition":
            mfunc = condition
        elif method == "minsingular":
            mfunc = minsingular
        elif method == "asada":
            mfunc = asada
        else:
            raise ValueError("Invalid method chosen")

        # Calculate manipulability based on supplied Jacobian
        if J is not None:
            w = [mfunc(self, J, q, axes)]

        # Otherwise use the q vector/matrix
        else:
            q = getmatrix(q, (None, self.n))
            w = np.zeros(q.shape[0])

            for k, qk in enumerate(q):
                Jk = self.jacob0(qk, **kwargs)
                w[k] = mfunc(self, Jk, qk, axes)

        if len(w) == 1:
            return w[0]
        else:
            return w

    def jacob0_dot(self, q=None, qd=None, J0=None, representation=None):
        r"""
        Derivative of Jacobian

        :param q: The joint configuration of the robot
        :type q: float ndarray(n)
        :param qd: The joint velocity of the robot
        :type qd: ndarray(n)
        :param J0: Jacobian in {0} frame
        :type J0: ndarray(6,n)
        :param representation: angular representation
        :type representation: str
        :return: The derivative of the manipulator Jacobian
        :rtype:  ndarray(6,n)

        ``robot.jacob_dot(q, qd)`` computes the rate of change of the
        Jacobian elements

        .. math::

            \dmat{J} = \frac{d \mat{J}}{d \vec{q}} \frac{d \vec{q}}{dt}

        where the first term is the rank-3 Hessian.

         If ``J0`` is already calculated for the joint
        coordinates ``q`` it can be passed in to to save computation time.

        It is computed as the mode-3 product of the Hessian tensor and the
        velocity vector.

        The derivative of an analytical Jacobian can be obtained by setting
        ``representation`` as

        ==================   ==================================
        ``representation``          Rotational representation
        ==================   ==================================
        ``'rpy/xyz'``        RPY angular rates in XYZ order
        ``'rpy/zyx'``        RPY angular rates in XYZ order
        ``'eul'``            Euler angular rates in ZYZ order
        ``'exp'``            exponential coordinate rates
        ==================   ==================================

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke

        :seealso: :func:`jacob0`, :func:`hessian0`
        """  # noqa
        # n = len(q)

        # J = r.jacob0(q)

        # H = r.hessian0(q)

        # ev = J @ qd
        # ew = ev[3:]

        # Γd = sm.smb.rotvelxform(T.R, inverse=True, representation=rep) @ ew

        if representation is None:

            if J0 is None:
                J0 = self.jacob0(q)
            H = self.hessian0(q, J0=J0)

        else:
            # determine analytic rotation
            T = self.fkine(q).A
            gamma = smb.r2x(smb.t2r(T), representation=representation)

            # get transformation angular velocity to analytic velocity
            Ai = smb.rotvelxform(
                gamma, representation=representation, inverse=True, full=True
            )

            # get analytic rate from joint rates
            omega = J0[3:, :] @ qd
            gamma_dot = Ai[3:, 3:] @ omega
            Ai_dot = smb.rotvelxform_inv_dot(gamma, gamma_dot, full=True)
            Ai_dot = sp.linalg.block_diag(np.zeros((3, 3)), Ai_dot)

            Jd = Ai_dot @ J0 + Ai @ Jd

            # not actually sure this can be written in closed form

            # H = smb.numhess(
            #     lambda q: self.jacob0_analytical(q, representation=representation), q
            # )
            Jd = Ai @ Jd
            return Jd

        return np.tensordot(H, qd, (0, 0))

    def jacobm(self, q=None, J=None, H=None, end=None, start=None, axes="all"):
        r"""
        Calculates the manipulability Jacobian. This measure relates the rate
        of change of the manipulability to the joint velocities of the robot.
        One of J or q is required. Supply J and H if already calculated to
        save computation time

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param J: The manipulator Jacobian in any frame
        :type J: float ndarray(6,n)
        :param H: The manipulator Hessian in any frame
        :type H: float ndarray(6,n,n)
        :param end: the final link or Gripper which the Hessian represents
        :type end: str or ELink or Gripper
        :param start: the first link which the Hessian represents
        :type start: str or ELink

        :return: The manipulability Jacobian
        :rtype: float ndarray(n)

        Yoshikawa's manipulability measure

        .. math::

            m(\vec{q}) = \sqrt{\mat{J}(\vec{q}) \mat{J}(\vec{q})^T}

        This method returns its Jacobian with respect to configuration

        .. math::

            \frac{\partial m(\vec{q})}{\partial \vec{q}}

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        end, start, _ = self._get_limit_links(end, start)
        # path, n, _ = self.get_path(end, start)

        if axes == "all":
            axes = [True, True, True, True, True, True]
        elif axes.startswith("trans"):
            axes = [True, True, True, False, False, False]
        elif axes.startswith("rot"):
            axes = [False, False, False, True, True, True]
        else:
            raise ValueError("axes must be all, trans or rot")

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, start=start, end=end)
        else:
            verifymatrix(J, (6, self.n))

        n = J.shape[1]

        if H is None:
            H = self.hessian0(J0=J, start=start, end=end)
        else:
            verifymatrix(H, (6, self.n, self.n))

        manipulability = self.manipulability(q, J=J, start=start, end=end, axes=axes)

        J = J[axes, :]
        H = H[:, axes, :]

        b = np.linalg.inv(J @ np.transpose(J))
        Jm = np.zeros((n, 1))

        for i in range(n):
            c = J @ np.transpose(H[i, :, :])
            Jm[i, 0] = manipulability * np.transpose(c.flatten("F")) @ b.flatten("F")

        return Jm

    @abstractmethod
    def ets(self, *args, **kwargs) -> ETS:
        pass

    def jacob0_analytical(
        self,
        q: ArrayLike,
        representation: str = "rpy/xyz",
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        tool: Union[ndarray, SE3, None] = None,
    ):
        r"""
        Manipulator analytical Jacobian in the ``start`` frame

        :param q: Joint coordinate vector
        :type q: Arraylike
        :param representation: angular representation
        :type representation: str
        :param end: the particular link or gripper whose velocity the Jacobian
            describes, defaults to the base link
        :param start: the link considered as the end-effector, defaults to the robots's end-effector
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None

        :return J: Manipulator Jacobian in the ``start`` frame

        - ``robot.jacob0_analytical(q)`` is the manipulator Jacobian matrix which maps
          joint  velocity to end-effector spatial velocity expressed in the
          ``start`` frame.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        ==================   ==================================
        ``representation``          Rotational representation
        ==================   ==================================
        ``'rpy/xyz'``        RPY angular rates in XYZ order
        ``'rpy/zyx'``        RPY angular rates in XYZ order
        ``'eul'``            Euler angular rates in ZYZ order
        ``'exp'``            exponential coordinate rates
        ==================   ==================================

        Example:
        .. runblock:: pycon
            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.ETS.Puma560()
            >>> puma.jacob0_analytical([0, 0, 0, 0, 0, 0])

        .. warning:: ``start`` and ``end`` must be on the same branch,
            with ``start`` closest to the base.
        """  # noqa
        return self.ets(start, end).jacob0_analytical(
            q, tool=tool, representation=representation
        )

    # --------------------------------------------------------------------- #

    @property
    def control_mode(self):
        """
        Get/set robot control mode (Robot superclass)

        - ``robot.control_type`` is the robot control mode

        :return: robot control mode
        :rtype: ndarray(n,)

        - ``robot.control_type = ...`` checks and sets the robot control mode

        .. note::  ???
        """
        return self._control_mode

    @control_mode.setter
    def control_mode(self, cn):
        if cn == "p" or cn == "v" or cn == "a":
            self._control_mode = cn
        else:
            raise ValueError("Control type must be one of 'p', 'v', or 'a'")

    # --------------------------------------------------------------------- #

    # TODO probably should be a static method
    def _get_graphical_backend(self, backend=None):

        default = self.default_backend

        # figure out the right default
        if backend is None:
            if isinstance(self, rtb.DHRobot):
                default = "pyplot"
            elif isinstance(self, rtb.ERobot2):
                default = "pyplot2"
            elif isinstance(self, rtb.ERobot):
                if self.hasgeometry:
                    default = "swift"
                else:
                    default = "pyplot"

        if backend is not None:
            backend = backend.lower()

        # find the right backend, modules are imported here on an as needs
        # basis
        if backend == "swift" or default == "swift":  # pragma nocover
            # swift was requested, is it installed?
            if isinstance(self, rtb.DHRobot):
                raise NotImplementedError(
                    "Plotting in Swift is not implemented for DHRobots yet"
                )
            try:
                # yes, use it
                from roboticstoolbox.backends.swift import Swift

                env = Swift()
                return env
            except ModuleNotFoundError:
                if backend == "swift":
                    print("Swift is not installed, " "install it using pip or conda")
                backend = "pyplot"

        elif backend == "vpython" or default == "vpython":  # pragma nocover
            # vpython was requested, is it installed?
            if not isinstance(self, rtb.DHRobot):
                raise NotImplementedError(
                    "Plotting in VPython is only implemented for DHRobots"
                )
            try:
                # yes, use it
                from roboticstoolbox.backends.VPython import VPython

                env = VPython()
                return env
            except ModuleNotFoundError:
                if backend == "vpython":
                    print("VPython is not installed, " "install it using pip or conda")
                backend = "pyplot"
        if backend is None:
            backend = default

        if backend == "pyplot":
            from roboticstoolbox.backends.PyPlot import PyPlot

            env = PyPlot()

        elif backend == "pyplot2":
            from roboticstoolbox.backends.PyPlot import PyPlot2

            env = PyPlot2()

        else:
            raise ValueError("unknown backend", backend)

        return env

    def plot(
        self,
        q,
        backend=None,
        block=False,
        dt=0.050,
        limits=None,
        vellipse=False,
        fellipse=False,
        fig=None,
        movie=None,
        loop=False,
        **kwargs,
    ):
        """
        Graphical display and animation

        :param q: The joint configuration of the robot.
        :type q: float ndarray(n)
        :param backend: The graphical backend to use, currently 'swift'
            and 'pyplot' are implemented. Defaults to 'swift' of an ``ERobot``
            and 'pyplot` for a ``DHRobot``
        :type backend: string
        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param dt: if q is a trajectory, this describes the delay in
            seconds between frames
        :type dt: float
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
            (this option is for 'pyplot' only)
        :type limits: ndarray(6)
        :param vellipse: (Plot Option) Plot the velocity ellipse at the
            end-effector (this option is for 'pyplot' only)
        :type vellipse: bool
        :param vellipse: (Plot Option) Plot the force ellipse at the
            end-effector (this option is for 'pyplot' only)
        :type vellipse: bool
        :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
            which the joint revolves around(revolute joint) or translates
            along (prosmatic joint) (this option is for 'pyplot' only)
        :type jointaxes: bool
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
            (this option is for 'pyplot' only)
        :type eeframe: bool
        :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
            plane. (this option is for 'pyplot' only)
        :type shadow: bool
        :param name: (Plot Option) Plot the name of the robot near its base
            (this option is for 'pyplot' only)
        :type name: bool
        :param movie: name of file in which to save an animated GIF
            (this option is for 'pyplot' only)
        :type movie: str

        :return: A reference to the environment object which controls the
            figure
        :rtype: Swift or PyPlot

        - ``robot.plot(q, 'pyplot')`` displays a graphical view of a robot
          based on the kinematic model and the joint configuration ``q``.
          This is a stick figure polyline which joins the origins of the
          link coordinate frames. The plot will autoscale with an aspect
          ratio of 1.

        If ``q`` (m,n) representing a joint-space trajectory it will create an
        animation with a pause of ``dt`` seconds between each frame.

        .. note::
            - By default this method will block until the figure is dismissed.
              To avoid this set ``block=False``.
            - For PyPlot, the polyline joins the origins of the link frames,
              but for some Denavit-Hartenberg models those frames may not
              actually be on the robot, ie. the lines to not neccessarily
              represent the links of the robot.

        :seealso: :func:`teach`
        """

        env = None

        env = self._get_graphical_backend(backend)

        q = getmatrix(q, (None, self.n))
        self.q = q[0, :]

        # Add the self to the figure in readonly mode
        # Add the self to the figure in readonly mode
        if q.shape[0] == 1:
            env.launch(self.name + " Plot", limits=limits, fig=fig)
        else:
            env.launch(self.name + " Trajectory Plot", limits=limits, fig=fig)

        env.add(self, readonly=True, **kwargs)

        if vellipse:
            vell = self.vellipse(centre="ee")
            env.add(vell)

        if fellipse:
            fell = self.fellipse(centre="ee")
            env.add(fell)

        # Stop lint error
        images = []  # list of images saved from each plot

        if movie is not None:
            loop = False

        while True:
            for qk in q:
                self.q = qk
                if vellipse:
                    vell.q = qk
                if fellipse:
                    fell.q = qk
                env.step(dt)

                if movie is not None:  # pragma nocover
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

        # Keep the plot open
        if block:  # pragma: no cover
            env.hold()

        return env

    # --------------------------------------------------------------------- #

    def fellipse(self, q=None, opt="trans", unit="rad", centre=[0, 0, 0]):
        """
        Create a force ellipsoid object for plotting with PyPlot

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param opt: 'trans' or 'rot' will plot either the translational or
            rotational force ellipsoid
        :type opt: string
        :param centre:
        :type centre: list or str('ee')

        :return: An EllipsePlot object
        :rtype: EllipsePlot

        - ``robot.fellipse(q)`` creates a force ellipsoid for the robot at
          pose ``q``. The ellipsoid is centered at the origin.

        - ``robot.fellipse()`` as above except the joint configuration is that
          stored in the robot object.

        .. note::
            - By default the ellipsoid related to translational motion is
              drawn.  Use ``opt='rot'`` to draw the rotational velocity
              ellipsoid.
            - By default the ellipsoid is drawn at the origin.  The option
              ``centre`` allows its origin to set to set to the specified
              3-vector, or the string "ee" ensures it is drawn at the
              end-effector position.

        """
        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError("ERobot fellipse not implemented yet")

        q = getunit(q, unit)
        ell = EllipsePlot(self, q, "f", opt, centre=centre)
        return ell

    def vellipse(self, q=None, opt="trans", unit="rad", centre=[0, 0, 0], scale=0.1):
        """
        Create a velocity ellipsoid object for plotting with PyPlot

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param opt: 'trans' or 'rot' will plot either the translational or
            rotational velocity ellipsoid
        :type opt: string
        :param centre:
        :type centre: list or str('ee')

        :return: An EllipsePlot object
        :rtype: EllipsePlot

        - ``robot.vellipse(q)`` creates a force ellipsoid for the robot at
          pose ``q``. The ellipsoid is centered at the origin.

        - ``robot.vellipse()`` as above except the joint configuration is that
          stored in the robot object.

        .. note::
            - By default the ellipsoid related to translational motion is
              drawn.  Use ``opt='rot'`` to draw the rotational velocity
              ellipsoid.
            - By default the ellipsoid is drawn at the origin.  The option
              ``centre`` allows its origin to set to set to the specified
              3-vector, or the string "ee" ensures it is drawn at the
              end-effector position.
        """
        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError("ERobot vellipse not implemented yet")

        q = getunit(q, unit)
        ell = EllipsePlot(self, q, "v", opt, centre=centre, scale=scale)
        return ell

    def plot_ellipse(
        self,
        ellipse,
        block=True,
        limits=None,
        jointaxes=True,
        eeframe=True,
        shadow=True,
        name=True,
    ):
        """
        Plot the an ellipsoid

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param ellipse: the ellipsoid to plot
        :type ellipse: EllipsePlot
        :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
            which the joint revolves around(revolute joint) or translates
            along (prosmatic joint)
        :type jointaxes: bool
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        :type eeframe: bool
        :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
            plane
        :type shadow: bool
        :param name: (Plot Option) Plot the name of the robot near its base
        :type name: bool

        :return: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        - ``robot.plot_ellipse(ellipsoid)`` displays the ellipsoid.

        .. note::
            - By default the ellipsoid is drawn at the origin.  The option
              ``centre`` allows its origin to set to set to the specified
              3-vector, or the string "ee" ensures it is drawn at the
              end-effector position.
        """

        if not isinstance(ellipse, EllipsePlot):  # pragma nocover
            raise TypeError(
                "ellipse must be of type " "roboticstoolbox.backend.PyPlot.EllipsePlot"
            )

        env = PyPlot()

        # Add the robot to the figure in readonly mode
        env.launch(ellipse.robot.name + " " + ellipse.name, limits=limits)

        env.add(ellipse, jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

        # Keep the plot open
        if block:  # pragma: no cover
            env.hold()

        return env

    def plot_fellipse(
        self,
        q=None,
        block=True,
        fellipse=None,
        limits=None,
        opt="trans",
        centre=[0, 0, 0],
        jointaxes=True,
        eeframe=True,
        shadow=True,
        name=True,
    ):
        """
        Plot the force ellipsoid for manipulator

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param fellipse: the vellocity ellipsoid to plot
        :type fellipse: EllipsePlot
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        :type limits: ndarray(6)
        :param opt: 'trans' or 'rot' will plot either the translational or
            rotational force ellipsoid
        :type opt: string
        :param centre: The coordinates to plot the fellipse [x, y, z] or "ee"
            to plot at the end-effector location
        :type centre: array_like or str
        :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
            which the joint revolves around(revolute joint) or translates
            along (prosmatic joint)
        :type jointaxes: bool
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        :type eeframe: bool
        :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
            plane
        :type shadow: bool
        :param name: (Plot Option) Plot the name of the robot near its base
        :type name: bool

        :return: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        - ``robot.plot_fellipse(q)`` displays the velocity ellipsoid for the
          robot at pose ``q``. The plot will autoscale with an aspect ratio
          of 1.

        - ``plot_fellipse()`` as above except the robot is plotted with joint
          coordinates stored in the robot object.

        - ``robot.plot_fellipse(vellipse)`` specifies a custon ellipse to plot.

        .. note::
            - By default the ellipsoid related to translational motion is
              drawn.  Use ``opt='rot'`` to draw the rotational velocity
              ellipsoid.
            - By default the ellipsoid is drawn at the origin.  The option
              ``centre`` allows its origin to set to set to the specified
              3-vector, or the string "ee" ensures it is drawn at the
              end-effector position.
        """

        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "Ellipse Plotting of ERobot's not implemented yet"
            )

        if q is not None:
            self.q = q

        if fellipse is None:
            fellipse = self.fellipse(q=q, opt=opt, centre=centre)

        return self.plot_ellipse(
            fellipse,
            block=block,
            limits=limits,
            jointaxes=jointaxes,
            eeframe=eeframe,
            shadow=shadow,
            name=name,
        )

    def plot_vellipse(
        self,
        q=None,
        block=True,
        vellipse=None,
        limits=None,
        opt="trans",
        centre=[0, 0, 0],
        jointaxes=True,
        eeframe=True,
        shadow=True,
        name=True,
    ):
        """
        Plot the velocity ellipsoid for manipulator

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param vellipse: the vellocity ellipsoid to plot
        :type vellipse: EllipsePlot
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        :type limits: ndarray(6)
        :param opt: 'trans' or 'rot' will plot either the translational or
            rotational velocity ellipsoid
        :type opt: string
        :param centre: The coordinates to plot the vellipse [x, y, z] or "ee"
            to plot at the end-effector location
        :type centre: array_like or str
        :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
            which the joint revolves around(revolute joint) or translates
            along (prosmatic joint)
        :type jointaxes: bool
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        :type eeframe: bool
        :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
            plane
        :type shadow: bool
        :param name: (Plot Option) Plot the name of the robot near its base
        :type name: bool

        :return: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        - ``robot.plot_vellipse(q)`` displays the velocity ellipsoid for the
          robot at pose ``q``. The plot will autoscale with an aspect ratio
          of 1.

        - ``plot_vellipse()`` as above except the robot is plotted with joint
          coordinates stored in the robot object.

        - ``robot.plot_vellipse(vellipse)`` specifies a custon ellipse to plot.

        .. note::
            - By default the ellipsoid related to translational motion is
              drawn.  Use ``opt='rot'`` to draw the rotational velocity
              ellipsoid.
            - By default the ellipsoid is drawn at the origin.  The option
              ``centre`` allows its origin to set to set to the specified
              3-vector, or the string "ee" ensures it is drawn at the
              end-effector position.
        """

        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "Ellipse Plotting of ERobot's not implemented yet"
            )

        if q is not None:
            self.q = q

        if vellipse is None:
            vellipse = self.vellipse(q=q, opt=opt, centre=centre)

        return self.plot_ellipse(
            vellipse,
            block=block,
            limits=limits,
            jointaxes=jointaxes,
            eeframe=eeframe,
            shadow=shadow,
            name=name,
        )

    # --------------------------------------------------------------------- #

    def teach(
        self,
        q=None,
        block=True,
        order="xyz",
        limits=None,
        jointaxes=True,
        jointlabels=False,
        vellipse=False,
        fellipse=False,
        eeframe=True,
        shadow=True,
        name=True,
        backend=None,
    ):
        """
        Graphical teach pendant

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint configuration of the robot (Optional,
                  if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param limits: Custom view limits for the plot. If not supplied will
                       autoscale, [x1, x2, y1, y2, z1, z2]
        :type limits: ndarray(6)
        :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
                          which the joint revolves around(revolute joint) or
                          translates along (prismatic joint)
        :type jointaxes: bool
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        :type eeframe: bool
        :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
            plane
        :type shadow: bool
        :param name: (Plot Option) Plot the name of the robot near its base
        :type name: bool

        :return: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        - ``robot.teach(q)`` creates a matplotlib plot which allows the user to
          "drive" a graphical robot using a graphical slider panel. The robot's
          inital joint configuration is ``q``. The plot will autoscale with an
          aspect ratio of 1.

        - ``robot.teach()`` as above except the robot's stored value of ``q``
            is used.

        .. note::
            - Program execution is blocked until the teach window is
              dismissed.  If ``block=False`` the method is non-blocking but
              you need to poll the window manager to ensure that the window
              remains responsive.
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

        # Add the self to the figure in readonly mode
        env.launch("Teach " + self.name, limits=limits)
        env.add(
            self,
            readonly=True,
            # jointaxes=jointaxes,
            # jointlabels=jointlabels,
            # eeframe=eeframe,
            # shadow=shadow,
            # name=name,
        )

        env._add_teach_panel(self, q)

        if vellipse:
            vell = self.vellipse(centre="ee", scale=0.5)
            env.add(vell)

        if fellipse:
            fell = self.fellipse(centre="ee")
            env.add(fell)

        # Keep the plot open
        if block:  # pragma: no cover
            env.hold()

        return env

    # --------------------------------------------------------------------- #

    def closest_point(
        self, q: ArrayLike, shape: Shape, inf_dist: float = 1.0, skip: bool = False
    ) -> Tuple[Union[int, None], Union[np.ndarray, None], Union[np.ndarray, None],]:
        """
        closest_point(shape, inf_dist) returns the minimum euclidean
        distance between this robot and shape, provided it is less than
        inf_dist. It will also return the points on self and shape in the
        world frame which connect the line of length distance between the
        shapes. If the distance is negative then the shapes are collided.

        :param shape: The shape to compare distance to
        :param inf_dist: The minimum distance within which to consider
            the shape
        :param skip: Skip setting all shape transforms based on q, use this
            option if using this method in conjuction with Swift to save time

        :returns: d, p1, p2 where d is the distance between the shapes,
            p1 and p2 are the points in the world frame on the respective
            shapes. The points returned are [x, y, z].
        """

        if not skip:
            self._update_link_tf(q)
            self._propogate_scene_tree()
            shape._propogate_scene_tree()

        d = 10000
        p1 = None
        p2 = None

        for link in self.links:
            td, tp1, tp2 = link.closest_point(shape, inf_dist, skip=True)

            if td is not None and td < d:
                d = td
                p1 = tp1
                p2 = tp2

        if d == 10000:
            d = None

        return d, p1, p2

    def iscollided(self, q, shape, skip=False):
        """
        collided(shape) checks if this robot and shape have collided
        :param shape: The shape to compare distance to
        :type shape: Shape
        :param skip: Skip setting all shape transforms based on q, use this
            option if using this method in conjuction with Swift to save time
        :type skip: boolean
        :returns: True if shapes have collided
        :rtype: bool
        """

        if not skip:
            self._update_link_tf(q)
            self._propogate_scene_tree()
            shape._propogate_scene_tree()

        for link in self.links:
            if link.iscollided(shape, skip=True):
                return True

        if isinstance(self, rtb.ERobot):
            for gripper in self.grippers:
                for link in gripper.links:
                    if link.iscollided(shape, skip=True):
                        return True

        return False

    def collided(self, q, shape, skip=False):
        """
        collided(shape) checks if this robot and shape have collided
        :param shape: The shape to compare distance to
        :type shape: Shape
        :param skip: Skip setting all shape transforms based on q, use this
            option if using this method in conjuction with Swift to save time
        :type skip: boolean
        :returns: True if shapes have collided
        :rtype: bool
        """
        warn("method collided is deprecated, use iscollided instead", FutureWarning)
        return self.iscollided(q, shape, skip=skip)

    def joint_velocity_damper(self, ps=0.05, pi=0.1, n=None, gain=1.0):
        """
        Formulates an inequality contraint which, when optimised for will
        make it impossible for the robot to run into joint limits. Requires
        the joint limits of the robot to be specified. See examples/mmc.py
        for use case

        :param ps: The minimum angle (in radians) in which the joint is
            allowed to approach to its limit
        :type ps: float
        :param pi: The influence angle (in radians) in which the velocity
            damper becomes active
        :type pi: float
        :param n: The number of joints to consider. Defaults to all joints
        :type n: int
        :param gain: The gain for the velocity damper
        :type gain: float

        :returns: Ain, Bin as the inequality contraints for an optisator
        :rtype: ndarray(6), ndarray(6)
        """

        if n is None:
            n = self.n

        Ain = np.zeros((n, n))
        Bin = np.zeros(n)

        for i in range(n):
            if self.q[i] - self.qlim[0, i] <= pi:
                Bin[i] = -gain * (((self.qlim[0, i] - self.q[i]) + ps) / (pi - ps))
                Ain[i, i] = -1
            if self.qlim[1, i] - self.q[i] <= pi:
                Bin[i] = gain * ((self.qlim[1, i] - self.q[i]) - ps) / (pi - ps)
                Ain[i, i] = 1

        return Ain, Bin

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


if __name__ == "__main__":

    pass

    # import roboticstoolbox as rtb

    # puma = rtb.models.DH.Puma560()
    # a = puma.copy()

    # from roboticstoolbox import ET2 as ET

    # e = ET.R() * ET.tx(1) * ET.R() * ET.tx(1)
    # print(e)
    # r = Robot2(e)

    # print(r.fkine([0, 0]))
    # print(r.jacob0([0, 0]))

    # r.plot([0.7, 0.7])
