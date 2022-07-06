#!/usr/bin/env python3
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

from os.path import splitext
import tempfile
import subprocess
import webbrowser
from numpy import (
    array,
    ndarray,
    isnan,
    zeros,
    eye,
    expand_dims,
    empty,
    concatenate,
    cross,
    arccos,
    dot,
)
from numpy.linalg import norm as npnorm, inv
from spatialmath import SE3, SE2
from spatialgeometry import Cylinder
from spatialmath.base.argcheck import getvector, islistof
from roboticstoolbox.robot.Link import Link, Link2, BaseLink
from roboticstoolbox.robot.ETS import ETS, ETS2
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.DHRobot import DHRobot
from roboticstoolbox.tools import xacro
from roboticstoolbox.tools import URDF
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.robot.Gripper import Gripper
from roboticstoolbox.tools.data import rtb_path_to_datafile
from roboticstoolbox.tools.params import rtb_get_param
from pathlib import PurePosixPath
from ansitable import ANSITable, Column
from spatialmath import (
    SpatialAcceleration,
    SpatialVelocity,
    SpatialInertia,
    SpatialForce,
)
from functools import lru_cache
from typing import Union, overload, Dict, List, Tuple, Optional
from copy import deepcopy

ArrayLike = Union[list, ndarray, tuple, set]


class BaseERobot(Robot):

    """
    Construct an ERobot object
    :param et_list: List of elementary transforms which represent the robot
        kinematics
    :type et_list: ET list
    :param name: Name of the robot
    :type name: str, optional
    :param manufacturer: Manufacturer of the robot
    :type manufacturer: str, optional
    :param base: Location of the base is the world frame
    :type base: SE3, optional
    :param tool: Offset of the flange of the robot to the end-effector
    :type tool: SE3, optional
    :param gravity: The gravity vector
    :type n: ndarray(3)
    An ERobot represents the kinematics of a serial-link manipulator with
    one or more branches.
    From ETS
    --------
    Example:
    .. runblock:: pycon
        >>> from roboticstoolbox import ETS, ERobot
        >>> ets = ETS.rz() * ETS.ry() * ETS.tz(1) * ETS.ry() * ETS.tz(1)
        >>> robot = ERobot(ets)
        >>> print(robot)
    The ETS is partitioned such that a new link frame is created **after** every
    joint variable.
    From list of Links
    -------------------
    Example:
    .. runblock:: pycon
        >>> from roboticstoolbox import ETS, ERobot
        >>> link1 = Link(ETS.rz(), name='link1')
        >>> link2 = Link(ETS.ry(), name='link2', parent=link1)
        >>> link3 = Link(ETS.tz(1) * ETS.ry(), name='link3', parent=link2)
        >>> link4 = Link(ETS.tz(1), name='ee', parent=link3)
        >>> robot = ERobot([link1, link2, link3, link4])
        >>> print(robot)
    A number of ``Link`` objects are created, each has a transform with
    respect to the previous frame, and all except the first link have a parent.
    The implicit parent of the first link is the base.
    The parent also can be specified as a string, and its name is mapped to the
    parent link by name in ``ERobot``.
    If no ``parent`` arguments are given it is assumed the links are in
    sequential order, and the parent hierarchy will be automatically
    established.
    .. runblock:: pycon
        >>> from roboticstoolbox import ETS, ERobot
        >>> robot = ERobot([
        >>>     Link(ETS.rz(), name='link1'),
        >>>     Link(ETS.ry(), name='link2'),
        >>>     Link(ETS.tz(1) * ETS.ry(), name='link3'),
        >>>     Link(ETS.tz(1), name='ee')
        >>>             ])
        >>> print(robot)
    Branched robots
    ---------------
    Example:
    .. runblock:: pycon
        >>> robot = ERobot([
        >>>    Link(ETS.rz(), name='link1'),
        >>>    Link(ETS.tx(1) * ETS.ty(-0.5) * ETS.rz(), name='link2', parent='link1'),
        >>>    Link(ETS.tx(1), name='ee_1', parent='link2'),
        >>>    Link(ETS.tx(1) * ETS.ty(0.5) * ETS.rz(), name='link3', parent='link1'),
        >>>    Link(ETS.tx(1), name='ee_2', parent='link3')
        >>>             ])
        >>> print(robot)
    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke
    """  # noqa E501

    def __init__(self, links, gripper_links=None, checkjindex=True, **kwargs):
        self._path_cache_fknm = {}
        self._path_cache = {}
        self._eye_fknm = eye(4)

        self._linkdict = {}
        self._n = 0
        self._ee_links = []

        # Ordered links, we reorder the input elinks to be in depth first
        # search order
        orlinks = []

        # check all the incoming Link objects
        n = 0
        for k, link in enumerate(links):
            # if link has no name, give it one
            if link.name is None or link.name == "":
                link.name = f"link-{k}"
            link.number = k + 1

            # put it in the link dictionary, check for duplicates
            if link.name in self._linkdict:
                raise ValueError(f"link name {link.name} is not unique")
            self._linkdict[link.name] = link

            if link.isjoint:
                n += 1

        # resolve parents given by name, within the context of
        # this set of links
        for link in links:
            if link.parent is None and link.parent_name is not None:
                link._parent = self._linkdict[link.parent_name]

        if all([link.parent is None for link in links]):
            # no parent links were given, assume they are sequential
            for i in range(len(links) - 1):
                links[i + 1]._parent = links[i]

        self._n = n

        # scan for base
        for link in links:
            # is this a base link?
            if link._parent is None:
                try:
                    if self._base_link is not None:
                        raise ValueError("Multiple base links")
                except AttributeError:
                    pass

                self._base_link = link
            else:
                # no, update children of this link's parent
                link._parent._children.append(link)

        if self.base_link is None:  # Pragma: nocover
            raise ValueError(
                "Invalid link configuration provided, must have a base link"
            )

        # Scene node, set links between the links
        for link in links:
            if link.parent is not None:
                link.scene_parent = link.parent

        # Set up the gripper, make a list containing the root of all
        # grippers
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
            self._n -= gripper.n

        # Set the ee links
        self.ee_links = []
        if len(gripper_links) == 0:
            for link in links:
                # is this a leaf node? and do we not have any grippers
                if len(link.children) == 0:
                    # no children, must be an end-effector
                    self.ee_links.append(link)
        else:
            for link in gripper_links:
                # use the passed in value
                self.ee_links.append(link.parent)  # type: ignore

        # assign the joint indices
        if all([link.jindex is None or link.ets._auto_jindex for link in links]):
            # no joints have an index
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
            # jindex set on all, check they are unique and contiguous
            if checkjindex:
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

        # self._nbranches = sum([link.nchildren == 0 for link in links])

        # Set up qlim
        qlim = zeros((2, self.n))
        j = 0

        for i in range(len(orlinks)):
            if orlinks[i].isjoint:
                qlim[:, j] = orlinks[i].qlim
                j += 1
        self._qlim = qlim

        self._valid_qlim = False
        for i in range(self.n):
            if any(qlim[:, i] != 0) and not any(isnan(qlim[:, i])):
                self._valid_qlim = True

        # Initialise Robot object
        super().__init__(orlinks, **kwargs)

        # Fix number of links for gripper links
        self._nlinks = len(links)

        for gripper in self.grippers:
            self._nlinks += len(gripper.links)

        # SceneNode, set a reference to the first link
        self.scene_children = [self.links[0]]  # type: ignore

    def __str__(self) -> str:
        """
        Pretty prints the ETS Model of the robot.
        :return: Pretty print of the robot model
        :rtype: str
        .. note::
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

        if isinstance(self, ERobot2):
            classname = "ERobot2"
        else:
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

    @overload
    def __getitem__(self: "ERobot", i: Union[int, str]) -> Link:
        ...

    @overload
    def __getitem__(self: "ERobot", i: slice) -> List[Link]:
        ...

    @overload
    def __getitem__(self: "ERobot2", i: Union[int, str]) -> Link2:
        ...

    @overload
    def __getitem__(self: "ERobot2", i: slice) -> List[Link2]:
        ...

    def __getitem__(self, i):
        """
        Get link

        :param i: link number or name
        :type i: int, slice or str
        :return: i'th link or named link
        :rtype: Link

        This also supports iterating over each link in the robot object,
        from the base to the tool.

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.Panda()
            >>> print(robot[1]) # print the 2nd link
            >>> print([link.a for link in robot])  # print all the a_j values

        .. note:: ``ERobot`` supports link lookup by name,
            eg. ``robot['link1']``
        """
        if isinstance(i, str):
            try:
                return self.link_dict[i]
            except KeyError:
                raise KeyError(f"link {i} not in link dictionary")
            except AttributeError:
                raise AttributeError(f"robot has no link dictionary")
        else:
            return self._links[i]

    # --------------------------------------------------------------------- #

    @overload
    def links(self: "ERobot") -> List[Link]:
        ...

    @overload
    def links(self: "ERobot2") -> List[Link2]:
        ...

    @property
    def links(self) -> List[Link]:
        """
        Robot links

        :return: A list of link objects
        """
        return self._links

    # --------------------------------------------------------------------- #

    @property
    def n(self) -> int:
        """
        Number of joints
        :return: number of variable joint in the robot's kinematic tree
        :rtype: int
        The sum of the number of revolute and prismatic joints.
        """
        return self._n

    # --------------------------------------------------------------------- #

    @property
    def grippers(self) -> List[Gripper]:
        """
        Grippers attached to the robot

        :return: A list of grippers

        """
        return self._grippers

    # --------------------------------------------------------------------- #
    @property
    def nbranches(self) -> int:
        """
        Number of branches

        :return: number of branches in the robot's kinematic tree
        :rtype: int

        Number of branches in this robot.  Computed as the number of links with
        zero children

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.ETS.Panda()
            >>> robot.nbranches

        :seealso: :func:`n`, :func:`nlinks`
        """
        # return self._nbranches
        return sum([link.nchildren == 0 for link in self.links]) + len(self.grippers)

    # --------------------------------------------------------------------- #

    @overload
    def elinks(self: "ERobot") -> List[Link]:
        ...

    @overload
    def elinks(self: "ERobot2") -> List[Link2]:
        ...

    @property
    def elinks(self) -> List[Link]:
        return self._links

    # --------------------------------------------------------------------- #

    @overload
    def link_dict(self: "ERobot") -> Dict[str, Link]:
        ...

    @overload
    def link_dict(self: "ERobot2") -> Dict[str, Link2]:
        ...

    @property
    def link_dict(self) -> Dict[str, Link]:
        return self._linkdict

    # --------------------------------------------------------------------- #

    @overload
    def base_link(self: "ERobot") -> Link:
        ...

    @overload
    def base_link(self: "ERobot2") -> Link2:
        ...

    @property
    def base_link(self) -> Link:
        return self._base_link

    @base_link.setter
    def base_link(self, link):
        if isinstance(link, Link):
            self._base_link = link
        else:
            raise TypeError("Must be a Link")

    # --------------------------------------------------------------------- #

    @overload
    def ee_links(self: "ERobot2") -> List[Link2]:
        ...

    @overload
    def ee_links(self: "ERobot") -> List[Link]:
        ...

    @property
    def ee_links(self) -> List[Link]:
        return self._ee_links

    @ee_links.setter
    def ee_links(self, link: Union[List[Link], Link]):
        if isinstance(link, Link):
            self._ee_links = [link]
        elif isinstance(link, list) and all([isinstance(x, Link) for x in link]):
            self._ee_links = link
        else:
            raise TypeError("expecting a Link or list of Links")

    # --------------------------------------------------------------------- #

    @property
    def reach(self) -> float:
        r"""
        Reach of the robot
        :return: Maximum reach of the robot
        :rtype: float
        A conservative estimate of the reach of the robot. It is computed as
        the sum of the translational ETs that define the link transform.
        .. note::
            - Probably an overestimate of reach
            - Used by numerical inverse kinematics to scale translational
              error.
            - For a prismatic joint, uses ``qlim`` if it is set
        .. warning:: Computed on the first access. If kinematic parameters
              subsequently change this will not be reflected.
        """
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

    # --------------------------------------------------------------------- #

    def hierarchy(self):
        """
        Pretty print the robot link hierachy
        :return: Pretty print of the robot model
        :rtype: str
        Example:
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

    # --------------------------------------------------------------------- #

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
        # return gripper.links[0].ets * ET.SE3(gripper.tool)
        return ETS(ET.SE3(gripper.tool))

    @lru_cache(maxsize=32)
    def ets(
        self,
        start: Union[Link, Gripper, str, None] = None,
        end: Union[Link, Gripper, str, None] = None,
    ) -> ETS:
        """
        ERobot to ETS

        :param start: start of path, defaults to ``base_link``
        :type start: Link or str, optional
        :param end: end of path, defaults to end-effector
        :type end: Link or str, optional
        :raises ValueError: a link does not belong to this ERobot
        :raises TypeError: a bad link argument
        :return: elementary transform sequence
        :rtype: ETS instance


        - ``robot.ets()`` is an ETS representing the kinematics from base to
          end-effector.
        - ``robot.ets(end=link)`` is an ETS representing the kinematics from
          base to the link ``link`` specified as a Link reference or a name.
        - ``robot.ets(start=l1, end=l2)`` is an ETS representing the kinematics
          from link ``l1`` to link ``l2``.

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

    # --------------------------------------------------------------------- #

    def segments(self) -> List[List[Union[Link, None]]]:
        """
        Segments of branched robot

        :return: Segment list
        :rtype: list of lists of Link

        For a single-chain robot with structure::

            L1 - L2 - L3

        the return is ``[[None, L1, L2, L3]]``

        For a robot with structure::

            L1 - L2 +-  L3 - L4
                    +- L5 - L6

        the return is ``[[None, L1, L2], [L2, L3, L4], [L2, L5, L6]]``

        .. note::
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

    def fkine_all(self, q):
        """
        Compute the pose of every link frame

        :param q: The joint configuration
        :type q:  darray(n)
        :return: Pose of all links
        :rtype: SE3 instance

        ``T = robot.fkine_all(q)`` is  an SE3 instance with ``robot.nlinks +
        1`` values:

        - ``T[0]`` is the base transform
        - ``T[i]`` is the pose of link whose ``number`` is ``i``

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """
        q = getvector(q)

        if isinstance(self, ERobot):
            Tbase = SE3(self.base)  # add base, also sets the type
        else:
            Tbase = SE2(self.base)  # add base, also sets the type

        linkframes = Tbase.__class__.Alloc(self.nlinks + 1)
        linkframes[0] = Tbase

        def recurse(Tall, Tparent, q, link):
            # if joint??
            T = Tparent
            while True:
                if isinstance(self, ERobot):
                    T *= SE3(link.A(q[link.jindex]))
                else:
                    T *= SE2(link.A(q[link.jindex]))

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

    # --------------------------------------------------------------------- #

    def showgraph(self, **kwargs):
        """
        Display a link transform graph in browser
        :param etsbox: Put the link ETS in a box, otherwise an edge label
        :type etsbox: bool
        :param jtype: Arrowhead to node indicates revolute or prismatic type
        :type jtype: bool
        :param static: Show static joints in blue and bold
        :type static: bool
        ``robot.showgraph()`` displays a graph of the robot's link frames
        and the ETS between them.  It uses GraphViz dot.
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
        Example::
            >>> import roboticstoolbox as rtb
            >>> panda = rtb.models.URDF.Panda()
            >>> panda.showgraph()
        .. image:: ../figs/panda-graph.svg
            :width: 600
        :seealso: :func:`dotfile`
        """

        # create the temporary dotfile
        dotfile = tempfile.TemporaryFile(mode="w")
        self.dotfile(dotfile, **kwargs)

        # rewind the dot file, create PDF file in the filesystem, run dot
        dotfile.seek(0)
        pdffile = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        subprocess.run("dot -Tpdf", shell=True, stdin=dotfile, stdout=pdffile)

        # open the PDF file in browser (hopefully portable), then cleanup
        webbrowser.open(f"file://{pdffile.name}")

    # --------------------------------------------------------------------- #

    def dotfile(self, filename, etsbox=False, ets="full", jtype=False, static=True):
        """
        Write a link transform graph as a GraphViz dot file
        :param file: Name of file to write to
        :type file: str or file
        :param etsbox: Put the link ETS in a box, otherwise an edge label
        :type etsbox: bool
        :param jtype: Arrowhead to node indicates revolute or prismatic type
        :type jtype: bool
        :param static: Show static joints in blue and bold
        :type static: bool
        The file can be processed using dot::
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
        .. note:: If ``filename`` is a file object then the file will *not*
            be closed after the GraphViz model is written.
        :seealso: :func:`showgraph`
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

    def dfs_links(self, start, func=None):
        """
        Visit all links from start in depth-first order and will apply
        func to each visited link
        :param start: the link to start at
        :type start: Link
        :param func: An optional function to apply to each link as it is found
        :type func: function
        :returns: A list of links
        :rtype: list of Link
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

    def _get_limit_links(
        self,
        end: Union[Gripper, Link, str, None] = None,
        start: Union[Gripper, Link, str, None] = None,
    ) -> Tuple[Link, Union[Link, Gripper], Union[None, SE3]]:
        """
        Get and validate an end-effector, and a base link
        :param end: end-effector or gripper to compute forward kinematics to
        :type end: str or Link or Gripper, optional
        :param start: name or reference to a base link, defaults to None
        :type start: str or Link, optional
        :raises ValueError: link not known or ambiguous
        :raises ValueError: [description]
        :raises TypeError: unknown type provided
        :return: end-effector link, base link, and tool transform of gripper
            if applicable
        :rtype: Link, Elink, SE3 or None
        Helper method to find or validate an end-effector and base link.
        """

        # Try cache
        # if self._cache_end is not None:
        #     return self._cache_end, self._cache_start, self._cache_end_tool

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

            # Cache result
            self._cache_end = end
            self._cache_end_tool = tool
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

        return end_ret, start_ret, tool

    def _getlink(
        self,
        link: Union[Link, Gripper, str, None],
        default: Union[Link, Gripper, str, None] = None,
    ) -> Link:
        """
        Validate reference to Link

        :param link: link

        :raises ValueError: link does not belong to this ERobot
        :raises TypeError: bad argument

        :return: link reference

        ``robot._getlink(link)`` is a validated reference to a Link within
        the ERobot ``robot``.  If ``link`` is:

        -  an ``Link`` reference it is validated as belonging to
          ``robot``.
        - a string, then it looked up in the robot's link name dictionary, and
          a Link reference returned.
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


# =========================================================================== #


class ERobot(BaseERobot):
    def __init__(self, arg, urdf_string=None, urdf_filepath=None, **kwargs):

        if isinstance(arg, ERobot):
            # We're passed an ERobot, clone it
            # We need to preserve the parent link as we copy

            # Copy each link within the robot
            links = [deepcopy(link) for link in arg.links]
            gripper_links = []

            for gripper in arg.grippers:
                glinks = []
                for link in gripper.links:
                    glinks.append(deepcopy(link))

                gripper_links.append(glinks[0])
                links = links + glinks

            # print(links[9] is gripper_links[0])
            # print(gripper_links)

            # Sever parent connection, but save the string
            # The constructor will piece this together for us
            for link in links:
                link._children = []
                if link.parent is not None:
                    link._parent_name = link.parent.name
                    link._parent = None

            # gripper_parents = []

            # # Make a list of old gripper links
            # for gripper in arg.grippers:
            #     gripper_parents.append(gripper.links[0].name)

            # gripper_links = []

            # def dfs(node, node_copy):
            #     for child in node.children:
            #         child_copy = child.copy(node_copy)
            #         links.append(child_copy)

            #         # If this link was a gripper link, add to the list
            #         if child_copy.name in gripper_parents:
            #             gripper_links.append(child_copy)

            #         dfs(child, child_copy)

            # link0 = arg.links[0]
            # links.append(arg.links[0].copy())
            # dfs(link0, links[0])

            # print(gripper_links[0].jindex)

            super().__init__(links, gripper_links=gripper_links, **kwargs)

            for i, gripper in enumerate(self.grippers):
                gripper.tool = arg.grippers[i].tool.copy()

            # if arg.qdlim is not None:
            #     self.qdlim = arg.qdlim

            self._urdf_string = arg.urdf_string
            self._urdf_filepath = arg.urdf_filepath

        else:
            self._urdf_string = urdf_string
            self._urdf_filepath = urdf_filepath

            if isinstance(arg, DHRobot):
                # we're passed a DHRobot object
                # TODO handle dynamic parameters if given
                arg = arg.ets

            if isinstance(arg, ETS):
                # we're passed an ETS string
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

            elif islistof(arg, Link):
                links = arg

            else:
                raise TypeError("constructor argument must be ETS or list of Link")

            super().__init__(links, **kwargs)

    @classmethod
    def URDF(cls, file_path, gripper=None):
        """
        Construct an ERobot object from URDF file
        :param file_path: [description]
        :type file_path: [type]
        :param gripper: index or name of the gripper link(s)
        :type gripper: int or str or list
        :return: [description]
        :rtype: [type]
        If ``gripper`` is specified, links from that link outward are removed
        from the rigid-body tree and folded into a ``Gripper`` object.
        """
        links, name, _, _ = ERobot.URDF_read(file_path)

        if gripper is not None:
            if isinstance(gripper, int):
                gripper = links[gripper]
            elif isinstance(gripper, str):
                for link in links:
                    if link.name == gripper:
                        gripper = link
                        break
                else:
                    raise ValueError(f"no link named {gripper}")
            else:
                raise TypeError("bad argument passed as gripper")

        links, name, urdf_string, urdf_filepath = ERobot.URDF_read(file_path)
        print(cls)
        return cls(
            links,
            name=name,
            gripper_links=gripper,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

    @property
    def urdf_string(self):
        return self._urdf_string

    @property
    def urdf_filepath(self):
        return self._urdf_filepath

    # --------------------------------------------------------------------- #

    def _to_dict(self, robot_alpha=1.0, collision_alpha=0.0):

        # self._set_link_fk(self.q)

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

    @staticmethod
    def URDF_read(file_path, tld=None, xacro_tld=None):
        """
        Read a URDF file as Links
        :param file_path: File path relative to the xacro folder
        :type file_path: str, in Posix file path fprmat
        :param tld: A custom top-level directory which holds the xacro data,
            defaults to None
        :type tld: str, optional
        :param xacro_tld: A custom top-level within the xacro data,
            defaults to None
        :type xacro_tld: str, optional
        :return: Links and robot name
        :rtype: tuple(Link list, str)
        File should be specified relative to ``RTBDATA/URDF/xacro``

        .. note:: If ``tld`` is not supplied, filepath pointing to xacro data should
            be directly under ``RTBDATA/URDF/xacro`` OR under ``./xacro`` relative
            to the model file calling this method. If ``tld`` is supplied, then
            ```file_path``` needs to be relative to ``tld``
        """

        # get the path to the class that defines the robot
        if tld is None:
            base_path = rtb_path_to_datafile("xacro")
        else:
            base_path = PurePosixPath(tld)
        # print("*** urdf_to_ets_args: ", classpath)
        # add on relative path to get to the URDF or xacro file
        # base_path = PurePath(classpath).parent.parent / 'URDF' / 'xacro'
        file_path = base_path / PurePosixPath(file_path)
        name, ext = splitext(file_path)

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

        return urdf.elinks, urdf.name, urdf_string, file_path

    # --------------------------------------------------------------------- #

    def get_path(self, end=None, start=None):
        """
        Find a path from start to end. The end must come after
        the start (ie end must be further away from the base link
        of the robot than start) in the kinematic chain and both links
        must be a part of the same branch within the robot structure. This
        method is a work in progress while an approach which generalises
        to all applications is designed.
        :param end: end-effector or gripper to compute forward kinematics to
        :type end: str or Link or Gripper, optional
        :param start: name or reference to a base link, defaults to None
        :type start: str or Link, optional
        :raises ValueError: link not known or ambiguous
        :return: the path from start to end
        :rtype: list of Link
        """
        path = []
        n = 0

        end, start, tool = self._get_limit_links(end=end, start=start)

        # This is way faster than doing if x in y method
        try:
            return self._path_cache[start.name][end.name]
        except KeyError:
            pass

        if start.name not in self._path_cache:
            self._path_cache[start.name] = {}
            # self._path_cache_fknm[start.name] = {}

        link = end

        path.append(link)
        if link.isjoint:
            n += 1

        while link != start:
            link = link.parent
            if link is None:
                raise ValueError(
                    f"cannot find path from {start.name} to" f" {end.name}"
                )
            path.append(link)
            if link.isjoint:
                n += 1

        path.reverse()
        # path_fknm = [x._fknm for x in path]

        if tool is None:
            tool = SE3()

        self._path_cache[start.name][end.name] = (path, n, tool)
        # self._path_cache_fknm[start.name][end.name] = (path_fknm, n, tool.A)

        return path, n, tool

    def fkine(
        self,
        q: ArrayLike,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        tool: Union[ndarray, SE3, None] = None,
        include_base: bool = True,
    ) -> SE3:
        """
        Forward kinematics

        :param q: Joint coordinates
        :type q: ArrayLike
        :param end: end-effector or gripper to compute forward kinematics to
        :param start: the link to compute forward kinematics from
        :param tool: tool transform, optional

        :return: The transformation matrix representing the pose of the
            end-effector

        - ``T = robot.fkine(q)`` evaluates forward kinematics for the robot at
          joint configuration ``q``.
        **Trajectory operation**:

        If ``q`` has multiple rows (mxn), it is considered a trajectory and the
        result is an ``SE3`` instance with ``m`` values.
        .. note::
            - For a robot with a single end-effector there is no need to
              specify ``end``
            - For a robot with multiple end-effectors, the ``end`` must
              be specified.
            - The robot's base tool transform, if set, is incorporated
              into the result.
            - A tool transform, if provided, is incorporated into the result.
            - Works from the end-effector link to the base

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """
        return SE3(
            self.ets(start, end).fkine(
                q, base=self._T, tool=tool, include_base=include_base
            ),
            check=False,
        )

    def jacob0(
        self,
        q: ArrayLike,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        tool: Union[ndarray, SE3, None] = None,
    ) -> ndarray:
        r"""
        Manipulator geometric Jacobian in the base frame

        :param q: Joint coordinate vector
        :type q: ArrayLike
        :param end: the particular link or gripper whose velocity the Jacobian
            describes, defaults to the end-effector if only one is present
        :param start: the link considered as the base frame, defaults to the robots's base frame
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None

        :return J: Manipulator Jacobian in the base frame

        - ``robot.jacobo(q)`` is the manipulator Jacobian matrix which maps
          joint  velocity to end-effector spatial velocity expressed in the
          end-effector frame.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        Example:
        .. runblock:: pycon
            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.ETS.Puma560()
            >>> puma.jacobe([0, 0, 0, 0, 0, 0])

        .. warning:: This is the geometric Jacobian as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a
            velocity twist as per the text by Lynch & Park.

        .. warning:: ``start`` and ``end`` must be on the same branch,
            with ``start`` closest to the base.
        """  # noqa
        return self.ets(start, end).jacob0(q, tool=tool)

    def jacobe(
        self,
        q: ArrayLike,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        tool: Union[ndarray, SE3, None] = None,
    ) -> ndarray:
        r"""
        Manipulator geometric Jacobian in the end-effector frame

        :param q: Joint coordinate vector
        :type q: ArrayLike
        :param end: the particular link or Gripper whose velocity the Jacobian
            describes, defaults to the end-effector if only one is present
        :param start: the link considered as the base frame, defaults to the robots's base frame
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None

        :return J: Manipulator Jacobian in the end-effector frame

        - ``robot.jacobe(q)`` is the manipulator Jacobian matrix which maps
          joint  velocity to end-effector spatial velocity expressed in the
          end-effector frame.
        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        Example:
        .. runblock:: pycon
            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.ETS.Puma560()
            >>> puma.jacobe([0, 0, 0, 0, 0, 0])

        .. warning:: This is the **geometric Jacobian** as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a
            velocity twist as per the text by Lynch & Park.

        .. warning:: ``start`` and ``end`` must be on the same branch,
            with ``start`` closest to the base.
        """  # noqa
        return self.ets(start, end).jacobe(q, tool=tool)

    def hessian0(
        self,
        q: Union[ArrayLike, None] = None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        J0: Union[ndarray, None] = None,
        tool: Union[ndarray, SE3, None] = None,
    ) -> ndarray:
        r"""
        Manipulator Hessian

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the world-coordinate frame. This
        function calulcates this based on the ETS of the robot. One of J0 or q
        is required. Supply J0 if already calculated to save computation time

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: ArrayLike
        :param end: the final link/Gripper which the Hessian represents
        :param start: the first link which the Hessian represents
        :param J0: The manipulator Jacobian in the 0 frame
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None
        
        :return: The manipulator Hessian in 0 frame
        
        This method computes the manipulator Hessian in the base frame.  If
        we take the time derivative of the differential kinematic relationship
        .. math::
            \nu    &= \mat{J}(\vec{q}) \dvec{q} \\
            \alpha &= \dmat{J} \dvec{q} + \mat{J} \ddvec{q}
        where
        .. math::
            \dmat{J} = \mat{H} \dvec{q}
        and :math:`\mat{H} \in \mathbb{R}^{6\times n \times n}` is the
        Hessian tensor.

        The elements of the Hessian are
        .. math::
            \mat{H}_{i,j,k} =  \frac{d^2 u_i}{d q_j d q_k}
        where :math:`u = \{t_x, t_y, t_z, r_x, r_y, r_z\}` are the elements
        of the spatial velocity vector.
        Similarly, we can write
        .. math::
            \mat{J}_{i,j} = \frac{d u_i}{d q_j}
        
        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """
        return self.ets(start, end).hessian0(q, J0=J0, tool=tool)

    def hessiane(
        self,
        q: Union[ArrayLike, None] = None,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        Je: Union[ndarray, None] = None,
        tool: Union[ndarray, SE3, None] = None,
    ) -> ndarray:
        r"""
        Manipulator Hessian

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the ee frame. This
        function calulcates this based on the ETS of the robot. One of Je or q
        is required. Supply Je if already calculated to save computation time

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: ArrayLike
        :param end: the final link/Gripper which the Hessian represents
        :param start: the first link which the Hessian represents
        :param Je: The manipulator Jacobian in the ee frame
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None
        
        :return: The manipulator Hessian in ee frame
        
        This method computes the manipulator Hessian in the ee frame.  If
        we take the time derivative of the differential kinematic relationship
        .. math::
            \nu    &= \mat{J}(\vec{q}) \dvec{q} \\
            \alpha &= \dmat{J} \dvec{q} + \mat{J} \ddvec{q}
        where
        .. math::
            \dmat{J} = \mat{H} \dvec{q}
        and :math:`\mat{H} \in \mathbb{R}^{6\times n \times n}` is the
        Hessian tensor.

        The elements of the Hessian are
        .. math::
            \mat{H}_{i,j,k} =  \frac{d^2 u_i}{d q_j d q_k}
        where :math:`u = \{t_x, t_y, t_z, r_x, r_y, r_z\}` are the elements
        of the spatial velocity vector.
        Similarly, we can write
        .. math::
            \mat{J}_{i,j} = \frac{d u_i}{d q_j}
        
        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """
        return self.ets(start, end).hessiane(q, Je=Je, tool=tool)

    def partial_fkine0(
        self,
        q: ArrayLike,
        n: int = 3,
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
    ):
        r"""
        Manipulator Forward Kinematics nth Partial Derivative

        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the ee frame. This
        function calulcates this based on the ETS of the robot. One of Je or q
        is required. Supply Je if already calculated to save computation time

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: ArrayLike
        :param end: the final link/Gripper which the Hessian represents
        :param start: the first link which the Hessian represents
        :param tool: a static tool transformation matrix to apply to the
            end of end, defaults to None

        :return: The nth Partial Derivative of the forward kinematics

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        return self.ets(start, end).partial_fkine0(q, n=n)

    def link_collision_damper(
        self,
        shape,
        q=None,
        di=0.3,
        ds=0.05,
        xi=1.0,
        end=None,
        start=None,
        collision_list=None,
    ):
        """
        Formulates an inequality contraint which, when optimised for will
        make it impossible for the robot to run into a collision. Requires
        See examples/neo.py for use case
        :param ds: The minimum distance in which a joint is allowed to
            approach the collision object shape
        :type ds: float
        :param di: The influence distance in which the velocity
            damper becomes active
        :type di: float
        :param xi: The gain for the velocity damper
        :type xi: float
        :param from_link: The first link to consider, defaults to the base
            link
        :type from_link: Link
        :param to_link: The last link to consider, will consider all links
            between from_link and to_link in the robot, defaults to the
            end-effector link
        :type to_link: Link
        :returns: Ain, Bin as the inequality contraints for an omptimisor
        :rtype: ndarray(6), ndarray(6)
        """

        end, start, _ = self._get_limit_links(start=start, end=end)

        links, n, _ = self.get_path(start=start, end=end)

        # if q is None:
        #     q = copy(self.q)
        # else:
        #     q = getvector(q, n)

        j = 0
        Ain = None
        bin = None

        def indiv_calculation(link, link_col, q):
            d, wTlp, wTcp = link_col.closest_point(shape, di)

            if d is not None:
                lpTcp = -wTlp + wTcp

                norm = lpTcp / d
                norm_h = expand_dims(concatenate((norm, [0, 0, 0])), axis=0)

                # tool = (self.fkine(q, end=link).inv() * SE3(wTlp)).A[:3, 3]

                # Je = self.jacob0(q, end=link, tool=tool)
                # Je[:3, :] = self._T[:3, :3] @ Je[:3, :]

                # n_dim = Je.shape[1]
                # dp = norm_h @ shape.v
                # l_Ain = zeros((1, self.n))

                Je = self.jacobe(q, start=self.base_link, end=link, tool=link_col.T)
                n_dim = Je.shape[1]
                dp = norm_h @ shape.v
                l_Ain = zeros((1, n))

                l_Ain[0, :n_dim] = norm_h @ Je
                l_bin = (xi * (d - ds) / (di - ds)) + dp
            else:
                l_Ain = None
                l_bin = None

            return l_Ain, l_bin

        for link in links:
            if link.isjoint:
                j += 1

            if collision_list is None:
                col_list = link.collision
            else:
                col_list = collision_list[j - 1]

            for link_col in col_list:
                l_Ain, l_bin = indiv_calculation(link, link_col, q)

                if l_Ain is not None and l_bin is not None:
                    if Ain is None:
                        Ain = l_Ain
                    else:
                        Ain = concatenate((Ain, l_Ain))

                    if bin is None:
                        bin = array(l_bin)
                    else:
                        bin = concatenate((bin, l_bin))

        return Ain, bin

    def vision_collision_damper(
        self,
        shape,
        camera=None,
        camera_n=0,
        q=None,
        di=0.3,
        ds=0.05,
        xi=1.0,
        end=None,
        start=None,
        collision_list=None,
    ):
        """
        Formulates an inequality contraint which, when optimised for will
        make it impossible for the robot to run into a line of sight.
        See examples/fetch_vision.py for use case
        :param camera: The camera link, either as a robotic link or SE3
            pose
        :type camera: ERobot or SE3
        :param camera_n: Degrees of freedom of the camera link
        :type camera_n: int
        :param ds: The minimum distance in which a joint is allowed to
            approach the collision object shape
        :type ds: float
        :param di: The influence distance in which the velocity
            damper becomes active
        :type di: float
        :param xi: The gain for the velocity damper
        :type xi: float
        :param from_link: The first link to consider, defaults to the base
            link
        :type from_link: ELink
        :param to_link: The last link to consider, will consider all links
            between from_link and to_link in the robot, defaults to the
            end-effector link
        :type to_link: ELink
        :returns: Ain, Bin as the inequality contraints for an omptimisor
        :rtype: ndarray(6), ndarray(6)
        """

        if start is None:
            start = self.base_link

        if end is None:
            end = self.ee_link

        links, n, _ = self.get_path(start=start, end=end)

        j = 0
        Ain = None
        bin = None

        def rotation_between_vectors(a, b):
            a = a / npnorm(a)
            b = b / npnorm(b)

            angle = arccos(dot(a, b))
            axis = cross(a, b)

            return SE3.AngleAxis(angle, axis)

        if isinstance(camera, ERobot):
            wTcp = camera.fkine(camera.q).A[:3, 3]
        elif isinstance(camera, SE3):
            wTcp = camera.t

        wTtp = shape.T[:3, -1]

        # Create line of sight object
        los_mid = SE3((wTcp + wTtp) / 2)
        los_orientation = rotation_between_vectors(array([0.0, 0.0, 1.0]), wTcp - wTtp)

        los = Cylinder(
            radius=0.001,
            length=npnorm(wTcp - wTtp),
            base=(los_mid * los_orientation),
        )

        def indiv_calculation(link, link_col, q):
            d, wTlp, wTvp = link_col.closest_point(los, di)

            if d is not None:
                lpTvp = -wTlp + wTvp

                norm = lpTvp / d
                norm_h = expand_dims(concatenate((norm, [0, 0, 0])), axis=0)

                tool = SE3((inv(self.fkine(q, end=link).A) @ SE3(wTlp).A)[:3, 3])

                Je = self.jacob0(q, end=link, tool=tool.A)
                Je[:3, :] = self._T[:3, :3] @ Je[:3, :]
                n_dim = Je.shape[1]

                if isinstance(camera, ERobot):
                    Jv = camera.jacob0(camera.q)
                    Jv[:3, :] = self._T[:3, :3] @ Jv[:3, :]

                    Jv *= npnorm(wTvp - shape.T[:3, -1]) / los.length

                    dpc = norm_h @ Jv
                    dpc = concatenate(
                        (
                            dpc[0, :-camera_n],
                            zeros(self.n - (camera.n - camera_n)),
                            dpc[0, -camera_n:],
                        )
                    )
                else:
                    dpc = zeros((1, self.n + camera_n))

                dpt = norm_h @ shape.v
                dpt *= npnorm(wTvp - wTcp) / los.length

                l_Ain = zeros((1, self.n + camera_n))
                l_Ain[0, :n_dim] = norm_h @ Je
                l_Ain -= dpc
                l_bin = (xi * (d - ds) / (di - ds)) + dpt
            else:
                l_Ain = None
                l_bin = None

            return l_Ain, l_bin

        for link in links:
            if link.isjoint:
                j += 1

            if collision_list is None:
                col_list = link.collision
            else:
                col_list = collision_list[j - 1]

            for link_col in col_list:
                l_Ain, l_bin = indiv_calculation(link, link_col, q)

                if l_Ain is not None and l_bin is not None:
                    if Ain is None:
                        Ain = l_Ain
                    else:
                        Ain = concatenate((Ain, l_Ain))

                    if bin is None:
                        bin = array(l_bin)
                    else:
                        bin = concatenate((bin, l_bin))

        return Ain, bin

    # inverse dynamics (recursive Newton-Euler) using spatial vector notation
    def rne(self, q, qd, qdd, symbolic=False, gravity=None):

        n = self.n

        # allocate intermediate variables
        Xup = SE3.Alloc(n)
        Xtree = SE3.Alloc(n)

        v = SpatialVelocity.Alloc(n)
        a = SpatialAcceleration.Alloc(n)
        f = SpatialForce.Alloc(n)
        I = SpatialInertia.Alloc(n)  # noqa
        s = []  # joint motion subspace

        if symbolic:
            Q = empty((n,), dtype="O")  # joint torque/force
        else:
            Q = empty((n,))  # joint torque/force

        # TODO Should the dynamic parameters of static links preceding joint be
        # somehow merged with the joint?

        # A temp variable to handle static joints
        Ts = SE3()

        # A counter through joints
        j = 0

        # initialize intermediate variables
        for link in self.links:
            if link.isjoint:
                I[j] = SpatialInertia(m=link.m, r=link.r)
                if symbolic and link.Ts is None:
                    Xtree[j] = SE3(eye(4, dtype="O"), check=False)
                else:
                    Xtree[j] = Ts * SE3(link.Ts, check=False)

                if link.v is not None:
                    s.append(link.v.s)

                # Increment the joint counter
                j += 1

                # Reset the Ts tracker
                Ts = SE3()
            else:
                # TODO Keep track of inertia and transform???
                Ts *= SE3(link.Ts, check=False)

        if gravity is None:
            a_grav = -SpatialAcceleration(self.gravity)
        else:
            a_grav = -SpatialAcceleration(gravity)

        # forward recursion
        for j in range(0, n):
            vJ = SpatialVelocity(s[j] * qd[j])

            # transform from parent(j) to j
            Xup[j] = SE3(self.links[j].A(q[j])).inv()

            if self.links[j].parent is None:
                v[j] = vJ
                a[j] = Xup[j] * a_grav + SpatialAcceleration(s[j] * qdd[j])
            else:
                jp = self.links[j].parent.jindex  # type: ignore
                v[j] = Xup[j] * v[jp] + vJ
                a[j] = Xup[j] * a[jp] + SpatialAcceleration(s[j] * qdd[j]) + v[j] @ vJ

            f[j] = I[j] * a[j] + v[j] @ (I[j] * v[j])

        # backward recursion
        for j in reversed(range(0, n)):

            # next line could be dot(), but fails for symbolic arguments
            Q[j] = sum(f[j].A * s[j])

            if self.links[j].parent is not None:
                jp = self.links[j].parent.jindex  # type: ignore
                f[jp] = f[jp] + Xup[j] * f[j]

        return Q

    # --------------------------------------------------------------------- #

    def ik_lm_chan(
        self,
        Tep: Union[ndarray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        λ: float = 1.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Chan's Method)

        :param Tep: The desired end-effector pose or pose trajectory
        :param end: the particular link or gripper to compute the pose of
        :param start: the link considered as the base frame, defaults to the robots's base frame
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return self.ets(start, end).ik_lm_chan(Tep, q0, ilimit, slimit, tol, reject_jl, we, λ)

    def ik_lm_wampler(
        self,
        Tep: Union[ndarray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        λ: float = 1.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Wamplers's Method)

        :param Tep: The desired end-effector pose or pose trajectory
        :param end: the particular link or gripper to compute the pose of
        :param start: the link considered as the base frame, defaults to the robots's base frame
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return self.ets(start, end).ik_lm_wampler(Tep, q0, ilimit, slimit, tol, reject_jl, we, λ)

    def ik_lm_sugihara(
        self,
        Tep: Union[ndarray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        λ: float = 1.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Sugihara's Method)

        :param Tep: The desired end-effector pose or pose trajectory
        :param end: the particular link or gripper to compute the pose of
        :param start: the link considered as the base frame, defaults to the robots's base frame
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return self.ets(start, end).ik_lm_sugihara(Tep, q0, ilimit, slimit, tol, reject_jl, we, λ)

    def ik_nr(
        self,
        Tep: Union[ndarray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        use_pinv: int = True,
        pinv_damping: float = 0.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Newton-Raphson Method)

        :param Tep: The desired end-effector pose or pose trajectory
        :param end: the particular link or gripper to compute the pose of
        :param start: the link considered as the base frame, defaults to the robots's base frame
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return self.ets(start, end).ik_nr(Tep, q0, ilimit, slimit, tol, reject_jl, we, use_pinv, pinv_damping)

    def ik_gn(
        self,
        Tep: Union[ndarray, SE3],
        end: Union[str, Link, Gripper, None] = None,
        start: Union[str, Link, Gripper, None] = None,
        q0: Union[ndarray, None] = None,
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        reject_jl: bool = True,
        we: Union[ndarray, None] = None,
        use_pinv: int = True,
        pinv_damping: float = 0.0,
    ) -> Tuple[ndarray, int, int, int, float]:
        """
        Numerical inverse kinematics by Levenberg-Marquadt optimization (Gauss-NewtonMethod)

        :param Tep: The desired end-effector pose or pose trajectory
        :param end: the particular link or gripper to compute the pose of
        :param start: the link considered as the base frame, defaults to the robots's base frame
        :param q0: initial joint configuration (default to random valid joint
            configuration contrained by the joint limits of the robot)
        :param ilimit: maximum number of iterations per search
        :param slimit: maximum number of search attempts
        :param tol: final error tolerance
        :param reject_jl: constrain the solution to being within the joint limits of
            the robot (reject solution with invalid joint configurations and perfrom
            another search up to the slimit)
        :param we: a mask vector which weights the end-effector error priority.
            Corresponds to translation in X, Y and Z and rotation about X, Y and Z
            respectively
        :param λ: value of lambda for the damping matrix Wn

        :return: inverse kinematic solution
        :rtype: tuple (q, success, iterations, searches, residual)

        ``sol = ets.ik_lm_chan(Tep)`` are the joint coordinates (n) corresponding
        to the robot end-effector pose ``Tep`` which is an ``SE3`` or ``ndarray`` object.
        This method can be used for robots with any number of degrees of freedom.
        The return value ``sol`` is a tuple with elements:

        ============    ==========  ===============================================
        Element         Type        Description
        ============    ==========  ===============================================
        ``q``           ndarray(n)  joint coordinates in units of radians or metres
        ``success``     int         whether a solution was found
        ``iterations``  int         total number of iterations
        ``searches``    int         total number of searches
        ``residual``    float       final value of cost function
        ============    ==========  ===============================================

        If ``success == 0`` the ``q`` values will be valid numbers, but the
        solution will be in error.  The amount of error is indicated by
        the ``residual``.

        **Joint Limits**:

        ``sol = robot.ikine_LM(T, slimit=100)`` which is the deafualt for this method.
        The solver will initialise a solution attempt with a random valid q0 and
        perform a maximum of ilimit steps within this attempt. If a solution is not
        found, this process is repeated up to slimit times.

        **Global search**:

        ``sol = robot.ikine_LM(T, reject_jl=True)`` is the deafualt for this method.
        By setting reject_jl to True, the solver will discard any solution which
        violates the defined joint limits of the robot. The solver will then
        re-initialise with a new random q0 and repeat the process up to slimit times.
        Note that finding a solution with valid joint coordinates takes longer than
        without.

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``we`` option where the ``we`` vector
        (6) specifies the Cartesian DOF (in the wrist coordinate frame) that
        will be ignored in reaching a solution.  The we vector has six
        elements that correspond to translation in X, Y and Z, and rotation
        about X, Y and Z respectively. The value can be 0 (for ignore)
        or above to assign a priority relative to other Cartesian DoF. The number
        of non-zero elements must equal the number of manipulator DOF.

        For example when using a 3 DOF manipulator tool orientation might
        be unimportant, in which case use the option ``we=[1, 1, 1, 0, 0, 0]``.



        .. note::

            - See `Toolbox kinematics wiki page
                <https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics>`_
            - Implements a Levenberg-Marquadt variable-damping solver.
            - The tolerance is computed on the norm of the error between
                current and desired tool pose.  This norm is computed from
                distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
                depends on the initial guess ``q0``.

        :references:
            TODO

        :seealso:
            TODO
        """

        return self.ets(start, end).ik_gn(Tep, q0, ilimit, slimit, tol, reject_jl, we, use_pinv, pinv_damping)



# =========================================================================== #


class ERobot2(BaseERobot):
    def __init__(self, arg, **kwargs):

        if isinstance(arg, ETS2):
            # we're passed an ETS string
            links = []
            # chop it up into segments, a link frame after every joint
            parent = None
            for j, ets_j in enumerate(arg.split()):
                elink = Link2(ETS2(ets_j), parent=parent, name=f"link{j:d}")
                parent = elink
                if (
                    elink.qlim is None
                    and elink.v is not None
                    and elink.v.qlim is not None
                ):
                    elink.qlim = elink.v.qlim
                links.append(elink)

        elif islistof(arg, Link2):
            links = arg
        else:
            raise TypeError("constructor argument must be ETS2 or list of Link2")

        super().__init__(links, **kwargs)

        # should just set it to None
        self.base = SE2()  # override superclass

    @property
    def base(self) -> SE2:
        """
        Get/set robot base transform (Robot superclass)

        - ``robot.base`` is the robot base transform

        :return: robot tool transform
        :rtype: SE2 instance

        - ``robot.base = ...`` checks and sets the robot base transform

        .. note:: The private attribute ``_base`` will be None in the case of
            no base transform, but this property will return ``SE3()`` which
            is an identity matrix.
        """
        if self._base is None:
            self._base = SE2()

        # return a copy, otherwise somebody with
        # reference to the base can change it
        return self._base.copy()

    @base.setter
    def base(self, T):
        if T is None:
            self._base = T
        elif isinstance(self, ERobot2):
            # 2D robot
            if isinstance(T, SE2):
                self._base = T
            elif SE2.isvalid(T):
                self._tool = SE2(T, check=True)
        else:
            raise ValueError("base must be set to None (no tool) or SE2")

    def jacob0(self, q, start=None, end=None):
        return self.ets(start, end).jacob0(q)

    def jacobe(self, q, start=None, end=None):
        return self.ets(start, end).jacobe(q)

    def fkine(self, q, end=None, start=None):

        return self.ets(start, end).fkine(q)


# --------------------------------------------------------------------- #

# def teach(
#         self,
#         q=None,
#         block=True,
#         limits=None,
#         vellipse=False,
#         fellipse=False,
#         eeframe=True,
#         name=False,
#         unit='rad',
#         backend='pyplot2'):
#     """
#     2D Graphical teach pendant
#     :param block: Block operation of the code and keep the figure open
#     :type block: bool
#     :param q: The joint configuration of the robot (Optional,
#         if not supplied will use the stored q values).
#     :type q: float ndarray(n)
#     :param limits: Custom view limits for the plot. If not supplied will
#         autoscale, [x1, x2, y1, y2]
#     :type limits: array_like(4)
#     :param vellipse: (Plot Option) Plot the velocity ellipse at the
#         end-effector
#     :type vellipse: bool
#     :param vellipse: (Plot Option) Plot the force ellipse at the
#         end-effector
#     :type vellipse: bool
#     :param eeframe: (Plot Option) Plot the end-effector coordinate frame
#         at the location of the end-effector. Uses three arrows, red,
#         green and blue to indicate the x, y, and z-axes.
#     :type eeframe: bool
#     :param name: (Plot Option) Plot the name of the robot near its base
#     :type name: bool
#     :param unit: angular units: 'rad' [default], or 'deg'
#     :type unit: str

#     :return: A reference to the PyPlot object which controls the
#         matplotlib figure
#     :rtype: PyPlot
#     - ``robot.teach2(q)`` creates a 2D matplotlib plot which allows the
#       user to "drive" a graphical robot using a graphical slider panel.
#       The robot's inital joint configuration is ``q``. The plot will
#       autoscale with an aspect ratio of 1.
#     - ``robot.teach2()`` as above except the robot's stored value of ``q``
#       is used.
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
#           If not set then
#             - For revolute joints they are assumed to be [-pi, +pi]
#             - For prismatic joint they are assumed unknown and an error
#               occurs.
#     """

#     if q is None:
#         q = zeros((self.n,))
#     else:
#         q = getvector(q, self.n)

#     if unit == 'deg':
#         q = self.toradians(q)

#     # Make an empty 3D figure
#     env = self._get_graphical_backend(backend)

#     # Add the robot to the figure in readonly mode
#     env.launch('Teach ' + self.name, limits=limits)
#     env.add(
#         self, readonly=True,
#         eeframe=eeframe, name=name)

#     env._add_teach_panel(self, q)

#     if limits is None:
#         limits = r_[-1, 1, -1, 1] * self.reach * 1.5
#         env.ax.set_xlim([limits[0], limits[1]])
#         env.ax.set_ylim([limits[2], limits[3]])

#     if vellipse:
#         vell = self.vellipse(centre='ee', scale=0.5)
#         env.add(vell)

#     if fellipse:
#         fell = self.fellipse(centre='ee')
#         env.add(fell)

#     # Keep the plot open
#     if block:           # pragma: no cover
#         env.hold()

#     return env


if __name__ == "__main__":  # pragma nocover

    e1 = Link(ETS(ET.Rz()), jindex=0)
    e2 = Link(ETS(ET.Rz()), jindex=1, parent=e1)
    e3 = Link(ETS(ET.Rz()), jindex=2, parent=e2)
    e4 = Link(ETS(ET.Rz()), jindex=5, parent=e3)

    ERobot([e1, e2, e3, e4])
