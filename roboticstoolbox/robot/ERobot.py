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
        pass


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
