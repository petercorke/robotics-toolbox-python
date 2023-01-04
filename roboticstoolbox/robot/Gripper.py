#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
from spatialmath import SE3
import spatialmath as sm
from spatialmath.base.argcheck import getvector
from roboticstoolbox.robot.Link import Link
from functools import lru_cache
from typing import Union, TypeVar, Generic, List, Callable
from roboticstoolbox.fknm import Robot_link_T
from roboticstoolbox.tools.types import ArrayLike, NDArray
from roboticstoolbox.robot.Link import BaseLink

# A generic type variable representing any subclass of BaseLink
LinkType = TypeVar("LinkType", bound=BaseLink)


class Gripper(Generic[LinkType]):
    def __init__(
        self,
        links: List[LinkType],
        name: str = "",
        tool: Union[NDArray, SE3, None] = None,
    ):

        self._n = 0

        self.name = name

        if tool is None:
            self.tool = sm.SE3()
        else:
            self.tool = tool

        for link in links:
            if link.isjoint:
                self._n += 1

        self.q = np.zeros(self.n)
        self._links = links

        # assign the joint indices
        if all(
            [
                link.jindex is None or link.ets._auto_jindex
                for link in links
                if link.isjoint
            ]
        ):

            jindex = [0]  # "mutable integer" hack

            def visit_link(link, jindex):
                # if it's a joint, assign it a jindex and increment it
                if link.isjoint:
                    link.jindex = jindex[0]
                    jindex[0] += 1

            # visit all links in DFS order
            self.dfs_links(links[0], lambda link: visit_link(link, jindex))

        elif all([link.jindex is not None for link in links if link.isjoint]):
            # jindex set on all, check they are unique and sequential
            jset = set(range(self.n))
            for link in links:
                if link.isjoint:
                    if link.jindex not in jset:
                        raise ValueError(
                            "gripper joint index {link.jindex} was "
                            "repeated or out of range"
                        )
                    jset -= set([link.jindex])
            if len(jset) > 0:  # pragma nocover # is impossible
                raise ValueError("gripper joints {jset} were not assigned")
        else:  # pragma nocover
            # must be a mixture of Links with/without jindex
            raise ValueError(
                "all gripper links must have a jindex, or none have a jindex"
            )

    def __str__(self):
        s = "Gripper("
        if self.name is not None:
            s += f'"{self.name}"'
        s += (
            f", connected to {self.links[0].parent_name}, {self.n} joints,"
            f" {len(self.links)} links"
        )
        s += ")"
        return s

    def __repr__(self):
        links = [link.__repr__() for link in self.links]

        tool = None if np.all(self.tool.A == np.eye(4)) else self.tool.A.__repr__()
        s = f'Gripper({links}, name="{self.name}", tool={tool})'

        return s

    def dfs_links(
        self, start: LinkType, func: Union[Callable[[Link], None], None] = None
    ):
        """
        Search links using depth first search

        Visit all links from start in depth-first order and will apply
        func to each visited link

        Parameters
        ----------
        start
            the link to start at
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

    @property
    def tool(self) -> SE3:
        """
        Get/set gripper tool transform

        - ``gripper.tool`` is the gripper tool transform as an SE3 object
        - ``gripper._tool`` is the gripper tool transform as a numpy array
        - ``gripper.tool = ...`` checks and sets the gripper tool transform

        Parameters
        ----------
        tool
            the new gripper tool transform (as an SE(3))

        Returns
        -------
        tool
            gripper tool transform



        """
        return SE3(self._tool, check=False)

    @tool.setter
    def tool(self, T: Union[SE3, NDArray]):
        if isinstance(T, SE3):
            self._tool = T.A
        else:
            self._tool = T

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
    def q(self) -> NDArray:
        """
        Get/set gripper joint configuration

        - ``gripper.q`` is the gripper joint configuration
        - ``gripper.q = ...`` checks and sets the joint configuration

        Parameters
        ----------
        q
            the new gripper joint configuration

        Returns
        -------
        q
            gripper joint configuration

        """

        return self._q

    @q.setter
    def q(self, q_new: ArrayLike):
        self._q = np.array(getvector(q_new, self.n))

    # --------------------------------------------------------------------- #

    @property
    def links(self) -> List[LinkType]:
        """
        Gripper links

        Returns
        -------
        links
            A list of link objects

        """
        return self._links

    # --------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        """
        Get/set gripper name

        - ``gripper.name`` is the gripper name
        - ``gripper.name = ...`` checks and sets the gripper name

        Parameters
        ----------
        name
            the new gripper name

        Returns
        -------
        name
            the current gripper name

        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

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
