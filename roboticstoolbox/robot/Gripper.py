#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
import spatialmath as sm
from spatialmath.base.argcheck import getvector
from roboticstoolbox.robot.Link import Link
from typing import List
from functools import lru_cache
from typing import Union
from fknm import Robot_link_T

ArrayLike = Union[list, np.ndarray, tuple, set]


class Gripper:
    def __init__(self, elinks, name="", tool=None):

        self._n = 0

        self.name = name

        if tool is None:
            self.tool = sm.SE3()
        else:
            self.tool = tool

        for link in elinks:
            if link.isjoint:
                self._n += 1

        self.q = np.zeros(self.n)
        self._links = elinks

        # assign the joint indices
        if all(
            [
                link.jindex is None or link.ets._auto_jindex
                for link in elinks
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
            self.dfs_links(elinks[0], lambda link: visit_link(link, jindex))

        elif all([link.jindex is not None for link in elinks if link.isjoint]):
            # jindex set on all, check they are unique and sequential
            jset = set(range(self.n))
            for link in elinks:
                if link.isjoint:
                    if link.jindex not in jset:
                        raise ValueError(
                            "gripper joint index {link.jindex} was "
                            "repeated or out of range"
                        )
                    jset -= set([link.jindex])
            if len(jset) > 0:  # pragma nocover # is impossible
                raise ValueError("gripper joints {jset} were not assigned")
        else:
            # must be a mixture of Links with/without jindex
            raise ValueError(
                "all gripper links must have a jindex, or none have a jindex"
            )

    def __str__(self):
        s = "Gripper("
        if self.name is not None:
            s += f'"{self.name}"'
        s += f", connected to {self.links[0].parent_name}, {self.n} joints, {len(self.links)} links"
        s += ")"
        return s

    def __repr__(self):
        return self.__str__()

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

    @property
    def n(self):
        return self._n

    @property
    def q(self):
        """
        Get/set gripper joint configuration

        - ``gripper.q`` is the gripper joint configuration

        :return: gripper joint configuration
        :rtype: ndarray(n,)

        - ``gripper.q = ...`` checks and sets the joint configuration

        .. note::  ???
        """
        return self._q

    @q.setter
    def q(self, q_new):
        self._q = getvector(q_new, self.n)

    # --------------------------------------------------------------------- #

    @property
    def links(self) -> List[Link]:
        """
        Gripper links

        :return: A list of link objects
        :rtype: list of Link subclass instances

        .. note:: It is probably more concise to index the robot object rather
            than the list of links, ie. the following are equivalent::

                robot.links[i]
                robot[i]
        """
        return self._links

    # --------------------------------------------------------------------- #

    @property
    def name(self):
        """
        Gripper name

        :return: The gripper name
        :rtype: string
        """
        return self._name

    @name.setter
    def name(self, new_name):
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
