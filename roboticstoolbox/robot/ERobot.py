#!/usr/bin/env python3
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

import sys
import os
from os.path import splitext
import tempfile
import subprocess
import webbrowser
import numpy as np
# import spatialmath as sp
from spatialmath import SE3
from spatialmath.base.argcheck import getvector, verifymatrix, getmatrix
from roboticstoolbox.robot.ELink import ELink, ETS
# from roboticstoolbox.backends.PyPlot.functions import \
#     _plot, _teach, _fellipse, _vellipse, _plot_ellipse, \
#     _plot2, _teach2
from roboticstoolbox.tools import xacro
from roboticstoolbox.tools import URDF
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.robot.Gripper import Gripper
from pathlib import PurePath, PurePosixPath
from ansitable import ANSITable, Column
from spatialmath import SpatialAcceleration, SpatialVelocity, \
    SpatialInertia, SpatialForce


class ERobot(Robot):
    """
    The ERobot. A superclass which represents the
    kinematics of a serial-link manipulator

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

    :references:
        - Kinematic Derivatives using the Elementary Transform Sequence,
          J. Haviland and P. Corke
    """

    # TODO do we need tool and base as well?

    def __init__(
            self,
            elinks,
            base_link=None,
            gripper_links=None,
            checkjindex=True,
            **kwargs
            ):

        self._ets = []
        self._linkdict = {}
        self._n = 0
        self._ee_links = []
        self._base_link = None

        # Ordered links, we reorder the input elinks to be in depth first
        # search order
        orlinks = []

        link_number = 0
        if isinstance(elinks, ETS):
            # were passed an ETS string
            ets = elinks
            elinks = []

            # chop it up into segments, a link frame after every joint
            start = 0
            for j, k in enumerate(ets.joints()):
                ets_j = ets[start:k+1]
                start = k + 1
                if j == 0:
                    parent = None
                else:
                    parent = elinks[-1]
                elink = ELink(ets_j, parent=parent, name=f"link{j:d}")
                elinks.append(elink)

            n = len(ets.joints())

            tool = ets[start:]
            if len(tool) > 0:
                elinks.append(ELink(tool, parent=elinks[-1], name="ee"))
        elif isinstance(elinks, list):
            # we're passed a list of ELinks

            # check all the incoming ELink objects
            n = 0
            for link in elinks:
                if isinstance(link, ELink):
                    # if link has no name, give it one
                    if link.name is None:
                        link.name = f"link-{link_number}"
                        link_number += 1

                    # put it in the link dictionary, check for duplicates
                    if link.name in self._linkdict:
                        raise ValueError(
                            f'link name {link.name} is not unique')
                    self._linkdict[link.name] = link
                else:
                    raise TypeError("Input can be only ELink")
                if link.isjoint:
                    n += 1
        else:
            raise TypeError('elinks must be a list of ELinks or an ETS')

        self._n = n

        # scan for base
        for link in elinks:
            # is this a base link?
            if link._parent is None:
                if self._base_link is not None:
                    raise ValueError('Multiple base links')
                self._base_link = link
            else:
                # no, update children of this link's parent
                link._parent._child.append(link)

        # Set up the gripper, make a list containing the root of all
        # grippers
        if gripper_links is not None:
            if isinstance(gripper_links, ELink):
                gripper_links = [gripper_links]
        else:
            gripper_links = []

        # An empty list to hold all grippers
        self.grippers = []

        # Make a gripper object for each gripper
        for link in gripper_links:
            g_links = self.dfs_links(link)

            # Remove gripper links from the robot
            for g_link in g_links:
                elinks.remove(g_link)

            # Save the gripper object
            self.grippers.append(Gripper(g_links))

        # Subtract the n of the grippers from the n of the robot
        for gripper in self.grippers:
            self._n -= gripper.n

        # Set the ee links
        self.ee_links = []
        if len(gripper_links) == 0:
            for link in elinks:
                # is this a leaf node? and do we not have any grippers
                if len(link.child) == 0:
                    # no children, must be an end-effector
                    self.ee_links.append(link)
        else:
            for link in gripper_links:
                # use the passed in value
                self.ee_links.append(link.parent)

        # assign the joint indices
        if all([link.jindex is None for link in elinks]):

            jindex = [0]  # "mutable integer" hack

            def visit_link(link, jindex):
                # if it's a joint, assign it a jindex and increment it
                if link.isjoint and link in elinks:
                    link.jindex = jindex[0]
                    jindex[0] += 1

                if link in elinks:
                    orlinks.append(link)

            # visit all links in DFS order
            self.dfs_links(
                self.base_link, lambda link: visit_link(link, jindex))

        elif all([link.jindex is not None for link in elinks if link.isjoint]):
            # jindex set on all, check they are unique and sequential
            if checkjindex:
                jset = set(range(self._n))
                for link in elinks:
                    if link.isjoint and link.jindex not in jset:
                        raise ValueError(
                            f'joint index {link.jindex} was '
                            'repeated or out of range')
                    jset -= set([link.jindex])
                if len(jset) > 0:  # pragma nocover  # is impossible
                    raise ValueError(f'joints {jset} were not assigned')
            orlinks = elinks
        else:
            # must be a mixture of ELinks with/without jindex
            raise ValueError(
                'all links must have a jindex, or none have a jindex')

        # Current joint angles of the robot
        # TODO should go to Robot class?
        self.q = np.zeros(self.n)
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)
        self.control_type = 'v'

        super().__init__(orlinks, **kwargs)

    def dfs_links(self, start, func=None):
        """
        Visit all links from start in depth-first order and will apply
        func to each visited link

        :param start: the link to start at
        :type start: ELink
        :param func: An optional function to apply to each link as it is found
        :type func: function

        :returns: A list of links
        :rtype: list of ELink
        """
        visited = []

        def vis_children(link):
            visited.append(link)
            if func is not None:
                func(link)

            for li in link.child:
                if li not in visited:
                    vis_children(li)

        vis_children(start)

        return visited

    # def dfs_path(self, l1, l2):
    #     path = []
    #     visited = [l1]

    #     def vis_children(link):
    #         visited.append(link)

    #         for li in link.child:
    #             if li not in visited:

    #                 if li == l2 or vis_children(li):
    #                     path.append(li)
    #                     return True
    #     vis_children(l1)
    #     path.append(l1)
    #     path.reverse()
    #     return path

    def to_dict(self):
        ob = {
            'links': [],
            'name': self.name,
            'n': self.n
        }

        self.fkine_all(self.q)

        for link in self.links:
            li = {
                'axis': [],
                'eta': [],
                'geometry': [],
                'collision': []
            }

            for et in link.ets():
                li['axis'].append(et.axis)
                li['eta'].append(et.eta)

            if link.v is not None:
                li['axis'].append(link.v.axis)
                li['eta'].append(link.v.eta)

            for gi in link.geometry:
                li['geometry'].append(gi.to_dict())

            for gi in link.collision:
                li['collision'].append(gi.to_dict())

            ob['links'].append(li)

        # Do the grippers now
        for gripper in self.grippers:
            for link in gripper.links:
                li = {
                    'axis': [],
                    'eta': [],
                    'geometry': [],
                    'collision': []
                }

                for et in link.ets():
                    li['axis'].append(et.axis)
                    li['eta'].append(et.eta)

                if link.v is not None:
                    li['axis'].append(link.v.axis)
                    li['eta'].append(link.v.eta)

                for gi in link.geometry:
                    li['geometry'].append(gi.to_dict())

                for gi in link.collision:
                    li['collision'].append(gi.to_dict())

                ob['links'].append(li)

        return ob

    def fk_dict(self):
        ob = {
            'links': []
        }

        self.fkine_all(self.q)

        # Do the robot
        for link in self.links:

            li = {
                'geometry': [],
                'collision': []
            }

            for gi in link.geometry:
                li['geometry'].append(gi.fk_dict())

            for gi in link.collision:
                li['collision'].append(gi.fk_dict())

            ob['links'].append(li)

        # Do the grippers now
        for gripper in self.grippers:
            for link in gripper.links:
                li = {
                    'geometry': [],
                    'collision': []
                }

                for gi in link.geometry:
                    li['geometry'].append(gi.fk_dict())

                for gi in link.collision:
                    li['collision'].append(gi.fk_dict())

                ob['links'].append(li)

        return ob

    # @classmethod
    # def urdf_to_ets(cls, file_path):
    #     name, ext = splitext(file_path)

    #     if ext == '.xacro':
    #         urdf_string = xacro.main(file_path)
    #         urdf = URDF.loadstr(urdf_string, file_path)

    #     return ERobot(
    #         urdf.elinks,
    #         name=urdf.name
    #     )

    def urdf_to_ets_args(self, file_path, tld=None):
        """
        [summary]

        :param file_path: File path relative to the xacro folder
        :type file_path: str, in Posix file path fprmat
        :param tld: top-level directory, defaults to None
        :type tld: str, optional
        :return: Links and robot name
        :rtype: tuple(ELink list, str)
        """

        # get the path to the class that defines the robot
        classpath = sys.modules[self.__module__].__file__
        # print("*** urdf_to_ets_args: ", classpath)
        # add on relative path to get to the URDF or xacro file
        base_path = PurePath(classpath).parent.parent / 'URDF' / 'xacro'
        file_path = base_path / PurePosixPath(file_path)
        name, ext = splitext(file_path)

        if ext == '.xacro':
            # it's a xacro file, preprocess it
            if tld is not None:
                tld = base_path / PurePosixPath(tld)
            urdf_string = xacro.main(file_path, tld)
            urdf = URDF.loadstr(urdf_string, file_path)
        else:  # pragma nocover
            urdf = URDF.loadstr(open(file_path).read(), file_path)

        return urdf.elinks, urdf.name

    # @classmethod
    # def dh_to_ets(cls, robot):
    #     """
    #     Converts a robot modelled with standard or modified DH parameters to
    #     an ERobot representation

    #     :param robot: The robot model to be converted
    #     :type robot: SerialLink
    #     :return: List of returned :class:`bluepy.btle.Characteristic` objects
    #     :rtype: ets class
    #     """
    #     ets = []
    #     q_idx = []
    #     M = 0

    #     for j in range(robot.n):
    #         L = robot.links[j]

    #         # Method for modified DH parameters
    #         if robot.mdh:

    #             # Append Tx(a)
    #             if L.a != 0:
    #                 ets.append(ET.Ttx(L.a))
    #                 M += 1

    #             # Append Rx(alpha)
    #             if L.alpha != 0:
    #                 ets.append(ET.TRx(L.alpha))
    #                 M += 1

    #             if L.is_revolute:
    #                 # Append Tz(d)
    #                 if L.d != 0:
    #                     ets.append(ET.Ttz(L.d))
    #                     M += 1

    #                 # Append Rz(q)
    #                 ets.append(ET.TRz(joint=j+1))
    #                 q_idx.append(M)
    #                 M += 1

    #             else:
    #                 # Append Tz(q)
    #                 ets.append(ET.Ttz(joint=j+1))
    #                 q_idx.append(M)
    #                 M += 1

    #                 # Append Rz(theta)
    #                 if L.theta != 0:
    #                     ets.append(ET.TRz(L.alpha))
    #                     M += 1

    #     return cls(
    #         ets,
    #         q_idx,
    #         robot.name,
    #         robot.manuf,
    #         robot.base,
    #         robot.tool)

    @property
    def qlim(self):
        v = np.zeros((2, self.n))
        j = 0

        for i in range(len(self.links)):
            if self.links[i].isjoint:
                v[:, j] = self.links[i].qlim
                j += 1

        return v

    # @property
    # def qdlim(self):
    #     return self.qdlim

# --------------------------------------------------------------------- #

    @property
    def n(self):
        return self._n
# --------------------------------------------------------------------- #

    @property
    def elinks(self):
        # return self._linkdict
        return self._links
# --------------------------------------------------------------------- #

    @property
    def link_dict(self):
        return self._linkdict
# --------------------------------------------------------------------- #

    @property
    def base_link(self):
        return self._base_link

    @base_link.setter
    def base_link(self, link):
        if isinstance(link, ELink):
            self._base_link = link
        else:
            # self._base_link = self.links[link]
            raise TypeError('Must be an ELink')
        # self._reset_fk_path()
# --------------------------------------------------------------------- #
    # TODO  get configuration string

    @property
    def ee_links(self):
        return self._ee_links

    # def add_ee(self, link):
    #     if isinstance(link, ELink):
    #         self._ee_link.append(link)
    #     else:
    #         raise ValueError('must be an ELink')
    #     self._reset_fk_path()

    @ee_links.setter
    def ee_links(self, link):
        if isinstance(link, ELink):
            self._ee_links = [link]
        elif isinstance(link, list) and \
                all([isinstance(x, ELink) for x in link]):
            self._ee_links = link
        else:
            raise TypeError('expecting an ELink or list of ELinks')
        # self._reset_fk_path()

    @property
    def reach(self):
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
            d = 0
            for link in self:
                for et in link.ets():
                    if et.isprismatic:
                        d += abs(et.eta)
                if link.isprismatic and link.qlim is not None:
                    d += link.qlim[1]
            self._reach = d
        return self._reach

# --------------------------------------------------------------------- #

    # @property
    # def ets(self):
    #     return self._ets

# --------------------------------------------------------------------- #

    # @property
    # def M(self):
    #     return self._M

# --------------------------------------------------------------------- #

    # @property
    # def q_idx(self):
    #     return self._q_idx

# --------------------------------------------------------------------- #

    # def ets(self, ee=None):
    #     if ee is None:
    #         if len(self.ee_links) == 1:
    #             link = self.ee_links[0]
    #         else:
    #             raise ValueError(
    #                 'robot has multiple end-effectors, specify one')
    #     # elif isinstance(ee, str) and ee in self._linkdict:
    #     #     ee = self._linkdict[ee]
    #     elif isinstance(ee, ELink) and ee in self._links:
    #         link = ee
    #     else:
    #         raise ValueError('end-effector is not valid')

    #     ets = ETS()

    #     # build the ETS string from ee back to root
    #     while link is not None:
    #         ets = link.ets() * ets
    #         link = link.parent

    #     return ets

    def ets(self, start=None, end=None, explored=None, path=None):
        """
        ERobot to ETS

        :param start: start of path, defaults to ``base_link``
        :type start: ELink or str, optional
        :param end: end of path, defaults to end-effector
        :type end: ELink or str, optional
        :raises ValueError: a link does not belong to this ERobot
        :raises TypeError: a bad link argument
        :return: elementary transform sequence
        :rtype: ETS instance

        - ``robot.ets()`` is an ETS representing the kinematics from base to
          end-effector.
        - ``robot.ets(end=link)`` is an ETS representing the kinematics from
          base to the link ``link`` specified as an ELink reference or a name.
        - ``robot.ets(start=l1, end=l2)`` is an ETS representing the kinematics
          from link ``l1`` to link ``l2``.
        """
        v = self._getlink(start, self.base_link)
        if end is None and len(self.ee_links) > 1:
            raise ValueError(
                'ambiguous, specify which end-effector is required')
        end = self._getlink(end, self.ee_links[0])

        if explored is None:
            explored = set()
        toplevel = path is None

        explored.add(v)
        if v == end:
            return path

        # unlike regular DFS, the neighbours of the node are its children
        # and its parent.

        # visit child nodes
        if toplevel:
            path = v.ets()
        for w in v.child:
            if w not in explored:
                p = self.ets(w, end, explored, path * w.ets())
                if p:
                    return p

        # visit parent node
        if toplevel:
            path = ETS()
        if v.parent is not None:
            w = v.parent
            if w not in explored:
                p = self.ets(w, end, explored, path * v.ets().inv())
                if p:
                    return p

        return None

    def config(self):
        s = ''
        for link in self.links:
            if link.v is not None:
                if link.v.isprismatic:
                    s += 'P'
                elif link.v.isrevolute:
                    s += 'R'
        return s

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
        os.remove(pdffile.name)

    def dotfile(self, filename, etsbox=False, jtype=False, static=True):
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
            file = open(filename, 'w')
        else:
            file = filename

        header = r"""digraph G {

graph [rankdir=LR];

"""

        def draw_edge(link, etsbox, jtype, static):
            # draw the edge
            if jtype:
                if link.isprismatic:
                    edge_options = \
                        'arrowhead="box", arrowtail="inv", dir="both"'
                elif link.isrevolute:
                    edge_options = \
                        'arrowhead="dot", arrowtail="inv", dir="both"'
                else:
                    edge_options = 'arrowhead="normal"'
            else:
                edge_options = 'arrowhead="normal"'

            if link.parent is None:
                parent = 'BASE'
            else:
                parent = link.parent.name

            if etsbox:
                # put the ets fragment in a box
                if not link.isjoint and static:
                    node_options = ', fontcolor="blue"'
                else:
                    node_options = ''
                file.write(
                    '  {}_ets [shape=box, style=rounded, '
                    'label="{}"{}];\n'.format(
                        link.name, link.ets().__str__(
                            q=f"q{link.jindex}"), node_options))
                file.write('  {} -> {}_ets;\n'.format(parent, link.name))
                file.write('  {}_ets -> {} [{}];\n'.format(
                    link.name, link.name, edge_options))
            else:
                # put the ets fragment as an edge label
                if not link.isjoint and static:
                    edge_options += 'fontcolor="blue"'
                file.write('  {} -> {} [label="{}", {}];\n'.format(
                    parent, link.name, link.ets().__str__(
                        q=f"q{link.jindex}"), edge_options))

        file.write(header)

        # add the base link
        file.write('  BASE [shape=square, style=filled, fillcolor=gray]\n')

        # add the links
        for link in self:
            # draw the link frame node (circle) or ee node (doublecircle)
            if link in self.ee_links:
                # end-effector
                node_options = \
                    'shape="doublecircle", color="blue", fontcolor="blue"'
            else:
                node_options = 'shape="circle"'

            file.write('  {} [{}];\n'.format(link.name, node_options))

            draw_edge(link, etsbox, jtype, static)

        for gripper in self.grippers:
            for link in gripper.links:
                file.write('  {} [shape=cds];\n'.format(link.name))
                draw_edge(link, etsbox, jtype, static)

        file.write('}\n')

        if isinstance(filename, str):
            close(file)  # noqa

# --------------------------------------------------------------------- #

    def fkine(self, q, endlink=None, startlink=None, tool=None):
        '''
        Forward kinematics

        :param q: Joint coordinates
        :type q: ndarray(n) or ndarray(m,n)
        :param endlink: end-effector to compute forward kinematics to
        :type endlink: str or ELink
        :param startlink: the link to compute forward kinematics from
        :type startlink: str or ELink
        :param tool: tool transform, optional
        :type tool: SE3
        :return: The transformation matrix representing the pose of the
            end-effector
        :rtype: SE3 instance

        - ``T = robot.fkine(q)`` evaluates forward kinematics for the robot at
          joint configuration ``q``.

        **Trajectory operation**:

        If ``q`` has multiple rows (mxn), it is considered a trajectory and the
        result is an ``SE3`` instance with ``m`` values.

        .. note::

            - For a robot with a single end-effector there is no need to
              specify ``endlink``
            - For a robot with multiple end-effectors, the ``endlink`` must
              be specified.
            - The robot's base tool transform, if set, is incorporated
              into the result.
            - A tool transform, if provided, is incorporated into the result.
            - Works from the end-effector link to the base

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke

        '''

        q = getmatrix(q, (None, self.n))

        if tool is not None:
            Ttool = tool.A

        endlink, startlink = self._get_limit_links(endlink, startlink)

        T = SE3.Empty()
        for k, qk in enumerate(q):

            link = endlink  # start with last link

            # add tool if provided
            if tool is None:
                Tk = link.A(qk[link.jindex], fast=True)
            else:
                Tk = link.A(qk[link.jindex], fast=True) @ Ttool

            # add remaining links, back toward the base
            while True:
                link = link.parent

                if link is None:
                    break

                Tk = link.A(qk[link.jindex], fast=True) @ Tk

                if link is startlink:
                    break

            # add base transform if it is set
            if self.base is not None and startlink == self.base_link:
                Tk = self.base.A @ Tk

            T.append(SE3(Tk))

        return T

    def fkine_all(self, q):
        '''
        Tall = robot.fkine_all(q) evaluates fkine for each joint within a
        robot and returns a trajecotry of poses.

        Tall = fkine_all() as above except uses the stored q value of the
        robot object.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :return T: Homogeneous transformation trajectory
        :rtype T: SE3 list

        .. note::

            - The robot's base transform, if present, are incorporated
              into the result.

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke

        '''

        q = getvector(q, self.n)

        for link in self.elinks:
            if link.isjoint:
                t = link.A(q[link.jindex])
            else:
                t = link.A()

            # Update the links internal transform wrt the base frame
            if link.parent is None:
                link._fk = self.base * t
            else:
                link._fk = link.parent._fk * t

            # Update the link model transforms as well
            for col in link.collision:
                col.wT = link._fk

            for gi in link.geometry:
                gi.wT = link._fk

        # Do the grippers now
        for gripper in self.grippers:
            for link in gripper.links:
                if link.isjoint:
                    t = link.A(gripper.q[link.jindex])
                else:
                    t = link.A()

                link._fk = link.parent._fk * t

                # Update the link model transforms as well
                for col in link.collision:
                    col.wT = link._fk

                for gi in link.geometry:
                    gi.wT = link._fk

    # def jacob0(self, q=None):
    #     """
    #     J0 = jacob0(q) is the manipulator Jacobian matrix which maps joint
    #     velocity to end-effector spatial velocity. v = J0*qd in the
    #     base frame.

    #     J0 = jacob0() as above except uses the stored q value of the
    #     robot object.

    #     :param q: The joint angles/configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)

    #     :return J: The manipulator Jacobian in ee frame
    #     :rtype: float ndarray(6,n)

    #     :references:
    #         - Kinematic Derivatives using the Elementary Transform
    #           Sequence, J. Haviland and P. Corke
    #     """

    #     if q is None:
    #         q = np.copy(self.q)
    #     else:
    #         q = getvector(q, self.n)

    #     T = (self.base.inv() * self.fkine(q)).A
    #     U = np.eye(4)
    #     j = 0
    #     J = np.zeros((6, self.n))

    #     for link in self._fkpath:

    #         for k in range(link.M):

    #             if k != link.q_idx:
    #                 U = U @ link.ets[k].T().A
    #             else:
    #                 # self._jacoblink(link, k, T)
    #                 U = U @ link.ets[k].T(q[j]).A
    #                 Tu = np.linalg.inv(U) @ T
    #                 n = U[:3, 0]
    #                 o = U[:3, 1]
    #                 a = U[:3, 2]
    #                 x = Tu[0, 3]
    #                 y = Tu[1, 3]
    #                 z = Tu[2, 3]

    #                 if link.ets[k].axis == 'Rz':
    #                     J[:3, j] = (o * x) - (n * y)
    #                     J[3:, j] = a

    #                 elif link.ets[k].axis == 'Ry':
    #                     J[:3, j] = (n * z) - (a * x)
    #                     J[3:, j] = o

    #                 elif link.ets[k].axis == 'Rx':
    #                     J[:3, j] = (a * y) - (o * z)
    #                     J[3:, j] = n

    #                 elif link.ets[k].axis == 'tx':
    #                     J[:3, j] = n
    #                     J[3:, j] = np.array([0, 0, 0])

    #                 elif link.ets[k].axis == 'ty':
    #                     J[:3, j] = o
    #                     J[3:, j] = np.array([0, 0, 0])

    #                 elif link.ets[k].axis == 'tz':
    #                     J[:3, j] = a
    #                     J[3:, j] = np.array([0, 0, 0])

    #                 j += 1

    #     return J

    def get_path(self, endlink=None, startlink=None):
        """
        Find a path from startlink to endlink. The endlink must come after
        the startlink (ie endlink must be further away from the base link
        of the robot than startlink) in the kinematic chain and both links
        must be a part of the same branch within the robot structure. This
        method is a work in progress while an approach which generalises
        to all applications is designed.

        :param endlink: name or reference to end-effector, defaults to None
        :type endlink: str or ELink, optional
        :param startlink: name or reference to a base link, defaults to None
        :type startlink: str or ELink, optional
        :raises ValueError: link not known or ambiguous
        :return: the path from startlink to endlink
        :rtype: list of Link
        """
        path = []
        n = 0

        endlink, startlink = self._get_limit_links(endlink, startlink)

        link = endlink

        path.append(link)
        if link.isjoint:
            n += 1

        while link != startlink:
            link = link.parent
            if link is None:
                raise ValueError(
                    f'cannot find path from {startlink.name} to'
                    f' {endlink.name}')
            path.append(link)
            if link.isjoint:
                n += 1

        path.reverse()

        return path, n

    def _get_limit_links(self, endlink=None, startlink=None):
        """
        Get and validate an end-effector, and a base link

        :param endlink: name or reference to end-effector, defaults to None
        :type endlink: str or ELink, optional
        :param startlink: name or reference to a base link, defaults to None
        :type startlink: str or ELink, optional
        :raises ValueError: link not known or ambiguous
        :raises ValueError: [description]
        :raises TypeError: unknown type provided
        :return: end-effector link, base link
        :rtype: ELink, Elink

        Helper method to find or validate an end-effector and base link.
        """
        if endlink is None:

            # if we have a gripper, use it
            if len(self.grippers) == 1:
                endlink = self.grippers[0].links[0]
            elif len(self.grippers) > 1:
                # if more than one gripper, user must choose
                raise ValueError('Must specify which gripper')

            # no grippers, use ee link if just one
            elif len(self.ee_links) == 1:
                endlink = self.ee_links[0]
            else:
                # if more than one EE, user must choose
                raise ValueError('Must specify which end-effector')
        else:
            # end effector is specified
            endlink = self._getlink(endlink)

        if startlink is None:
            startlink = self.base_link
        else:
            # start effector is specified
            startlink = self._getlink(startlink)

        return endlink, startlink

    def _getlink(self, link, default=None):
        """
        Validate reference to ELink

        :param link: link
        :type link: ELink or str
        :raises ValueError: link does not belong to this ERobot
        :raises TypeError: bad argument
        :return: link reference
        :rtype: ELink

        ``robot._getlink(link)`` is a validated reference to an ELink within
        the ERobot ``robot``.  If ``link`` is:

        -  an ``ELink`` reference it is validated as belonging to
          ``robot``.
        - a string, then it looked up in the robot's link name dictionary, and
          an ELink reference returned.
        """
        if link is None:
            link = default

        if isinstance(link, str):
            if link in self.link_dict:
                return self.link_dict[link]

            raise ValueError(f'no link named {link}')

        elif isinstance(link, ELink):
            if link in self.links:
                return link
            else:
                for gripper in self.grippers:
                    if link in gripper.links:
                        return link

                raise ValueError('link not in robot links')
        else:
            raise TypeError('unknown argument')

    def jacob0(self, q, endlink=None, startlink=None, offset=None, T=None):
        """
        Manipulator geometric Jacobian in the base frame

        :param q: Joint coordinate vector
        :type q: ndarray(n)
        :param endlink: the particular link whose velocity the Jacobian describes, defaults
            to the end-effector if only one is present
        :type endlink: str or ELink
        :param startlink: the link considered as the base frame, defaults to the robots's base frame
        :type startlink: str or ELink
        :param offset: a static offset transformation matrix to apply to the
            end of endlink, defaults to None
        :type offset: SE3, optional
        :param T: The transformation matrix of the reference point which the
            Jacobian represents with respect to the base frame. Use this to
            avoid caluclating forward kinematics to save time, defaults
            to None
        :type T: SE3, optional
        :return J: Manipulator Jacobian in the base frame
        :rtype: ndarray(6,n)

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

        .. note:: This is the geometric Jacobian as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a 
            velocity twist as per the text by Lynch & Park.

        .. warning:: ``startlink`` and ``endlink`` must be on the same branch,
            with ``startlink`` closest to the base.
        """  # noqa

        if offset is None:
            offset = SE3()

        path, n = self.get_path(endlink, startlink)

        q = getvector(q, self.n)

        if T is None:
            T = self.base.inv() * \
                self.fkine(q, endlink=endlink, startlink=startlink) * offset
        T = T.A
        U = np.eye(4)
        j = 0
        J = np.zeros((6, n))

        for link in path:

            if link.isjoint:
                U = U @ link.A(q[link.jindex], fast=True)

                if link == endlink:
                    U = U @ offset.A

                Tu = np.linalg.inv(U) @ T
                n = U[:3, 0]
                o = U[:3, 1]
                a = U[:3, 2]
                x = Tu[0, 3]
                y = Tu[1, 3]
                z = Tu[2, 3]

                if link.v.axis == 'Rz':
                    J[:3, j] = (o * x) - (n * y)
                    J[3:, j] = a

                elif link.v.axis == 'Ry':
                    J[:3, j] = (n * z) - (a * x)
                    J[3:, j] = o

                elif link.v.axis == 'Rx':
                    J[:3, j] = (a * y) - (o * z)
                    J[3:, j] = n

                elif link.v.axis == 'tx':
                    J[:3, j] = n
                    J[3:, j] = np.array([0, 0, 0])

                elif link.v.axis == 'ty':
                    J[:3, j] = o
                    J[3:, j] = np.array([0, 0, 0])

                elif link.v.axis == 'tz':
                    J[:3, j] = a
                    J[3:, j] = np.array([0, 0, 0])

                j += 1
            else:
                U = U @ link.A(fast=True)

        return J

    def jacobe(self, q, endlink=None, startlink=None, offset=None, T=None):
        """
        Manipulator geometric Jacobian in the end-effector frame

        :param q: Joint coordinate vector
        :type q: ndarray(n)
        :param endlink: the particular link whose velocity the Jacobian describes, defaults
            to the end-effector if only one is present
        :type endlink: str or ELink
        :param startlink: the link considered as the base frame, defaults to the robots's base frame
        :type startlink: str or ELink
        :param offset: a static offset transformation matrix to apply to the
            end of endlink, defaults to None
        :type offset: SE3, optional
        :param T: The transformation matrix of the reference point which the
            Jacobian represents with respect to the base frame. Use this to
            avoid caluclating forward kinematics to save time, defaults
            to None
        :type T: SE3, optional
        :return J: Manipulator Jacobian in the end-effector frame
        :rtype: ndarray(6,n)

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

        .. note:: This is the **geometric Jacobian** as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a 
            velocity twist as per the text by Lynch & Park.

        .. warning:: ``startlink`` and ``endlink`` must be on the same branch,
            with ``startlink`` closest to the base.
        """  # noqa

        q = getvector(q, self.n)

        if offset is None:
            offset = SE3()

        endlink, startlink = self._get_limit_links(endlink, startlink)

        path, n = self.get_path(endlink, startlink)

        if T is None:
            T = self.base.inv() * \
                self.fkine(q, endlink=endlink, startlink=startlink) * offset

        J0 = self.jacob0(q, endlink, startlink, offset, T)
        Je = self.jacobev(q, endlink, startlink, offset, T) @ J0
        return Je

    def partial_fkine0(self, q, n, J0=None, endlink=None, startlink=None):
        endlink, startlink = self._get_limit_links(endlink, startlink)

        def cross(a, b):
            x = a[1] * b[2] - a[2] * b[1]
            y = a[2] * b[0] - a[0] * b[2]
            z = a[0] * b[1] - a[1] * b[0]
            return np.array([x, y, z])

        _, nl = self.get_path(endlink, startlink)

        J = self.jacob0(q, endlink=endlink, startlink=startlink)
        H = self.hessian0(q, J, endlink, startlink)

        d = [J, H]
        size = [6, nl, nl]
        count = np.array([0, 0])
        c = 2

        def add_indices(indices, c):
            total = len(indices * 2)
            new_indices = []

            for i in range(total):
                j = i // 2
                new_indices.append([])
                new_indices[i].append(indices[j][0].copy())
                new_indices[i].append(indices[j][1].copy())

                # if even number
                if i % 2 == 0:
                    new_indices[i][0].append(c)
                # if odd number
                else:
                    new_indices[i][1].append(c)

            return new_indices

        def add_pdi(pdi):
            total = len(pdi * 2)
            new_pdi = []

            for i in range(total):
                j = i // 2
                new_pdi.append([])
                new_pdi[i].append(pdi[j][0])
                new_pdi[i].append(pdi[j][1])

                # if even number
                if i % 2 == 0:
                    new_pdi[i][0] += 1
                # if odd number
                else:
                    new_pdi[i][1] += 1

            return new_pdi

        # these are the indices used for the hessian
        indices = [[[1], [0]]]

        # the are the pd indices used in the corss prods
        pdi = [[0, 0]]

        while len(d) != n:
            size.append(nl)
            count = np.r_[count, 0]
            indices = add_indices(indices, c)
            pdi = add_pdi(pdi)
            c += 1

            pd = np.zeros(size)

            for i in range(nl ** c):

                rot = np.zeros(3)
                trn = np.zeros(3)

                for j in range(len(indices)):
                    pdr0 = d[pdi[j][0]]
                    pdr1 = d[pdi[j][1]]

                    idx0 = count[indices[j][0]]
                    idx1 = count[indices[j][1]]

                    rot += cross(
                        pdr0[(slice(3, 6), *idx0)],
                        pdr1[(slice(3, 6), *idx1)])

                    trn += cross(
                        pdr0[(slice(3, 6), *idx0)],
                        pdr1[(slice(0, 3), *idx1)])

                pd[(slice(0, 3), *count)] = trn
                pd[(slice(3, 6), *count)] = rot

                count[0] += 1
                for j in range(len(count)):
                    if count[j] == nl:
                        count[j] = 0
                        if (j != len(count) - 1):
                            count[j + 1] += 1

            d.append(pd)

        return d[-1]

    # def third(self, q=None, J0=None, endlink=None, startlink=None):
    #     endlink, startlink = self._get_limit_links(endlink, startlink)
    #     path, n = self.get_path(endlink, startlink)

    #     def cross(a, b):
    #         x = a[1] * b[2] - a[2] * b[1]
    #         y = a[2] * b[0] - a[0] * b[2]
    #         z = a[0] * b[1] - a[1] * b[0]
    #         return np.array([x, y, z])

    #     if J0 is None:
    #         q = getvector(q, n)
    #         J0 = self.jacob0(q, endlink=endlink)
    #     else:
    #         verifymatrix(J0, (6, n))

    #     H0 = self.hessian0(q, J0, endlink, startlink)

    #     L = np.zeros((6, n, n, n))

    #     for l in range(n):
    #         for k in range(n):
    #             for j in range(n):

    #                 L[:3, j, k, l] = cross(H0[3:, k, l], J0[:3, j]) + \
    #                     cross(J0[3:, k], H0[:3, j, l])

    #                 L[3:, j, k, l] = cross(H0[3:, k, l], J0[3:, j]) + \
    #                     cross(J0[3:, k], H0[3:, j, l])

    #     return L

    def hessian0(self, q=None, J0=None, endlink=None, startlink=None):
        """
        The manipulator Hessian tensor maps joint acceleration to end-effector
        spatial acceleration, expressed in the world-coordinate frame. This
        function calulcates this based on the ETS of the robot. One of J0 or q
        is required. Supply J0 if already calculated to save computation time

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param J0: The manipulator Jacobian in the 0 frame
        :type J0: float ndarray(6,n)
        :param endlink: the final link which the Hessian represents
        :type endlink: str or ELink
        :param startlink: the first link which the Hessian represents
        :type startlink: str or ELink

        :return: The manipulator Hessian in 0 frame
        :rtype: float ndarray(6,n,n)

        H[i,j,k] is d2 u_i / dq_j dq_k

        where u = {t_x, t_y, t_z, r_x, r_y, r_z}

        J[i,j] is d u_i / dq_j

        where u = {t_x, t_y, t_z, ζ_x, ζ_y, ζ_z}

        v = J qd

        a = Jd qd + J qdd

        Jd = H qd

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        endlink, startlink = self._get_limit_links(endlink, startlink)
        path, n = self.get_path(endlink, startlink)

        def cross(a, b):
            x = a[1] * b[2] - a[2] * b[1]
            y = a[2] * b[0] - a[0] * b[2]
            z = a[0] * b[1] - a[1] * b[0]
            return np.array([x, y, z])

        if J0 is None:
            q = getvector(q, n)
            J0 = self.jacob0(q, endlink=endlink)
        else:
            verifymatrix(J0, (6, n))

        H = np.zeros((6, n, n))

        for j in range(n):
            for i in range(j, n):

                H[:3, i, j] = cross(J0[3:, j], J0[:3, i])
                H[3:, i, j] = cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[:3, j, i] = H[:3, i, j]

        return H

    def jacobm(self, q=None, J=None, H=None, endlink=None, startlink=None):
        """
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
        :param endlink: the final link which the Hessian represents
        :type endlink: str or ELink
        :param startlink: the first link which the Hessian represents
        :type startlink: str or ELink

        :return: The manipulability Jacobian
        :rtype: float ndarray(n)

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        endlink, startlink = self._get_limit_links(endlink, startlink)
        path, n = self.get_path(endlink, startlink)

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, startlink=startlink, endlink=endlink)
        else:
            verifymatrix(J, (6, n))

        if H is None:
            H = self.hessian0(J0=J, startlink=startlink, endlink=endlink)
        else:
            verifymatrix(H, (6, n, n))

        manipulability = self.manipulability(
            q, J=J, startlink=startlink, endlink=endlink)
        b = np.linalg.inv(J @ np.transpose(J))
        Jm = np.zeros((n, 1))

        for i in range(n):
            c = J @ np.transpose(H[:, :, i])
            Jm[i, 0] = manipulability * \
                np.transpose(c.flatten('F')) @ b.flatten('F')

        return Jm

    def __str__(self):
        """
        Pretty prints the ETS Model of the robot. Will output angles in
        degrees

        :return: Pretty print of the robot model
        :rtype: str

        Constant links are shown in blue.
        End-effector links are prefixed with an @
        """
        table = ANSITable(
            Column("id", headalign="^"),
            Column("link", headalign="^"),
            Column("parent", headalign="^"),
            Column("joint", headalign="^"),
            Column("ETS", headalign="^", colalign="<"),
            border="thin")
        for k, link in enumerate(self):
            color = "" if link.isjoint else "<<blue>>"
            ee = "@" if link in self.ee_links else ""
            ets = link.ets()
            table.row(
                k,
                color + ee + link.name,
                link.parent.name if link.parent is not None else "-",
                link._joint_name if link.parent is not None else "",
                ets.__str__(f"q{link._jindex}"))

        s = str(table)
        s += self.configurations_str()

        return s

    def hierarchy(self):
        """
        Pretty print the robot link hierachy

        :return: Pretty print of the robot model
        :rtype: str

        Example:

        .. runblock:: pycon

            import roboticstoolbox as rtb
            robot = rtb.models.URDF.Panda()
            robot.hierarchy()

        """

        # link = self.base_link

        def recurse(link, indent=0):
            print(' ' * indent * 2, link.name)
            for child in link.child:
                recurse(child, indent+1)

        recurse(self.base_link)

    def jacobev(
            self, q, endlink=None, startlink=None,
            offset=None, T=None):
        """
        Jv = jacobev(q) is the spatial velocity Jacobian, at joint
        configuration q, which relates the velocity in the base frame to the
        velocity in the end-effector frame.

        :param q: Joint coordinate vector
        :type q: ndarray(n)
        :param endlink: the final link which the Jacobian represents
        :type endlink: str or ELink
        :param startlink: the first link which the Jacobian represents
        :type startlink: str or ELink
        :param offset: a static offset transformation matrix to apply to the
            end of endlink, defaults to None
        :type offset: SE3, optional
        :param T: The transformation matrix of the reference point which the
            Jacobian represents with respect to the base frame. Use this to
            avoid caluclating forward kinematics to save time, defaults
            to None
        :type T: SE3, optional

        :returns J: The velocity Jacobian in ee frame
        :rtype J: float ndarray(6,6)

        """

        endlink, startlink = self._get_limit_links(endlink, startlink)

        if T is None:
            T = self.base.inv() * \
                self.fkine(q, endlink=endlink, startlink=startlink)
            if offset is not None:
                T *= offset
        R = (T.R).T

        Jv = np.zeros((6, 6))
        Jv[:3, :3] = R
        Jv[3:, 3:] = R

        return Jv

    def jacob0v(
            self, q, endlink=None, startlink=None,
            offset=None, T=None):
        """
        Jv = jacob0v(q) is the spatial velocity Jacobian, at joint
        configuration q, which relates the velocity in the end-effector frame
        to velocity in the base frame

        :param q: Joint coordinate vector
        :type q: ndarray(n)
        :param endlink: the final link which the Jacobian represents
        :type endlink: str or ELink
        :param startlink: the first link which the Jacobian represents
        :type startlink: str or ELink
        :param offset: a static offset transformation matrix to apply to the
            end of endlink, defaults to None
        :type offset: SE3, optional
        :param T: The transformation matrix of the reference point which the
            Jacobian represents with respect to the base frame. Use this to
            avoid caluclating forward kinematics to save time, defaults
            to None
        :type T: SE3, optional

        :returns J: The velocity Jacobian in 0 frame
        :rtype J: float ndarray(6,6)

        """

        endlink, startlink = self._get_limit_links(endlink, startlink)

        if T is None:
            T = self.base.inv() * \
                self.fkine(q, endlink=endlink, startlink=startlink)
            if offset is not None:
                T *= offset
        R = (T.R)

        Jv = np.zeros((6, 6))
        Jv[:3, :3] = R
        Jv[3:, 3:] = R

        return Jv

    def link_collision_damper(
            self, shape, q=None, di=0.3, ds=0.05, xi=1.0,
            endlink=None, startlink=None):
        '''
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
        :type from_link: ELink
        :param to_link: The last link to consider, will consider all links
            between from_link and to_link in the robot, defaults to the
            end-effector link
        :type to_link: ELink

        :returns: Ain, Bin as the inequality contraints for an omptimisor
        :rtype: ndarray(6), ndarray(6)
        '''

        if startlink is None:
            startlink = self.base_link

        if endlink is None:
            endlink = self.ee_link

        links, n = self.get_path(startlink=startlink, endlink=endlink)

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, n)

        j = 0
        Ain = None
        bin = None

        def indiv_calculation(link, link_col, q):
            d, wTlp, wTcp = link_col.closest_point(shape, di)

            if d is not None:
                lpTcp = wTlp.inv() * wTcp
                norm = lpTcp.t / d
                norm_h = np.expand_dims(np.r_[norm, 0, 0, 0], axis=0)

                Je = self.jacobe(
                    q, startlink=self.base_link, endlink=link,
                    offset=link_col.base)
                n_dim = Je.shape[1]
                dp = norm_h @ shape.v
                l_Ain = np.zeros((1, n))
                l_Ain[0, :n_dim] = norm_h @ Je
                l_bin = (xi * (d - ds) / (di - ds)) + dp
            else:
                l_Ain = None
                l_bin = None

            return l_Ain, l_bin, d, wTcp

        for link in links:
            if link.isjoint:
                j += 1

            for link_col in link.collision:
                l_Ain, l_bin, d, wTcp = indiv_calculation(
                                                link, link_col, q)

                if l_Ain is not None and l_bin is not None:
                    if Ain is None:
                        Ain = l_Ain
                    else:
                        Ain = np.r_[Ain, l_Ain]

                    if bin is None:
                        bin = np.array(l_bin)
                    else:
                        bin = np.r_[bin, l_bin]

        return Ain, bin

    # inverse dynamics (recursive Newton-Euler) using spatial vector notation
    def rne(robot, q, qd, qdd, gravity=None):

        n = robot.n

        # allocate intermediate variables
        Xup = SE3.Alloc(n)
        Xtree = SE3.Alloc(n)

        v = SpatialVelocity.Alloc(n)
        a = SpatialAcceleration.Alloc(n)
        f = SpatialForce.Alloc(n)
        I = SpatialInertia.Alloc(n)  # noqa
        s = [None for i in range(n)]   # joint motion subspace
        Q = np.zeros((n,))   # joint torque/force

        # initialize intermediate variables
        for j, link in enumerate(robot):
            I[j] = SpatialInertia(m=link.m, r=link.r)
            Xtree[j] = link.Ts
            s[j] = link.v.s

        if gravity is None:
            a_grav = SpatialAcceleration(robot.gravity)
        else:
            a_grav = SpatialAcceleration(gravity)

        # forward recursion
        for j in range(0, n):
            vJ = SpatialVelocity(s[j] * qd[j])

            # transform from parent(j) to j
            Xup[j] = robot[j].A(q[j]).inv()

            if robot[j].parent is None:
                v[j] = vJ
                a[j] = Xup[j] * a_grav + SpatialAcceleration(s[j] * qdd[j])
            else:
                jp = robot[j].parent.jindex
                v[j] = Xup[j] * v[jp] + vJ
                a[j] = Xup[j] * a[jp] \
                    + SpatialAcceleration(s[j] * qdd[j]) \
                    + v[j] @ vJ

            f[j] = I[j] * a[j] + v[j] @ (I[j] * v[j])

        # backward recursion
        for j in reversed(range(0, n)):
            Q[j] = f[j].dot(s[j])

            if robot[j].parent is not None:
                jp = robot[j].parent.jindex
                f[jp] = f[jp] + Xup[j] * f[j]

        return Q


if __name__ == "__main__":  # pragma nocover

    import roboticstoolbox as rtb
    np.set_printoptions(precision=4, suppress=True)

    p = rtb.models.ETS.Puma560()
    p.fkine(p.qz)
    p.jacob0(p.qz)
    p.jacobe(p.qz)

    # robot = rtb.models.ETS.Panda()
    # print(robot)
    # print(robot.base, robot.tool)
    # print(robot.ee_links)
    # ets = robot.ets()
    # print(ets)
    # print('n', ets.n)
    # ets2 = ets.compile()
    # print(ets2)

    # q = np.random.rand(7)
    # # print(ets.eval(q))
    # # print(ets2.eval(q))

    # J1 = robot.jacob0(q)
    # J2 = ets2.jacob0(q)
    # print(J1-J2)

    # print(robot[2].v, robot[2].v.jindex)
    # print(robot[2].Ts)
