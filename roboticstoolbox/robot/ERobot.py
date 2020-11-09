#!/usr/bin/env python3
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

import sys
from os.path import splitext
import numpy as np
# import spatialmath as sp
from spatialmath import SE3
from spatialmath.base.argcheck import getvector, verifymatrix
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
    # TODO doco for ee_link, can be a list

    def __init__(
            self,
            elinks,
            base_link=None,
            gripper_links=None,
            **kwargs
            ):

        self._ets = []
        self._linkdict = {}
        self._n = 0
        self._ee_links = []
        self._base_link = None

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
            # were passed a list of ELinks

            # check all the incoming ELink objects
            n = 0
            for link in elinks:
                if isinstance(link, ELink):
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

            # visit all links in DFS order
            self.dfs_links(
                self.base_link, lambda link: visit_link(link, jindex))

        elif all([link.jindex is not None for link in elinks]):
            # jindex set on all, check they are unique and sequential
            jset = set(range(self._n))
            for link in elinks:
                if link.jindex not in jset:
                    raise ValueError(
                        'joint index {link.jindex} was '
                        'repeated or out of range')
                jset -= set(link.jindex)
            if len(jset) > 0:
                raise ValueError('joints {jset} were not assigned')
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

        super().__init__(elinks, **kwargs)

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

    def dfs_path(self, l1, l2):
        path = []
        visited = [l1]

        def vis_children(link):
            visited.append(link)

            for li in link.child:
                if li not in visited:

                    if li == l2 or vis_children(li):
                        path.append(li)
                        return True
        vis_children(l1)
        path.append(l1)
        path.reverse()
        return path

    def to_dict(self):
        ob = {
            'links': [],
            'name': self.name,
            'n': self.n
        }

        self.fkine_all()

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

        self.fkine_all()

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
        else:
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
            raise ValueError('must be an ELink')
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
            raise ValueError('expecting an ELink or list of ELinks')
        # self._reset_fk_path()
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

    def ets(self, ee=None):
        if ee is None:
            if len(self.ee_links) == 1:
                link = self.ee_links[0]
            else:
                raise ValueError(
                    'robot has multiple end-effectors, specify one')
        # elif isinstance(ee, str) and ee in self._linkdict:
        #     ee = self._linkdict[ee]
        elif isinstance(ee, ELink) and ee in self._links:
            link = ee
        else:
            raise ValueError('end-effector is not valid')

        ets = ETS()

        # build the ETS string from ee back to root
        while link is not None:
            ets = link.ets() * ets
            link = link.parent

        return ets

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

    # def fkine(self, q=None):
    #     '''
    #     Evaluates the forward kinematics of a robot based on its ETS and
    #     joint angles q.

    #     T = fkine(q) evaluates forward kinematics for the robot at joint
    #     configuration q.

    #     T = fkine() as above except uses the stored q value of the
    #     robot object.

    #     Trajectory operation:
    #     Calculates fkine for each point on a trajectory of joints q where
    #     q is (nxm) and the returning SE3 in (m)

    #     :param q: The joint angles/configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :return: The transformation matrix representing the pose of the
    #         end-effector
    #     :rtype: SE3

    #     :notes:
    #         - The robot's base or tool transform, if present, are incorporated
    #           into the result.

    #     :references:
    #         - Kinematic Derivatives using the Elementary Transform
    #           Sequence, J. Haviland and P. Corke

    #     '''

    #     trajn = 1

    #     if q is None:
    #         q = self.q

    #     try:
    #         q = getvector(q, self.n, 'col')
    #     except ValueError:
    #         trajn = q.shape[1]
    #         verifymatrix(q, (self.n, trajn))

    #     for i in range(trajn):
    #         j = 0
    #         tr = self.base

    #         for link in self._fkpath:
    #             if link.isjoint:
    #                 T = link.A(q[j, i])
    #                 j += 1
    #             else:
    #                 T = link.A()

    #             tr = tr * T

    #         tr = tr * self.tool

    #         if i == 0:
    #             t = SE3(tr)
    #         else:
    #             t.append(tr)

    #     return t

    def fkine(self, q=None, from_link=None, to_link=None):

        if from_link is None:
            from_link = self.base_link

        if to_link is None:
            to_link = self.ee_links[0]

        trajn = 1

        if q is None:
            q = self.q

        path, n = self.get_path(from_link, to_link)

        use_jindex = True

        try:
            q = getvector(q, self.n, 'col')

        except ValueError:
            try:
                q = getvector(q, n, 'col')
                use_jindex = False
                j = 0
            except ValueError:
                trajn = q.shape[1]
                verifymatrix(q, (self.n, trajn))

        for i in range(trajn):
            tr = self.base.A
            for link in path:
                if link.isjoint:
                    if use_jindex:
                        T = link.A(q[link.jindex, i], fast=True)
                    else:
                        T = link.A(q[j, i], fast=True)
                        j += 1
                else:
                    T = link.A(fast=True)

                tr = tr @ T

            if i == 0:
                t = SE3(tr)
            else:
                t.append(SE3(tr))

        return t

    def fkine_all(self, q=None):
        '''
        Tall = fkine_all(q) evaluates fkine for each joint within a robot and
        returns a trajecotry of poses.

        Tall = fkine_all() as above except uses the stored q value of the
        robot object.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :return T: Homogeneous transformation trajectory
        :rtype T: SE3 list

        :notes:
            - The robot's base transform, if present, are incorporated
              into the result.

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke

        '''

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        for link in self.links:
            if link.isjoint:
                t = link.A(q[link.jindex])
            else:
                t = link.A()

            if link.parent is None:
                link._fk = self.base * t
            else:
                link._fk = link.parent._fk * t

            # Update the collision objects transform as well
            for col in link.collision:
                col.wT = link._fk

            for gi in link.geometry:
                gi.wT = link._fk

        # Do the grippers now
        for gripper in self.grippers:
            for link in gripper.links:
                # print(link.jindex)
                if link.isjoint:
                    t = link.A(gripper.q[link.jindex])
                else:
                    t = link.A()

                link._fk = link.parent._fk * t

                # Update the collision objects transform as well
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

    def get_path(self, from_link, to_link):
        path = []
        n = 0
        link = to_link

        path.append(link)
        if link.isjoint:
            n += 1

        while link != from_link:
            link = link.parent
            path.append(link)
            if link.isjoint:
                n += 1

        path.reverse()

        return path, n

    def jacob0(
            self, q=None, from_link=None, to_link=None,
            offset=None, T=None):

        if from_link is None:
            from_link = self.base_link

        if to_link is None:
            to_link = self.ee_links[0]

        if offset is None:
            offset = SE3()

        path, n = self.get_path(from_link, to_link)

        if q is None:
            q = np.copy(self.q)
        else:
            try:
                q = getvector(q, n)
            except ValueError:
                q = getvector(q, self.n)

        if T is None:
            T = (self.base.inv()
                 * self.fkine(q, from_link=from_link, to_link=to_link)
                 * offset)

        T = T.A
        U = np.eye(4)
        j = 0
        J = np.zeros((6, n))

        for link in path:

            if link.isjoint:
                U = U @ link.A(q[j], fast=True)

                if link == to_link:
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

    def jacobe(self, q=None, from_link=None, to_link=None, offset=None):
        """
        Je = jacobe(q) is the manipulator Jacobian matrix which maps joint
        velocity to end-effector spatial velocity. v = Je*qd in the
        end-effector frame.

        Je = jacobe() as above except uses the stored q value of the
        robot object.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :return J: The manipulator Jacobian in ee frame
        :rtype: float ndarray(6,n)

        """

        if from_link is None:
            from_link = self.base_link

        if to_link is None:
            to_link = self.ee_links[0]

        if offset is None:
            offset = SE3()

        if q is None:
            q = np.copy(self.q)
        # else:
        #     q = getvector(q, n)

        T = (self.base.inv()
             * self.fkine(q, from_link=from_link, to_link=to_link)
             * offset)

        J0 = self.jacob0(q, from_link, to_link, offset, T)
        Je = self.jacobev(q, from_link, to_link, offset, T) @ J0
        return Je

    def hessian0(self, q=None, J0=None, from_link=None, to_link=None):
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
        :return: The manipulator Hessian in 0 frame
        :rtype: float ndarray(6,n,n)

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        if from_link is None:
            from_link = self.base_link

        if to_link is None:
            to_link = self.ee_links[0]

        path, n = self.get_path(from_link, to_link)

        if J0 is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, n)

            J0 = self.jacob0(q, from_link, to_link)
        else:
            verifymatrix(J0, (6, n))

        H = np.zeros((6, n, n))

        for j in range(n):
            for i in range(j, n):

                H[:3, i, j] = np.cross(J0[3:, j], J0[:3, i])
                H[3:, i, j] = np.cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[:3, j, i] = H[:3, i, j]

        return H

    def manipulability(self, q=None, J=None, from_link=None, to_link=None):
        """
        Calculates the manipulability index (scalar) robot at the joint
        configuration q. It indicates dexterity, that is, how isotropic the
        robot's motion is with respect to the 6 degrees of Cartesian motion.
        The measure is high when the manipulator is capable of equal motion
        in all directions and low when the manipulator is close to a
        singularity. One of J or q is required. Supply J if already
        calculated to save computation time

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param J: The manipulator Jacobian in any frame
        :type J: float ndarray(6,n)
        :return: The manipulability index
        :rtype: float

        :references:
            - Analysis and control of robot manipulators with redundancy,
              T. Yoshikawa,
            - Robotics Research: The First International Symposium (M. Brady
              and R. Paul, eds.), pp. 735-747, The MIT press, 1984.

        """

        if from_link is None:
            from_link = self.base_link

        if to_link is None:
            to_link = self.ee_links[0]

        path, n = self.get_path(from_link, to_link)

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, n)

            J = self.jacob0(q, from_link, to_link)
        else:
            verifymatrix(J, (6, n))

        return np.sqrt(np.linalg.det(J @ np.transpose(J)))

    def jacobm(self, q=None, J=None, H=None, from_link=None, to_link=None):
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
        :return: The manipulability Jacobian
        :rtype: float ndarray(n)

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        if from_link is None:
            from_link = self.base_link

        if to_link is None:
            to_link = self.ee_links[0]

        path, n = self.get_path(from_link, to_link)

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, n)

            J = self.jacob0(q, from_link, to_link)
        else:
            verifymatrix(J, (6, n))

        if H is None:
            H = self.hessian0(J0=J, from_link=from_link, to_link=to_link)
        else:
            verifymatrix(H, (6, n, n))

        manipulability = self.manipulability(
            J=J, from_link=from_link, to_link=to_link)
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
            Column("ETS", headalign="^", colalign=">"),
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
            self, q=None, from_link=None, to_link=None,
            offset=None, T=None):
        """
        Jv = jacobev(q) is the spatial velocity Jacobian, at joint
        configuration q, which relates the velocity in the base frame to the
        velocity in the end-effector frame.

        Jv = jacobev() as above except uses the stored q value of the
        robot object.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :returns J: The velocity Jacobian in ee frame
        :rtype J: float ndarray(6,6)

        """

        if from_link is None:
            from_link = self.base_link

        if to_link is None:
            to_link = self.ee_links[0]

        if offset is None:
            offset = SE3()

        if T is None:
            r = (self.base.inv() * self.fkine(
                    q, from_link, to_link) * offset).R
            r = np.linalg.inv(r)
        else:
            r = np.linalg.inv(T.R)

        Jv = np.zeros((6, 6))
        Jv[:3, :3] = r
        Jv[3:, 3:] = r

        return Jv

    def jacob0v(self, q=None):
        """
        Jv = jacob0v(q) is the spatial velocity Jacobian, at joint
        configuration q, which relates the velocity in the end-effector frame
        to velocity in the base frame

        Jv = jacob0v() as above except uses the stored q value of the
        robot object.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :returns J: The velocity Jacobian in 0 frame
        :rtype J: float ndarray(6,6)

        """

        r = (self.base.inv() * self.fkine(q)).R

        Jv = np.zeros((6, 6))
        Jv[:3, :3] = r
        Jv[3:, 3:] = r

        return Jv

    def joint_velocity_damper(self, ps=0.05, pi=0.1, n=None, gain=1.0):
        '''
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
        '''

        if n is None:
            n = self.n

        Ain = np.zeros((n, n))
        Bin = np.zeros(n)

        for i in range(n):
            if self.q[i] - self.qlim[0, i] <= pi:
                Bin[i] = -gain * (
                    ((self.qlim[0, i] - self.q[i]) + ps) / (pi - ps))
                Ain[i, i] = -1
            if self.qlim[1, i] - self.q[i] <= pi:
                Bin[i] = gain * (
                    (self.qlim[1, i] - self.q[i]) - ps) / (pi - ps)
                Ain[i, i] = 1

        return Ain, Bin

    def link_collision_damper(
            self, shape, q=None, di=0.3, ds=0.05, xi=1.0,
            from_link=None, to_link=None):

        if from_link is None:
            from_link = self.base_link

        if to_link is None:
            to_link = self.ee_link

        links, n = self.get_path(from_link, to_link)

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
                    q, from_link=self.base_link, to_link=link,
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
                                                link, link_col, q[:j])

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

    def closest_point(self, shape, inf_dist=1.0):
        '''
        closest_point(shape, inf_dist) returns the minimum euclidean
        distance between this robot and shape, provided it is less than
        inf_dist. It will also return the points on self and shape in the
        world frame which connect the line of length distance between the
        shapes. If the distance is negative then the shapes are collided.
        :param shape: The shape to compare distance to
        :type shape: Shape
        :param inf_dist: The minimum distance within which to consider
            the shape
        :type inf_dist: float
        :returns: d, p1, p2 where d is the distance between the shapes,
            p1 and p2 are the points in the world frame on the respective
            shapes
        :rtype: float, SE3, SE3
        '''

        d = 10000
        p1 = None,
        p2 = None

        for link in self.links:
            td, tp1, tp2 = link.closest_point(shape, inf_dist)

            if td is not None and td < d:
                d = td
                p1 = tp1
                p2 = tp2

        if d == 10000:
            d = None

        return d, p1, p2

    def collided(self, shape):
        '''
        collided(shape) checks if this robot and shape have collided
        :param shape: The shape to compare distance to
        :type shape: Shape
        :returns: True if shapes have collided
        :rtype: bool
        '''

        for link in self.links:
            if link.collided(shape):
                return True

        return False

    # def teach(
    #         self, block=True, q=None, limits=None,
    #         jointaxes=True, eeframe=True, shadow=True, name=True):
    #     '''
    #     Graphical teach pendant

    #     env = teach() creates a matplotlib plot which allows the user to
    #     "drive" a graphical robot using a graphical slider panel. The
    #     robot's inital joint configuration is robot.q. This will block the
    #     programs execution. The plot will autoscale with an aspect ratio of 1.  # noqa

    #     env = teach(q) as above except the robot's initial configuration is
    #     set to q.

    #     env = teach(block=False) as avove except the plot is non-blocking. Note  # noqa
    #     that the plot will exit when the python script finishes executing.

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param q: The joint angles/configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param limits: Custom view limits for the plot. If not supplied will
    #         autoscale, [x1, x2, y1, y2, z1, z2]
    #     :type limits: ndarray(6)
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

    #     :retrun: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     :notes:
    #         - The slider limits are derived from the joint limit properties.
    #           If not set then
    #             - For revolute joints they are assumed to be [-pi, +pi]
    #             - For prismatic joint they are assumed unknown and an error
    #               occurs.

    #     '''

    #     if q is not None:
    #         self.q = q

    #     # try:
    #     return _teach(
    #         self, block, limits=limits,
    #         jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)
    #     # except ModuleNotFoundError:
    #     #     print(
    #     #         'Could not find matplotlib.'
    #     #         ' Matplotlib required for this function')

    # def plot(
    #         self, block=True, q=None, dt=50, limits=None,
    #         vellipse=False, fellipse=False,
    #         jointaxes=True, eeframe=True, shadow=True, name=True):
    #     '''
    #     Graphical display and animation

    #     env = plot() displays a graphical view of a robot based on the
    #     kinematic model, at it's stored q value. A stick figure polyline
    #     joins the origins of the link coordinate frames. This method will be
    #     blocking. The plot will autoscale with an aspect ratio of 1.

    #     env = plot(q) as above except the robot is plotted with joint angles q  # noqa

    #     env = plot(block=False) as avove except the plot in non-blocking. Note  # noqa
    #     that the plot will exit when the python script finishes executing.

    #     env = plot(q, dt) as above except q is an nxm trajectory of joint
    #     angles. This creates an animation of the robot moving through the
    #     trajectories with a gap dt milliseconds in between.

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param q: The joint angles/configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param dt: if q is a trajectory, this describes the delay in
    #         milliseconds between frames
    #     :type dt: int
    #     :param limits: Custom view limits for the plot. If not supplied will
    #         autoscale, [x1, x2, y1, y2, z1, z2]
    #     :type limits: ndarray(6)
    #     :param vellipse: (Plot Option) Plot the velocity ellipse at the
    #         end-effector
    #     :type vellipse: bool
    #     :param vellipse: (Plot Option) Plot the force ellipse at the
    #         end-effector
    #     :type vellipse: bool
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

    #     :retrun: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     '''

    #     # try:
    #     return _plot(
    #         self, block, q, dt, limits,
    #         vellipse=vellipse, fellipse=fellipse,
    #         jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)
    #     # except ModuleNotFoundError:
    #     #     print(
    #     #         'Could not find matplotlib.'
    #     #         ' Matplotlib required for this function')

    # def plot2(
    #         self, block=True, q=None, dt=50, limits=None,
    #         vellipse=False, fellipse=False,
    #         eeframe=True, name=False):
    #     '''
    #     2D Graphical display and animation

    #     env = plot2() displays a 2D graphical view of a robot based on the
    #     kinematic model, at it's stored q value. A stick figure polyline
    #     joins the origins of the link coordinate frames. This method will be
    #     blocking. The plot will autoscale with an aspect ratio of 1.

    #     env = plot2(q) as above except the robot is plotted with joint angles q  # noqa

    #     env = plot2(block=False) as avove except the plot in non-blocking. Note  # noqa
    #     that the plot will exit when the python script finishes executing.

    #     env = plot2(q, dt) as above except q is an nxm trajectory of joint
    #     angles. This creates an animation of the robot moving through the
    #     trajectories with a gap dt milliseconds in between.

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param q: The joint angles/configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param dt: if q is a trajectory, this describes the delay in
    #         milliseconds between frames
    #     :type dt: int
    #     :param limits: Custom view limits for the plot. If not supplied will
    #         autoscale, [x1, x2, y1, y2, z1, z2]
    #     :type limits: ndarray(6)
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

    #     :retrun: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     '''

    #     # try:
    #     return _plot2(
    #         self, block, q, dt, limits,
    #         vellipse=vellipse, fellipse=fellipse,
    #         eeframe=eeframe, name=name)
    #     # except ModuleNotFoundError:
    #     #     print(
    #     #         'Could not find matplotlib.'
    #     #         ' Matplotlib required for this function')

    # def teach2(
    #         self, block=True, q=None, limits=None,
    #         eeframe=True, name=False):
    #     '''
    #     2D Graphical teach pendant

    #     env = teach2() creates a 2D matplotlib plot which allows the user to
    #     "drive" a graphical robot using a graphical slider panel. The
    #     robot's inital joint configuration is robot.q. This will block the
    #     programs execution. The plot will autoscale with an aspect ratio of 1.  # noqa

    #     env = teach2(q) as above except the robot's initial configuration is
    #     set to q.

    #     env = teach2(block=False) as avove except the plot is non-blocking.
    #     Note that the plot will exit when the python script finishes
    #     executing.

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param q: The joint angles/configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param limits: Custom view limits for the plot. If not supplied will
    #         autoscale, [x1, x2, y1, y2, z1, z2]
    #     :type limits: ndarray(6)
    #     :param eeframe: (Plot Option) Plot the end-effector coordinate frame
    #         at the location of the end-effector. Uses three arrows, red,
    #         green and blue to indicate the x, y, and z-axes.
    #     :type eeframe: bool
    #     :param name: (Plot Option) Plot the name of the robot near its base
    #     :type name: bool

    #     :retrun: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     :notes:
    #         - The slider limits are derived from the joint limit properties.
    #           If not set then
    #             - For revolute joints they are assumed to be [-pi, +pi]
    #             - For prismatic joint they are assumed unknown and an error
    #               occurs.

    #     '''

    #     if q is not None:
    #         self.q = q

    #     # try:
    #     return _teach2(
    #         self, block, limits=limits,
    #         eeframe=eeframe, name=name)
    #     # except ModuleNotFoundError:
    #     #     print(
    #     #         'Could not find matplotlib.'
    #     #         ' Matplotlib required for this function')

    # def vellipse(self, q=None, opt='trans', centre=[0, 0, 0]):
    #     '''
    #     Create a velocity ellipsoid object for plotting

    #     env = vellipse() creates a velocity ellipsoid for the robot at
    #     pose robot.q. The ellipsoid is centered at the origin.

    #     env = vellipse(q) as above except the robot is plotted with joint
    #     angles q

    #     env = vellipse(opt) as above except opt is 'trans' or 'rot' will
    #     plot either the translational or rotational velocity ellipsoid.

    #     env = vellipse(centre) as above except centre is either a 3
    #     vector or 'ee' which is the centre location of the ellipsoid

    #     :param q: The joint angles/configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param opt: 'trans' or 'rot' will plot either the translational or
    #         rotational velocity ellipsoid
    #     :type opt: string
    #     :param centre:
    #     :type centre: list or str('ee')

    #     :retrun: An EllipsePlot object
    #     :rtype: EllipsePlot

    #     '''

    #     return _vellipse(self, q=q, opt=opt, centre=centre)

    # def fellipse(self, q=None, opt='trans', centre=[0, 0, 0]):
    #     '''
    #     Create a force ellipsoid object for plotting

    #     env = fellipse() creates a force ellipsoid for the robot at
    #     pose robot.q. The ellipsoid is centered at the origin.

    #     env = fellipse(q) as above except the robot is plotted with joint
    #     angles q

    #     env = fellipse(opt) as above except opt is 'trans' or 'rot' will
    #     plot either the translational or rotational force ellipsoid.

    #     env = fellipse(centre) as above except centre is either a 3
    #     vector or 'ee' which is the centre location of the ellipsoid

    #     :param q: The joint angles/configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)
    #     :param opt: 'trans' or 'rot' will plot either the translational or
    #         rotational force ellipsoid
    #     :type opt: string
    #     :param centre:
    #     :type centre: list or str('ee')

    #     :retrun: An EllipsePlot object
    #     :rtype: EllipsePlot

    #     '''

    #     return _fellipse(self, q=q, opt=opt, centre=centre)

    # def plot_vellipse(
    #         self, block=True, q=None, vellipse=None,
    #         limits=None, opt='trans', centre=[0, 0, 0],
    #         jointaxes=True, eeframe=True, shadow=True, name=True):
    #     '''
    #     Plot the velocity ellipsoid for seriallink manipulator

    #     env = plot_vellipse() displays the velocity ellipsoid for the robot at  # noqa
    #     pose robot.q. The ellipsoid is centered at the origin. This method
    #     will be blocking. The plot will autoscale with an aspect ratio of 1.

    #     env = plot_vellipse(block=False) as avove except the plot in
    #     non-blocking. Note that the plot will exit when the python script
    #     finishes executing.

    #     env = plot_vellipse(q) as above except the robot is plotted with joint  # noqa
    #     angles q

    #     env = plot_vellipse(vellipse) specifies a custon ellipse to plot. If
    #     not supplied this function calculates the vellipse based on q

    #     env = plot_vellipse(limits) as above except the view limits of the
    #     plot are set by limits.

    #     env = plot_vellipse(opt) as above except opt is 'trans' or 'rot' will
    #     plot either the translational or rotational velocity ellipsoid.

    #     env = plot_vellipse(centre) as above except centre is either a 3
    #     vector or 'ee' which is the centre location of the ellipsoid

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param q: The joint angles/configuration of the robot (Optional,
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
    #     :param centre: The coordinates to plot the vellipse [x, y, z] or 'ee'
    #         to plot at the end-effector location
    #     :type centre: list or str('ee')
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

    #     :retrun: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     '''

    #     if q is not None:
    #         self.q = q

    #     if vellipse is None:
    #         vellipse = self.vellipse(q=q, opt=opt, centre=centre)

    #     return _plot_ellipse(
    #         vellipse, block, limits,
    #         jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

    # def plot_fellipse(
    #         self, block=True, q=None, fellipse=None,
    #         limits=None, opt='trans', centre=[0, 0, 0],
    #         jointaxes=True, eeframe=True, shadow=True, name=True):
    #     '''
    #     Plot the force ellipsoid for seriallink manipulator

    #     env = plot_fellipse() displays the force ellipsoid for the robot at
    #     pose robot.q. The ellipsoid is centered at the origin. This method
    #     will be blocking. The plot will autoscale with an aspect ratio of 1.

    #     env = plot_fellipse(block=False) as avove except the plot in
    #     non-blocking. Note that the plot will exit when the python script
    #     finishes executing.

    #     env = plot_fellipse(q) as above except the robot is plotted with joint  # noqa
    #     angles q

    #     env = plot_fellipse(fellipse) specifies a custon ellipse to plot. If
    #     not supplied this function calculates the fellipse based on q

    #     env = plot_fellipse(limits) as above except the view limits of the
    #     plot are set by limits.

    #     env = plot_fellipse(opt) as above except opt is 'trans' or 'rot' will
    #     plot either the translational or rotational force ellipsoid.

    #     env = plot_fellipse(centre) as above except centre is either a 3
    #     vector or 'ee' which is the centre location of the ellipsoid

    #     :param block: Block operation of the code and keep the figure open
    #     :type block: bool
    #     :param q: The joint angles/configuration of the robot (Optional,
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
    #     :param centre: The coordinates to plot the fellipse [x, y, z] or 'ee'
    #         to plot at the end-effector location
    #     :type centre: list or str('ee')
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

    #     :retrun: A reference to the PyPlot object which controls the
    #         matplotlib figure
    #     :rtype: PyPlot

    #     '''

    #     if q is not None:
    #         self.q = q

    #     if fellipse is None:
    #         fellipse = self.fellipse(q=q, opt=opt, centre=centre)

    #     return _plot_ellipse(
    #         fellipse, block, limits,
    #         jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)


if __name__ == "__main__":

    import roboticstoolbox as rtb
    np.set_printoptions(precision=4, suppress=True)

    p=rtb.models.URDF.Panda()
    print(p[1].m)

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
