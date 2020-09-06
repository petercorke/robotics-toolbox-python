#!/usr/bin/env python3
"""
Created on Tue Apr 24 15:48:52 2020
@author: Jesse Haviland
"""

from os.path import splitext
import numpy as np
# import spatialmath as sp
from spatialmath import SE3
from spatialmath.base.argcheck import getvector, verifymatrix
from spatialmath.base import tr2rpy, r2q
from ropy.robot.ELink import ELink
from ropy.backend.PyPlot.functions import \
    _plot, _teach, _fellipse, _vellipse, _plot_ellipse, \
    _plot2, _teach2
from ropy.backend import xacro
from ropy.backend import URDF


class ETS(object):
    """
    The Elementary Transform Sequence (ETS). A superclass which represents the
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

    def __init__(
            self,
            elinks,
            name='noname',
            base_link=None,
            ee_link=None,
            manufacturer='',
            base=None,
            tool=None,
            gravity=np.array([0, 0, 9.81])):

        super(ETS, self).__init__()

        if base is None:
            self.base = SE3()
        else:
            self.base = base

        self.tool = SE3()
        self.gravity = gravity

        # Verify elinks
        if not isinstance(elinks, list):
            raise TypeError('The links must be stored in a list.')

        self._ets = []
        self._n = 0
        self._M = 0
        self._q_idx = []

        # Set up a dictionary for looking up links by name
        for link in elinks:
            if isinstance(link, ELink):
                self._M += 1
            else:
                raise TypeError("Input can be only ELink")

        # Set up references between links, a bi-directional linked list
        # Also find the top of the tree
        self.root = []
        self.end = []
        for link in elinks:
            for li in link.parent:
                li._child.append(link)

            if len(link.parent) == 0:
                self.root.append(link)

        # Find the bottom of the tree
        for link in elinks:
            if len(link.child) == 0:
                self.end.append(link)

        # If the base link is not defined, use the root of the tree
        if base_link is None:
            self._base_link = self.root[0]   # Needs to be private attrib
        else:
            self._base_link = base_link      # Needs to be private attrib

        # If the ee link is not defined, use the bottom of the tree
        if ee_link is None:
            self._ee_link = self.end[-1]
        else:
            self._ee_link = ee_link

        def add_links(link, lst, q_idx):

            if link.jtype == link.VARIABLE:
                q_idx.append(len(lst))

            lst.append(link)

        # Figure out the order of links with resepect to joint variables
        self.bfs_link(
            lambda link: add_links(link, self._ets, self._q_idx))
        self._n = len(self._q_idx)

        self._reset_fk_path()
        self.name = name
        self.manuf = manufacturer

        # Current joint angles of the robot
        self.q = np.zeros(self.n)
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)
        self.control_type = 'v'

    def _reset_fk_path(self):
        # Pre-calculate the forward kinematics path
        self._fkpath = self.dfs_path(self.base_link, self.ee_link)

    def bfs_link(self, func):
        queue = self.root

        for li in queue:
            func(li)

        def vis_children(link):
            for li in link.child:
                if li not in queue:
                    queue.append(li)
                    func(li)

        while len(queue) > 0:
            link = queue.pop(0)
            vis_children(link)

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
            'n': self.n,
            'M': self.M,
            'q_idx': self.q_idx
        }

        self.allfkine()

        for link in self.ets:
            li = {
                'axis': [],
                'eta': [],
                'q_idx': link.q_idx,
                'geometry': [],
                't': link._fk.t.tolist(),
                'q': r2q(link._fk.R).tolist()
            }

            for et in link.ets:
                li['axis'].append(et.axis)
                li['eta'].append(et.eta)

            for gi in link.geometry:
                g_fk = link._fk * gi.base
                if gi.scale is not None:
                    scale = gi.scale.tolist()
                else:
                    scale = [1, 1, 1]
                li['geometry'].append({
                    'filename': gi.filename,
                    'scale': scale,
                    't': g_fk.t.tolist(),
                    'q': r2q(g_fk.R).tolist()
                })

            ob['links'].append(li)

        return ob

    def fk_dict(self):
        ob = {
            'links': []
        }

        self.allfkine()
        # print(Tall)

        for link in self.ets:

            li = {
                't': link._fk.t.tolist(),
                'q': r2q(link._fk.R).tolist()
            }

            ob['links'].append(li)

        return ob

    # @classmethod
    # def urdf_to_ets(cls, file_path):
    #     name, ext = splitext(file_path)

    #     if ext == '.xacro':
    #         urdf_string = xacro.main(file_path)
    #         urdf = URDF.loadstr(urdf_string, file_path)

    #     return ETS(
    #         urdf.elinks,
    #         name=urdf.name
    #     )

    @staticmethod
    def urdf_to_ets_args(file_path):
        name, ext = splitext(file_path)

        if ext == '.xacro':
            urdf_string = xacro.main(file_path)
            urdf = URDF.loadstr(urdf_string, file_path)

        return urdf.elinks, urdf.name

    # @classmethod
    # def dh_to_ets(cls, robot):
    #     """
    #     Converts a robot modelled with standard or modified DH parameters to an
    #     ETS representation

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

        for i in range(self.M):
            if self.ets[i].jtype == self.ets[i].VARIABLE:
                v[:, j] = self.ets[i].qlim
                j += 1

        return v

    @property
    def base_link(self):
        return self._base_link

    @property
    def ee_link(self):
        return self._ee_link

    @property
    def q(self):
        return self._q

    @property
    def qd(self):
        return self._qd

    @property
    def qdd(self):
        return self._qdd

    @property
    def control_type(self):
        return self._control_type

    @property
    def ets(self):
        return self._ets

    @property
    def name(self):
        return self._name

    @property
    def manuf(self):
        return self._manuf

    @property
    def base(self):
        return self._base

    @property
    def tool(self):
        return self._tool

    @property
    def n(self):
        return self._n

    @property
    def M(self):
        return self._M

    @property
    def q_idx(self):
        return self._q_idx

    @property
    def gravity(self):
        return self._gravity

    @name.setter
    def name(self, name_new):
        self._name = name_new

    @manuf.setter
    def manuf(self, manuf_new):
        self._manuf = manuf_new

    @gravity.setter
    def gravity(self, gravity_new):
        self._gravity = getvector(gravity_new, 3, 'col')

    @q.setter
    def q(self, q_new):
        q_new = getvector(q_new, self.n)
        self._q = q_new

    @qd.setter
    def qd(self, qd_new):
        self._qd = getvector(qd_new, self.n)

    @qdd.setter
    def qdd(self, qdd_new):
        self._qdd = getvector(qdd_new, self.n)

    @control_type.setter
    def control_type(self, cn):
        if cn == 'p' or cn == 'v' or cn == 'a':
            self._control_type = cn
        else:
            raise ValueError(
                'Control type must be one of \'p\', \'v\', or \'a\'')

    @base.setter
    def base(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._base = T

    @tool.setter
    def tool(self, T):
        if not isinstance(T, SE3): # pragma nocover
            T = SE3(T)
        self._tool = T

    @base_link.setter
    def base_link(self, link):
        if isinstance(link, ELink):
            self._base_link = link
        else:
            self._base_link = self.ets[link]
        self._reset_fk_path()

    @ee_link.setter
    def ee_link(self, link):
        if isinstance(link, ELink):
            self._ee_link = link
        else:
            self._ee_link = self.ets[link]
        self._reset_fk_path()

    def fkine(self, q=None):
        '''
        Evaluates the forward kinematics of a robot based on its ETS and
        joint angles q.

        T = fkine(q) evaluates forward kinematics for the robot at joint
        configuration q.

        T = fkine() as above except uses the stored q value of the
        robot object.

        Trajectory operation:
        Calculates fkine for each point on a trajectory of joints q where
        q is (nxm) and the returning SE3 in (m)

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :return: The transformation matrix representing the pose of the
            end-effector
        :rtype: SE3

        :notes:
            - The robot's base or tool transform, if present, are incorporated
              into the result.

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke

        '''

        trajn = 1

        if q is None:
            q = self.q

        try:
            q = getvector(q, self.n, 'col')
        except ValueError:
            trajn = q.shape[1]
            verifymatrix(q, (self.n, trajn))

        for i in range(trajn):
            j = 0
            tr = self.base

            for link in self._fkpath:
                if link.jtype == link.VARIABLE:
                    T = link.A(q[j, i])
                    j += 1
                else:
                    T = link.A()

                tr = tr * T

            tr = tr * self.tool

            if i == 0:
                t = SE3(tr)
            else:
                t.append(tr)

        return t

    def allfkine(self, q=None):
        '''
        Tall = allfkine(q) evaluates fkine for each joint within a robot and
        returns a trajecotry of poses.

        Tall = allfkine() as above except uses the stored q value of the
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

        # t = self.base
        # Tall = SE3()
        j = 0

        if self.ets[0].jtype == self.ets[0].VARIABLE:
            self.ets[0]._fk = self.base * self.ets[0].A(q[j])
            j += 1
        else:
            self.ets[0]._fk = self.base * self.ets[0].A()

        for i in range(1, self.M):
            if self.ets[i].jtype == self.ets[i].VARIABLE:
                t = self.ets[i].A(q[j])
                j += 1
            else:
                t = self.ets[i].A()

            self.ets[i]._fk = self.ets[i].parent[0]._fk * t

    def jacob0(self, q=None):
        """
        J0 = jacob0(q) is the manipulator Jacobian matrix which maps joint
        velocity to end-effector spatial velocity. v = J0*qd in the
        base frame.

        J0 = jacob0() as above except uses the stored q value of the
        robot object.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :return J: The manipulator Jacobian in ee frame
        :rtype: float ndarray(6,n)

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        T = (self.base.inv() * self.fkine(q)).A
        U = np.eye(4)
        j = 0
        J = np.zeros((6, self.n))

        for link in self._fkpath:

            for k in range(link.M):

                if k != link.q_idx:
                    U = U @ link.ets[k].T().A
                else:
                    if link.ets[k]._axis == 'Rz':
                        U = U @ link.ets[k].T(q[j]).A
                        Tu = np.linalg.inv(U) @ T

                        n = U[:3, 0]
                        o = U[:3, 1]
                        a = U[:3, 2]
                        y = Tu[1, 3]
                        x = Tu[0, 3]

                        J[:3, j] = (o * x) - (n * y)
                        J[3:, j] = a

                        j += 1
                    if link.ets[k]._axis == 'Ry':
                        U = U @ link.ets[k].T(q[j]).A
                        Tu = np.linalg.inv(U) @ T

                        n = U[:3, 0]
                        o = U[:3, 1]
                        a = U[:3, 2]
                        z = Tu[2, 3]
                        x = Tu[0, 3]

                        J[:3, j] = (n * z) - (a * x)
                        J[3:, j] = o

                        j += 1
                    if link.ets[k]._axis == 'Rx':
                        U = U @ link.ets[k].T(q[j]).A
                        Tu = np.linalg.inv(U) @ T

                        n = U[:3, 0]
                        o = U[:3, 1]
                        a = U[:3, 2]
                        y = Tu[1, 3]
                        z = Tu[2, 3]

                        J[:3, j] = (a * y) - (o * z)
                        J[3:, j] = n

                        j += 1
                    elif link.ets[k]._axis == 'tx':
                        U = U @ link.ets[k].T(q[j]).A
                        n = U[:3, 0]

                        J[:3, j] = n
                        J[3:, j] = np.array([0, 0, 0])

                        j += 1
                    elif link.ets[k]._axis == 'ty':
                        U = U @ link.ets[k].T(q[j]).A
                        o = U[:3, 1]

                        J[:3, j] = o
                        J[3:, j] = np.array([0, 0, 0])

                        j += 1
                    elif link.ets[k]._axis == 'tz':
                        U = U @ link.ets[k].T(q[j]).A
                        a = U[:3, 2]

                        J[:3, j] = a
                        J[3:, j] = np.array([0, 0, 0])

                        j += 1

        return J

    def jacobe(self, q=None):
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

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        J0 = self.jacob0(q)
        Je = self.jacobev(q) @ J0
        return Je

    def hessian0(self, q=None, J0=None):
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

        if J0 is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J0 = self.jacob0(q)
        else:
            verifymatrix(J0, (6, self.n))

        H = np.zeros((6, self.n, self.n))

        for j in range(self.n):
            for i in range(j, self.n):

                H[:3, i, j] = np.cross(J0[3:, j], J0[:3, i])
                H[3:, i, j] = np.cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[:3, j, i] = H[:3, i, j]

        return H

    def manipulability(self, q=None, J=None):
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

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q)
        else:
            verifymatrix(J, (6, self.n))

        return np.sqrt(np.linalg.det(J @ np.transpose(J)))

    def jacobm(self, q=None, J=None, H=None):
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

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q)
        else:
            verifymatrix(J, (6, self.n))

        if H is None:
            H = self.hessian0(J0=J)
        else:
            verifymatrix(H, (6, self.n, self.n))

        manipulability = self.manipulability(J=J)
        b = np.linalg.inv(J @ np.transpose(J))
        Jm = np.zeros((self.n, 1))

        for i in range(self.n):
            c = J @ np.transpose(H[:, :, i])
            Jm[i, 0] = manipulability * \
                np.transpose(c.flatten('F')) @ b.flatten('F')

        return Jm

    # def __str__(self):
    #     """
    #     Pretty prints the ETS Model of the robot. Will output angles in degrees

    #     :return: Pretty print of the robot model
    #     :rtype: str
    #     """
    #     axes = ''

    #     for i in range(self.n):
    #         axes += self.ets[self.q_idx[i]].axis

    #     rpy = tr2rpy(self.tool.A, unit='deg')

    #     for i in range(3):
    #         if rpy[i] == 0:
    #             rpy[i] = 0

    #     model = '\n%s (%s): %d axis, %s, ETS\n'\
    #         'Elementary Transform Sequence:\n'\
    #         '%s\n'\
    #         'tool:  t = (%g, %g, %g),  RPY/xyz = (%g, %g, %g) deg' % (
    #             self.name, self.manuf, self.n, axes,
    #             self.ets,
    #             self.tool.A[0, 3], self.tool.A[1, 3],
    #             self.tool.A[2, 3], rpy[0], rpy[1], rpy[2]
    #         )

    #     return model

    def jacobev(self, q=None):
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

        r = (self.base.inv() * self.fkine(q)).R
        r = np.linalg.inv(r)

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

    # def teach(
    #         self, block=True, q=None, limits=None,
    #         jointaxes=True, eeframe=True, shadow=True, name=True):
    #     '''
    #     Graphical teach pendant

    #     env = teach() creates a matplotlib plot which allows the user to
    #     "drive" a graphical robot using a graphical slider panel. The
    #     robot's inital joint configuration is robot.q. This will block the
    #     programs execution. The plot will autoscale with an aspect ratio of 1.

    #     env = teach(q) as above except the robot's initial configuration is
    #     set to q.

    #     env = teach(block=False) as avove except the plot is non-blocking. Note
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

    #     env = plot(q) as above except the robot is plotted with joint angles q

    #     env = plot(block=False) as avove except the plot in non-blocking. Note
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

    #     env = plot2(q) as above except the robot is plotted with joint angles q

    #     env = plot2(block=False) as avove except the plot in non-blocking. Note
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
    #     programs execution. The plot will autoscale with an aspect ratio of 1.

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

    #     env = plot_vellipse() displays the velocity ellipsoid for the robot at
    #     pose robot.q. The ellipsoid is centered at the origin. This method
    #     will be blocking. The plot will autoscale with an aspect ratio of 1.

    #     env = plot_vellipse(block=False) as avove except the plot in
    #     non-blocking. Note that the plot will exit when the python script
    #     finishes executing.

    #     env = plot_vellipse(q) as above except the robot is plotted with joint
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

    #     env = plot_fellipse(q) as above except the robot is plotted with joint
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
