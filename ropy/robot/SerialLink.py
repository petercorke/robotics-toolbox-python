#!/usr/bin/env python

import numpy as np
from ropy.robot.Link import Link
from ropy.robot.fkine import fkine
from ropy.robot.jocobe import jacobe
from ropy.robot.jocob0 import jacob0
from spatialmath.base.argcheck import getvector, ismatrix, isscalar
import spatialmath.base as sp
from spatialmath import SE3


class SerialLink(object):
    """
    A superclass for arm type robots. A concrete class that represents a
    serial-link arm-type robot.  Each link and joint in the chain is
    described by a Link-class object using Denavit-Hartenberg parameters
    (standard or modified).

    Note: Link subclass elements passed in must be all standard, or all
          modified, DH parameters.

    :param name: Name of the robot
    :type name: string
    :param manufacturer: Manufacturer of the robot
    :type manufacturer: string
    :param base: Locaation of the base
    :type base: float np.ndarray(4,4)
    :param tool: Location of the tool
    :type tool: float np.ndarray(4,4)
    :param links: Series of links which define the robot
    :type links: List[n]
    :param mdh: 0 if standard D&H, else 1
    :type mdh: int
    :param n: Number of joints in the robot
    :type n: int
    :param T: The current pose of the robot
    :type T: float np.ndarray(4,4)
    :param q: The current joint angles of the robot
    :type q: float np.ndarray(1,n)

    Examples
    --------
    >>> L[0] = Revolute('d', 0, 'a', a1, 'alpha', np.pi/2)

    >>> L[1] = Revolute('d', 0, 'a', a2, 'alpha', 0)

    >>> twolink = SerialLink(L, 'name', 'two link');

    See Also
    --------
    ropy.robot.ets : A superclass which represents the kinematics of a
                     serial-link manipulator
    ropy.robot.Link : A link superclass for all link types
    ropy.robot.Revolute : A revolute link class

    Reference::
    - Robotics, Vision & Control, Chaps 7-9,
      P. Corke, Springer 2011.
    - Robot, Modeling & Control,
      M.Spong, S. Hutchinson & M. Vidyasagar, Wiley 2006.
    """

    def __init__(
            self,
            L,
            name='noname',
            manufacturer='',
            base=SE3(),
            tool=SE3(),
            gravity=np.array([[0], [0], [9.81]])
            ):

        self.name = name
        self.manuf = manufacturer
        self.base = base
        self.tool = tool
        self.gravity = gravity
        self._T = SE3()

        super(SerialLink, self).__init__()

        # Verify L
        if not isinstance(L, list):
            raise TypeError('The links L must be stored in a list.')

        length = len(L)
        self._links = []
        self._n = 0

        for i in range(length):
            if isinstance(L[i], Link):
                self._links.append(L[i])
                self._n += 1

            elif isinstance(L[i], SerialLink):
                for j in range(L[i].n):
                    self._links.append(L[i].links[j])
                    self._n += 1

            else:
                raise TypeError("Input can be only Link or SerialLink")

        # Current joint angles of the robot
        self.q = np.zeros(self.n)

        # Check the DH convention
        self._check_dh()

    def __add__(self, L):
        nlinks = []

        # TODO - Should I do a deep copy here a physically copy the Links
        # and not just the references?
        # Copy Link references to new list
        for i in range(self.n):
            nlinks.append(self.links[i])

        if isinstance(L, Link):
            nlinks.append(L)
        elif isinstance(L, SerialLink):
            for j in range(L.n):
                nlinks.append(L.links[j])
        else:
            raise TypeError("Can only combine SerialLinks with other "
                            "SerialLinks or Links")

        return SerialLink(
            nlinks,
            name=self.name,
            manufacturer=self.manuf,
            base=self.base,
            tool=self.tool,
            gravity=self.gravity)

    def _check_dh(self):
        self._mdh = self.links[0].mdh
        for i in range(self.n):
            if not self.links[i].mdh == self.mdh:
                raise ValueError('Robot has mixed D&H links conventions.')

    @property
    def name(self):
        return self._name

    @property
    def manuf(self):
        return self._manuf

    @property
    def links(self):
        return self._links

    @property
    def base(self):
        return self._base

    @property
    def tool(self):
        return self._tool

    @property
    def T(self):
        return self._T

    @property
    def gravity(self):
        return self._gravity

    @property
    def n(self):
        return self._n

    @property
    def mdh(self):
        return self._mdh

    @property
    def q(self):
        return self._q

    @name.setter
    def name(self, name_new):
        self._name = name_new

    @manuf.setter
    def manuf(self, manuf_new):
        self._manuf = manuf_new

    @gravity.setter
    def gravity(self, gravity_new):
        self._gravity = getvector(gravity_new, 3, 'col')

    @name.setter
    def name(self, name_new):
        self._name = name_new

    @base.setter
    def base(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._base = T

    @tool.setter
    def tool(self, T):
        if not isinstance(T, SE3):
            T = SE3(T)
        self._tool = T

    @q.setter
    def q(self, q_new):
        self._q = getvector(q_new, self.n, 'col')

    def A(self, joints, q):
        """
        Transforms between link frames for the J'th joint.  Q is a vector
        (1xN) of joint variables. For:
        - standard DH parameters, this is from frame {J-1} to frame {J}.
        - modified DH parameters, this is from frame {J} to frame {J+1}.

        Notes:
        - Base and tool transforms are not applied.

        :param joints:
        :type joints: int, tuple or 2 element list
        :param q: The joint angles/configuration of the robot
        :type q: float np.ndarray(1,n)

        :return T: The transform between link 0 and joints or joints[0]
            and joints[1]
        :rtype T: SE3
        """

        if not isscalar(joints):
            joints = getvector(joints, 2)
            j0 = joints[0]
            jn = joints[1]
        else:
            j0 = 0
            jn = joints

        if jn > self.n:
            raise ValueError("The joints value out of range")

        q = getvector(q, self.n)

        T = SE3()

        for i in range(j0, jn):
            T = T * self.links[i].A(q[i])

        return T

    def islimit(self, q):
        """
        Joint limit test

        :param q: The joint angles/configuration of the robot
        :type q: float np.ndarray(1,n)

        :return v: is a vector of boolean values, one per joint, False if q[i]
            is within the joint limits, else True
        :rtype v: bool list
        """

        q = getvector(q, self.n)
        v = []

        for i in range(self.n):
            v.append(self.links[i].islimit(q[i]))

        return v

    def isspherical(self):
        """
        Test for spherical wrist. Tests if the robot has a spherical wrist,
        that is, the last 3 axes are revolute and their axes intersect at
        a point.

        :return: True if spherical wrist
        :rtype: bool
        """
        if self.n < 3:
            return False

        L = self.links[self.n-3:self.n]

        alpha = [-np.pi/2, np.pi/2]

        if L[0].a == 0 and L[1].a == 0 and \
                L[1].d == 0 and \
                (
                    (L[0].alpha == alpha[0] and L[1].alpha == alpha[1]) or
                    (L[0].alpha == alpha[1] and L[1].alpha == alpha[0])
                ) and \
                L[0].sigma == 0 and L[1].sigma == 0 and L[2].sigma == 0:

            return True
        else:
            return False

    def payload(self, m, p=np.zeros(3)):
        """
        Add payload mass adds a payload with point mass m at position p
        in the end-effector coordinate frame. payload(0) removes added
        payload.

        :param m: mass (kg)
        :type m: float
        :param p: position in end-effector frame
        :type p: float np.ndarray(3,1)
        """

        p = getvector(p, 3, out='col')
        lastlink = self.links(self.n-1)

        lastlink.m = m
        lastlink.r = p

    def jointdynamics(self, q, qd):
        """
        Transfer function of joint actuator. Returns a vector of N 
        continuous-time transfer function objects that represent the
        transfer function 1/(Js+B) for each joint based on the dynamic
        parameters of the robot and the configuration q (1xN).
        n is the number of robot joints.

        :param q: The joint angles/configuration of the robot
        :type q: float np.ndarray(1,n)
        :param qd: The joint velocities of the robot
        :type qd: float np.ndarray(1,n)
        """

        # TODO a tf object implementation?
        pass


    # """
    # The spatial velocity Jacobian which relates the velocity in end-effector
    # frame to velocity in the base frame.

    # Parameters
    # ----------
    # q : float np.ndarray(1,n)
    #     The joint angles/configuration of the robot

    # Returns
    # -------
    # J : float np.ndarray(6,n)
    #     The velocity Jacobian in 0 frame

    # Examples
    # --------
    # >>> J = panda.jacob0v(np.array([1,1,1,1,1,1,1]))
    # >>> J = panda.J0v

    # See Also
    # --------
    # ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame
    # ropy.robot.m : Calculates the manipulability index of the robot
    # ropy.robot.Jm : Calculates the manipiulability Jacobian
    # ropy.robot.fkine : Calculates the forward kinematics of a robot
    # """
    # def jacob0v(self, q):
    #     r = self.fkine(q)[0:3,0:3]

    #     Jv = np.zeros((6,6))
    #     Jv[:3,:3] = r
    #     Jv[3:,3:] = r

    #     return Jv

    # """
    # The spatial velocity Jacobian which relates the velocity in base
    # frame to velocity in the end-effector frame.

    # Parameters
    # ----------
    # q : float np.ndarray(1,n)
    #     The joint angles/configuration of the robot

    # Returns
    # -------
    # J : float np.ndarray(6,n)
    #     The velocity Jacobian in ee frame

    # Examples
    # --------
    # >>> J = panda.jacobev(np.array([1,1,1,1,1,1,1]))
    # >>> J = panda.Jev

    # See Also
    # --------
    # ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame
    # ropy.robot.m : Calculates the manipulability index of the robot
    # ropy.robot.Jm : Calculates the manipiulability Jacobian
    # ropy.robot.fkine : Calculates the forward kinematics of a robot
    # """
    # def jacobev(self, q):
    #     r = self.fkine(q)[0:3,0:3]
    #     r = np.linalg.inv(r)

    #     Jv = np.zeros((6,6))
    #     Jv[:3,:3] = r
    #     Jv[3:,3:] = r

    #     return Jv
