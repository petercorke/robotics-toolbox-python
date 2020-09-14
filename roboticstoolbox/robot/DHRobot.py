#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
from functools import wraps
from roboticstoolbox.robot.DHLink import DHLink
from roboticstoolbox.tools.null import null
from spatialmath.base.argcheck import \
    getvector, ismatrix, isscalar, verifymatrix
from spatialmath.base.transforms3d import tr2delta, tr2eul
from spatialmath import SE3, Twist3
from scipy.optimize import minimize, Bounds, LinearConstraint
from frne import init, frne, delete
from roboticstoolbox.backend.PyPlot.functions import \
    _plot, _teach, _fellipse, _vellipse, _plot_ellipse, \
    _plot2, _teach2
from roboticstoolbox.robot.Dynamics import Dynamics


class DHRobot(Dynamics):
    """
    A superclass for arm type robots. A concrete class that represents a
    serial-link arm-type robot.  Each link and joint in the chain is
    described by a Link-class object using Denavit-Hartenberg parameters
    (standard or modified).

    :param L: Series of links which define the robot
    :type L: list(n)
    :param name: Name of the robot
    :type name: string
    :param manufacturer: Manufacturer of the robot
    :type manufacturer: string
    :param base: Locaation of the base
    :type base: SE3
    :param tool: Location of the tool
    :type tool: SE3
    :param gravity: The gravity vector
    :type n: ndarray(3)

    :notes:
        - Link subclass elements passed in must be all standard, or all
          modified, DH parameters.

    :references:
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
            gravity=np.array([0, 0, 9.81])):

        self.name = name
        self.manufacturer = manufacturer
        self.base = base
        self.tool = tool
        self.gravity = gravity

        super().__init__()

        # Verify L
        if not isinstance(L, list):
            raise TypeError('The links L must be stored in a list.')

        length = len(L)
        self._links = []
        self._n = 0

        for i in range(length):
            if isinstance(L[i], DHLink):
                self._links.append(L[i])
                self._n += 1
                L[i].id = self._n

            elif isinstance(L[i], DHRobot):
                for j in range(L[i].n):
                    self._links.append(L[i].links[j])
                    self._n += 1
                    L[i].id = self._n

            else:
                raise TypeError("Input can be only DHLink or DHRobot")

        # Current joint angles of the robot
        self.q = np.zeros(self.n)
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)

        self.control_type = 'v'

        # Check the DH convention
        self._mdh = self.links[0].mdh
        if not all([link.mdh == self.mdh for link in self.links]):
            raise ValueError('Robot has mixed D&H link conventions')

        # rne parameters
        self._rne_init = False
        self._rne_ob = None
        self._rne_changed = False

    def __str__(self):
        """
        Pretty prints the DH Model of the robot. Will output angles in degrees

        :return: Pretty print of the robot model
        :rtype: str
        """
        axes = ''
        L = ''

        for i in range(self.n):
            L += str(self.links[i]) + '\n'

            if not self.links[i].sigma:
                axes += 'R'
            else:
                axes += 'P'

        if not self.mdh:
            dh = 'std DH'
        else:
            dh = 'mod DH'

        rpy = self.tool.rpy()

        for i in range(3):
            if rpy[i] == 0:
                rpy[i] = 0

        model = '\n%s (%s): %d axis, %s, %s\n'\
            'Parameters:\n'\
            '%s\n'\
            'tool:  t = (%g, %g, %g),  RPY/xyz = (%g, %g, %g) deg' % (
                self.name, self.manufacturer, self.n, axes, dh,
                L,
                self.tool.A[0, 3], self.tool.A[1, 3],
                self.tool.A[2, 3], rpy[0], rpy[1], rpy[2]
            )

        return model

    def __add__(self, L):
        nlinks = []

        # TODO - Should I do a deep copy here a physically copy the DHLinks
        # and not just the references?
        # Copy DHLink references to new list
        for i in range(self.n):
            nlinks.append(self.links[i])

        if isinstance(L, DHLink):
            nlinks.append(L)
        elif isinstance(L, DHRobot):
            for j in range(L.n):
                nlinks.append(L.links[j])
        else:
            raise TypeError("Can only combine DHRobots with other "
                            "DHRobots or DHLinks")

        return DHRobot(
            nlinks,
            name = self.name,
            manufacturer = self.manufacturer,
            base = self.base,
            tool = self.tool,
            gravity = self.gravity)

    def _copy(self):
        L = []

        for i in range(self.n):
            L.append(self.links[i]._copy())

        r2 = DHRobot(
            L,
            name = self.name,
            manufacturer=  self.manufacturer,
            base = self.base,
            tool = self.tool,
            gravity = self.gravity)

        r2.q = self.q
        r2.qd = self.qd
        r2.qdd = self.qdd

        return r2

    def _init_rne(self):
        # Compress link data into a 1D array
        L = np.zeros(24 * self.n)

        for i in range(self.n):
            j = i * 24
            L[j] = self.links[i].alpha
            L[j + 1] = self.links[i].a
            L[j + 2] = self.links[i].theta
            L[j + 3] = self.links[i].d
            L[j + 4] = self.links[i].sigma
            L[j + 5] = self.links[i].offset
            L[j + 6] = self.links[i].m
            L[j + 7:j + 10] = self.links[i].r.flatten()
            L[j + 10:j + 19] = self.links[i].I.flatten()
            L[j + 19] = self.links[i].Jm
            L[j + 20] = self.links[i].G
            L[j + 21] = self.links[i].B
            L[j + 22:j + 24] = self.links[i].Tc.flatten()

        self._rne_ob = init(self.n, self.mdh, L, self.gravity[:, 0])

    def _check_rne(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not args[0]._rne_init or args[0]._rne_changed:
                args[0]._init_rne()
                args[0]._rne_init = True
                args[0]._rne_changed = False
            return func(*args, **kwargs)
        return wrapper

    def _listen_rne(func):
        @wraps(func)
        def wrapper(*args):
            args[0]._rne_changed = True
            return func(*args)
        return wrapper

    @property
    def name(self):
        return self._name

    @property
    def manufacturer(self):
        return self._manufacturer

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
    def d(self):
        v = []
        for i in range(self.n):
            v.append(self.links[i].d)
        return v

    @property
    def a(self):
        v = []
        for i in range(self.n):
            v.append(self.links[i].a)
        return v

    @property
    def theta(self):
        v = []
        for i in range(self.n):
            v.append(self.links[i].theta)
        return v

    @property
    def r(self):
        v = np.copy(self.links[0].r)
        for i in range(1, self.n):
            v = np.c_[v, self.links[i].r]
        return v

    @property
    def offset(self):
        v = []
        for i in range(self.n):
            v.append(self.links[i].offset)
        return v

    @property
    def qlim(self):
        v = np.copy(self.links[0].qlim)
        for i in range(1, self.n):
            v = np.c_[v, self.links[i].qlim]
        return v

    @manufacturer.setter
    def manufacturer(self, manufacturer_new):
        self._manufacturer = manufacturer_new

    @control_type.setter
    def control_type(self, cn):
        if cn == 'p' or cn == 'v' or cn == 'a':
            self._control_type = cn
        else:
            raise ValueError(
                'Control type must be one of \'p\', \'v\', or \'a\'')

    @gravity.setter
    @_listen_rne
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
        self._q = getvector(q_new, self.n)

    @qd.setter
    def qd(self, qd_new):
        self._qd = getvector(qd_new, self.n)

    @qdd.setter
    def qdd(self, qdd_new):
        self._qdd = getvector(qdd_new, self.n)

    def A(self, joints, q=None):
        """
        Link forward kinematics.

        T = A(joints, q) transforms between link frames for the J'th joint.
        q is a vector (n) of joint variables. For:
        - standard DH parameters, this is from frame {J-1} to frame {J}.
        - modified DH parameters, this is from frame {J} to frame {J+1}.

        T = A(joints) as above except uses the stored q value of the
        robot object.

        :param joints: Joints to transform to (int) or between (list/tuple)
        :type joints: int, tuple or 2 element list
        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values)
        :type q: float ndarray(1,n)

        :return T: The transform between link 0 and joints or joints[0]
            and joints[1]
        :rtype T: SE3

        :notes:
            - Base and tool transforms are not applied.

        """

        if not isscalar(joints):
            joints = getvector(joints, 2)
            j0 = int(joints[0])
            jn = int(joints[1])
        else:
            j0 = 0
            jn = int(joints)

        jn += 1

        if jn > self.n:
            raise ValueError("The joints value out of range")

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        T = SE3()

        for i in range(j0, jn):
            T = T * self.links[i].A(q[i])

        return T

    def islimit(self, q=None):
        """
        Joint limit test.

        v = islimit(q) returns a list of boolean values indicating if the
        joint configuration q is in vialation of the joint limits.

        v = jointlimit() as above except uses the stored q value of the
        robot object.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values)
        :type q: float ndarray(n)

        :return v: is a vector of boolean values, one per joint, False if q[i]
            is within the joint limits, else True
        :rtype v: bool list

        """

        if q is None:
            q = np.copy(self.q)
        else:
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

        L = self.links[self.n - 3:self.n]

        alpha = [-np.pi / 2, np.pi / 2]

        if L[0].a == 0 and L[1].a == 0 and L[1].d == 0 and (
                (L[0].alpha == alpha[0] and L[1].alpha == alpha[1])
                or (L[0].alpha == alpha[1] and L[1].alpha == alpha[0])
        ) and L[0].sigma == 0 and L[1].sigma == 0 and L[2].sigma == 0:

            return True
        else:
            return False

    def isprismatic(self):
        """
        Identify prismatic joints.

        p = isprismatic() returns a bool list identifying the prismatic joints
        within the robot

        :return: a list of bool variables, one per joint, true if
            the corresponding joint is prismatic, otherwise false.
        :rtype: bool list

        """

        p = []

        for i in range(self.n):
            p.append(self.links[i].isprismatic())

        return p

    def isrevolute(self):
        """
        Identify revolute joints.

        p = isrevolute() returns a bool list identifying the revolute joints
        within the robot

        :return: a list of bool variables, one per joint, true if
            the corresponding joint is revolute, otherwise false.
        :rtype: bool list

        """

        p = []

        for i in range(self.n):
            p.append(self.links[i].isrevolute())

        return p

    def todegrees(self, q=None):
        """
        Convert joint angles to degrees.

        qdeg = toradians(q) converts joint coordinates q to degrees.

        qdeg = toradians() as above except uses the stored q value of the
        robot object.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values)
        :type q: float ndarray(n)

        :return: a vector of joint coordinates where those elements
            corresponding to revolute joints are converted from radians to
            degrees. Elements corresponding to prismatic joints are copied
            unchanged.
        :rtype: float ndarray(n)

        """

        if q is None:
            qdeg = np.copy(self.q)
        else:
            qdeg = getvector(q, self.n)

        k = self.isrevolute()
        qdeg[k] *= 180 / np.pi
        return qdeg

    def toradians(self, q):
        """
        Convert joint angles to radians.

        qrad = toradians(q) converts joint coordinates q to radians.

        :param q: The joint angles/configuration of the robot (Not optional,
            stored q is always radians)
        :type q: float ndarray(n)

        :return: a vector of joint coordinates where those elements
            corresponding to revolute joints are converted from degrees to
            radians. Elements corresponding to prismatic joints are copied
            unchanged.
        :rtype: float ndarray(n)

        """

        qrad = getvector(q, self.n)

        k = self.isrevolute()
        qrad[k] *= np.pi / 180
        return qrad

    def twists(self, q=None):
        """
        Joint axis as  twists

        :param q: The joint angles/configuration of the robot
        :type q: array_like (n)
        :return tw: a vector of Twist objects
        :rtype tw: float ndarray(n,)
        :return T0: Represents the pose of the tool
        :rtype T0: SE3

        - ``tw, T0 = twists(q)`` calculates a vector of Twist objects (n) that
          represent the axes of the joints for the robot with joint coordinates
          ``q`` (n). Also returns T0 which is an SE3 object representing the pose of
          the tool.

        - ``tw, T0 = twists()`` as above but the joint coordinates are taken to be
          zero.

        """

        if q is None:
            q = np.zeros((self.n))

        T = self.fkine_all(q)
        tw = Twist3.Alloc(self.n)
        if not self.mdh:
            # DH case
            for j in range(self.n):
                if j == 0:
                    if self.links[j].sigma == 0:
                        tw[j] = Twist3.R([0, 0, 1], [0, 0, 0])  # revolute
                    else:
                        tw[j] = Twist3.P([0, 0, 1])  # prismatic
                else:
                    if self.links[j].sigma == 0:
                        tw[j] = Twist3.R(T[j-1].a, T[j-1].t)  # revolute
                    else:
                        tw[j] = Twist3.P(T[j-1].a)  # prismatic
        else:
            # MDH case
            for j in range(self.n):
                if self.links[j].sigma == 0:
                    tw[j] = Twist3.R(T[j].a, T[j].t)
                else:
                    tw[j] = Twist3.P(T[j].a, T[j].t)

        return tw, T[-1]

    def fkine(self, q=None):
        '''
        T = fkine(q) evaluates forward kinematics for the robot at joint
        configuration q.

        T = fkine() as above except uses the stored q value of the
        robot object.

        Trajectory operation:
        for each point on a trajectory of joints q

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values). If q is a matrix
            the rows are interpreted as the generalized joint coordinates
            for a sequence of points along a trajectory. q(j,i) is the
            j'th joint parameter for the i'th trajectory point.
        :type q: float ndarray(n) or (mxn)

        :return T: Homogeneous transformation matrix or trajectory
        :rtype T: SE3 or SE3 list

        :notes:
            - The robot's base or tool transform, if present, are incorporated
              into the result.
            - Joint offsets, if defined, are added to q before the forward
              kinematics are computed.

        '''

        cols = 0
        if q is None:
            q = np.copy(self.q)
        elif isinstance(q, np.ndarray) and q.ndim == 2 and q.shape[0] > 1:
            cols = q.shape[0]
            ismatrix(q, (cols, self.n))
        else:
            q = getvector(q, self.n)

        if cols == 0:
            # Single configuration
            t = self.base
            for i in range(self.n):
                t = t * self.links[i].A(q[i])
            t = t * self.tool
        else:
            # Trajectory

            for i in range(cols):
                tr = self.base
                for j in range(self.n):
                    tr *= self.links[j].A(q[j, i])
                tr = tr * self.tool

                if i == 0:
                    t = SE3(tr)
                else:
                    t.append(tr)

        return t

    def fkine_all(self, q=None):
        '''
        Tall = allfkine(q) evaluates fkine for each joint within a robot and
        returns a sequence of link frame poses.

        Tall = allfkine() as above except uses the stored q value of the
        robot object.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)

        :return T: Homogeneous transformation trajectory
        :rtype T: SE3 list

        :notes:
            - The robot's base or tool transform, if present, are incorporated
              into the result.
            - Joint offsets, if defined, are added to q before the forward
              kinematics are computed.

        '''

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        t = self.base
        Tall = SE3()
        for i in range(self.n):
            t = t * self.links[i].A(q[i])

            if i == 0:
                Tall = SE3(t)
            else:
                Tall.append(t)

        return Tall

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

        n = self.n
        L = self.links
        J = np.zeros((6, self.n))

        U = self.tool.A

        for j in range(n - 1, -1, -1):
            if self.mdh == 0:
                # standard DH convention
                U = L[j].A(q[j]).A @ U

            if not L[j].sigma:
                # revolute axis
                d = np.array([
                    [-U[0, 0] * U[1, 3] + U[1, 0] * U[0, 3]],
                    [-U[0, 1] * U[1, 3] + U[1, 1] * U[0, 3]],
                    [-U[0, 2] * U[1, 3] + U[1, 2] * U[0, 3]]
                ])
                delta = np.expand_dims(U[2, :3], axis=1)  # nz oz az
            else:
                # prismatic axis
                d = np.expand_dims(U[2, :3], axis=1)      # nz oz az
                delta = np.zeros((3, 1))

            J[:, j] = np.squeeze(np.concatenate((d, delta)))

            if self.mdh != 0:
                # modified DH convention
                U = L[j].A(q[j]).A @ U

        return J

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

        """

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        J0 = self.jacob0v(q) @ self.jacobe(q)

        return J0

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

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        r = self.fkine(q).R

        Jv = np.zeros((6, 6))
        Jv[:3, :3] = r
        Jv[3:, 3:] = r

        return Jv

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

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        r = self.fkine(q).R
        r = np.linalg.inv(r)

        Jv = np.zeros((6, 6))
        Jv[:3, :3] = r
        Jv[3:, 3:] = r

        return Jv

    def ikcon(self, T, q0=None):
        """
        Inverse kinematics by optimization with joint limits

        q, success, err = ikcon(T, q0) calculates the joint coordinates (1xn)
        corresponding to the robot end-effector pose T which is an SE3 object
        or homogenenous transform matrix (4x4), and N is the number of robot
        joints. Initial joint coordinates Q0 used for the minimisation.

        q, success, err = ikcon(T) as above but q0 is set to 0.

        Trajectory operation:
        In all cases if T is a vector of SE3 objects or a homogeneous
        transform sequence (4x4xm) then returns the joint coordinates
        corresponding to each of the transforms in the sequence. q is mxn
        where n is the number of robot joints. The initial estimate of q
        for each time step is taken as the solution from the previous time
        step. Retruns trajectory of joints q (mxn), list of success (m) and
        list of errors (m)

        :param T: The desired end-effector pose
        :type T: SE3 or SE3 trajectory
        :param q0: initial joint configuration (default all zeros)
        :type q0: float ndarray(n) (default all zeros)

        :retrun q: The calculated joint values
        :rtype q: float ndarray(n)
        :retrun success: IK solved (True) or failed (False)
        :rtype success: bool
        :retrun error: Final pose error
        :rtype error: float

        :notes:
            - Joint limits are considered in this solution.
            - Can be used for robots with arbitrary degrees of freedom.
            - In the case of multiple feasible solutions, the solution
              returned depends on the initial choice of q0.
            - Works by minimizing the error between the forward kinematics
              of the joint angle solution and the end-effector frame as an
              optimisation.
            - The objective function (error) is described as:
              sumsqr( (inv(T)*robot.fkine(q) - eye(4)) * omega )
              Where omega is some gain matrix, currently not modifiable.

        """

        if not isinstance(T, SE3):
            T = SE3(T)

        trajn = len(T)

        try:
            if q0 is not None:
                q0 = getvector(q0, self.n, 'row')
            else:
                q0 = np.zeros((trajn, self.n))
        except ValueError:
            verifymatrix(q0, (trajn, self.n))

        # create output variables
        qstar = np.zeros((trajn, self.n))
        error = []
        exitflag = []

        reach = np.sum(np.abs([self.a, self.d]))
        omega = np.diag([1, 1, 1, 3 / reach])

        def cost(q, T, omega):
            return np.sum(
                (
                    (np.linalg.pinv(T.A) @ self.fkine(q).A - np.eye(4)) @
                    omega) ** 2
            )

        bnds = Bounds(self.qlim[0, :], self.qlim[1, :])

        for i in range(trajn):
            Ti = T[i]
            res = minimize(
                lambda q: cost(q, Ti, omega),
                q0[i, :], bounds=bnds, options={'gtol': 1e-6})
            qstar[i, :] = res.x
            error.append(res.fun)
            exitflag.append(res.success)

        if trajn > 1:
            return qstar, exitflag, error
        else:
            return qstar[0, :], exitflag[0], error[0]

    def ikine(
            self, T,
            ilimit=500,
            rlimit=100,
            tol=1e-10,
            Y=0.1,
            Ymin=0,
            mask=None,
            q0=None,
            search=False,
            slimit=100,
            transpose=None):
        """
        Inverse kinematics by optimization without joint limits

        ``q, failure, reason = ikine(T)`` are the joint coordinates (n)
        corresponding to the robot end-effector pose ``T`` which is an ``SE3``
        instance. ``failure`` is True if the solver failed, and ``reason``
        contains details of the failure.

        This method can be used for robots with any number of degrees of
        freedom.

        Trajectory operation:
        If ``T`` contains multiple values, ie. a trajectory, then returns the
        joint coordinates corresponding to each of the pose values in ``T``.
        ``q`` is mxn where n is the number of robot joints. The initial
        estimate of ``q`` for each time step is taken as the solution from the
        previous time step. Returns trajectory of joints ``q`` (mxn), list of
        failure (m) and list of error reasons (m).

        :param T: The desired end-effector pose
        :type T: SE3 or SE3 trajectory
        :param ilimit: maximum number of iterations
        :type ilimit: int (default 500)
        :param rlimit: maximum number of consecutive step rejections
        :type rlimit: int (default 100)
        :param tol: final error tolerance
        :type tol: float (default 1e-10)
        :param Y: initial value of lambda
        :type Y: float (default 0.1)
        :param Ymin: minimum allowable value of lambda
        :type Ymin: float (default 0)
        :param mask: mask vector that correspond to translation in X, Y and Z
            and rotation about X, Y and Z respectively.
        :type mask: float ndarray(6)
        :param q0: initial joint configuration (default all zeros)
        :type q0: float ndarray(n) (default all zeros)
        :param search: search over all configurations
        :type search: bool
        :param slimit: maximum number of search attempts
        :type slimit: int (default 100)
        :param transpose: use Jacobian transpose with step size A, rather
            than Levenberg-Marquadt
        :type transpose: float

        :return q: The calculated joint values
        :rtype q: float ndarray(n)
        :return failure: IK solver failed
        :rtype failure: bool or list of bool
        :return error: If failed, what went wrong
        :rtype error: List of str

        Underactuated robots:
        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the 'mask' option where the mask vector (1x6)
        specifies the Cartesian DOF (in the wrist coordinate frame) that will
        be ignored in reaching a solution.  The mask vector has six elements
        that correspond to translation in X, Y and Z, and rotation about X, Y
        and Z respectively. The value should be 0 (for ignore) or 1. The
        number of non-zero elements should equal the number of manipulator
        DOF.

        For example when using a 3 DOF manipulator rotation orientation might
        be unimportant in which case use the option: mask = [1 1 1 0 0 0].

        For robots with 4 or 5 DOF this method is very difficult to use since
        orientation is specified by T in world coordinates and the achievable
        orientations are a function of the tool position.

        :notes:
            - Solution is computed iteratively.
            - Implements a Levenberg-Marquadt variable step size solver.
            - The tolerance is computed on the norm of the error between
              current and desired tool pose.  This norm is computed from
              distances and angles without any kind of weighting.
            - The inverse kinematic solution is generally not unique, and
              depends on the initial guess q0 (defaults to 0).
            - The default value of q0 is zero which is a poor choice for most
              manipulators (eg. puma560, twolink) since it corresponds to a
              kinematic singularity.
            - Such a solution is completely general, though much less
              efficient than specific inverse kinematic solutions derived
              symbolically, like ikine6s or ikine3.
            - This approach allows a solution to be obtained at a singularity,
              but the joint angles within the null space are arbitrarily
              assigned.
            - Joint offsets, if defined, are added to the inverse kinematics
              to generate q.
            - Joint limits are not considered in this solution.
            - The 'search' option peforms a brute-force search with initial
              conditions chosen from the entire configuration space.
            - If the search option is used any prismatic joint must have
              joint limits defined.

        :references:
            - Robotics, Vision & Control, P. Corke, Springer 2011,
              Section 8.4.

        """

        if not isinstance(T, SE3):
            T = SE3(T)

        trajn = len(T)
        err = []

        try:
            if q0 is not None:
                if trajn == 1:
                    q0 = getvector(q0, self.n, 'row')
                else:
                    verifymatrix(q0, (trajn, self.n))
            else:
                q0 = np.zeros((trajn, self.n))
        except ValueError:
            verifymatrix(q0, (trajn, self.n))

        if mask is not None:
            mask = getvector(mask, 6)
        else:
            mask = np.ones(6)

        if search:
            # Randomised search for a starting point
            search = False
            # quiet = True

            for k in range(slimit):

                q0n = np.zeros(self.n)
                for j in range(self.n):
                    qlim = self.links[j].qlim
                    if np.sum(np.abs(qlim)) == 0:
                        if not self.links[j].sigma:
                            q0n[j] = np.random.rand() * 2 * np.pi - np.pi
                        else:
                            raise ValueError('For a prismatic joint, '
                                             'search requires joint limits')
                    else:
                        q0n[j] = np.random.rand() * (qlim[1] - qlim[0]) + \
                            qlim[0]

                # fprintf('Trying q = %s\n', num2str(q))

                q, _, _ = self.ikine(
                    T,
                    ilimit,
                    rlimit,
                    tol,
                    Y,
                    Ymin,
                    mask,
                    q0n,
                    search,
                    slimit,
                    transpose)

                if not np.sum(np.abs(q)) == 0:
                    return q, True, err

            q = np.array([])
            return q, False, err

        if not self.n >= np.sum(mask):
            raise ValueError('Number of robot DOF must be >= the same number '
                             'of 1s in the mask matrix')
        W = np.diag(mask)

        # Preallocate space for results
        qt = np.zeros((len(T), self.n))

        # Total iteration count
        tcount = 0

        # Rejected step count
        rejcount = 0

        failed = []
        nm = 0

        revolutes = []
        for i in range(self.n):
            revolutes.append(not self.links[i].sigma)

        for i in range(len(T)):
            iterations = 0
            q = np.copy(q0[i, :])
            Yl = Y

            while True:
                # Update the count and test against iteration limit
                iterations += 1

                if iterations > ilimit:
                    err.append('ikine: iteration limit {0} exceeded '
                               ' (pose {1}), final err {2}'.format(
                                   ilimit, i, nm))
                    failed.append(True)
                    break

                e = tr2delta(self.fkine(q).A, T[i].A)

                # Are we there yet
                if np.linalg.norm(W @ e) < tol:
                    # print(iterations)
                    failed.append(False)
                    break

                # Compute the Jacobian
                J = self.jacobe(q)

                JtJ = J.T @ W @ J

                if transpose is not None:
                    # Do the simple Jacobian transpose with constant gain
                    dq = transpose * J.T @ e
                else:
                    # Do the damped inverse Gauss-Newton with
                    # Levenberg-Marquadt
                    dq = np.linalg.inv(
                        JtJ + ((Yl + Ymin) * np.eye(self.n))
                    ) @ J.T @ W @ e

                    # Compute possible new value of
                    qnew = q + dq

                    # And figure out the new error
                    enew = tr2delta(self.fkine(qnew).A, T[i].A)

                    # Was it a good update?
                    if np.linalg.norm(W @ enew) < np.linalg.norm(W @ e):
                        # Step is accepted
                        q = qnew
                        e = enew
                        Yl = Yl / 2
                        rejcount = 0
                    else:
                        # Step is rejected, increase the damping and retry
                        Yl = Yl * 2
                        rejcount += 1
                        if rejcount > rlimit:
                            err.append(
                                'ikine: rejected-step limit {0} exceeded '
                                '(pose {1}), final err {2}'.format(
                                    rlimit, i, np.linalg.norm(W @ enew)))
                            failed.append(True)
                            break

                # Wrap angles for revolute joints
                k = (q > np.pi) & revolutes
                q[k] -= 2 * np.pi

                k = (q < -np.pi) & revolutes
                q[k] += + 2 * np.pi

                nm = np.linalg.norm(W @ e)

            qt[i, :] = q
            tcount += iterations

        if any(failed):
            err.append(
                'failed to converge: try a different '
                'initial value of joint coordinates')

        if trajn == 1:
            qt = qt[0, :]
            failed = failed[0]

        return qt, failed, err

    def ikine3(self, T, left=True, elbow_up=True):
        """
        Analytical inverse kinematics for three link robots

        q = ikine3(T) is the joint coordinates (3) corresponding to the robot
        end-effector pose T represented by the homogenenous transform.  This
        is a analytic solution for a 3-axis robot (such as the first three
        joints of a robot like the Puma 560). This will have the arm to the
        left and the elbow up.

        q = ikine3(T, left, elbow_up) as above except the arm location and
        elbow position can be specified.

        Trajectory operation:
        In all cases if T is a vector of SE3 objects (m) or a homogeneous
        transform sequence (4x4xm) then returns the joint coordinates
        corresponding to each of the transforms in the sequence. Q is 3xm.

        :param T: The desired end-effector pose
        :type T: SE3 or SE3 trajectory
        :param left: True for arm to the left (default), else arm to the right
        :type left: bool
        :param elbow_up: True for elbow up (default), else elbow down
        :type elbow_up: bool

        :retrun q: The calculated joint values
        :rtype q: float ndarray(n)

        :notes:
            - The same as IKINE6S without the wrist.
            - The inverse kinematic solution is generally not unique, and
              depends on the configuration string.
            - Joint offsets, if defined, are added to the inverse kinematics
              to generate q.

        :reference:
            - Inverse kinematics for a PUMA 560 based on the equations by Paul
              and Zhang. From The International Journal of Robotics Research
              Vol. 5, No. 2, Summer 1986, p. 32-44

        """

        if not self.n == 3:
            raise ValueError(
                "Function only applicable to three degree of freedom robots")

        if self.mdh:
            raise ValueError(
                "Function only applicable to robots with standard DH "
                "parameters")

        if not self.isrevolute() == [True, True, True]:
            raise ValueError(
                "Function only applicable to robots with revolute joints")

        if not isinstance(T, SE3):
            T = SE3(T)

        trajn = len(T)

        qt = np.zeros((trajn, 3))

        for j in range(trajn):
            theta = np.zeros(3)

            a2 = self.links[1].a
            a3 = self.links[2].a
            d3 = self.links[2].d

            # undo base transformation
            Ti = np.linalg.inv(self.base.A) @ T[j].A

            # The following parameters are extracted from the Homogeneous
            # Transformation
            Px = Ti[0, 3]
            Py = Ti[1, 3]
            Pz = Ti[2, 3]

            # The configuration parameter determines what n1,n2 values are
            # used and how many solutions are determined which have values
            # of -1 or +1.

            if left:
                n1 = -1
            else:
                n1 = 1

            if not elbow_up:
                if n1 == 1:
                    n2 = -1
                else:
                    n2 = 1
            else:
                if n1 == 1:
                    n2 = 1
                else:
                    n2 = -1

            # Solve for theta[0]
            # based on the configuration parameter n1

            r = np.sqrt(Px**2 + Py**2)

            if n1 == 1:
                theta[0] = np.arctan2(Py, Px) + np.arcsin(d3 / r)
            else:
                theta[0] = np.arctan2(Py, Px) + np.pi - np.arcsin(d3 / r)

            # Solve for theta[1]
            # based on the configuration parameter n2

            V114 = Px * np.cos(theta[0]) + Py * np.sin(theta[0])
            r = np.sqrt(V114**2 + Pz**2)

            Psi = np.arccos(
                (a2**2 - d3**2 - a3**2 + V114**2 + Pz**2)
                / (2.0 * a2 * r))

            theta[1] = np.arctan2(Pz, V114) + n2 * Psi

            # Solve for theta[2]
            num = np.cos(theta[1]) * V114 + np.sin(theta[1]) * Pz - a2
            den = np.cos(theta[1]) * Pz - np.sin(theta[1]) * V114
            theta[2] = np.arctan2(a3, d3) - np.arctan2(num, den)

            # remove the link offset angles
            for i in range(3):
                theta[i] -= self.links[i].offset

            # Append to trajectory
            qt[j, :] = theta

        if trajn == 1:
            return qt[0, :]
        else:
            return qt

    def ikine6s(self, T, left=True, elbow_up=True, wrist_flip=False):
        """
        Analytical inverse kinematics

        q, err = ikine6s(T) are the joint coordinates (n) corresponding to the
        robot end-effector pose T which is an SE3 object or homogenenous
        transform matrix (4x4), and n is the number of robot joints. This
        is an analytic solution for a 6-axis robot with a spherical wrist
        (the most common form for industrial robot arms).

        q, err = ikine6s(T, left, elbow_up, wrist_flip) as above except the
        arm location, elbow position, and wrist orientation can be specified.

        Trajectory operation:
        In all cases if T is a vector of SE3 objects (1xM) or a homogeneous
        transform sequence (4x4xM) then the inverse kinematics is computed for
        all m poses resulting in q (mxn) with each row representing the joint
        angles at the corresponding pose.

        :param T: The desired end-effector pose
        :type T: SE3 or SE3 trajectory
        :param left: True for arm to the left (default), else arm to the right
        :type left: bool
        :param elbow_up: True for elbow up (default), else elbow down
        :type elbow_up: bool
        :param wrist_flip: False for wrist not flipped (default), else wrist
            flipped (rotated by 180 deg)
        :type wrist_flip: bool

        :return q: The calculated joint values
        :rtype q: float ndarray(n)
        :return err: Any errors encountered
        :rtype err: list String

        :notes:
            - Treats a number of specific cases:
                - Robot with no shoulder offset
                - Robot with a shoulder offset (has lefty/righty configuration)
                - Robot with a shoulder offset and a prismatic third joint
                  (like Stanford arm)
            - The Puma 560 arms with shoulder and elbow offsets (4 lengths
              parameters)
            - The Kuka KR5 with many offsets (7 length parameters)
            - The inverse kinematics for the various cases determined using
              ikine_sym.
            - The inverse kinematic solution is generally not unique, and
              depends on the configuration string.
            - Joint offsets, if defined, are added to the inverse kinematics
              to generate q.
            - Only applicable for standard Denavit-Hartenberg parameters

        :reference:
            - Inverse kinematics for a PUMA 560,
              Paul and Zhang,
              The International Journal of Robotics Research,
              Vol. 5, No. 2, Summer 1986, p. 32-44

        """

        if not self.n == 6:
            raise ValueError(
                "Function only applicable to six degree of freedom robots")

        if self.mdh:
            raise ValueError(
                "Function only applicable to robots with standard DH "
                "parameters")

        if not self.isspherical():
            raise ValueError(
                "Function only applicable to robots with a spherical wrist")

        if not isinstance(T, SE3):
            T = SE3(T)

        trajn = len(T)

        sol = [1, 1, 1]

        if not left:
            sol[0] = 2

        if not elbow_up:
            sol[1] = 2

        if wrist_flip:
            sol[2] = 2

        if self._is_simple():
            self.ikineType = 'nooffset'
        elif self._is_puma():
            self.ikineType = 'puma'
        elif self._is_offset():
            self.ikineType = 'offset'
        elif self._is_rrp():
            self.ikineType = 'rrp'
        else:
            raise ValueError('This kinematic structure not supported')

        q = np.zeros((trajn, self.n))
        err = []

        for j in range(trajn):

            theta = np.zeros(self.n)

            # Undo base and tool transformations
            Ti = np.linalg.inv(self.base.A) @ T[j].A @ \
                np.linalg.inv(self.tool.A)

            if self.ikineType == 'puma':
                # Puma model with shoulder and elbow offsets
                # - Inverse kinematics for a PUMA 560,
                #   Paul and Zhang,
                #   The International Journal of Robotics Research,
                #   Vol. 5, No. 2, Summer 1986, p. 32-44

                a2 = self.links[1].a
                a3 = self.links[2].a
                d1 = self.links[0].d
                d3 = self.links[2].d
                d4 = self.links[3].d

                # The following parameters are extracted from the Homogeneous
                # Transformation as defined in equation 1, p. 34

                Px = Ti[0, 3]
                Py = Ti[1, 3]
                Pz = Ti[2, 3] - d1

                # Solve for theta[0]
                # r is defined in equation 38, p. 39.
                # theta[0] uses equations 40 and 41, p.39,
                # based on the configuration parameter n1

                r = np.sqrt(Px**2 + Py**2)
                if sol[0] == 1:
                    theta[0] = np.arctan2(Py, Px) + np.pi - np.arcsin(d3 / r)
                else:
                    theta[0] = np.arctan2(Py, Px) + np.arcsin(d3 / r)

                # Solve for theta[1]
                # V114 is defined in equation 43, p.39.
                # r is defined in equation 47, p.39.
                # Psi is defined in equation 49, p.40.
                # theta[1] uses equations 50 and 51, p.40, based on the
                # configuration parameter n2
                if sol[1] == 1:
                    n2 = -1
                else:
                    n2 = 1

                if sol[0] == 2:
                    n2 = -n2

                V114 = Px * np.cos(theta[0]) + Py * np.sin(theta[0])

                r = np.sqrt(V114**2 + Pz**2)

                Psi = np.arccos(
                    (a2**2 - d4**2 - a3**2 + V114**2 + Pz**2)
                    / (2.0 * a2 * r))

                if np.isnan(Psi):
                    theta = []
                else:
                    theta[1] = np.arctan2(Pz, V114) + n2 * Psi

                    # Solve for theta[2]
                    # theta[2] uses equation 57, p. 40.
                    num = np.cos(theta[1]) * V114 + np.sin(theta[1]) * Pz - a2
                    den = np.cos(theta[1]) * Pz - np.sin(theta[1]) * V114
                    theta[2] = np.arctan2(a3, d4) - np.arctan2(num, den)

            elif self.ikineType == 'nooffset':
                a2 = self.links[1].a
                a3 = self.links[2].a
                d1 = self.links[0].d

                px = Ti[0, 3]
                py = Ti[1, 3]
                pz = Ti[2, 3]

                # Autogenerated code
                if self.links[0].alpha < 0:
                    if sol[0] == 1:
                        temp = -px - py * 1j
                        if np.abs(temp) == 0:
                            temp = 0
                        theta[0] = np.angle(temp)
                    else:
                        theta[0] = np.angle(px + py * 1j)

                    print(theta[0])

                    S1 = np.sin(theta[0])
                    C1 = np.cos(theta[0])

                    if sol[1] == 1:
                        theta[1] = -np.angle(a2*d1*-2.0+a2*pz*2.0-C1*a2*px*2.0j-S1*a2*py*2.0j)+np.angle(d1*pz*2.0j-a2**2*1j+a3**2*1j-d1**2*1j-pz**2*1j-C1**2*px**2*1j-np.sqrt(0j+(a2*d1*2.0-a2*pz*2.0)**2+(C1*a2*px*2.0+S1*a2*py*2.0)**2-(d1*pz*-2.0+a2**2-a3**2+d1**2+pz**2+C1**2*px**2+S1**2*py**2+C1*S1*px*py*2.0)**2)-S1**2*py**2*1j-C1*S1*px*py*2.0j)  # noqa
                    else:
                        theta[1] = -np.angle(a2*d1*-2.0+a2*pz*2.0-C1*a2*px*2.0j-S1*a2*py*2.0j)+np.angle(d1*pz*2.0j-a2**2*1j+a3**2*1j-d1**2*1j-pz**2*1j-C1**2*px**2*1j+np.sqrt(0j+(a2*d1*2.0-a2*pz*2.0)**2+(C1*a2*px*2.0+S1*a2*py*2.0)**2-(d1*pz*-2.0+a2**2-a3**2+d1**2+pz**2+C1**2*px**2+S1**2*py**2+C1*S1*px*py*2.0)**2)-S1**2*py**2*1j-C1*S1*px*py*2.0j)  # noqa

                    S2 = np.sin(theta[1])
                    C2 = np.cos(theta[1])

                    if sol[2] == 1:
                        theta[2] = -np.angle(a2*-1j+C2*d1-C2*pz+S2*d1*1j-S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j-np.sqrt(0j+(-a2+S2*d1-S2*pz+C1*C2*px+C2*S1*py)**2+(-C2*d1+C2*pz+C1*S2*px+S1*S2*py)**2-a3**2))  # noqa
                    else:
                        theta[2] = -np.angle(a2*-1j+C2*d1-C2*pz+S2*d1*1j-S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j+np.sqrt(0j+(-a2+S2*d1-S2*pz+C1*C2*px+C2*S1*py)**2+(-C2*d1+C2*pz+C1*S2*px+S1*S2*py)**2-a3**2))  # noqa

                else:
                    if sol[0] == 1:
                        theta[0] = np.angle(px + py * 1j)
                    else:
                        temp = -px - py * 1j
                        if np.abs(temp) == 0:
                            temp = 0
                        theta[0] = np.angle(temp)

                    S1 = np.sin(theta[0])
                    C1 = np.cos(theta[0])

                    if sol[1] == 1:
                        theta[1] = -np.angle(a2*d1*2.0-a2*pz*2.0-C1*a2*px*2.0j-S1*a2*py*2.0j)+np.angle(d1*pz*2.0j-a2**2*1j+a3**2*1j-d1**2*1j-pz**2*1j-C1**2*px**2*1j-np.sqrt(0j+(a2*d1*2.0-a2*pz*2.0)**2+(C1*a2*px*2.0+S1*a2*py*2.0)**2-(d1*pz*-2.0+a2**2-a3**2+d1**2+pz**2+C1**2*px**2+S1**2*py**2+C1*S1*px*py*2.0)**2)-S1**2*py**2*1j-C1*S1*px*py*2.0j)  # noqa
                    else:
                        theta[1] = -np.angle(a2*d1*2.0-a2*pz*2.0-C1*a2*px*2.0j-S1*a2*py*2.0j)+np.angle(d1*pz*2.0j-a2**2*1j+a3**2*1j-d1**2*1j-pz**2*1j-C1**2*px**2*1j+np.sqrt(0j+(a2*d1*2.0-a2*pz*2.0)**2+(C1*a2*px*2.0+S1*a2*py*2.0)**2-(d1*pz*-2.0+a2**2-a3**2+d1**2+pz**2+C1**2*px**2+S1**2*py**2+C1*S1*px*py*2.0)**2)-S1**2*py**2*1j-C1*S1*px*py*2.0j)  # noqa

                    S2 = np.sin(theta[1])
                    C2 = np.cos(theta[1])

                    if sol[2] == 1:
                        theta[2] = -np.angle(a2*-1j-C2*d1+C2*pz-S2*d1*1j+S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j-np.sqrt(0j+(-a2-S2*d1+S2*pz+C1*C2*px+C2*S1*py)**2+(C2*d1-C2*pz+C1*S2*px+S1*S2*py)**2-a3**2))  # noqa
                    else:
                        theta[2] = -np.angle(a2*-1j-C2*d1+C2*pz-S2*d1*1j+S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j+np.sqrt(0j+(-a2-S2*d1+S2*pz+C1*C2*px+C2*S1*py)**2+(C2*d1-C2*pz+C1*S2*px+S1*S2*py)**2-a3**2))  # noqa

            elif self.ikineType == 'offset':
                # General case with 6 length parameters
                a1 = self.links[0].a
                a2 = self.links[1].a
                a3 = self.links[2].a
                d1 = self.links[0].d
                d2 = self.links[1].d
                d3 = self.links[2].d

                px = Ti[0, 3]
                py = Ti[1, 3]
                pz = Ti[2, 3]

                # Autogenerated code
                if self.links[0].alpha < 0:

                    if sol[0] == 1:
                        theta[0] = -np.angle(-px+py*1j)+np.angle(d2*1j+d3*1j-np.sqrt(0j+d2*d3*-2.0-d2**2-d3**2+px**2+py**2))  # noqa
                    else:
                        theta[0] = np.angle(d2*1j+d3*1j+np.sqrt(0j+d2*d3*-2.0-d2**2-d3**2+px**2+py**2))-np.angle(-px+py*1j)  # noqa

                    S1 = np.sin(theta[0])
                    C1 = np.cos(theta[0])

                    if sol[1] == 1:
                        theta[1] = np.angle(d1*pz*2.0j-np.sqrt(0j+d1*pz**3*4.0+d1**3*pz*4.0-a1**4-a2**4-a3**4-d1**4-py**4-pz**4-C1**4*px**4+C1**2*py**4*2.0-C1**4*py**4+a1**2*a2**2*2.0+a1**2*a3**2*2.0+a2**2*a3**2*2.0-a1**2*d1**2*2.0+a2**2*d1**2*2.0+a3**2*d1**2*2.0-a1**2*py**2*6.0-a2**2*py**2*2.0+a3**2*py**2*2.0-a1**2*pz**2*2.0+a2**2*pz**2*2.0+a3**2*pz**2*2.0-d1**2*py**2*2.0-d1**2*pz**2*6.0-py**2*pz**2*2.0+d1*py**2*pz*4.0+C1**3*a1*px**3*4.0+S1**3*a1*py**3*4.0-C1**2*a1**2*px**2*6.0+C1**2*a2**2*px**2*2.0+C1**2*a3**2*px**2*2.0+C1**2*a1**2*py**2*6.0+C1**2*a2**2*py**2*1.0e1-C1**2*a3**2*py**2*2.0-C1**4*a2**2*py**2*1.2e1+C1**6*a2**2*py**2*4.0-C1**2*d1**2*px**2*2.0+C1**2*d1**2*py**2*2.0-C1**2*px**2*py**2*6.0+C1**4*px**2*py**2*6.0-C1**2*px**2*pz**2*2.0+C1**2*py**2*pz**2*2.0+S1**6*a2**2*py**2*4.0+C1*a1**3*px*4.0+S1*a1**3*py*4.0+a1**2*d1*pz*4.0-a2**2*d1*pz*4.0-a3**2*d1*pz*4.0-C1*a1*a2**2*px*4.0-C1*a1*a3**2*px*4.0-C1*S1*px**3*py*4.0+C1*a1*d1**2*px*4.0+C1*a1*px*py**2*1.2e1+C1*a1*px*pz**2*4.0+S1*a1*a2**2*py*4.0-S1*a1*a3**2*py*4.0+S1*a1*d1**2*py*4.0+S1*a1*px**2*py*4.0+S1*a1*py*pz**2*4.0-C1*S1**3*px*py**3*4.0+C1*S1**3*px**3*py*4.0-C1**3*a1*px*py**2*1.2e1-S1**3*a1*a2**2*py*8.0+C1**2*d1*px**2*pz*4.0-C1**2*d1*py**2*pz*4.0-S1**3*a1*px**2*py*4.0-C1*a1*d1*px*pz*8.0-S1*a1*d1*py*pz*8.0-C1*S1*a1**2*px*py*1.2e1-C1*S1*a2**2*px*py*4.0+C1*S1*a3**2*px*py*4.0-C1*S1*d1**2*px*py*4.0-C1*S1*px*py*pz**2*4.0-C1**2*S1*a1*a2**2*py*8.0+C1**2*S1*a1*px**2*py*8.0+C1*S1**3*a2**2*px*py*8.0+C1**3*S1*a2**2*px*py*8.0+C1*S1*d1*px*py*pz*8.0)-a1**2*1j-a2**2*1j+a3**2*1j-d1**2*1j-py**2*1j-pz**2*1j-C1**2*px**2*1j+C1**2*py**2*1j+C1*a1*px*2.0j+S1*a1*py*2.0j-C1*S1*px*py*2.0j)-np.angle(-a2*(a1*-1j+d1-pz+C1*px*1j+S1**3*py*1j+C1**2*S1*py*1j))  # noqa
                    else:
                        theta[1] = np.angle(d1*pz*2.0j+np.sqrt(0j+d1*pz**3*4.0+d1**3*pz*4.0-a1**4-a2**4-a3**4-d1**4-py**4-pz**4-C1**4*px**4+C1**2*py**4*2.0-C1**4*py**4+a1**2*a2**2*2.0+a1**2*a3**2*2.0+a2**2*a3**2*2.0-a1**2*d1**2*2.0+a2**2*d1**2*2.0+a3**2*d1**2*2.0-a1**2*py**2*6.0-a2**2*py**2*2.0+a3**2*py**2*2.0-a1**2*pz**2*2.0+a2**2*pz**2*2.0+a3**2*pz**2*2.0-d1**2*py**2*2.0-d1**2*pz**2*6.0-py**2*pz**2*2.0+d1*py**2*pz*4.0+C1**3*a1*px**3*4.0+S1**3*a1*py**3*4.0-C1**2*a1**2*px**2*6.0+C1**2*a2**2*px**2*2.0+C1**2*a3**2*px**2*2.0+C1**2*a1**2*py**2*6.0+C1**2*a2**2*py**2*1.0e1-C1**2*a3**2*py**2*2.0-C1**4*a2**2*py**2*1.2e1+C1**6*a2**2*py**2*4.0-C1**2*d1**2*px**2*2.0+C1**2*d1**2*py**2*2.0-C1**2*px**2*py**2*6.0+C1**4*px**2*py**2*6.0-C1**2*px**2*pz**2*2.0+C1**2*py**2*pz**2*2.0+S1**6*a2**2*py**2*4.0+C1*a1**3*px*4.0+S1*a1**3*py*4.0+a1**2*d1*pz*4.0-a2**2*d1*pz*4.0-a3**2*d1*pz*4.0-C1*a1*a2**2*px*4.0-C1*a1*a3**2*px*4.0-C1*S1*px**3*py*4.0+C1*a1*d1**2*px*4.0+C1*a1*px*py**2*1.2e1+C1*a1*px*pz**2*4.0+S1*a1*a2**2*py*4.0-S1*a1*a3**2*py*4.0+S1*a1*d1**2*py*4.0+S1*a1*px**2*py*4.0+S1*a1*py*pz**2*4.0-C1*S1**3*px*py**3*4.0+C1*S1**3*px**3*py*4.0-C1**3*a1*px*py**2*1.2e1-S1**3*a1*a2**2*py*8.0+C1**2*d1*px**2*pz*4.0-C1**2*d1*py**2*pz*4.0-S1**3*a1*px**2*py*4.0-C1*a1*d1*px*pz*8.0-S1*a1*d1*py*pz*8.0-C1*S1*a1**2*px*py*1.2e1-C1*S1*a2**2*px*py*4.0+C1*S1*a3**2*px*py*4.0-C1*S1*d1**2*px*py*4.0-C1*S1*px*py*pz**2*4.0-C1**2*S1*a1*a2**2*py*8.0+C1**2*S1*a1*px**2*py*8.0+C1*S1**3*a2**2*px*py*8.0+C1**3*S1*a2**2*px*py*8.0+C1*S1*d1*px*py*pz*8.0)-a1**2*1j-a2**2*1j+a3**2*1j-d1**2*1j-py**2*1j-pz**2*1j-C1**2*px**2*1j+C1**2*py**2*1j+C1*a1*px*2.0j+S1*a1*py*2.0j-C1*S1*px*py*2.0j)-np.angle(-a2*(a1*-1j+d1-pz+C1*px*1j+S1**3*py*1j+C1**2*S1*py*1j))  # noqa

                    S2 = np.sin(theta[1])
                    C2 = np.cos(theta[1])

                    if sol[2] == 1:
                        theta[2] = np.angle(a3*1j-np.sqrt(0j+d1*pz*-2.0+a1**2+a2**2-a3**2+d1**2+py**2+pz**2+C1**2*px**2-C1**2*py**2+C2*a1*a2*2.0-C1*a1*px*2.0-S2*a2*d1*2.0-S1*a1*py*2.0+S2*a2*pz*2.0-C1*C2*a2*px*2.0-C2*S1*a2*py*2.0+C1*S1*px*py*2.0+0j))-np.angle(a2*-1j-C2*a1*1j+C2*d1-C2*pz+S2*a1+S2*d1*1j-S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)  # noqa
                    else:
                        theta[2] = -np.angle(a2*-1j-C2*a1*1j+C2*d1-C2*pz+S2*a1+S2*d1*1j-S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j+np.sqrt(0j+d1*pz*-2.0+a1**2+a2**2-a3**2+d1**2+py**2+pz**2+C1**2*px**2-C1**2*py**2+C2*a1*a2*2.0-C1*a1*px*2.0-S2*a2*d1*2.0-S1*a1*py*2.0+S2*a2*pz*2.0-C1*C2*a2*px*2.0-C2*S1*a2*py*2.0+C1*S1*px*py*2.0))  # noqa

                else:
                    if sol[0] == 1:
                        theta[0] = -np.angle(px-py*1j)+np.angle(d2*1j+d3*1j-np.sqrt(0j+d2*d3*-2.0-d2**2-d3**2+px**2+py**2))  # noqa
                    else:
                        theta[0] = -np.angle(px-py*1j)+np.angle(d2*1j+d3*1j+np.sqrt(0j+d2*d3*-2.0-d2**2-d3**2+px**2+py**2))  # noqa

                    S1 = np.sin(theta[0])
                    C1 = np.cos(theta[0])

                    if sol[1] == 1:
                        theta[1] = np.angle(d1*pz*2.0j-np.sqrt(0j+d1*pz**3*4.0+d1**3*pz*4.0-a1**4-a2**4-a3**4-d1**4-py**4-pz**4-C1**4*px**4+C1**2*py**4*2.0-C1**4*py**4+a1**2*a2**2*2.0+a1**2*a3**2*2.0+a2**2*a3**2*2.0-a1**2*d1**2*2.0+a2**2*d1**2*2.0+a3**2*d1**2*2.0-a1**2*py**2*6.0-a2**2*py**2*2.0+a3**2*py**2*2.0-a1**2*pz**2*2.0+a2**2*pz**2*2.0+a3**2*pz**2*2.0-d1**2*py**2*2.0-d1**2*pz**2*6.0-py**2*pz**2*2.0+d1*py**2*pz*4.0+C1**3*a1*px**3*4.0+S1**3*a1*py**3*4.0-C1**2*a1**2*px**2*6.0+C1**2*a2**2*px**2*2.0+C1**2*a3**2*px**2*2.0+C1**2*a1**2*py**2*6.0+C1**2*a2**2*py**2*1.0e1-C1**2*a3**2*py**2*2.0-C1**4*a2**2*py**2*1.2e1+C1**6*a2**2*py**2*4.0-C1**2*d1**2*px**2*2.0+C1**2*d1**2*py**2*2.0-C1**2*px**2*py**2*6.0+C1**4*px**2*py**2*6.0-C1**2*px**2*pz**2*2.0+C1**2*py**2*pz**2*2.0+S1**6*a2**2*py**2*4.0+C1*a1**3*px*4.0+S1*a1**3*py*4.0+a1**2*d1*pz*4.0-a2**2*d1*pz*4.0-a3**2*d1*pz*4.0-C1*a1*a2**2*px*4.0-C1*a1*a3**2*px*4.0-C1*S1*px**3*py*4.0+C1*a1*d1**2*px*4.0+C1*a1*px*py**2*1.2e1+C1*a1*px*pz**2*4.0+S1*a1*a2**2*py*4.0-S1*a1*a3**2*py*4.0+S1*a1*d1**2*py*4.0+S1*a1*px**2*py*4.0+S1*a1*py*pz**2*4.0-C1*S1**3*px*py**3*4.0+C1*S1**3*px**3*py*4.0-C1**3*a1*px*py**2*1.2e1-S1**3*a1*a2**2*py*8.0+C1**2*d1*px**2*pz*4.0-C1**2*d1*py**2*pz*4.0-S1**3*a1*px**2*py*4.0-C1*a1*d1*px*pz*8.0-S1*a1*d1*py*pz*8.0-C1*S1*a1**2*px*py*1.2e1-C1*S1*a2**2*px*py*4.0+C1*S1*a3**2*px*py*4.0-C1*S1*d1**2*px*py*4.0-C1*S1*px*py*pz**2*4.0-C1**2*S1*a1*a2**2*py*8.0+C1**2*S1*a1*px**2*py*8.0+C1*S1**3*a2**2*px*py*8.0+C1**3*S1*a2**2*px*py*8.0+C1*S1*d1*px*py*pz*8.0)-a1**2*1j-a2**2*1j+a3**2*1j-d1**2*1j-py**2*1j-pz**2*1j-C1**2*px**2*1j+C1**2*py**2*1j+C1*a1*px*2.0j+S1*a1*py*2.0j-C1*S1*px*py*2.0j)-np.angle(-a2*(a1*-1j-d1+pz+C1*px*1j+S1**3*py*1j+C1**2*S1*py*1j))  # noqa
                    else:
                        theta[1] = np.angle(d1*pz*2.0j+np.sqrt(0j+d1*pz**3*4.0+d1**3*pz*4.0-a1**4-a2**4-a3**4-d1**4-py**4-pz**4-C1**4*px**4+C1**2*py**4*2.0-C1**4*py**4+a1**2*a2**2*2.0+a1**2*a3**2*2.0+a2**2*a3**2*2.0-a1**2*d1**2*2.0+a2**2*d1**2*2.0+a3**2*d1**2*2.0-a1**2*py**2*6.0-a2**2*py**2*2.0+a3**2*py**2*2.0-a1**2*pz**2*2.0+a2**2*pz**2*2.0+a3**2*pz**2*2.0-d1**2*py**2*2.0-d1**2*pz**2*6.0-py**2*pz**2*2.0+d1*py**2*pz*4.0+C1**3*a1*px**3*4.0+S1**3*a1*py**3*4.0-C1**2*a1**2*px**2*6.0+C1**2*a2**2*px**2*2.0+C1**2*a3**2*px**2*2.0+C1**2*a1**2*py**2*6.0+C1**2*a2**2*py**2*1.0e1-C1**2*a3**2*py**2*2.0-C1**4*a2**2*py**2*1.2e1+C1**6*a2**2*py**2*4.0-C1**2*d1**2*px**2*2.0+C1**2*d1**2*py**2*2.0-C1**2*px**2*py**2*6.0+C1**4*px**2*py**2*6.0-C1**2*px**2*pz**2*2.0+C1**2*py**2*pz**2*2.0+S1**6*a2**2*py**2*4.0+C1*a1**3*px*4.0+S1*a1**3*py*4.0+a1**2*d1*pz*4.0-a2**2*d1*pz*4.0-a3**2*d1*pz*4.0-C1*a1*a2**2*px*4.0-C1*a1*a3**2*px*4.0-C1*S1*px**3*py*4.0+C1*a1*d1**2*px*4.0+C1*a1*px*py**2*1.2e1+C1*a1*px*pz**2*4.0+S1*a1*a2**2*py*4.0-S1*a1*a3**2*py*4.0+S1*a1*d1**2*py*4.0+S1*a1*px**2*py*4.0+S1*a1*py*pz**2*4.0-C1*S1**3*px*py**3*4.0+C1*S1**3*px**3*py*4.0-C1**3*a1*px*py**2*1.2e1-S1**3*a1*a2**2*py*8.0+C1**2*d1*px**2*pz*4.0-C1**2*d1*py**2*pz*4.0-S1**3*a1*px**2*py*4.0-C1*a1*d1*px*pz*8.0-S1*a1*d1*py*pz*8.0-C1*S1*a1**2*px*py*1.2e1-C1*S1*a2**2*px*py*4.0+C1*S1*a3**2*px*py*4.0-C1*S1*d1**2*px*py*4.0-C1*S1*px*py*pz**2*4.0-C1**2*S1*a1*a2**2*py*8.0+C1**2*S1*a1*px**2*py*8.0+C1*S1**3*a2**2*px*py*8.0+C1**3*S1*a2**2*px*py*8.0+C1*S1*d1*px*py*pz*8.0)-a1**2*1j-a2**2*1j+a3**2*1j-d1**2*1j-py**2*1j-pz**2*1j-C1**2*px**2*1j+C1**2*py**2*1j+C1*a1*px*2.0j+S1*a1*py*2.0j-C1*S1*px*py*2.0j)-np.angle(-a2*(a1*-1j+d1-pz+C1*px*1j+S1**3*py*1j+C1**2*S1*py*1j))  # noqa
                        print(theta[1])

                    S2 = np.sin(theta[1])
                    C2 = np.cos(theta[1])

                    if sol[2] == 1:
                        theta[2] = np.angle(a3*1j-np.sqrt(0j+d1*pz*-2.0+a1**2+a2**2-a3**2+d1**2+py**2+pz**2+C1**2*px**2-C1**2*py**2+C2*a1*a2*2.0-C1*a1*px*2.0+S2*a2*d1*2.0-S1*a1*py*2.0-S2*a2*pz*2.0-C1*C2*a2*px*2.0-C2*S1*a2*py*2.0+C1*S1*px*py*2.0))-np.angle(a2*-1j-C2*a1*1j-C2*d1+C2*pz+S2*a1-S2*d1*1j+S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)  # noqa
                    else:
                        theta[2] = -np.angle(a2*-1j-C2*a1*1j-C2*d1+C2*pz+S2*a1-S2*d1*1j+S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j+np.sqrt(0j+d1*pz*-2.0+a1**2+a2**2-a3**2+d1**2+py**2+pz**2+C1**2*px**2-C1**2*py**2+C2*a1*a2*2.0-C1*a1*px*2.0+S2*a2*d1*2.0-S1*a1*py*2.0-S2*a2*pz*2.0-C1*C2*a2*px*2.0-C2*S1*a2*py*2.0+C1*S1*px*py*2.0))  # noqa

            elif self.ikineType == 'rrp':
                # RRP (Stanford arm like)
                px = Ti[0, 3]
                py = Ti[1, 3]
                pz = Ti[2, 3]
                d1 = self.links[0].d
                d2 = self.links[1].d

                # Autogenerated code
                if self.links[0].alpha < 0:
                    if sol[0] == 1:
                        theta[0] = -np.angle(-px+py*1j)+np.angle(d2*1j-np.sqrt(0j+-d2**2+px**2+py**2))  # noqa
                    else:
                        theta[0] = np.angle(d2*1j+np.sqrt(0j+-d2**2+px**2+py**2))-np.angle(-px+py*1j)  # noqa

                    S1 = np.sin(theta[0])
                    C1 = np.cos(theta[0])

                    if sol[1] == 1:
                        theta[1] = np.angle(d1-pz-C1*px*1j-S1*py*1j)  # noqa
                    else:
                        theta[1] = np.angle(-d1+pz+C1*px*1j+S1*py*1j)  # noqa

                    S2 = np.sin(theta[1])
                    C2 = np.cos(theta[1])

                    theta[2] = -C2*d1+C2*pz+C1*S2*px+S1*S2*py  # noqa

                else:
                    if sol[0] == 1:
                        theta[0] = -np.angle(px-py*1j)+np.angle(d2*1j-np.sqrt(0j+-d2**2+px**2+py**2))  # noqa
                    else:
                        theta[0] = -np.angle(px-py*1j)+np.angle(d2*1j+np.sqrt(0j+-d2**2+px**2+py**2))  # noqa

                    S1 = np.sin(theta[0])
                    C1 = np.cos(theta[0])

                    if sol[1] == 1:
                        theta[1] = np.angle(-d1+pz-C1*px*1j-S1*py*1j)  # noqa
                    else:
                        theta[1] = np.angle(d1-pz+C1*px*1j+S1*py*1j)  # noqa

                    print(theta[1])

                    S2 = np.sin(theta[1])
                    C2 = np.cos(theta[1])

                    theta[2] = -C2*d1+C2*pz-C1*S2*px-S1*S2*py  # noqa

            if not np.all(np.isnan(theta)):
                # Solve for the wrist rotation
                # We need to account for some random translations between the
                # first and last 3 joints (d4) and also d6,a6,alpha6 in the
                # final frame.

                # Transform of first 3 joints
                T13 = self.A([0, 2], theta)

                # T = T13 * Tz(d4) * R * Tz(d6) Tx(a5)
                Td4 = SE3(0, 0, self.links[3].d)      # Tz(d4)

                # Tz(d6) Tx(a5) Rx(alpha6)
                Tt = SE3(self.links[5].a, 0, self.links[5].d) * \
                    SE3.Rx(self.links[5].alpha)

                R = np.linalg.inv(Td4.A) @ np.linalg.inv(T13.A) @ Ti @ \
                    np.linalg.inv(Tt.A)

                # The spherical wrist implements Euler angles
                if sol[2] == 1:
                    theta[3:6] = tr2eul(R, flip=True)
                else:
                    theta[3:6] = tr2eul(R)

                if self.links[3].alpha > 0:
                    theta[4] = -theta[4]

                # Remove the link offset angles
                for k in range(self.n):
                    theta[k] -= self.links[k].offset

                q[j, :] = theta
            else:
                err.append('point not reachable')

        if trajn == 1:
            return q[0, :], err
        else:
            return q, err

    def _is_simple(self):
        L = self.links
        alpha = [-np.pi / 2, 0, np.pi / 2]
        s = (L[1].d == 0 and L[2].d == 0) and ((
            L[0].alpha == alpha[0]
            and L[1].alpha == alpha[1]
            and L[2].alpha == alpha[2]
        ) or (
            L[0].alpha == -alpha[0]
            and L[1].alpha == -alpha[1]
            and L[2].alpha == -alpha[2]
        )) and \
            (not L[0].sigma and not L[1].sigma and not L[2].sigma) and \
            L[0].a == 0

        return s

    def _is_offset(self):
        L = self.links
        alpha = [-np.pi / 2, 0, np.pi / 2]
        s = ((
            L[0].alpha == alpha[0]
            and L[1].alpha == alpha[1]
            and L[2].alpha == alpha[2]
        ) or (
            L[0].alpha == -alpha[0]
            and L[1].alpha == -alpha[1]
            and L[2].alpha == -alpha[2]
        )) and (not L[0].sigma and not L[1].sigma and not L[2].sigma)

        return s

    def _is_rrp(self):
        L = self.links
        alpha = [-np.pi / 2, np.pi / 2, 0]
        s = (L[1].a == 0 and L[2].a == 0) and ((
            L[0].alpha == alpha[0]
            and L[1].alpha == alpha[1]
            and L[2].alpha == alpha[2]
        ) or (
            L[0].alpha == -alpha[0]
            and L[1].alpha == -alpha[1]
            and L[2].alpha == -alpha[2]
        )) and not L[0].sigma and not L[1].sigma and L[2].sigma

        return s

    def _is_puma(self):
        L = self.links
        alpha = [np.pi / 2, 0, -np.pi / 2]
        s = (
            L[1].d == 0
            and L[0].a == 0
            and not L[2].d == 0
            and not L[2].a == 0 and (
                L[0].alpha == alpha[0]
                and L[1].alpha == alpha[1]
                and L[2].alpha == alpha[2]
            ) and (not L[0].sigma and not L[1].sigma and not L[2].sigma))

        return s

    def ikinem(self, T, q0=None, pweight=1.0, stiffness=0.0,
               qlimits=True, ilimit=1000, nolm=False):
        """
        Numerical inverse kinematics with joint limits
        q, success, err = ikinem(T) is the joint coordinates corresponding to
        the robot end-effector pose T which is a homogenenous transform.

        q, success, err = R.ikinem(T, q0, pweight, stiffness, qlimits, ilimit,
        nolm) as above except with options defined such as the initial
        estimate of the joint coordinates q0.

        Trajectory operation:
        In all cases if T is 4x4xm it is taken as a homogeneous transform
        sequence and ikinem(T) returns the joint coordinates corresponding to
        each of the transforms in the sequence. q is mxn where n is the number
        of robot joints. The initial estimate of q for each time step is taken
        as the solution from the previous time step.

        :param T: The desired end-effector pose
        :type T: SE3 or SE3 trajectory
        :param pweight: weighting on position error norm compared to rotation
            error (default 1)
        :type pweight: float
        :param stiffness: Stiffness used to impose a smoothness contraint on
            joint angles, useful when n is large (default 0)
        :type stiffness: float
        :param qlimits: Enforce joint limits (default True)
        :type qlimits: bool
        :param ilimit: Iteration limit (default 1000)
        :type ilimit: bool
        :param nolm: Disable Levenberg-Marquadt
        :type nolm: bool

        :retrun q: The calculated joint values
        :rtype q: float ndarray(n)
        :retrun success: IK solved (True) or failed (False)
        :rtype success: bool
        :retrun error: Final pose error
        :rtype error: float

        :notes:
            - PROTOTYPE CODE UNDER DEVELOPMENT, intended to do numerical
              inverse kinematics with joint limits
            - The inverse kinematic solution is generally not unique, and
              depends on the initial guess q0 (defaults to 0).
            - The function to be minimized is highly nonlinear and the
              solution is often trapped in a local minimum, adjust q0 if this
              happens.
            - The default value of q0 is zero which is a poor choice for most
              manipulators (eg. puma560, twolink) since it corresponds to a
              kinematic singularity.
            - Such a solution is completely general, though much less
              efficient than specific inverse kinematic solutions derived
              symbolically, like ikine6s or ikine3.
            - Uses Levenberg-Marquadt minimizer LMFsolve if it can be found,
              if 'nolm' is not given, and 'qlimits' false
            - The error function to be minimized is computed on the norm of
              the error between current and desired tool pose.  This norm is
              computed from distances and angles and 'pweight' can be used to
              scale the position error norm to be congruent with rotation
              error norm.
            - This approach allows a solution to obtained at a singularity,
              but the joint angles within the null space are arbitrarily
              assigned.
            - Joint offsets, if defined, are added to the inverse kinematics
              to generate q.
            - Joint limits become explicit contraints if 'qlimits' is set.

        """

        if not isinstance(T, SE3):
            T = SE3(T)

        trajn = len(T)

        if q0 is None:
            q0 = np.zeros((trajn, self.n))

        verifymatrix(q0, (trajn, self.n))

        qt = np.zeros((trajn, self.n))
        success = []
        err = []
        col = 2

        # Define the cost function to minimise
        def cost(q, T, pweight, col, stiffness):
            Tq = self.fkine(q)

            # find the pose error in SE(3)
            dT = T.t - Tq.t

            # translation error
            E = np.linalg.norm(dT) * pweight

            # Rotation error
            # Find dot product of
            dd = np.dot(T.A[0:3, col], Tq.A[0:3, col])
            E += np.arccos(dd)**2 * 1000

            if stiffness > 0:
                # Enforce a continuity constraint on joints, minimum bend
                E += np.sum(np.diff(q)**2) * stiffness

            return E

        for i in range(trajn):

            Ti = T[i]

            if qlimits:
                bnds = Bounds(self.qlim[0, :], self.qlim[1, :])

                res = minimize(
                    lambda q: cost(q, Ti, pweight, col, stiffness),
                    q0[i, :], bounds=bnds,
                    options={'gtol': 1e-6, 'maxiter': ilimit})
            else:
                # No joint limits, unconstrained optimization
                res = minimize(
                    lambda q: cost(q, Ti, pweight, col, stiffness),
                    q0[i, :],
                    options={'gtol': 1e-6, 'maxiter': ilimit})

            if res.success and i < trajn - 1:
                q0[i + 1, :] = res.x

            qt[i, :] = res.x
            success.append(res.success)
            err.append(res.fun)

        if trajn == 1:
            return qt[0, :], success[0], err[0]
        else:
            return qt, success, err

    def ikunc(self, T, q0=None, ilimit=1000):
        """
        Inverse manipulator by optimization without joint limits

        q, success, err = ikunc(T) are the joint coordinates (n) corresponding
        to the robot end-effector pose T which is an SE3 object or
        homogenenous transform matrix (4x4), and n is the number of robot
        joints. Also returns success and err which is the scalar final value
        of the objective function.

        q, success, err = robot.ikunc(T, q0, ilimit) as above but specify the
        initial joint coordinates q0 used for the minimisation.

        Trajectory operation:
        In all cases if T is a vector of SE3 objects (m) or a homogeneous
        transform sequence (4x4xm) then returns the joint coordinates
        corresponding to each of the transforms in the sequence. q is mxn
        where n is the number of robot joints. The initial estimate of q
        for each time step is taken as the solution from the previous time
        step.

        :param T: The desired end-effector pose
        :type T: SE3 or SE3 trajectory
        :param ilimit: Iteration limit (default 1000)
        :type ilimit: bool

        :retrun q: The calculated joint values
        :rtype q: float ndarray(n)
        :retrun success: IK solved (True) or failed (False)
        :rtype success: bool
        :retrun error: Final pose error
        :rtype error: float

        :notes:
            - Joint limits are not considered in this solution.
            - Can be used for robots with arbitrary degrees of freedom.
            - In the case of multiple feasible solutions, the solution
              returned depends on the initial choice of q0
            - Works by minimizing the error between the forward kinematics of
              the joint angle solution and the end-effector frame as an
              optimisation.
            - The objective function (error) is described as:
              sumsqr( (inv(T)*robot.fkine(q) - eye(4)) * omega )
              Where omega is some gain matrix, currently not modifiable.

        """

        if not isinstance(T, SE3):
            T = SE3(T)

        trajn = len(T)

        if q0 is None:
            q0 = np.zeros((trajn, self.n))

        verifymatrix(q0, (trajn, self.n))

        qt = np.zeros((trajn, self.n))
        success = []
        err = []

        reach = np.sum(np.abs([self.a, self.d]))
        omega = np.diag([1, 1, 1, 3 / reach])

        def sumsqr(arr):
            return np.sum(np.power(arr, 2))

        for i in range(trajn):

            Ti = T[i]

            res = minimize(
                lambda q: sumsqr(((
                    np.linalg.inv(Ti.A) @ self.fkine(q).A) - np.eye(4)) @
                    omega),
                q0[i, :],
                options={'gtol': 1e-6, 'maxiter': ilimit})

            qt[i, :] = res.x
            success.append(res.success)
            err.append(res.fun)

        if trajn == 1:
            return qt[0, :], success[0], err[0]
        else:
            return qt, success, err

    @_check_rne
    def rne(self, q, qd=None, qdd=None, grav=None, fext=None):
        r"""
        Inverse dynamics

        :param q: The joint angles/configuration of the robot (Optional,
        if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param qd: The joint velocities of the robot
        :type qd: float ndarray(n)
        :param qdd: The joint accelerations of the robot
        :type qdd: float ndarray(n)
        :param grav: Gravity vector to overwrite robots gravity value
        :type grav: float ndarray(6)
        :param fext: Specify wrench acting on the end-effector
             :math:`W=[F_x F_y F_z M_x M_y M_z]`
        :type fext: float ndarray(6)

        ``tau = rne(q, qd, qdd, grav, fext)`` is the joint torque required for
        the robot to achieve the specified joint position ``q`` (1xn), velocity
        ``qd`` (1xn) and acceleration ``qdd`` (1xn), where n is the number of
        robot joints. ``fext`` describes the wrench acting on the end-effector

        Trajectory operation:
        If q, qd and qdd (mxn) are matrices with m cols representing a
        trajectory then tau (mxn) is a matrix with cols corresponding to each
        trajectory step.

        :notes:
            - The torque computed contains a contribution due to armature
              inertia and joint friction.
            - If a model has no dynamic parameters set the result is zero.

        """

        trajn = 1

        try:
            q = getvector(q, self.n, 'row')
            qd = getvector(qd, self.n, 'row')
            qdd = getvector(qdd, self.n, 'row')
        except ValueError:
            trajn = q.shape[0]
            verifymatrix(q, (trajn, self.n))
            verifymatrix(qd, (trajn, self.n))
            verifymatrix(qdd, (trajn, self.n))

        if grav is None:
            grav = self.gravity

        # The c function doesn't handle base rotation, so we need to hack the
        # gravity vector instead
        grav = self.base.R.T @ grav
        grav = getvector(grav, 3)

        if fext is None:
            fext = np.zeros(6)
        else:
            fext = getvector(fext, 6)

        tau = np.zeros((trajn, self.n))

        for i in range(trajn):
            tau[i, :] = frne(
                self._rne_ob, q[i, :], qd[i, :], qdd[i, :], grav, fext)

        if trajn == 1:
            return tau[0, :]
        else:
            return tau

    def delete_rne(self):
        """
        Frees the memory holding the robot object in c if the robot object
        has been initialised in c.
        """
        if self._rne_init:
            delete(self._rne_ob)
            self._rne_init = False
            self._rne_changed = False
            self._rne_ob = None

    def jacob_dot(self, q=None, qd=None):
        '''
        Jqd = jacob_dot(q, qd) is the product (6) of the derivative of the
        manipulator Jacobian (in the world frame) and the joint rates.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param qd: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored qd values).
        :type qd: float ndarray(n)

        :retrun Jdot: The derivative of the manipulator Jacobian
        :rtype Jdot: float ndarray(n)

        :notes:
            - This term appears in the formulation for operational space
              control xdd = J(q)qdd + Jdot(q)qd
            - Written as per the reference and not very efficient.

        :references:
            - Fundamentals of Robotics Mechanical Systems (2nd ed)
              J. Angleles, Springer 2003.
            - A unified approach for motion and force control of robot
              manipulators: The operational space formulation
              O Khatib, IEEE Journal on Robotics and Automation, 1987.

        '''

        if q is None:
            q = self.q

        if qd is None:
            qd = self.qd

        q = getvector(q, self.n)
        qd = getvector(qd, self.n)

        # Using the notation of Angeles:
        #   [Q,a] ~ [R,t] the per link transformation
        #   P ~ R   the cumulative rotation t2r(Tj) in world frame
        #   e       the last column of P, the local frame z axis in world
        #           coordinates
        #   w       angular velocity in base frame
        #   ed      deriv of e
        #   r       is distance from final frame
        #   rd      deriv of r
        #   ud      ??

        Q = np.zeros((3, 3, self.n))
        a = np.zeros((3, self.n))
        P = np.zeros((3, 3, self.n))
        e = np.zeros((3, self.n))
        w = np.zeros((3, self.n))
        ed = np.zeros((3, self.n))
        rd = np.zeros((3, self.n))
        r = np.zeros((3, self.n))
        ud = np.zeros((3, self.n))
        v = np.zeros((6, self.n))

        for i in range(self.n):
            T = self.links[i].A(q[i])
            Q[:, :, i] = T.R
            a[:, i] = T.t

        P[:, :, 0] = Q[:, :, 0]
        e[:, 0] = [0, 0, 1]

        for i in range(1, self.n):
            P[:, :, i] = P[:, :, i - 1] @ Q[:, :, i]
            e[:, i] = P[:, 2, i]

        # Step 1
        w[:, 0] = qd[0] * e[:, 0]

        for i in range(self.n - 1):
            w[:, i + 1] = (
                qd[i + 1] *
                np.array([0, 0, 1]) +
                Q[:, :, i].T @ w[:, i])

        # Step 2
        ed[:, 0] = np.array([0, 0, 1])

        for i in range(1, self.n):
            ed[:, i] = np.cross(w[:, i], e[:, i])

        # Step 3
        rd[:, self.n - 1] = np.cross(w[:, self.n - 1], a[:, self.n - 1])

        for i in range(self.n - 2, -1, -1):
            rd[:, i] = np.cross(w[:, i], a[:, i]) + Q[:, :, i] @ rd[:, i + 1]

        r[:, self.n - 1] = a[:, self.n - 1]

        for i in range(self.n - 2, -1, -1):
            r[:, i] = a[:, i] + Q[:, :, i] @ r[:, i + 1]

        ud[:, 0] = np.cross(e[:, 0], rd[:, 0])

        for i in range(1, self.n):
            ud[:, i] = \
                np.cross(ed[:, i], r[:, i]) + np.cross(e[:, i], rd[:, i])

        # Step 4
        # Swap ud and ed
        v[:, self.n - 1] = \
            qd[self.n - 1] * np.r_[ud[:, self.n - 1], ed[:, self.n - 1]]

        for i in range(self.n - 2, -1, -1):
            Ui = np.r_[
                np.c_[Q[:, :, i], np.zeros((3, 3))],
                np.c_[np.zeros((3, 3)), Q[:, :, i]]]

            v[:, i] = (
                qd[i] *
                np.r_[ud[:, i], ed[:, i]] +
                Ui @ v[:, i + 1])

        Jdot = v[:, 0]

        return Jdot

    def maniplty(self, q=None, method='yoshikawa', axes=[1, 1, 1, 1, 1, 1]):
        '''
        Manipulability measure

        m = maniplty(q) is the yoshikawa manipulability index (scalar) for the
        robot at the joint configuration q (n) where n is the number of robot
        joints.  It indicates dexterity, that is, how isotropic the robot's
        motion is with respect to the 6 degrees of Cartesian motion. The
        measure is high when the manipulator is capable of equal motion in all
        directions and low when the manipulator is close to a singularity.
        Yoshikawa's manipulability measure is based on the shape of the
        velocity ellipsoid and depends only on kinematic parameters.

        m = maniplty(q, method='asada') as above except computes the asada
        manipulability measure. Asada's manipulability measure is based on the
        shape of the acceleration ellipsoid which in turn is a function of the
        Cartesian inertia matrix and the dynamic parameters. The scalar
        measure computed here is the ratio of the smallest/largest ellipsoid
        axis. Ideally the ellipsoid would be spherical, giving a ratio of 1,
        but in practice will be less than 1.

        m = maniplty(q, method, axes) as above except axes specity which of
        the 6 degrees-of-freedom to concider in the measurement. For example
        set axes=[1, 1, 1, 0, 0, 0] to consider only translation or
        axes=[0, 0, 0, 1, 1, 1] to consider only rotation. Defaults to all
        motion.

        If q is a matrix (mxn) then m (mx1) is a vector of manipulability
        indices for each joint configuration specified by a row of q.

        [m, CI] = maniplty(q, OPTIONS) as above, but for the case of the Asada
        measure returns the Cartesian inertia matrix CI.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param method: Which method to use, 'yoshikawa' (default) or 'asada'
        :type method: string
        :param axes: The degrees-of-freedom to be included for manipulability
        :type axes: int list

        :notes:
            - The 'all' option includes rotational and translational
              dexterity, but this involves adding different units. It can be
              more useful to look at the translational and rotational
              manipulability separately.
            - Examples in the RVC book (1st edition) can be replicated by
              using the 'all' option

        :references:
            - Analysis and control of robot manipulators with redundancy,
              T. Yoshikawa,
              Robotics Research: The First International Symposium
              (M. Brady and R. Paul, eds.),
              pp. 735-747, The MIT press, 1984.
            - A geometrical representation of manipulator dynamics and its
              application to arm design, H. Asada,
              Journal of Dynamic Systems, Measurement, and Control,
              vol. 105, p. 131, 1983.
            - Robotics, Vision & Control, P. Corke, Springer 2011.

        '''

        def yoshi(robot, q, axes):
            J = robot.jacob0(q)
            J = J[axes, :]
            m2 = np.linalg.det(J @ J.T)
            m2 = np.maximum(0.0, m2)  # clip it to positive
            m = np.sqrt(m2)
            return m

        def asada(robot, q, axes, dof):
            J = robot.jacob0(q)

            if np.linalg.matrix_rank(J) < 6:
                return 0, np.zeros((dof, dof))

            Ji = np.linalg.pinv(J)
            M = robot.inertia(q)
            Mx = Ji.T @ M @ Ji
            d = np.where(axes)[0]
            Mx = Mx[d]
            Mx = Mx[:, d.tolist()]
            e, _ = np.linalg.eig(Mx)
            m = np.min(e) / np.max(e)

            return m, Mx

        axes = getvector(axes, 6)
        axes = axes > 0

        trajn = 1

        if q is None:
            q = self.q

        try:
            q = getvector(q, self.n, 'row')
        except ValueError:
            trajn = q.shape[0]
            verifymatrix(q, (trajn, self.n))

        w = np.zeros(trajn)

        if method == 'yoshikawa':
            for i in range(trajn):
                w[i] = yoshi(self, q[i, :], axes)

            if trajn == 1:
                return w[0]
            else:
                return w

        elif method == 'asada':
            dof = np.sum(axes)
            mx = np.zeros((dof, dof, trajn))

            for i in range(trajn):
                w[i], mx[:, :, i] = asada(self, q[i, :], axes, dof)

            if trajn == 1:
                return w[0], mx[:, :, 0]
            else:
                return w, mx

        else:
            raise ValueError(
                'Invalid method chosen. Must be \'yoshikawa\' or \'asada\'.')

    def qmincon(self, q=None):
        '''
        qs, success, err = qmincon(q) exploits null space motion and returns
        a set of joint angles qs (n) that result in the same end-effector
        pose but are away from the joint coordinate limits. n is the number
        of robot joints. Success retruns True for successful optimisation.
        err which is the scalar final value of the objective function.

        Trajectory operation:
        In all cases if q is mxn it is taken as a pose sequence and qmincon()
        returns the adjusted joint coordinates (mxn) corresponding to each of
        the poses in the sequence.

        err and success are also m and indicate the results of optimisation
        for the corresponding trajectory step.

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :retrun qs: The calculated joint values
        :rtype qs: float ndarray(n)
        :retrun success: Optimisation solved (True) or failed (False)
        :rtype success: bool
        :retrun err: Final value of the objective function
        :rtype err: float

        :notes:
            - Robot must be redundant.

        '''

        def sumsqr(A):
            return np.sum(A**2)

        def cost(x, ub, lb, qm, N):
            return sumsqr(
                (2 * (N @ x + qm) - ub - lb) / (ub - lb))

        trajn = 1

        if q is None:
            q = self.q

        try:
            q = getvector(q, self.n, 'col')
        except ValueError:
            trajn = q.shape[1]

        qstar = np.zeros((self.n, trajn))
        error = np.zeros(trajn)
        success = np.zeros(trajn)

        lb = self.qlim[0, :]
        ub = self.qlim[1, :]

        for i in range(trajn):

            qm = q[:, i]
            J = self.jacobe(qm)

            N = null(J)

            x0 = np.zeros(N.shape[1])
            A = np.r_[N, -N]
            b = np.r_[ub - qm, qm - lb].reshape(A.shape[0],)

            con = LinearConstraint(A, -np.inf, b)

            res = minimize(
                lambda x: cost(x, ub, lb, qm, N),
                x0, constraints=con)

            qstar[:, i] = qm + N @ res.x
            error[i] = res.fun
            success[i] = res.success

        if trajn == 1:
            return qstar[:, 0], success[0], error[0]
        else:
            return qstar, success, error

    def teach(
            self, q=None, block=True, limits=None,
            jointaxes=True, eeframe=True, shadow=True, name=True):
        '''
        Graphical teach pendant

        env = teach() creates a matplotlib plot which allows the user to
        "drive" a graphical robot using a graphical slider panel. The
        robot's inital joint configuration is robot.q. This will block the
        programs execution. The plot will autoscale with an aspect ratio of 1.

        env = teach(q) as above except the robot's initial configuration is
        set to q.

        env = teach(block=False) as avove except the plot is non-blocking. Note
        that the plot will exit when the python script finishes executing.

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        :type limits: ndarray(6)
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

        :retrun: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        :notes:
            - The slider limits are derived from the joint limit properties.
              If not set then
                - For revolute joints they are assumed to be [-pi, +pi]
                - For prismatic joint they are assumed unknown and an error
                  occurs.

        '''

        if q is not None:
            self.q = q

        # try:
        return _teach(
            self, block, limits=limits,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)
        # except ModuleNotFoundError:
        #     print(
        #         'Could not find matplotlib.'
        #         ' Matplotlib required for this function')

    def plot(
            self, q=None, block=True, dt=50, limits=None,
            vellipse=False, fellipse=False,
            jointaxes=True, eeframe=True, shadow=True, name=True, movie=None):
        '''
        Graphical display and animation

        env = plot() displays a graphical view of a robot based on the
        kinematic model, at it's stored q value. A stick figure polyline
        joins the origins of the link coordinate frames. This method will be
        blocking. The plot will autoscale with an aspect ratio of 1.

        env = plot(q) as above except the robot is plotted with joint angles q

        env = plot(block=False) as avove except the plot in non-blocking. Note
        that the plot will exit when the python script finishes executing.

        env = plot(q, dt) as above except q is an mxn trajectory of joint
        angles. This creates an animation of the robot moving through the
        trajectories with a gap dt milliseconds in between.

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param dt: if q is a trajectory, this describes the delay in
            milliseconds between frames
        :type dt: int
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        :type limits: ndarray(6)
        :param vellipse: (Plot Option) Plot the velocity ellipse at the
            end-effector
        :type vellipse: bool
        :param vellipse: (Plot Option) Plot the force ellipse at the
            end-effector
        :type vellipse: bool
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
        :param movie: name of file in which to save an animated GIF 
        :type movie: str

        :retrun: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        '''

        # try:
        return _plot(
            self, block, q, dt, limits,
            vellipse=vellipse, fellipse=fellipse,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name,
            movie=movie)
        # except ModuleNotFoundError:
        #     print(
        #         'Could not find matplotlib.'
        #         ' Matplotlib required for this function')

    def vellipse(self, q=None, opt='trans', centre=[0, 0, 0]):
        '''
        Create a velocity ellipsoid object for plotting

        env = vellipse() creates a velocity ellipsoid for the robot at
        pose robot.q. The ellipsoid is centered at the origin.

        env = vellipse(q) as above except the robot is plotted with joint
        angles q

        env = vellipse(opt) as above except opt is 'trans' or 'rot' will
        plot either the translational or rotational velocity ellipsoid.

        env = vellipse(centre) as above except centre is either a 3
        vector or 'ee' which is the centre location of the ellipsoid

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param opt: 'trans' or 'rot' will plot either the translational or
            rotational velocity ellipsoid
        :type opt: string
        :param centre:
        :type centre: list or str('ee')

        :retrun: An EllipsePlot object
        :rtype: EllipsePlot

        '''

        return _vellipse(self, q=q, opt=opt, centre=centre)

    def fellipse(self, q=None, opt='trans', centre=[0, 0, 0]):
        '''
        Create a force ellipsoid object for plotting

        env = fellipse() creates a force ellipsoid for the robot at
        pose robot.q. The ellipsoid is centered at the origin.

        env = fellipse(q) as above except the robot is plotted with joint
        angles q

        env = fellipse(opt) as above except opt is 'trans' or 'rot' will
        plot either the translational or rotational force ellipsoid.

        env = fellipse(centre) as above except centre is either a 3
        vector or 'ee' which is the centre location of the ellipsoid

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param opt: 'trans' or 'rot' will plot either the translational or
            rotational force ellipsoid
        :type opt: string
        :param centre:
        :type centre: list or str('ee')

        :retrun: An EllipsePlot object
        :rtype: EllipsePlot

        '''

        return _fellipse(self, q=q, opt=opt, centre=centre)

    def plot_vellipse(
            self, q=None, block=True, vellipse=None,
            limits=None, opt='trans', centre=[0, 0, 0],
            jointaxes=True, eeframe=True, shadow=True, name=True):
        '''
        Plot the velocity ellipsoid for seriallink manipulator

        env = plot_vellipse() displays the velocity ellipsoid for the robot at
        pose robot.q. The ellipsoid is centered at the origin. This method
        will be blocking. The plot will autoscale with an aspect ratio of 1.

        env = plot_vellipse(block=False) as avove except the plot in
        non-blocking. Note that the plot will exit when the python script
        finishes executing.

        env = plot_vellipse(q) as above except the robot is plotted with joint
        angles q

        env = plot_vellipse(vellipse) specifies a custon ellipse to plot. If
        not supplied this function calculates the vellipse based on q

        env = plot_vellipse(limits) as above except the view limits of the
        plot are set by limits.

        env = plot_vellipse(opt) as above except opt is 'trans' or 'rot' will
        plot either the translational or rotational velocity ellipsoid.

        env = plot_vellipse(centre) as above except centre is either a 3
        vector or 'ee' which is the centre location of the ellipsoid

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint angles/configuration of the robot (Optional,
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
        :param centre: The coordinates to plot the vellipse [x, y, z] or 'ee'
            to plot at the end-effector location
        :type centre: list or str('ee')
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

        :retrun: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        '''

        if q is not None:
            self.q = q

        if vellipse is None:
            vellipse = self.vellipse(q=q, opt=opt, centre=centre)

        return _plot_ellipse(
            vellipse, block, limits,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

    def plot_fellipse(
            self, q=None, block=True, fellipse=None,
            limits=None, opt='trans', centre=[0, 0, 0],
            jointaxes=True, eeframe=True, shadow=True, name=True):
        '''
        Plot the force ellipsoid for seriallink manipulator

        env = plot_fellipse() displays the force ellipsoid for the robot at
        pose robot.q. The ellipsoid is centered at the origin. This method
        will be blocking. The plot will autoscale with an aspect ratio of 1.

        env = plot_fellipse(block=False) as avove except the plot in
        non-blocking. Note that the plot will exit when the python script
        finishes executing.

        env = plot_fellipse(q) as above except the robot is plotted with joint
        angles q

        env = plot_fellipse(fellipse) specifies a custon ellipse to plot. If
        not supplied this function calculates the fellipse based on q

        env = plot_fellipse(limits) as above except the view limits of the
        plot are set by limits.

        env = plot_fellipse(opt) as above except opt is 'trans' or 'rot' will
        plot either the translational or rotational force ellipsoid.

        env = plot_fellipse(centre) as above except centre is either a 3
        vector or 'ee' which is the centre location of the ellipsoid

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint angles/configuration of the robot (Optional,
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
        :param centre: The coordinates to plot the fellipse [x, y, z] or 'ee'
            to plot at the end-effector location
        :type centre: list or str('ee')
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

        :retrun: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        '''

        if q is not None:
            self.q = q

        if fellipse is None:
            fellipse = self.fellipse(q=q, opt=opt, centre=centre)

        return _plot_ellipse(
            fellipse, block, limits,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

    def plot2(
            self, q=None, block=True, dt=50, limits=None,
            vellipse=False, fellipse=False,
            eeframe=True, name=False):
        '''
        2D Graphical display and animation

        env = plot2() displays a 2D graphical view of a robot based on the
        kinematic model, at it's stored q value. A stick figure polyline
        joins the origins of the link coordinate frames. This method will be
        blocking. The plot will autoscale with an aspect ratio of 1.

        env = plot2(q) as above except the robot is plotted with joint angles q

        env = plot2(block=False) as avove except the plot in non-blocking. Note
        that the plot will exit when the python script finishes executing.

        env = plot2(q, dt) as above except q is an mxn trajectory of joint
        angles. This creates an animation of the robot moving through the
        trajectories with a gap dt milliseconds in between.

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param dt: if q is a trajectory, this describes the delay in
            milliseconds between frames
        :type dt: int
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        :type limits: ndarray(6)
        :param vellipse: (Plot Option) Plot the velocity ellipse at the
            end-effector
        :type vellipse: bool
        :param vellipse: (Plot Option) Plot the force ellipse at the
            end-effector
        :type vellipse: bool
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        :type eeframe: bool
        :param name: (Plot Option) Plot the name of the robot near its base
        :type name: bool

        :retrun: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        '''

        # try:
        return _plot2(
            self, block, q, dt, limits,
            vellipse=vellipse, fellipse=fellipse,
            eeframe=eeframe, name=name)
        # except ModuleNotFoundError:
        #     print(
        #         'Could not find matplotlib.'
        #         ' Matplotlib required for this function')

    def teach2(
            self, q=None, block=True, limits=None,
            eeframe=True, name=False):
        '''
        2D Graphical teach pendant

        env = teach2() creates a 2D matplotlib plot which allows the user to
        "drive" a graphical robot using a graphical slider panel. The
        robot's inital joint configuration is robot.q. This will block the
        programs execution. The plot will autoscale with an aspect ratio of 1.

        env = teach2(q) as above except the robot's initial configuration is
        set to q.

        env = teach2(block=False) as avove except the plot is non-blocking.
        Note that the plot will exit when the python script finishes
        executing.

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        :type limits: ndarray(6)
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        :type eeframe: bool
        :param name: (Plot Option) Plot the name of the robot near its base
        :type name: bool

        :retrun: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        :notes:
            - The slider limits are derived from the joint limit properties.
              If not set then
                - For revolute joints they are assumed to be [-pi, +pi]
                - For prismatic joint they are assumed unknown and an error
                  occurs.

        '''

        if q is not None:
            self.q = q

        # try:
        return _teach2(
            self, block, limits=limits,
            eeframe=eeframe, name=name)
        # except ModuleNotFoundError:
        #     print(
        #         'Could not find matplotlib.'
        #         ' Matplotlib required for this function')

class SerialLink(DHRobot):
    def __init__(self, *args, **kwargs):
        print('SerialLink is deprecated, use DHRobot instead')
        super().__init__(*args, **kwargs)