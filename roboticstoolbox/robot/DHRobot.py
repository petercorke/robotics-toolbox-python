#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np
from roboticstoolbox.robot import Robot  # DHLink
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.DHLink import DHLink  # HACK
from spatialmath.base.argcheck import \
    getvector, isscalar, verifymatrix, getmatrix
from spatialmath.base.transforms3d import tr2jac, trinv
from spatialmath import SE3, Twist3
import spatialmath.base.symbolic as sym
from scipy.optimize import minimize, Bounds
from roboticstoolbox.backends.PyPlot.functions import \
    _plot, _teach, _fellipse, _vellipse, _plot_ellipse, \
    _plot2, _teach2
from roboticstoolbox.robot.DHDynamics import DHDynamicsMixin
from ansitable import ANSITable, Column


class DHRobot(Robot, DHDynamicsMixin):
    """
    Class for robots defined using Denavit-Hartenberg notation

    :param L: List of links which define the robot
    :type L: list(n)
    :param name: Name of the robot
    :type name: str
    :param manufacturer: Manufacturer of the robot
    :type manufacturer: str
    :param base: Locaation of the base
    :type base: SE3
    :param tool: Location of the tool
    :type tool: SE3
    :param gravity: The gravity vector
    :type gravity: ndarray(3)

    A concrete superclass for arm type robots defined using Denavit-Hartenberg
    notation, that represents a serial-link arm-type robot.  Each link and joint
    in the chain is described by a DHLink-class object using Denavit-Hartenberg
    parameters (standard or modified).

    .. note:: Link subclass elements passed in must be all standard, or all
          modified, DH parameters.

    :reference:

        - Robotics, Vision & Control, Chaps 7-9,
          P. Corke, Springer 2011.
        - Robot, Modeling & Control,
          M.Spong, S. Hutchinson & M. Vidyasagar, Wiley 2006.

    """

    def __init__(
            self,
            L,
            **kwargs):

        # Verify L
        if not isinstance(L, list):
            raise TypeError('The links L must be stored in a list.')

        links = []
        self._n = 0

        # TODO rewrite using extend
        for i in range(len(L)):
            if isinstance(L[i], DHLink):
                # got a link
                links.append(L[i])
                self._n += 1
                L[i].id = self._n
                L[i].name = f"link{self._n}"

            elif isinstance(L[i], DHRobot):
                # got a robot
                for j in range(L[i].n):
                    links.append(L[i].links[j])
                    self._n += 1
                    L[i].id = self._n
                    L[i].name = f"link{self._n}"
            else:
                raise TypeError("Input can be only DHLink or DHRobot")
            
        super().__init__(links, **kwargs)

        # Check the DH convention
        self._mdh = self.links[0].mdh
        if not all([link.mdh == self.mdh for link in self.links]):
            raise ValueError('Robot has mixed D&H link conventions')

        # rne parameters
        self._rne_ob = None

    def __str__(self):
        """
        Pretty prints the DH Model of the robot. Will output angles in degrees

        :return: Pretty print of the robot model
        :rtype: str
        """
        s = ""
        deg = 180 / np.pi

        def qstr(j, link):
            j += 1
            if link.flip:
                s = f"-q{j:d}"
            else:
                s = f" q{j:d}"

            if L.offset != 0:
                sign = "+" if L.offset > 0 else "-"
                offset = abs(L.offset)
                if link.isprismatic():
                    s += f" {sign} {offset:}"
                else:
                    s += f" {sign} {offset * deg:.3g}\u00b0"
            return s

        def angle(theta, fmt=None):
            if sym.issymbol(theta):
                return "<<red>>" + str(theta)
            else:
                if fmt is not None:
                    return fmt.format(theta * deg) + "\u00b0"
                else:
                    return str(theta * deg) + "\u00b0"

        has_qlim = any([link._qlim is not None for link in self])
        if has_qlim:
            qlim_columns = [
                Column("q⁻", headalign="^"), Column("q⁺", headalign="^"),
            ]
            qlim = self.qlim

        else:
            qlim_columns = []
        if self.mdh:
            # MDH format
            table = ANSITable(
                Column("aⱼ₋₁", headalign="^"),
                Column("⍺ⱼ₋₁", headalign="^"),
                Column("θⱼ", headalign="^"),
                Column("dⱼ", headalign="^"),
                *qlim_columns, 
                border="thick"
                )
            for j, L in enumerate(self):
                if has_qlim:
                    if L.isprismatic():
                        ql = [qlim[0, j], qlim[1, j]]
                    else:
                        ql = [angle(qlim[k, j], "{:.1f}") for k in [0, 1]]
                else:
                    ql = []
                if L.isprismatic():
                    table.row(L.a, angle(L.alpha), angle(L.theta), qstr(j, L), *ql)
                else:
                    table.row(L.a, angle(L.alpha), qstr(j, L), L.d, *ql)
        else:
            # DH format
            table = ANSITable(
                Column("θⱼ", headalign="^", colalign="<"),
                Column("dⱼ", headalign="^"),
                Column("aⱼ", headalign="^"),
                Column("⍺ⱼ", headalign="^"),
                *qlim_columns,
                border="thick"
                )
            for j, L in enumerate(self):
                if has_qlim:
                    if L.isprismatic():
                        ql = [qlim[0, j], qlim[1, j]]
                    else:
                        ql = [angle(qlim[k, j], "{:.1f}") for k in [0, 1]]
                else:
                    ql = []
                if L.isprismatic():
                    table.row(
                        angle(L.theta), qstr(j, L), L.a, angle(L.alpha), *ql)
                else:
                    table.row(qstr(j, L), L.d, L.a, angle(L.alpha), *ql)

        s = str(table)

        # show tool and base
        if self._tool is not None or self._tool is not None:
            table = ANSITable(
                Column("", colalign=">"),
                Column("", colalign="<"),
                border="thin", header=False)
            if self._base is not None:
                table.row(
                    "base", self._base.printline(
                        orient="rpy/xyz", fmt="{:.2g}", file=None))
            if self._tool is not None:
                table.row(
                    "tool", self._tool.printline(
                        orient="rpy/xyz", fmt="{:.2g}", file=None))
            s += "\n" + str(table)

        # show named configurations
        s += self.configurations_str()

        return s

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
            name=self.name,
            manufacturer=self.manufacturer,
            base=self.base,
            tool=self.tool,
            gravity=self.gravity)

    def copy(self):
        """
        Copy a robot

        :return: A deepish copy of the robot
        :rtype: Robot subclass instance
        """

        L = [link.copy() for link in self]

        new = DHRobot(
            L,
            name=self.name,
            manufacturer=self.manufacturer,
            base=self.base,
            tool=self.tool,
            gravity=self.gravity)

        new.q = self.q
        new.qd = self.qd
        new.qdd = self.qdd

        return new

    @property
    def mdh(self):
        """
        Modified Denavit-Hartenberg status

        :return: whether robot is defined using modified Denavit-Hartenberg
            notation
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.mdh
            >>> panda = rtb.models.DH.Panda()
            >>> panda.mdh

        """
        return self._mdh

    @property
    def d(self):
        r"""
        Link offset values

        :return: List of link offset values :math:`d_j`.
        :rtype: ndarray(n,)

        The following are equivalent::

                robot.links[j].d
                robot.d[j]

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.d
        """
        v = []
        for i in range(self.n):
            v.append(self.links[i].d)
        return v

    @property
    def a(self):
        r"""
        Link length values

        :return: List of link length values :math:`a_j`.
        :rtype: ndarray(n,)

        The following are equivalent::

                robot.links[j].a
                robot.a[j]

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.a
        """
        v = []
        for i in range(self.n):
            v.append(self.links[i].a)
        return v

    @property
    def theta(self):
        r"""
        Joint angle values

        :return: List of joint angle values :math:`\theta_j`.
        :rtype: ndarray(n,)

        The following are equivalent::

                robot.links[j].theta
                robot.theta[j]

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.theta
        """
        v = []
        for i in range(self.n):
            v.append(self.links[i].theta)
        return v

    @property
    def alpha(self):
        r"""
        Link twist values

        :return: List of link twist values :math:`\alpha_j`.
        :rtype: ndarray(n,)

        The following are equivalent::

                robot.links[j].alpha
                robot.alpha[j]

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.alpha
        """
        v = []
        for i in range(self.n):
            v.append(self.links[i].theta)
        return v

    @property
    def r(self):
        r"""
        Link centre of mass values

        :return: Array of link centre of mass values :math:`r_j`.
        :rtype: ndarray(3,n)

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.r
        """
        # TODO tidyup
        v = np.copy(self.links[0].r)
        for i in range(1, self.n):
            v = np.c_[v, self.links[i].r]
        return v

    @property
    def offset(self):
        r"""
        Joint offset values

        :return: List of joint offset values :math:`\bar{q}_j`.
        :rtype: ndarray(n,)

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.offset
        """
        v = []
        for i in range(self.n):
            v.append(self.links[i].offset)
        return v

    @property
    def reach(self):
        r"""
        Reach of the robot

        :return: Maximum reach of the robot
        :rtype: float

        The reach is computed as :math:`\sum_j |a_j| + |d_j|`

        .. note:: Probably an overestimate.
        """
        return np.sum(np.abs([self.a, self.d]))

    def A(self, j, q=None):
        """
        Link forward kinematics

        :param j: Joints to compute link transform for
        :type j: int, 2-tuple
        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values)
        :type q: float ndarray(1,n)
        :return T: The transform between link frames
        :rtype T: SE3

        - ``robot.A(j, q)`` transform between link frames {0} and {n}.  ``q``
          is a vector (n) of joint variables.
        - ``robot.A([j1, j2], q)`` as above between link frames {j1} and {j2}.
        - ``robot.A(j)`` as above except uses the stored q value of the
          robot object.

        .. note:: Base and tool transforms are not applied.

        """

        if isscalar(j):
            j0 = 0
            jn = int(j)
        else:
            j = getvector(j, 2)
            j0 = int(j[0])
            jn = int(j[1])

        jn += 1

        if jn > self.n:
            raise ValueError("The joints value out of range")

        q = getvector(q)

        T = SE3()
        for i in range(j0, jn):
            T *= self.links[i].A(q[i])

        return T

    def islimit(self, q=None):
        """
        Joint limit test

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values)
        :type q: ndarray(n
        :return v: is a vector of boolean values, one per joint, False if
            ``q[j]`` is within the joint limits, else True
        :rtype v: bool list

        - ``robot.islimit(q)`` is a list of boolean values indicating if the
          joint configuration ``q`` is in violation of the joint limits.

        - ``robot.jointlimit()`` as above except uses the stored q value of the
          robot object.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.islimit([0, 0, -4, 4, 0, 0])

        """
        q = self._getq(q)

        return [link.islimit(qk) for (link, qk) in zip(self, q)]

    def isspherical(self):
        """
        Test for spherical wrist

        :return: True if spherical wrist
        :rtype: bool

        Tests if the robot has a spherical wrist, that is, the last 3 axes are
        revolute and their axes intersect at a point.

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.isspherical()

        """
        if self.n < 3:
            return False

        L = self.links[self.n - 3:self.n]

        alpha = [-np.pi / 2, np.pi / 2]

        return L[0].a == 0 \
            and L[1].a == 0 \
            and L[1].d == 0 \
            and (
                (L[0].alpha == alpha[0] and L[1].alpha == alpha[1])
                or
                (L[0].alpha == alpha[1] and L[1].alpha == alpha[0])
            ) \
            and L[0].sigma == 0 \
            and L[1].sigma == 0 \
            and L[2].sigma == 0

    def isprismatic(self):
        """
        Identify prismatic joints

        :return: a list of bool variables, one per joint, true if
            the corresponding joint is prismatic, otherwise false.
        :rtype: list of bool

        ``robot.isprismatic()`` is a bool list identifying the prismatic joints
        within the robot

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.isprismatic()
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.isprismatic()
        """
        return [link.isprismatic() for link in self]

    def isrevolute(self):
        """
        Identify revolute joints

        :return: a list of bool variables, one per joint, true if
            the corresponding joint is revolute, otherwise false.
        :rtype: list of bool

        ``robot.isrevolute()`` is a bool list identifying the revolute joints
        within the robot

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.isprismatic()
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.isrevolute()
        """

        p = []

        for i in range(self.n):
            p.append(self.links[i].isrevolute())

        return p

    def config(self):
        """
        Return the joint configuration string

        :return: joint configuration string
        :rtype: str

        A string with one letter per joint: ``R`` for a revolute
        joint, and ``P`` for a prismatic joint.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.config()
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.config()
        """
        return ''.join(['R' if L.isrevolute() else 'P' for L in self])

    def todegrees(self, q=None):
        """
        Convert joint angles to degrees

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values)
        :type q: ndarray(n)
        :return: a vector of joint coordinates in degrees and metres
        :rtype: ndarray(n)

        ``robot.todegrees(q)`` converts joint coordinates ``q`` to degrees
        taking into account whether elements of ``q`` correspond to revolute
        or prismatic joints, ie. prismatic joint values are not converted.

        ``robot.todegrees()`` as above except uses the stored q value of the
        robot object.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> from math import pi
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.todegrees([pi/4, pi/8, 2, -pi/4, pi/6, pi/3])
        """

        q = self._getq(q)
        revolute = self.isrevolute()

        return np.array([
            q[k] * 180.0 / np.pi if
            revolute[k] else q[k] for k in range(len(q))
        ])

    def toradians(self, q):
        """
        Convert joint angles to radians

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values)
        :type q: ndarray(n)
        :return: a vector of joint coordinates in radians and metres
        :rtype: ndarray(n)

        ``robot.toradians(q)`` converts joint coordinates ``q`` to radians
        taking into account whether elements of ``q`` correspond to revolute
        or prismatic joints, ie. prismatic joint values are not converted.

        ``robot.toradians()`` as above except uses the stored q value of the
        robot object.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.toradians([10, 20, 2, 30, 40, 50])
        """

        q = self._getq(q)
        revolute = self.isrevolute()

        return np.array(
            [q[k] * np.pi / 180.0 if revolute[k] else q[k] for k in range(len(q))]
                    )

    def twists(self, q=None):
        """
        Joint axes as  twists

        :param q: The joint configuration of the robot
        :type q: array_like (n)
        :return: a vector of Twist objects
        :rtype: float ndarray(n,)
        :return: Represents the pose of the tool
        :rtype: SE3 instance

        - ``tw, T0 = twists(q)`` calculates a vector of Twist objects (n) that
          represent the axes of the joints for the robot with joint coordinates
          ``q`` (n). Also returns T0 which is an SE3 object representing the
          pose of the tool.

        - ``tw, T0 = twists()`` as above but the joint coordinates are taken
          to be zero.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> tw, T0 = robot.twists(robot.qz)
            >>> tw
            >>> T0

        """

        if q is None:
            q = np.zeros((self.n))

        T = self.fkine_all(q)
        tw = Twist3.Alloc(self.n)
        if self.mdh:
            # MDH case
            for j, link in enumerate(self):
                if link.sigma == 0:
                    tw[j] = Twist3.R(T[j].a, T[j].t)
                else:
                    tw[j] = Twist3.P(T[j].a)
        else:
            # DH case
            for j, link in enumerate(self):
                if j == 0:
                    # first link case
                    if link.sigma == 0:
                        tw[j] = Twist3.R([0, 0, 1], [0, 0, 0])  # revolute
                    else:
                        tw[j] = Twist3.P([0, 0, 1])  # prismatic
                else:
                    # subsequent links
                    if link.sigma == 0:
                        tw[j] = Twist3.R(T[j-1].a, T[j-1].t)  # revolute
                    else:
                        tw[j] = Twist3.P(T[j-1].a)  # prismatic

        return tw, T[-1]

    def ets(self):

        # optionally start with the base transform
        if self._base is None:
            ets = ETS()
        else:
            ets = ETS.SE3(self._base)

        # add the links
        for link in self:
            ets *= link.ets()

        # optionally add the base transform
        if self._tool is not None:
            ets *= ETS.SE3(self._tool)

        return ets
            
    def fkine(self, q=None):
        """
        Forward kinematics

        :param q: The joint configuration (Optional,
            if not supplied will use the stored q values). 
        :type q: ndarray(n) or ndarray(m,n)
        :return: Forward kinematics as an SE(3) matrix 
        :rtype: SE3 instance

        - ``robot.fkine(q)`` computes the forward kinematics for the robot at 
          joint configuration ``q``.

        - ``robot.fkine()`` as above except uses the stored ``q`` value of the
          robot object.

        If q is a 2D array, the rows are interpreted as the generalized joint
        coordinates for a sequence of points along a trajectory. ``q[k,j]`` is
        the j'th joint coordinate for the k'th trajectory configuration, and
        the returned ``SE3`` instance contains ``n`` values.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.fkine([0, 0, 0, 0, 0, 0])

        .. note::

            - The robot's base or tool transform, if present, are incorporated
              into the result.
            - Joint offsets, if defined, are added to ``q`` before the forward
              kinematics are computed.
        """
        q = self._getq(q)

        T = SE3.Empty()
        for qr in getmatrix(q, (None, self.n)):

            first = True
            for q, L in zip(qr, self.links):
                if first:
                    Tr = L.A(q)
                    first = False
                else:
                    Tr *= L.A(q)

            if self._base is not None:
                Tr = self._base * Tr
            if self._tool is not None:
                Tr = Tr * self._tool
            T.append(Tr)

        return T

    def fkine_all(self, q=None, old=True):
        """
        Forward kinematics for all link frames

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values). 
        :type q: ndarray(n) or ndarray(m,n)
        :param old: "old" behaviour, defaults to True
        :type old: bool, optional
        :return: Forward kinematics as an SE(3) matrix 
        :rtype: SE3 instance with ``n`` values

        - ``fkine_all(q)`` evaluates fkine for each joint within a robot and
          returns a sequence of link frame poses.

        - ``fkine_all()`` as above except uses the stored q value of the
          robot object.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> T = puma.fkine_all([0, 0, 0, 0, 0, 0])
            >>> len(T)

        .. note::
            - Old behaviour is to return a list of ``n`` frames {1} to {n}, but
              if ``old=False`` it returns ``n``+1 frames {0} to {n}, ie. it
              includes the base frame.
            - The robot's base or tool transform, if present, are incorporated
              into the result.
            - Joint offsets, if defined, are added to q before the forward
              kinematics are computed.
        """
        q = self._getq(q)

        if self._base is not None:
            Tj = self._base
        else:
            Tj = SE3()
        first = True
        Tall = Tj
        # print(Tj)
        for q, L in zip(q, self.links):
            if first:
                # print(q, L.A(q))
                Tj *= L.A(q)
                if old:
                    Tall = Tj
                else:
                    Tall.append(Tj)
                first = False
            else:
                Tj *= L.A(q)
                Tall.append(Tj)
        return Tall

    def jacobe(self, q=None):
        r"""
        Manipulator Jacobian in end-effector frame

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: ndarray(n)
        :return J: The manipulator Jacobian in the end-effector frame
        :rtype: ndarray(6,n)

        - ``robot.jacobe(q)`` is the manipulator Jacobian matrix which maps joint
          velocity to end-effector spatial velocity.

        - ``robot.jacobe(q)` as above except uses the stored q value of the
          robot object.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.jacobe([0, 0, 0, 0, 0, 0])
        """

        if q is None:
            q = np.copy(self.q)
        else:
            q = getvector(q, self.n)

        n = self.n
        L = self.links
        J = np.zeros((6, self.n), dtype=q.dtype)

        U = self.tool.A

        for j in range(n - 1, -1, -1):
            if self.mdh == 0:
                # standard DH convention
                U = L[j].A(q[j]).A @ U

            if not L[j].sigma:
                # revolute axis
                d = np.array([
                    -U[0, 0] * U[1, 3] + U[1, 0] * U[0, 3],
                    -U[0, 1] * U[1, 3] + U[1, 1] * U[0, 3],
                    -U[0, 2] * U[1, 3] + U[1, 2] * U[0, 3]
                ])
                delta = U[2, :3]   # nz oz az
            else:
                # prismatic axis
                d = U[2, :3]      # nz oz az
                delta = np.zeros((3,))

            J[:, j] = np.r_[d, delta]

            if self.mdh != 0:
                # modified DH convention
                U = L[j].A(q[j]).A @ U

        return J

    def jacob0(self, q=None, T=None):
        r"""
        Manipulator Jacobian in world frame

        :param q: Joint coordinate
        :type q: ndarray(n)
        :param T: Forward kinematics if known, SE(3 matrix)
        :type T: SE3 instance
        :return J: The manipulator Jacobian in the world frame
        :rtype: ndarray(6,n)

        - ``robot.jacobe(q)`` is the manipulator Jacobian matrix which maps joint
          velocity to end-effector spatial velocity.

        - ``robot.jacobe()`` as above except uses the stored q value of the
        robot object.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{0}\!\nu = \mathbf{J}_0(q) \dot{q}`.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.jacob0([0, 0, 0, 0, 0, 0])
        """
        q = self._getq(q)

        if T is None:
            T = self.fkine(q)

        return tr2jac(trinv(T.A)) @ self.jacobe(q)

    def manipulability(self, q=None, method='yoshikawa', axes='all'):
        """
        Manipulability measure

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: ndarray(n), or ndarray(m,n)
        :param method: method to use, "yoshikawa" (default) or "asada"
        :type method: str
        :param axes: Task space axes to consider: "all" [default],
            "trans", "rot"
        :type axes: str
        :return: manipulability
        :rtype: float or ndarray(m)

        - ``manipulability(q)`` is the Yoshikawa manipulability index (scalar)
          for the robot at the joint configuration ``q``.  It indicates
          dexterity, that is, how isotropic the robot's motion is with respect
          to the 6 degrees of Cartesian motion. The measure is high when the
          manipulator is capable of equal motion in all directions and low when
          the manipulator is close to a singularity. Yoshikawa's manipulability
          measure is based on the shape of the velocity ellipsoid and depends
          only on kinematic parameters.

        - ``manipulability(q, method='asada')`` as above except computes the
          Asada manipulability measure. Asada's manipulability measure is based
          on the shape of the acceleration ellipsoid which in turn is a
          function of the Cartesian inertia matrix and the dynamic parameters.
          The scalar measure computed here is the ratio of the smallest/largest
          ellipsoid axis. Ideally the ellipsoid would be spherical, giving a
          ratio of 1, but in practice will be less than 1.

        - ``maniplty(q, method, axes)`` as above except ``axes`` specify which of
          the 6 degrees-of-freedom to consider in the measurement. For example
          ``axes="trans"`` to consider only translation or
          ``axes="rot"`` to consider only rotation. Defaults to ``"all"``
          motion.

        If ``q`` is a matrix (m,n) then the result (m,) is a vector of
        manipulability indices for each joint configuration specified by a row
        of ``q``.

        .. note::
            - The "all" option includes rotational and translational
              dexterity, but this involves adding different units. It can be
              more useful to look at the translational and rotational
              manipulability separately.
            - Examples in the RVC book (1st edition) can be replicated by
              using the "all" option

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

        """
        if axes == 'all':
            axes = [1, 1, 1, 1, 1, 1]
        elif axes.startswith('trans'):
            axes = [1, 1, 1, 0, 0, 0]
        elif axes.startswith('rot'):
            axes = [0, 0, 0, 1, 1, 1]
        else:
            raise ValueError('axes must be all, trans or rot')

        def yoshi(robot, q, axes):
            J = robot.jacob0(q)
            J = J[axes, :]
            m2 = np.linalg.det(J @ J.T)
            # m2 = np.maximum(0.0, m2)  # clip it to positive
            return np.sqrt(abs(m2))

        def asada(robot, q, axes, dof):
            J = robot.jacob0(q)

            if np.linalg.matrix_rank(J) < 6:
                return 0

            Ji = np.linalg.pinv(J)
            M = robot.inertia(q)
            Mx = Ji.T @ M @ Ji
            d = np.where(axes)[0]
            Mx = Mx[d]
            Mx = Mx[:, d.tolist()]
            e, _ = np.linalg.eig(Mx)

            return np.min(e) / np.max(e)

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

            for i in range(trajn):
                w[i] = asada(self, q[i, :], axes, dof)

            if trajn == 1:
                return w[0]
            else:
                return w

        else:
            raise ValueError(
                'Invalid method chosen. Must be \'yoshikawa\' or \'asada\'.')

    def jacob_dot(self, q=None, qd=None):
        """
        Derivative of Jacobian

        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param qd: The joint configuration of the robot (Optional,
            if not supplied will use the stored qd values).
        :type qd: ndarray(n)
        :return: The derivative of the manipulator Jacobian
        :rtype:  ndarray(n)

        ``robot.jacob_dot(q, qd)`` computes the product (6) of the temporal
        derivative of the manipulator Jacobian (in the world frame) and the
        joint rates :math:`\frac{d}{dt} \dot{\mathbf{J}}(q) \dot{q}`.

        .. note::
            - This term appears in the formulation for operational space
              control :math:`\ddot{x} = \mathbf{J}(q) \ddot{q} + \dot{\mathbf{J}}(q) \dot{q}`.
            - Written as per the reference and not very efficient.

        :references:
            - Fundamentals of Robotics Mechanical Systems (2nd ed)
              J. Angleles, Springer 2003.
            - A unified approach for motion and force control of robot
              manipulators: The operational space formulation
              O Khatib, IEEE Journal on Robotics and Automation, 1987.

        """

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

    def plot(
            self, q=None, block=True, dt=0.050, limits=None,
            vellipse=False, fellipse=False,
            jointaxes=True, eeframe=True, shadow=True, name=True, movie=None):
        """
        Graphical display and animation

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param dt: if q is a trajectory, this describes the delay in
            seconds between frames
        :type dt: float
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

        :return: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        - ``robot.plot(q)`` displays a graphical view of a robot based on the
          kinematic model and the joint configuration ``q``. This is a stick
          figure polyline which joins the origins of the link coordinate frames. 
          The plot will autoscale with an aspect ratio of 1.

        - ``robot.plot()`` as above but use the stored ``q`` value.

        If ``q`` (m,n) representing a joint-space trajectory it will create an
        animation with a pause of ``dt`` seconds between each frame.

        .. note::
            - By default this method will block until the figure is dismissed.
              To avoid this set ``block=False``.
            - The polyline joins the origins of the link frames, but for
              some Denavit-Hartenberg models those frames may not actually
              be on the robot, ie. the lines to not neccessarily represent
              the links of the robot.

        :seealso: :func:`teach`
        """

        # try:
        return _plot(
            self, block, q, int(dt * 1000), limits,
            vellipse=vellipse, fellipse=fellipse,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name,
            movie=movie)
        # except ModuleNotFoundError:
        #     print(
        #         'Could not find matplotlib.'
        #         ' Matplotlib required for this function')

    def teach(
            self, q=None, block=True, limits=None,
            jointaxes=True, eeframe=True, shadow=True, name=True):
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

        - ``robot.teach()`` as above except the robot's stored value of ``q`` is used.

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

    def vellipse(self, q=None, opt='trans', centre=[0, 0, 0]):
        """
        Create a velocity ellipsoid object for plotting

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

        return _vellipse(self, q=q, opt=opt, centre=centre)

    def fellipse(self, q=None, opt='trans', centre=[0, 0, 0]):
        '''
        Create a force ellipsoid object for plotting

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

        '''

        return _fellipse(self, q=q, opt=opt, centre=centre)

    def plot_vellipse(
            self, q=None, block=True, vellipse=None,
            limits=None, opt='trans', centre=[0, 0, 0],
            jointaxes=True, eeframe=True, shadow=True, name=True):
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
          robot at pose ``q``. The plot will autoscale with an aspect ratio of 1.

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
          robot at pose ``q``. The plot will autoscale with an aspect ratio of 1.

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
        """
        2D Graphical display and animation

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint configuration of the robot (Optional,
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

        :return: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        - ``robot.plot2(q)`` displays a 2D graphical view of a robot based on the
          kinematic model and the joint configuration ``q``. This is a stick
          figure polyline which joins the origins of the link coordinate frames. 
          The plot will autoscale with an aspect ratio of 1.

        - ``robot.plot2()`` as above but use the stored ``q`` value.

        If ``q`` (m,n) representing a joint-space trajectory it will create an
        animation with a pause of ``dt`` seconds between each frame.

        .. note::
            - By default this method will block until the figure is dismissed.
              To avoid this set ``block=False``.
            - The polyline joins the origins of the link frames, but for
              some Denavit-Hartenberg models those frames may not actually
              be on the robot, ie. the lines to not neccessarily represent
              the links of the robot.

        :seealso: :func:`teach2`

        """

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

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint configuration of the robot (Optional,
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

        :return: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        - ``robot.teach2(q)`` creates a 2D matplotlib plot which allows the user
          to "drive" a graphical robot using a graphical slider panel. The
          robot's inital joint configuration is ``q``. The plot will autoscale
          with an aspect ratio of 1.

        - ``robot.teach2()`` as above except the robot's stored value of ``q``
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

    def ikine_6s(self, T, config, _ikfunc):
        # Undo base and tool transformations, but if they are not
        # set, skip the operation.  Nicer for symbolics
        if self._base is not None:
            T = self.base.inv() * T
        if self._tool is not None:
            T = self.tool.inv() * T

        q = np.zeros((len(T), 6))
        for k, Tk in enumerate(T):
            theta = _ikfunc(Tk, config)

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

                R = Td4.inv() * T13.inv() * Tk * Tt.inv()

                # The spherical wrist implements Euler angles
                if 'f' in config:
                    eul = R.eul(flip=True)
                else:
                    eul = R.eul()
                theta = np.r_[theta, eul]
                if self.links[3].alpha > 0:
                    theta[4] = -theta[4]

                # Remove the link offset angles
                theta = theta - self.offset

                q[k, :] = theta
            
            if q.shape[0] == 1:
                return q[0]
            else:
                return q


    # def ikine_6s(self, T, left=True, elbow_up=True, wrist_flip=False):
    #     """
    #     Analytical inverse kinematics

    #     q, err = ikine6s(T) are the joint coordinates (n) corresponding to the
    #     robot end-effector pose T which is an SE3 object or homogenenous
    #     transform matrix (4x4), and n is the number of robot joints. This
    #     is an analytic solution for a 6-axis robot with a spherical wrist
    #     (the most common form for industrial robot arms).

    #     q, err = ikine6s(T, left, elbow_up, wrist_flip) as above except the
    #     arm location, elbow position, and wrist orientation can be specified.

    #     Trajectory operation:
    #     In all cases if T is a vector of SE3 objects (1xM) or a homogeneous
    #     transform sequence (4x4xM) then the inverse kinematics is computed for
    #     all m poses resulting in q (mxn) with each row representing the joint
    #     angles at the corresponding pose.

    #     :param T: The desired end-effector pose
    #     :type T: SE3 or SE3 trajectory
    #     :param left: True for arm to the left (default), else arm to the right
    #     :type left: bool
    #     :param elbow_up: True for elbow up (default), else elbow down
    #     :type elbow_up: bool
    #     :param wrist_flip: False for wrist not flipped (default), else wrist
    #         flipped (rotated by 180 deg)
    #     :type wrist_flip: bool

    #     :return q: The calculated joint values
    #     :rtype q: float ndarray(n)
    #     :return err: Any errors encountered
    #     :rtype err: list String

    #     :notes:
    #         - Treats a number of specific cases:
    #             - Robot with no shoulder offset
    #             - Robot with a shoulder offset (has lefty/righty configuration)
    #             - Robot with a shoulder offset and a prismatic third joint
    #               (like Stanford arm)
    #         - The Puma 560 arms with shoulder and elbow offsets (4 lengths
    #           parameters)
    #         - The Kuka KR5 with many offsets (7 length parameters)
    #         - The inverse kinematics for the various cases determined using
    #           ikine_sym.
    #         - The inverse kinematic solution is generally not unique, and
    #           depends on the configuration string.
    #         - Joint offsets, if defined, are added to the inverse kinematics
    #           to generate q.
    #         - Only applicable for standard Denavit-Hartenberg parameters

    #     :reference:
    #         - Inverse kinematics for a PUMA 560,
    #           Paul and Zhang,
    #           The International Journal of Robotics Research,
    #           Vol. 5, No. 2, Summer 1986, p. 32-44

    #     """

    #     if not self.n == 6:
    #         raise ValueError(
    #             "Function only applicable to six degree of freedom robots")

    #     if self.mdh:
    #         raise ValueError(
    #             "Function only applicable to robots with standard DH "
    #             "parameters")

    #     if not self.isspherical():
    #         raise ValueError(
    #             "Function only applicable to robots with a spherical wrist")

    #     if not isinstance(T, SE3):
    #         T = SE3(T)

    #     trajn = len(T)

    #     sol = [1, 1, 1]

    #     if not left:
    #         sol[0] = 2

    #     if not elbow_up:
    #         sol[1] = 2

    #     if wrist_flip:
    #         sol[2] = 2

    #     if self._is_simple():
    #         self.ikineType = 'nooffset'
    #     elif self._is_puma():
    #         self.ikineType = 'puma'
    #     elif self._is_offset():
    #         self.ikineType = 'offset'
    #     elif self._is_rrp():
    #         self.ikineType = 'rrp'
    #     else:
    #         raise ValueError('This kinematic structure not supported')

    #     q = np.zeros((trajn, self.n))
    #     err = []

    #     for j in range(trajn):

    #         theta = np.zeros(self.n)

    #         # Undo base and tool transformations
    #         Ti = np.linalg.inv(self.base.A) @ T[j].A @ \
    #             np.linalg.inv(self.tool.A)

    #         if self.ikineType == 'puma':
    #             # Puma model with shoulder and elbow offsets
    #             # - Inverse kinematics for a PUMA 560,
    #             #   Paul and Zhang,
    #             #   The International Journal of Robotics Research,
    #             #   Vol. 5, No. 2, Summer 1986, p. 32-44

    #             a2 = self.links[1].a
    #             a3 = self.links[2].a
    #             d1 = self.links[0].d
    #             d3 = self.links[2].d
    #             d4 = self.links[3].d

    #             # The following parameters are extracted from the Homogeneous
    #             # Transformation as defined in equation 1, p. 34

    #             Px = Ti[0, 3]
    #             Py = Ti[1, 3]
    #             Pz = Ti[2, 3] - d1

    #             # Solve for theta[0]
    #             # r is defined in equation 38, p. 39.
    #             # theta[0] uses equations 40 and 41, p.39,
    #             # based on the configuration parameter n1

    #             r = np.sqrt(Px**2 + Py**2)
    #             if sol[0] == 1:
    #                 theta[0] = np.arctan2(Py, Px) + np.pi - np.arcsin(d3 / r)
    #             else:
    #                 theta[0] = np.arctan2(Py, Px) + np.arcsin(d3 / r)

    #             # Solve for theta[1]
    #             # V114 is defined in equation 43, p.39.
    #             # r is defined in equation 47, p.39.
    #             # Psi is defined in equation 49, p.40.
    #             # theta[1] uses equations 50 and 51, p.40, based on the
    #             # configuration parameter n2
    #             if sol[1] == 1:
    #                 n2 = -1
    #             else:
    #                 n2 = 1

    #             if sol[0] == 2:
    #                 n2 = -n2

    #             V114 = Px * np.cos(theta[0]) + Py * np.sin(theta[0])

    #             r = np.sqrt(V114**2 + Pz**2)

    #             Psi = np.arccos(
    #                 (a2**2 - d4**2 - a3**2 + V114**2 + Pz**2)
    #                 / (2.0 * a2 * r))

    #             if np.isnan(Psi):
    #                 theta = []
    #             else:
    #                 theta[1] = np.arctan2(Pz, V114) + n2 * Psi

    #                 # Solve for theta[2]
    #                 # theta[2] uses equation 57, p. 40.
    #                 num = np.cos(theta[1]) * V114 + np.sin(theta[1]) * Pz - a2
    #                 den = np.cos(theta[1]) * Pz - np.sin(theta[1]) * V114
    #                 theta[2] = np.arctan2(a3, d4) - np.arctan2(num, den)

    #         elif self.ikineType == 'nooffset':
    #             a2 = self.links[1].a
    #             a3 = self.links[2].a
    #             d1 = self.links[0].d

    #             px = Ti[0, 3]
    #             py = Ti[1, 3]
    #             pz = Ti[2, 3]

    #             # Autogenerated code
    #             if self.links[0].alpha < 0:
    #                 if sol[0] == 1:
    #                     temp = -px - py * 1j
    #                     if np.abs(temp) == 0:
    #                         temp = 0
    #                     theta[0] = np.angle(temp)
    #                 else:
    #                     theta[0] = np.angle(px + py * 1j)

    #                 print(theta[0])

    #                 S1 = np.sin(theta[0])
    #                 C1 = np.cos(theta[0])

    #                 if sol[1] == 1:
    #                     theta[1] = -np.angle(a2*d1*-2.0+a2*pz*2.0-C1*a2*px*2.0j-S1*a2*py*2.0j)+np.angle(d1*pz*2.0j-a2**2*1j+a3**2*1j-d1**2*1j-pz**2*1j-C1**2*px**2*1j-np.sqrt(0j+(a2*d1*2.0-a2*pz*2.0)**2+(C1*a2*px*2.0+S1*a2*py*2.0)**2-(d1*pz*-2.0+a2**2-a3**2+d1**2+pz**2+C1**2*px**2+S1**2*py**2+C1*S1*px*py*2.0)**2)-S1**2*py**2*1j-C1*S1*px*py*2.0j)  # noqa
    #                 else:
    #                     theta[1] = -np.angle(a2*d1*-2.0+a2*pz*2.0-C1*a2*px*2.0j-S1*a2*py*2.0j)+np.angle(d1*pz*2.0j-a2**2*1j+a3**2*1j-d1**2*1j-pz**2*1j-C1**2*px**2*1j+np.sqrt(0j+(a2*d1*2.0-a2*pz*2.0)**2+(C1*a2*px*2.0+S1*a2*py*2.0)**2-(d1*pz*-2.0+a2**2-a3**2+d1**2+pz**2+C1**2*px**2+S1**2*py**2+C1*S1*px*py*2.0)**2)-S1**2*py**2*1j-C1*S1*px*py*2.0j)  # noqa

    #                 S2 = np.sin(theta[1])
    #                 C2 = np.cos(theta[1])

    #                 if sol[2] == 1:
    #                     theta[2] = -np.angle(a2*-1j+C2*d1-C2*pz+S2*d1*1j-S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j-np.sqrt(0j+(-a2+S2*d1-S2*pz+C1*C2*px+C2*S1*py)**2+(-C2*d1+C2*pz+C1*S2*px+S1*S2*py)**2-a3**2))  # noqa
    #                 else:
    #                     theta[2] = -np.angle(a2*-1j+C2*d1-C2*pz+S2*d1*1j-S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j+np.sqrt(0j+(-a2+S2*d1-S2*pz+C1*C2*px+C2*S1*py)**2+(-C2*d1+C2*pz+C1*S2*px+S1*S2*py)**2-a3**2))  # noqa

    #             else:
    #                 if sol[0] == 1:
    #                     theta[0] = np.angle(px + py * 1j)
    #                 else:
    #                     temp = -px - py * 1j
    #                     if np.abs(temp) == 0:
    #                         temp = 0
    #                     theta[0] = np.angle(temp)

    #                 S1 = np.sin(theta[0])
    #                 C1 = np.cos(theta[0])

    #                 if sol[1] == 1:
    #                     theta[1] = -np.angle(a2*d1*2.0-a2*pz*2.0-C1*a2*px*2.0j-S1*a2*py*2.0j)+np.angle(d1*pz*2.0j-a2**2*1j+a3**2*1j-d1**2*1j-pz**2*1j-C1**2*px**2*1j-np.sqrt(0j+(a2*d1*2.0-a2*pz*2.0)**2+(C1*a2*px*2.0+S1*a2*py*2.0)**2-(d1*pz*-2.0+a2**2-a3**2+d1**2+pz**2+C1**2*px**2+S1**2*py**2+C1*S1*px*py*2.0)**2)-S1**2*py**2*1j-C1*S1*px*py*2.0j)  # noqa
    #                 else:
    #                     theta[1] = -np.angle(a2*d1*2.0-a2*pz*2.0-C1*a2*px*2.0j-S1*a2*py*2.0j)+np.angle(d1*pz*2.0j-a2**2*1j+a3**2*1j-d1**2*1j-pz**2*1j-C1**2*px**2*1j+np.sqrt(0j+(a2*d1*2.0-a2*pz*2.0)**2+(C1*a2*px*2.0+S1*a2*py*2.0)**2-(d1*pz*-2.0+a2**2-a3**2+d1**2+pz**2+C1**2*px**2+S1**2*py**2+C1*S1*px*py*2.0)**2)-S1**2*py**2*1j-C1*S1*px*py*2.0j)  # noqa

    #                 S2 = np.sin(theta[1])
    #                 C2 = np.cos(theta[1])

    #                 if sol[2] == 1:
    #                     theta[2] = -np.angle(a2*-1j-C2*d1+C2*pz-S2*d1*1j+S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j-np.sqrt(0j+(-a2-S2*d1+S2*pz+C1*C2*px+C2*S1*py)**2+(C2*d1-C2*pz+C1*S2*px+S1*S2*py)**2-a3**2))  # noqa
    #                 else:
    #                     theta[2] = -np.angle(a2*-1j-C2*d1+C2*pz-S2*d1*1j+S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j+np.sqrt(0j+(-a2-S2*d1+S2*pz+C1*C2*px+C2*S1*py)**2+(C2*d1-C2*pz+C1*S2*px+S1*S2*py)**2-a3**2))  # noqa

    #         elif self.ikineType == 'offset':
    #             # General case with 6 length parameters
    #             a1 = self.links[0].a
    #             a2 = self.links[1].a
    #             a3 = self.links[2].a
    #             d1 = self.links[0].d
    #             d2 = self.links[1].d
    #             d3 = self.links[2].d

    #             px = Ti[0, 3]
    #             py = Ti[1, 3]
    #             pz = Ti[2, 3]

    #             # Autogenerated code
    #             if self.links[0].alpha < 0:

    #                 if sol[0] == 1:
    #                     theta[0] = -np.angle(-px+py*1j)+np.angle(d2*1j+d3*1j-np.sqrt(0j+d2*d3*-2.0-d2**2-d3**2+px**2+py**2))  # noqa
    #                 else:
    #                     theta[0] = np.angle(d2*1j+d3*1j+np.sqrt(0j+d2*d3*-2.0-d2**2-d3**2+px**2+py**2))-np.angle(-px+py*1j)  # noqa

    #                 S1 = np.sin(theta[0])
    #                 C1 = np.cos(theta[0])

    #                 if sol[1] == 1:
    #                     theta[1] = np.angle(d1*pz*2.0j-np.sqrt(0j+d1*pz**3*4.0+d1**3*pz*4.0-a1**4-a2**4-a3**4-d1**4-py**4-pz**4-C1**4*px**4+C1**2*py**4*2.0-C1**4*py**4+a1**2*a2**2*2.0+a1**2*a3**2*2.0+a2**2*a3**2*2.0-a1**2*d1**2*2.0+a2**2*d1**2*2.0+a3**2*d1**2*2.0-a1**2*py**2*6.0-a2**2*py**2*2.0+a3**2*py**2*2.0-a1**2*pz**2*2.0+a2**2*pz**2*2.0+a3**2*pz**2*2.0-d1**2*py**2*2.0-d1**2*pz**2*6.0-py**2*pz**2*2.0+d1*py**2*pz*4.0+C1**3*a1*px**3*4.0+S1**3*a1*py**3*4.0-C1**2*a1**2*px**2*6.0+C1**2*a2**2*px**2*2.0+C1**2*a3**2*px**2*2.0+C1**2*a1**2*py**2*6.0+C1**2*a2**2*py**2*1.0e1-C1**2*a3**2*py**2*2.0-C1**4*a2**2*py**2*1.2e1+C1**6*a2**2*py**2*4.0-C1**2*d1**2*px**2*2.0+C1**2*d1**2*py**2*2.0-C1**2*px**2*py**2*6.0+C1**4*px**2*py**2*6.0-C1**2*px**2*pz**2*2.0+C1**2*py**2*pz**2*2.0+S1**6*a2**2*py**2*4.0+C1*a1**3*px*4.0+S1*a1**3*py*4.0+a1**2*d1*pz*4.0-a2**2*d1*pz*4.0-a3**2*d1*pz*4.0-C1*a1*a2**2*px*4.0-C1*a1*a3**2*px*4.0-C1*S1*px**3*py*4.0+C1*a1*d1**2*px*4.0+C1*a1*px*py**2*1.2e1+C1*a1*px*pz**2*4.0+S1*a1*a2**2*py*4.0-S1*a1*a3**2*py*4.0+S1*a1*d1**2*py*4.0+S1*a1*px**2*py*4.0+S1*a1*py*pz**2*4.0-C1*S1**3*px*py**3*4.0+C1*S1**3*px**3*py*4.0-C1**3*a1*px*py**2*1.2e1-S1**3*a1*a2**2*py*8.0+C1**2*d1*px**2*pz*4.0-C1**2*d1*py**2*pz*4.0-S1**3*a1*px**2*py*4.0-C1*a1*d1*px*pz*8.0-S1*a1*d1*py*pz*8.0-C1*S1*a1**2*px*py*1.2e1-C1*S1*a2**2*px*py*4.0+C1*S1*a3**2*px*py*4.0-C1*S1*d1**2*px*py*4.0-C1*S1*px*py*pz**2*4.0-C1**2*S1*a1*a2**2*py*8.0+C1**2*S1*a1*px**2*py*8.0+C1*S1**3*a2**2*px*py*8.0+C1**3*S1*a2**2*px*py*8.0+C1*S1*d1*px*py*pz*8.0)-a1**2*1j-a2**2*1j+a3**2*1j-d1**2*1j-py**2*1j-pz**2*1j-C1**2*px**2*1j+C1**2*py**2*1j+C1*a1*px*2.0j+S1*a1*py*2.0j-C1*S1*px*py*2.0j)-np.angle(-a2*(a1*-1j+d1-pz+C1*px*1j+S1**3*py*1j+C1**2*S1*py*1j))  # noqa
    #                 else:
    #                     theta[1] = np.angle(d1*pz*2.0j+np.sqrt(0j+d1*pz**3*4.0+d1**3*pz*4.0-a1**4-a2**4-a3**4-d1**4-py**4-pz**4-C1**4*px**4+C1**2*py**4*2.0-C1**4*py**4+a1**2*a2**2*2.0+a1**2*a3**2*2.0+a2**2*a3**2*2.0-a1**2*d1**2*2.0+a2**2*d1**2*2.0+a3**2*d1**2*2.0-a1**2*py**2*6.0-a2**2*py**2*2.0+a3**2*py**2*2.0-a1**2*pz**2*2.0+a2**2*pz**2*2.0+a3**2*pz**2*2.0-d1**2*py**2*2.0-d1**2*pz**2*6.0-py**2*pz**2*2.0+d1*py**2*pz*4.0+C1**3*a1*px**3*4.0+S1**3*a1*py**3*4.0-C1**2*a1**2*px**2*6.0+C1**2*a2**2*px**2*2.0+C1**2*a3**2*px**2*2.0+C1**2*a1**2*py**2*6.0+C1**2*a2**2*py**2*1.0e1-C1**2*a3**2*py**2*2.0-C1**4*a2**2*py**2*1.2e1+C1**6*a2**2*py**2*4.0-C1**2*d1**2*px**2*2.0+C1**2*d1**2*py**2*2.0-C1**2*px**2*py**2*6.0+C1**4*px**2*py**2*6.0-C1**2*px**2*pz**2*2.0+C1**2*py**2*pz**2*2.0+S1**6*a2**2*py**2*4.0+C1*a1**3*px*4.0+S1*a1**3*py*4.0+a1**2*d1*pz*4.0-a2**2*d1*pz*4.0-a3**2*d1*pz*4.0-C1*a1*a2**2*px*4.0-C1*a1*a3**2*px*4.0-C1*S1*px**3*py*4.0+C1*a1*d1**2*px*4.0+C1*a1*px*py**2*1.2e1+C1*a1*px*pz**2*4.0+S1*a1*a2**2*py*4.0-S1*a1*a3**2*py*4.0+S1*a1*d1**2*py*4.0+S1*a1*px**2*py*4.0+S1*a1*py*pz**2*4.0-C1*S1**3*px*py**3*4.0+C1*S1**3*px**3*py*4.0-C1**3*a1*px*py**2*1.2e1-S1**3*a1*a2**2*py*8.0+C1**2*d1*px**2*pz*4.0-C1**2*d1*py**2*pz*4.0-S1**3*a1*px**2*py*4.0-C1*a1*d1*px*pz*8.0-S1*a1*d1*py*pz*8.0-C1*S1*a1**2*px*py*1.2e1-C1*S1*a2**2*px*py*4.0+C1*S1*a3**2*px*py*4.0-C1*S1*d1**2*px*py*4.0-C1*S1*px*py*pz**2*4.0-C1**2*S1*a1*a2**2*py*8.0+C1**2*S1*a1*px**2*py*8.0+C1*S1**3*a2**2*px*py*8.0+C1**3*S1*a2**2*px*py*8.0+C1*S1*d1*px*py*pz*8.0)-a1**2*1j-a2**2*1j+a3**2*1j-d1**2*1j-py**2*1j-pz**2*1j-C1**2*px**2*1j+C1**2*py**2*1j+C1*a1*px*2.0j+S1*a1*py*2.0j-C1*S1*px*py*2.0j)-np.angle(-a2*(a1*-1j+d1-pz+C1*px*1j+S1**3*py*1j+C1**2*S1*py*1j))  # noqa

    #                 S2 = np.sin(theta[1])
    #                 C2 = np.cos(theta[1])

    #                 if sol[2] == 1:
    #                     theta[2] = np.angle(a3*1j-np.sqrt(0j+d1*pz*-2.0+a1**2+a2**2-a3**2+d1**2+py**2+pz**2+C1**2*px**2-C1**2*py**2+C2*a1*a2*2.0-C1*a1*px*2.0-S2*a2*d1*2.0-S1*a1*py*2.0+S2*a2*pz*2.0-C1*C2*a2*px*2.0-C2*S1*a2*py*2.0+C1*S1*px*py*2.0+0j))-np.angle(a2*-1j-C2*a1*1j+C2*d1-C2*pz+S2*a1+S2*d1*1j-S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)  # noqa
    #                 else:
    #                     theta[2] = -np.angle(a2*-1j-C2*a1*1j+C2*d1-C2*pz+S2*a1+S2*d1*1j-S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j+np.sqrt(0j+d1*pz*-2.0+a1**2+a2**2-a3**2+d1**2+py**2+pz**2+C1**2*px**2-C1**2*py**2+C2*a1*a2*2.0-C1*a1*px*2.0-S2*a2*d1*2.0-S1*a1*py*2.0+S2*a2*pz*2.0-C1*C2*a2*px*2.0-C2*S1*a2*py*2.0+C1*S1*px*py*2.0))  # noqa

    #             else:
    #                 if sol[0] == 1:
    #                     theta[0] = -np.angle(px-py*1j)+np.angle(d2*1j+d3*1j-np.sqrt(0j+d2*d3*-2.0-d2**2-d3**2+px**2+py**2))  # noqa
    #                 else:
    #                     theta[0] = -np.angle(px-py*1j)+np.angle(d2*1j+d3*1j+np.sqrt(0j+d2*d3*-2.0-d2**2-d3**2+px**2+py**2))  # noqa

    #                 S1 = np.sin(theta[0])
    #                 C1 = np.cos(theta[0])

    #                 if sol[1] == 1:
    #                     theta[1] = np.angle(d1*pz*2.0j-np.sqrt(0j+d1*pz**3*4.0+d1**3*pz*4.0-a1**4-a2**4-a3**4-d1**4-py**4-pz**4-C1**4*px**4+C1**2*py**4*2.0-C1**4*py**4+a1**2*a2**2*2.0+a1**2*a3**2*2.0+a2**2*a3**2*2.0-a1**2*d1**2*2.0+a2**2*d1**2*2.0+a3**2*d1**2*2.0-a1**2*py**2*6.0-a2**2*py**2*2.0+a3**2*py**2*2.0-a1**2*pz**2*2.0+a2**2*pz**2*2.0+a3**2*pz**2*2.0-d1**2*py**2*2.0-d1**2*pz**2*6.0-py**2*pz**2*2.0+d1*py**2*pz*4.0+C1**3*a1*px**3*4.0+S1**3*a1*py**3*4.0-C1**2*a1**2*px**2*6.0+C1**2*a2**2*px**2*2.0+C1**2*a3**2*px**2*2.0+C1**2*a1**2*py**2*6.0+C1**2*a2**2*py**2*1.0e1-C1**2*a3**2*py**2*2.0-C1**4*a2**2*py**2*1.2e1+C1**6*a2**2*py**2*4.0-C1**2*d1**2*px**2*2.0+C1**2*d1**2*py**2*2.0-C1**2*px**2*py**2*6.0+C1**4*px**2*py**2*6.0-C1**2*px**2*pz**2*2.0+C1**2*py**2*pz**2*2.0+S1**6*a2**2*py**2*4.0+C1*a1**3*px*4.0+S1*a1**3*py*4.0+a1**2*d1*pz*4.0-a2**2*d1*pz*4.0-a3**2*d1*pz*4.0-C1*a1*a2**2*px*4.0-C1*a1*a3**2*px*4.0-C1*S1*px**3*py*4.0+C1*a1*d1**2*px*4.0+C1*a1*px*py**2*1.2e1+C1*a1*px*pz**2*4.0+S1*a1*a2**2*py*4.0-S1*a1*a3**2*py*4.0+S1*a1*d1**2*py*4.0+S1*a1*px**2*py*4.0+S1*a1*py*pz**2*4.0-C1*S1**3*px*py**3*4.0+C1*S1**3*px**3*py*4.0-C1**3*a1*px*py**2*1.2e1-S1**3*a1*a2**2*py*8.0+C1**2*d1*px**2*pz*4.0-C1**2*d1*py**2*pz*4.0-S1**3*a1*px**2*py*4.0-C1*a1*d1*px*pz*8.0-S1*a1*d1*py*pz*8.0-C1*S1*a1**2*px*py*1.2e1-C1*S1*a2**2*px*py*4.0+C1*S1*a3**2*px*py*4.0-C1*S1*d1**2*px*py*4.0-C1*S1*px*py*pz**2*4.0-C1**2*S1*a1*a2**2*py*8.0+C1**2*S1*a1*px**2*py*8.0+C1*S1**3*a2**2*px*py*8.0+C1**3*S1*a2**2*px*py*8.0+C1*S1*d1*px*py*pz*8.0)-a1**2*1j-a2**2*1j+a3**2*1j-d1**2*1j-py**2*1j-pz**2*1j-C1**2*px**2*1j+C1**2*py**2*1j+C1*a1*px*2.0j+S1*a1*py*2.0j-C1*S1*px*py*2.0j)-np.angle(-a2*(a1*-1j-d1+pz+C1*px*1j+S1**3*py*1j+C1**2*S1*py*1j))  # noqa
    #                 else:
    #                     theta[1] = np.angle(d1*pz*2.0j+np.sqrt(0j+d1*pz**3*4.0+d1**3*pz*4.0-a1**4-a2**4-a3**4-d1**4-py**4-pz**4-C1**4*px**4+C1**2*py**4*2.0-C1**4*py**4+a1**2*a2**2*2.0+a1**2*a3**2*2.0+a2**2*a3**2*2.0-a1**2*d1**2*2.0+a2**2*d1**2*2.0+a3**2*d1**2*2.0-a1**2*py**2*6.0-a2**2*py**2*2.0+a3**2*py**2*2.0-a1**2*pz**2*2.0+a2**2*pz**2*2.0+a3**2*pz**2*2.0-d1**2*py**2*2.0-d1**2*pz**2*6.0-py**2*pz**2*2.0+d1*py**2*pz*4.0+C1**3*a1*px**3*4.0+S1**3*a1*py**3*4.0-C1**2*a1**2*px**2*6.0+C1**2*a2**2*px**2*2.0+C1**2*a3**2*px**2*2.0+C1**2*a1**2*py**2*6.0+C1**2*a2**2*py**2*1.0e1-C1**2*a3**2*py**2*2.0-C1**4*a2**2*py**2*1.2e1+C1**6*a2**2*py**2*4.0-C1**2*d1**2*px**2*2.0+C1**2*d1**2*py**2*2.0-C1**2*px**2*py**2*6.0+C1**4*px**2*py**2*6.0-C1**2*px**2*pz**2*2.0+C1**2*py**2*pz**2*2.0+S1**6*a2**2*py**2*4.0+C1*a1**3*px*4.0+S1*a1**3*py*4.0+a1**2*d1*pz*4.0-a2**2*d1*pz*4.0-a3**2*d1*pz*4.0-C1*a1*a2**2*px*4.0-C1*a1*a3**2*px*4.0-C1*S1*px**3*py*4.0+C1*a1*d1**2*px*4.0+C1*a1*px*py**2*1.2e1+C1*a1*px*pz**2*4.0+S1*a1*a2**2*py*4.0-S1*a1*a3**2*py*4.0+S1*a1*d1**2*py*4.0+S1*a1*px**2*py*4.0+S1*a1*py*pz**2*4.0-C1*S1**3*px*py**3*4.0+C1*S1**3*px**3*py*4.0-C1**3*a1*px*py**2*1.2e1-S1**3*a1*a2**2*py*8.0+C1**2*d1*px**2*pz*4.0-C1**2*d1*py**2*pz*4.0-S1**3*a1*px**2*py*4.0-C1*a1*d1*px*pz*8.0-S1*a1*d1*py*pz*8.0-C1*S1*a1**2*px*py*1.2e1-C1*S1*a2**2*px*py*4.0+C1*S1*a3**2*px*py*4.0-C1*S1*d1**2*px*py*4.0-C1*S1*px*py*pz**2*4.0-C1**2*S1*a1*a2**2*py*8.0+C1**2*S1*a1*px**2*py*8.0+C1*S1**3*a2**2*px*py*8.0+C1**3*S1*a2**2*px*py*8.0+C1*S1*d1*px*py*pz*8.0)-a1**2*1j-a2**2*1j+a3**2*1j-d1**2*1j-py**2*1j-pz**2*1j-C1**2*px**2*1j+C1**2*py**2*1j+C1*a1*px*2.0j+S1*a1*py*2.0j-C1*S1*px*py*2.0j)-np.angle(-a2*(a1*-1j+d1-pz+C1*px*1j+S1**3*py*1j+C1**2*S1*py*1j))  # noqa
    #                     print(theta[1])

    #                 S2 = np.sin(theta[1])
    #                 C2 = np.cos(theta[1])

    #                 if sol[2] == 1:
    #                     theta[2] = np.angle(a3*1j-np.sqrt(0j+d1*pz*-2.0+a1**2+a2**2-a3**2+d1**2+py**2+pz**2+C1**2*px**2-C1**2*py**2+C2*a1*a2*2.0-C1*a1*px*2.0+S2*a2*d1*2.0-S1*a1*py*2.0-S2*a2*pz*2.0-C1*C2*a2*px*2.0-C2*S1*a2*py*2.0+C1*S1*px*py*2.0))-np.angle(a2*-1j-C2*a1*1j-C2*d1+C2*pz+S2*a1-S2*d1*1j+S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)  # noqa
    #                 else:
    #                     theta[2] = -np.angle(a2*-1j-C2*a1*1j-C2*d1+C2*pz+S2*a1-S2*d1*1j+S2*pz*1j+C1*C2*px*1j-C1*S2*px+C2*S1*py*1j-S1*S2*py)+np.angle(a3*1j+np.sqrt(0j+d1*pz*-2.0+a1**2+a2**2-a3**2+d1**2+py**2+pz**2+C1**2*px**2-C1**2*py**2+C2*a1*a2*2.0-C1*a1*px*2.0+S2*a2*d1*2.0-S1*a1*py*2.0-S2*a2*pz*2.0-C1*C2*a2*px*2.0-C2*S1*a2*py*2.0+C1*S1*px*py*2.0))  # noqa

    #         elif self.ikineType == 'rrp':
    #             # RRP (Stanford arm like)
    #             px = Ti[0, 3]
    #             py = Ti[1, 3]
    #             pz = Ti[2, 3]
    #             d1 = self.links[0].d
    #             d2 = self.links[1].d

    #             # Autogenerated code
    #             if self.links[0].alpha < 0:
    #                 if sol[0] == 1:
    #                     theta[0] = -np.angle(-px+py*1j)+np.angle(d2*1j-np.sqrt(0j+-d2**2+px**2+py**2))  # noqa
    #                 else:
    #                     theta[0] = np.angle(d2*1j+np.sqrt(0j+-d2**2+px**2+py**2))-np.angle(-px+py*1j)  # noqa

    #                 S1 = np.sin(theta[0])
    #                 C1 = np.cos(theta[0])

    #                 if sol[1] == 1:
    #                     theta[1] = np.angle(d1-pz-C1*px*1j-S1*py*1j)  # noqa
    #                 else:
    #                     theta[1] = np.angle(-d1+pz+C1*px*1j+S1*py*1j)  # noqa

    #                 S2 = np.sin(theta[1])
    #                 C2 = np.cos(theta[1])

    #                 theta[2] = -C2*d1+C2*pz+C1*S2*px+S1*S2*py  # noqa

    #             else:
    #                 if sol[0] == 1:
    #                     theta[0] = -np.angle(px-py*1j)+np.angle(d2*1j-np.sqrt(0j+-d2**2+px**2+py**2))  # noqa
    #                 else:
    #                     theta[0] = -np.angle(px-py*1j)+np.angle(d2*1j+np.sqrt(0j+-d2**2+px**2+py**2))  # noqa

    #                 S1 = np.sin(theta[0])
    #                 C1 = np.cos(theta[0])

    #                 if sol[1] == 1:
    #                     theta[1] = np.angle(-d1+pz-C1*px*1j-S1*py*1j)  # noqa
    #                 else:
    #                     theta[1] = np.angle(d1-pz+C1*px*1j+S1*py*1j)  # noqa

    #                 print(theta[1])

    #                 S2 = np.sin(theta[1])
    #                 C2 = np.cos(theta[1])

    #                 theta[2] = -C2*d1+C2*pz-C1*S2*px-S1*S2*py  # noqa

    #         if not np.all(np.isnan(theta)):
    #             # Solve for the wrist rotation
    #             # We need to account for some random translations between the
    #             # first and last 3 joints (d4) and also d6,a6,alpha6 in the
    #             # final frame.

    #             # Transform of first 3 joints
    #             T13 = self.A([0, 2], theta)

    #             # T = T13 * Tz(d4) * R * Tz(d6) Tx(a5)
    #             Td4 = SE3(0, 0, self.links[3].d)      # Tz(d4)

    #             # Tz(d6) Tx(a5) Rx(alpha6)
    #             Tt = SE3(self.links[5].a, 0, self.links[5].d) * \
    #                 SE3.Rx(self.links[5].alpha)

    #             R = np.linalg.inv(Td4.A) @ np.linalg.inv(T13.A) @ Ti @ \
    #                 np.linalg.inv(Tt.A)

    #             # The spherical wrist implements Euler angles
    #             if sol[2] == 1:
    #                 theta[3:6] = tr2eul(R, flip=True)
    #             else:
    #                 theta[3:6] = tr2eul(R)

    #             if self.links[3].alpha > 0:
    #                 theta[4] = -theta[4]

    #             # Remove the link offset angles
    #             for k in range(self.n):
    #                 theta[k] -= self.links[k].offset

    #             q[j, :] = theta
    #         else:
    #             err.append('point not reachable')

    #     if trajn == 1:
    #         return q[0, :], err
    #     else:
    #         return q, err

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

        # TODO use getmatrix
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


    # def jacob0v(self, q=None):
    #     """
    #     Jv = jacob0v(q) is the spatial velocity Jacobian, at joint
    #     configuration q, which relates the velocity in the end-effector frame
    #     to velocity in the base frame

    #     Jv = jacob0v() as above except uses the stored q value of the
    #     robot object.

    #     :param q: The joint configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)

    #     :returns J: The velocity Jacobian in 0 frame
    #     :rtype J: float ndarray(6,6)

    #     """

    #     if q is None:
    #         q = np.copy(self.q)
    #     else:
    #         q = getvector(q, self.n)

    #     R = self.fkine(q).R

    #     Jv = np.zeros((6, 6), dtype=R.dtype)
    #     Jv[:3, :3] = R
    #     Jv[3:, 3:] = R

    #     return Jv

    # def jacobev(self, q=None):
    #     """
    #     Jv = jacobev(q) is the spatial velocity Jacobian, at joint
    #     configuration q, which relates the velocity in the base frame to the
    #     velocity in the end-effector frame.

    #     Jv = jacobev() as above except uses the stored q value of the
    #     robot object.

    #     :param q: The joint configuration of the robot (Optional,
    #         if not supplied will use the stored q values).
    #     :type q: float ndarray(n)

    #     :returns J: The velocity Jacobian in ee frame
    #     :rtype J: float ndarray(6,6)

    #     """

    #     if q is None:
    #         q = np.copy(self.q)
    #     else:
    #         q = getvector(q, self.n)

    #     R = self.fkine(q).R

    #     Jv = np.zeros((6, 6), dtype=R.dtype)
    #     Jv[:3, :3] = R.T
    #     Jv[3:, 3:] = R.T

    #     return Jv

class SerialLink(DHRobot):
    def __init__(self, *args, **kwargs):
        print('SerialLink is deprecated, use DHRobot instead')
        super().__init__(*args, **kwargs)

    

if __name__ == "__main__":

    import roboticstoolbox as rtb
    # import spatialmath.base.symbolic as sym

    planar = rtb.models.DH.Planar2()
    # J = puma.jacob0(puma.qn)
    # print(J)
    # print(puma.manipulability(puma.qn))
    # print(puma.manipulability(puma.qn, 'asada'))
    #tw, T0 = puma.twists(puma.qz)
    print(planar)

    puma = rtb.models.DH.Puma560()
    print(puma)
    puma.base = None
    print('base', puma.base)
    print('tool', puma.tool)

    print(puma.ets())

    puma[2].flip = True
    puma[3].offset = 1
    puma[4].flip = True
    puma[4].offset = -1
    print(puma)
    print(puma.ets())

    print(puma.dyntable())