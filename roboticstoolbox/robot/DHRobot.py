#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from collections import namedtuple
from roboticstoolbox.tools.data import rtb_path_to_datafile
import warnings
import copy
import numpy as np
from roboticstoolbox.robot.Robot import Robot  # DHLink
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.DHLink import DHLink  # HACK
from spatialmath.base.argcheck import getvector, isscalar, verifymatrix, getmatrix

# from spatialmath import base
from spatialmath.base import tr2jac, tr2eul, tr2rpy, t2r, eul2jac, rpy2jac
from spatialmath import SE3, Twist3
import spatialmath.base.symbolic as sym

# from scipy.optimize import minimize, Bounds
from ansitable import ANSITable, Column
from scipy.linalg import block_diag
from roboticstoolbox.robot.DHLink import _check_rne
from frne import init, frne, delete

iksol = namedtuple("IKsolution", "q, success, reason")


class DHRobot(Robot):
    """
    Class for robots defined using Denavit-Hartenberg notation

    :param L: List of links which define the robot
    :type L: list(n)
    :param name: Name of the robot
    :type name: str
    :param manufacturer: Manufacturer of the robot
    :type manufacturer: str
    :param base: Location of the base
    :type base: SE3
    :param tool: Location of the tool
    :type tool: SE3
    :param gravity: Gravitational acceleration vector
    :type gravity: ndarray(3)

    A concrete superclass for arm type robots defined using Denavit-Hartenberg
    notation, that represents a serial-link arm-type robot.  Each link and
    joint in the chain is described by a DHLink-class object using
    Denavit-Hartenberg parameters (standard or modified).

    .. note:: Link subclass elements passed in must be all standard, or all
          modified, DH parameters.

    :reference:

        - Robotics, Vision & Control, Chaps 7-9,
          P. Corke, Springer 2011.
        - Robot, Modeling & Control,
          M.Spong, S. Hutchinson & M. Vidyasagar, Wiley 2006.

    """

    def __init__(self, links, meshdir=None, **kwargs):

        # Verify L
        if not isinstance(links, list):
            raise TypeError("The links must be stored in a list.")

        all_links = []
        self._n = 0

        for link in links:
            if isinstance(link, DHLink):
                # got a link
                all_links.append(link)
                link.number = self._n + 1
                link.jindex = self._n
                self._n += 1

                link.name = f"link{self._n}"

            elif isinstance(link, DHRobot):
                # link is actually a robot

                # copy the links
                rlinks = copy.copy(link.links)
                for rlink in rlinks:
                    all_links.append(rlink)
                    rlink.number = self._n
                    rlink.jindex = self._n
                    self._n += 1

                    rlink.name = f"link{self._n}"
            else:
                raise TypeError("Input can be only DHLink or DHRobot")

        super().__init__(all_links, **kwargs)

        self.ee_links = [self.links[-1]]

        # Check the DH convention
        self._mdh = self.links[0].mdh
        if not all([link.mdh == self.mdh for link in self.links]):
            raise ValueError("Robot has mixed D&H link conventions")

        # load meshes if required
        if meshdir is not None:
            self.meshdir = rtb_path_to_datafile(meshdir)
            self.basemesh = self.meshdir / "link0.stl"
            for j, link in enumerate(self._links, start=1):
                link.mesh = self.meshdir / "link{:d}.stl".format(j)
            self._hasgeometry = True
        else:
            self.basemesh = None

        # rne parameters
        self._rne_ob = None

    def __str__(self):
        """
        Pretty prints the DH Model of the robot. Will output angles in degrees

        :return: Pretty print of the robot model
        :rtype: str
        """

        s = f"DHRobot: {self.name}"

        if self.manufacturer is not None and len(self.manufacturer) > 0:
            s += f" (by {self.manufacturer})"
        s += f", {self.n} joints ({self.structure})"

        deg = 180 / np.pi

        if self._hasdynamics:
            s += ", dynamics"
        if any([link.mesh is not None for link in self.links]):
            s += ", geometry"

        if self.mdh:
            dh = "modified"
        else:
            dh = "standard"
        s += f", {dh} DH parameters\n"

        def qstr(j, link):
            j += 1
            if link.flip:
                s = f"-q{j:d}"
            else:
                s = f" q{j:d}"

            if L.offset != 0:
                sign = "+" if L.offset > 0 else "-"
                offset = abs(L.offset)
                if link.isprismatic:
                    s += f" {sign} {offset:}"
                else:
                    s += f" {sign} {offset * deg:.3g}\u00b0"
            return s

        def angle(theta, fmt=None):
            if sym.issymbol(theta):  # pragma nocover
                return "<<red>>" + str(theta)
            else:
                if fmt is not None:
                    return fmt.format(theta * deg) + "\u00b0"
                else:
                    return str(theta * deg) + "\u00b0"

        has_qlim = any([link._qlim is not None for link in self])
        if has_qlim:
            qlim_columns = [
                Column("q⁻", headalign="^"),
                Column("q⁺", headalign="^"),
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
                border="thick",
            )
            for j, L in enumerate(self):
                if has_qlim:
                    if L.isprismatic:
                        ql = [qlim[0, j], qlim[1, j]]
                    else:
                        ql = [angle(qlim[k, j], "{:.1f}") for k in [0, 1]]
                else:
                    ql = []
                if L.isprismatic:
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
                border="thick",
            )
            for j, L in enumerate(self):
                if has_qlim:
                    if L.isprismatic:
                        ql = [qlim[0, j], qlim[1, j]]
                    else:
                        ql = [angle(qlim[k, j], "{:.1f}") for k in [0, 1]]
                else:
                    ql = []
                if L.isprismatic:
                    table.row(
                        angle(L.theta), qstr(j, L), f"{L.a:.4g}", angle(L.alpha), *ql
                    )
                else:
                    table.row(
                        qstr(j, L), f"{L.d:.4g}", f"{L.a:.4g}", angle(L.alpha), *ql
                    )

        s += str(table)

        # show tool and base
        if self._tool is not None or self._tool is not None:
            table = ANSITable(
                Column("", colalign=">"),
                Column("", colalign="<"),
                border="thin",
                header=False,
            )
            if self._base is not None:
                table.row(
                    "base",
                    self._base.printline(orient="rpy/xyz", fmt="{:.2g}", file=None),
                )
            if self._tool is not None:
                table.row(
                    "tool",
                    self._tool.printline(orient="rpy/xyz", fmt="{:.2g}", file=None),
                )
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
            raise TypeError(
                "Can only combine DHRobots with other " "DHRobots or DHLinks"
            )

        return DHRobot(
            nlinks,
            name=self.name,
            manufacturer=self.manufacturer,
            base=self.base,
            tool=self.tool,
            gravity=self.gravity,
        )

    # def copy(self):
    #     """
    #     Copy a robot

    #     :return: A deepish copy of the robot
    #     :rtype: Robot subclass instance
    #     """

    #     L = [link.copy() for link in self]

    #     new = DHRobot(
    #         L,
    #         name=self.name,
    #         manufacturer=self.manufacturer,
    #         base=self.base,
    #         tool=self.tool,
    #         gravity=self.gravity)

    #     new.q = self.q
    #     new.qd = self.qd
    #     new.qdd = self.qdd

    #     return new

    # --------------------------------------------------------------------- #

    def _set_link_fk(self, q):
        """
        robot._set_link_fk(q) evaluates fkine for each link within a
        robot and stores that pose in a private variable within the link.

        This method is not for general use.

        :param q: The joint angles/configuration of the robot
        :type q: float ndarray(n)

        .. note::

            - The robot's base transform, if present, are incorporated
              into the result.
        """

        q = getvector(q, self.n)

        # t = self.base

        tall = self.fkine_all(q, old=True)

        for i, link in enumerate(self.links):

            # Update the link model transforms
            for col in link.collision:
                col.wT = tall[i]

            for gi in link.geometry:
                gi.wT = tall[i]

    # --------------------------------------------------------------------- #

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
            v.append(self.links[i].alpha)
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

        A conservative estimate of the reach of the robot. It is computed as
        :math:`\sum_j |a_j| + |d_j|` where :math:`d_j` is taken as the maximum
        joint coordinate (``qlim``) if the joint is primsmatic.

        .. note::

            - This is the *length sum* referred to in Craig's book
            - Probably an overestimate of the actual reach
            - Used by numerical inverse kinematics to scale translational
              error.
            - For a prismatic joint, uses ``qlim`` if it is set

        .. warning:: Computed on the first access. If kinematic parameters
              subsequently change this will not be reflected.
        """
        if self._reach is None:
            d = 0
            for link in self:
                d += abs(link.a) + (link.d)
                if link.isprismatic and link.qlim is not None:
                    d += link.qlim[1]
            self._reach = d
        return self._reach

    @property
    def nbranches(self):
        """
        Number of branches

        :return: number of branches in the robot's kinematic tree
        :rtype: int

        Number of branches in this robot.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Panda()
            >>> robot.nbranches

        :seealso: :func:`n`, :func:`nlinks`
        """
        return 1

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

        - ``robot.A(j, q)`` transform between link frames {0} and {j}.  ``q``
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

        L = self.links[self.n - 3 : self.n]

        alpha = [-np.pi / 2, np.pi / 2]

        return (
            L[0].a == 0
            and L[1].a == 0
            and L[1].d == 0
            and (
                (L[0].alpha == alpha[0] and L[1].alpha == alpha[1])
                or (L[0].alpha == alpha[1] and L[1].alpha == alpha[0])
            )
            and L[0].sigma == 0
            and L[1].sigma == 0
            and L[2].sigma == 0
        )

    def dhunique(self):
        """
        Print the unique DH parameters

        Print a table showing all the non-zero DH parameters, and their
        values.  This is the minimum set of kinematic parameters required
        to describe the robot.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.dhunique()
        """

        table = ANSITable(
            Column("param"),
            Column("value", headalign="^", colalign="<", fmt="{:.4g}"),
            border="thin",
        )
        for j, link in enumerate(self):
            if link.isprismatic:
                if link.theta != 0:
                    table.row(f"θ{j}", link.theta)
            elif link.isrevolute:
                if link.d != 0:
                    table.row(f"d{j}", link.d)
            if link.a != 0:
                table.row(f"a{j}", link.a)
            if link.alpha != 0:
                table.row(f"⍺{j}", link.alpha)
        table.print()

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
                    tw[j] = Twist3.UnitRevolute(T[j].a, T[j].t)
                else:
                    tw[j] = Twist3.UnitPrismatic(T[j].a)
        else:
            # DH case
            for j, link in enumerate(self):
                if j == 0:
                    # first link case
                    if link.sigma == 0:
                        # revolute
                        tw[j] = Twist3.UnitRevolute([0, 0, 1], [0, 0, 0])
                    else:
                        tw[j] = Twist3.UnitPrismatic([0, 0, 1])  # prismatic
                else:
                    # subsequent links
                    if link.sigma == 0:
                        tw[j] = Twist3.UnitRevolute(T[j - 1].a, T[j - 1].t)  # revolute
                    else:
                        tw[j] = Twist3.UnitPrismatic(T[j - 1].a)  # prismatic

        return tw, T[-1]

    def ets(self):
        """
        Robot kinematics as an elemenary transform sequence

        :return: elementary transform sequence
        :rtype: ETS

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.ets()
        """

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

    def fkine(self, q, **kwargs):
        """
        Forward kinematics

        :param q: The joint configuration
        :type q: ndarray(n) or ndarray(m,n)
        :return: Forward kinematics as an SE(3) matrix
        :rtype: SE3 instance

        - ``robot.fkine(q)`` computes the forward kinematics for the robot at
          joint configuration ``q``.

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

    def fkine_path(self, q, old=None):
        """
        Compute the pose of every link frame

        :param q: The joint configuration
        :type q:  darray(n)
        :return: Pose of all links
        :rtype: SE3 instance

        ``T = robot.fkine_path(q)`` is  an SE3 instance with ``robot.nlinks +
        1`` values:

        - ``T[0]`` is the base transform
        - ``T[i+1]`` is the pose of link whose ``number`` is ``i``

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """
        T = self.base
        q = getvector(q)
        Tj = T

        for q, L in zip(q, self.links):
            Tj *= L.A(q)
            T.append(Tj)

        if self._tool is not None:
            T[-1] *= self._tool

        return T

    def segments(self):

        segments = [None]
        segments.extend(self.links)
        return [segments]

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

        Tj = self.base.copy()
        Tall = Tj

        for q, L in zip(q, self.links):
            Tj *= L.A(q)
            Tall.append(Tj)
        return Tall

    def jacobe(self, q, half=None, **kwargs):
        r"""
        Manipulator Jacobian in end-effector frame

        :param q: Joint coordinate vector
        :type q: ndarray(n)
        :param half: return half Jacobian: 'trans' or 'rot'
        :type half: str
        :return J: The manipulator Jacobian in the end-effector frame
        :rtype: ndarray(6,n)

        - ``robot.jacobe(q)`` is the manipulator Jacobian matrix which maps
          joint  velocity to end-effector spatial velocity.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{E}\!\nu = \mathbf{J}_m(q) \dot{q}`.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.jacobe([0, 0, 0, 0, 0, 0])

        .. warning:: This is the **geometric Jacobian** as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a
            velocity twist as per the text by Lynch & Park.
        """  # noqa

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
                d = np.array(
                    [
                        -U[0, 0] * U[1, 3] + U[1, 0] * U[0, 3],
                        -U[0, 1] * U[1, 3] + U[1, 1] * U[0, 3],
                        -U[0, 2] * U[1, 3] + U[1, 2] * U[0, 3],
                    ]
                )
                delta = U[2, :3]  # nz oz az
            else:
                # prismatic axis
                d = U[2, :3]  # nz oz az
                delta = np.zeros((3,))

            J[:, j] = np.r_[d, delta]

            if self.mdh != 0:
                # modified DH convention
                U = L[j].A(q[j]).A @ U

        # return top or bottom half if asked
        if half is not None:
            if half == "trans":
                return J[:3, :]
            elif half == "rot":
                return J[3:, :]
            else:
                raise ValueError("bad half specified")

        return J

    def jacob0(self, q=None, T=None, half=None, analytical=None, start=None, end=None):
        r"""
        Manipulator Jacobian in world frame

        :param q: Joint coordinate vector
        :type q: ndarray(n)
        :param T: Forward kinematics if known, SE(3 matrix)
        :type T: SE3 instance
        :param half: return half Jacobian: 'trans' or 'rot'
        :type half: str
        :param analytical: return analytical Jacobian instead of geometric Jacobian (default)
        :type analytical: str
        :return J: The manipulator Jacobian in the world frame
        :rtype: ndarray(6,n)

        - ``robot.jacob0(q)`` is the manipulator Jacobian matrix which maps
          joint velocity to end-effector spatial velocity.

        End-effector spatial velocity :math:`\nu = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)^T`
        is related to joint velocity by :math:`{}^{0}\!\nu = \mathbf{J}_0(q) \dot{q}`.

        ``analytical`` can be one of:

            =============  ==================================
            Value          Rotational representation
            =============  ==================================
            ``'rpy/xyz'``  RPY angular rates in XYZ order
            ``'rpy/zyx'``  RPY angular rates in XYZ order
            ``'eul'``      Euler angular rates in ZYZ order
            ``'exp'``      exponential coordinate rates
            =============  ==================================

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.jacob0([0, 0, 0, 0, 0, 0])

        .. warning:: The **geometric Jacobian** is as described in texts by
            Corke, Spong etal., Siciliano etal.  The end-effector velocity is
            described in terms of translational and angular velocity, not a
            velocity twist as per the text by Lynch & Park.

        .. note:: ``T`` can be passed in to save the cost of computing forward
            kinematics which is needed to transform velocity from end-effector
            frame to world frame.

        """  # noqa
        q = getvector(q, self.n)

        if T is None:
            T = self.fkine(q)
        T = T.A

        # compute Jacobian in EE frame and transform to world frame
        J0 = tr2jac(T) @ self.jacobe(q)

        # compute rotational transform if analytical Jacobian required
        if analytical is not None:

            if analytical == "rpy/xyz":
                rpy = tr2rpy(T, "xyz")
                A = rpy2jac(rpy, "xyz")
            elif analytical == "rpy/zyx":
                rpy = tr2rpy(T, "zyx")
                A = rpy2jac(rpy, "zyx")
            elif analytical == "eul":
                eul = tr2eul(T)
                A = eul2jac(eul)
            elif analytical == "exp":
                # TODO: move to SMTB.base, Horner form with skew(v)
                (theta, v) = trlog(t2r(T))
                A = (
                    np.eye(3, 3)
                    - (1 - math.cos(theta)) / theta * skew(v)
                    + (theta - math.sin(theta)) / theta * skew(v) ** 2
                )
            else:
                raise ValueError("bad analytical value specified")

            J0 = block_diag(np.eye(3, 3), np.linalg.inv(A)) @ J0

        # TODO optimize computation above if half matrix is returned

        # return top or bottom half if asked
        if half is not None:
            if half == "trans":
                J0 = J0[:3, :]
            elif half == "rot":
                J0 = J0[3:, :]
            else:
                raise ValueError("bad half specified")
        return J0

    def hessian0(self, q=None, J0=None, start=None, end=None):
        r"""
        Manipulator Hessian in base frame

        :param q: joint coordinates
        :type q: array_like
        :param J0: Jacobian in {0} frame
        :type J0: ndarray(6,n)
        :return: Hessian matrix
        :rtype: ndarray(6,n,n)

        This method calculcates the Hessisan in the base frame. One of ``J0`` or
        ``q`` is required. If ``J0`` is already calculated for the joint
        coordinates ``q`` it can be passed in to to save computation time.

        If we take the time derivative of the differential kinematic
        relationship

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

        :seealso: :func:`jacob0`, :func:`jacob_dot`
        """

        return self.ets().hessian0(q, J0)

    def _get_limit_links(self, end=None, start=None):
        # For compatibility with ERobot

        return None, None, None

    # -------------------------------------------------------------------------- #

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
            L[j + 7 : j + 10] = self.links[i].r.flatten()
            L[j + 10 : j + 19] = self.links[i].I.flatten()
            L[j + 19] = self.links[i].Jm
            L[j + 20] = self.links[i].G
            L[j + 21] = self.links[i].B
            L[j + 22 : j + 24] = self.links[i].Tc.flatten()

        # we negate gravity here, since the C code has the sign wrong
        self._rne_ob = init(self.n, self.mdh, L, -self.gravity)

    def delete_rne(self):
        """
        Frees the memory holding the robot object in c if the robot object
        has been initialised in c.
        """
        if self._rne_ob is not None:
            delete(self._rne_ob)
            self._dynchanged = False
            self._rne_ob = None

    @_check_rne
    def rne(self, q, qd=None, qdd=None, gravity=None, fext=None):
        r"""
        Inverse dynamics

        :param q: Joint coordinates
        :type q: ndarray(n)
        :param qd: Joint velocity
        :type qd: ndarray(n)
        :param qdd: The joint accelerations of the robot
        :type qdd: ndarray(n)
        :param gravity: Gravitational acceleration to override robot's gravity
            value
        :type gravity: ndarray(6)
        :param fext: Specify wrench acting on the end-effector
                     :math:`W=[F_x F_y F_z M_x M_y M_z]`
        :type fext: ndarray(6)

        ``tau = rne(q, qd, qdd, grav, fext)`` is the joint torque required for
        the robot to achieve the specified joint position ``q`` (1xn), velocity
        ``qd`` (1xn) and acceleration ``qdd`` (1xn), where n is the number of
        robot joints. ``fext`` describes the wrench acting on the end-effector

        Trajectory operation:
        If q, qd and qdd (mxn) are matrices with m cols representing a
        trajectory then tau (mxn) is a matrix with cols corresponding to each
        trajectory step.

        .. note::
            - The torque computed contains a contribution due to armature
              inertia and joint friction.
            - If a model has no dynamic parameters set the result is zero.

        :seealso: :func:`rne_python`
        """
        trajn = 1

        try:
            q = getvector(q, self.n, "row")
            qd = getvector(qd, self.n, "row")
            qdd = getvector(qdd, self.n, "row")
        except ValueError:
            trajn = q.shape[0]
            verifymatrix(q, (trajn, self.n))
            verifymatrix(qd, (trajn, self.n))
            verifymatrix(qdd, (trajn, self.n))

        if gravity is None:
            gravity = self.gravity
        else:
            gravity = getvector(gravity, 3)

        # The c function doesn't handle base rotation, so we need to hack the
        # gravity vector instead
        gravity = self.base.R.T @ gravity

        if fext is None:
            fext = np.zeros(6)
        else:
            fext = getvector(fext, 6)

        tau = np.zeros((trajn, self.n))

        for i in range(trajn):
            tau[i, :] = frne(
                # we negate gravity here, since the C code has the sign wrong
                self._rne_ob,
                q[i, :],
                qd[i, :],
                qdd[i, :],
                -gravity,
                fext,
            )

        if trajn == 1:
            return tau[0, :]
        else:
            return tau

    def rne_python(
        self,
        Q,
        QD=None,
        QDD=None,
        gravity=None,
        fext=None,
        debug=False,
        basewrench=False,
    ):
        """
        Compute inverse dynamics via recursive Newton-Euler formulation

        :param Q: Joint coordinates
        :param QD: Joint velocity
        :param QDD: Joint acceleration
        :param gravity: gravitational acceleration, defaults to attribute
            of self
        :type gravity: array_like(3), optional
        :param fext: end-effector wrench, defaults to None
        :type fext: array-like(6), optional
        :param debug: print debug information to console, defaults to False
        :type debug: bool, optional
        :param basewrench: compute the base wrench, defaults to False
        :type basewrench: bool, optional
        :raises ValueError: for misshaped inputs
        :return: Joint force/torques
        :rtype: NumPy array

        Recursive Newton-Euler for standard Denavit-Hartenberg notation.

        - ``rne_dh(q, qd, qdd)`` where the arguments have shape (n,) where n is
          the number of robot joints.  The result has shape (n,).
        - ``rne_dh(q, qd, qdd)`` where the arguments have shape (m,n) where n
          is the number of robot joints and where m is the number of steps in
          the joint trajectory.  The result has shape (m,n).
        - ``rne_dh(p)`` where the input is a 1D array ``p`` = [q, qd, qdd] with
          shape (3n,), and the result has shape (n,).
        - ``rne_dh(p)`` where the input is a 2D array ``p`` = [q, qd, qdd] with
          shape (m,3n) and the result has shape (m,n).

        .. note::
            - This is a pure Python implementation and slower than the .rne()
            which is written in C.
            - This version supports symbolic model parameters
            - Verified against MATLAB code

        :seealso: :func:`rne`
        """

        def removesmall(x):
            return x

        n = self.n

        if self.symbolic:
            dtype = "O"
        else:
            dtype = None

        z0 = np.array([0, 0, 1], dtype=dtype)

        if gravity is None:
            gravity = self.gravity  # default gravity from the object
        else:
            gravity = getvector(gravity, 3)

        if fext is None:
            fext = np.zeros((6,), dtype=dtype)
        else:
            fext = getvector(fext, 6)

        if debug:
            print("grav", gravity)
            print("fext", fext)

        # unpack the joint coordinates and derivatives
        if Q is not None and QD is None and QDD is None:
            # single argument case
            Q = getmatrix(Q, (None, self.n * 3))
            q = Q[:, 0:n]
            qd = Q[:, n : 2 * n]
            qdd = Q[:, 2 * n :]

        else:
            # 3 argument case
            q = getmatrix(Q, (None, self.n))
            qd = getmatrix(QD, (None, self.n))
            qdd = getmatrix(QDD, (None, self.n))

        nk = q.shape[0]

        tau = np.zeros((nk, n), dtype=dtype)
        if basewrench:
            wbase = np.zeros((nk, n), dtype=dtype)

        for k in range(nk):
            # take the k'th row of data
            q_k = q[k, :]
            qd_k = qd[k, :]
            qdd_k = qdd[k, :]

            if debug:
                print("q_k", q_k)
                print("qd_k", qd_k)
                print("qdd_k", qdd_k)
                print()

            # joint vector quantities stored columwise in matrix
            #  m suffix for matrix
            Fm = np.zeros((3, n), dtype=dtype)
            Nm = np.zeros((3, n), dtype=dtype)
            # if robot.issym
            #     pstarm = sym([]);
            # else
            #     pstarm = [];
            pstarm = np.zeros((3, n), dtype=dtype)
            Rm = []

            # rotate base velocity and acceleration into L1 frame
            # base has zero angular velocity
            w = np.zeros((3,), dtype=dtype)
            # base has zero angular acceleration
            wd = np.zeros((3,), dtype=dtype)
            vd = -gravity

            if self._base is not None:
                Rb = t2r(self.base.A).T
                w = Rb @ w
                wd = Rb @ wd
                vd = Rb @ gravity

            # ----------------  initialize some variables ----------------- #

            for j in range(n):
                link = self.links[j]

                # compute the link rotation matrix
                if link.sigma == 0:
                    # revolute axis
                    Tj = link.A(q_k[j]).A
                    d = link.d
                else:
                    # prismatic
                    Tj = link.A(link.theta).A
                    d = q_k[j]

                # compute pstar:
                #   O_{j-1} to O_j in {j}, negative inverse of link xform
                alpha = link.alpha
                if self.mdh:
                    pstar = np.r_[link.a, -d * sym.sin(alpha), d * sym.cos(alpha)]
                    if j == 0:
                        if self._base:
                            Tj = self._base.A @ Tj
                            pstar = self._base.A @ pstar
                else:
                    pstar = np.r_[link.a, d * sym.sin(alpha), d * sym.cos(alpha)]

                # stash them for later
                Rm.append(t2r(Tj))
                pstarm[:, j] = pstar

            # -----------------  the forward recursion -------------------- #

            for j, link in enumerate(self.links):

                Rt = Rm[j].T  # transpose!!
                pstar = pstarm[:, j]
                r = link.r

                # statement order is important here

                if self.mdh:
                    if link.isrevolute:
                        # revolute axis
                        w_ = Rt @ w + z0 * qd_k[j]
                        wd_ = Rt @ wd + z0 * qdd_k[j] + _cross(Rt @ w, z0 * qd_k[j])
                        vd_ = Rt @ _cross(wd, pstar) + _cross(w, _cross(w, pstar)) + vd
                    else:
                        # prismatic axis
                        w_ = Rt @ w
                        wd_ = Rt @ wd
                        vd_ = (
                            Rt @ (_cross(wd, pstar) + _cross(w, _cross(w, pstar)) + vd)
                            + 2 * _cross(Rt @ w, z0 * qd_k[j])
                            + z0 * qdd_k[j]
                        )
                    # trailing underscore means new value, update here
                    w = w_
                    wd = wd_
                    vd = vd_
                else:
                    if link.isrevolute:
                        # revolute axis
                        wd = Rt @ (wd + z0 * qdd_k[j] + _cross(w, z0 * qd_k[j]))
                        w = Rt @ (w + z0 * qd_k[j])
                        vd = _cross(wd, pstar) + _cross(w, _cross(w, pstar)) + Rt @ vd
                    else:
                        # prismatic axis
                        w = Rt @ w
                        wd = Rt @ wd
                        vd = (
                            Rt @ (z0 * qdd_k[j] + vd)
                            + _cross(wd, pstar)
                            + 2 * _cross(w, Rt @ z0 * qd_k[j])
                            + _cross(w, _cross(w, pstar))
                        )

                vhat = _cross(wd, r) + _cross(w, _cross(w, r)) + vd
                Fm[:, j] = link.m * vhat
                Nm[:, j] = link.I @ wd + _cross(w, link.I @ w)

                if debug:
                    print("w:     ", removesmall(w))
                    print("wd:    ", removesmall(wd))
                    print("vd:    ", removesmall(vd))
                    print("vdbar: ", removesmall(vhat))
                    print()

            if debug:
                print("Fm\n", Fm)
                print("Nm\n", Nm)

            # -----------------  the backward recursion -------------------- #

            f = fext[:3]  # force/moments on end of arm
            nn = fext[3:]

            for j in range(n - 1, -1, -1):
                link = self.links[j]
                r = link.r

                #
                # order of these statements is important, since both
                # nn and f are functions of previous f.
                #
                if self.mdh:
                    if j == (n - 1):
                        R = np.eye(3, dtype=dtype)
                        pstar = np.zeros((3,), dtype=dtype)
                    else:
                        R = Rm[j + 1]
                        pstar = pstarm[:, j + 1]

                    f_ = R @ f + Fm[:, j]
                    nn_ = (
                        R @ nn
                        + _cross(pstar, R @ f)
                        + _cross(pstar, Fm[:, j])
                        + Nm[:, j]
                    )
                    f = f_
                    nn = nn_

                else:
                    pstar = pstarm[:, j]
                    if j == (n - 1):
                        R = np.eye(3, dtype=dtype)
                    else:
                        R = Rm[j + 1]

                    nn = (
                        R @ (nn + _cross(R.T @ pstar, f))
                        + _cross(pstar + r, Fm[:, j])
                        + Nm[:, j]
                    )
                    f = R @ f + Fm[:, j]

                if debug:
                    print("f: ", removesmall(f))
                    print("n: ", removesmall(nn))

                R = Rm[j]
                if self.mdh:
                    if link.isrevolute:
                        # revolute axis
                        t = nn @ z0
                    else:
                        # prismatic
                        t = f @ z0
                else:
                    if link.isrevolute:
                        # revolute axis
                        t = nn @ (R.T @ z0)
                    else:
                        # prismatic
                        t = f @ (R.T @ z0)

                # add joint inertia and friction
                #  no Coulomb friction if model is symbolic
                tau[k, j] = (
                    t
                    + link.G ** 2 * link.Jm * qdd_k[j]
                    - link.friction(qd_k[j], coulomb=not self.symbolic)
                )
                if debug:
                    print(
                        f"j={j:}, G={link.G:}, Jm={link.Jm:}, friction={link.friction(qd_k[j], coulomb=False):}"
                    )  # noqa
                    print()

            # compute the base wrench and save it
            if basewrench:
                R = Rm[0]
                nn = R @ nn
                f = R @ f
                wbase[k, :] = np.r_[f, nn]

        # if self.symbolic:
        #     # simplify symbolic expressions
        #     print(
        #       'start symbolic simplification, this might take a while...')
        #     # from sympy import trigsimp

        #     # tau = trigsimp(tau)
        #     # consider using multiprocessing to spread over cores
        #     #  https://stackoverflow.com/questions/33844085/using-multiprocessing-with-sympy  # noqa
        #     print('done')
        #     if tau.shape[0] == 1:
        #         return tau.reshape(self.n)
        #     else:
        #         return tau

        if tau.shape[0] == 1:
            return tau.flatten()
        else:
            return tau

    # -------------------------------------------------------------------------- #

    def ikine_6s(self, T, config, ikfunc):
        # Undo base and tool transformations, but if they are not
        # set, skip the operation.  Nicer for symbolics
        if self._base is not None:
            T = self.base.inv() * T
        if self._tool is not None:
            T = self.tool.inv() * T

        # q = np.zeros((6,))
        solutions = []

        for k, Tk in enumerate(T):
            # get model specific solution for first 3 joints
            theta = ikfunc(self, Tk, config)

            if isinstance(theta, np.ndarray):
                # Solve for the wrist rotation
                # We need to account for some random translations between the
                # first and last 3 joints (d4) and also d6,a6,alpha6 in the
                # final frame.

                # Transform of first 3 joints
                T13 = self.A([0, 2], theta)

                # T = T13 * Tz(d4) * R * Tz(d6) Tx(a5)
                Td4 = SE3(0, 0, self.links[3].d)  # Tz(d4)

                # Tz(d6) Tx(a5) Rx(alpha6)
                Tt = SE3(self.links[5].a, 0, self.links[5].d) * SE3.Rx(
                    self.links[5].alpha
                )

                R = Td4.inv() * T13.inv() * Tk * Tt.inv()

                # The spherical wrist implements Euler angles
                if "f" in config:
                    eul = R.eul(flip=True)
                else:
                    eul = R.eul()
                theta = np.r_[theta, eul]
                if self.links[3].alpha > 0:
                    theta[4] = -theta[4]

                # Remove the link offset angles
                theta = theta - self.offset

                solution = iksol(theta, True, "")

            else:
                # ikfunc can return None or a str reason
                if theta is None:
                    solution = iksol(None, False, "")
                else:
                    solution = iksol(None, False, theta)

            solutions.append(solution)

        if len(T) == 1:
            return solutions[0]
        else:
            return iksol(
                np.vstack([sol.q for sol in solutions]),
                np.array([sol.success for sol in solutions]),
                [sol.reason for sol in solutions],
            )

    def config_validate(self, config, allowables):
        """
        Validate a configuration string

        :param config: a configuration string
        :type config: str
        :param allowable: [description]
        :type allowable: tuple of str
        :raises ValueError: bad character in configuration string
        :return: configuration string
        :rtype: str

        For analytic inverse kinematics the Toolbox uses a string whose
        letters indicate particular solutions, eg. for the Puma 560

            =========  ===================
            Character  Meaning
            =========  ===================
            'l'        lefty
            'r'        righty
            'u'        elbow up
            'd'        elbow down
            'n'        wrist not flipped
            'f'        wrist flipped
            =========  ===================

        This method checks that the configuration string is valid and adds
        default values for missing characters.  For example:

            config = self.config_validate(config, ('lr', 'ud', 'nf'))

        indicates the valid characters, and the first character in each
        string is the default, ie. if neither 'l' or 'r' is given then
        'l' will be added to the string.

        """
        for c in config:
            if not any([c in allowable for allowable in allowables]):
                raise ValueError(f"bad config specifier <{c}>")
        for allowable in allowables:
            if all([a not in config for a in allowable]):
                config += allowable[0]
        return config


class SerialLink(DHRobot):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SerialLink is deprecated, use DHRobot instead", DeprecationWarning
        )
        super().__init__(*args, **kwargs)


def _cross(a, b):
    return np.r_[
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    ]


if __name__ == "__main__":  # pragma nocover

    import roboticstoolbox as rtb

    # import spatialmath.base.symbolic as sym

    # planar = rtb.models.DH.Planar2()
    # J = puma.jacob0(puma.qn)
    # print(J)
    # print(puma.manipulability(puma.qn))
    # print(puma.manipulability(puma.qn, 'asada'))
    # tw, T0 = puma.twists(puma.qz)
    # print(planar)

    puma = rtb.models.DH.Puma560()
    print(puma)
    print(puma.jacob0(puma.qn, analytical="eul"))
    # puma.base = None
    # print('base', puma.base)
    # print('tool', puma.tool)

    # print(puma.ets())

    # puma[2].flip = True
    # puma[3].offset = 1
    # puma[4].flip = True
    # puma[4].offset = -1
    # print(puma)
    # print(puma.ets())

    # print(puma.dyntable())
