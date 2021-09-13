# import sys
from abc import ABC, abstractproperty
import copy
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3, SE2
from spatialmath.base.argcheck import (
    isvector,
    getvector,
    getmatrix,
    getunit,
    verifymatrix,
)
from ansitable import ANSITable, Column
from roboticstoolbox.backends.PyPlot import PyPlot
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
from roboticstoolbox.robot.Dynamics import DynamicsMixin
from roboticstoolbox.robot.IK import IKMixin

try:
    from matplotlib import colors
    from matplotlib import cm

    _mpl = True
except ImportError:  # pragma nocover
    pass

_default_backend = None

# TODO maybe this needs to be abstract
# ikine functions need: fkine, jacobe, qlim methods from subclass


class Robot(ABC, DynamicsMixin, IKMixin):

    _color = True

    def __init__(
        self,
        links,
        name="noname",
        manufacturer="",
        comment="",
        base=None,
        tool=None,
        gravity=None,
        keywords=(),
        symbolic=False,
    ):

        self.name = name
        self.manufacturer = manufacturer
        self.comment = comment
        self.symbolic = symbolic
        self.tool = tool
        self._reach = None
        self._base = base

        # if base is None:
        #     self.base = SE3()

        if keywords is not None and not isinstance(keywords, (tuple, list)):
            raise TypeError("keywords must be a list or tuple")
        else:
            self.keywords = keywords

        # gravity is in the negative-z direction.  This is the negative of the
        # MATLAB Toolbox case (which was wrong).
        if gravity is None:
            gravity = np.array([0, 0, -9.81])
        self.gravity = gravity

        # validate the links, must be a list of Link subclass objects
        if not isinstance(links, list):
            raise TypeError("The links must be stored in a list.")

        self._hasdynamics = False
        self._hasgeometry = False
        self._hascollision = False

        for link in links:
            if not isinstance(link, rtb.Link):
                raise TypeError("links should all be Link subclass")

            # add link back to roboto
            link._robot = self

            if link.hasdynamics:
                self._hasdynamics = True
            if link.geometry:
                self._hasgeometry = []
            if link.collision:
                self._hascollision = True

            if isinstance(link, rtb.ELink):
                if len(link.geometry) > 0:
                    self._hasgeometry = True
        self._links = links
        self._nlinks = len(links)

        # Current joint angles of the robot
        self.q = np.zeros(self.n)
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)

        self.control_type = "v"

        self._configdict = {}

        self._dynchanged = False

        # URDF Parser Attempt
        # # Search mesh dir for meshes
        # if urdfdir is not None:
        #     # Parse the URDF to obtain file paths and scales
        #     data = self._get_stl_file_paths_and_scales(urdfdir)
        #     # Obtain the base mesh
        #     self.basemesh = [data[0][0], data[1][0], data[2][0]]
        #     # Save the respective meshes to each link
        #     for idx in range(1, self.n+1):
        #         self._links[idx-1].mesh = [data[0][idx], data[1][idx],
        #         data[2][idx]]
        # else:
        #     self.basemesh = None

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return str(self)

    def __getitem__(self, i):
        """
        Get link (Robot superclass)

        :param i: link number or name
        :type i: int or str
        :return: i'th link or named link
        :rtype: Link subclass

        This also supports iterating over each link in the robot object,
        from the base to the tool.

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> print(robot[1]) # print the 2nd link
            >>> print([link.a for link in robot])  # print all the a_j values

        .. note:: ``ERobot`` supports link lookup by name,
            eg. ``robot['link1']``
        """
        if isinstance(i, str):
            try:
                return self.link_dict[i]
            except KeyError:
                raise KeyError(f"link {i} not in link dictionary")
            except AttributeError:
                raise AttributeError(f"robot has no link dictionary")
        else:
            return self._links[i]

    # URDF Parser Attempt
    # @staticmethod
    # def _get_stl_file_paths_and_scales(urdf_path):
    #     data = [[], [], []]  # [ [filenames] , [scales] , [origins] ]
    #
    #     name, ext = splitext(urdf_path)
    #
    #     if ext == '.xacro':
    #         urdf_string = xacro.main(urdf_path)
    #         urdf = URDF.loadstr(urdf_string, urdf_path)
    #
    #         for link in urdf.links:
    #             data[0].append(link.visuals[0].geometry.mesh.filename)
    #             data[1].append(link.visuals[0].geometry.mesh.scale)
    #             data[2].append(SE3(link.visuals[0].origin))
    #
    #     return data

    def dynchanged(self, what=None):
        """
        Dynamic parameters have changed (Robot superclass)

        Called from a property setter to inform the robot that the cache of
        dynamic parameters is invalid.

        :seealso: :func:`roboticstoolbox.Link._listen_dyn`
        """
        self._dynchanged = True
        if what != "gravity":
            self._hasdynamics = True

    def _getq(self, q=None):
        """
        Get joint coordinates (Robot superclass)

        :param q: passed value, defaults to None
        :type q: array_like, optional
        :return: passed or value from robot state
        :rtype: ndarray(n,)
        """
        if q is None:
            return self.q
        elif isvector(q, self.n):
            return getvector(q, self.n)
        else:
            return getmatrix(q, (None, self.n))

    @property
    def n(self):
        """
        Number of joints (Robot superclass)

        :return: Number of joints
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.n

        :seealso: :func:`nlinks`, :func:`nbranches`
        """
        return self._n

    @property
    def nlinks(self):
        """
        Number of links (Robot superclass)

        :return: Number of links
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.nlinks

        :seealso: :func:`n`, :func:`nbranches`
        """
        return self._nlinks

    @abstractproperty
    def nbranches(self):
        """
        Number of branches (Robot superclass)

        :return: Number of branches
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.nbranches

        :seealso: :func:`n`, :func:`nlinks`
        """
        return self._n

    @property
    def hasdynamics(self):
        """
        Robot has dynamic parameters (Robot superclass)

        :return: Robot has dynamic parameters
        :rtype: bool

        At least one link has associated dynamic parameters.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.hasdynamics:
        """
        return self._hasdynamics

    @property
    def hasgeometry(self):
        """
        Robot has geometry model (Robot superclass)

        :return: Robot has geometry model
        :rtype: bool

        At least one link has associated mesh to describe its shape.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.hasgeometry

        :seealso: :func:`hascollision`
        """
        return self._hasgeometry

    @property
    def hascollision(self):
        """
        Robot has collision model (Robot superclass)

        :return: Robot has collision model
        :rtype: bool

        At least one link has associated collision model.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.hascollision

        :seealso: :func:`hasgeometry`
        """
        return self._hascollision

    @property
    def qrandom(self):
        """
        Return a random joint configuration

        :return: Random joint configuration :rtype: ndarray(n)

        The value for each joint is uniform randomly distributed  between the
        limits set for the robot.

        .. note:: The joint limit for all joints must be set.

        :seealso: :func:`Robot.qlim`, :func:`Link.qlim`
        """
        qlim = self.qlim
        if np.any(np.isnan(qlim)):
            raise ValueError("some joint limits not defined")
        return np.random.uniform(low=qlim[0, :], high=qlim[1, :], size=(self.n,))

    @property
    def default_backend(self):
        """
        Get default graphical backend

        :return: backend name
        :rtype: str

        Get the default graphical backend, used when no explicit backend is
        passed to ``Robot.plot``.
        """
        return _default_backend

    @default_backend.setter
    def default_backend(self, be):
        """
        Set default graphical backend

        :param be: backend name
        :type be: str

        Set the default graphical backend, used when no explicit backend is
        passed to ``Robot.plot``.  The default set here will be overridden if
        the particular ``Robot`` subclass cannot support it.
        """
        _default_backend = be

    def addconfiguration(self, name, q, unit="rad"):
        """
        Add a named joint configuration (Robot superclass)

        :param name: Name of the joint configuration
        :type name: str
        :param q: Joint configuration
        :type q: ndarray(n) or list

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.qz
            >>> robot.addconfiguration("mypos", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            >>> robot.mypos
        """
        v = getvector(q, self.n)
        v = getunit(v, unit)
        self._configdict[name] = v
        setattr(self, name, v)

    def configurations_str(self):
        deg = 180 / np.pi

        # TODO: factor this out of DHRobot
        def angle(theta, fmt=None):

            if fmt is not None:
                try:
                    return fmt.format(theta * deg) + "\u00b0"
                except TypeError:
                    pass

            # pragma nocover
            return str(theta * deg) + "\u00b0"

        # show named configurations
        if len(self._configdict) > 0:
            table = ANSITable(
                Column("name", colalign=">"),
                *[
                    Column(f"q{j:d}", colalign="<", headalign="<")
                    for j in range(self.n)
                ],
                border="thin",
            )

            for name, q in self._configdict.items():
                qlist = []
                for j, c in enumerate(self.structure):
                    if c == "P":
                        qlist.append(f"{q[j]: .3g}")
                    else:
                        qlist.append(angle(q[j], "{: .3g}"))
                table.row(name, *qlist)

            return "\n" + str(table)
        else:  # pragma nocover
            return ""

    @property
    def structure(self):
        """
        Return the joint structure string

        :return: joint configuration string
        :rtype: str

        A string with one letter per joint: ``R`` for a revolute
        joint, and ``P`` for a prismatic joint.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.structure
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.structure

        .. note:: Fixed joints, that maintain a constant link relative pose,
            are not included.  ``len(self.structure) == self.n``.
        """
        structure = []
        for link in self:
            if link.isrevolute:
                structure.append("R")
            elif link.isprismatic:
                structure.append("P")

        return "".join(structure)

    @property
    def revolutejoints(self):
        """
        Revolute joints as bool array

        :return: array of joint type, True if revolute
        :rtype: bool(n)

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.revolutejoints()
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.revolutejoints()

        .. note:: Fixed joints, that maintain a constant link relative pose,
            are not included.  ``len(self.structure) == self.n``.

        :seealso: :func:`Link.isrevolute`, :func:`prismaticjoints`
        """
        return [link.isrevolute for link in self if link.isjoint]

    # TODO not very efficient
    # TODO keep a mapping from joint to link
    def isrevolute(self, j):
        return self.revolutejoints[j]

    def isprismatic(self, j):
        return self.prismaticjoints[j]

    @property
    def prismaticjoints(self):
        """
        Revolute joints as bool array

        :return: array of joint type, True if prismatic
        :rtype: bool(n)

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> puma.prismaticjoints()
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.prismaticjoints()

        .. note:: Fixed joints, that maintain a constant link relative pose,
            are not included.  ``len(self.structure) == self.n``.

        :seealso: :func:`Link.isprismatic`, :func:`revolutejoints`
        """
        return [link.isprismatic for link in self if link.isjoint]

    def todegrees(self, q):
        """
        Convert joint angles to degrees

        :param q: The joint configuration of the robot
        :type q: ndarray(n) or ndarray(m,n)
        :return: a vector of joint coordinates in degrees and metres
        :rtype: ndarray(n)  or ndarray(m,n)

        ``robot.todegrees(q)`` converts joint coordinates ``q`` to degrees
        taking into account whether elements of ``q`` correspond to revolute
        or prismatic joints, ie. prismatic joint values are not converted.

        If ``q`` is a matrix, with one column per joint, the conversion is
        performed columnwise.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> from math import pi
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.todegrees([pi/4, pi/8, 2, -pi/4, pi/6, pi/3])
        """

        q = getmatrix(q, (None, self.n))

        for j, revolute in enumerate(self.revolutejoints):
            if revolute:
                q[:, j] *= 180.0 / np.pi
        if q.shape[0] == 1:
            return q[0]
        else:
            return q

    def toradians(self, q):
        """
        Convert joint angles to radians

        :param q: The joint configuration of the robot
        :type q: ndarray(n)  or ndarray(m,n)
        :return: a vector of joint coordinates in radians and metres
        :rtype: ndarray(n)  or ndarray(m,n)

        ``robot.toradians(q)`` converts joint coordinates ``q`` to radians
        taking into account whether elements of ``q`` correspond to revolute
        or prismatic joints, ie. prismatic joint values are not converted.

        If ``q`` is a matrix, with one column per joint, the conversion is
        performed columnwise.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> stanford = rtb.models.DH.Stanford()
            >>> stanford.toradians([10, 20, 2, 30, 40, 50])
        """

        q = getmatrix(q, (None, self.n))

        for j, revolute in enumerate(self.revolutejoints):
            if revolute:
                q[:, j] *= np.pi / 180.0
        if q.shape[0] == 1:
            return q[0]
        else:
            return q

    def linkcolormap(self, linkcolors="viridis"):
        """
        Create a colormap for robot joints

        :param linkcolors: list of colors or colormap, defaults to "viridis"
        :type linkcolors: list or str, optional
        :return: color map
        :rtype: matplotlib.colors.ListedColormap

        - ``cm = robot.linkcolormap()`` is an n-element colormap that gives a
          unique color for every link.  The RGBA colors for link ``j`` are
          ``cm(j)``.
        - ``cm = robot.linkcolormap(cmap)`` as above but ``cmap`` is the name
          of a valid matplotlib colormap.  The default, example above, is the
          ``viridis`` colormap.
        - ``cm = robot.linkcolormap(list of colors)`` as above but a
          colormap is created from a list of n color names given as strings,
          tuples or hexstrings.

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> cm = robot.linkcolormap("inferno")
            >>> print(cm(range(6))) # cm(i) is 3rd color in colormap
            >>> cm = robot.linkcolormap(
            >>>     ['red', 'g', (0,0.5,0), '#0f8040', 'yellow', 'cyan'])
            >>> print(cm(range(6)))

        .. note::

            - Colormaps have 4-elements: red, green, blue, alpha (RGBA)
            - Names of supported colors and colormaps are defined in the
              matplotlib documentation.

                - `Specifying colors
                <https://matplotlib.org/3.1.0/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py>`_
                - `Colormaps
                <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py>`_
        """  # noqa

        if isinstance(linkcolors, list) and len(linkcolors) == self.n:
            # provided a list of color names
            return colors.ListedColormap(linkcolors)
        else:
            # assume it is a colormap name
            return cm.get_cmap(linkcolors, 6)

    def jtraj(self, T1, T2, t, **kwargs):
        """
        Joint-space trajectory between SE(3) poses

        :param T1: initial end-effector pose
        :type T1: SE3 instance
        :param T2: final end-effector pose
        :type T2: SE3 instance
        :param t: time vector or number of steps
        :type t: ndarray(m) or int
        :param kwargs: arguments passed to the IK solver
        :return: trajectory
        :rtype: Trajectory instance

        ``traj = obot.jtraj(T1, T2, t)`` is a trajectory object whose
        attribute ``traj.q`` is a row-wise joint-space trajectory.

        The initial and final poses are mapped to joint space using inverse
        kinematics:

        - if the object has an analytic solution ``ikine_a`` that will be used,
        - otherwise the general numerical algorithm ``ikine_min`` will be used.


        """

        if hasattr(self, "ikine_a"):
            ik = self.ikine_a
        else:
            ik = self.ikine_min

        q1 = ik(T1, **kwargs)
        q2 = ik(T2, **kwargs)

        return rtb.jtraj(q1.q, q2.q, t)

    def manipulability(self, q=None, J=None, method="yoshikawa", axes="all", **kwargs):
        """
        Manipulability measure

        :param q: Joint coordinates, one of J or q required
        :type q: ndarray(n), or ndarray(m,n)
        :param J: Jacobian in world frame if already computed, one of J or
            q required
        :type J: ndarray(6,n)
        :param method: method to use, "yoshikawa" (default), "condition",
            "minsingular"  or "asada"
        :type method: str
        :param axes: Task space axes to consider: "all" [default],
            "trans", "rot" or "both"
        :type axes: str
        :param kwargs: extra arguments to pass to ``jacob0``
        :return: manipulability
        :rtype: float or ndarray(m)

        - ``manipulability(q)`` is the scalar manipulability index
          for the robot at the joint configuration ``q``.  It indicates
          dexterity, that is, how well conditioned the robot is for motion
          with respect to the 6 degrees of Cartesian motion.  The values is
          zero if the robot is at a singularity.

        Various measures are supported:

        +-------------------+-------------------------------------------------+
        | Measure           |       Description                               |
        +-------------------+-------------------------------------------------+
        | ``"yoshikawa"``   | Volume of the velocity ellipsoid, *distance*    |
        |                   | from singularity [Yoshikawa85]_                 |
        +-------------------+-------------------------------------------------+
        | ``"invcondition"``| Inverse condition number of Jacobian, isotropy  |
        |                   | of the velocity ellipsoid [Klein87]_            |
        +-------------------+-------------------------------------------------+
        | ``"minsingular"`` | Minimum singular value of the Jacobian,         |
        |                   | *distance*  from singularity [Klein87]_         |
        +-------------------+-------------------------------------------------+
        | ``"asada"``       | Isotropy of the task-space acceleration         |
        |                   | ellipsoid which is a function of the Cartesian  |
        |                   | inertia matrix which depends on the inertial    |
        |                   | parameters [Asada83]_                           |
        +-------------------+-------------------------------------------------+

        **Trajectory operation**:

        If ``q`` is a matrix (m,n) then the result (m,) is a vector of
        manipulability indices for each joint configuration specified by a row
        of ``q``.

        .. note::

            - Invokes the ``jacob0`` method of the robot if ``J`` is not passed
            - The "all" option includes rotational and translational
              dexterity, but this involves adding different units. It can be
              more useful to look at the translational and rotational
              manipulability separately.
            - Examples in the RVC book (1st edition) can be replicated by
              using the "all" option
            - Asada's measure requires inertial a robot model with inertial
              parameters.

        :references:

        .. [Yoshikawa85] Manipulability of Robotic Mechanisms. Yoshikawa T.,
                The International Journal of Robotics Research.
                1985;4(2):3-9. doi:10.1177/027836498500400201
        .. [Asada83] A geometrical representation of manipulator dynamics and
                its application to arm design, H. Asada,
                Journal of Dynamic Systems, Measurement, and Control,
                vol. 105, p. 131, 1983.
        .. [Klein87] Dexterity Measures for the Design and Control of
                Kinematically Redundant Manipulators. Klein CA, Blaho BE.
                The International Journal of Robotics Research.
                1987;6(2):72-83. doi:10.1177/027836498700600206

        - Robotics, Vision & Control, Chap 8, P. Corke, Springer 2011.

        """
        if isinstance(axes, list) and len(axes) == 6:
            pass
        elif axes == "all":
            axes = [True, True, True, True, True, True]
        elif axes.startswith("trans"):
            axes = [True, True, True, False, False, False]
        elif axes.startswith("rot"):
            axes = [False, False, False, True, True, True]
        elif axes == "both":
            return (
                self.manipulability(q, J, method, axes="trans", **kwargs),
                self.manipulability(q, J, method, axes="rot", **kwargs),
            )
        else:
            raise ValueError("axes must be all, trans, rot or both")

        def yoshikawa(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            if J.shape[0] == J.shape[1]:
                # simplified case for square matrix
                return abs(np.linalg.det(J))
            else:
                m2 = np.linalg.det(J @ J.T)
                return np.sqrt(abs(m2))

        def condition(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            return 1 / np.linalg.cond(J)  # return 1/cond(J)

        def minsingular(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            s = np.linalg.svd(J, compute_uv=False)
            return s[-1]  # return last/smallest singular value of J

        def asada(robot, J, q, axes, **kwargs):
            # dof = np.sum(axes)
            if np.linalg.matrix_rank(J) < 6:
                return 0
            Ji = np.linalg.pinv(J)
            Mx = Ji.T @ robot.inertia(q) @ Ji
            d = np.where(axes)[0]
            Mx = Mx[d]
            Mx = Mx[:, d.tolist()]
            e, _ = np.linalg.eig(Mx)
            return np.min(e) / np.max(e)

        # choose the handler function
        if method == "yoshikawa":
            mfunc = yoshikawa
        elif method == "invcondition":
            mfunc = condition
        elif method == "minsingular":
            mfunc = minsingular
        elif method == "asada":
            mfunc = asada
        else:
            raise ValueError("Invalid method chosen")

        # Calculate manipulability based on supplied Jacobian
        if J is not None:
            w = [mfunc(self, J, q, axes)]

        # Otherwise use the q vector/matrix
        else:
            q = getmatrix(q, (None, self.n))
            w = np.zeros(q.shape[0])

            for k, qk in enumerate(q):
                Jk = self.jacob0(qk, **kwargs)
                w[k] = mfunc(self, Jk, qk, axes)

        if len(w) == 1:
            return w[0]
        else:
            return w

    def jacob_dot(self, q=None, qd=None, J0=None):
        r"""
        Derivative of Jacobian

        :param q: The joint configuration of the robot
        :type q: float ndarray(n)
        :param qd: The joint velocity of the robot
        :type qd: ndarray(n)
        :param J0: Jacobian in {0} frame
        :type J0: ndarray(6,n)
        :return: The derivative of the manipulator Jacobian
        :rtype:  ndarray(6,n)

        ``robot.jacob_dot(q, qd)`` computes the rate of change of the
        Jacobian elements.  If ``J0`` is already calculated for the joint
        coordinates ``q`` it can be passed in to to save computation time.

        It is computed as the mode-3 product of the Hessian tensor and the
        velocity vector.

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke

        :seealso: :func:`jacob0`, :func:`hessian0`
        """  # noqa
        n = len(q)
        if J0 is None:
            J0 = self.jacob0(q)
        H = self.hessian0(q, J0)

        # Jd = H qd using mode 3 product
        Jd = np.zeros((6, n))
        for i in range(n):
            Jd += H[:, :, i] * qd[i]

        return Jd

    def jacobm(self, q=None, J=None, H=None, end=None, start=None, axes="all"):
        r"""
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
        :param end: the final link or Gripper which the Hessian represents
        :type end: str or ELink or Gripper
        :param start: the first link which the Hessian represents
        :type start: str or ELink

        :return: The manipulability Jacobian
        :rtype: float ndarray(n)

        Yoshikawa's manipulability measure

        .. math::

            m(\vec{q}) = \sqrt{\mat{J}(\vec{q}) \mat{J}(\vec{q})^T}

        This method returns its Jacobian with respect to configuration

        .. math::

            \frac{\partial m(\vec{q})}{\partial \vec{q}}

        :references:
            - Kinematic Derivatives using the Elementary Transform
              Sequence, J. Haviland and P. Corke
        """

        end, start, _ = self._get_limit_links(end, start)
        # path, n, _ = self.get_path(end, start)

        if axes == "all":
            axes = [True, True, True, True, True, True]
        elif axes.startswith("trans"):
            axes = [True, True, True, False, False, False]
        elif axes.startswith("rot"):
            axes = [False, False, False, True, True, True]
        else:
            raise ValueError("axes must be all, trans or rot")

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, start=start, end=end)
        else:
            verifymatrix(J, (6, self.n))

        n = J.shape[1]

        if H is None:
            H = self.hessian0(J0=J, start=start, end=end)
        else:
            verifymatrix(H, (6, self.n, self.n))

        manipulability = self.manipulability(q, J=J, start=start, end=end, axes=axes)

        J = J[axes, :]
        H = H[axes, :, :]

        b = np.linalg.inv(J @ np.transpose(J))
        Jm = np.zeros((n, 1))

        for i in range(n):
            c = J @ np.transpose(H[:, :, i])
            Jm[i, 0] = manipulability * np.transpose(c.flatten("F")) @ b.flatten("F")

        return Jm

    # --------------------------------------------------------------------- #

    @property
    def name(self):
        """
        Get/set robot name (Robot superclass)

        - ``robot.name`` is the robot name

        :return: robot name
        :rtype: str

        - ``robot.name = ...`` checks and sets therobot name
        """
        return self._name

    @name.setter
    def name(self, name_new):
        self._name = name_new

    # --------------------------------------------------------------------- #

    @property
    def manufacturer(self):
        """
        Get/set robot manufacturer's name (Robot superclass)

        - ``robot.manufacturer`` is the robot manufacturer's name

        :return: robot manufacturer's name
        :rtype: str

        - ``robot.manufacturer = ...`` checks and sets the manufacturer's name
        """
        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, manufacturer_new):
        self._manufacturer = manufacturer_new

    # --------------------------------------------------------------------- #

    @property
    def links(self):
        """
        Robot links (Robot superclass)

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
    def base(self):
        """
        Get/set robot base transform (Robot superclass)

        - ``robot.base`` is the robot base transform

        :return: robot tool transform
        :rtype: SE3 instance

        - ``robot.base = ...`` checks and sets the robot base transform

        .. note:: The private attribute ``_base`` will be None in the case of
            no base transform, but this property will return ``SE3()`` which
            is an identity matrix.
        """
        if self._base is None:
            if isinstance(self, rtb.ERobot2):
                self._base = SE2()
            else:
                self._base = SE3()

        # return a copy, otherwise somebody with
        # reference to the base can change it
        return self._base.copy()

    @base.setter
    def base(self, T):
        # if not isinstance(T, SE3):
        #     T = SE3(T)
        if T is None:
            self._base = T
        elif isinstance(self, rtb.ERobot2):
            # 2D robot
            if isinstance(T, SE2):
                self._base = T
            elif SE2.isvalid(T):
                self._tool = SE2(T, check=True)
        elif isinstance(self, rtb.Robot):
            # all other 3D robots
            if isinstance(T, SE3):
                self._base = T
            elif SE3.isvalid(T):
                self._tool = SE3(T, check=True)

        else:
            raise ValueError("base must be set to None (no tool), SE2, or SE3")

    # --------------------------------------------------------------------- #

    @property
    def tool(self):
        """
        Get/set robot tool transform (Robot superclass)

        - ``robot.tool`` is the robot name

        :return: robot tool transform
        :rtype: SE3 instance

        - ``robot.tool = ...`` checks and sets the robot tool transform

        .. note:: The private attribute ``_tool`` will be None in the case of
            no tool transform, but this property will return ``SE3()`` which
            is an identity matrix.
        """
        if self._tool is None:
            return SE3()
        else:
            return self._tool

    @tool.setter
    def tool(self, T):
        # if not isinstance(T, SE3):
        #     T = SE3(T)
        # this is allowed to be none, it's helpful for symbolics rather than
        # having an identity matrix
        if T is None or isinstance(T, SE3):
            self._tool = T
        elif SE3.isvalid(T):
            self._tool = SE3(T, check=False)
        else:
            raise ValueError("tool must be set to None (no tool) or an SE3")

    @property
    def qlim(self):
        r"""
        Joint limits (Robot superclass)

        :return: Array of joint limit values
        :rtype: ndarray(2,n)
        :exception ValueError: unset limits for a prismatic joint

        Limits are extracted from the link objects.  If joints limits are
        not set for:

            - a revolute joint [-𝜋. 𝜋] is returned
            - a prismatic joint an exception is raised

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.qlim
        """
        # TODO tidy up
        limits = np.zeros((2, self.n))
        j = 0
        for link in self:
            if link.isrevolute:
                if link.qlim is None:
                    v = np.r_[-np.pi, np.pi]
                else:
                    v = link.qlim
            elif link.isprismatic:
                if link.qlim is None:
                    raise ValueError("undefined prismatic joint limit")
                else:
                    v = link.qlim
            else:
                # fixed link
                continue

            limits[:, j] = v
            j += 1
        return limits

    # TODO, the remaining functions, I have only a hazy understanding
    # of how they work
    # --------------------------------------------------------------------- #

    @property
    def q(self):
        """
        Get/set robot joint configuration (Robot superclass)

        - ``robot.q`` is the robot joint configuration

        :return: robot joint configuration
        :rtype: ndarray(n,)

        - ``robot.q = ...`` checks and sets the joint configuration

        .. note::  ???
        """
        return self._q

    @q.setter
    def q(self, q_new):
        self._q = getvector(q_new, self.n)

    # --------------------------------------------------------------------- #

    @property
    def qd(self):
        """
        Get/set robot joint velocity (Robot superclass)

        - ``robot.qd`` is the robot joint velocity

        :return: robot joint velocity
        :rtype: ndarray(n,)

        - ``robot.qd = ...`` checks and sets the joint velocity

        .. note::  ???
        """
        return self._qd

    @qd.setter
    def qd(self, qd_new):
        self._qd = getvector(qd_new, self.n)

    # --------------------------------------------------------------------- #

    @property
    def qdd(self):
        """
        Get/set robot joint acceleration (Robot superclass)

        - ``robot.qdd`` is the robot joint acceleration

        :return: robot joint acceleration
        :rtype: ndarray(n,)

        - ``robot.qdd = ...`` checks and sets the robot joint acceleration

        .. note::  ???
        """
        return self._qdd

    @qdd.setter
    def qdd(self, qdd_new):
        self._qdd = getvector(qdd_new, self.n)

    # --------------------------------------------------------------------- #

    # TODO could we change this to control_mode ?
    @property
    def control_type(self):
        """
        Get/set robot control mode (Robot superclass)

        - ``robot.control_type`` is the robot control mode

        :return: robot control mode
        :rtype: ndarray(n,)

        - ``robot.control_type = ...`` checks and sets the robot control mode

        .. note::  ???
        """
        return self._control_type

    @control_type.setter
    def control_type(self, cn):
        if cn == "p" or cn == "v" or cn == "a":
            self._control_type = cn
        else:
            raise ValueError("Control type must be one of 'p', 'v', or 'a'")

    # --------------------------------------------------------------------- #

    # TODO probably should be a static method
    def _get_graphical_backend(self, backend=None):

        default = self.default_backend

        # figure out the right default
        if backend is None:
            if isinstance(self, rtb.DHRobot):
                default = "pyplot"
            elif isinstance(self, rtb.ERobot2):
                default = "pyplot2"
            elif isinstance(self, rtb.ERobot):
                if self.hasgeometry:
                    default = "swift"
                else:
                    default = "pyplot"

        if backend is not None:
            backend = backend.lower()

        # find the right backend, modules are imported here on an as needs
        # basis
        if backend == "swift" or default == "swift":  # pragma nocover
            # swift was requested, is it installed?
            if isinstance(self, rtb.DHRobot):
                raise NotImplementedError(
                    "Plotting in Swift is not implemented for DHRobots yet"
                )
            try:
                # yes, use it
                from roboticstoolbox.backends.swift import Swift

                env = Swift()
                return env
            except ModuleNotFoundError:
                if backend == "swift":
                    print("Swift is not installed, " "install it using pip or conda")
                backend = "pyplot"

        elif backend == "vpython" or default == "vpython":  # pragma nocover
            # vpython was requested, is it installed?
            if not isinstance(self, rtb.DHRobot):
                raise NotImplementedError(
                    "Plotting in VPython is only implemented for DHRobots"
                )
            try:
                # yes, use it
                from roboticstoolbox.backends.VPython import VPython

                env = VPython()
                return env
            except ModuleNotFoundError:
                if backend == "vpython":
                    print("VPython is not installed, " "install it using pip or conda")
                backend = "pyplot"
        if backend is None:
            backend = default

        if backend == "pyplot":
            from roboticstoolbox.backends.PyPlot import PyPlot

            env = PyPlot()

        elif backend == "pyplot2":
            from roboticstoolbox.backends.PyPlot import PyPlot2

            env = PyPlot2()

        else:
            raise ValueError("unknown backend", backend)

        return env

    def plot(
        self,
        q,
        backend=None,
        block=False,
        dt=0.050,
        limits=None,
        vellipse=False,
        fellipse=False,
        fig=None,
        movie=None,
        loop=False,
        **kwargs,
    ):
        """
        Graphical display and animation

        :param q: The joint configuration of the robot.
        :type q: float ndarray(n)
        :param backend: The graphical backend to use, currently 'swift'
            and 'pyplot' are implemented. Defaults to 'swift' of an ``ERobot``
            and 'pyplot` for a ``DHRobot``
        :type backend: string
        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param dt: if q is a trajectory, this describes the delay in
            seconds between frames
        :type dt: float
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
            (this option is for 'pyplot' only)
        :type limits: ndarray(6)
        :param vellipse: (Plot Option) Plot the velocity ellipse at the
            end-effector (this option is for 'pyplot' only)
        :type vellipse: bool
        :param vellipse: (Plot Option) Plot the force ellipse at the
            end-effector (this option is for 'pyplot' only)
        :type vellipse: bool
        :param jointaxes: (Plot Option) Plot an arrow indicating the axes in
            which the joint revolves around(revolute joint) or translates
            along (prosmatic joint) (this option is for 'pyplot' only)
        :type jointaxes: bool
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
            (this option is for 'pyplot' only)
        :type eeframe: bool
        :param shadow: (Plot Option) Plot a shadow of the robot in the x-y
            plane. (this option is for 'pyplot' only)
        :type shadow: bool
        :param name: (Plot Option) Plot the name of the robot near its base
            (this option is for 'pyplot' only)
        :type name: bool
        :param movie: name of file in which to save an animated GIF
            (this option is for 'pyplot' only)
        :type movie: str

        :return: A reference to the environment object which controls the
            figure
        :rtype: Swift or PyPlot

        - ``robot.plot(q, 'pyplot')`` displays a graphical view of a robot
          based on the kinematic model and the joint configuration ``q``.
          This is a stick figure polyline which joins the origins of the
          link coordinate frames. The plot will autoscale with an aspect
          ratio of 1.

        If ``q`` (m,n) representing a joint-space trajectory it will create an
        animation with a pause of ``dt`` seconds between each frame.

        .. note::
            - By default this method will block until the figure is dismissed.
              To avoid this set ``block=False``.
            - For PyPlot, the polyline joins the origins of the link frames,
              but for some Denavit-Hartenberg models those frames may not
              actually be on the robot, ie. the lines to not neccessarily
              represent the links of the robot.

        :seealso: :func:`teach`
        """

        env = None

        env = self._get_graphical_backend(backend)

        q = getmatrix(q, (None, self.n))
        self.q = q[0, :]

        # Add the self to the figure in readonly mode
        # Add the self to the figure in readonly mode
        if q.shape[0] == 1:
            env.launch(self.name + " Plot", limits=limits, fig=fig)
        else:
            env.launch(self.name + " Trajectory Plot", limits=limits, fig=fig)

        env.add(self, readonly=True, **kwargs)

        if vellipse:
            vell = self.vellipse(centre="ee")
            env.add(vell)

        if fellipse:
            fell = self.fellipse(centre="ee")
            env.add(fell)

        # Stop lint error
        images = []  # list of images saved from each plot

        if movie is not None:
            loop = False

        while True:
            for qk in q:
                self.q = qk
                if vellipse:
                    vell.q = qk
                if fellipse:
                    fell.q = qk
                env.step(dt)

                if movie is not None:  # pragma nocover
                    images.append(env.getframe())

            if movie is not None:  # pragma nocover
                # save it as an animated GIF
                images[0].save(
                    movie,
                    save_all=True,
                    append_images=images[1:],
                    optimize=False,
                    duration=dt,
                    loop=0,
                )
            if not loop:
                break

        # Keep the plot open
        if block:  # pragma: no cover
            env.hold()

        return env

    # --------------------------------------------------------------------- #

    def fellipse(self, q=None, opt="trans", unit="rad", centre=[0, 0, 0]):
        """
        Create a force ellipsoid object for plotting with PyPlot

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

        """
        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError("ERobot fellipse not implemented yet")

        q = getunit(q, unit)
        ell = EllipsePlot(self, q, "f", opt, centre=centre)
        return ell

    def vellipse(self, q=None, opt="trans", unit="rad", centre=[0, 0, 0], scale=0.1):
        """
        Create a velocity ellipsoid object for plotting with PyPlot

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
        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError("ERobot vellipse not implemented yet")

        q = getunit(q, unit)
        ell = EllipsePlot(self, q, "v", opt, centre=centre, scale=scale)
        return ell

    def plot_ellipse(
        self,
        ellipse,
        block=True,
        limits=None,
        jointaxes=True,
        eeframe=True,
        shadow=True,
        name=True,
    ):
        """
        Plot the an ellipsoid

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param ellipse: the ellipsoid to plot
        :type ellipse: EllipsePlot
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

        - ``robot.plot_ellipse(ellipsoid)`` displays the ellipsoid.

        .. note::
            - By default the ellipsoid is drawn at the origin.  The option
              ``centre`` allows its origin to set to set to the specified
              3-vector, or the string "ee" ensures it is drawn at the
              end-effector position.
        """

        if not isinstance(ellipse, EllipsePlot):  # pragma nocover
            raise TypeError(
                "ellipse must be of type " "roboticstoolbox.backend.PyPlot.EllipsePlot"
            )

        env = PyPlot()

        # Add the robot to the figure in readonly mode
        env.launch(ellipse.robot.name + " " + ellipse.name, limits=limits)

        env.add(ellipse, jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

        # Keep the plot open
        if block:  # pragma: no cover
            env.hold()

        return env

    def plot_fellipse(
        self,
        q=None,
        block=True,
        fellipse=None,
        limits=None,
        opt="trans",
        centre=[0, 0, 0],
        jointaxes=True,
        eeframe=True,
        shadow=True,
        name=True,
    ):
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
          robot at pose ``q``. The plot will autoscale with an aspect ratio
          of 1.

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

        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "Ellipse Plotting of ERobot's not implemented yet"
            )

        if q is not None:
            self.q = q

        if fellipse is None:
            fellipse = self.fellipse(q=q, opt=opt, centre=centre)

        return self.plot_ellipse(
            fellipse,
            block=block,
            limits=limits,
            jointaxes=jointaxes,
            eeframe=eeframe,
            shadow=shadow,
            name=name,
        )

    def plot_vellipse(
        self,
        q=None,
        block=True,
        vellipse=None,
        limits=None,
        opt="trans",
        centre=[0, 0, 0],
        jointaxes=True,
        eeframe=True,
        shadow=True,
        name=True,
    ):
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
          robot at pose ``q``. The plot will autoscale with an aspect ratio
          of 1.

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

        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "Ellipse Plotting of ERobot's not implemented yet"
            )

        if q is not None:
            self.q = q

        if vellipse is None:
            vellipse = self.vellipse(q=q, opt=opt, centre=centre)

        return self.plot_ellipse(
            vellipse,
            block=block,
            limits=limits,
            jointaxes=jointaxes,
            eeframe=eeframe,
            shadow=shadow,
            name=name,
        )

    # --------------------------------------------------------------------- #

    def teach(
        self,
        q=None,
        block=True,
        order="xyz",
        limits=None,
        jointaxes=True,
        jointlabels=False,
        vellipse=False,
        fellipse=False,
        eeframe=True,
        shadow=True,
        name=True,
        backend=None,
    ):
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

        - ``robot.teach()`` as above except the robot's stored value of ``q``
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
        """

        if q is None:
            q = np.zeros((self.n,))
        else:
            q = getvector(q, self.n)

        # Make an empty 3D figure
        env = self._get_graphical_backend(backend)

        # Add the self to the figure in readonly mode
        env.launch("Teach " + self.name, limits=limits)
        env.add(
            self,
            readonly=True,
            jointaxes=jointaxes,
            jointlabels=jointlabels,
            eeframe=eeframe,
            shadow=shadow,
            name=name,
        )

        env._add_teach_panel(self, q)

        if vellipse:
            vell = self.vellipse(centre="ee", scale=0.5)
            env.add(vell)

        if fellipse:
            fell = self.fellipse(centre="ee")
            env.add(fell)

        # Keep the plot open
        if block:  # pragma: no cover
            env.hold()

        return env

    # --------------------------------------------------------------------- #

    def closest_point(self, q, shape, inf_dist=1.0, skip=False):
        """
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
        :param skip: Skip setting all shape transforms based on q, use this
            option if using this method in conjuction with Swift to save time
        :type skip: boolean
        :returns: d, p1, p2 where d is the distance between the shapes,
            p1 and p2 are the points in the world frame on the respective
            shapes. The points returned are [x, y, z].
        :rtype: float, ndarray(1x3), ndarray(1x3)
        """

        if not skip:
            self._set_link_fk(q)

        d = 10000
        p1 = None
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

    def collided(self, q, shape, skip=False):
        """
        collided(shape) checks if this robot and shape have collided
        :param shape: The shape to compare distance to
        :type shape: Shape
        :param skip: Skip setting all shape transforms based on q, use this
            option if using this method in conjuction with Swift to save time
        :type skip: boolean
        :returns: True if shapes have collided
        :rtype: bool
        """

        if not skip:
            self._set_link_fk(q)

        for link in self.links:
            if link.collided(shape):
                return True

        if isinstance(self, rtb.ERobot):
            for gripper in self.grippers:
                for link in gripper.links:
                    if link.collided(shape):
                        return True

        return False

    def joint_velocity_damper(self, ps=0.05, pi=0.1, n=None, gain=1.0):
        """
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
        """

        if n is None:
            n = self.n

        Ain = np.zeros((n, n))
        Bin = np.zeros(n)

        for i in range(n):
            if self.q[i] - self.qlim[0, i] <= pi:
                Bin[i] = -gain * (((self.qlim[0, i] - self.q[i]) + ps) / (pi - ps))
                Ain[i, i] = -1
            if self.qlim[1, i] - self.q[i] <= pi:
                Bin[i] = gain * ((self.qlim[1, i] - self.q[i]) - ps) / (pi - ps)
                Ain[i, i] = 1

        return Ain, Bin


if __name__ == "__main__":

    from roboticstoolbox import ETS2 as ET

    e = ET.r() * ET.tx(1) * ET.r() * ET.tx(1)
    # print(e)
    # r = Robot2(e)

    # print(r.fkine([0, 0]))
    # print(r.jacob0([0, 0]))

    # r.plot([0.7, 0.7])
