import sys
import copy
import numpy as np
# import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath.base.argcheck import isvector, getvector, getmatrix, \
    verifymatrix, getunit
from roboticstoolbox.robot.Link import Link
from spatialmath.base.transforms3d import tr2delta
# from roboticstoolbox.tools import urdf
# from roboticstoolbox.tools import xacro
from pathlib import PurePath, PurePosixPath
from scipy.optimize import minimize, Bounds, LinearConstraint
from matplotlib import colors
from matplotlib import cm
from roboticstoolbox.tools.null import null
from ansitable import ANSITable, Column

# TODO maybe this needs to be abstract
# ikine functions need: fkine, jacobe, qlim methods from subclass


class Robot:

    _color = True

    def __init__(
            self,
            links,
            name='noname',
            manufacturer='',
            comment='',
            base=None,
            tool=None,
            gravity=None,
            meshdir=None,
            keywords=(),
            symbolic=False):

        self.name = name
        self.manufacturer = manufacturer
        self.comment = comment
        self.symbolic = symbolic
        self.base = base
        self.tool = tool
        self.basemesh = None
        if keywords is not None and not isinstance(keywords, (tuple, list)):
            raise TypeError('keywords must be a list or tuple')
        else:
            self.keywords = keywords

        if gravity is None:
            gravity = np.array([0, 0, 9.81])
        self.gravity = gravity

        # validate the links, must be a list of Link subclass objects
        if not isinstance(links, list):
            raise TypeError('The links must be stored in a list.')
        for link in links:
            if not isinstance(link, Link):
                raise TypeError('links should all be Link subclass')
            link._robot = self
        self._links = links

        # Current joint angles of the robot
        self.q = np.zeros(self.n)
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)

        self.control_type = 'v'

        self._configdict = {}

        self._dynchanged = True

        # this probably should go down to DHRobot
        if meshdir is not None:
            classpath = sys.modules[self.__module__].__file__
            self.meshdir = PurePath(classpath).parent / PurePosixPath(meshdir)
            self.basemesh = self.meshdir / "link0.stl"
            for j, link in enumerate(self._links, start=1):
                link.mesh = self.meshdir / "link{:d}.stl".format(j)

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

    def __getitem__(self, i):
        """
        Get link (Robot superclass)

        :param i: link number
        :type i: int
        :return: i'th link of robot
        :rtype: Link subclass

        This also supports iterating over each link in the robot object,
        from the base to the tool.

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> print(robot[1]) # print the 2nd link
            >>> print([link.a for link in robot])  # print all the a_j values

        """
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

    def dynchanged(self):
        """
        Dynamic parameters have changed (Robot superclass)

        Called from a property setter to inform the robot that the cache of
        dynamic parameters is invalid.

        :seealso: :func:`roboticstoolbox.Link._listen_dyn`
        """
        self._dynchanged = True

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

        """
        return self._n

    def addconfiguration(self, name, q, unit='rad'):
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
                return fmt.format(theta * deg) + "\u00b0"
            else:
                return str(theta * deg) + "\u00b0"

        config = self.config()
        # show named configurations
        if len(self._configdict) > 0:
            table = ANSITable(
                Column("name", colalign=">"),
                *[Column(f"q{j:d}", colalign="<", headalign="<") for j in range(self.n)],
                border="thin")

            for name, q in self._configdict.items():
                qlist = []
                for i, c in enumerate(config):
                    if c == 'P':
                        qlist.append(f"{q[i]: .3g}")
                    else:
                        qlist.append(angle(q[i], "{: .3g}"))
                table.row(name, *qlist)

            return "\n" + str(table)
        else:
            return ""

    def dyntable(self):
        """
        Pretty print the dynamic parameters (Robot superclass)

        The dynamic parameters are printed in a table, with one row per link.

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.links[2].dyntable()

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.dyntable()
        """
        table = ANSITable(
            Column("j", colalign=">", headalign="^"),
            Column("m", colalign="<", headalign="^"),
            Column("r", colalign="<", headalign="^"),
            Column("I", colalign="<", headalign="^"),
            Column("Jm", colalign="<", headalign="^"),
            Column("B", colalign="<", headalign="^"),
            Column("Tc", colalign="<", headalign="^"),
            Column("G", colalign="<", headalign="^"), border="thin")

        for j, link in enumerate(self):
            table.row(link.name, *link._dyn2list())
        return str(table)



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
            >>> cm = robot.linkcolormap(['red', 'g', (0,0.5,0), '#0f8040', 'yellow', 'cyan'])
            >>> print(cm(range(6)))

        .. note::
            
            - Colormaps have 4-elements: red, green, blue, alpha (RGBA)
            - Names of supported colors and colormaps are defined in the matplotlib
              documentation.

                - `Specifying colors <https://matplotlib.org/3.1.0/tutorials/colors/colors.html#sphx-glr-tutorials-colors-colors-py>`_
                - `Colormaps <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py>`_
        """

        if isinstance(linkcolors, list) and len(linkcolors) == self.n:
            # provided a list of color names
            return colors.ListedColormap(linkcolors)
        else:
            # assume it is a colormap name
            return cm.get_cmap(linkcolors, 6)
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
            return SE3()
        else:
            return self._base

    @base.setter
    def base(self, T):
        # if not isinstance(T, SE3):
        #     T = SE3(T)
        if T is None or isinstance(T, SE3):
            self._base = T
        elif SE3.isvalid(T):
            self._tool = SE3(T, check=False)
        else:
            raise ValueError('base must be set to None (no tool) or an SE3')
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
            raise ValueError('tool must be set to None (no tool) or an SE3')

# --------------------------------------------------------------------- #

    @property
    def gravity(self):
        """
        Get/set default gravitational acceleration (Robot superclass)

        - ``robot.name`` is the default gravitational acceleration

        :return: robot name
        :rtype: ndarray(3,)

        - ``robot.name = ...`` checks and sets default gravitational
          acceleration

        .. note:: If the z-axis is upward, out of the Earth, this should be
            a positive number.
        """
        return self._gravity

    @gravity.setter
    def gravity(self, gravity_new):
        self._gravity = getvector(gravity_new, 3)
        self.dynchanged()

# --------------------------------------------------------------------- #

    @property
    def qlim(self):
        r"""
        Joint limits (Robot superclass)

        :return: Array of joint limit values
        :rtype: ndarray(2,n)

        Example:

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> robot = rtb.models.DH.Puma560()
            >>> robot.qlim
        """
        # TODO tidy up
        limits = np.zeros((2, self.n))
        for j, link in enumerate(self):
            if link.qlim is None:
                if link.isrevolute():
                    v = np.r_[-np.pi, np.pi]
                else:
                    raise ValueError('undefined prismatic joint limit')
            else:
                v = link.qlim
            limits[:, j] = v
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
        if cn == 'p' or cn == 'v' or cn == 'a':
            self._control_type = cn
        else:
            raise ValueError(
                'Control type must be one of \'p\', \'v\', or \'a\'')

# ===================================================================== #

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
        Numerical inverse kinematics by optimization (Robot superclass)

        :param T: The desired end-effector pose
        :type T: SE3 with 1 or ``m`` values
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

        :return: The calculated joint values
        :rtype: float ndarray(n)
        :return: IK solver failed
        :rtype: bool or list of bool
        :return: If failed, what went wrong
        :rtype: List of str

        ``q, failure, reason = ikine(T)`` are the joint coordinates (n)
        corresponding to the robot end-effector pose ``T`` which is an ``SE3``
        instance. ``failure`` is True if the solver failed, and ``reason``
        contains details of the failure.

        This method can be used for robots with any number of degrees of
        freedom.

        **Trajectory operation:**

        If ``T`` contains ``m`` > 1 values, ie. a trajectory, then returns the
        joint coordinates corresponding to each of the pose values in ``T``.
        ``q`` (m,n) where ``n`` is the number of robot joints. The initial
        estimate of ``q`` for each time step is taken as the solution from the
        previous time step. Returns trajectory of joints ``q`` (m,n), list of
        failure (m) and list of error reasons (m).

        **Underactuated robots:**

        For the case where the manipulator has fewer than 6 DOF the
        solution space has more dimensions than can be spanned by the
        manipulator joint coordinates.

        In this case we specify the ``mask`` option where the ``mask`` vector (6)
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

        .. note::

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

            qlim = self.qlim
            qspan = qlim[1] - qlim[0]  # range of joint motion

            for k in range(slimit):
                q0n = np.random.rand(self.n) * qspan + qlim[0, :]

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
                    dq = transpose * J.T @ e    # lgtm [py/multiple-definition]
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

# --------------------------------------------------------------------- #

    def ikunc(self, T, q0=None, ilimit=1000):
        """
        Inverse manipulator by optimization without joint limits (Robot
        superclass)

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

        .. note::
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

        omega = np.diag([1, 1, 1, 3 / self.reach])

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

# --------------------------------------------------------------------- #

    def ikcon(self, T, q0=None):
        """
        Inverse kinematics by optimization with joint limits (Robot superclass)

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

        .. note::
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

        omega = np.diag([1, 1, 1, 3 / self.reach])

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

# --------------------------------------------------------------------- #

    def qmincon(self, q=None):
            """
            Move away from joint limits

            :param q: Joint coordinates
            :type q: ndarray(n)
            :retrun qs: The calculated joint values
            :rtype qs: ndarray(n)
            :return: Optimisation solved (True) or failed (False)
            :rtype: bool
            :return: Final value of the objective function
            :rtype: float

            ``qs, success, err = qmincon(q)`` exploits null space motion and returns
            a set of joint angles ``qs`` (n) that result in the same end-effector
            pose but are away from the joint coordinate limits. ``n`` is the number
            of robot joints. ``success`` is True for successful optimisation.
            ``err`` is the scalar final value of the objective function.

            **Trajectory operation**

            In all cases if ``q`` is (m,n) it is taken as a pose sequence and
            ``qmincon()`` returns the adjusted joint coordinates (m,n) corresponding
            to each of the configurations in the sequence.

            ``err`` and ``success`` are also (m) and indicate the results of
            optimisation for the corresponding trajectory step.

            .. note:: Robot must be redundant.

            """

            def sumsqr(A):
                return np.sum(A**2)

            def cost(x, ub, lb, qm, N):
                return sumsqr(
                    (2 * (N @ x + qm) - ub - lb) / (ub - lb))

            q = getmatrix(q, (None, self.n))

            qstar = np.zeros((q.shape[0], self.n))
            error = np.zeros(q.shape[0])
            success = np.zeros(q.shape[0])

            lb = self.qlim[0, :]
            ub = self.qlim[1, :]

            for k, qk in enumerate(q):

                J = self.jacobe(qk)

                N = null(J)

                x0 = np.zeros(N.shape[1])
                A = np.r_[N, -N]
                b = np.r_[ub - qk, qk - lb].reshape(A.shape[0],)

                con = LinearConstraint(A, -np.inf, b)

                res = minimize(
                    lambda x: cost(x, ub, lb, qk, N),
                    x0, constraints=con)

                qstar[k, :] = qk + N @ res.x
                error[k] = res.fun
                success[k] = res.success

            if q.shape[0] == 1:
                return qstar[0, :], success[0], error[0]
            else:
                return qstar, success, error

# --------------------------------------------------------------------- #

    def friction(self, qd):
        r"""
        Manipulator joint friction (Robot superclass)

        :param qd: The joint velocities of the robot
        :type qd: ndarray(n)

        :return: The joint friction forces/torques for the robot
        :rtype: ndarray(n,)

        ``robot.friction(qd)`` is a vector of joint friction
        forces/torques for the robot moving with joint velocities ``qd``.

        The friction model includes:

        - Viscous friction which is a linear function of velocity.
        - Coulomb friction which is proportional to sign(qd).

        .. math::

            \tau_j = G^2 B \dot{q}_j + |G_j| \left\{ \begin{array}{ll}
                \tau_{C,j}^+ & \mbox{if $\dot{q}_j > 0$} \\
                \tau_{C,j}^- & \mbox{if $\dot{q}_j < 0$} \end{array} \right.

        .. note::

            - The friction value should be added to the motor output torque to
              determine the nett torque. It has a negative value when qd > 0.
            - The returned friction value is referred to the output of the
              gearbox.
            - The friction parameters in the Link object are referred to the
              motor.
            - Motor viscous friction is scaled up by :math:`G^2`.
            - Motor Coulomb friction is scaled up by math:`G`.
            - The appropriate Coulomb friction value to use in the
              non-symmetric case depends on the sign of the joint velocity,
              not the motor velocity.
            - Coulomb friction is zero for zero joint velocity, stiction is
              not modeled.
            - The absolute value of the gear ratio is used.  Negative gear
              ratios are tricky: the Puma560 robot has negative gear ratio for
              joints 1 and 3.
            - The absolute value of the gear ratio is used. Negative gear
              ratios are tricky: the Puma560 has negative gear ratio for
              joints 1 and 3.

        :seealso: :func:`Robot.nofriction`, :func:`Link.friction`
        """

        qd = getvector(qd, self.n)
        tau = np.zeros(self.n)

        for i in range(self.n):
            tau[i] = self.links[i].friction(qd[i])

        return tau

# --------------------------------------------------------------------- #

    def nofriction(self, coulomb=True, viscous=False):
        """
        Remove manipulator joint friction (Robot superclass)

        :param coulomb: set the Coulomb friction to 0
        :type coulomb: bool
        :param viscous: set the viscous friction to 0
        :type viscous: bool
        :return: A copy of the robot with dynamic parameters perturbed
        :rtype: Robot subclass

        ``nofriction()`` copies the robot and returns
        a robot with the same link parameters except the Coulomb and/or viscous
        friction parameter are set to zero.

        :seealso: :func:`Robot.friction`, :func:`Link.nofriction`
        """

        # shallow copy the robot object
        self.delete_rne()  # remove the inherited C pointers
        nf = copy.copy(self)
        nf.name = 'NF/' + self.name

        # add the modified links (copies)
        nf._links = [link.nofriction(coulomb, viscous) for link in self.links]

        return nf
