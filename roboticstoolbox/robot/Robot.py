import sys
import copy
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath.base.argcheck import isvector, getvector, getmatrix, \
    verifymatrix, getunit
from roboticstoolbox.robot.Link import Link
from spatialmath.base.transforms3d import tr2delta
# from roboticstoolbox.tools import urdf
# from roboticstoolbox.tools import xacro
from pathlib import PurePath, PurePosixPath
from scipy.optimize import minimize, Bounds, LinearConstraint
from roboticstoolbox.tools.null import null
from ansitable import ANSITable, Column

from roboticstoolbox.backends.PyPlot import PyPlot, PyPlot2
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
from roboticstoolbox.backends.Swift import Swift

from roboticstoolbox.robot.Dynamics import DynamicsMixin
from roboticstoolbox.robot.IK import IKMixin

try:
    from matplotlib import colors
    from matplotlib import cm
    _mpl = True
except ImportError:    # pragma nocover
    pass

try:
    import PIL
    _pil_exists = True
except ImportError:    # pragma nocover
    _pil_exists = False

# TODO maybe this needs to be abstract
# ikine functions need: fkine, jacobe, qlim methods from subclass


class Robot(DynamicsMixin, IKMixin):

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
        self._reach = None

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

    def copy(self):
        return copy.deepcopy(self)

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
            else:  # pragma nocover
                return str(theta * deg) + "\u00b0"

        config = self.config()
        # show named configurations
        if len(self._configdict) > 0:
            table = ANSITable(
                Column("name", colalign=">"),
                *[
                    Column(f"q{j:d}", colalign="<", headalign="<")
                    for j in range(self.n)],
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
        else:  # pragma nocover
            return ""

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

    def manipulability(
            self, q=None, J=None, method='yoshikawa',
            axes='all', **kwargs):
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
            "trans" or "rot"
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
        if axes == 'all':
            axes = [True, True, True, True, True, True]
        elif axes.startswith('trans'):
            axes = [True, True, True, False, False, False]
        elif axes.startswith('rot'):
            axes = [False, False, False, True, True, True]
        else:
            raise ValueError('axes must be all, trans or rot')

        def yoshikawa(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            if J.shape[0] == J.shape[1]:
                # simplified case for square matrix
                return abs(np.linalg.det(J))
            else:
                m2 = np.linalg.det(J @ J.T)
                return np.sqrt(abs(m2))

        def invcondition(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            return 1 / np.linalg.cond(J)  # return 1/cond(J)

        def minsingular(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            s = np.linalg.svd(J, compute_uv=False)
            return s[-1]  # return last/smallest singular value of J

        def asada(robot, J, q, axes, **kwargs):
            dof = np.sum(axes)
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
        if method == 'yoshikawa':
            mfunc = yoshikawa
        elif method == 'invcondition':
            mfunc = condition
        elif method == 'minsingular':
            mfunc = minsingular
        elif method == 'asada':
            mfunc = asada
        else:
            raise ValueError(
                "Invalid method chosen")

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
                if link.isrevolute:
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

# --------------------------------------------------------------------- #

    def plot(
            self, q, backend=None, block=True, dt=0.050,
            limits=None, vellipse=False, fellipse=False,
            jointaxes=True, eeframe=True, shadow=True, name=True, movie=None
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

        if backend is None:
            if isinstance(self, rtb.DHRobot):
                backend = 'pyplot'
            else:
                backend = 'swift'

        if backend.lower() == 'swift':  # pragma nocover
            if isinstance(self, rtb.ERobot):
                env = self._plot_swift(q=q, block=block)
            elif isinstance(self, rtb.DHRobot):
                raise NotImplementedError(
                    'Plotting in Swift is not implemented for DHRobots yet')

        elif backend.lower() == 'pyplot':
            # if isinstance(self, rtb.ERobot):  # pragma nocover
            #     raise NotImplementedError(
            #         'Plotting in PyPlot is not implemented for ERobots yet')
            # elif isinstance(self, rtb.DHRobot):
            env = self._plot_pyplot(
                q=q, block=block, dt=dt, limits=limits, vellipse=vellipse,
                fellipse=fellipse, jointaxes=jointaxes, eeframe=eeframe,
                shadow=shadow, name=name, movie=movie)

        return env

    def _plot_pyplot(
            self, q, block, dt, limits,
            vellipse, fellipse,
            jointaxes, eeframe, shadow, name, movie):

        # Make an empty 3D figure
        env = PyPlot()

        q = getmatrix(q, (None, self.n))

        # Add the self to the figure in readonly mode
        if q.shape[0] == 1:
            env.launch(self.name + ' Plot', limits)
        else:
            env.launch(self.name + ' Trajectory Plot', limits)

        env.add(
            self, readonly=True,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

        if vellipse:
            vell = self.vellipse(centre='ee')
            env.add(vell)

        if fellipse:
            fell = self.fellipse(centre='ee')
            env.add(fell)

        # Stop lint error
        images = []  # list of images saved from each plot

        if movie is not None:   # pragma nocover
            if not _pil_exists:
                raise RuntimeError(
                    'to save movies PIL must be installed:\npip3 install PIL')
            # make the background white, looks better than grey stipple
            env.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            env.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            env.ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        for qk in q:
            self.q = qk
            env.step(dt)

            if movie is not None:  # pragma nocover
                # render the frame and save as a PIL image in the list
                canvas = env.fig.canvas
                img = PIL.Image.frombytes(
                    'RGB', canvas.get_width_height(),
                    canvas.tostring_rgb())
                images.append(img)

        if movie is not None:  # pragma nocover
            # save it as an animated GIF
            images[0].save(
                movie,
                save_all=True, append_images=images[1:], optimize=False,
                duration=dt, loop=0)

        # Keep the plot open
        if block:           # pragma: no cover
            env.hold()

        return env

    def _plot_swift(self, q, block):   # pragma nocover

        # Make an empty 3D figure
        env = Swift()

        q = getmatrix(q, (None, self.n))
        self.q = q[0, :]

        # Add the self to the figure in readonly mode
        env.launch()

        env.add(
            self, readonly=True)

        for qk in q:
            self.q = qk
            env.step()

        # Keep the plot open
        if block:           # pragma: no cover
            env.hold()

        return env

# --------------------------------------------------------------------- #

    def fellipse(self, q=None, opt='trans', centre=[0, 0, 0]):
        '''
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

        '''
        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "ERobot fellipse not implemented yet")

        ell = EllipsePlot(self, q, 'f', opt, centre=centre)
        return ell

    def vellipse(self, q=None, opt='trans', centre=[0, 0, 0]):
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
            raise NotImplementedError(
                "ERobot vellipse not implemented yet")

        ell = EllipsePlot(self, q, 'v', opt, centre=centre)
        return ell

    def plot_ellipse(
            self, ellipse, block=True, limits=None,
            jointaxes=True, eeframe=True, shadow=True, name=True):
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
                'ellipse must be of type '
                'roboticstoolbox.backend.PyPlot.EllipsePlot')

        env = PyPlot()

        # Add the robot to the figure in readonly mode
        env.launch(ellipse.robot.name + ' ' + ellipse.name, limits=limits)

        env.add(
            ellipse,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

        # Keep the plot open
        if block:           # pragma: no cover
            env.hold()

        return env

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
                "Ellipse Plotting of ERobot's not implemented yet")

        if q is not None:
            self.q = q

        if fellipse is None:
            fellipse = self.fellipse(q=q, opt=opt, centre=centre)

        return self.plot_ellipse(
            fellipse, block=block, limits=limits,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

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
                "Ellipse Plotting of ERobot's not implemented yet")

        if q is not None:
            self.q = q

        if vellipse is None:
            vellipse = self.vellipse(q=q, opt=opt, centre=centre)

        return self.plot_ellipse(
            vellipse, block=block, limits=limits,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

# --------------------------------------------------------------------- #

    def plot2(
            self, q, block=True, dt=0.05, limits=None,
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

        - ``robot.plot2(q)`` displays a 2D graphical view of a robot based on
          the kinematic model and the joint configuration ``q``. This is a
          stick figure polyline which joins the origins of the link coordinate
          frames. The plot will autoscale with an aspect ratio of 1.

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

        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "2D Plotting of ERobot's not implemented yet")

        # Make an empty 2D figure
        env = PyPlot2()

        q = getmatrix(q, (None, self.n))

        # Add the self to the figure in readonly mode
        if q.shape[0] == 1:
            env.launch(self.name + ' Plot', limits)
        else:
            env.launch(self.name + ' Trajectory Plot', limits)

        env.add(
            self, readonly=True,
            eeframe=eeframe, name=name)

        if vellipse:
            vell = self.vellipse(centre='ee')
            env.add(vell)

        if fellipse:
            fell = self.fellipse(centre='ee')
            env.add(fell)

        for qk in q:
            self.q = qk
            env.step()

        # Keep the plot open
        if block:           # pragma: no cover
            env.hold()

        return env

# --------------------------------------------------------------------- #

    def teach(
            self, q=None, block=True, order='xyz', limits=None,
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

        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "2D Plotting of ERobot's not implemented yet")

        if q is not None:
            self.q = q

        # Make an empty 3D figure
        env = PyPlot()

        # Add the self to the figure in readonly mode
        env.launch('Teach ' + self.name, limits=limits)
        env.add(
            self, readonly=True,
            jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

        env._add_teach_panel(self)

        # Keep the plot open
        if block:           # pragma: no cover
            env.hold()

        return env

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

        - ``robot.teach2(q)`` creates a 2D matplotlib plot which allows the
          user to "drive" a graphical robot using a graphical slider panel.
          The robot's inital joint configuration is ``q``. The plot will
          autoscale with an aspect ratio of 1.

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

        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "2D Plotting of ERobot's not implemented yet")

        if q is not None:
            self.q = q

        # Make an empty 3D figure
        env = PyPlot2()

        # Add the robot to the figure in readonly mode
        env.launch('Teach ' + self.name, limits=limits)
        env.add(
            self, readonly=True,
            eeframe=eeframe, name=name)

        env._add_teach_panel(self)

        # Keep the plot open
        if block:           # pragma: no cover
            env.hold()

        return env

# --------------------------------------------------------------------- #

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
