"""
Python Navigation Abstract Class
@Author: Peter Corke
@Author: Kristian Gibson
"""

from abc import ABC

# from scipy import integrate
# from scipy.ndimage import interpolation
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *
from spatialmath.base.argcheck import getvector
from spatialmath.base.graphics import axes_logic, plotvol2, axes_get_scale

# from spatialmath import SE2, SE3
from matplotlib import cm
from abc import ABC
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from roboticstoolbox.mobile.OccGrid import BaseOccupancyGrid, BinaryOccupancyGrid
from roboticstoolbox.mobile.Animations import VehiclePolygon
from colored import fg, attr

try:
    from progress.bar import FillingCirclesBar

    _progress = True
except ImportError:
    _progress = False


class PlannerBase(ABC):
    r"""
    Mobile robot motion planner (superclass)

    :param occgrid: occupancy grid, defaults to None
    :type occgrid: :class:`OccGrid` instance of ndarray(N,M), optional
    :param start: start position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to None
    :type start: array_like(2) or array_like(3), optional
    :param goal: goal position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to None
    :type goal: array_like(2) or array_like(3), optional
    :param inflate: obstacle inflation, defaults to 0
    :type inflate: float, optional
    :param ndims: dimensionality of the planning, either 2 for :math:`\mathbb{R}^2` or
        3 for :math:`\SE{2}`
    :param ndims: int, optional
    :param verbose: verbosity, defaults to False
    :type verbose: bool, optional
    :param msgcolor: color for message channel printing
    :type msgcolor: str, defaults to yellow
    :param seed: seed provided to private random number generator, defaults to None
    :type seed: int, optional

    Superclass for all mobile robot motion planners.  Key functionality
    includes:

    - encapsulates an occupancy grid and optionally inflates it
    - validates ``start`` and ``goal`` if given
    - encapsulates a private random number generator with specifiable seed
    - encapsulates state such as start, goal, and the plan
    - provides a message channel for diagnostic output

    The start and goal can be specifed in various ways:

    - at constructor time by the arguments ``start`` or ``goal``
    - by assigning the attributes ``start`` or ``goal``
    - at planning time by specifying ``goal`` to :meth:`plan`
    - at query time by specifying ``start`` to :meth:`query`

    :seealso: :class:`OccGrid`
    """

    def __init__(
        self,
        occgrid=None,
        inflate=0,
        ndims=None,
        start=None,
        goal=None,
        verbose=False,
        msgcolor="yellow",
        progress=True,
        marker=None,
        seed=None,
        **unused,
    ):

        self._occgrid = None
        if ndims is None:
            raise ValueError("ndims must be specified")
        self._ndims = ndims
        self._verbose = verbose
        self._msgcolor = msgcolor
        self._seed = seed
        self._private_random = np.random.default_rng(seed=seed)
        self._inflate = inflate

        self._progress = progress and _progress

        self.marker = marker

        if occgrid is not None:
            if isinstance(occgrid, np.ndarray) and occgrid.ndim == 2:
                # it's a NumPy array
                self._occgrid = BinaryOccupancyGrid(occgrid)
            elif isinstance(occgrid, BinaryOccupancyGrid):
                self._occgrid = occgrid  # original occgrid for reference

            if inflate > 0:
                self._occgrid0 = self._occgrid.copy()
                self._occgrid.inflate(inflate)
            else:
                self._occgrid0 = self._occgrid

        self._start = self.validate_endpoint(start)
        self._goal = self.validate_endpoint(goal)

    def __str__(self):
        """
        Compact representation of the planner

        :return: pretty printed representation
        :rtype: str
        """
        s = f"{self.__class__.__name__}: "
        if hasattr(self, "_occgrid") and self._occgrid is not None:
            s += "\n  " + str(self._occgrid)
        if self._start is not None:
            s += f"\n  Start: {self.start}"
        if self._goal is not None:
            s += f"\n  Goal: {self.goal}"
        return s

    def __repr__(self):
        return str(self)

    @property
    def start(self):
        r"""
        Set/get start point or configuration (superclass)

        :getter: Return start point or configuration
        :rtype: ndarray(2) or ndarray(3)
        :setter: Set start point or configuration
        :param: array_like(2) or array_like(3)

        The start is either a point :math:`(x, y)` or a configuration :math:`(x, y, \theta)`.

        :seealso: :meth:`goal`
        """
        return self._start

    @start.setter
    def start(self, start):
        if start is not None:
            if self.isoccupied(start):
                raise ValueError("Start location inside obstacle")
            self._start = getvector(start)

    @property
    def goal(self):
        r"""
        Set/get goal point or configuration (superclass)

        :getter: Return goal pointor configuration
        :rtype: ndarray(2) or ndarray(3)
        :setter: Set goal point or configuration
        :param: array_like(2) or array_like(3)

        The goal is either a point :math:`(x, y)` or a configuration :math:`(x, y, \theta)`.

        :seealso: :meth:`goal`
        """
        return self._goal

    @goal.setter
    def goal(self, goal):
        r"""
        Set goal point or configuration for planning

        :param goal: Set goal :math:`(x, y)` or configuration :math:`(x, y, \theta)`
        :type goal: array_like(2) or array_like(3)
        :raises ValueError: if goal point is occupied
        """
        if goal is not None:
            if self.isoccupied(goal):
                raise ValueError("Goal location inside obstacle")
            self._goal = getvector(goal)

    @property
    def verbose(self):
        """
        Get verbosity

        :return: verbosity
        :rtype: bool

        If ``verbosity`` print more diagnostic messages to the planner's
        message channel.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        """
        Set verbosity

        :param v: verbosity
        :type v: bool

        If ``verbosity`` print more diagnostic messages to the planner's
        message channel.
        """
        self._verbose = v

    @property
    def random(self):
        """
        Get private random number generator

        :return: NumPy random number generator
        :rtype: :class:`numpy.random.Generator`

        Has methods including:

        - :meth:`integers(low, high, size, endpoint) <numpy.random.Generator.integers>`
        - :meth:`random(size) <numpy.random.Generator.random>`
        - :meth:`uniform(low, high, size) <numpy.random.Generator.uniform>`
        - :meth:`normal(mean, std, size) <numpy.random.Generator.normal>`
        - :meth:`multivariate_normal(mean, covar, size) <numpy.random.Generator.multivariate_normal>`

        The generator is initialized with the seed provided at constructor
        time.

        :seealso: :func:`numpy.random.default_rng`
        """
        return self._private_random

    def random_init(self, seed=None):
        """
        Initialize private random number generator

        :param seed: random number seed, defaults to value given to constructor
        :type seed: int

        The private random number generator is initialized.  The seed is ``seed``
        or the value given to the constructor.  If None, the generator will
        be randomly seeded using a seed from the operating system.
        """
        if seed is None:
            seed = self._seed

        self._private_random = np.random.default_rng(seed=seed)

    # def randinit(self):
    #     if self._seed is not None:
    #         self._private_random = np.random.default_rng(seed=self._seed)

    def plan(self):
        r"""
        Plan path (superclass)

        :param start: start position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to value passed to constructor
        :type start: array_like(2) or array_like(3), optional
        :param goal: goal position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to value passed to constructor
        :type goal: array_like(2) or array_like(3), optional

        The implementation depends on the particular planner.  Some may have
        no planning phase.  The plan may also depend on just the start or goal.
        """
        pass

    @property
    def occgrid(self):
        """
        Occupancy grid

        :return: occupancy grid used for planning
        :rtype: :class:`OccGrid` or subclass or None

        Returns the occupancy grid that was optionally inflated at constructor time.

        :seealso: :meth:`validate_endpoint` :meth:`isoccupied`
        """
        return self._occgrid

    def isoccupied(self, p):
        """
        Test if point is occupied (superclass)

        :param p: world coordinate (x, y)
        :type p: array_like(2)
        :return: occupancy status of corresponding grid cell
        :rtype: bool

        The world coordinate is transformed and the status of the occupancy
        grid cell is returned.  If the point lies outside the bounds of
        the occupancy grid return True (obstacle)

        If there is no occupancy grid this function always returns False (free).

        :seealso:  :meth:`occgrid` :meth:`validate_endpoint` :meth:`BinaryOccGrid.isoccupied`
        """
        if self.occgrid is None:
            return False
        else:
            return self.occgrid.isoccupied(p)

    def validate_endpoint(self, p, dtype=None):
        """
        Validate start or goal point

        :param p: the point
        :type p: array_like(2)
        :param dtype: data type for point coordinates, defaults to None
        :type dtype: str, optional
        :raises ValueError: point is inside obstacle
        :return: point as a NumPy array of specified dtype
        :rtype: ndarray(2)

        The coordinate is tested to be a free cell in the occupancy grid.

        :seealso: :meth:`isoccupied` :meth:`occgrid`
        """
        if p is not None:
            p = getvector(p, self._ndims, dtype=dtype)
            if self.isoccupied(p):
                raise ValueError("Point is inside obstacle")
        return p

    def progress_start(self, n):
        """
        Initialize a progress bar (superclass)

        :param n: Number of iterations in the operation
        :type n: int

        Create a progress bar for an operation that has ``n`` steps, for
        example::

            planner.progress_start(100)
            for i in range(100):
                # ...
                planner.progress_next()
            planner.progress_end()

        .. warning: Requires that the ``progress`` package is installed.

        :seealso: :meth:`progress_next` :meth:`progress_end`
        """
        if _progress:
            self._bar = FillingCirclesBar(
                self.__class__.__name__, max=n, suffix="%(percent).1f%% - %(eta)ds"
            )

    def progress_next(self):
        """
        Increment a progress bar (superclass)

        Create a progress bar for an operation that has ``n`` steps, for
        example::

            planner.progress_start(100)
            for i in range(100):
                # ...
                planner.progress_next()
            planner.progress_end()

        .. warning: Requires that the ``progress`` package is installed.

        :seealso: :meth:`progress_start` :meth:`progress_end`
        """
        if _progress:
            self._bar.next()

    def progress_end(self):
        """
        Finalize a progress bar  (superclass)

        Remove/cleanip a progress bar, for
        example::

            planner.progress_start(100)
            for i in range(100):
                # ...
                planner.progress_next()
            planner.progress_end()

        .. warning: Requires that the ``progress`` package is installed.

        :seealso: :meth:`progress_start` :meth:`progress_next`
        """
        if _progress:
            self._bar.finish()

    def query(
        self, start=None, goal=None, dtype=None, next=True, animate=False, movie=None
    ):
        r"""
        Find a path from start to goal using planner (superclass)

        :param start: start position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to value specified to constructor
        :type start: array_like(), optional
        :param goal: goal position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to value specified to constructor
        :type goal: array_like(), optional
        :param dtype: data type for point coordinates, defaults to None
        :type dtype: str, optional
        :param next: invoke :meth:`next` method of class, defaults to True
        :type next: bool, optional
        :param animate: show the vehicle path, defaults to False
        :type animate: bool, optional
        :return: path from start to goal, one point :math:`(x, y)` or configuration :math:`(x, y, \theta)` per row
        :rtype: ndarray(N,2) or ndarray(N,3)

        Find a path from ``start`` to ``goal`` using a previously computed plan.

        This is a generic method that works for any planner
        (:math:`\mathbb{R}^2` or :math:`\SE{2}`) that can incrementally
        determine the next point on the path.  The method performs the following
        steps:

        - Validate start and goal
        - If ``animate``, visualize the environment using :meth:`plot`
        - Iterate on the class's :meth:`next` method until the ``goal`` is
          achieved, and if ``animate`` plot points.

        :seealso: :meth:`next` :meth:`plan`
        """
        # make sure start and goal are set and valid
        self.start = self.validate_endpoint(start, dtype=dtype)
        self.goal = self.validate_endpoint(goal, dtype=dtype)

        # if movie is not None:
        #     animate = True

        if animate:
            self.plot()

        # movie = MovieWriter(movie)

        robot = self._start
        path = [robot]

        while next:
            if animate:
                plt.plot(robot[0], robot[1], "y.", 12)
                plt.pause(0.05)

            # get next point on the path
            robot = self.next(robot)

            # are we are done?
            if robot is None:
                path.append(self._goal)
                return np.array(path).astype(int)

            path.append(robot)

    def plot(
        self,
        path=None,
        line=None,
        line_r=None,
        configspace=False,
        unwrap=True,
        direction=None,
        background=True,
        path_marker=None,
        path_marker_reverse=None,
        start_marker=None,
        goal_marker=None,
        start_vehicle=None,
        goal_vehicle=None,
        start=None,
        goal=None,
        ax=None,
        block=None,
        bgargs={},
        **unused,
    ):
        r"""
        Plot vehicle path (superclass)

        :param path: path, defaults to None
        :type path: (N, 2) or ndarray(N, 3)
        :param direction: travel direction associated with each point on path, is either >0 or <0, defaults to None
        :type direction: array_like(N), optional
        :param line: line style for forward motion, default is striped yellow on black
        :type line: sequence of dict of arguments for ``plot``
        :param line_r: line style for reverse motion, default is striped red on black
        :type line_r: sequence of dict of arguments for ``plot``

        :param configspace: plot the path in 3D configuration space, input must be 3xN.
            Start and goal style will be given by ``qstart_marker`` and ``qgoal_marker``, defaults to False
        :type configspace: bool, optional
        :param unwrap: for configuration space plot unwrap :math:`\theta` so
            there are no discontinuities at :math:`\pm \pi`, defaults to True
        :type unwrap: bool, optional
        :param background: plot occupancy grid if present, default True
        :type background: bool, optional
        :param start_marker: style for marking start point
        :type start_marker: dict, optional
        :param goal_marker: style for marking goal point
        :type goal_marker: dict, optional
        :param start_vehicle: style for vehicle animation object at start configuration
        :type start_vehicle: dict
        :param goal_vehicle: style for vehicle animation object at goal configuration
        :type goal_vehicle: dict
        :param start: start position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to value previously set
        :type start: array_like(2) or array_like(3), optional
        :param goal: goal position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to value previously set
        :type goal: array_like(2) or array_like(3), optional
        :param bgargs: arguments passed to :meth:`plot_bg`, defaults to None
        :type bgargs: dict, optional
        :param ax: axes to plot into
        :type ax: matplotlib axes
        :param block: block after displaying the plot
        :type block: bool, optional

        Plots the start and goal location/pose if they are specified by
        ``start`` or ``goal`` or were set by the object constructor or
        ``plan`` or ``query`` method.

        If the ``start`` and ``goal`` have length 2, planning in
        :math:`\mathbb{R}^2`, then markers are drawn using styles specified by
        ``start_marker`` and ``end_marker`` which are dicts using Matplotlib
        keywords, for example::

            planner.plot(path, start=dict(marker='s', color='b'))

        If the ``start`` and ``goal`` have length 3, planning in :math:`\SE{2}`,
        and ``configspace`` is False, then direction-indicating markers are used
        to display start and goal configuration. These are also given as dicts
        but have two items: ``'shape'`` which is the shape of the polygonal
        marker and is either ``'triangle'`` or ``'car'``.  The second item
        ``'args'`` is passed to :func:`base.plot_poly` and Matplotlib and could
        have values such as ``filled=True`` or ``color``.

        If ``configspace`` is False then a 3D plot is created and the start and
        goal are indicated by Matplotlib markers specified by the dicts
        ``start_marker`` and ``end_marker``

        Default values are provided for all markers:

            - the start point is a circle
            - the goal point is a star
            - the start vehicle style is a ``VehiclePolygon(shape='car')`` as
              an unfilled outline
            - the goal vehicle style is a ``VehiclePolygon(shape='car')`` as
              a transparent filled shape

        If ``background`` is True then the background of the plot is either or
        both of:

        - the occupancy grid
        - the distance field of the planner

        Additional arguments ``bgargs`` can be passed through to :meth:`plot_bg`

        If ``path`` is specified it has one column per point and either 2 or 3 rows:

        - 2 rows describes motion in the :math:`xy`-plane and a 2D plot  is created
        - 3 rows describes motion in the :math:`xy\theta`-configuration space. By
          default only the :math:`xy`-plane is plotted unless ``configspace``
          is True in which case motion in :math:`xy\theta`-configuration space
          is shown as a 3D plot.

        If the planner supports bi-directional motion then the ``direction``
        option gives the direction for each point on the path.

        Forward motion segments are drawn using style information from ``line``
        while reverse motion segments are drawn using style information from
        ``line_r``. These are each a sequence of dicts of Matplotlib plot
        options which can draw an arbitrary number of lines on top of each
        other.  The default::

            line = (
                    {color:'black', linewidth:4},
                    {color:'yellow', linewidth:3, dashes:(5,5)}
                )

        will draw a blackline of width 4 with a dashed yellow line of width 3
        plotted on top, giving a line of alternating black and yellow dashes.

        :seealso: :meth:`plot_bg` :func:`base.plot_poly`
        """
        # create default markers
        # passed to Matplotlib plot()
        if start_marker is None:
            start_marker = {
                "marker": "o",
                "markeredgecolor": "k",
                "markerfacecolor": "y",
                "markersize": 10,
                "zorder": 10,
                "linestyle": "none",
            }
        if goal_marker is None:
            goal_marker = {
                "marker": "*",
                "markeredgecolor": "k",
                "markerfacecolor": "y",
                "markersize": 16,
                "zorder": 10,
                "linestyle": "none",
            }

        # create defaut line styles
        if line is None:
            line = (
                {"color": "black", "linewidth": 4},
                {"color": "yellow", "linewidth": 3, "dashes": (5, 5)},
            )
        if line_r is None:
            line_r = (
                {"color": "black", "linewidth": 4},
                {"color": "red", "linewidth": 3, "dashes": (5, 5)},
            )

        # passed to VehiclePolygon
        if start_vehicle is None:
            start_vehicle = {"facecolor": "none", "edgecolor": "k", "linewidth": 2}

        if goal_vehicle is None:
            goal_vehicle = {"alpha": 0.5}

        ndims = self._ndims

        if ndims == 3 and not configspace:
            ndims = 2
            if path is not None:
                path = path[:, :2]

        if configspace and ndims < 3 and path is not None:
            raise ValueError(f"path should have {ndims} rows")

        ax = axes_logic(ax, ndims)

        # plot occupancy grid background
        if background:
            self.plot_bg(ax=ax, **bgargs)

        # mark the path
        if path is not None:
            if ndims == 2:
                # 2D case
                if direction is not None:
                    # bidirectional motion
                    direction = np.array(direction)
                    if direction.shape[0] != path.shape[0]:
                        raise ValueError(
                            "direction vector must have same length as path"
                        )

                    while len(direction) > 0:
                        dir = direction[0]
                        change = np.argwhere(dir != direction)
                        if len(change) == 0:
                            k = -1
                        else:
                            k = change[0, 0]

                        for style in line if dir > 0 else line_r:
                            ax.plot(path[:k, 0], path[:k, 1], zorder=9, **style)

                        if len(change) == 0:
                            break
                        direction = direction[k - 1 :]
                        direction[0] = direction[1]
                        path = path[k - 1 :, :]

                else:
                    # forward motion only
                    for style in line:
                        ax.plot(path[:, 0], path[:, 1], zorder=9, **style)

            elif ndims == 3:
                # 3D case
                if direction is not None:
                    # bidirectional motion

                    direction = np.array(direction)
                    if direction.shape[0] != path.shape[0]:
                        raise ValueError(
                            "direction vector must have same length as path"
                        )
                    theta = path[:, 2]
                    if unwrap:
                        theta = np.unwrap(theta)

                    while len(direction) > 0:
                        dir = direction[0]
                        change = np.argwhere(dir != direction)
                        if len(change) == 0:
                            k = -1
                        else:
                            k = change[0, 0]

                        for style in line if dir > 0 else line_r:
                            ax.plot(path[:k, 0], path[:k, 1], theta[:k], **style)

                        if len(change) == 0:
                            break
                        direction = direction[k - 1 :]
                        direction[0] = direction[1]
                        path = path[k - 1 :, :]
                        theta = theta[k - 1 :]

                else:
                    # forward motion only
                    theta = path[:, 2]
                    if unwrap:
                        theta = np.unwrap(theta)
                    for style in line:
                        ax.plot(path[:, 0], path[:, 1], theta, **style)

        # mark start and goal if requested
        if start is not None:
            start = self.validate_endpoint(start)
        else:
            start = self.start
        if goal is not None:
            self.goal = self.validate_endpoint(goal)
        else:
            goal = self.goal

        if ndims == 2 and self._ndims == 2:
            # proper 2d plot
            if start is not None:
                ax.plot(start[0], start[1], label="start", **start_marker)
            if goal is not None:
                ax.plot(goal[0], goal[1], label="goal", **goal_marker)

        elif ndims == 2 and self._ndims == 3:
            # 2d projection of 3d plot, show start/goal configuration
            scale = axes_get_scale(ax) / 10

            if self.marker is None:
                self.marker = VehiclePolygon(shape="car", scale=scale)

            if start is not None:
                self.marker.plot(start, **start_vehicle)
            if goal is not None:
                self.marker.plot(goal, **goal_vehicle)

        elif ndims == 3:
            # 3d plot

            if start is not None:
                ax.plot(start[0], start[1], start[2], label="start", **start_marker)
            if goal is not None:

                if path is not None and unwrap:
                    theta = theta[-1]
                else:
                    theta = goal[2]
                plt.plot(goal[0], goal[1], theta, label="goal", **goal_marker)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if ndims == 2:
            ax.set_aspect("equal")
        else:
            ax.set_zlabel(r"$\theta$")

        if block:
            plt.show(block=block)

        return ax

    def _qmarker(self, shape):
        h = 0.3
        t = 0.8  # start of head taper
        c = 0.5  # centre x coordinate
        w = 1  # width in x direction
        if shape == "car":
            return np.array(
                [
                    [-c, h],
                    [t - c, h],
                    [w - c, 0],
                    [t - c, -h],
                    [-c, -h],
                ]
            ).T
        elif shape == "triangle":
            return np.array(
                [
                    [-c, h],
                    [w, 0],
                    [-c, -h],
                    [-c, h],
                ]
            ).T

    def plot_bg(
        self,
        distance=None,
        cmap="gray",
        ax=None,
        inflated=True,
        colorbar=True,
        block=None,
        **unused,
    ):
        """
        Plot background (superclass)

        :param distance: override distance field, defaults to None
        :type distance: ndarray(N,M), optional
        :param cmap: Specify a colormap for the distance field, defaults to 'gray'
        :type cmap: str or Colormap, optional

        Displays the background which is either the occupancy grid or a distance
        field.  The distance field encodes the distance of a point from the goal, small
        distance is dark, a large distance is bright.

        If the planner has an occupancy grid then that will be displayed with:
            - free cells in white
            - occupied cells in red
            - inflated occupied cells in pink

        If distance is provided, or the planner has a distancemap attribute
        the the distance field will be used as the background and obstacle cells
        (actual or inflated) will be shown in red. A colorbar is added.
        """
        if self._occgrid is None:
            return

        if isinstance(self._occgrid, BaseOccupancyGrid):
            ax = plotvol2(dim=self._occgrid.workspace, ax=ax)
        else:
            ax = axes_logic(ax, 2)

        # create color map for free space + obstacle:
        #   free space, color index = 1, white, alpha=0 to allow background and grid lines to show
        #   obstacle, color index = 2, red, alpha=1

        if self._inflate > 0 and inflated:
            # 0 background (white, transparent)
            # 1 inflated obstacle (pink)
            # 2 original obstacle (red)
            colors = [(1, 1, 1, 0), (1, 0.75, 0.8, 1), (1, 0, 0, 1)]
            image = self.occgrid.grid.astype(int) + self._occgrid0.grid.astype(int)
        else:
            # 0 background
            # 1 obstacle
            colors = [(1, 1, 1, 0), (1, 0, 0, 1)]
            image = self.occgrid.grid

        if distance is None and hasattr(self, "distancemap"):
            distance = self.distancemap

        if distance is not None:
            # distance field with obstacles

            # find largest finite value

            v = distance.ravel()
            vmax = max(v[np.isfinite(v)])

            # create a copy of greyscale color map
            c_map = copy.copy(mpl.cm.get_cmap(cmap))
            # c_map.set_bad(color=(1,0,0,1))  # nan and inf are red

            # change all inf to large value, so they are not 'bad' ie. red
            distance[np.isinf(distance)] = 2 * vmax
            c_map.set_over(color=(0, 0, 1))  # ex-infs are now blue

            # display image
            norm = mpl.colors.Normalize(vmin=0, vmax=vmax, clip=False)
            ax.imshow(
                distance,
                origin="lower",
                interpolation=None,
                cmap=c_map,
                norm=norm,
            )
            ax.grid(True, alpha=0.1, color=(1, 1, 1))

            # add colorbar
            scalar_mappable_c_map = cm.ScalarMappable(cmap=c_map, norm=norm)
            if colorbar is True:
                plt.colorbar(
                    scalar_mappable_c_map,
                    # shrink=0.75,
                    # aspect=20 * 0.75,
                    label="Distance",
                )

            elif isinstance(colorbar, dict):
                if "label" not in colorbar:
                    colorbar["label"] = "Distance"
                plt.colorbar(scalar_mappable_c_map, **colorbar)
            # overlay obstacles
            c_map = mpl.colors.ListedColormap(colors)
            self.occgrid.plot(image, cmap=c_map, zorder=1)

        else:
            # occupancy grid only

            # overlay obstacles
            c_map = mpl.colors.ListedColormap(colors)
            # self.occgrid.plot(image, cmap=c_map, zorder=1)
            self.occgrid.plot(cmap=c_map, zorder=1, ax=ax)

        ax.set_facecolor((1, 1, 1))  # create white background
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
        ax.grid(True, zorder=0)

        # lock axis limits to current value
        # ax.set_xlim(ax.get_xlim())
        # ax.set_ylim(ax.get_ylim())

        # plt.draw()
        if block is not None:
            plt.show(block=block)

    def message(self, s, color=None):
        """
        Print message to message channel

        :param s: message to print
        :type s: str
        :param color: color to print it, defaults to color specified at
            constructor time.
        :type color: str, optional

        """
        if self.verbose:
            if color is None:
                color = self._msgcolor
            print(fg(color), "Planner:: " + s, attr(0))

    # @staticmethod
    # def show_distance(d):
    #     d[np.isinf(d)] = None
    #     ax = plt.gca()
    #     c_map = plt.get_cmap("Greys")
    #     plt.clim(0, np.max(d[:]))
    #     plt.figimage(d)
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.show()


class MovieWriter:
    def __init__(self, filename=None, interval=0.1, fig=None):
        """
        Save animation as a movie file

        :param filename: name of movie file, or tuple containing filename and
            frame interval
        :type filename: str or tuple(str, float)
        :param interval: frame interval, defaults to 0.1
        :type interval: float, optional
        :param fig: figure to record for the movie
        :type fig: figure reference

        Example::

            movie = MovieWriter(filename)

            while ...
                movie.add()

            movie.done()

        To avoid extra user-logic, if ``MovieWriter`` is called with ``filename`` equal to None,
        then the writer will do nothing when the ``add`` and ``done`` methods are called.
        """
        # Set up formatting for the movie files
        if filename is None:
            self.writer = None
            return

        if isinstance(filename, (tuple, list)):
            filename, interval = filename

        if os.path.exists(filename):
            print("overwriting movie", filename)
        else:
            print("creating movie", filename)
        self.writer = animation.FFMpegWriter(
            fps=round(1 / interval), extra_args=["-vcodec", "libx264"]
        )
        if fig is None:
            fig = plt.gcf()
        self.writer.setup(fig, filename)
        self.filename = filename

    def add(self):
        """
        Add frame to the movie
        """
        if self.writer is not None:
            self.writer.grab_frame()

    def done(self):
        if self.writer is not None:
            self.writer.finish()
            self.writer = None
