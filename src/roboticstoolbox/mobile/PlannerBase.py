"""
Superclass for mobile robot motion planners, and a movie writer

@Author: Peter Corke
@Author: Kristian Gibson
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from roboticstoolbox.tools.plot import apply_block
import numpy as np
from colored import attr, fg
from matplotlib import animation
from spatialmath.base.argcheck import getvector
from spatialmath.base.graphics import axes_get_scale, axes_logic
from spatialmath.base.types import R2, R3, ArrayLike2, ArrayLike3

from roboticstoolbox.mobile.Animations import VehiclePolygon
from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from roboticstoolbox.tools.types import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

try:
    from progress.bar import FillingCirclesBar

    _progress = True
except ImportError:
    _progress = False

# Default Matplotlib styles.  A style given by the caller is merged over these,
# the caller's values win.  The path styles are sequences of layers that are
# drawn on top of each other, and a caller's replaces the whole sequence.

#: forward motion: a wide black line with a dashed yellow line on top
PATH_STYLE: tuple[dict[str, Any], ...] = (
    {"color": "black", "linewidth": 4},
    {"color": "yellow", "linewidth": 3, "dashes": (5, 5)},
)
#: reverse motion: a wide black line with a dashed red line on top
PATH_REVERSE_STYLE: tuple[dict[str, Any], ...] = (
    {"color": "black", "linewidth": 4},
    {"color": "red", "linewidth": 3, "dashes": (5, 5)},
)
#: start point: a circle
START_STYLE: dict[str, Any] = {
    "marker": "o",
    "markeredgecolor": "k",
    "markerfacecolor": "y",
    "markersize": 10,
    "zorder": 10,
    "linestyle": "none",
}
#: goal point: a star
GOAL_STYLE: dict[str, Any] = {
    "marker": "*",
    "markeredgecolor": "k",
    "markerfacecolor": "y",
    "markersize": 16,
    "zorder": 10,
    "linestyle": "none",
}
#: start configuration: an unfilled vehicle outline
START_VEHICLE_STYLE: dict[str, Any] = {
    "facecolor": "none",
    "edgecolor": "k",
    "linewidth": 2,
}
#: goal configuration: a transparent filled vehicle
GOAL_VEHICLE_STYLE: dict[str, Any] = {"alpha": 0.5}


def _layers(
    style: dict[str, Any] | Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """A style given as one dict or as a sequence of dicts, as a tuple of dicts"""
    if isinstance(style, dict):
        return (style,)
    return tuple(style)


def _warn_unused(unused: dict[str, Any], stacklevel: int = 3) -> None:
    """
    Warn about keyword arguments that are accepted and ignored

    RTB 1.4.4 and earlier silently ignored unknown keyword arguments of the
    planners, and code in the book relies on that, for example
    ``dx.plot(labelvalues=True)``.  They are ignored with a warning so that a
    mistyped option is not silent.

    :param unused: the unknown keyword arguments
    :param stacklevel: of the warning, relative to the caller of this function
    """
    for name in unused:
        warnings.warn(
            f"unknown argument {name!r} is ignored",
            DeprecationWarning,
            stacklevel=stacklevel,
        )


class NoPathError(RuntimeError):
    """
    A valid planning problem has no solution

    Raised by a planner when the start, goal and environment are valid but there
    is no path between them, for example the goal is unreachable or the robot is
    trapped.  Invalid input raises :class:`ValueError` instead.  This is a
    :class:`RuntimeError`, so code that catches that continues to work.
    """


class PlannerBase:
    r"""
    Mobile robot motion planner (superclass)

    :param occgrid: occupancy grid, defaults to None
    :type occgrid: :class:`BinaryOccupancyGrid` or ndarray(N,M), optional
    :param inflate: obstacle inflation in world units, defaults to 0
    :param ndims: dimensionality of the planning, either 2 for :math:`\mathbb{R}^2`
        or 3 for :math:`\SE{2}`
    :param start: start position :math:`(x, y)` or configuration :math:`(x, y, \theta)`,
        defaults to None
    :param goal: goal position :math:`(x, y)` or configuration :math:`(x, y, \theta)`,
        defaults to None
    :param verbose: print diagnostic messages to the message channel, defaults to False
    :param msgcolor: color for message channel printing, defaults to ``"yellow"``
    :param progress: show a progress bar during planning if the ``progress``
        package is installed, defaults to True
    :param marker: vehicle animation used to display start and goal
        configurations when planning in :math:`\SE{2}`, defaults to a car
        shaped :class:`~roboticstoolbox.mobile.Animations.VehiclePolygon`
    :param seed: seed provided to private random number generator, defaults to None
    :param animate: planners that can animate their operation do so, defaults
        to False
    :param movie: save the animation as a movie file, defaults to None.  Either
        the name of the file or a tuple (filename, frame interval).  Implies
        ``animate``.  Can be overridden by the ``movie`` argument of the
        planner's methods.
    :raises ValueError: ``ndims`` is not given, ``occgrid`` is not a
        :class:`BinaryOccupancyGrid` or a 2D array, or ``start`` or ``goal``
        is not of length ``ndims`` or lies inside an obstacle

    Superclass for all mobile robot motion planners.  Key functionality
    includes:

    - encapsulates an occupancy grid and optionally inflates it
    - validates ``start`` and ``goal`` if given, see :meth:`validate_start_goal`
    - encapsulates a private random number generator with specifiable seed
    - encapsulates state such as start, goal, and the plan
    - provides a message channel for diagnostic output

    Positions, returned paths and plot axes use world coordinates. Grid-based
    planners convert positions to the nearest cell centre internally, using the
    occupancy grid's origin and cell size. A raw array has unit cells and origin
    (0, 0). Angles are in radians.

    The start and goal can be specified in various ways:

    - at constructor time by the arguments ``start`` or ``goal``
    - by assigning the attributes ``start`` or ``goal``
    - at planning time by specifying ``goal`` to the planner's ``plan`` method
    - at query time by specifying ``start`` to the planner's ``query`` method

    The planners do not share one call signature, since the algorithms need
    different things at different times.  The verbs a planner has are the ones
    that make sense for it, and they fall into four families:

    .. list-table::
       :header-rows: 1
       :widths: 30 40 30

       * - Family
         - Planners
         - Verbs
       * - plan for a goal, then query many starts
         - DistanceTransformPlanner, DstarPlanner, RRTPlanner
         - ``plan(goal)``, ``query(start)``
       * - build a map of the space, then query both ends
         - PRMPlanner, LatticePlanner
         - ``plan(npoints)`` or ``plan(iterations)``, ``query(start, goal)``
       * - nothing to plan
         - DubinsPlanner, ReedsSheppPlanner, QuinticPolyPlanner,
           CurvaturePolyPlanner
         - ``query(start, goal)``
       * - reactive
         - Bug2
         - ``run(start, goal)``

    A planner raises ``ValueError`` for invalid arguments and ``NoPathError``
    when a valid problem has no solution.

    .. seealso:: :class:`~roboticstoolbox.mobile.OccGrid.BinaryOccupancyGrid`
    """

    def __init__(
        self,
        occgrid: BinaryOccupancyGrid | NDArray | None = None,
        inflate: float = 0,
        ndims: int | None = None,
        start: ArrayLike2 | ArrayLike3 | None = None,
        goal: ArrayLike2 | ArrayLike3 | None = None,
        verbose: bool = False,
        msgcolor: str = "yellow",
        progress: bool = True,
        marker: Any = None,
        seed: int | None = None,
        animate: bool = False,
        movie: str | tuple[str, float] | None = None,
        **unused: Any,
    ) -> None:
        # called by the constructor of a subclass, so the caller is 3 above
        _warn_unused(unused, stacklevel=4)

        self._occgrid: BinaryOccupancyGrid | None = None
        self._occgrid0: BinaryOccupancyGrid | None = None
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
        self._animate = animate
        self._movie = movie
        self._writer: MovieWriter | None = None

        if occgrid is not None:
            if isinstance(occgrid, np.ndarray) and occgrid.ndim == 2:
                # it's a NumPy array
                self._occgrid = BinaryOccupancyGrid(occgrid)
            elif isinstance(occgrid, BinaryOccupancyGrid):
                self._occgrid = occgrid  # original occgrid for reference
            else:
                raise ValueError("occgrid must be a BinaryOccupancyGrid or a 2D array")

            if inflate > 0:
                self._occgrid0 = self._occgrid.copy()
                self._occgrid.inflate(inflate)
            else:
                self._occgrid0 = self._occgrid

        self._start: NDArray | None = self.validate_endpoint(start)
        self._goal: NDArray | None = self.validate_endpoint(goal)

    def __str__(self) -> str:
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

    def __repr__(self) -> str:
        """
        Compact representation of the planner, same as :meth:`__str__`
        """
        return str(self)

    @property
    def start(self) -> R2 | R3 | None:
        r"""
        Set/get start point or configuration (superclass)

        :getter: Return start point or configuration, None if not set
        :setter: Set start point or configuration, None is ignored
        :raises ValueError: (setter) not of length ``ndims``, or inside an obstacle

        The start is either a point :math:`(x, y)` or a configuration :math:`(x, y, \theta)`.

        .. seealso:: :meth:`goal`
        """
        return self._start

    @start.setter
    def start(self, start: ArrayLike2 | ArrayLike3 | None) -> None:
        if start is not None:
            pv = getvector(start, self._ndims)
            if self.isoccupied(pv):
                raise ValueError("Start location inside obstacle")
            self._start = pv

    @property
    def goal(self) -> R2 | R3 | None:
        r"""
        Set/get goal point or configuration (superclass)

        :getter: Return goal point or configuration, None if not set
        :setter: Set goal point or configuration, None is ignored
        :raises ValueError: (setter) not of length ``ndims``, or inside an obstacle

        The goal is either a point :math:`(x, y)` or a configuration :math:`(x, y, \theta)`.

        .. seealso:: :meth:`start`
        """
        return self._goal

    @goal.setter
    def goal(self, goal: ArrayLike2 | ArrayLike3 | None) -> None:
        if goal is not None:
            pv = getvector(goal, self._ndims)
            if self.isoccupied(pv):
                raise ValueError("Goal location inside obstacle")
            self._goal = pv

    @property
    def verbose(self) -> bool:
        """
        Get verbosity

        :return: verbosity

        If ``verbosity`` print more diagnostic messages to the planner's
        message channel.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool) -> None:
        """
        Set verbosity

        :param v: verbosity

        If ``verbosity`` print more diagnostic messages to the planner's
        message channel.
        """
        self._verbose = v

    @property
    def animate(self) -> bool:
        """
        Get or set animation of the planner's operation

        :getter: True if planners that can animate will do so
        :setter: Set the default for methods that take an ``animate`` argument

        A movie file given to the constructor also implies animation.

        .. seealso:: :meth:`animation_start`
        """
        return self._animate or self._movie is not None

    @animate.setter
    def animate(self, animate: bool) -> None:
        self._animate = animate

    @property
    def movie(self) -> str | tuple[str, float] | None:
        """
        Get or set the movie file for the animation

        :getter: filename, or tuple (filename, frame interval), None for no movie
        :setter: Set the default movie file, None for no movie
        """
        return self._movie

    @movie.setter
    def movie(self, movie: str | tuple[str, float] | None) -> None:
        self._movie = movie

    def animation_start(
        self,
        animate: bool | None = None,
        movie: str | tuple[str, float] | None = None,
        draw: Callable[[], Any] | None = None,
    ) -> bool:
        """
        Begin an animation

        :param animate: animate this call, defaults to the planner's :attr:`animate`
        :param movie: movie file for this call, defaults to the planner's
            :attr:`movie`
        :param draw: function that draws the static background of the animation
        :return: True if the caller should animate

        The planner decides what is drawn and when, this helper owns the pacing
        and recording.  If animating, ``draw`` is called, then the movie file if
        any is opened on the current figure.  The caller then calls
        :meth:`animation_frame` after updating its artists, and
        :meth:`animation_end` when done.

        A ``movie`` implies animation.

        .. seealso:: :meth:`animation_frame` :meth:`animation_end`
        """
        if movie is None:
            movie = self._movie
        if animate is None:
            animate = self._animate
        if movie is not None:
            animate = True

        if not animate:
            return False

        if draw is not None:
            draw()
        self._writer = MovieWriter(movie)
        return True

    def animation_frame(self, pause: float = 0.05) -> None:
        """
        Show the current state of the animation

        :param pause: time to pause for the display to update, defaults to 0.05 s

        Updates the display and adds a frame to the movie, if there is one.

        .. seealso:: :meth:`animation_start`
        """
        if pause > 0:
            plt.pause(pause)
        if self._writer is not None:
            self._writer.add()

    def animation_end(self) -> None:
        """
        End an animation, closing the movie file if there is one

        .. seealso:: :meth:`animation_start`
        """
        if self._writer is not None:
            self._writer.done()
            self._writer = None

    @property
    def random(self) -> np.random.Generator:
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

        .. seealso:: :func:`numpy.random.default_rng`
        """
        return self._private_random

    def random_init(self, seed: int | None = None) -> None:
        """
        Initialize private random number generator

        :param seed: random number seed, defaults to value given to constructor

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

    @property
    def occgrid(self) -> BinaryOccupancyGrid | None:
        """
        Occupancy grid

        :return: occupancy grid used for planning, None if there is none

        Returns the occupancy grid that was optionally inflated at constructor time.

        .. seealso:: :meth:`validate_endpoint` :meth:`~roboticstoolbox.mobile.PlannerBase.isoccupied`
        """
        return self._occgrid

    def isoccupied(self, p: ArrayLike2 | ArrayLike3) -> bool:
        """
        Test if point is occupied (superclass)

        :param p: world coordinate (x, y), or configuration (x, y, :math:`\theta`)
        :return: occupancy status of corresponding grid cell

        The world coordinate is transformed and the status of the occupancy
        grid cell is returned.  If the point lies outside the bounds of
        the occupancy grid return True (obstacle)

        If there is no occupancy grid this function always returns False (free).

        .. seealso:: :meth:`occgrid` :meth:`validate_endpoint` ``isoccupied``
        """
        if self.occgrid is None:
            return False
        else:
            return self.occgrid.isoccupied(p[:2])

    def validate_endpoint(
        self, p: ArrayLike2 | ArrayLike3 | None, dtype: Any = None
    ) -> R2 | R3 | None:
        """
        Validate start or goal point

        :param p: the point, or None
        :param dtype: data type for point coordinates, defaults to None
        :raises ValueError: point is of the wrong length, or is inside an obstacle
        :return: point as a NumPy array of specified dtype, None if ``p`` is None

        The coordinate is tested to be a free cell in the occupancy grid.

        .. seealso:: :meth:`~roboticstoolbox.mobile.PlannerBase.isoccupied` :meth:`occgrid`
        """
        if p is not None:
            pv = getvector(p, self._ndims, dtype=dtype)
            if self.isoccupied(pv):
                raise ValueError("Point is inside obstacle")
            return pv
        return None

    def progress_start(self, n: int) -> None:
        """
        Initialize a progress bar (superclass)

        :param n: Number of iterations in the operation

        Create a progress bar for an operation that has ``n`` steps, for
        example::

            planner.progress_start(100)
            for i in range(100):
                # ...
                planner.progress_next()
            planner.progress_end()

        .. warning:: Requires that the ``progress`` package is installed.

        .. seealso:: :meth:`progress_next` :meth:`progress_end`
        """
        if self._progress:
            self._bar = FillingCirclesBar(
                self.__class__.__name__, max=n, suffix="%(percent).1f%% - %(eta)ds"
            )

    def progress_next(self) -> None:
        """
        Increment a progress bar (superclass)

        Advance a progress bar by one step, for
        example::

            planner.progress_start(100)
            for i in range(100):
                # ...
                planner.progress_next()
            planner.progress_end()

        .. warning:: Requires that the ``progress`` package is installed.

        .. seealso:: :meth:`progress_start` :meth:`progress_end`
        """
        if self._progress:
            self._bar.next()

    def progress_end(self) -> None:
        """
        Finalize a progress bar  (superclass)

        Remove/cleanup a progress bar, for
        example::

            planner.progress_start(100)
            for i in range(100):
                # ...
                planner.progress_next()
            planner.progress_end()

        .. warning:: Requires that the ``progress`` package is installed.

        .. seealso:: :meth:`progress_start` :meth:`progress_next`
        """
        if self._progress:
            self._bar.finish()

    def validate_start_goal(
        self,
        start: ArrayLike2 | ArrayLike3 | None = None,
        goal: ArrayLike2 | ArrayLike3 | None = None,
        dtype: Any = None,
    ) -> None:
        r"""
        Validate and store the start and goal

        :param start: start position :math:`(x, y)` or configuration
            :math:`(x, y, \theta)`, defaults to the value already set
        :param goal: goal position :math:`(x, y)` or configuration
            :math:`(x, y, \theta)`, defaults to the value already set
        :param dtype: data type for the coordinates, defaults to None
        :raises ValueError: ``start`` or ``goal`` is of the wrong length or is
            inside an obstacle

        Each given value is tested against the planner's dimensionality and the
        occupancy grid, and then stored as :attr:`start` or :attr:`goal`.  A
        value of None leaves the existing value unchanged.  If either is invalid
        neither is stored.

        .. seealso:: :meth:`validate_endpoint` :attr:`start` :attr:`goal`
        """
        start_ = self.validate_endpoint(start, dtype=dtype)
        goal_ = self.validate_endpoint(goal, dtype=dtype)
        if start_ is not None:
            self._start = start_
        if goal_ is not None:
            self._goal = goal_

    def plot(
        self,
        path: NDArray | None = None,
        path_style: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        path_reverse_style: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        configspace: bool = False,
        unwrap: bool = True,
        direction: ArrayLike | None = None,
        background: bool = True,
        path_marker: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        path_marker_reverse: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        start_style: dict[str, Any] | None = None,
        goal_style: dict[str, Any] | None = None,
        start_vehicle_style: dict[str, Any] | None = None,
        goal_vehicle_style: dict[str, Any] | None = None,
        start: ArrayLike2 | ArrayLike3 | None = None,
        goal: ArrayLike2 | ArrayLike3 | None = None,
        ax: Axes | None = None,
        block: bool | None = None,
        bgargs: dict[str, Any] | None = None,
        *,
        line: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        line_r: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        start_marker: dict[str, Any] | None = None,
        goal_marker: dict[str, Any] | None = None,
        start_vehicle: dict[str, Any] | None = None,
        goal_vehicle: dict[str, Any] | None = None,
        labels: bool = False,
        label_style: dict[str, Any] | None = None,
        labelvalues: bool = False,
        inflated: bool | None = None,
        colorbar: bool | dict[str, Any] | None = None,
        colorbar_style: dict[str, Any] | None = None,
        cmap: str | None = None,
        **unused: Any,
    ) -> Axes:
        r"""
        Plot vehicle path (superclass)

        :param path: path, one point per row, defaults to None
        :type path: ndarray(N, 2) or ndarray(N, 3)
        :param path_style: style for forward motion, one dict of Matplotlib line
            options or a sequence of them drawn on top of each other, defaults
            to :data:`PATH_STYLE`, a dashed yellow line on black
        :param path_reverse_style: style for reverse motion, as ``path_style``,
            defaults to :data:`PATH_REVERSE_STYLE`, a dashed red line on black
        :param configspace: plot path in configuration space, defaults to False
        :param unwrap: for configuration space plot unwrap :math:`\theta` so
            there are no discontinuities at :math:`\pm \pi`, defaults to True
        :param direction: travel direction associated with each point on path,
            either >0 or <0, defaults to None (all forward)
        :type direction: array_like(N)
        :param background: plot occupancy grid if present, default True
        :param start_style: Matplotlib options for the start point, merged over
            :data:`START_STYLE`
        :param goal_style: Matplotlib options for the goal point, merged over
            :data:`GOAL_STYLE`
        :param start_vehicle_style: options for the vehicle drawn at the start
            configuration, merged over :data:`START_VEHICLE_STYLE`
        :param goal_vehicle_style: options for the vehicle drawn at the goal
            configuration, merged over :data:`GOAL_VEHICLE_STYLE`
        :param start: start position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to value previously set
        :param goal: goal position :math:`(x, y)` or configuration :math:`(x, y, \theta)`, defaults to value previously set.  The planner's own goal is not changed.
        :param ax: axes to plot into, defaults to a new axes
        :param block: ``None`` leaves showing the figure to the environment (no
            ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
            it and waits until it is closed; ignored by notebook backends,
            defaults to None
        :param bgargs: arguments passed to :meth:`~roboticstoolbox.mobile.PlannerBase.plot_bg`, defaults to None
        :param labels: write the distance value in each cell of the distance
            field, defaults to False.  Useful for small grids, see
            :meth:`~roboticstoolbox.mobile.PlannerBase.plot_bg`.
        :param label_style: Matplotlib text options for ``labels``
        :param labelvalues: deprecated, use ``labels``
        :param inflated: show the inflated obstacles, passed to :meth:`~roboticstoolbox.mobile.PlannerBase.plot_bg`
        :param colorbar: show a colorbar for the distance field, passed to
            :meth:`~roboticstoolbox.mobile.PlannerBase.plot_bg`
        :param colorbar_style: options for the colorbar, passed to
            :meth:`~roboticstoolbox.mobile.PlannerBase.plot_bg`
        :param cmap: colormap for the distance field, passed to :meth:`~roboticstoolbox.mobile.PlannerBase.plot_bg`
        :param path_marker: deprecated, use ``path_style``
        :param path_marker_reverse: deprecated, use ``path_reverse_style``
        :param line: deprecated, use ``path_style``
        :param line_r: deprecated, use ``path_reverse_style``
        :param start_marker: deprecated, use ``start_style``
        :param goal_marker: deprecated, use ``goal_style``
        :param start_vehicle: deprecated, use ``start_vehicle_style``
        :param goal_vehicle: deprecated, use ``goal_vehicle_style``
        :raises ValueError: ``configspace`` is True but the path is not in configuration space
        :return: the axes plotted into

        Plots the start and goal location/pose if they are specified by
        ``start`` or ``goal`` or were set by the object constructor or
        ``plan`` or ``query`` method.

        If the ``start`` and ``goal`` have length 2, planning in
        :math:`\mathbb{R}^2`, then markers are drawn using styles specified by
        ``start_style`` and ``goal_style``, which are dicts using Matplotlib
        keywords, for example::

            planner.plot(path, start_style=dict(marker='s', color='b'))

        If the ``start`` and ``goal`` have length 3, planning in :math:`\SE{2}`,
        and ``configspace`` is False, then the start and goal configuration are
        displayed as a vehicle, using ``start_vehicle_style`` and
        ``goal_vehicle_style``.

        If ``configspace`` is True then a 3D plot is created and the start and
        goal are indicated by Matplotlib markers specified by the dicts
        ``start_style`` and ``goal_style``.

        Default values are provided for all styles:

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

        Additional arguments ``bgargs`` can be passed through to :meth:`~roboticstoolbox.mobile.PlannerBase.plot_bg`

        If ``path`` is specified it has one column per point and either 2 or 3 rows:

        - 2 rows describes motion in the :math:`xy`-plane and a 2D plot  is created
        - 3 rows describes motion in the :math:`xy\theta`-configuration space. By
          default only the :math:`xy`-plane is plotted unless ``configspace``
          is True in which case motion in :math:`xy\theta`-configuration space
          is shown as a 3D plot.

        If the planner supports bi-directional motion then the ``direction``
        option gives the direction for each point on the path.

        Forward motion segments are drawn using the layers of ``path_style``
        while reverse motion segments are drawn using ``path_reverse_style``.
        These are each one dict, or a sequence of dicts, of Matplotlib plot
        options which can draw an arbitrary number of lines on top of each
        other.  The default::

            path_style = (
                    {color:'black', linewidth:4},
                    {color:'yellow', linewidth:3, dashes:(5,5)}
                )

        will draw a black line of width 4 with a dashed yellow line of width 3
        plotted on top, giving a line of alternating black and yellow dashes.
        A style given by the caller replaces these layers.

        .. deprecated:: 1.5.0
            The arguments ``line``, ``line_r``, ``path_marker``,
            ``path_marker_reverse``, ``start_marker``, ``goal_marker``,
            ``start_vehicle`` and ``goal_vehicle`` are deprecated, use
            ``path_style``, ``path_reverse_style``, ``start_style``,
            ``goal_style``, ``start_vehicle_style`` and ``goal_vehicle_style``.
            ``path_marker`` and ``path_marker_reverse`` were accepted but had
            no effect.

        .. seealso:: :meth:`~roboticstoolbox.mobile.PlannerBase.plot_bg` ``plot_poly``

        .. note:: Unknown keyword arguments are ignored, with a
            ``DeprecationWarning``, as they were silently by RTB 1.4.4 and
            earlier.

        .. deprecated:: 1.5.0
            ``labelvalues`` is deprecated, use ``labels``.  It is the name used
            in the book, where it had no effect.

        .. versionchanged:: 1.5.0
            ``block=False`` shows the figure without waiting, as documented. It used to do nothing,
            the same as ``None``.
        """
        _warn_unused(unused)
        # deprecated names, mapped to the new ones
        for old, new, value in (
            ("line", "path_style", line),
            ("path_marker", "path_style", path_marker),
            ("line_r", "path_reverse_style", line_r),
            ("path_marker_reverse", "path_reverse_style", path_marker_reverse),
            ("start_marker", "start_style", start_marker),
            ("goal_marker", "goal_style", goal_marker),
            ("start_vehicle", "start_vehicle_style", start_vehicle),
            ("goal_vehicle", "goal_vehicle_style", goal_vehicle),
        ):
            if value is None:
                continue
            warnings.warn(
                f"{old} is deprecated, use {new} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if new == "path_style" and path_style is None:
                path_style = value
            elif new == "path_reverse_style" and path_reverse_style is None:
                path_reverse_style = value
            elif new == "start_style" and start_style is None:
                start_style = value  # type: ignore[assignment]
            elif new == "goal_style" and goal_style is None:
                goal_style = value  # type: ignore[assignment]
            elif new == "start_vehicle_style" and start_vehicle_style is None:
                start_vehicle_style = value  # type: ignore[assignment]
            elif new == "goal_vehicle_style" and goal_vehicle_style is None:
                goal_vehicle_style = value  # type: ignore[assignment]

        # the caller's styles are merged over the defaults, and are not modified
        start_marker_style = START_STYLE | (start_style or {})
        goal_marker_style = GOAL_STYLE | (goal_style or {})
        line_layers = _layers(PATH_STYLE if path_style is None else path_style)
        line_r_layers = _layers(
            PATH_REVERSE_STYLE if path_reverse_style is None else path_reverse_style
        )
        start_vehicle_options = START_VEHICLE_STYLE | (start_vehicle_style or {})
        goal_vehicle_options = GOAL_VEHICLE_STYLE | (goal_vehicle_style or {})
        bgargs = dict(bgargs or {})
        if labelvalues:
            warnings.warn(
                "labelvalues is deprecated, use labels instead",
                DeprecationWarning,
                stacklevel=2,
            )
            labels = True
        if labels:
            bgargs["labels"] = True
            bgargs.setdefault("label_style", label_style)
        # arguments of the background that are given here rather than in bgargs
        for bg_name, bg_value in (
            ("inflated", inflated),
            ("colorbar", colorbar),
            ("colorbar_style", colorbar_style),
            ("cmap", cmap),
        ):
            if bg_value is not None:
                bgargs.setdefault(bg_name, bg_value)

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

                        for style in line_layers if dir > 0 else line_r_layers:
                            ax.plot(path[:k, 0], path[:k, 1], zorder=9, **style)

                        if len(change) == 0:
                            break
                        direction = direction[k - 1 :]
                        direction[0] = direction[1]
                        path = path[k - 1 :, :]

                else:
                    # forward motion only
                    for style in line_layers:
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

                        for style in line_layers if dir > 0 else line_r_layers:
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
                    for style in line_layers:
                        ax.plot(path[:, 0], path[:, 1], theta, **style)

        # mark start and goal if requested
        # a start or goal given here is only for this plot, the planner's own
        # start and goal are not changed
        start_pt = self.validate_endpoint(start) if start is not None else self.start
        goal_pt = self.validate_endpoint(goal) if goal is not None else self.goal

        if ndims == 2 and self._ndims == 2:
            # proper 2d plot
            if start_pt is not None:
                ax.plot(start_pt[0], start_pt[1], label="start", **start_marker_style)
            if goal_pt is not None:
                ax.plot(goal_pt[0], goal_pt[1], label="goal", **goal_marker_style)

        elif ndims == 2 and self._ndims == 3:
            # 2d projection of 3d plot, show start/goal configuration
            scale = axes_get_scale(ax) / 10

            if self.marker is None:
                self.marker = VehiclePolygon(shape="car", scale=scale)

            if start_pt is not None:
                self.marker.plot(start_pt, **start_vehicle_options)
            if goal_pt is not None:
                self.marker.plot(goal_pt, **goal_vehicle_options)

        elif ndims == 3:
            # 3d plot

            if start_pt is not None:
                ax.plot(
                    start_pt[0],
                    start_pt[1],
                    start_pt[2],
                    label="start",
                    **start_marker_style,
                )
            if goal_pt is not None:
                if path is not None and unwrap:
                    theta_goal = theta[-1]
                else:
                    theta_goal = goal_pt[2]
                ax.plot(
                    goal_pt[0],
                    goal_pt[1],
                    theta_goal,
                    label="goal",
                    **goal_marker_style,
                )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if ndims == 2:
            ax.set_aspect("equal")
        else:
            ax.set_zlabel(r"$\theta$")  # type: ignore[union-attr]

        apply_block(block)

        return ax

    def _qmarker(self, shape: str) -> NDArray | None:
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
        return None

    def plot_bg(
        self,
        distance: NDArray | None = None,
        cmap: str = "gray",
        ax: Axes | None = None,
        inflated: bool = True,
        colorbar: bool | dict[str, Any] = True,
        block: bool | None = None,
        colorbar_style: dict[str, Any] | None = None,
        labels: bool = False,
        label_style: dict[str, Any] | None = None,
        labelvalues: bool = False,
        **unused: Any,
    ) -> None:
        """
        Plot background (superclass)

        :param distance: override distance field, defaults to None.  The array
            is not modified.
        :type distance: ndarray(N,M)
        :param cmap: Specify a colormap for the distance field, defaults to 'gray'
        :type cmap: str or Colormap
        :param ax: axes to plot into, defaults to the current axes
        :param inflated: show the inflated obstacles, defaults to True
        :param colorbar: show a colorbar for the distance field, defaults to True
        :param block: ``None`` leaves showing the figure to the environment (no
            ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
            it and waits until it is closed; ignored by notebook backends,
            defaults to None
        :param colorbar_style: options for the colorbar, passed to
            :meth:`matplotlib.figure.Figure.colorbar`, defaults to None.  The
            label defaults to "Distance".
        :param labels: write the value in each cell of the distance field,
            defaults to False
        :param labelvalues: deprecated, use ``labels``
        :param label_style: Matplotlib text options for the values, merged over
            red, centered ``fontsize`` 8 text, as in the book, with a thin
            white outline (``path_effects``) so that it is legible on any gray.
            The axes are set to show every cell whole.

        .. deprecated:: 1.5.0
            Giving a dict as ``colorbar`` is deprecated, use ``colorbar=True``
            with ``colorbar_style``.

        Displays the background which is either the occupancy grid or a distance
        field.  The distance field encodes the distance of a point from the goal, small
        distance is dark, a large distance is bright.

        If the planner has an occupancy grid then that will be displayed with:
            - free cells in white
            - occupied cells in red
            - inflated occupied cells in pink

        If distance is provided, or the planner has a distancemap attribute
        the distance field will be used as the background and obstacle cells
        (actual or inflated) will be shown in red. A colorbar is added.

        .. note:: Unknown keyword arguments are ignored, with a
            ``DeprecationWarning``, as they were silently by RTB 1.4.4 and
            earlier.

        .. deprecated:: 1.5.0
            ``labelvalues`` is deprecated, use ``labels``.  It is the name used
            in the book, where it had no effect.

        """
        _warn_unused(unused)
        if labelvalues:
            warnings.warn(
                "labelvalues is deprecated, use labels instead",
                DeprecationWarning,
                stacklevel=2,
            )
            labels = True
        occgrid = self._occgrid
        if occgrid is None:
            return

        if distance is None and hasattr(self, "distancemap"):
            distance = self.distancemap

        # The original map owns the geometry; the planner supplies its overlays.
        assert self._occgrid0 is not None  # nosec B101: narrows the type for mypy
        self._occgrid0.plot(
            field=distance,
            inflated=occgrid.grid if self._inflate > 0 and inflated else None,
            cmap=cmap if distance is not None else None,
            ax=ax,
            colorbar=colorbar,
            colorbar_style=colorbar_style,
            labels=labels,
            label_style=label_style,
            block=block,
        )

    def message(self, s: str, color: str | None = None) -> None:
        """
        Print message to message channel

        :param s: message to print
        :param color: color to print it, defaults to color specified at
            constructor time.

        The message is printed only if :attr:`verbose` is True.

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
    def __init__(
        self,
        filename: str | tuple[str, float] | list[Any] | None = None,
        interval: float = 0.1,
        fig: Any = None,
    ) -> None:
        """
        Save animation as a movie file

        :param filename: name of movie file, or tuple containing filename and
            frame interval
        :type filename: str or tuple(str, float)
        :param interval: frame interval, defaults to 0.1
        :param fig: figure to record for the movie, defaults to the current figure

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

    def add(self) -> None:
        """
        Add frame to the movie
        """
        if self.writer is not None:
            self.writer.grab_frame()

    def done(self) -> None:
        """
        Finish the movie and close the file
        """
        if self.writer is not None:
            self.writer.finish()
            self.writer = None
