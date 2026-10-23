"""
Distance transform path planner

@Author: Peter Corke, original MATLAB code and Python version
@Author: Kristian Gibson, initial MATLAB port
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
from roboticstoolbox.tools.plot import apply_block
import numpy as np
from matplotlib import colormaps
from spatialmath import base
from spatialmath.base.types import ArrayLike2

from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from roboticstoolbox.mobile.PlannerBase import NoPathError, PlannerBase
from roboticstoolbox.tools.types import NDArray

#: the names of the distance metrics, as accepted by :func:`distancexform`
METRICS = ("euclidean", "manhattan", "cityblock")


class DistanceTransformPlanner(PlannerBase):
    r"""
    Distance transform path planner

    :param occgrid: occupancy grid
    :type occgrid: :class:`~roboticstoolbox.mobile.OccGrid.BinaryOccupancyGrid` or ndarray(h,w)
    :param metric: distance metric, one of: "euclidean" [default], "manhattan"
        ("cityblock" is a synonym)
    :param distance: deprecated, use ``metric``
    :param kwargs: common planner options, see :class:`PlannerBase`
    :raises ValueError: ``metric`` is not recognised, or ``metric`` and
        ``distance`` are both given and are different

    .. deprecated:: 1.5.0
        ``distance`` is deprecated, use ``metric``.  It was never an argument
        (it was silently ignored), but code printed in the book uses it.

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 :math:`\mathbb{R}^2`, discrete
    Obstacle avoidance   Yes, occupancy grid
    Curvature            Discontinuous
    Motion               Omnidirectional
    ==================   ========================

    Creates a planner that finds the path between two points in the 2D grid
    using omnidirectional motion and avoiding occupied cells.  The path
    comprises a set of 4- or 8-way connected points in adjacent cells.  Also
    known as the wavefront, grassfire or brushfire planning algorithm.

    The map is described by a 2D occupancy ``occgrid`` whose elements are zero
    if traversable or nonzero if untraversable, ie. an obstacle.

    Positions, paths and distances are in world units. Crossing a cell
    horizontally or vertically travels one ``cellsize``; a diagonal travels
    :math:`\sqrt{2}` times that distance. Endpoints are rounded to cell centres.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import DistanceTransformPlanner
        >>> import numpy as np
        >>> simplegrid = np.zeros((6, 6));
        >>> simplegrid[2:5, 3:5] = 1
        >>> dx = DistanceTransformPlanner(simplegrid, goal=(1, 1), metric="manhattan");
        >>> dx.plan()
        >>> path = dx.query(start=(5, 4))
        >>> print(path.T)

    .. warning:: The distance planner is iterative and implemented in Python, will
        be slow for very large occupancy grids.

    :author: Peter Corke

    .. seealso:: :meth:`plan` :meth:`query` :class:`PlannerBase`
    """

    def __init__(
        self,
        occgrid: BinaryOccupancyGrid | NDArray | None = None,
        metric: str = "euclidean",
        *,
        distance: str | None = None,
        **kwargs: Any,
    ) -> None:
        if distance is not None:
            warnings.warn(
                "distance is deprecated, use metric instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if metric != "euclidean" and metric.lower() != distance.lower():
                raise ValueError(
                    f"metric {metric!r} and distance {distance!r} are different"
                )
            metric = distance
        if metric.lower() not in METRICS:
            raise ValueError(f"unknown metric {metric!r}, must be one of {METRICS}")

        super().__init__(occgrid=occgrid, ndims=2, **kwargs)
        self._metric = metric
        self._distancemap: NDArray | None = None

    @property
    def metric(self) -> str:
        """
        Get the distance metric

        :return: the metric, either "euclidean" or "manhattan"
        """
        return self._metric

    @property
    def distancemap(self) -> NDArray | None:
        """
        Get the distance map

        :return: distance map
        :rtype: ndarray(h,w)

        The 2D array, the same size as the passed occupancy grid, has elements
        equal to nan if they contain an obstacle, otherwise the minimum
        obstacle-free distance to the goal using the particular distance metric.
        """
        return self._distancemap

    def __str__(self) -> str:
        """
        Compact representation of the planner

        :return: the base class description, the metric and whether a distance
            map has been computed
        """
        s = super().__str__()
        s += f"\n  Distance metric: {self._metric}"
        if self.distancemap is not None:
            s += ", Distance map: computed "
        else:
            s += ", Distance map: empty "

        return s

    def plan(
        self,
        goal: ArrayLike2 | None = None,
        animate: bool | None = None,
        summary: bool = False,
    ) -> None:
        r"""
        Plan path using distance transform

        :param goal: goal position :math:`(x, y)`, defaults to previously set value
        :param animate: show the iterations of the algorithm, defaults to the
            planner's :attr:`animate`
        :param summary: print the number of iterations and of unreachable cells,
            defaults to False
        :raises ValueError: no goal is given here or to the constructor, or
            there is no occupancy grid

        Compute the distance transform for all non-obstacle cells, which is the
        minimum obstacle-free distance to the goal using the particular distance
        metric.

        .. seealso:: :meth:`query`
        """
        if goal is not None:
            self.goal = goal

        if self._goal is None:
            raise ValueError("No goal specified here or in constructor")

        if self.occgrid is None:
            raise ValueError("an occupancy grid is required")

        if animate is None:
            animate = self._animate

        self._distancemap = (
            distancexform(
                self.occgrid.grid,
                goal=self.occgrid.w2g(self._goal),
                metric=self._metric,
                animate=animate,
                summary=summary,
            )
            * self.occgrid.cellsize
        )

    def query(
        self,
        start: ArrayLike2 | None = None,
        goal: ArrayLike2 | None = None,
        dtype: Any = None,
        animate: bool | None = None,
        movie: str | tuple[str, float] | None = None,
        *,
        next: bool | None = None,
    ) -> NDArray:
        r"""
        Find a path from start to goal using the distance map

        :param start: start position :math:`(x, y)`, defaults to value specified
            to constructor
        :param goal: goal position :math:`(x, y)`, defaults to value specified
            to constructor
        :param dtype: data type of the returned path, defaults to int for unit
            cells at (0, 0), otherwise float
        :param animate: show the vehicle path, defaults to the planner's
            :attr:`animate`
        :param movie: save the animation as a movie file, filename or tuple
            (filename, frame interval), defaults to the planner's :attr:`movie`.
            Implies ``animate``.
        :param next: deprecated, has no effect
        :raises ValueError: there is no start point
        :raises RuntimeError: no plan has been computed
        :raises NoPathError: the goal cannot be reached from ``start``
        :return: path from start to goal, one point :math:`(x, y)` per row
        :rtype: ndarray(N,2)

        Repeatedly steps to the neighbouring cell with the lowest distance
        until the goal is reached.

        .. deprecated:: 1.5.0
            ``next`` was accepted but had no effect.

        .. seealso:: :meth:`plan`
        """
        if next is not None:
            warnings.warn(
                "next is deprecated and has no effect",
                DeprecationWarning,
                stacklevel=2,
            )

        self.validate_start_goal(start, goal)
        if self._start is None:
            raise ValueError("no start point")

        if self.distancemap is None:
            raise RuntimeError("no plan computed, call plan() first")

        animating = self.animation_start(animate, movie, draw=self.plot)

        assert self.occgrid is not None  # nosec B101: narrows the type for mypy
        robot = self.occgrid.g2w(self.occgrid.w2g(self._start))
        path: list[NDArray] = [robot]

        try:
            while True:
                if animating:
                    plt.plot(robot[0], robot[1], "y.", markersize=12)
                    self.animation_frame()

                # get next point on the path
                step = self.next(robot)

                # are we are done?
                if step is None:
                    break

                robot = step
                path.append(robot)
        finally:
            self.animation_end()

        assert self._goal is not None  # nosec B101: narrows the type for mypy
        endpoint = self.occgrid.g2w(self.occgrid.w2g(self._goal))
        if not np.array_equal(path[-1], endpoint):
            path.append(endpoint)
        if dtype is None:
            dtype = (
                int
                if self.occgrid.cellsize == 1 and np.all(self.occgrid.g2w((0, 0)) == 0)
                else float
            )
        return np.array(path).astype(dtype)

    def next(self, position: ArrayLike2) -> NDArray | None:
        """
        Find next point on the path

        :param position: current robot position in world coordinates
        :raises RuntimeError: no plan has been computed
        :raises NoPathError: there is no neighbouring cell with a finite distance
        :return: next robot position, None if the next point is the goal
        :rtype: ndarray(2)

        Return the robot position that is one step closer to the goal. Called
        by :meth:`query` to find a path from start to goal.

        .. seealso:: :meth:`plan` :meth:`query`
        """
        if self.distancemap is None:
            raise RuntimeError("no plan computed, call plan() first")

        directions = np.array(
            [
                # dy  dx
                [-1, -1],
                [0, -1],
                [1, -1],
                [-1, 0],
                [-1, 1],
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=int,
        )

        assert self.occgrid is not None and self._goal is not None  # nosec B101: narrows the type for mypy
        self.validate_endpoint(position)
        x, y = self.occgrid.w2g(position)
        goal = self.occgrid.w2g(self._goal)
        if np.array_equal((x, y), goal):
            return None

        distancemap = self.distancemap
        rows, columns = distancemap.shape
        min_dist = distancemap[y, x]
        min_dir = None
        for d in directions:
            r = y + d[0]
            c = x + d[1]
            # a neighbouring cell outside the map is not a candidate, note that
            # a negative index would wrap around to the far side of the map
            if 0 <= r < rows and 0 <= c < columns and distancemap[r, c] < min_dist:
                min_dir = d
                min_dist = distancemap[r, c]

        if min_dir is None or np.isinf(min_dist):
            raise NoPathError("the goal is not reachable from this position")

        x = x + min_dir[1]
        y = y + min_dir[0]

        next = np.r_[x, y]
        if all(next == goal):
            return None
        else:
            return self.occgrid.g2w(next)

    def plot_3d(
        self,
        path: NDArray | None = None,
        path_style: dict[str, Any] | None = None,
        block: bool | None = True,
        *,
        ls: dict[str, Any] | None = None,
    ) -> Any:
        """
        Plot path on 3D cost surface

        :param path: robot path, one point per row, defaults to None
        :type path: ndarray(N,2)
        :param path_style: dictionary of Matplotlib line options for the path,
            defaults to None
        :param block: ``None`` leaves showing the figure to the environment (no
            ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
            it and waits until it is closed; ignored by notebook backends,
            defaults to True
        :param ls: deprecated, use ``path_style``
        :raises ValueError: no plan has been computed
        :return: Matplotlib 3D axes
        :rtype: Axes

        Creates a 3D plot showing distance from the goal as a cost surface.
        Overlays the path if given.

        .. deprecated:: 1.5.0
            ``ls`` is deprecated, use ``path_style``.

        .. warning:: The visualization is poor because of Matplotlib's poor hidden
            line/surface handling.
        """
        if ls is not None:
            warnings.warn(
                "ls is deprecated, use path_style instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if path_style is None:
                path_style = ls

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        distance = self._distancemap
        if distance is None:
            raise ValueError("no distance map computed, you need to plan")
        assert self.occgrid is not None  # nosec B101: narrows the type for mypy
        X, Y = np.meshgrid(
            self.occgrid.xmin + np.arange(distance.shape[1]) * self.occgrid.cellsize,
            self.occgrid.ymin + np.arange(distance.shape[0]) * self.occgrid.cellsize,
        )
        ax.plot_surface(
            X,
            Y,
            distance,
            linewidth=1,
            antialiased=False,  # cmap='gray',
        )

        if path is not None:
            # k = sub2ind(np.shape(self._distancemap), p[:, 1], p[:, 0])
            cells = np.array([self.occgrid.w2g(p) for p in path])
            height = distance[cells[:, 1], cells[:, 0]]
            ax.plot(path[:, 0], path[:, 1], height, **(path_style or {}))

        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        apply_block(block)
        return ax


def distancexform(
    occgrid: NDArray,
    goal: ArrayLike2,
    metric: str = "cityblock",
    animate: bool = False,
    summary: bool = False,
) -> NDArray:
    """
    Distance transform for path planning

    :param occgrid: Occupancy grid, 0 is free, >0 is occupied/obstacle
    :type occgrid: ndarray(h,w)
    :param goal: Goal position (x,y)
    :param metric: distance metric, one of "cityblock" (or "manhattan") or
        "euclidean", defaults to 'cityblock'
    :param animate: animate the iterations of the algorithm, defaults to False
    :param summary: print the number of iterations and of unreachable cells,
        defaults to False
    :raises ValueError: ``metric`` is not recognised
    :return: Distance transform matrix
    :rtype: ndarray(h,w)

    Implements the grass/brush fire algorithm to compute, for every reachable
    cell in the occupancy grid, its distance from the goal.

    The result is an array, the same size as the occupancy grid ``occgrid``,
    where each cell contains the distance to the goal according to the chosen
    ``metric``.  In addition:

        - Obstacle cells will be set to ``nan``
        - Unreachable cells, ie. free cells _inside obstacles_ will be set
          to ``inf``

    The cells of the passed occupancy grid are:
        - zero, cell is free or driveable
        - one, cell is an obstacle, or not driveable
    """

    # build the matrix for performing distance transform:
    # - obstacles are nan
    # - other cells are inf
    # - goal is zero

    goal = base.getvector(goal, 2, dtype=int)

    distance = occgrid.astype(np.float32)
    distance[occgrid > 0] = np.nan  # assign nan to obstacle cells
    distance[occgrid == 0] = np.inf  # assign inf to other cells
    distance[int(goal[1]), int(goal[0])] = 0  # assign zero to goal

    # create the appropriate distance matrix D
    if metric.lower() in ("manhattan", "cityblock"):
        # fmt: off
        D = np.array([
                [ np.inf,   1,   np.inf],
                [      1,   0,        1],
                [ np.inf,   1,   np.inf]
            ])
        # fmt: on
    elif metric.lower() == "euclidean":
        r2 = np.sqrt(2)
        # fmt: off
        D = np.array([
                [ r2,   1,   r2],
                [  1,   0,    1],
                [ r2,   1,   r2]
                ])
        # fmt: on
    else:
        raise ValueError(f"unknown metric {metric!r}, must be one of {METRICS}")

    # get ready to iterate
    count = 0
    ninf: float = np.inf  # number of infinities in the map

    h = None
    while True:
        distance = grassfire_step(distance, D)
        distance[occgrid > 0] = np.nan  # reinsert nans for obstacles

        count += 1

        if animate:
            # TODO, needs work to update colorbar and be faster
            display = distance.copy()
            display[np.isinf(display)] = 0
            if h is None:
                plt.figure()
                plt.xlabel("x")
                plt.ylabel("y")
                plt.pause(0.001)
                cmap = colormaps.get_cmap("gray").with_extremes(bad="red", over="white")
                h = plt.imshow(display, cmap=cmap)
                plt.colorbar(label="distance")
            else:
                h.remove()
                h = plt.imshow(display, cmap=cmap)
            plt.pause(0.001)

        ninfnow = np.isinf(distance).sum()  # current number of Infs
        if ninfnow == ninf:
            # stop if the number of Infs left in the map had stopped reducing
            # it may never get to zero if there are unreachable cells in the map
            break

        ninf = int(ninfnow)

    if summary:
        print(f"{count:d} iterations, {ninf:d} unreachable cells")
    return distance


def grassfire_step(G: NDArray, D: NDArray) -> NDArray:
    """
    One step of the grassfire algorithm

    :param G: distance matrix, ``nan`` for an obstacle, ``inf`` for a cell not
        yet reached
    :type G: ndarray(h,w)
    :param D: 3x3 matrix of the distance to each neighbour
    :type D: ndarray(3,3)
    :return: distance matrix in which each cell is the minimum, over its
        neighbours, of the neighbour's distance plus the distance to it
    :rtype: ndarray(h,w)
    """
    # pad with inf
    H = np.pad(G, max(D.shape) // 2, "constant", constant_values=np.inf)
    rows, columns = G.shape

    # inspired by https://landscapearchaeology.org/2018/numpy-loops/
    minimum = np.full(G.shape, np.inf)
    for y in range(3):
        for x in range(3):
            v = H[y : rows + y, x : columns + x] + D[y, x]
            # we use fmin() because it ignores NaNs, behaves like MATLAB min()
            minimum = np.fmin(minimum, v)

    return minimum
