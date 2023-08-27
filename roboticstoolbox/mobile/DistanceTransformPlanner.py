"""
@Author: Peter Corke, original MATLAB code and Python version
@Author: Kristian Gibson, initial MATLAB port
"""
from numpy import disp
from scipy import integrate
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *
from spatialmath.pose2d import SE2
from spatialmath import base
from scipy.ndimage import *
import matplotlib.pyplot as plt
from matplotlib import cm
from roboticstoolbox.mobile.PlannerBase import PlannerBase


class DistanceTransformPlanner(PlannerBase):
    r"""
    Distance transform path planner

    :param occgrid: occupancy grid
    :type occgrid: :class:`BinaryOccGrid` or ndarray(h,w)
    :param metric: distane metric, one of: "euclidean" [default], "manhattan"
    :type metric: str optional
    :param kwargs: common planner options, see :class:`PlannerBase`

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
    comprises a set of 4- or -way connected points in adjacent cells.  Also
    known as the wavefront, grassfire or brushfire planning algorithm.

    The map is described by a 2D occupancy ``occgrid`` whose elements are zero
    if traversable of nonzero if untraversable, ie. an obstacle.

    The cells are assumed to be unit squares. Crossing the cell horizontally or
    vertically is a travel distance of 1, and for the Euclidean distance
    measure, the crossing the cell diagonally is a distance of :math:`\sqrt{2}`.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import DistanceTransformPlanner
        >>> import numpy as np
        >>> simplegrid = np.zeros((6, 6));
        >>> simplegrid[2:5, 3:5] = 1
        >>> dx = DistanceTransformPlanner(simplegrid, goal=(1, 1), distance="manhattan");
        >>> dx.plan()
        >>> path = dx.query(start=(5, 4))
        >>> print(path.T)

    .. warning:: The distance planner is iterative and implemented in Python, will
        be slow for very large occupancy grids.

    :author: Peter Corke

    :seealso: :meth:`plan` :meth:`query` :class:`PlannerBase`
    """

    def __init__(self, occgrid=None, metric="euclidean", **kwargs):

        super().__init__(occgrid=occgrid, ndims=2, **kwargs)
        self._metric = metric
        self._distancemap = None

    @property
    def metric(self):
        """
        Get the distance metric

        :return: Get the metric, either "euclidean" or "manhattan"
        :rtype: str
        """
        return self._metric

    @property
    def distancemap(self):
        """
        Get the distance map

        :return: distance map
        :rtype: ndarray(h,w)

        The 2D array, the same size as the passed occupancy grid, has elements
        equal to nan if they contain an obstacle, otherwise the minimum
        obstacle-free distance to the goal using the particular distance metric.
        """
        return self._distancemap

    def __str__(self):
        s = super().__str__()
        s += f"\n  Distance metric: {self._metric}"
        if self.distancemap is not None:
            s += ", Distance map: computed "
        else:
            s += ", Distance map: empty "

        return s

    def plan(self, goal=None, animate=False, summary=False):
        r"""
        Plan path using distance transform

        :param goal: goal position :math:`(x, y)`, defaults to previously set value
        :type goal: array_like(2), optional

        Compute the distance transform for all non-obstacle cells, which is the
        minimum obstacle-free distance to the goal using the particular distance
        metric.

        :seealso: :meth:`query`
        """
        # show = None
        # if animate:
        #     show = 0.05
        # else:
        #     show = 0

        if goal is not None:
            self.goal = goal

        if self._goal is None:
            raise ValueError("No goal specified here or in constructor")

        self._distancemap = distancexform(
            self.occgrid.grid,
            goal=self._goal,
            metric=self._metric,
            animate=animate,
            summary=summary,
        )

    def next(self, position):
        """
        Find next point on the path

        :param position: current robot position
        :type position: array_like(2)
        :raises RuntimeError: no plan has been computed
        :return: next robot position
        :rtype: ndarray(2)

        Return the robot position that is one step closer to the goal. Called
        by :meth:`query` to find a path from start to goal.

        :seealso: :meth:`plan` :meth:`query`
        """
        if self.distancemap is None:
            Error("No distance map computed, you need to plan.")

        directions = np.array(
            [
                # dy  dx
                [-1, -1],
                [0, -1],
                [1, -1],
                [-1, 0],
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=int,
        )

        x = int(position[0])
        y = int(position[1])

        min_dist = np.inf
        for d in directions:
            try:
                if self._distancemap[y + d[0], x + d[1]] < min_dist:
                    min_dir = d
                    min_dist = self._distancemap[y + d[0], x + d[1]]
            except:
                # come here if the neighbouring cell is outside the map bounds
                # raise RuntimeError(f"Unexpected error finding next min dist at {d}")
                pass

        if np.isinf(min_dist):
            raise RuntimeError("no minimum found, shouldn't happen")

        x = x + min_dir[1]
        y = y + min_dir[0]

        next = np.r_[x, y]
        if all(next == self._goal):
            return None
        else:
            return next

    def plot_3d(self, path=None, ls=None):
        """
        Plot path on 3D cost surface

        :param path: robot path, defaults to None
        :type path: ndarray(N,2), optional
        :param ls: dictionary of Matplotlib linestyle options, defaults to None
        :type ls: dict, optional
        :return: Matplotlib 3D axes
        :rtype: Axes

        Creates a 3D plot showing distance from the goal as a cost surface.
        Overlays the path if given.

        .. warning:: The visualization is poor because of Matplotlib's poor hidden
            line/surface handling.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        distance = self._distancemap
        X, Y = np.meshgrid(np.arange(distance.shape[1]), np.arange(distance.shape[0]))
        surf = ax.plot_surface(
            X, Y, distance, linewidth=1, antialiased=False  # cmap='gray',
        )

        if path is not None:
            # k = sub2ind(np.shape(self._distancemap), p[:, 1], p[:, 0])
            height = distance[path[:, 1], path[:, 0]]
            ax.plot(path[:, 0], path[:, 1], height, **ls)

        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        plt.show()
        return ax


import numpy as np


def distancexform(occgrid, goal, metric="cityblock", animate=False, summary=False):
    """
    Distance transform for path planning

    :param occgrid: Occupancy grid, 0 is free, >0 is occupied/obstacle
    :type occgrid: NumPy ndarray
    :param goal: Goal position (x,y)
    :type goal: 2-element array-like
    :param metric: distance metric, defaults to 'cityblock'
    :type metric: str, optional
    :param animate: animate the iterations of the algorithm
    :return: Distance transform matrix
    :rtype: NumPy ndarray

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
    distance[goal[1], goal[0]] = 0  # assign zero to goal

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

    # get ready to iterate
    count = 0
    ninf = np.inf  # number of infinities in the map

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
                ax = plt.gca()
                plt.pause(0.001)
                cmap = cm.get_cmap("gray")
                cmap.set_bad("red")
                cmap.set_over("white")
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

        ninf = ninfnow

    if summary:
        print(f"{count:d} iterations, {ninf:d} unreachable cells")
    return distance


def grassfire_step(G, D):

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


if __name__ == "__main__":
    pass

    # # make a simple map, as per the MOOCs
    # occgrid = np.zeros((10,10))
    # occgrid[3:6,2:7] = 1
    # occgrid[4,3:5] = 0  # hole in the obstacle
    # # occgrid[7:8,7] = 1  # extra bit

    # print(occgrid)
    # print()

    # goal=[5,7]
    # np.set_printoptions(precision=1)
    # dx = distancexform(occgrid, goal, metric='Euclidean')
    # print(dx)

    # from roboticstoolbox import DistanceTransformPlanner, rtb_loadmat

    # house = rtb_loadmat('data/house.mat')
    # floorplan = house['floorplan']
    # places = house['places']

    # dx = DistanceTransformPlanner(floorplan)
    # print(dx)
    # dx.goal = [1,2]
    # dx.plan(places.kitchen)
    # path = dx.query(places.br3)
    # dx.plot(path, block=True)
