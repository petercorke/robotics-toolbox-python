"""
Probabilistic roadmap (PRM) path planner

@Author: Peter Corke, original MATLAB code and Python version
@Author: Kristian Gibson, initial MATLAB port
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from pgraph import UGraph
from spatialmath.base.types import ArrayLike2

from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from roboticstoolbox.mobile.PlannerBase import NoPathError, PlannerBase
from roboticstoolbox.tools.types import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: default Matplotlib style of the roadmap vertices and edges, a user's options
#: are merged over these
VERTEX_STYLE: dict[str, Any] = {"markersize": 4}
EDGE_STYLE: dict[str, Any] = {"linewidth": 0.5}


class PRMPlanner(PlannerBase):
    r"""
    Probabilistic roadmap path planner

    :param occgrid: occupancy grid
    :type occgrid: :class:`~roboticstoolbox.mobile.OccGrid.BinaryOccupancyGrid` or ndarray(h,w)
    :param npoints: number of random points, defaults to 100
    :param dist_thresh: distance threshold, an edge is only added to the
        roadmap between two vertices that are closer than this distance,
        defaults to 0.3 times the largest dimension of the occupancy grid
    :param collision_step: spacing in cells of the points tested along a
        candidate edge, defaults to None for the approximate test described
        below
    :param kwargs: common planner options, see :class:`PlannerBase`
    :raises ValueError: there is no occupancy grid, or ``collision_step`` is
        not positive

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 Cartesian space
    Obstacle avoidance   Yes, occupancy grid
    Curvature            Discontinuous
    Motion               Omnidirectional
    ==================   ========================

    Creates a planner that finds the path between two points in the
    plane using omnidirectional motion.  The path comprises a set of way points.

    A candidate edge is added to the roadmap if it is shorter than
    ``dist_thresh`` and a test finds it free of obstacles.  By default the test
    samples the edge once per world unit, with only the endpoints tested for
    edges shorter than about 2.5 units, so a wall one cell thick can be stepped
    over.  If ``collision_step`` is given the edge is instead sampled at
    intervals of at most ``collision_step`` cells, and both endpoints are
    always tested.  A value of 0.5 or less cannot step over a wall one cell
    thick, and smaller values give no gain for the increased cost.  Larger
    values are allowed, and are the user's responsibility.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import PRMPlanner
        >>> import numpy as np
        >>> simplegrid = np.zeros((6, 6));
        >>> simplegrid[2:5, 3:5] = 1
        >>> prm = PRMPlanner(simplegrid);
        >>> prm.plan()
        >>> path = prm.query(start=(5, 4), goal=(1,1))
        >>> print(path.T)

    :author: Peter Corke
    :seealso: :class:`PlannerBase`
    """

    def __init__(
        self,
        occgrid: BinaryOccupancyGrid | NDArray | None = None,
        npoints: int = 100,
        dist_thresh: float | None = None,
        collision_step: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(occgrid, ndims=2, **kwargs)

        if collision_step is not None and collision_step <= 0:
            raise ValueError("collision_step must be positive")
        self._collision_step = collision_step

        if self.occgrid is None:
            raise ValueError("PRMPlanner requires an occupancy grid")
        if dist_thresh is None:
            dist_thresh = 0.3 * self.occgrid.maxdim
        self._dist_thresh = dist_thresh

        self._npoints = npoints
        self._dist_thresh0 = self.dist_thresh
        self._graph: UGraph | None = None

    def __str__(self) -> str:
        """
        Compact representation of the planner

        :return: the base class description, and the roadmap if there is one
        """
        s = super().__str__()
        if self.graph is not None:
            s += "\n  " + str(self.graph)
        return s

    @property
    def npoints(self) -> int:
        """
        Number of points in the roadmap

        :return: Number of points
        """
        return self._npoints

    @property
    def collision_step(self) -> float | None:
        """
        Spacing of the points tested along an edge

        :return: spacing in cells, None for the default approximate test
        """
        return self._collision_step

    @property
    def dist_thresh(self) -> float:
        """
        Distance threshold

        :return: distance threshold

        Edges are created between points if the distance between them is less
        than this value.
        """
        return self._dist_thresh

    @property
    def graph(self) -> UGraph | None:
        """
        Roadmap graph

        :return: roadmap as an undirected graph
        :rtype: :class:`pgraph.UGraph` instance
        """
        return self._graph

    def _create_roadmap(self, npoints: int, dist_thresh: float | None) -> None:
        occgrid = self.occgrid
        graph = self._graph
        if occgrid is None or graph is None:
            raise ValueError("no occupancy grid or graph")

        self.progress_start(npoints)

        for _ in range(npoints):
            # find a random point in freespace
            while True:
                # pick a random unoccupied point
                x = self.random.uniform(occgrid.xmin, occgrid.xmax)
                y = self.random.uniform(occgrid.ymin, occgrid.ymax)
                if not occgrid.isoccupied((x, y)):
                    break

            # add it as a vertex to the graph
            graph.add_vertex([x, y])

        # The roadmap is undirected, so each pair of vertices is considered
        # once, from the earlier vertex.  Adding the edge from each end would
        # make two parallel edges, which are counted twice and drawn as two arcs.
        vertices = list(graph)
        for i, vertex in enumerate(vertices):
            # find distance from vertex to all the later vertices, as
            # (distance, vertex) tuples
            distances = [
                (vertex.distance(othervertex), othervertex)
                for othervertex in vertices[i + 1 :]
            ]

            # sort into ascending distance
            distances.sort(key=lambda x: x[0])

            # create edges to vertex if permissible
            for distance, othervertex in distances:
                # test if below distance threshold
                if dist_thresh is not None and distance > dist_thresh:
                    break  # sorted into ascending order, so we are done

                # test if obstacle free path connecting them
                if self._test_path(vertex, othervertex):
                    # add an edge
                    graph.add_edge(vertex, othervertex, cost=distance)
            self.progress_next()

        self.progress_end()

    def _test_path(self, v1: Any, v2: Any, npoints: int | None = None) -> bool:
        # vector from v1 to v2
        direction = v2.coord - v1.coord

        # figure the number of points, essentially the line length
        # The sampling density is in cells, while graph edges are in world units.
        if npoints is None:
            length = float(np.linalg.norm(direction))
            if self._collision_step is None:
                assert self.occgrid is not None
                npoints = round(length / self.occgrid.cellsize)
            else:
                assert self.occgrid is not None
                step = self._collision_step * self.occgrid.cellsize
                npoints = max(2, math.ceil(length / step) + 1)

        # test each point along the line from v1 to v2
        assert self.occgrid is not None
        for s in np.linspace(0, 1, npoints):
            if self.occgrid.isoccupied(v1.coord + s * direction):
                return False
        return True

    def plan(
        self,
        npoints: int | None = None,
        dist_thresh: float | None = None,
        animate: bool | None = None,
    ) -> None:
        """
        Plan PRM path

        :param npoints: number of random points, defaults to ``npoints`` given
            to constructor
        :param dist_thresh: distance threshold, defaults to ``dist_thresh`` given
            to constructor
        :param animate: deprecated, animating the planning is not implemented

        Create a probabilistic roadmap.  This is a graph connecting points
        randomly selected from the free space of the occupancy grid. Edges are
        created between points if the distance between them is less than
        ``dist_thresh``.

        The roadmap is a :class:`pgraph.UGraph`.

        .. deprecated:: 1.4.5
            ``animate`` was accepted but had no effect.

        :seealso: :meth:`query` :meth:`graph`
        """
        if animate:
            warnings.warn(
                "animate is deprecated, animating the planning is not implemented",
                DeprecationWarning,
                stacklevel=2,
            )

        self.message("create the graph")

        if npoints is None:
            npoints = self.npoints

        if dist_thresh is None:
            dist_thresh = self.dist_thresh

        self._graph = UGraph()

        self.random_init()  # reset the random number generator
        self._create_roadmap(npoints, dist_thresh)

    def query(
        self,
        start: ArrayLike2 | None = None,
        goal: ArrayLike2 | None = None,
    ) -> NDArray:
        """
        Find a path from start to goal using planner

        :param start: start position :math:`(x, y)`, defaults to previously set value
        :param goal: goal position :math:`(x, y)`, defaults to previously set value
        :return: path from start to goal, one point :math:`(x, y)` per row
        :rtype: ndarray(N,2)
        :raises RuntimeError: :meth:`plan` has not been called
        :raises NoPathError: no path exists through the roadmap
        :raises ValueError: there is no start or goal

        The path is a sparse sequence of waypoints, with variable distance
        between them.

        .. warning:: Waypoints 1 to N-2 are part of the roadmap, while waypoints
            0 and N-1 are the start and goal respectively.  The first and last
            motion segment is not guaranteed to be obstacle free.

        """
        if self.graph is None or self.graph.n == 0:
            raise RuntimeError("no plan computed, call plan() first")

        self.validate_start_goal(start, goal)

        # find roadmap vertices closest to start and goal
        vstart, _ = self.graph.closest(self.start)
        vgoal, _ = self.graph.closest(self.goal)

        # find A* path through the roadmap
        out = self.graph.path_Astar(vstart, vgoal)
        if out is None:
            raise NoPathError("no path found through the roadmap")
        path = [v.coord for v in out[0]]

        path.insert(0, self.start)  # insert start at head of path
        path.append(self.goal)  # append goal to end of path

        return np.array(path)

    def plot(
        self,
        *args: Any,
        vertex_style: dict[str, Any] | None = None,
        edge_style: dict[str, Any] | None = None,
        vertex: dict[str, Any] | None = None,
        edge: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot PRM path

        :param args: positional arguments passed to :meth:`PlannerBase.plot`
        :param vertex_style: Matplotlib options for the vertices, merged over
            :data:`VERTEX_STYLE`
        :param edge_style: Matplotlib options for the edges, merged over
            :data:`EDGE_STYLE`
        :param vertex: deprecated, use ``vertex_style``
        :param edge: deprecated, use ``edge_style``
        :param kwargs: keyword arguments passed to :meth:`PlannerBase.plot`
        :raises RuntimeError: :meth:`plan` has not been called
        :return: the axes plotted into

        Displays:

        - the planner background (obstacles)
        - the roadmap graph
        - the path

        .. deprecated:: 1.4.5
            ``vertex`` and ``edge`` are deprecated, use ``vertex_style`` and
            ``edge_style``.

        :seealso: :meth:`PlannerBase.plot` :meth:`UGraph.plot`
        """
        for old, new, value in (
            ("vertex", "vertex_style", vertex),
            ("edge", "edge_style", edge),
        ):
            if value is None:
                continue
            warnings.warn(
                f"{old} is deprecated, use {new} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if new == "vertex_style" and vertex_style is None:
                vertex_style = value
            elif new == "edge_style" and edge_style is None:
                edge_style = value

        if self._graph is None:
            raise RuntimeError("no plan computed, call plan() first")

        # plot the obstacles and path
        ax = super().plot(*args, **kwargs)

        # add the roadmap graph
        self._graph.plot(
            labels=False,
            vertex_opt=VERTEX_STYLE | (vertex_style or {}),
            edge_opt=EDGE_STYLE | (edge_style or {}),
            ax=ax,
        )
        return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from roboticstoolbox import rtb_load_matfile

    house = rtb_load_matfile("data/house.mat")
    floorplan = house["floorplan"]
    places = house["places"]

    prm = PRMPlanner(occgrid=floorplan, seed=0)
    prm.plan(npoints=50)
    path = prm.query(places.br3, places.kitchen)
    prm.plot(path)
    plt.show(block=True)
