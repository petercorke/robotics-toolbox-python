"""
Python PRM
@Author: Peter Corke, original MATLAB code and Python version
@Author: Kristian Gibson, initial MATLAB port
"""
# from multiprocessing.sharedctypes import Value
# from numpy import disp
# from scipy import integrate
# from spatialmath.base.animate import Animate
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *

# from spatialmath.pose2d import SE2
# from spatialmath.base import animate
from scipy.ndimage import *
from matplotlib import cm, pyplot as plt
from roboticstoolbox.mobile.PlannerBase import PlannerBase
from pgraph import UGraph

# from progress.bar import FillingCirclesBar


class PRMPlanner(PlannerBase):
    r"""
    Distance transform path planner

    :param occgrid: occupancy grid
    :type occgrid: :class:`BinaryOccGrid` or ndarray(h,w)
    :param npoints: number of random points, defaults to 100
    :type npoints: int, optional
    :param dist_thresh: distance threshold, a new point is only added to the
        roadmap if it is closer than this distance to an existing vertex,
        defaults to None
    :type dist_thresh: float, optional
    :param Planner: probabilistic roadmap path planner
    :param kwargs: common planner options, see :class:`PlannerBase`

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

    def __init__(self, occgrid=None, npoints=100, dist_thresh=None, **kwargs):
        super().__init__(occgrid, ndims=2, **kwargs)

        if dist_thresh is None:
            self._dist_thresh = 0.3 * self.occgrid.maxdim

        self._npoints = npoints
        # self._npoints0 = npoints
        self._dist_thresh0 = self.dist_thresh
        self._graph = None
        self._v_goal = None
        self._v_start = None
        self._local_goal = None
        self._local_path = None
        self._v_path = None
        self._g_path = None

    def __str__(self):
        s = super().__str__()
        if self.graph is not None:
            s += "\n  " + str(self.graph)
        return s

    @property
    def npoints(self):
        """
        Number of points in the roadmap

        :return: Number of points
        :rtype: int
        """
        return self._npoints

    @property
    def dist_thresh(self):
        """
        Distance threshold

        :return: distance threshold
        :rtype: float

        Edges are created between points if the distance between them is less
        than this value.
        """
        return self._dist_thresh

    # @property
    # def npoints0(self):
    #     return self._npoints0

    # @property
    # def dist_thresh0(self):
    #     return self._dist_thresh0

    @property
    def graph(self):
        """
        Roadmap graph

        :return: roadmap as an undirected graph
        :rtype: :class:`pgraph.UGraph` instance
        """
        return self._graph

    def _create_roadmap(self, npoints, dist_thresh, animate=None):
        # a = Animate(animate, fps=5)
        self.progress_start(npoints)

        x = None
        y = None
        for j in range(npoints):
            # find a random point in freespace
            while True:
                # pick a random unoccupied point
                x = self.random.uniform(self.occgrid.xmin, self.occgrid.xmax)
                y = self.random.uniform(self.occgrid.ymin, self.occgrid.ymax)
                if not self.occgrid.isoccupied((x, y)):
                    break

            # add it as a vertex to the graph
            vnew = self.graph.add_vertex([x, y])

        # compute distance between vertices
        for vertex in self.graph:
            # find distance from vertex to all other vertices
            distances = []
            for othervertex in self.graph:
                # skip self
                if vertex is othervertex:
                    continue
                # add (distance, vertex) tuples to list
                distances.append((vertex.distance(othervertex), othervertex))

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
                    self.graph.add_edge(vertex, othervertex, cost=distance)
            self.progress_next()

        self.progress_end()
        # if animate is not None:
        #     self.plot()
        #     if not np.empty(movie):
        #         a.add()

    def _test_path(self, v1, v2, npoints=None):
        # vector from v1 to v2
        dir = v2.coord - v1.coord

        # figure the number of points, essentially the line length
        # TODO: should delegate this test to the OccGrid object and do it
        # world units
        if npoints is None:
            npoints = int(round(np.linalg.norm(dir)))

        # test each point along the line from v1 to v2
        for s in np.linspace(0, 1, npoints):
            if self.occgrid.isoccupied(v1.coord + s * dir):
                return False
        return True

    def plan(self, npoints=None, dist_thresh=None, animate=None):
        """
        Plan PRM path

        :param npoints: number of random points, defaults to ``npoints`` given
            to constructor
        :type npoints: int, optional
        :param dist_thresh: distance threshold, defaults to ``dist_thresh`` given
            to constructor
        :type dist_thresh: float, optional
        :param animate: animate the planning algorithm iterations, defaults to False
        :type animate: bool, optional

        Create a probablistic roadmap.  This is a graph connecting points
        randomly selected from the free space of the occupancy grid. Edges are
        created between points if the distance between them is less than
        ``dist_thresh``.

        The roadmap is a pgraph :obj:`~pgraph.PGraph.UGraph`
        :class:`~pgraph.UGraph`
        :class:`~pgraph.PGraph.UGraph`

        :seealso: :meth:`query` :meth:`graph`
        """

        self.message("create the graph")

        if npoints is None:
            npoints = self.npoints

        if dist_thresh is None:
            dist_thresh = self.dist_thresh

        self._graph = UGraph()
        self._v_path = np.array([])

        self.random_init()  # reset the random number generator
        self._create_roadmap(npoints, dist_thresh, animate)

    def query(self, start, goal, **kwargs):
        """
        Find a path from start to goal using planner

        :param start: start position :math:`(x, y)`, defaults to previously set value
        :type start: array_like(), optional
        :param goal: goal position :math:`(x, y)`, defaults to previously set value
        :type goal: array_like(), optional
        :param kwargs: options passed to :meth:`PlannerBase.query`
        :return: path from start to goal, one point :math:`(x, y)` per row
        :rtype: ndarray(N,2)

        The path is a sparse sequence of waypoints, with variable distance
        between them.

        .. warning:: Waypoints 1 to N-2 are part of the roadmap, while waypoints
            0 and N-1 are the start and goal respectively.  The first and last
            motion segment is not guaranteed to be obstacle free.

        """
        if self.graph.n == 0:
            raise RuntimeError("no plan computed")

        super().query(start=start, goal=goal, next=False, **kwargs)

        # find roadmap vertices closest to start and goal
        vstart, _ = self.graph.closest(self.start)
        vgoal, _ = self.graph.closest(self.goal)

        # find A* path through the roadmap
        out = self.graph.path_Astar(vstart, vgoal)
        if out is None:
            raise RuntimeError("no path found")
        path = [v.coord for v in out[0]]

        path.insert(0, start)  # insert start at head of path
        path.append(goal)  # append goal to end of path

        return np.array(path)

    def plot(self, *args, vertex={}, edge={}, **kwargs):
        """
        Plot PRM path

        :param vertex: vertex style, defaults to {}
        :type vertex: dict, optional
        :param edge: edge style, defaults to {}
        :type edge: dict, optional

        Displays:

        - the planner background (obstacles)
        - the roadmap graph
        - the path

        :seealso: :meth:`UGraph.plot`
        """
        # plot the obstacles and path
        ax = super().plot(*args, **kwargs)
        print("plotted background")
        vertex = {**dict(markersize=4), **vertex}
        edge = {**dict(linewidth=0.5), **edge}

        # add the roadmap graph
        self.graph.plot(text=False, vopt=vertex, eopt=edge, ax=ax)
        print("plotted roadmap")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from roboticstoolbox import *

    house = rtb_load_matfile("data/house.mat")
    floorplan = house["floorplan"]
    places = house["places"]

    occgrid = floorplan.copy()

    prm = PRMPlanner(occgrid=floorplan, seed=0)
    prm.plan(npoints=50)
    prm.plot()

    # start and goal position
    # start = (10, 10)
    # goal = (50, 50)

    # occgrid = np.zeros((100, 100))
    # occgrid[20:40, 15:30] = 1

    # prm = PRMPlanner(occgrid=occgrid, verbose=True)

    # prm.plan()

    # prm.plot()

    # path = prm.query(start, goal)
    # print(path)

    # prm.plot(path, path_marker=dict(zorder=8, linewidth=2, markersize=6, color='k'))
    # prm.plot(ax=plt.gca(), text=False, vertex=dict(markersize=4), edge=dict(linewidth=0.5))
    plt.show(block=True)
