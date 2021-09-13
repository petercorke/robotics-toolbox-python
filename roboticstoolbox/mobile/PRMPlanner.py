"""
Python PRM
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing

Not ready for use yet.
"""
from numpy import disp
from scipy import integrate
from spatialmath.base.animate import Animate
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *
from spatialmath.pose2d import SE2
from spatialmath.base import animate
from scipy.ndimage import *
from matplotlib import cm, pyplot as plt
from roboticstoolbox.mobile.PlannerBase import PlannerBase
from pgraph import UGraph
from progress.bar import FillingCirclesBar


class PRMPlanner(PlannerBase):
    r"""
    Distance transform path planner

    :param occgrid: occupancy grid
    :type curvature: OccGrid or ndarray(w,h)
    :param npoints: number of random points, defaults to 100
    :type npoints: int, optional
    :param dist_thresh: distance threshold, a new point is only added to the
        roadmap if it is closer than this distance to an existing vertex,
        defaults to None
    :type dist_thresh: float, optional
    :param Planner: probabilistic roadmap path planner
    :type Planner: PRMPlanner instance

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 Cartesian space
    Obstacle avoidance   Yes
    Curvature            Discontinuous
    Motion               Forwards only
    ==================   ========================

    Creates a planner that finds the path between two points in the
    plane using forward motion.  The path comprises a set of way points.

    :author: Peter Corke_
    :seealso: :class:`Planner`
    """
    def __init__(self, occ_grid=None, npoints=100, dist_thresh=None, **kwargs):
        super().__init__(ndims=2, **kwargs)

        if dist_thresh is None:
            self._dist_thresh = 0.3 * self.occgrid.maxdim

        self._npoints = npoints
        self._npoints0 = npoints
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
        s += '\n  ' + str(self.graph)
        return s

    @property
    def npoints(self):
        return self._npoints

    @property
    def npoints0(self):
        return self._npoints0

    @property
    def dist_thresh(self):
        return self._dist_thresh

    @property
    def dist_thresh0(self):
        return self._dist_thresh0

    @property
    def graph(self):
        return self._graph

    @property
    def v_goal(self):
        return self._v_goal

    @property
    def v_start(self):
        return self._v_start

    @property
    def local_goal(self):
        return self._local_goal

    @property
    def local_path(self):
        return self._local_path

    @property
    def v_path(self):
        return self._v_path


    def create_roadmap(self, animate=None):
        # a = Animate(animate, fps=5)
        self.progress_start(self.npoints)

        x = None
        y = None
        for j in range(self.npoints):
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
            distances.sort(key=lambda x:x[0])

            # create edges to vertex if permissible
            for distance, othervertex in distances:
                # test if below distance threshold
                if self.dist_thresh is not None and distance > self.dist_thresh:
                    break # sorted into ascending order, so we are done

                # test if obstacle free path connecting them
                if self.test_path(vertex, othervertex):
                    # add an edge
                    self.graph.add_edge(vertex, othervertex, cost=distance)
            self.progress_next()

        self.progress_end()
            # if animate is not None:
            #     self.plot()
            #     if not np.empty(movie):
            #         a.add()

    def test_path(self, v1, v2, npoints=None):
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
        """

        self.message('create the graph')

        if npoints is None:
            npoints = self.npoints0
        if dist_thresh is None:
            dist_thresh = self.dist_thresh0;

        self._npoints = npoints
        self._dist_thresh = dist_thresh

        self._graph = UGraph()
        self._v_path = np.array([])

        self.randinit()  # reset the random number generator
        self.create_roadmap(animate)

    def query(self, start, goal, **kwargs):
        if self.graph.n == 0:
            Error('RTB:PRM:noplan:query: no plan: run the planner')

        super().query(start=start, goal=goal, next=False, **kwargs)

        # find vertices closest to start and goal
        vstart, _ = self.graph.closest(self.start)
        vgoal, _ = self.graph.closest(self.goal)

        out = self.graph.path_Astar(vstart, vgoal)
        if out is None:
            print('no path found')
            return None

        path = [v.coord for v in out[0]]
        path.insert(0, start)
        path.append(goal)
        return np.array(path)


    def next(self, p):
        if all(p[:] == self.goal):
            n = np.array([])
            return n

        if len(self._local_path) == 0:
            if np.empty(self._g_path):
                self._local_path = self.bresenham(p, self._goal)
                self._local_path = self._local_path[1:len(self._local_path), :]
                self._local_goal = np.array([])
            else:
                self._local_goal = self._g_path[0]
                self._g_path = self._g_path[1:len(self._g_path)]

                self._local_path = bresenham(p, self.graph.coord(self._local_goal))
                self._local_path = self._local_path[1:len(self._local_path), :]
                self.graph.highlight_node(self._local_goal)

        n = np.transpose(self._local_path[0, :])
        self._local_pathh = self._local_path[1:len(self._local_path), :]
        return n

    def plot(self, *args, vertex=None, edge=None, **kwargs):
        super().plot(*args, **kwargs)
        if vertex is None:
            vertex=dict(markersize=4)
        if edge is None:
             edge=dict(linewidth=0.5)

        self.graph.plot(text=False, vertex=vertex, edge=edge)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # start and goal position
    start = (10, 10)
    goal = (50, 50)
    robot_size = 5.0  # [m]

    occgrid = np.zeros((100, 100))
    occgrid[20:40, 15:30] = 1

    prm = PRMPlanner(occgrid=occgrid, verbose=True, inflate=5)

    prm.plan()
    path = prm.query(start, goal)
    print(path)

    prm.plot(path, path_marker=dict(zorder=8, linewidth=2, markersize=6, color='k'))
    prm.graph.plot(ax=plt.gca(), text=False, vertex=dict(markersize=4), edge=dict(linewidth=0.5))
    plt.show(block=True)