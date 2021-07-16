"""
Python RRT
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing

Not ready for use yet.
"""
from scipy import integrate
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *
from spatialmath.pose2d import SE2
from spatialmath.base import *
from spatialmath.base.animate import *
from scipy.ndimage import *
from matplotlib import cm
from roboticstoolbox.mobile.Planner import Planner
from pgraph import DGraph, DVertex
from roboticstoolbox.mobile.DubinsPlanner import path_planning

class RRTVertex(DVertex):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vel = 0
        self.path = []


class RRTPlanner(Planner):

    def __init__(self, vehicle=None,
                 private=False, reset=False,
                 npoints=100, simtime=0.5, speed=1, root=(0, 0, 0),
                 rev_cost=1, **kwargs):

        super().__init__(**kwargs)
        self._graph = None
        self._vehicle = vehicle
        self._npoints = npoints
        self._simtime = simtime
        self._speed = speed
        self._root = root
        self._rev_cost = rev_cost
        self._v_goal = None
        self._v_start = None

    @property
    def npoints(self):
        return self._npoints

    @property
    def graph(self):
        return self._graph

    @property
    def vehicle(self):
        return self._vehicle

    @property
    def v_goal(self):
        return self._v_goal

    @property
    def v_start(self):
        return self._v_start

    def plan(self, progress=True, samples=False, goalsampling=0.1, root=None, goal=None, maxlen=np.Inf):

        self.message('create the graph')
        self._graph = DGraph(metric=lambda x: np.linalg.norm(x[:2]))

        # if np.empty(self._root):
        #     Error('no root node specified')

        # if not isvector(self._root, 3):
        #     Error('root must be 3-vector')

        assert(not self.isoccupied(root[:2]), 'root node cell is occupied')

        vroot = RRTVertex(root)
        v_root = self._graph.add_vertex(vroot)

        npoints = 0
        curvature = self.vehicle.curvature_max

        while npoints < self.npoints:
            if goal is not None and self.random.uniform() > goalsampling:
                # pick a random unoccupied point
                while True:
                    x = self.random.uniform(self.occgrid.xmin, self.occgrid.xmax)
                    y = self.random.uniform(self.occgrid.ymin, self.occgrid.ymax)
                    if not self.occgrid.isoccupied((x, y)):
                        break

                theta = self.random.uniform(-np.pi, np.pi)
                rand = np.r_[x, y, theta]
            else:
                # with probability goalsampling give the goal
                rand = goal

            vnear, d = self.graph.closest(rand)

            if d > maxlen:
                continue

            near = vnear.coord

            self.message(f"random point ({rand[:2]}) node {vnear.name}")

            path, length, *_ = self.find_path(near, rand, curvature)

            for p in path:
                if self.isoccupied(p[:2]):
                    self.message('path cross an obstacle')
                    break
            else:
                vnew = RRTVertex(rand)
                vnew.path = path
                self.graph.add_vertex(vnew)
                self.graph.add_edge(vnear, vnew, cost=length)

                npoints += 1

        self.message('graph create done')

    def query(self, start, goal):
        if self.graph.n == 0:
            Error('RTB:PRM:noplan:query: no plan: run the planner')

        self.check_points(start, goal, dim=3)

        # find vertices closest to start and goal
        vstart, _ = self.graph.closest(self.start)
        vgoal, _ = self.graph.closest(self.goal)

        out = self.graph.path_Astar(vstart, vgoal)
        if out is None:
            print('no path found')
            return None

        self.astar = out
        self.message(f"A* path cost {out[1]}")

        path = [self._start]
        path.extend([v.coord for v in out[0]])
        path.append(self._goal)

        return np.array(path)

    def __str__(self):
        s = str(self)
        s += "\nregion: X: " +  str(self._x_range) + " Y:" + str(self._y_range)
        s += "\nsim time: " + str(self._simtime)
        s += "\nspeed: " + str(self._speed)
        s += "\nGraph:"
        s += "\n" + str(self._graph)
        if self.vehicle is not None:
            s += "\n Vehicle: " + self.vehicle
        return s

    def find_path(self, x0=None, xg=None, curvature=1, N=None):

        path = path_planning(x0, xg, curvature, step_size=0.1)
        return path


if __name__ == "__main__":

    from roboticstoolbox.mobile.Vehicle import Bicycle

    # start and goal position
    start = (10, 10, 0)
    goal = (50, 50, 0)
    robot_size = 5.0  # [m]

    occgrid = np.zeros((100, 100))
    occgrid[20:40, 15:30] = 1

    vehicle = Bicycle(steer_max=0.4, l=2)

    rrt = RRTPlanner(occgrid=occgrid, vehicle=vehicle, verbose=False, inflate=5, seed=0)

    rrt.plan(root=start, goal=goal)
    path = rrt.query(start, goal)
    print(path)

    path = []
    for vertex in rrt.astar[0]:
        path.extend(vertex.path)
    path = np.array(path)
    rrt.plot(path[:,:2], path_marker=dict(zorder=8, linewidth=2, markersize=6, color='k'))
    for vertex in rrt.graph:
        path = np.array(vertex.path)
        if path.size > 0:
            plt.plot(path[:,0], path[:,1])

    plt.show(block=True)