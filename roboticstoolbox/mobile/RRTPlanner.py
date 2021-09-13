import math
from collections import namedtuple
from roboticstoolbox.mobile.OccGrid import PolygonMap

# import rvcprint
from roboticstoolbox import *
import numpy as np
import matplotlib.pyplot as plt

from spatialmath import Polygon2, SE2, base
from roboticstoolbox.mobile.PlannerBase import PlannerBase
from roboticstoolbox.mobile.DubinsPlanner import DubinsPlanner

# from roboticstoolbox.mobile.OccupancyGrid import OccupancyGrid
from pgraph import DGraph


class RRTPlanner(PlannerBase):
    def __init__(
        self,
        map=None,
        vehicle=None,
        curvature=None,
        expand_dis=3.0,
        path_resolution=0.5,
        stepsize=0.2,
        showsamples=False,
        npoints=50,
        **kwargs
    ):

        super().__init__(ndims=2, **kwargs)

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.npoints = npoints
        self.map = map
        self.showsamples = showsamples

        self.g = DGraph(metric="SE2")

        self.vehicle = vehicle
        if curvature is None:
            if vehicle is not None:
                curvature = vehicle.curvature_max
            else:
                curvature = 1

        print("curvature", curvature)
        self.dubins = DubinsPlanner(curvature=curvature, stepsize=stepsize)

        # self.goal_yaw_th = np.deg2rad(1.0)
        # self.goal_xy_th = 0.5

    def plan(self, goal, animation=True, search_until_npoints=True):
        """
        execute planning

        animation: flag for animation on or off
        """
        # TODO use validate
        self.goal = np.r_[goal]
        # self.goal = np.r_[goal]
        self.randinit()

        v = self.g.add_vertex(coord=goal)
        v.path = None

        self.progress_start(self.npoints)
        count = 0
        while count < self.npoints:

            random_point = self.qrandom_free()

            if self.showsamples:
                plt.plot(random_point[0], random_point[1], "ok", markersize=2)

            vnearest, d = self.g.closest(random_point)

            if d > 6:
                continue
            path, pstatus = self.dubins.query(random_point, vnearest.coord)
            if path is None:
                continue

            collision = False
            for config in path:
                if self.map.iscollision(self.vehicle.polygon(config)):
                    collision = True
                    break
            if collision:
                # print('collision')
                continue

            if pstatus.length > 6:
                # print('too long')
                continue

            # we have a valid configuration to add to the graph
            count += 1
            self.progress_next()

            # add new vertex to graph
            vnew = self.g.add_vertex(random_point)
            self.g.add_edge(vnew, vnearest, cost=pstatus.length)
            vnew.path = path

            self.vehicle.polygon(random_point).plot(color="b", alpha=0.1)
            plt.show()

        self.progress_end()

    def query(self, start):
        self._start = start
        vstart, d = self.g.closest(start)

        vpath, cost, _ = self.g.path_UCS(vstart, self.g[0])

        print(vpath)
        # stack the Dubins path segments
        path = np.empty((0, 3))
        for vertex in vpath:
            if vertex.path is not None:
                path = np.vstack((path, vertex.path))

        status = namedtuple("RRTStatus", ["length", "initial_d", "vertices"])(
            cost, d, vpath
        )

        return path, status

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):

        dx = x - self.goal.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def qrandom(self):
        return self.random.uniform(
            low=(self.map.workspace[0], self.map.workspace[2], -np.pi),
            high=(self.map.workspace[1], self.map.workspace[3], np.pi),
        )

    def qrandom_free(self):

        # iterate for a random freespace configuration
        while True:
            q = self.random()
            if not self.map.iscollision(self.vehicle.polygon(q)):
                return q

    def iscollision(self, q):
        return self.map.iscollision(self.vehicle.polygon(q))


if __name__ == "__main__":

    from roboticstoolbox.mobile.Vehicle import Bicycle

    # start and goal configuration
    qs = (2, 8, -np.pi / 2)
    qg = (8, 2, -np.pi / 2)

    # obstacle map
    map = PolygonMap(workspace=[0, 10])
    map.add([(5, 50), (5, 6), (6, 6), (6, 50)])
    # map.add([(5, 0), (6, 0), (6, 4), (5, 4)])
    map.add([(5, 4), (5, -50), (6, -50), (6, 4)])

    l = 3
    w = 1.5
    v0 = Polygon2([(-l / 2, w / 2), (-l / 2, -w / 2), (l / 2, -w / 2), (l / 2, w / 2)])

    vehicle = Bicycle(steer_max=0.4, l=2, polygon=v0)

    rrt = RRTPlanner(map=map, vehicle=vehicle, verbose=False, seed=0)

    rrt.plan(goal=qg)
    path, status = rrt.query(start=qs)
    rrt.plot(path)

    plt.show(block=True)
