# ======================================================================== #

# The following code is based on code from Python Robotics
# https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning
# RRTDubins planning
# Author: Atsushi Sakai
# Copyright (c) 2016 - 2022 Atsushi Sakai and other contributors: https://github.com/AtsushiSakai/PythonRobotics/contributors
# Released under the MIT license: https://github.com/AtsushiSakai/PythonRobotics/blob/master/LICENSE

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
    r"""
    Rapidly exploring tree planner

    :param map: occupancy grid
    :type map: :class:`PolygonMap`
    :param vehicle: vehicle kinematic model
    :type vehicle: :class:`VehicleBase` subclass
    :param curvature: maximum path curvature, defaults to 1.0
    :type curvature: float, optional
    :param stepsize: spacing between points on the path, defaults to 0.2
    :type stepsize: float, optional
    :param npoints: number of vertices in random tree, defaults to 50
    :type npoints: int, optional

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 :math:`\SE{2}`
    Obstacle avoidance   Yes, polygons
    Curvature            Discontinuous
    Motion               Bidirectional
    ==================   ========================

    Creates a planner that finds the obstacle-free path between two
    configurations in the plane using forward and backward motion.  The path
    comprises multiple Dubins curves comprising straight lines, or arcs with
    curvature of :math:`\pm` ``curvature``. Motion along the segments may be in
    the forward or backward direction.

    Polygons are used for obstacle avoidance:

    - the environment is defined by a set of polygons represented by a :class:`PolygonMap`
    - the vehicle is defined by a single polygon specified by the ``polygon``
      argument to its constructor

    Example::

        from roboticstoolbox import RRTPlanner
        from spatialmath import Polygon2
        from math import pi

        # create polygonal obstacles
        map = PolygonMap(workspace=[0, 10])
        map.add([(5, 50), (5, 6), (6, 6), (6, 50)])
        map.add([(5, 4), (5, -50), (6, -50), (6, 4)])

        # create outline polygon for vehicle
        l, w = 3, 1.5
        vpolygon = Polygon2([(-l/2, w/2), (-l/2, -w/2), (l/2, -w/2), (l/2, w/2)])

        # create vehicle model
        vehicle = Bicycle(steer_max=1, L=2, polygon=vpolygon)

        # create planner
        rrt = RRTPlanner(map=map, vehicle=vehicle, npoints=50, seed=0)
        # start and goal configuration
        qs = (2, 8, -pi/2)
        qg = (8, 2, -pi/2)

        # plan path
        rrt.plan(goal=qg)
        path, status = rrt.query(start=qs)
        print(path[:5,:])
        print(status)


    :seealso: :class:`DubinsPlanner` :class:`Vehicle` :class:`PlannerBase`
    """

    def __init__(
        self,
        map,
        vehicle,
        curvature=1.0,
        stepsize=0.2,
        npoints=50,
        **kwargs,
    ):
        super().__init__(ndims=2, **kwargs)

        self.npoints = npoints
        self.map = map

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

    def plan(self, goal, showsamples=True, showvalid=True, animate=False):
        r"""
        Plan paths to goal using RRT

        :param goal: goal pose :math:`(x, y, \theta)`, defaults to previously set value
        :type goal: array_like(3), optional
        :param showsamples: display position part of configurations overlaid on the map, defaults to True
        :type showsamples: bool, optional
        :param showvalid: display valid configurations as vehicle polygons overlaid on the map, defaults to False
        :type showvalid: bool, optional
        :param animate: update the display as configurations are tested, defaults to False
        :type animate: bool, optional

        Compute a rapidly exploring random tree with its root at the ``goal``.
        The tree will have ``npoints`` vertices spread uniformly randomly over
        the workspace which is an attribute of the ``map``.

        For every new point added, a Dubins path is computed to the nearest
        vertex already in the graph.  Each configuration on that path, with
        spacing of ``stepsize``, is tested for obstacle intersection.

        The configurations tested are displayed (translation only) if ``showsamples`` is
        True.  The valid configurations are displayed as vehicle polygones if ``showvalid``
        is True.  If ``animate`` is True these points are displayed during the search
        process, otherwise a single figure is displayed at the end.

        :seealso: :meth:`query`
        """
        # TODO use validate
        self.goal = np.r_[goal]
        # self.goal = np.r_[goal]
        self.random_init()

        v = self.g.add_vertex(coord=goal)
        v.path = None

        if showsamples or showvalid:
            self.map.plot()

        self.progress_start(self.npoints)
        count = 0
        while count < self.npoints:
            random_point = self.qrandom_free()

            if showsamples:
                plt.plot(random_point[0], random_point[1], "ok", markersize=2)
                if animate:
                    plt.pause(0.02)

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

            if showvalid:
                self.vehicle.polygon(random_point).plot(color="b", alpha=0.1)
                if animate:
                    plt.pause(0.02)

        if (showvalid or showsamples) and not animate:
            plt.show(block=False)

        self.progress_end()

    def query(self, start):
        r"""
        Find a path from start configuration

        :param start: start configuration :math:`(x, y, \theta)`
        :type start: array_like(3), optional
        :return: path and status
        :rtype: ndarray(N,3), namedtuple

        The path comprises points equally spaced at a distance of ``stepsize``.

        The returned status value has elements:

        +---------------+---------------------------------------------------+
        | Element       |  Description                                      |
        +---------------+---------------------------------------------------+
        | ``length``    | total path length                                 |
        +-------------+-----------------------------------------------------+
        | ``initial_d`` | distance from start to first vertex in graph      |
        +---------------+---------------------------------------------------+
        | ``vertices``  | sequence of vertices in the graph                 |
        +---------------+---------------------------------------------------+

        """
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

    # def _generate_final_course(self, goal_ind):
    #     path = [[self.end.x, self.end.y]]
    #     node = self.node_list[goal_ind]
    #     while node.parent is not None:
    #         path.append([node.x, node.y])
    #         node = node.parent
    #     path.append([node.x, node.y])

    #     return path

    # def _calc_dist_to_goal(self, x, y):

    #     dx = x - self.goal.x
    #     dy = y - self.end.y
    #     return math.hypot(dx, dy)

    def qrandom(self):
        r"""
        Random configuration

        :return: random configuration :math:`(x, y, \theta)`
        :rtype: ndarray(3)

        Returns a random configuration where position :math:`(x, y)`
        lies within the bounds of the ``map`` associated with this planner.

        :seealso: :meth:`qrandom_free`
        """
        return self.random.uniform(
            low=(self.map.workspace[0], self.map.workspace[2], -np.pi),
            high=(self.map.workspace[1], self.map.workspace[3], np.pi),
        )

    def qrandom_free(self):
        r"""
        Random obstacle free configuration

        :return: random configuration :math:`(x, y, \theta)`
        :rtype: ndarray(3)

        Returns a random obstacle free configuration where position :math:`(x,
        y)` lies within the bounds of the ``map`` associated with this planner.
        Iterates on :meth:`qrandom`

        :seealso: :meth:`qrandom` :meth:`iscollision`
        """
        # iterate for a random freespace configuration
        while True:
            q = self.qrandom()
            if not self.iscollision(q):
                return q

    def iscollision(self, q):
        r"""
        Test if configuration is collision

        :param q: vehicle configuration :math:`(x, y, \theta)`
        :type q: array_like(3)
        :return: collision status
        :rtype: bool

        Transforms the vehicle polygon and tests for intersection against
        the polygonal obstacle map.
        """
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
