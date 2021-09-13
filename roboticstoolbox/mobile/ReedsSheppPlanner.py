"""

Reeds Shepp path planner sample code

author Atsushi Sakai(@Atsushi_twi)

The MIT License (MIT)

Copyright (c) 2016 - 2021 Atsushi Sakai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

"""
import math
from collections import namedtuple
from roboticstoolbox.mobile.PlannerBase import PlannerBase
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import *
from spatialmath import base


class Path:

    def __init__(self):
        self.lengths = []
        self.ctypes = []
        self.L = 0.0
        self.x = []
        self.y = []
        self.yaw = []
        self.directions = []


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def straight_left_straight(x, y, phi):
    phi = base.wrap_0_2pi(phi)
    if y > 0.0 and 0.0 < phi < math.pi * 0.99:
        xd = - y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v
    elif y < 0.0 < phi < math.pi * 0.99:
        xd = - y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = -math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v

    return False, 0.0, 0.0, 0.0


def set_path(paths, lengths, ctypes):
    path = Path()
    path.ctypes = ctypes
    path.lengths = lengths

    # check same path exist
    for tpath in paths:
        typeissame = (tpath.ctypes == path.ctypes)
        if typeissame:
            if sum(np.abs(tpath.lengths)) - sum(np.abs(path.lengths)) <= 0.01:
                return paths  # not insert path

    path.L = sum([abs(i) for i in lengths])

    # Base.Test.@test path.L >= 0.01
    if path.L >= 0.01:
        paths.append(path)

    return paths


def straight_curve_straight(x, y, phi, paths):
    flag, t, u, v = straight_left_straight(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "L", "S"])

    flag, t, u, v = straight_left_straight(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "R", "S"])

    return paths


def polar(x, y):
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan2(y, x)
    return r, theta


def left_straight_left(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= 0.0:
        v = base.wrap_0_2pi(phi - t)
        if v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def left_right_left(x, y, phi):
    u1, t1 = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))

    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = base.wrap_0_2pi(t1 + 0.5 * u + math.pi)
        v = base.wrap_0_2pi(phi - t + u)

        if t >= 0.0 >= u:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def curve_curve_curve(x, y, phi, paths):
    flag, t, u, v = left_right_left(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "R", "L"])

    flag, t, u, v = left_right_left(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "R", "L"])

    flag, t, u, v = left_right_left(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "L", "R"])

    flag, t, u, v = left_right_left(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "L", "R"])

    # backwards
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    flag, t, u, v = left_right_left(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["L", "R", "L"])

    flag, t, u, v = left_right_left(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["L", "R", "L"])

    flag, t, u, v = left_right_left(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["R", "L", "R"])

    flag, t, u, v = left_right_left(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["R", "L", "R"])

    return paths


def curve_straight_curve(x, y, phi, paths):
    flag, t, u, v = left_straight_left(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "L"])

    flag, t, u, v = left_straight_left(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "L"])

    flag, t, u, v = left_straight_left(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "R"])

    flag, t, u, v = left_straight_left(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "R"])

    flag, t, u, v = left_straight_right(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "R"])

    flag, t, u, v = left_straight_right(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "R"])

    flag, t, u, v = left_straight_right(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "L"])

    flag, t, u, v = left_straight_right(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "L"])

    return paths


def left_straight_right(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = base.wrap_0_2pi(t1 + theta)
        v = base.wrap_0_2pi(t - phi)

        if t >= 0.0 and v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def generate_path(q0, q1, max_curvature):
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature

    paths = []
    paths = straight_curve_straight(x, y, dth, paths)
    paths = curve_straight_curve(x, y, dth, paths)
    paths = curve_curve_curve(x, y, dth, paths)

    return paths


def interpolate(ind, length, mode, max_curvature, origin_x, origin_y, origin_yaw, path_x, path_y, path_yaw, directions):
    if mode == "S":
        path_x[ind] = origin_x + length / max_curvature * math.cos(origin_yaw)
        path_y[ind] = origin_y + length / max_curvature * math.sin(origin_yaw)
        path_yaw[ind] = origin_yaw
    else:  # curve
        ldx = math.sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - math.cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - math.cos(length)) / -max_curvature
        gdx = math.cos(-origin_yaw) * ldx + math.sin(-origin_yaw) * ldy
        gdy = -math.sin(-origin_yaw) * ldx + math.cos(-origin_yaw) * ldy
        path_x[ind] = origin_x + gdx
        path_y[ind] = origin_y + gdy

    if mode == "L":  # left turn
        path_yaw[ind] = origin_yaw + length
    elif mode == "R":  # right turn
        path_yaw[ind] = origin_yaw - length

    if length > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1

    return path_x, path_y, path_yaw, directions


def generate_local_course(total_length, lengths, mode, max_curvature, step_size):
    n_point = math.trunc(total_length / step_size) + len(lengths) + 4

    px = [0.0 for _ in range(n_point)]
    py = [0.0 for _ in range(n_point)]
    pyaw = [0.0 for _ in range(n_point)]
    directions = [0.0 for _ in range(n_point)]
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    ll = 0.0

    for (m, l, i) in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        # set origin state
        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]

        ind -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = - d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = interpolate(
                ind, pd, m, max_curvature, ox, oy, oyaw, px, py, pyaw, directions)
            pd += d

        ll = l - pd - d  # calc remain length

        ind += 1
        px, py, pyaw, directions = interpolate(
            ind, l, m, max_curvature, ox, oy, oyaw, px, py, pyaw, directions)

    # remove unused data
    while px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]

    paths = generate_path(q0, q1, maxc)
    for path in paths:
        x, y, yaw, directions = generate_local_course(
            path.L, path.lengths, path.ctypes, maxc, step_size * maxc)

        # convert global coordinate
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2])
                  * iy + q0[0] for (ix, iy) in zip(x, y)]
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2])
                  * iy + q0[1] for (ix, iy) in zip(x, y)]
        path.yaw = [pi_2_pi(iyaw + q0[2]) for iyaw in yaw]
        path.directions = directions
        path.lengths = [length / maxc for length in path.lengths]
        path.L = path.L / maxc

    return paths


def reeds_shepp_path_planning(start, goal, maxc, step_size):

    s_x, s_y, s_yaw = start
    g_x, g_y, g_yaw = goal

    paths = calc_paths(s_x, s_y, s_yaw, g_x, g_y, g_yaw, maxc, step_size)

    if not paths:
        return None

    minL = float("Inf")
    best_path_index = -1
    for i, _ in enumerate(paths):
        if paths[i].L <= minL:
            minL = paths[i].L
            best_path_index = i

    bpath = paths[best_path_index]

    return bpath

# ============================================================================

class ReedsSheppPlanner(PlannerBase):
    r"""
    Reeds-Shepp path planner

    :param curvature: maximum path curvature, defaults to 1.0
    :type curvature: float, optional
    :param stepsize: spacing between points on the path, defaults to 0.1
    :type stepsize: float, optional
    :param Planner: Reeds-Shepp path planner
    :type Planner: ReedsSheppPlanner instance

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 Configuration space
    Obstacle avoidance   No
    Curvature            Discontinuous
    Motion               Bidirectional
    ==================   ========================

    Creates a planner that finds the path between two configurations in the
    plane using forward and backward motion.  The path comprises upto 3 segments
    that are straight lines or arcs with :math:`\pm` ``curvature``.

    :reference: Optimal paths for a car that goes both forwards and backwards,
        Reeds, J.A. and L.A. Shepp, Pacific J. Math., 145 (1990), 
        pp. 367–393. 

    :author: Atsushi Sakai `PythonRobotics <https://github.com/AtsushiSakai/PythonRobotics>`_
    :seealso: :class:`Planner`
    """
    def __init__(self, curvature=1, stepsize=0.1, **kwargs):
        super().__init__(ndims=3, **kwargs)
        self._curvature = curvature
        self._stepsize = stepsize

    def __str__(self):
        s = super().__str__() + f"\n  curvature={self.curvature}, stepsize={self.stepsize}"

    def query(self, start, goal, **kwargs):
        r"""
        Find Reeds-Shepp path

        :param start: start configuration :math:`(x, y, \theta)`
        :type start: array_like(3), optional
        :param goal: goal configuration :math:`(x, y, \theta)`
        :type goal: array_like(3), optional
        :return: path and status
        :rtype: ndarray(N,3), namedtuple

        The returned status value has elements:

        +-------------+-----------------------------------------------------+
        | Element     |  Description                                        |
        +-------------+-----------------------------------------------------+
        |``segments`` | a list containing the type of each path segment as  |
        |             | a single letter code: either "L", "R" or "S" for    |
        |             | left turn, right turn or straight line respectively.|
        +-------------+-----------------------------------------------------+
        |``lengths``  | the length of each path segment. The sign of the    |
        |             |length indicates the direction of travel.            |
        +-------------+-----------------------------------------------------+
        |``direction``| the direction of motion at each point on the path   |
        +-------------+-----------------------------------------------------+
    
        .. note:: The direction of turning is reversed when travelling 
            backwards.

        :seealso: :meth:`Planner.query`
        """
        super().query(start=start, goal=goal, next=False, **kwargs)

        bpath = reeds_shepp_path_planning(
            start=self.start, goal=self.goal,
            maxc=self._curvature, step_size=self._stepsize)

        path = np.c_[bpath.x, bpath.y, bpath.yaw]

        status = namedtuple('ReedsSheppStatus', 
            ['segments', 'length', 'seglengths', 'direction'])

        return path, status(bpath.ctypes, sum([abs(l) for l in bpath.lengths]),
            bpath.lengths, bpath.directions)


if __name__ == '__main__':
    from math import pi

    # start = (-1, -4, -np.radians(20))
    # goal = (5, 5, np.radians(25))


    start = (0, 0, 0)
    goal = (0, 0, pi)

    reedsshepp = ReedsSheppPlanner(curvature=1.0, stepsize=0.1)
    path, status = reedsshepp.query(start, goal)
    print(status)

    # px, py, pyaw, mode, clen = reeds_shepp_path_planning(
    #     start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature, step_size)

    # if show_animation:  # pragma: no cover
    #     plt.cla()
    #     plt.plot(px, py, label="final course " + str(mode))

    #     # plotting
    #     plot_arrow(start_x, start_y, start_yaw)
    #     plot_arrow(end_x, end_y, end_yaw)

    #     plt.legend()
    #     plt.grid(True)
    #     plt.axis("equal")
    #     plt.show(block=True)

    reedsshepp.plot(path=path, direction=status.direction, configspace=True)
    plt.show(block=True)