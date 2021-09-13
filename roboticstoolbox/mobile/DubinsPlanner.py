"""

Dubins path planner sample code

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
from roboticstoolbox.mobile import PlannerBase

import matplotlib.pyplot as plt
import numpy as np
from spatialmath import *
from spatialmath import base

def left_straight_left(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d + sa - sb

    mode = ["L", "S", "L"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((cb - ca), tmp0)
    t = base.wrap_0_2pi(-alpha + tmp1)
    p = math.sqrt(p_squared)
    q = base.wrap_0_2pi(beta - tmp1)

    return t, p, q, mode


def right_straight_right(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d - sa + sb
    mode = ["R", "S", "R"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((ca - cb), tmp0)
    t = base.wrap_0_2pi(alpha - tmp1)
    p = math.sqrt(p_squared)
    q = base.wrap_0_2pi(-beta + tmp1)

    return t, p, q, mode


def left_straight_right(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    t = base.wrap_0_2pi(-alpha + tmp2)
    q = base.wrap_0_2pi(-base.wrap_0_2pi(beta) + tmp2)

    return t, p, q, mode

def right_straight_left(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    t = base.wrap_0_2pi(alpha - tmp2)
    q = base.wrap_0_2pi(beta - tmp2)

    return t, p, q, mode


def right_left_right(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["R", "L", "R"]
    tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp_rlr) > 1.0:
        return None, None, None, mode

    p = base.wrap_0_2pi(2 * math.pi - math.acos(tmp_rlr))
    t = base.wrap_0_2pi(alpha - math.atan2(ca - cb, d - sa + sb) + base.wrap_0_2pi(p / 2.0))
    q = base.wrap_0_2pi(alpha - beta - t + base.wrap_0_2pi(p))
    return t, p, q, mode


def left_right_left(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["L", "R", "L"]
    tmp_lrl = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (- sa + sb)) / 8.0
    if abs(tmp_lrl) > 1:
        return None, None, None, mode
    p = base.wrap_0_2pi(2 * math.pi - math.acos(tmp_lrl))
    t = base.wrap_0_2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
    q = base.wrap_0_2pi(base.wrap_0_2pi(beta) - alpha - t + base.wrap_0_2pi(p))

    return t, p, q, mode


def dubins_path_planning_from_origin(end_x, end_y, end_yaw, curvature,
                                     step_size):
    dx = end_x
    dy = end_y
    D = math.hypot(dx, dy)
    d = D * curvature

    theta = base.wrap_0_2pi(math.atan2(dy, dx))
    alpha = base.wrap_0_2pi(- theta)
    beta = base.wrap_0_2pi(end_yaw - theta)

    planners = [left_straight_left, right_straight_right, left_straight_right,
                right_straight_left, right_left_right,
                left_right_left]

    best_cost = float("inf")
    bt, bp, bq, best_mode = None, None, None, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)
        if t is None:
            continue

        cost = (abs(t) + abs(p) + abs(q))
        if best_cost > cost:
            bt, bp, bq, best_mode = t, p, q, mode
            best_cost = cost
    lengths = [bt, bp, bq]

    x_list, y_list, yaw_list, directions = generate_local_course(
        sum(lengths), lengths, best_mode, curvature, step_size)

    return x_list, y_list, yaw_list, best_mode, best_cost, lengths


def interpolate(ind, length, mode, max_curvature, origin_x, origin_y,
                origin_yaw, path_x, path_y, path_yaw, directions):
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


def generate_local_course(total_length, lengths, mode, max_curvature,
                          step_size):
    n_point = math.trunc(total_length / step_size) + len(lengths) + 4

    path_x = [0.0 for _ in range(n_point)]
    path_y = [0.0 for _ in range(n_point)]
    path_yaw = [0.0 for _ in range(n_point)]
    directions = [0.0 for _ in range(n_point)]
    index = 1

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
        origin_x, origin_y, origin_yaw = \
            path_x[index], path_y[index], path_yaw[index]

        index -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = - d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            index += 1
            path_x, path_y, path_yaw, directions = interpolate(
                index, pd, m, max_curvature, origin_x, origin_y, origin_yaw,
                path_x, path_y, path_yaw, directions)
            pd += d

        ll = l - pd - d  # calc remain length

        index += 1
        path_x, path_y, path_yaw, directions = interpolate(
            index, l, m, max_curvature, origin_x, origin_y, origin_yaw,
            path_x, path_y, path_yaw, directions)

    if len(path_x) <= 1:
        return [], [], [], []

    # remove unused data
    while len(path_x) >= 1 and path_x[-1] == 0.0:
        path_x.pop()
        path_y.pop()
        path_yaw.pop()
        directions.pop()

    return path_x, path_y, path_yaw, directions

def path_planning(start, goal, curvature, step_size=0.1):
    """
    Dubins path planner

    input:
        s_x x position of start point [m]
        s_y y position of start point [m]
        s_yaw yaw angle of start point [rad]
        g_x x position of end point [m]
        g_y y position of end point [m]
        g_yaw yaw angle of end point [rad]
        c curvature [1/m]

    """
    s_x, s_y, s_yaw = start
    g_x, g_y, g_yaw = goal

    g_x = g_x - s_x
    g_y = g_y - s_y

    l_rot = base.rot2(s_yaw)
    le_xy = np.stack([g_x, g_y]).T @ l_rot
    le_yaw = g_yaw - s_yaw

    lp_x, lp_y, lp_yaw, mode, length, lengths = dubins_path_planning_from_origin(
        le_xy[0], le_xy[1], le_yaw, curvature, step_size)

    rot = base.rot2(-s_yaw)
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = [base.wrap_mpi_pi(i_yaw + s_yaw) for i_yaw in lp_yaw]

    path = np.c_[x_list, y_list, yaw_list]
    return path, length, mode, lengths

# ============================================================================
class DubinsPlanner(PlannerBase):
    r"""
    Dubins path planner

    :param curvature: maximum path curvature, defaults to 1.0
    :type curvature: float, optional
    :param stepsize: spacing between points on the path, defaults to 0.1
    :type stepsize: float, optional
    :return: Dubins path planner
    :rtype: DubinsPlanner instance

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 Configuration space
    Obstacle avoidance   No
    Curvature            Discontinuous
    Motion               Forwards only
    ==================   ========================

    Creates a planner that finds the path between two configurations in the
    plane using forward motion only.  The path comprises upto 3 segments that are
    straight lines or arcs with :math:`\pm` ``curvature``.

    :reference: On Curves of Minimal Length with a Constraint on Average 
        Curvature, and with Prescribed Initial and Terminal Positions and 
        Tangents,  Dubins, L.E. (July 1957), American Journal of Mathematics.
        79(3): 497–516.
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
        Find a Dubins path

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

        :seealso: :meth:`Planner.query`
        """
        super().query(start=start, goal=goal, next=False, **kwargs)

        path, length, mode, lengths = path_planning(
            start=self.start, goal=self.goal,
            curvature=self._curvature, step_size=self._stepsize)

        status = namedtuple('DubinsStatus', ['segments', 'length', 'seglengths'])
        
        return path, status(mode, sum(lengths), lengths)

if __name__ == '__main__':
    from math import pi

    start = (1, 1, pi/4)
    goal = (-3, -3, -pi/4)

    start = (0, 0, pi/2)
    goal = (1, 0, pi/2)

    dubins = DubinsPlanner(curvature=1.0)
    path, status = dubins.query(start, goal)

    print(path)
    print(status)
    dubins.plot(path, configspace=True)

    plt.show(block=True)
    # plt.plot(path_x, path_y, label="final course " + "".join(mode))

    # # plotting
    # plot_arrow(start_x, start_y, start_yaw)
    # plot_arrow(end_x, end_y, end_yaw)

    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show(block=True)
