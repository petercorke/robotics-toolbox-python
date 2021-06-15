"""

Quintic Polynomials Planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Local Path planning And Motion Control For Agv In Positioning](http://ieeexplore.ieee.org/document/637936/)

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

import matplotlib.pyplot as plt
import numpy as np
from roboticstoolbox.mobile.Planner import Planner

# parameter
MAX_T = 100.0  # maximum time to the goal [s]
MIN_T = 5.0  # minimum time to the goal[s]

show_animation = True


class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


def quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt):
    """
    quintic polynomial planner

    input
        s_x: start x position [m]
        s_y: start y position [m]
        s_yaw: start yaw angle [rad]
        sa: start accel [m/ss]
        gx: goal x position [m]
        gy: goal y position [m]
        gyaw: goal yaw angle [rad]
        ga: goal accel [m/ss]
        max_accel: maximum accel [m/ss]
        max_jerk: maximum jerk [m/sss]
        dt: time tick [s]

    return
        time: time result
        rx: x position result list
        ry: y position result list
        ryaw: yaw angle result list
        rv: velocity result list
        ra: accel result list

    """

    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)
    vxg = gv * math.cos(gyaw)
    vyg = gv * math.sin(gyaw)

    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)
    axg = ga * math.cos(gyaw)
    ayg = ga * math.sin(gyaw)

    time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

    for T in np.arange(MIN_T, MAX_T, MIN_T):
        xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
        yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)

        time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

        for t in np.arange(0.0, T + dt, dt):
            time.append(t)
            rx.append(xqp.calc_point(t))
            ry.append(yqp.calc_point(t))

            vx = xqp.calc_first_derivative(t)
            vy = yqp.calc_first_derivative(t)
            v = np.hypot(vx, vy)
            yaw = math.atan2(vy, vx)
            rv.append(v)
            ryaw.append(yaw)

            ax = xqp.calc_second_derivative(t)
            ay = yqp.calc_second_derivative(t)
            a = np.hypot(ax, ay)
            if len(rv) >= 2 and rv[-1] - rv[-2] < 0.0:
                a *= -1
            ra.append(a)

            jx = xqp.calc_third_derivative(t)
            jy = yqp.calc_third_derivative(t)
            j = np.hypot(jx, jy)
            if len(ra) >= 2 and ra[-1] - ra[-2] < 0.0:
                j *= -1
            rj.append(j)

        if max([abs(i) for i in ra]) <= max_accel and max([abs(i) for i in rj]) <= max_jerk:
            print("find path!!")
            break

    return time, np.c_[rx, ry, ryaw], rv, ra, rj


class QuinticPolyPlanner(Planner):

    def __init__(self, dt=0.1, start_vel=0, start_acc=0, goal_vel=0, goal_acc=0,
            max_acc=1, max_jerk=0.5):
        super().__init__()
        self.dt = dt
        self.start_vel = start_vel
        self.start_acc = start_acc
        self.goal_vel = goal_vel
        self.goal_acc = goal_acc
        self.max_acc = max_acc
        self.max_jerk = max_jerk


    def query(self, start, goal):
        self._start = start
        self._goal = goal

        time, path, v, a, j = quintic_polynomials_planner(
            start[0], start[1], start[2], self.start_vel, self.start_acc,
            goal[0], goal[1], goal[2], self.start_vel, self.start_acc, 
            self.max_acc, self.max_jerk, dt=self.dt)

        status = namedtuple('QuinticPolyStatus', ['t', 'vel', 'acc', 'jerk'])(
                time, v, a, j)
        
        return path, status

if __name__ == '__main__':
    from math import pi

    start = (10, 10, np.deg2rad(10.0))
    goal = (30, -10, np.deg2rad(20.0))

    quintic = QuinticPolyPlanner(start_vel=1)
    path, status = quintic.query(start, goal)

    print(path)
    print(status)
    quintic.plot(path=path[:, :2])

    plt.figure()
    plt.subplot(311)
    plt.plot(status.t, status.vel, "-r")
    plt.ylabel("velocity (m/s)")
    plt.grid(True)

    plt.subplot(312)
    plt.plot(status.t, status.acc, "-r")
    plt.ylabel("accel (m/s2)")
    plt.grid(True)

    plt.subplot(313)
    plt.plot(status.t, status.jerk, "-r")
    plt.xlabel("Time (s)")
    plt.ylabel("jerk (m/s3)")
    plt.grid(True)

    plt.show(block=True)


