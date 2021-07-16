
import math
import scipy.integrate
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from roboticstoolbox.mobile.Planner import Planner


def solvepath(poly, s_f, x0=[0, 0, 0], **kwargs):

    def dotfunc(t, x, poly):
        theta = x[2]
        k = poly[0] * t ** 3 + poly[1] * t ** 2 + poly[2] * t + poly[3]
        return math.cos(theta), math.sin(theta), k

    sol = scipy.integrate.solve_ivp(dotfunc, [0, s_f], start, args=(poly,), **kwargs)
    return sol.y

def costfunc(unknowns, start, goal):
    # final cost of path from start with params
    # p[0:4] is polynomial
    # p[4] is s_f
    path = solvepath(poly=unknowns[:4], s_f=unknowns[4], x0=start)
    return np.linalg.norm(path[:, -1] - np.r_[goal])


class CurvaturePolyPlanner(Planner):

    def query(self, start, goal):
        goal = np.r_[goal]
        start = np.r_[start]
        self._start = start
        self._goal = goal

        delta = goal[:2] - start[:2]
        d = np.linalg.norm(delta)
        theta = math.atan2(delta[1], delta[0])
        sol = scipy.optimize.minimize(costfunc, [0, 0, 0, 1, theta, d], args=(start, goal,))

        path = solvepath(sol.x[:4], sol.x[4], dense_output=True, max_step = 1e-2)

        status = namedtuple('CurvaturePolyStatus', ['s_f', 'poly'])(sol.x[5], sol.x[:5])

        return path.T, status

if __name__ == '__main__':
    from math import pi

    # start = (1, 1, pi/4)
    # goal = (-3, -3, -pi/4)
    start = (0, 0, -pi/4)
    goal = (1, 2, pi/4)

    # start = (0, 0, pi/2)
    # goal = (1, 0, pi/2)

    planner = CurvaturePolyPlanner()
    path, status = planner.query(start, goal)

    print(status)
    planner.plot(path, block=True)
