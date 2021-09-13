
import math
import scipy.integrate
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from roboticstoolbox.mobile import *

def solvepath(x, q0=[0, 0, 0], **kwargs):
    # x[:4] is 4 coeffs of curvature polynomial
    # x[4] is total path length
    # q0 is initial state of the vehicle
    maxcurvature = 0

    def dotfunc(s, q, poly):
        # q = (x, y, θ)
        # qdot = (cosθ, sinθ, ϰ)
        k = poly[0] * s ** 3 + poly[1] * s ** 2 + poly[2] * s + poly[3]
        # k = ((poly[0] * s + poly[1]) * s  + poly[2]) * s + poly[3]

        # save maximum curvature for this path solution
        nonlocal maxcurvature
        maxcurvature = max(maxcurvature, abs(k))

        theta = q[2]
        return math.cos(theta), math.sin(theta), k

    cpoly = x[:4]
    s_f = x[4]
    sol = scipy.integrate.solve_ivp(dotfunc, [0, s_f], q0, args=(cpoly,), **kwargs)
    return sol.y, maxcurvature

def xcurvature(x):
    # inequality constraint function, must be non-negative
    _, maxcurvature = solvepath(x, q0=(0, 0, 0))
    return maxcurvature

def costfunc(x, start, goal):
    # final cost of path from start with params
    # p[0:4] is polynomial: k0, a, b, c
    # p[4] is s_f

    # integrate the path for this curvature polynomial and length
    # path is 3xN
    path, _ = solvepath(x, q0=start)

    # cost is configuration error at end of path
    e = np.linalg.norm(path[:, -1] - np.r_[goal])

    return e
class CurvaturePolyPlanner(PlannerBase):

    def __init__(self, curvature=None):
        super().__init__(ndims=3)
        self.curvature = curvature

    def query(self, start, goal):
        goal = np.r_[goal]
        start = np.r_[start]
        self._start = start
        self._goal = goal

        # initial estimate of path length is Euclidean distance 
        d = np.linalg.norm(goal[:2] - start[:2])
        # state vector is kappa_0, a, b, c, s_f

        if self.curvature is not None:
            nlcontraints = (scipy.optimize.NonlinearConstraint(xcurvature, 0, self.curvature),)
        else:
            nlcontraints = ()

        sol = scipy.optimize.minimize(costfunc, [0, 0, 0, 0, d],
            constraints=nlcontraints,
            bounds=[(None, None), (None, None), (None, None), (None, None), (d, None)],
            args=(start, goal))
        print(sol)
        path, maxcurvature = solvepath(sol.x, q0=start, dense_output=True, max_step = 1e-2)

        status = namedtuple('CurvaturePolyStatus', ['length', 'maxcurvature', 'poly'])(sol.x[4], maxcurvature, sol.x[:4])

        return path.T, status

if __name__ == '__main__':
    from math import pi

    # start = (1, 1, pi/4)
    # goal = (-3, -3, -pi/4)
    start = (0, 0, -pi/4)
    goal = (1, 2, pi/4)

    start = (0, 0, pi/2)
    goal = (1, 0, pi/2)

    planner = CurvaturePolyPlanner()
    path, status = planner.query(start, goal)
    print('start', path[0,:])
    print('goal', path[-1, :])

    print(status)
    planner.plot(path, block=True)

    ## attempt polynomial scaling, doesnt seem to work
    # sf = status.s_f
    # c = status.poly
    # print(c)

    # print(solvepath(np.r_[c, sf], start))

    # for i in range(4):
    #     c[i] /= sf ** (3-i)

    # print(solvepath(np.r_[c, 1], start))
    # print(c)