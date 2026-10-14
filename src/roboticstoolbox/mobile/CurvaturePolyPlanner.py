from __future__ import annotations

import math
from typing import Any, Callable, NamedTuple
import scipy.integrate
import scipy.optimize
import numpy as np

# import matplotlib.pyplot as plt
from roboticstoolbox.mobile.PlannerBase import PlannerBase
from roboticstoolbox.tools.types import ArrayLike, NDArray


def solvepath(
    x: ArrayLike, q0: ArrayLike = [0, 0, 0], **kwargs: Any
) -> tuple[NDArray, float]:
    # x[:4] is 4 coeffs of curvature polynomial
    # x[4] is total path length
    # q0 is initial state of the vehicle
    maxcurvature = 0

    def dotfunc(s: float, q: ArrayLike, poly: ArrayLike) -> tuple[float, float, float]:
        # q = (x, y, θ)
        # qdot = (cosθ, sinθ, ϰ)
        k = poly[0] * s**3 + poly[1] * s**2 + poly[2] * s + poly[3]
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


def xcurvature(x: ArrayLike) -> float:
    # inequality constraint function, must be non-negative
    _, maxcurvature = solvepath(x, q0=(0, 0, 0))
    return maxcurvature


def costfunc(x: ArrayLike, start: ArrayLike, goal: ArrayLike) -> float:
    # final cost of path from start with params
    # p[0:4] is polynomial: k0, a, b, c
    # p[4] is s_f

    # integrate the path for this curvature polynomial and length
    # path is 3xN
    path, _ = solvepath(x, q0=start)

    # cost is configuration error at end of path
    e = np.linalg.norm(path[:, -1] - np.r_[goal])

    return e


class CurvaturePolyStatus(NamedTuple):
    """Length, maximum curvature and curvature polynomial of the path"""

    length: float
    maxcurvature: float
    poly: NDArray


class CurvaturePolyPlanner(PlannerBase):
    def __init__(self, curvature: float | None = None) -> None:
        r"""
        Continuous curvature path planner

        :param curvature: , defaults to None
        :type curvature: _type_, optional

        ==================   ========================
        Feature              Capability
        ==================   ========================
        Plan                 :math:`\SE{2}`
        Obstacle avoidance   No
        Curvature            Continuous
        Motion               Forwards only
        ==================   ========================

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import DubinsPlanner
            >>> from math import pi
            >>> planner = CurvaturePolyPlanner()
            >>> path, status = planner.query(start=(0, 0, pi/2), goal=(1, 0, pi/2))
            >>> print(path[:5,:])
            >>> print(status)
        """
        super().__init__(ndims=3)
        self.curvature = curvature

    def query(  # type: ignore[override]
        self, start: ArrayLike, goal: ArrayLike
    ) -> tuple[NDArray, CurvaturePolyStatus]:
        r"""
        Find a path between two configurations

        :param start: start configuration :math:`(x, y, \theta)`
        :type start: array_like(3), optional
        :param goal: goal configuration :math:`(x, y, \theta)`
        :type goal: array_like(3), optional
        :return: path and status
        :rtype: ndarray(N,3), namedtuple

        The path comprises points equally spaced at a distance of ``stepsize``.

        The returned status value has elements:

        +-------------------+-----------------------------------------------+
        | Element           |  Description                                  |
        +-------------------+-----------------------------------------------+
        | ``length``        | total path length                             |
        +-------------------+-----------------------------------------------+
        | ``maxcurvature``  | maximum curvature on path                     |
        +-------------------+-----------------------------------------------+
        | ``poly``          | curvature polynomial coefficients             |
        +-------------------+-----------------------------------------------+

        """
        goal_: NDArray = np.r_[goal]
        start_: NDArray = np.r_[start]
        self._start = start_
        self._goal = goal_

        # initial estimate of path length is Euclidean distance
        d = np.linalg.norm(goal_[:2] - start_[:2])
        # state vector is kappa_0, a, b, c, s_f

        if self.curvature is not None:
            nlcontraints = (
                scipy.optimize.NonlinearConstraint(xcurvature, 0, self.curvature),
            )
        else:
            nlcontraints = ()  # type: ignore[assignment]

        sol = scipy.optimize.minimize(
            costfunc,
            [0, 0, 0, 0, d],
            constraints=nlcontraints,
            bounds=[(None, None), (None, None), (None, None), (None, None), (d, None)],
            args=(start_, goal_),
        )
        print(sol)
        path, maxcurvature = solvepath(
            sol.x, q0=start_, dense_output=True, max_step=1e-2
        )

        status = CurvaturePolyStatus(sol.x[4], maxcurvature, sol.x[:4])

        return path.T, status


if __name__ == "__main__":
    from math import pi

    # start = (1, 1, pi/4)
    # goal = (-3, -3, -pi/4)
    # start = (0, 0, -pi/4)
    # goal = (1, 2, pi/4)

    start = (0, 0, pi / 2)
    goal = (1, 0, pi / 2)

    planner = CurvaturePolyPlanner()
    path, status = planner.query(start, goal)
    print("start", path[0, :])
    print("goal", path[-1, :])

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
