"""
Continuous curvature path planner
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import scipy.integrate
import scipy.optimize
from spatialmath.base.types import ArrayLike3

from roboticstoolbox.mobile.PlannerBase import PlannerBase
from roboticstoolbox.tools.types import ArrayLike, NDArray


def solvepath(
    x: ArrayLike, q0: ArrayLike3 = (0, 0, 0), **kwargs: Any
) -> tuple[NDArray, float]:
    """
    Integrate the path for a curvature polynomial

    :param x: the four coefficients of the curvature polynomial followed by the
        total path length
    :param q0: initial configuration :math:`(x, y, \\theta)`, defaults to (0, 0, 0)
    :param kwargs: options passed to :func:`scipy.integrate.solve_ivp`
    :return: path, one configuration per column, and the maximum curvature
    :rtype: ndarray(3,N), float
    """
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
    """
    Maximum curvature of the path, the inequality constraint of the optimization
    """
    _, maxcurvature = solvepath(x, q0=(0, 0, 0))
    return maxcurvature


def costfunc(x: ArrayLike, start: ArrayLike3, goal: ArrayLike3) -> float:
    """
    Cost of a path, the configuration error at its end

    :param x: the four coefficients of the curvature polynomial followed by the
        total path length
    :param start: start configuration
    :param goal: goal configuration
    :return: distance of the end of the path from ``goal``
    """
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
    """
    Continuous curvature path planner
    """

    def __init__(self, curvature: float | None = None) -> None:
        r"""
        Continuous curvature path planner

        :param curvature: upper bound on the curvature of the path, defaults to
            None for no bound

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

            >>> from roboticstoolbox import CurvaturePolyPlanner
            >>> from math import pi
            >>> planner = CurvaturePolyPlanner()
            >>> path, status = planner.query(start=(0, 0, pi/2), goal=(1, 0, pi/2))
            >>> print(path[:5,:])
            >>> print(status)
        """
        super().__init__(ndims=3)
        self.curvature = curvature

    def query(
        self, start: ArrayLike3, goal: ArrayLike3
    ) -> tuple[NDArray, CurvaturePolyStatus]:
        r"""
        Find a path between two configurations

        :param start: start configuration :math:`(x, y, \theta)`
        :param goal: goal configuration :math:`(x, y, \theta)`
        :return: path and status
        :rtype: ndarray(N,3), namedtuple

        The curvature of the path is a cubic polynomial in the distance along
        the path, whose coefficients and the path length are found by numerical
        optimization.  The path is integrated in steps of at most 0.01.

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
        self.validate_start_goal(start, goal)
        start_, goal_ = self.start, self.goal
        assert start_ is not None and goal_ is not None  # nosec B101: narrows the type for mypy

        # initial estimate of path length is Euclidean distance
        d = np.linalg.norm(goal_[:2] - start_[:2])
        # state vector is kappa_0, a, b, c, s_f

        constraints: tuple[Any, ...] = ()
        if self.curvature is not None:
            constraints = (
                scipy.optimize.NonlinearConstraint(xcurvature, 0, self.curvature),
            )

        sol = scipy.optimize.minimize(
            costfunc,
            [0, 0, 0, 0, d],
            constraints=constraints,
            bounds=[(None, None), (None, None), (None, None), (None, None), (d, None)],
            args=(start_, goal_),
        )
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
