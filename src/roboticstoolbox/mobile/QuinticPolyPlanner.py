"""
Quintic polynomial path planner
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from spatialmath.base.types import ArrayLike3

from roboticstoolbox.mobile.PlannerBase import NoPathError, PlannerBase
from roboticstoolbox.tools.types import NDArray

# ======================================================================== #

# The following code is modified from Python Robotics
# https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning
# Quintic Polynomial Planner
# Author: Atsushi Sakai
# Copyright (c) 2016 - 2022 Atsushi Sakai and other contributors: https://github.com/AtsushiSakai/PythonRobotics/contributors
# Released under the MIT license: https://github.com/AtsushiSakai/PythonRobotics/blob/master/LICENSE


class _QuinticPolynomial:
    def __init__(
        self,
        xs: float,
        vxs: float,
        axs: float,
        xe: float,
        vxe: float,
        axe: float,
        time: float,
    ) -> None:
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array(
            [
                [time**3, time**4, time**5],
                [3 * time**2, 4 * time**3, 5 * time**4],
                [6 * time, 12 * time**2, 20 * time**3],
            ]
        )
        b = np.array(
            [
                xe - self.a0 - self.a1 * time - self.a2 * time**2,
                vxe - self.a1 - 2 * self.a2 * time,
                axe - 2 * self.a2,
            ]
        )
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t: float) -> float:
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * t**2
            + self.a3 * t**3
            + self.a4 * t**4
            + self.a5 * t**5
        )

        return xt

    def calc_first_derivative(self, t: float) -> float:
        xt = (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * t**2
            + 4 * self.a4 * t**3
            + 5 * self.a5 * t**4
        )

        return xt

    def calc_second_derivative(self, t: float) -> float:
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t: float) -> float:
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt


def quintic_polynomials_planner(
    sx: float,
    sy: float,
    syaw: float,
    sv: float,
    sa: float,
    gx: float,
    gy: float,
    gyaw: float,
    gv: float,
    ga: float,
    max_accel: float,
    max_jerk: float,
    dt: float,
    MIN_T: float,
    MAX_T: float,
) -> tuple[list[float], NDArray, list[float], list[float], list[float]]:
    """
    Quintic polynomial planner

    :param sx: start x position [m]
    :param sy: start y position [m]
    :param syaw: start yaw angle [rad]
    :param sv: start speed [m/s]
    :param sa: start acceleration [m/s^2]
    :param gx: goal x position [m]
    :param gy: goal y position [m]
    :param gyaw: goal yaw angle [rad]
    :param gv: goal speed [m/s]
    :param ga: goal acceleration [m/s^2]
    :param max_accel: maximum acceleration [m/s^2]
    :param max_jerk: maximum jerk [m/s^3]
    :param dt: time step [s]
    :param MIN_T: minimum path time, also the step in the time searched [s]
    :param MAX_T: maximum path time [s]
    :raises NoPathError: no path time in the range meets the acceleration
        and jerk limits
    :return: time, path, velocity, acceleration and jerk
    :rtype: tuple(list, ndarray(N,3), list, list, list)

    The path time is searched from ``MIN_T`` to ``MAX_T`` in steps of ``MIN_T``
    and the first time for which the maximum acceleration and jerk are within
    their limits is used.  The path is :math:`(x, y, \\theta)` for each time
    step.
    """

    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)
    vxg = gv * math.cos(gyaw)
    vyg = gv * math.sin(gyaw)

    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)
    axg = ga * math.cos(gyaw)
    ayg = ga * math.sin(gyaw)

    time: list[float] = []
    rx: list[float] = []
    ry: list[float] = []
    ryaw: list[float] = []
    rv: list[float] = []
    ra: list[float] = []
    rj: list[float] = []

    for T in np.arange(MIN_T, MAX_T, MIN_T):
        xqp = _QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
        yqp = _QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)

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

        if (
            max([abs(i) for i in ra]) <= max_accel
            and max([abs(i) for i in rj]) <= max_jerk
        ):
            break
    else:
        raise NoPathError(
            "no path time between MIN_T and MAX_T meets the acceleration and jerk limits"
        )

    return time, np.c_[rx, ry, ryaw], rv, ra, rj


# ====================== RTB wrapper ============================= #


# Copyright (c) 2022 Peter Corke: https://github.com/petercorke/robotics-toolbox-python
# Released under the MIT license: https://github.com/AtsushiSakai/PythonRobotics/blob/master/LICENSE
class QuinticPolyStatus(NamedTuple):
    """Time, velocity, acceleration and jerk along the path"""

    t: list[float]
    vel: list[float]
    acc: list[float]
    jerk: list[float]


class QuinticPolyPlanner(PlannerBase):
    r"""
    Quintic polynomial path planner

    :param dt: time step, defaults to 0.1
    :param start_vel: initial velocity, defaults to 0
    :param start_acc: initial acceleration, defaults to 0
    :param goal_vel: goal velocity, defaults to 0
    :param goal_acc: goal acceleration, defaults to 0
    :param max_acc: maximum acceleration, defaults to 1
    :param max_jerk: maximum jerk, defaults to 0.5
    :param min_t: minimum path time, defaults to 5
    :param max_t: maximum path time, defaults to 100

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 :math:`\SE{2}`
    Obstacle avoidance   No
    Curvature            Continuous
    Motion               Forwards only
    ==================   ========================

    Creates a planner that finds the path between two configurations in the
    plane using forward motion only.  The path is a continuous quintic polynomial
    for x and y

    .. math::

            x(t) &= a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5 \\
            y(t) &= b_0 + b_1 t + b_2 t^2 + b_3 t^3 + b_4 t^4 + b_5 t^5

    that meets the given constraints.  Trajectory time is given as a range.

    :reference: "Local Path Planning And Motion Control For AGV In
        Positioning",  Takahashi, T. Hongo, Y. Ninomiya and G.
        Sugimoto; Proceedings. IEEE/RSJ International Workshop on
        Intelligent Robots and Systems (IROS '89)  doi: 10.1109/IROS.1989.637936

    .. note:: The path time is searched in the interval [``min_t``, ``max_t``] in steps
        of ``min_t``.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import QuinticPolyPlanner
        >>> import numpy as np
        >>> start = (10, 10, np.deg2rad(10.0))
        >>> goal = (30, -10, np.deg2rad(20.0))
        >>> quintic = QuinticPolyPlanner(start_vel=1)
        >>> path, status = quintic.query(start, goal)
        >>> print(path[:5,:])

    :thanks: based on quintic polynomial planning from `Python Robotics <https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning>`_

    .. seealso:: :class:`PlannerBase` :class:`CurvaturePolyPlanner`
    """

    def __init__(
        self,
        dt: float = 0.1,
        start_vel: float = 0,
        start_acc: float = 0,
        goal_vel: float = 0,
        goal_acc: float = 0,
        max_acc: float = 1,
        max_jerk: float = 0.5,
        min_t: float = 5,
        max_t: float = 100,
    ) -> None:

        super().__init__(ndims=3)
        self.dt = dt
        self.start_vel = start_vel
        self.start_acc = start_acc
        self.goal_vel = goal_vel
        self.goal_acc = goal_acc
        self.max_acc = max_acc
        self.max_jerk = max_jerk
        self.min_t = min_t
        self.max_t = max_t

    def query(
        self, start: ArrayLike3, goal: ArrayLike3
    ) -> tuple[NDArray, QuinticPolyStatus]:
        r"""
        Find a quintic polynomial path

        :param start: start configuration :math:`(x, y, \theta)`
        :param goal: goal configuration :math:`(x, y, \theta)`
        :raises NoPathError: no path time in the range ``min_t`` to ``max_t``
            meets the acceleration and jerk limits
        :return: path and status
        :rtype: ndarray(N,3), namedtuple

        The returned status value has elements:

        ==========  ===================================================
        Element     Description
        ==========  ===================================================
        ``t``       time to execute the path
        ``vel``     velocity profile along the path
        ``acc``     acceleration profile along the path
        ``jerk``    jerk profile along the path
        ==========  ===================================================


        """
        # stored as arrays, as for the other planners
        self.validate_start_goal(start, goal)
        start_ = self.start
        goal_ = self.goal
        assert start_ is not None and goal_ is not None  # nosec B101: narrows the type for mypy

        time, path, v, a, j = quintic_polynomials_planner(
            start_[0],
            start_[1],
            start_[2],
            self.start_vel,
            self.start_acc,
            goal_[0],
            goal_[1],
            goal_[2],
            self.goal_vel,
            self.goal_acc,
            self.max_acc,
            self.max_jerk,
            dt=self.dt,
            MIN_T=self.min_t,
            MAX_T=self.max_t,
        )

        status = QuinticPolyStatus(time, v, a, j)

        return path, status


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    start = (10, 10, np.deg2rad(10.0))
    goal = (30, -10, np.deg2rad(20.0))

    quintic = QuinticPolyPlanner(start_vel=1)
    path, status = quintic.query(start, goal)

    print(status)
    quintic.plot(path)

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
