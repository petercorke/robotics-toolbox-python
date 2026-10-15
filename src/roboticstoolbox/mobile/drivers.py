"""
Driving agents (drivers) for mobile robot vehicles

@Author: Peter Corke, original MATLAB code and Python version
@Author: Kristian Gibson, initial MATLAB port
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from math import atan2, pi
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from spatialmath import base
from spatialmath.base.types import R2, ArrayLike2, Points2

import roboticstoolbox as rtb
from roboticstoolbox.mobile.workspace import WorkspaceLike, resolve_workspace
from roboticstoolbox.tools.types import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from roboticstoolbox.mobile.Vehicle import VehicleBase

# default Matplotlib styles, a user supplied dict is merged over these
GOAL_STYLE: dict[str, Any] = {
    "marker": "D",
    "markersize": 6,
    "color": "r",
    "linestyle": "None",
}
PATH_STYLE: dict[str, Any] = {"color": "r", "linewidth": 3, "alpha": 0.7, "zorder": 0}


class VehicleDriverBase(ABC):
    """
    Abstract vehicle driver class

    Abstract class for an agent that can drive a mobile robot.

    :seealso: :class:`RandomPath` :class:`PurePursuit`
    """

    def __init__(
        self,
        workspace: WorkspaceLike | None = None,
        speed: float = 1,
        headinggain: float = 0.3,
        goal_style: dict[str, Any] | None = None,
        verbose: bool = False,
        goalmarkerstyle: dict[str, Any] | None = None,
    ) -> None:
        """
        Abstract vehicle driver

        :param workspace: dimension of workspace, see
            :func:`spatialmath.base.graphics.expand_dims`, defaults to None
        :type workspace: scalar, array_like(2), array_like(4)
        :param speed: forward speed, defaults to 1
        :param headinggain: gain of the proportional controller that steers the
            vehicle towards the current goal, defaults to 0.3
        :param goal_style: Matplotlib line style of the marker used to display
            the current goal, merged over the default :data:`GOAL_STYLE` (a red
            diamond)
        :param verbose: display debug information, defaults to False
        :param goalmarkerstyle: deprecated, use ``goal_style``

        .. deprecated:: 1.4.5
            ``goalmarkerstyle`` is deprecated, use ``goal_style`` instead.

        The workspace of the robot is a rectangular region defined by ``workspace``
        that is specified by (see :func:`~spatialmath.base.graphics.plotvol2`):

        ==============  =======  =======
        ``workspace``   x-range  y-range
        ==============  =======  =======
        A (scalar)      -A:A     -A:A
        [A, B]           A:B      A:B
        [A, B, C, D]     A:B      C:D
        ==============  =======  =======
        """
        # a scalar, 2-vector, 4-vector or an object with a workspace attribute
        self._workspace: NDArray | None = resolve_workspace(workspace)

        # the vehicle under control, set by the vehicle's control setter
        self._veh: VehicleBase | None = None

        # Matplotlib Line2D showing the current goal, once displayed
        self._goal_marker: Any = None

        self._speed = speed
        self._headinggain = headinggain
        self._verbose = verbose

        if goalmarkerstyle is not None:
            warnings.warn(
                "goalmarkerstyle is deprecated, use goal_style instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if goal_style is None:
                goal_style = goalmarkerstyle

        self._goal_style: dict[str, Any] = GOAL_STYLE | (goal_style or {})

    @abstractmethod
    def demand(self) -> R2:
        """
        Compute speed and steering angle

        :return: speed and steering angle demand for :class:`VehicleBase`

        When an instance of a :class:`VehicleDriverBase` class is attached as
        the control for an instance of a :class:`VehicleBase` class, this method
        is called at each time step to provide the control input.

        Has access to the vehicle and its state through the :meth:`vehicle`
        property.
        """

    @abstractmethod
    def init(self, ax: Axes | None = None) -> None:
        """
        Initialize driving agent

        :param ax: axes in which to draw markers, defaults to None

        Called at the start of a simulation run.  Used to initialize state
        including random number generator state.
        """

    @property
    def vehicle(self) -> VehicleBase | None:
        """
        Set/get the vehicle under control

        :getter: return the :class:`VehicleBase` instance, or None
        :setter: set the :class:`VehicleBase` instance

        .. note:: The setter is invoked by ``vehicle.control = driver``
        """
        return self._veh

    @vehicle.setter
    def vehicle(self, v: VehicleBase) -> None:
        self._veh = v

    def _init_goal_marker(self, ax: Axes | None, label: str = "goal") -> None:
        """
        Create the goal marker, to be moved by :meth:`driveto`

        :param ax: axes in which to draw the goal, None for no graphics
        :param label: legend label for the marker
        """
        self._goal_marker = None
        if ax is not None:
            self._goal_marker = plt.plot(
                np.nan, np.nan, **self._goal_style, label=label
            )[0]

    def _attached_vehicle(self) -> VehicleBase:
        """The vehicle under control, which must have been attached"""
        if self._veh is None:
            raise RuntimeError("driver is not attached to a vehicle")
        return self._veh

    def __repr__(self) -> str:
        """
        Convert to string, same as :meth:`__str__`

        :return: driver parameters and state in a compact human readable format
        """
        return str(self)

    @property
    def workspace(self) -> NDArray | None:
        """
        Size of robot driving workspace

        :return: workspace bounds [xmin, xmax, ymin, ymax], or None
        :rtype: ndarray(4) or None

        Returns the bounds of the workspace as specified by constructor
        option ``workspace`` or, if that was not given, the workspace of the
        vehicle that the driver is attached to.  None if neither has one.
        """
        if self._workspace is not None:
            return self._workspace
        if self._veh is not None:
            # the vehicle's own attribute, its property falls back to this driver
            return self._veh._workspace
        return None

    def driveto(self, goal: ArrayLike2) -> R2:
        """
        Speed and steering demand to drive towards a goal point

        :param goal: goal point (x, y)
        :return: speed and steering angle demand

        The speed demand is the constant ``speed``.  The steering angle demand
        is the heading error to the goal, :math:`\\Delta\\theta`, scaled by
        ``headinggain``.

        If the goal marker is displayed, it is moved to the goal.
        """
        veh = self._attached_vehicle()
        if self._goal_marker is not None:
            self._goal_marker.set_data([goal[0]], [goal[1]])
        goal_heading = atan2(goal[1] - veh._x[1], goal[0] - veh._x[0])
        delta_heading = base.angdiff(goal_heading, veh._x[2])
        if veh.verbose or self._verbose:
            print(
                f"t={veh._t:.1f}, pos=({veh._x[0]:.1f}, {veh._x[1]:.1f}), ",
                f"goal_heading={goal_heading * 180 / pi:.1f}, delta_heading={delta_heading * 180 / pi:.1f}",
            )

        return np.r_[self._speed, self._headinggain * delta_heading]


# ========================================================================= #


class RandomPath(VehicleDriverBase):
    def __init__(
        self, dthresh: float = 0.05, seed: int | None = 0, **kwargs: Any
    ) -> None:
        """
        Driving agent for random path

        :param dthresh: distance threshold as a fraction of the larger workspace
            dimension, defaults to 0.05
        :param seed: random number seed, defaults to 0, None for a random seed
        :param kwargs: arguments passed to :class:`VehicleDriverBase`, in
            particular ``workspace`` which is required
        :raises ValueError: no workspace specified

        Returns a *driver* object that drives the attached vehicle to a
        sequence of random waypoints.

        The driver is connected to the vehicle by::

            veh = Bicycle(control=driver)

        or::

            veh = Bicycle()
            veh.control = driver

        The waypoints are positioned inside a rectangular region defined by
        the ``workspace``, which must be given::

            driver = RandomPath(workspace=10)

        .. note::
            - It is possible in some cases for the vehicle to move outside the desired
              region, for instance if moving to a waypoint near the edge, the limited
              turning circle may cause the vehicle to temporarily move outside.
            - The vehicle chooses a new waypoint when it is closer than ``dthresh``
              times the larger workspace dimension to the current waypoint.
            - Uses its own random number generator so as to not influence the performance
              of other randomized algorithms such as path planning. Set
              ``seed=None`` to have it randomly initialized from the operating
              system.

        :seealso: :class:`Bicycle` :class:`Unicycle` :func:`~spatialmath.base.graphics.plotvol2`
        """

        # TODO options to specify region, maybe accept a Map object?
        super().__init__(**kwargs)

        self._d_prev = np.inf
        self._random = np.random.default_rng(seed)
        self._seed = seed
        self._goal: NDArray | None = None

        # the workspace is the driver's own or, once attached, the vehicle's
        self._dthresh_fraction = dthresh
        self._dthresh = 0.0
        if self.workspace is not None:
            self._set_threshold()

    def _set_threshold(self) -> NDArray:
        """
        Compute the waypoint distance threshold from the workspace

        :raises ValueError: neither the driver nor its vehicle has a workspace
        :return: workspace bounds
        """
        workspace = self.workspace
        if workspace is None:
            raise ValueError(
                "RandomPath requires a workspace, given to the driver or its vehicle"
            )
        self._dthresh = self._dthresh_fraction * max(
            workspace[1] - workspace[0], workspace[3] - workspace[2]
        )
        return workspace

    def __str__(self) -> str:
        """
        Convert to string

        :return: driver parameters and state in a compact human readable format
        """

        s = "RandomPath driver object\n"
        ws = self.workspace
        if ws is not None:
            s += (
                f"  X {ws[0]} : {ws[1]}; Y {ws[2]} : {ws[3]}, dthresh={self._dthresh}\n"
            )
        s += f"  current goal={self._goal}"
        return s

    def init(self, ax: Axes | None = None) -> None:
        """
        Initialize random path driving agent

        :param ax: axes in which to draw the current goal, defaults to None

        Called at the start of a simulation run.  Ensures that the
        random number generator is reseeded so that the sequence of random
        waypoints is repeatable.
        """

        # the workspace may have come from the vehicle, which is now attached
        self._set_threshold()

        if self._seed is not None:
            self._random = np.random.default_rng(self._seed)

        self._goal = None
        self._init_goal_marker(ax, label="random waypoint")

    def demand(self) -> R2:
        """
        Compute speed and steering angle for the current random waypoint

        :return: speed and steering angle demand for :class:`VehicleBase`

        This is the speed and steering angle that drives the vehicle toward the
        current waypoint.  When the vehicle is within ``dthresh`` times the
        larger workspace dimension of the waypoint, a new random waypoint is
        chosen.

        :seealso: :class:`VehicleBase`
        """

        if self._goal is None:
            self._new_goal()

        # if nearly at goal point, choose the next one
        assert self._goal is not None  # set by _new_goal
        d = np.linalg.norm(self._attached_vehicle()._x[0:2] - self._goal)
        if d < self._dthresh or abs(d - self._d_prev) < 1e-3:
            self._new_goal()
        # elif d > 2 * self._d_prev:
        #     self.choose_goal()
        self._d_prev = d

        return self.driveto(self._goal)

    def _new_goal(self) -> None:
        """Choose a new random goal, invoked from :meth:`demand`"""

        workspace = self._set_threshold()
        veh = self._attached_vehicle()

        # choose a uniform random goal within inner 80% of driving area
        while True:
            r = self._random.uniform(0.1, 0.9)
            gx = workspace[0:2] @ np.r_[r, 1 - r]

            r = self._random.uniform(0.1, 0.9)
            gy = workspace[2:4] @ np.r_[r, 1 - r]

            self._goal = np.r_[gx, gy]

            # check not too close to last goal
            if np.linalg.norm(self._goal - veh._x[0:2]) > 2 * self._dthresh:
                break

        if veh.verbose or self._verbose:
            print(f"set goal: {self._goal}")


# ========================================================================= #


class PurePursuit(VehicleDriverBase):
    def __init__(
        self,
        path: Points2,
        lookahead: float = 1,
        path_style: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Driving agent for pure pursuit

        :param path: path as a set of waypoints, one column per point
        :type path: ndarray(2,n)
        :param lookahead: lookahead distance, defaults to 1
        :param path_style: Matplotlib line style used to display the path,
            merged over the default :data:`PATH_STYLE` (a wide, translucent red
            line)
        :param kwargs: arguments passed to :class:`VehicleDriverBase`
        :raises ValueError: ``path`` is not a 2xN array

        Returns a *driver* object that drives the attached vehicle along the
        path using the pure pursuit algorithm: the vehicle steers toward the
        first point on the path that is at least ``lookahead`` away from the
        vehicle, beyond the closest point.

        The driver is connected to the vehicle by::

            veh = Bicycle(control=driver)

        or::

            veh = Bicycle()
            veh.control = driver

        If the simulation is animated, the path is plotted using ``path_style``
        and the current goal point, which changes at every time step, is shown
        using ``goal_style``.

        :seealso: :class:`RandomPath` :class:`VehicleBase`
        """
        super().__init__(**kwargs)

        # this should be a capability of spatialmath.base.getmatrix(), see
        # https://github.com/rai-opensource/spatialmath-python/issues/255
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[0] != 2:
            raise ValueError("path must be a 2xN array, one column per point")

        self._path = path
        self._lookahead = lookahead

        self._path_style: dict[str, Any] = PATH_STYLE | (path_style or {})

    def __str__(self) -> str:
        """
        Convert to string

        :return: driver parameters and state in a compact human readable format
        """

        s = "PurePursuit driver object\n"
        if self._workspace is None:
            return s + f"  lookahead={self._lookahead}\n"
        s += (
            f"  X {self._workspace[0]} : {self._workspace[1]}; "
            f"Y {self._workspace[2]} : {self._workspace[3]}, "
            f"lookahead={self._lookahead}\n"
        )
        return s

    def init(self, ax: Axes | None = None) -> None:
        """
        Initialize pure pursuit driving agent

        :param ax: axes in which to draw the path and waypoints, defaults to None

        Called at the start of a simulation run.  If ``ax`` is given, plot the
        path and create the goal marker.
        """
        if ax is not None:
            plt.plot(
                self._path[0, :], self._path[1, :], **self._path_style, label="path"
            )
        self._init_goal_marker(ax, label="pursuit goal")

    def demand(self) -> R2:
        """
        Compute speed and steering angle to follow the path

        :return: speed and steering angle demand for :class:`VehicleBase`

        The goal is the first point on the path, beyond the point closest to the
        vehicle, that is at least ``lookahead`` away from the vehicle.  If there
        is no such point the goal is the end of the path.
        """

        cp = self._attached_vehicle()._x[0:2]  # current point

        # find closest point on the path to current point
        d = np.linalg.norm(self._path.T - cp, axis=1)  # rely on implicit expansion
        i = np.argmin(d)

        # find all points on the path at least lookahead distance away
        (k,) = np.where(d[i + 1 :] >= self._lookahead)  # find all points beyond horizon
        if len(k) == 0:
            # no such points, we must be near the end, goal is the end
            goal = self._path[:, -1]
        else:
            # many such points, take the first one beyond look ahead distance
            goal = self._path[:, i + 1 + k[0]]

        return self.driveto(goal)


# ========================================================================= #

if __name__ == "__main__":
    # import unittest

    import roboticstoolbox as rtb

    # veh = rtb.Bicycle(control=rtb.RandomPath(workspace=10), animation="car")
    # print(veh.control)
    # veh.run(20, animate=True)

    # create the path
    path = np.array([[10.0, 10], [10, 60], [80, 80], [50, 10]]).T
    veh = rtb.Bicycle(
        control=rtb.PurePursuit(path, speed=4, lookahead=5, workspace=[0, 100]),
        animation="car",
    )
    print(veh)
    print(veh.control)
    veh.run(60, animate=True)
