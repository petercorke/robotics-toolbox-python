"""
Python Vehicle
@Author: Peter Corke, original MATLAB code and Python version
@Author: Kristian Gibson, initial MATLAB port
"""
from abc import ABC, abstractmethod
import warnings
from math import pi, sin, cos, tan, atan2
import numpy as np
from scipy import integrate, linalg, interpolate

import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.transforms as mtransforms

from spatialmath import SE2, base
import roboticstoolbox as rtb


class VehicleDriverBase(ABC):
    """
    Abstract Vehicle driver class

    Abtract class that can drive a mobile robot.

    :seealso: :class:`RandomPath`
    """

    @abstractmethod
    def demand(self):
        """
        Compute speed and heading

        :return: speed and steering for :class:`VehicleBase`

        When an instance of a :class:`VehicleDriverBase` class is attached as
        the control for an instance of a :class:`VehicleBase` class, this method
        is called at each time step to provide the control input.

        Has access to the vehicle and its state through the :meth:`vehicle`
        property.
        """
        pass

    @abstractmethod
    def init(self):
        """
        Initialize driving agent

        Called at the start of a simulation run.  Used to initialize state
        including random number generator state.
        """
        pass

    @property
    def vehicle(self):
        """
        Set/get the vehicle under control

        :getter: return :class:`VehicleBase` instance
        :setter: set :class:`VehicleBase` instance

        .. note:: The setter is invoked by ``vehicle.control = driver``
        """
        return self._veh

    @vehicle.setter
    def vehicle(self, v):

        self._veh = v

    def __repr__(self):
        return str(self)


# ========================================================================= #


class RandomPath(VehicleDriverBase):
    def __init__(
        self,
        workspace,
        speed=1,
        dthresh=0.05,
        seed=0,
        headinggain=0.3,
        goalmarkerstyle=None,
    ):
        """
        Driving agent for random path

        :param workspace: dimension of workspace, see :func:`spatialmath.base.exand_dims`
        :type workspace: scalar, array_like(2), array_like(4)
        :param speed: forward speed, defaults to 1
        :type speed: float, optional
        :param dthresh: distance threshold, defaults to 0.05
        :type dthresh: float, optional

        :raises ValueError: bad workspace specified

        Returns a *driver* object that drives the attached vehicle to a
        sequence of random waypoints.

        The driver is connected to the vehicle by::

            Vehicle(control=driver)

        or::

            veh = Vehicle()
            veh.control = driver

        The waypoints are positioned inside a rectangular region defined by
        the vehicle that is specified by  (see ``plotvol2``):

        ==============  =======  =======
        ``workspace``   x-range  y-range
        ==============  =======  =======
        A (scalar)      -A:A     -A:A
        [A, B]           A:B      A:B
        [A, B, C, D]     A:B      C:D
        ==============  =======  =======


        .. note::
            - It is possible in some cases for the vehicle to move outside the desired
              region, for instance if moving to a waypoint near the edge, the limited
              turning circle may cause the vehicle to temporarily move outside.
            - The vehicle chooses a new waypoint when it is closer than ``dthresh``
              to the current waypoint.
            - Uses its own random number generator so as to not influence the performance
              of other randomized algorithms such as path planning. Set ``seed=None`` to have it randomly initialized from the
              operating system.

        :seealso: :class:`Bicycle` :class:`Unicycle` :func:`~spatialmath.base.graphics.plotvol2`
        """

        # TODO options to specify region, maybe accept a Map object?

        if hasattr(workspace, "workspace"):
            # workspace can be defined by an object with a workspace attribute
            self._workspace = base.expand_dims(workspace.workspace)
        else:
            self._workspace = base.expand_dims(workspace)

        self._speed = speed
        self._dthresh = dthresh * np.diff(self._workspace[0:2])
        self._goal_marker = None
        if goalmarkerstyle is None:
            self._goal_marker_style = {
                "marker": "D",
                "markersize": 6,
                "color": "r",
                "linestyle": "None",
            }
        else:
            self._goal_marker_style = goalmarkerstyle
        self._headinggain = headinggain

        self._d_prev = np.inf
        self._random = np.random.default_rng(seed)
        self._seed = seed
        self.verbose = True
        self._goal = None
        self._dthresh = dthresh * max(
            self._workspace[1] - self._workspace[0],
            self._workspace[3] - self._workspace[2],
        )

        self._veh = None

    def __str__(self):
        """%RandomPath.char Convert to string
        %
        % s = R.char() is a string showing driver parameters and state in in
        % a compact human readable format."""

        s = "RandomPath driver object\n"
        s += (
            f"  X {self._workspace[0]} : {self._workspace[1]}; Y {self._workspace[0]} :"
            f" {self._workspace[1]}, dthresh={self._dthresh}\n"
        )
        s += f"  current goal={self._goal}"
        return s

    @property
    def workspace(self):
        """
        Size of robot driving workspace

        :return: workspace bounds [xmin, xmax, ymin, ymax]
        :rtype: ndarray(4)

        Returns the bounds of the workspace as specified by constructor
        option ``workspace``
        """
        return self._workspace

    def init(self, ax=None):
        """
        Initialize random path driving agent

        :param ax: axes in which to draw via points, defaults to None
        :type ax: Axes, optional

        Called at the start of a simulation run.  Ensures that the
        random number generated is reseeded to ensure that
        the sequence of random waypoints is repeatable.
        """

        if self._seed is not None:
            self._random = np.random.default_rng(self._seed)

        self._goal = None
        # delete(driver.h_goal);   % delete the goal
        # driver.h_goal = [];
        if ax is not None:
            self._goal_marker = plt.plot(
                np.nan, np.nan, **self._goal_marker_style, label="random waypoint"
            )[0]

    def demand(self):
        """
        Compute speed and heading for random waypoint
            %
            % [SPEED,STEER] = R.demand() is the speed and steer angle to
            % drive the vehicle toward the next waypoint.  When the vehicle is
            % within R.dtresh a new waypoint is chosen.
            %
            % See also Vehicle."""

        if self._goal is None:
            self._new_goal()

        # if nearly at goal point, choose the next one
        d = np.linalg.norm(self._veh._x[0:2] - self._goal)
        if d < self._dthresh or abs(d - self._d_prev) < 1e-3:
            self._new_goal()
        # elif d > 2 * self._d_prev:
        #     self.choose_goal()
        self._d_prev = d

        speed = self._speed

        goal_heading = atan2(
            self._goal[1] - self._veh._x[1], self._goal[0] - self._veh._x[0]
        )
        delta_heading = base.angdiff(goal_heading, self._veh._x[2])

        return np.r_[speed, self._headinggain * delta_heading]

    ## private method, invoked from demand() to compute a new waypoint

    def _new_goal(self):

        # choose a uniform random goal within inner 80% of driving area
        while True:
            r = self._random.uniform(0.1, 0.9)
            gx = self._workspace[0:2] @ np.r_[r, 1 - r]

            r = self._random.uniform(0.1, 0.9)
            gy = self._workspace[2:4] @ np.r_[r, 1 - r]

            self._goal = np.r_[gx, gy]

            # check not too close to last goal
            if np.linalg.norm(self._goal - self._veh._x[0:2]) > 2 * self._dthresh:
                break

        if self._veh.verbose:
            print(f"set goal: {self._goal}")

        # update the goal marker
        if self._goal_marker is not None:
            self._goal_marker.set_xdata(self._goal[0])
            self._goal_marker.set_ydata(self._goal[1])


# ========================================================================= #


class PurePursuit(VehicleDriverBase):
    def __init__(self, speed=1, radius=1):
        pass

    def __str__(self):
        pass

    def init(self):
        pass

    def demand(self):
        pass


# ========================================================================= #

if __name__ == "__main__":

    import unittest
