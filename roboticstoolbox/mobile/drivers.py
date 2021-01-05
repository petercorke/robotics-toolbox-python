"""
Python Vehicle
@Author: Kristian Gibson
@Author: Peter Corke
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


class VehicleDriver(ABC):

    @abstractmethod
    def demand():
        pass

    @abstractmethod
    def init():
        pass

    @abstractmethod
    def vehicle():
        pass    

# ========================================================================= #

class RandomPath(VehicleDriver):
    """
        RandomPath Vehicle driver class

    Create a "driver" object capable of steering a Vehicle subclass object through random 
    waypoints within a rectangular region and at constant speed.

    The driver object is connected to a Vehicle object by the latter's
    add_driver() method.  The driver's demand() method is invoked on every
    call to the Vehicle's step() method.

    Methods::
    init       reset the random number generator
    demand     speed and steer angle to next waypoint
    display    display the state and parameters in human readable form
    char       convert to string
    plot      
    Properties::
    goal          current goal/waypoint coordinate
    veh           the Vehicle object being controlled
    dim           dimensions of the work space (2x1) [m]
    speed         speed of travel [m/s]
    dthresh       proximity to waypoint at which next is chosen [m]

    Example::

    veh = Bicycle(V);
    veh.add_driver( RandomPath(20, 2) );

    Notes::
    - It is possible in some cases for the vehicle to move outside the desired
    region, for instance if moving to a waypoint near the edge, the limited
    turning circle may cause the vehicle to temporarily move outside.
    - The vehicle chooses a new waypoint when it is closer than property
    closeenough to the current waypoint.
    - Uses its own random number stream so as to not influence the performance
    of other randomized algorithms such as path planning.

    Reference::

    Robotics, Vision & Control, Chap 6,
    Peter Corke,
    Springer 2011

    See also Vehicle, Bicycle, Unicycle.

    """

    def __init__(self, dim, speed=1, dthresh=0.05, seed=None, headinggain=1, goalmarkerstyle=None):
        """
        Driving agent for random path

        :param dim: dimension of workspace, see spatialmath.plotvol2
        :type dim: scalar, array_like(2), array_like(4)
        :param speed: forward speed, defaults to 1
        :type speed: float, optional
        :param dthresh: distance threshold, defaults to 0.05
        :type dthresh: float, optional

        :raises ValueError: [description]

        Returns a *driver* object that drives the attached vehicle to a 
        sequence of random waypoints.

        The driver is connected to the vehicle by::

            Vehicle(control=driver)
            
            veh = Vehicle()
            veh.control = driver
        
        The waypoints are positioned inside a rectangular region defined by
        the vehicle that is controlled.
          - D scalar; X: -D to +D, Y: -D to +D
              - D (1x2); X: -D(1) to +D(1), Y: -D(2) to +D(2)
        %     - D (1x4); X: D(1) to D(2), Y: D(3) to D(4)
        
        % Options::
        % 'speed',S      Speed along path (default 1m/s).
        % 'dthresh',D    Distance from goal at which next goal is chosen.
        %
        % See also Vehicle.
        :seealso: :func:`spatialmath.plotvol2`
        """
        
        # TODO options to specify region, maybe accept a Map object?
        

        self._dim = base.expand_dims(dim)
        
        self._speed = speed
        self._dthresh = dthresh * np.diff(self._dim[0:2])
        self._goal_marker = None
        if goalmarkerstyle is None:
            self._goal_marker_style = {
                'marker': 'D',
                'markersize': 6, 
                'color': 'r',
                }
        else:
            self._goal_marker_style = goalmarkerstyle
        self._headinggain = headinggain
        
        self._d_prev = np.inf
        self._rand = np.random.default_rng(seed)
        self.verbose = True
        self._goal = None
        self._dthresh = dthresh * max(
                self._dim[1] - self._dim[0], 
                self._dim[3] - self._dim[2]
                                    )

        self._veh = None

    def __str__(self):
        """%RandomPath.char Convert to string
        %
        % s = R.char() is a string showing driver parameters and state in in 
        % a compact human readable format. """

        s = 'RandomPath driver object\n'
        s += f"  X {self._dim[0]} : {self._dim[1]}; Y {self._dim[0]} : {self._dim[1]}, dthresh={self._dthresh}\n"
        s += f"  current goal={self._goal}"
        return s

    @property
    def vehicle(self):
        return self._veh

    @vehicle.setter
    def vehicle(self, v):
        """
        Connect to the vehicle under control

        :param v: [description]
        :type v: [type]
        """
        self._veh = v

    def init(self, ax=None):
        """
        Initialize random path driving agent
        
        %RandomPath.init Reset random number generator
        %
        % R.init() resets the random number generator used to create the waypoints.
        % This enables the sequence of random waypoints to be repeated.
        %
        % Notes::
        % - Called by Vehicle.run.
        %
        % See also RANDSTREAM.
        """
        self._goal = None
        # delete(driver.h_goal);   % delete the goal
        # driver.h_goal = [];
        if ax is not None:
            self._goal_marker = plt.plot(np.nan, np.nan, **self._goal_marker_style)[0]

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
        if d < self._dthresh:
            self._new_goal()
        # elif d > 2 * self._d_prev:
        #     self.choose_goal()
        # self._d_prev = d

        speed = self._speed

        goal_heading = atan2(
            self._goal[1] - self._veh._x[1], 
            self._goal[0] - self._veh._x[0]
                )
        delta_heading = base.angdiff(goal_heading, self._veh._x[2])

        return np.r_[speed, self._headinggain * delta_heading]

    ## private method, invoked from demand() to compute a new waypoint
    
    def _new_goal(self):
        
        # choose a uniform random goal within inner 80% of driving area
        while True:
            r = self._rand.uniform(0.1, 0.9)
            gx = self._dim[0:2] @ np.r_[r, 1-r]

            r = self._rand.uniform(0.1, 0.9)
            gy = self._dim[2:4] @ np.r_[r, 1-r]

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

class PurePursuit(VehicleDriver):

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


