from abc import ABC
import numpy as np
import scipy as sp
from math import pi, sin, cos
import matplotlib.pyplot as plt
from spatialmath import base
import roboticstoolbox as rtb


"""
Sensor Sensor superclass

An abstract superclass to represent robot navigation sensorself.

Methods::
  plot        plot a line from robot to map feature
  display     print the parameters in human readable form
  char        convert to string

Properties::
robot   The Vehicle object on which the sensor is mounted
map     The PointMap object representing the landmarks around the robot

Reference::

  Robotics, Vision & Control,
  Peter Corke,
  Springer 2011

See also RangeBearingSensor, EKF, Vehicle, Landmarkself.

"""

class Sensor(ABC):
    # TODO, pose option, wrt vehicle

        # robot
        # map
        
        # verbose
        
        # ls
        # animate     # animate sensor measurements
        # interval    # measurement return subsample factor
        # fail
        # delay
        

    def __init__(self, robot, map, every=1, fail=[], animate=False, delay=0.1, verbose=False):
        """Sensor.Sensor Sensor object constructor
        %
        # S = Sensor(VEHICLE, MAP, OPTIONS) is a sensor mounted on a vehicle
        # described by the Vehicle subclass object VEHICLE and observing landmarks
        # in a map described by the LandmarkMap class object self.
        %
        # Options::
        # 'animate'    animate the action of the laser scanner
        # 'ls',LS      laser scan lines drawn with style ls (default 'r-')
        # 'skip', I    return a valid reading on every I'th call
        # 'fail',T     sensor simulates failure between timesteps T=[TMIN,TMAX]
        %
        # Notes::
        # - Animation shows a ray from the vehicle position to the selected
        #   landmark.
        """
        self._robot = robot
        self._map = map
        self._every = every
        self._fail =  fail
        
        self._count = 1
        self._verbose = verbose

        self.delay = 0.1

        self._animate = animate

        self._rand = np.random.default_rng()


    def __str__(self):
        """
        Convert sensor parameters to a string
        %
        # s = self.char() is a string showing sensor parameters in
        # a compact human readable format.
        """
        s = f"{self.__class__.__name__} sensor class\n"
        s += "  " + str(self.map)
        return s

    @property
    def robot(self):
        return self._robot

    @property
    def map(self):
        return self._map

    @property
    def sensor(self):
        return self._sensor

    @property
    def verbose(self):
        return self._verbose
    
    def plot(s, jf):
        """Sensor.plot Plot sensor reading
        %
        # self.plot(J) draws a line from the robot to the J'th map feature.
        %
        # Notes::
        # - The line is drawn using the linestyle given by the property ls
        # - There is a delay given by the property delay
        """
        # if isempty(self.ls)
        #     return
        # end
        
        # h = findobj(gca, 'tag', 'sensor')
        # if isempty(h)
        #     # no sensor line, create one
        #     h = plot(0, 0, self.ls, 'tag', 'sensor')
        # end
        
        # # there is a sensor line animate it
        
        # if jf == 0
        #     set(h, 'Visible', 'off')
        # else
        #     xi = self.self.map(:,jf)
        #     set(h, 'Visible', 'on', 'XData', [self.robot.x[0], xi[0]], 'YData', [self.robot.x[1], xi[1]])
        # end
        # pause(self.delay)

        # drawnow
        pass



"""RangeBearingSensor Range and bearing sensor class

A concrete subclass of the Sensor class that implements a range and bearing
angle sensor that provides robot-centric measurements of landmark points in 
the world. To enable this it holds a references to a map of the world (LandmarkMap object)
and a robot (Vehicle subclass object) that moves in SE[1].

The sensor observes landmarks within its angular field of view between
the minimum and maximum range.

Methods::

reading   range/bearing observation of random landmark
h         range/bearing observation of specific landmark
Hx        Jacobian matrix with respect to vehicle pose dh/dx 
Hp        Jacobian matrix with respect to landmark position dh/dp 
Hw        Jacobian matrix with respect to noise dh/dw
-
g         feature position given vehicle pose and observation
Gx        Jacobian matrix with respect to vehicle pose dg/dx 
Gz        Jacobian matrix with respect to observation dg/dz

Properties (read/write)::
W            measurement covariance matrix (2x2)
interval     valid measurements returned every interval'th call to reading()
landmarklog  time history of observed landmarks

Reference::

  Robotics, Vision & Control, Chap 6,
  Peter Corke,
  Springer 2011

See also Sensor, Vehicle, LandmarkMap, EKF."""

# ======================================================================== #

class RangeBearingSensor(Sensor):

    # properties
    #     W           # measurment covariance
    #     r_range     # range limits
    #     theta_range # angle limits

    #     randstream  # random stream just for Sensors
        
    #     landmarklog  # time history of observed landmarks        
    # end

    # properties (SetAccess = private)
    #     count       # number of reading()s
    # end


    def __init__(self, robot, map, 
            line_style=None,
            poly_style=None,
            covar=None, 
            range=None,
            angle=None,
            **kwargs):
        """RangeBearingSensor.RangeBearingSensor Range and bearing sensor constructor
        %
        # S = RangeBearingSensor(VEHICLE, MAP, OPTIONS) is an object
        # representing a range and bearing angle sensor mounted on the Vehicle
        # subclass object VEHICLE and observing an environment of known landmarks
        # represented by the LandmarkMap object self.  The sensor covariance is W
        # (2x2) representing range and bearing covariance.
        %
        # The sensor has specified angular field of view and minimum and maximum
        # range.
        %
        # Options::
        # 'covar',W               covariance matrix (2x2)
        # 'range',xmax            maximum range of sensor
        # 'range',[xmin xmax]     minimum and maximum range of sensor
        # 'angle',TH              angular field of view, from -TH to +TH
        # 'angle',[THMIN THMAX]   detection for angles betwen THMIN
        #                         and THMAX
        # 'skip',K                return a valid reading on every K'th call
        # 'fail',[TMIN TMAX]      sensor simulates failure between 
        #                         timesteps TMIN and TMAX
        # 'animate'               animate sensor readings
        %
        # See also options for Sensor constructor.
        %
        # See also RangeBearingSensor.reading, Sensor.Sensor, Vehicle, LandmarkMap, EKF.
        """


        # call the superclass constructor
        super().__init__(robot, map, **kwargs)

        self._line_style = line_style
        self._poly_style = poly_style

        if covar is None:
            self._covar = np.zeros((2,2))
        else:
            self._covar = covar
        self._W = self._covar

        if range is None:
            self._r_range = None
        elif len(range) == 1:
            self._r_range = [0, range]
        elif len(range) == 2:
            self._r_range = range

        if angle is None:
            self._theta_range = None
        elif len(angle) == 1:
            self._theta_range = [-angle, angle]
        elif len(angle) == 2:
            self._theta_range = angle

        self._landmarklog = []


    def init(self):
        self._landmarklog = []
        self._count = 1
    
    def selectFeature(self):
        return self.randstream.randi(self._sensor.nlandmarks)


    def reading(self):
        """
        Choose landmark and return observation
        %
        # [Z,K] = self.reading() is an observation of a random visible landmark where
        # Z=[R,THETA] is the range and bearing with additive Gaussian noise of
        # covariance W (property W). K is the index of the map feature that was
        # observed.
        %
        # The landmark is chosen randomly from the set of all visible landmarks,
        # those within the angular field of view and range limitself.  If no valid
        # measurement, ie. no features within range, interval subsampling enabled
        # or simulated failure the return is Z=[] and K=0.
        %
        # Notes::
        # - Noise with covariance W (property W) is added to each row of Z.
        # - If 'animate' option set then show a line from the vehicle to the
        #   landmark
        # - If 'animate' option set and the angular and distance limits are set
        #   then display that region as a shaded polygon.
        # - Implements sensor failure and subsampling if specified to constructor.
        %
        # See also RangeBearingSensor.h.
        """
        
        # TODO probably should return K=0 to indicated invalid
        
        # model a sensor that emits readings every interval samples
        self._count += 1

        # check conditions for NOT returning a value
        z = []
        jf = 0
        # sample interval
        if self._count % self._every != 0:
            return (None, None)

        # simulated failure, fail is a list of 2-tuples giving (start,end) times
        # for a sensor failure
        if self._fail is not None:
            if any([start <= self._count < end for start, end in self._fail]):
                return (None, None)
        
        # create a polygon to indicate the active sensing area based on range+angle limits
        # if self.animate && ~isempty(self.theta_range) && ~isempty(self.r_range)
        #     h = findobj(gca, 'tag', 'sensor-area')
        #     if isempty(h)
                
        #         th=linspace(self.theta_range[0], self.theta_range[1], 20)
        #         x = self.r_range[1] * cos(th)
        #         y = self.r_range[1] * sin(th)
        #         if self.r_range[0] > 0
        #             th = flip(th)
        #             x = [x self.r_range[0] * cos(th)]
        #             y = [y self.r_range[0] * sin(th)]
        #         else
        #             x = [x 0]
        #             y = [y 0]
        #         end
        #         # no sensor zone, create one
        #         plot_poly([x; y], 'fillcolor', 'r', 'alpha', 0.1, 'edgecolor', 'none', 'animate', 'tag', 'sensor-area')
        #     else
        #         hg = get(h, 'Parent')
        #         plot_poly(h, self.robot.x)


        # if range and bearing angle limits are in place look for
        # any landmarks that match criteria

        # get range/bearing to all landmarks, one per row
        z = self.h(self.robot.x)
        zk = [(z, k) for k, z in enumerate(z)]
        # a list of tuples, each tuple is ((range, bearing), k)
        
        if self._r_range is not None:
            zk = filter(lambda zk: self._r_range[0] <= zk[0][0] <= self._r_range[1], zk)

        if self._theta_range is not None:
            # find all within angular range as well
            zk = filter(lambda zk: self._theta_range[0] <= zk[0][1] <= self._theta_range[1], zk)
            
        if len(zk) > 1:
            # more than 1 in range, pick a random one
            i = self._rand.integers(len(zk))
            z = zk[i][0]
            jf = zk[i][1]
            if self.verbose:
                print(f"Sensor:: feature {jf}: ({z[0]}, {z[1]})")
        else:
            if self.verbose:
                print('Sensor:: no features\n')
            return (None, None)

        # compute the range and bearing from robot to feature
        # z = self.h(self.robot.x, jf)  
        
        if self._animate:
            self.plot(jf)
        
        # add the reading to the landmark log
        self._landmarklog.append(jf)
    
        return z


    def h(self, xv, jf=None):
        """
        Landmark range and bearing
        %
        # Z = self.h(X, K) is a sensor observation (1x2), range and bearing, from vehicle at 
        # pose X (1x3) to the K'th landmark.
        %
        # Z = self.h(X, P) as above but compute range and bearing to a landmark at coordinate P.
        %
        # Z = self.h(X) as above but computes range and bearing to all
        # map featureself.  Z has one row per landmark.
        %
        # Notes::
        # - Noise with covariance W (propertyW) is added to each row of Z.
        # - Supports vectorized operation where XV (Nx3) and Z (Nx2).
        # - The landmark is assumed visible, field of view and range liits are not
        #   applied.
        %
        # See also RangeBearingSensor.reading, RangeBearingSensor.Hx, RangeBearingSensor.Hw, RangeBearingSensor.Hp.
        """
        # get the landmarks, one per row
        if jf is None:
            # self.h(XV)   all landmarks
            dx = self.map.x - xv[0]
            dy = self.map.y - xv[1]
        elif base.isinteger(jf):
            # self.h(XV, JF)
            xlm = self.map.landmark(jf)
            dx = xlm[0] - xv[0]
            dy = xlm[1] - xv[1]
        else:
            # self.h(XV, XF)
            xlm = base.getvector(jf, 2)
            dx = xlm[0] - xv[0]
            dy = xlm[1] - xv[1]

        # compute range and bearing (Vectorized code)

        z = np.c_[
                np.sqrt(dx**2 + dy**2), 
                base.angdiff(np.arctan2(dy, dx), xv[2]) 
            ]  # range & bearing as columns

        # add noise with covariance W
        z += self._rand.normal(size=z.shape) @ sp.linalg.sqrtm(self._W)

        if z.shape[0] == 1:
            return z[0]
        else:
            return z

    def Hx(self, xv, jf):
        """
        Jacobian dh/dx
        %
        # J = self.Hx(X, K) returns the Jacobian dh/dx (2x3) at the vehicle
        # state X (3x1) for map landmark K.
        %
        # J = self.Hx(X, P) as above but for a landmark at coordinate P.
        %
        # See also RangeBearingSensor.h.
        """
        if base.isinteger(jf):
            # landmark index provided
            xf = self.sensor.landmark(jf)
        else:
            # assume it is a coordinate
            xf = base.getvector(jf, 2)

        Delta = xf - xv[0:2]
        r = base.norm(Delta)
        return np.array([
            [-Delta[0] / r,    -Delta[1] / r,      0],
            [ Delta[1] / r**2, -Delta[0] / r**2,  -1],
                ])

    def Hp(self, xv, jf):
        """
        Jacobian dh/dp
        %
        # J = self.Hp(X, K) is the Jacobian dh/dp (2x2) at the vehicle
        # state X (3x1) for map landmark K.
        %
        # J = self.Hp(X, P) as above but for a landmark at coordinate P (1x2).
        %
        # See also RangeBearingSensor.h.
        """
        if base.isinteger(jf):
            xf = self.map.landmark(jf)
        else:
            xf = jf

        Delta = xf - xv[0:2]
        r = base.norm(Delta)
        return np.array([
            [ Delta([0]) / r,      Delta([1]) / r],
            [-Delta([1]) / r**2,    Delta([0]) / r**2],
            ])

    def Hw(self, xv, jf):
        """
        Jacobian dh/dw
        %
        # J = self.Hw(X, K) is the Jacobian dh/dw (2x2) at the vehicle
        # state X (3x1) for map landmark K.
        %
        # See also RangeBearingSensor.h.
        """
        return np.eye(2)

    def g(self, xv, z):
        """
        Compute landmark location
        %
        # P = self.g(X, Z) is the world coordinate (2x1) of a feature given
        # the observation Z (1x2) from a vehicle state with X (3x1).
        %
        # See also RangeBearingSensor.Gx, RangeBearingSensor.Gz.
        """
        range = z[0]
        bearing = z[1] + xv[2]  # bearing angle in vehicle frame

        return np.r_[
            xv[0] + range * cos(bearing),
            xv[1] + range * sin(bearing)
                ]

    def Gx(self, xv, z):
        """RangeBearingSensor.Gxv Jacobian dg/dx
        %
        # J = self.Gx(X, Z) is the Jacobian dg/dx (2x3) at the vehicle state X (3x1) for
        # sensor observation Z (2x1).
        %
        # See also RangeBearingSensor.g.
        """
        theta = xv[2]
        r = z[0]
        bearing = z[1]

        return np.array([
            [1,   0,   -r*sin(theta + bearing)],
            [0,   1,    r*cos(theta + bearing)],
                ])

    def Gz(self, xv, z):
        """
        Jacobian dg/dz
        %
        # J = self.Gz(X, Z) is the Jacobian dg/dz (2x2) at the vehicle state X (3x1) for
        # sensor observation Z (2x1).
        %
        # See also RangeBearingSensor.g.
        """
        theta = xv[2]
        r = z[0]
        bearing = z[1]
        return np.array([
            [cos(theta + bearing),   -r*sin(theta + bearing)],
            [sin(theta + bearing),    r*cos(theta + bearing)],
            ])

    def __str__(self):
        s = super().__str__()
        s += f"\n  W = {self._W.tolist()}\n"

        s += f"  sampled every {self._every} samples\n"
        if self._r_range is not None:
            s += f"  range: {self._r_range[0]} to {self._r_range[0]}\n"
        if self._theta_range is not None:
            s += f"  angle: {self._theta_range[0]} to {self._theta_range[0]}\n"
        return s.rstrip()

if __name__ == "__main__":
    
    pass
    