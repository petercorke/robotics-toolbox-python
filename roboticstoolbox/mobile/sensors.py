from abc import ABC
import numpy as np
import scipy as sp
from math import pi, sin, cos
import matplotlib.pyplot as plt
from spatialmath import base
import roboticstoolbox as rtb
from collections.abc import Iterable

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


class SensorBase(ABC):
    # TODO, pose option, wrt vehicle

    # robot
    # map

    # verbose

    # ls
    # animate     # animate sensor measurements
    # interval    # measurement return subsample factor
    # fail
    # delay

    def __init__(
        self,
        robot,
        map,
        every=1,
        fail=[],
        plot=False,
        delay=0.1,
        seed=0,
        verbose=False,
    ):
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
        self._fail = fail

        self._verbose = verbose

        self.delay = 0.1

        self._animate = plot

        self._seed = seed
        self.init()

    def init(self):
        """
        Initialize sensor (superclass)

        - reseed the random number generator
        - reset the counter for handling the ``every`` and ``fail`` options
        """
        self._random = np.random.default_rng(self._seed)
        self._count = 0

    def __str__(self):
        """
        Convert sensor parameters to a string (superclass)
        %
        # s = self.char() is a string showing sensor parameters in
        # a compact human readable format.
        """
        s = f"{self.__class__.__name__} sensor class\n"
        s += "  " + str(self.map)
        return s

    def __repr__(self):
        return str(self)

    @property
    def robot(self):
        """
        Robot associated with sensor (superclass)

        :return: robot
        :rtype: :class:`VehicleBase` subclass
        """
        return self._robot

    @property
    def map(self):
        """
        Landmark map associated with sensor (superclass)

        :return: robot
        :rtype: :class:`LandmarkMap`
        """
        return self._map

    @property
    def random(self):
        """
        Get private random number generator (superclass)

        :return: NumPy random number generator
        :rtype: :class:`numpy.random.Generator`

        Has methods including:

            - :meth:`integers(low, high, size, endpoint) <numpy.random.Generator.integers>`
            - :meth:`random(size) <numpy.random.Generator.random>`
            - :meth:`uniform(low, high, size) <numpy.random.Generator.uniform>`
            - :meth:`normal(mean, std, size) <numpy.random.Generator.normal>`
            - :meth:`multivariate_normal(mean, covar, size) <numpy.random.Generator.multivariate_normal>`

        The generator is initialized with the seed provided at constructor
        time every time :meth:`init` is called.

        :seealso: :meth:`init`
        """
        return self._random

    @property
    def verbose(self):
        """
        Get verbosity state

        :return: verbosity
        :rtype: bool
        """
        return self._verbose

    def plot(self, id):
        """
        Plot sensor observation

        :param id: landmark id
        :type id: int

        Draws a line from the robot to landmark ``id``.

        .. note::
            - The line is drawn using the ``line_style`` given at constructor time

        """
        pass

        # if isempty(self.ls)
        #     return
        # end

        # h = findobj(gca, 'tag', 'sensor')
        # if isempty(h)
        #     # no sensor line, create one
        #     h = plot(0, 0, self.ls, 'tag', 'sensor')
        # end

        # # there is a sensor line animate it

        # if lm_id == 0
        #     set(h, 'Visible', 'off')
        # else
        #     xi = self.self.map(:,lm_id)
        #     set(h, 'Visible', 'on', 'XData', [self.robot.x[0], xi[0]], 'YData', [self.robot.x[1], xi[1]])
        # end
        # pause(self.delay)

        # drawnow


# ======================================================================== #

# visibility function, for one id, or return list of visible
# covar can be 2x2 or (2,)
# .W property
class RangeBearingSensor(SensorBase):
    def __init__(
        self,
        robot,
        map,
        line_style=None,
        poly_style=None,
        covar=None,
        range=None,
        angle=None,
        plot=False,
        seed=0,
        **kwargs,
    ):

        r"""
        Range and bearing angle sensor

        :param robot: model of robot carrying the sensor
        :type robot: :class:`VehicleBase` subclass
        :param map: map of landmarks
        :type map: :class:`LandmarkMap` instance
        :param polygon: polygon style for sensing region, see :class:`~spatialmath.base.graphics.plot_polygon`, defaults to None
        :type polygon: dict, optional
        :param covar: covariance matrix for sensor readings, defaults to None
        :type covar: ndarray(2,2), optional
        :param range: maximum range :math:`r_{max}` or range span :math:`[r_{min}, r_{max}]`, defaults to None
        :type range: float or array_like(2), optional
        :param angle: angular field of view, from :math:`[-\theta, \theta]` defaults to None
        :type angle: float, optional
        :param plot: animate the sensor beams, defaults to False
        :type plot: bool, optional
        :param seed: random number seed, defaults to 0
        :type seed: int, optional
        :param kwargs: arguments passed to :class:`SensorBase`

        Sensor object that returns the range and bearing angle :math:`(r,
        \beta)` to a point landmark from a robot-mounted sensor.  The sensor
        measurements are corrupted with zero-mean Gaussian noise with covariance
        ``covar``.

        The sensor can have a maximum range, or a minimum and maximum range. The
        sensor can also have a restricted angular field of view.

        The sensing region can be displayed by setting the polygon parameter
        which can show an outline or a filled polygon.  This is updated every
        time :meth:`reading` is called, based on the current configuration of
        the ``robot``.

        .. runblock:: pycon

            >>> from roboticstoolbox import Bicycle, LandmarkMap, RangeBearingSensor
            >>> from math import pi
            >>> robot = Bicycle()
            >>> map = LandmarkMap(20)
            >>> sensor = RangeBearingSensor(robot, map, range=(0.5, 20), angle=pi/4)
            >>> print(sensor)

        :seealso: :class:`~roboticstoolbox.mobile.LandmarkMap` :class:`~roboticstoolbox.mobile.EKF`
        """

        # TODO change plot option to animate, but RVC3 uses plot

        # call the superclass constructor
        super().__init__(robot, map, **kwargs)

        self._line_style = line_style
        self._poly_style = poly_style

        if covar is None:
            self._W = np.zeros((2, 2))
        elif base.isvector(covar, 2):
            self._W = np.diag(covar)
        elif base.ismatrix(covar, (2, 2)):
            self._W = covar
        else:
            raise ValueError("bad value for covar, must have shape (2,) or (2,2)")

        if range is None:
            self._r_range = None
        elif isinstance(range, Iterable):
            self._r_range = base.getvector(range, 2)
        else:
            self._r_range = [0, range]

        if angle is None:
            self._theta_range = None
        elif isinstance(angle, Iterable):
            self._theta_range = base.getvector(angle, 2)
        else:
            self._theta_range = [-angle, angle]

        self._animate = plot
        self._landmarklog = []

        self._random = np.random.default_rng(seed)

    def __str__(self):
        s = super().__str__()
        s += f"\n  W = {base.array2str(self._W)}\n"

        s += f"  sampled every {self._every} samples\n"
        if self._r_range is not None:
            s += f"  range: ({self._r_range[0]}: {self._r_range[1]})\n"
        if self._theta_range is not None:
            s += f"  angle: ({self._theta_range[0]:.3g}: {self._theta_range[1]:.3g})\n"
        return s.rstrip()

    def init(self):
        """
        Initialize sensor

        - reseed the random number generator
        - reset the counter for handling the ``every`` and ``fail`` options
        - reset the landmark log
        - initalize plots

        :seealso: :meth:`SensorBase.init`
        """
        super().init()
        self._landmarklog = []

        if self._animate:
            self.map.plot()

    @property
    def W(self):
        """
        Get sensor covariance

        :return: sensor covariance
        :rtype: ndarray(2,2)

        Returns the value of the sensor covariance matrix passed to
        the constructor.
        """
        return self._covar

    def reading(self):
        r"""
        Choose landmark and return observation

        :return: range and bearing angle to a landmark, and landmark id
        :rtype: ndarray(2), int

        Returns an observation of a random visible landmark (range, bearing) and
        the ``id`` of that landmark. The landmark is chosen randomly from the
        set of all visible landmarks, those within the angular field of view and
        range limit.

        If constructor argument ``every`` is set then only return a valid
        reading on every ``every`` calls.

        If constructor argument ``fail`` is set then do not return a reading
        during that specified time interval.

        If no valid reading is available then return (None, None)

        .. runblock:: pycon

            >>> from roboticstoolbox import Bicycle, LandmarkMap, RangeBearingSensor
            >>> from math import pi
            >>> robot = Bicycle()
            >>> map = LandmarkMap(20)
            >>> sensor = RangeBearingSensor(robot, map, range=(0.5, 20), angle=pi/4)
            >>> print(sensor.reading())

        .. note::

            - Noise with covariance ``W`` (set by constructor) is added to the
              reading
            - If ``animate`` option is set then show a line from the vehicle to
              the landmark
                - If ``animate`` option set and the angular and distance limits
                  are set then display the sensor field of view as a polygon.

        :seealso: :meth:`h`
        """

        # TODO probably should return K=0 to indicated invalid

        # model a sensor that emits readings every interval samples
        self._count += 1

        # check conditions for NOT returning a value
        z = []
        lm_id = -1
        # sample interval
        if self._count % self._every != 0:
            self._landmarklog.append(lm_id)
            return (None, None)

        # simulated failure, fail is a list of 2-tuples giving (start,end) times
        # for a sensor failure
        if self._fail is not None:
            if any([start <= self._count < end for start, end in self._fail]):
                self._landmarklog.append(lm_id)
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

        zk = self.visible()
        if len(zk) > 1:
            # more than 1 visible landmark, pick a random one
            i = self._random.integers(len(zk))
            z = zk[i][0]
            lm_id = zk[i][1]
            if self.verbose:
                print(f"Sensor:: feature {lm_id}: ({z[0]}, {z[1]})")
        elif len(zk) == 1:
            # just 1 visible landmark
            z = zk[0][0]
            lm_id = zk[0][1]
            if self.verbose:
                print(f"Sensor:: feature {lm_id}: ({z[0]}, {z[1]})")
        else:
            if self.verbose:
                print("Sensor:: no features\n")
            self._landmarklog.append(lm_id)
            return (None, None)

        # compute the range and bearing from robot to feature
        # z = self.h(self.robot.x, lm_id)

        if self._animate:
            self.plot(lm_id)

        # add the reading to the landmark log
        self._landmarklog.append(lm_id)

        # add noise with covariance W
        z += self._random.multivariate_normal((0, 0), self._W)

        return z, lm_id

    def visible(self):
        """
        List of all visible landmarks

        :return: list of visible landmarks
        :rtype: list of int

        Return a list of the id of all landmarks that are visible, that is, it
        lies with the sensing range and field of view of the sensor at the
        robot's current configuration.

        :seealso: :meth:`isvisible` :meth:`h`
        """
        # get range/bearing to all landmarks
        z = self.h(self.robot.x)
        zk = [(z, k) for k, z in enumerate(z)]
        # a list of tuples, each tuple is ((range, bearing), k)

        if self._r_range is not None:
            zk = filter(lambda zk: self._r_range[0] <= zk[0][0] <= self._r_range[1], zk)

        if self._theta_range is not None:
            # find all within angular range as well
            zk = filter(
                lambda zk: self._theta_range[0] <= zk[0][1] <= self._theta_range[1], zk
            )

        return list(zk)

    def isvisible(self, id):
        """
        Test if landmark is visible

        :param id: landmark id
        :type id: int
        :return: visibility
        :rtype: bool

        The landmark ``id`` is visible if it lies with the sensing range and
        field of view of the sensor at the robot's current configuration.

        :seealso: :meth:`visible` :meth:`h`
        """
        z = self.h(self.robot.x, id)

        return (
            (self._r_range is None) or self._r_range[0] <= z[0] <= self._r_range[1]
        ) and (
            (self._theta_range is None)
            or self._theta_range[0] <= z[1] <= self._theta_range[1]
        )

    def h(self, x, landmark=None):
        r"""
        Landmark observation function

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3), array_like(N,3)
        :param landmark: landmark id or position, defaults to None
        :type landmark: int or array_like(2), optional
        :return: range and bearing angle to landmark math:`(r,\beta)`
        :rtype: ndarray(2) or ndarray(N,2)

        Return the range and bearing to a landmark:

        - ``.h(x)`` is range and bearing to all landmarks, one row per landmark
        - ``.h(x, id)`` is range and bearing to landmark ``id``
        - ``.h(x, p)`` is range and bearing to landmark with coordinates ``p``

        .. runblock:: pycon

            >>> from roboticstoolbox import Bicycle, LandmarkMap, RangeBearingSensor
            >>> from math import pi
            >>> robot = Bicycle()
            >>> map = LandmarkMap(20)
            >>> sensor = RangeBearingSensor(robot, map, range=(0.5, 20), angle=pi/4)
            >>> z = sensor.h((1, 2, pi/2), 3)
            >>> print(z)

        .. note::
            - Noise with covariance (property ``W``) is added to each row of ``z``.
            - Performs fast vectorized operation where ``x`` is an ndarray(n,3).
            - The landmark is assumed to be visible, field of view and range limits are not
              applied.

        :seealso: :meth:`reading` :meth:`Hx` :meth:`Hw` :meth:`Hp`
        """
        # get the landmarks, one per row

        if isinstance(x, np.ndarray) and x.ndim == 2:
            # x is Nx3 set of vehicle states, do vectorized form
            # used by particle filter
            x, y, t = x.T
        else:
            x, y, t = x

        if landmark is None:
            # self.h(XV)   all landmarks
            dx = self.map.landmarks[0, :] - x
            dy = self.map.landmarks[1, :] - y
        elif base.isinteger(landmark):
            # landmark id
            # self.h(XV, JF)
            xlm = self.map[landmark]
            dx = xlm[0] - x
            dy = xlm[1] - y
        else:
            # landmark position
            # self.h(XV, XF)
            xlm = base.getvector(landmark, 2)
            dx = xlm[0] - x
            dy = xlm[1] - y

        # compute range and bearing (Vectorized code)

        z = np.c_[
            np.sqrt(dx**2 + dy**2), base.angdiff(np.arctan2(dy, dx), t)
        ]  # range & bearing as columns

        if z.shape[0] == 1:
            return z[0]
        else:
            return z

    def Hx(self, x, landmark):
        r"""
        Jacobian dh/dx

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param arg: landmark id or coordinate
        :type arg: int or array_like(2)
        :return: Jacobian matrix
        :rtype: ndarray(2,3)

        Compute the Jacobian of the observation function with respect to vehicle
        configuration :math:`\partial h/\partial x`

        - ``sensor.Hx(q, id)`` is Jacobian for landmark ``id``
        - ``sensor.h(q, p)`` is Jacobian for landmark with coordinates ``p``

        :seealso: :meth:`h` :meth:`Hp` :meth:`Hw`
        """

        if base.isinteger(landmark):
            # landmark index provided
            xf = self.map[landmark]
        else:
            # assume it is a coordinate
            xf = base.getvector(landmark, 2)

        Delta = xf - x[:2]
        r = base.norm(Delta)
        # fmt: off
        return np.array([
            [-Delta[0] / r,    -Delta[1] / r,      0],
            [ Delta[1] / r**2, -Delta[0] / r**2,  -1],
                ])
        # fmt: on

    def Hp(self, x, landmark):
        r"""
        Jacobian dh/dp

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param arg: landmark id or coordinate
        :type arg: int or array_like(2)
        :return: Jacobian matrix
        :rtype: ndarray(2,2)

        Compute the Jacobian of the observation function with respect
        to landmark position :math:`\partial h/\partial p`

        - ``sensor.Hp(x, id)`` is Jacobian for landmark ``id``
        - ``sensor.Hp(x, p)`` is Jacobian for landmark with coordinates ``p``

        :seealso: :meth:`h` :meth:`Hx` :meth:`Hw`
        """
        if base.isinteger(landmark):
            xf = self.map.landmark(landmark)
        else:
            xf = landmark
        x = base.getvector(x, 3)

        Delta = xf - x[:2]
        r = base.norm(Delta)
        # fmt: off
        return np.array([
            [ Delta[0] / r,      Delta[1] / r],
            [-Delta[1] / r**2,   Delta[0] / r**2],
            ])
        # fmt: on

    def Hw(self, x, landmark):
        r"""
        Jacobian dh/dw

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param arg: landmark id or coordinate
        :type arg: int or array_like(2)
        :return: Jacobian matrix
        :rtype: ndarray(2,2)

        Compute the Jacobian of the observation function with respect
        to sensor noise :math:`\partial h/\partial w`

        - ``sensor.Hw(x, id)`` is Jacobian for landmark ``id``
        - ``sensor.Hw(x, p)`` is Jacobian for landmark with coordinates ``p``

        .. note:: ``x`` and ``landmark`` are not used to compute this.

        :seealso: :meth:`h` :meth:`Hx` :meth:`Hp`
        """
        return np.eye(2)

    def g(self, x, z):
        r"""
        Landmark position from sensor observation

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param z: landmark observation :math:`(r, \beta)`
        :type z: array_like(2)
        :return: landmark position :math:`(x, y)`
        :rtype: ndarray(2)

        Compute the world coordinate  of a landmark given
        the observation ``z`` from a vehicle state with ``x``.

        :seealso: :meth:`h` :meth:`Gx` :meth:`Gz`
        """

        range = z[0]
        bearing = z[1] + x[2]  # bearing angle in vehicle frame

        # fmt: off
        return np.r_[
            x[0] + range * cos(bearing),
            x[1] + range * sin(bearing)
                ]
        # fmt: on

    def Gx(self, x, z):
        r"""
        Jacobian dg/dx

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param z: landmark observation :math:`(r, \beta)`
        :type z: array_like(2)
        :return: Jacobian matrix
        :rtype: ndarray(2,3)

        Compute the Jacobian of the landmark position function with respect
        to landmark position :math:`\partial g/\partial x`

        :seealso: :meth:`g`
        """
        theta = x[2]
        r, bearing = z

        # fmt: off
        return np.array([
            [1,   0,   -r*sin(theta + bearing)],
            [0,   1,    r*cos(theta + bearing)],
                ])
        # fmt: on

    def Gz(self, x, z):
        r"""
        Jacobian dg/dz

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param z: landmark observation :math:`(r, \beta)`
        :type z: array_like(2)
        :return: Jacobian matrix
        :rtype: ndarray(2,2)

        Compute the Jacobian of the landmark position function with respect
        to sensor observation :math:`\partial g/\partial z`

        :seealso: :meth:`g`
        """
        theta = x[2]
        r, bearing = z
        # fmt: off
        return np.array([
            [cos(theta + bearing),   -r * sin(theta + bearing)],
            [sin(theta + bearing),    r * cos(theta + bearing)],
            ])
        # fmt: on


if __name__ == "__main__":

    from roboticstoolbox import Bicycle, LandmarkMap, RangeBearingSensor
    from math import pi

    robot = Bicycle()
    map = LandmarkMap(20)
    sensor = RangeBearingSensor(robot, map, range=(0.5, 20), angle=pi / 4)
    print(sensor.reading())
    print(sensor.visible())
    print(sensor.isvisible(3))
    print(sensor.isvisible(4))
