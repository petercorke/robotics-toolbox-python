from __future__ import annotations

import warnings
from abc import ABC
from collections.abc import Iterable
from math import cos, pi, sin
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from spatialmath import base

from roboticstoolbox.tools.types import ArrayLike, NDArray

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
    """
    Superclass for sensors
    """

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
        robot: Any,
        map: Any,
        every: int = 1,
        fail: list[tuple[int, int]] = [],
        plot: bool = False,
        delay: float = 0.1,
        seed: int | None = 0,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
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
        if kwargs:
            raise TypeError(
                f"SensorBase.__init__() got unexpected keyword argument(s): "
                f"{', '.join(kwargs)}"
            )

        self._robot = robot
        self._map = map
        self._every = every
        self._fail = fail

        self._verbose = verbose

        self.delay = 0.1

        self._animate = plot

        self._seed = seed
        self.init()

    def init(self) -> None:
        """
        Initialize sensor (superclass)

        - reseed the random number generator
        - reset the counter for handling the ``every`` and ``fail`` options
        """
        self._random = np.random.default_rng(self._seed)
        self._count = 0

    def __str__(self) -> str:
        """
        Convert sensor parameters to a string (superclass)
        %
        # s = self.char() is a string showing sensor parameters in
        # a compact human readable format.
        """
        s = f"{self.__class__.__name__} sensor class\n"
        s += "  " + str(self.map)
        return s

    def __repr__(self) -> str:
        return str(self)

    @property
    def robot(self) -> Any:
        """
        Robot associated with sensor (superclass)

        :return: robot
        :rtype: :class:`VehicleBase` subclass
        """
        return self._robot

    @property
    def map(self) -> Any:
        """
        Landmark map associated with sensor (superclass)

        :return: robot
        :rtype: ``LandmarkMap``
        """
        return self._map

    @property
    def random(self) -> np.random.Generator:
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

        .. seealso:: :meth:`init`
        """
        return self._random

    @property
    def verbose(self) -> bool:
        """
        Get verbosity state

        :return: verbosity
        :rtype: bool
        """
        return self._verbose

    def plot(self, id: int) -> None:
        """
        Plot sensor observation

        :param id: landmark id

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
    """
    Range and bearing angle sensor
    """

    def __init__(
        self,
        robot: Any,
        map: Any,
        line_style: dict[str, Any] | None = None,
        poly_style: dict[str, Any] | None = None,
        covar: ArrayLike | None = None,
        range: float | ArrayLike | None = None,
        angle: float | ArrayLike | None = None,
        plot: bool = False,
        seed: int | None = 0,
        **kwargs: Any,
    ) -> None:
        r"""
        Range and bearing angle sensor

        :param robot: model of robot carrying the sensor
        :type robot: :class:`VehicleBase` subclass
        :param map: map of landmarks
        :type map: ``LandmarkMap`` instance
        :param line_style: style of the line to the observed landmark, defaults to None
        :type line_style: dict, optional
        :param poly_style: style for the sensing region polygon, defaults to None
        :type poly_style: dict, optional
        :param covar: covariance matrix for sensor readings, or the variance
            of each of range and bearing, defaults to None (noise free)
        :type covar: ndarray(2,2) or array_like(2), optional
        :param range: maximum range :math:`r_{max}` or range span :math:`[r_{min}, r_{max}]`, defaults to None
        :type range: float or array_like(2), optional
        :param angle: angular field of view, either :math:`\theta` for the
            span :math:`[-\theta, \theta]` or the span :math:`[\theta_{min}, \theta_{max}]`,
            defaults to None
        :type angle: float or array_like(2), optional
        :param plot: plot the map and a line to each observed landmark, defaults to False
        :param seed: random number seed, defaults to 0
        :param kwargs: arguments passed to :class:`SensorBase`

        Sensor object that returns the range and bearing angle :math:`(r,
        \beta)` to a point landmark from a robot-mounted sensor.  The sensor
        measurements are corrupted with zero-mean Gaussian noise with covariance
        ``covar``.

        The sensor can have a maximum range, or a minimum and maximum range. The
        sensor can also have a restricted angular field of view.

        If ``plot`` is True the landmark map is displayed and a line is drawn
        from the robot to the landmark observed each time :meth:`reading` is
        called.

        ``line_style`` styles the landmark beam and ``poly_style`` styles the
        sensing region when ``plot=True``. The region is shown when both range
        and angle limits are supplied.

        .. runblock:: pycon

            >>> from roboticstoolbox import Bicycle, LandmarkMap, RangeBearingSensor
            >>> from math import pi
            >>> robot = Bicycle()
            >>> map = LandmarkMap(20)
            >>> sensor = RangeBearingSensor(robot, map, range=(0.5, 20), angle=pi/4)
            >>> print(sensor)

        .. seealso:: ``LandmarkMap`` :class:`~roboticstoolbox.mobile.EKF`

        .. deprecated:: 1.4.0
            The ``animate`` keyword, use ``plot`` instead.
        """

        animate = kwargs.pop("animate", None)
        if animate is not None:
            warnings.warn(
                "animate is deprecated, use plot instead",
                DeprecationWarning,
                stacklevel=2,
            )
            plot = animate

        # call the superclass constructor
        super().__init__(robot, map, **kwargs)

        self._line_style = line_style
        self._poly_style = poly_style

        self._W: NDArray
        if covar is None:
            self._W = np.zeros((2, 2))
        elif base.isvector(covar, 2):
            self._W = np.diag(covar)
        elif base.ismatrix(covar, (2, 2)):
            self._W = np.asarray(covar)
        else:
            raise ValueError("bad value for covar, must have shape (2,) or (2,2)")

        self._r_range: NDArray | list[float] | None
        if range is None:
            self._r_range = None
        elif isinstance(range, Iterable):
            self._r_range = base.getvector(range, 2)
        else:
            self._r_range = [0, range]

        self._theta_range: NDArray | list[float] | None
        if angle is None:
            self._theta_range = None
        elif isinstance(angle, Iterable):
            self._theta_range = base.getvector(angle, 2)
        else:
            self._theta_range = [-angle, angle]

        self._animate = plot
        self._landmarklog: list[int] = []

        self._random = np.random.default_rng(seed)
        self._plot_ax: Axes | None = None
        self._plot_line: Line2D | None = None
        self._plot_poly: Polygon | None = None
        if self._animate:
            self._init_plot()

    def _init_plot(self) -> None:
        """Create map axes and discard previous sensor artists."""
        if self._plot_line is not None and self._plot_line.axes is not None:
            self._plot_line.remove()
        if self._plot_poly is not None and self._plot_poly.axes is not None:
            self._plot_poly.remove()
        self.map.plot()
        self._plot_ax = plt.gca()
        self._plot_line = None
        self._plot_poly = None

    def __str__(self) -> str:
        s = super().__str__()
        s += f"\n  W = {base.array2str(self._W)}\n"

        s += f"  sampled every {self._every} samples\n"
        if self._r_range is not None:
            s += f"  range: ({self._r_range[0]}: {self._r_range[1]})\n"
        if self._theta_range is not None:
            s += f"  angle: ({self._theta_range[0]:.3g}: {self._theta_range[1]:.3g})\n"
        return s.rstrip()

    def init(self) -> None:
        """
        Initialize sensor

        - reseed the random number generator
        - reset the counter for handling the ``every`` and ``fail`` options
        - reset the landmark log
        - initialize plots

        .. seealso:: :meth:`SensorBase.init`
        """
        super().init()
        self._landmarklog = []

        if self._animate:
            self._init_plot()

    @property
    def W(self) -> NDArray:
        """
        Get sensor covariance

        :return: sensor covariance
        :rtype: ndarray(2,2)

        Returns the value of the sensor covariance matrix passed to
        the constructor.
        """
        return self._W

    def reading(self) -> tuple[NDArray | None, int | None]:
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

        .. seealso:: :meth:`h`
        """

        # TODO probably should return K=0 to indicated invalid

        # model a sensor that emits readings every interval samples
        self._count += 1

        # check conditions for NOT returning a value
        z: NDArray = np.empty((0,))
        lm_id = -1
        # sample interval
        if self._count % self._every != 0:
            if self._animate:
                self.plot(None)
            self._landmarklog.append(lm_id)
            return (None, None)

        # simulated failure, fail is a list of 2-tuples giving (start,end) times
        # for a sensor failure
        if self._fail is not None:
            if any([start <= self._count < end for start, end in self._fail]):
                if self._animate:
                    self.plot(None)
                self._landmarklog.append(lm_id)
                return (None, None)

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
            if self._animate:
                self.plot(None)
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

    def plot(self, id: int | None) -> None:
        """Update the sensing region and the beam to the observed landmark.

        :param id: landmark index, or None if no landmark is observed
        """
        if self._plot_ax is None:
            self._init_plot()
        ax = self._plot_ax
        if ax is None:
            return  # pragma: no cover

        pose = self.robot.x
        if self._r_range is not None and self._theta_range is not None:
            theta = np.linspace(
                float(self._theta_range[0]), float(self._theta_range[1]), 40
            ) + pose[2]
            direction = np.column_stack((np.cos(theta), np.sin(theta)))
            outer = self._r_range[1] * direction
            if self._r_range[0] > 0:
                inner = self._r_range[0] * direction[::-1]
            else:
                inner = np.zeros((1, 2))
            vertices = np.vstack((outer, inner)) + pose[:2]
            if self._plot_poly is None:
                poly_style: dict[str, Any] = {
                    "facecolor": "red",
                    "alpha": 0.1,
                    "edgecolor": "none",
                }
                poly_style.update(self._poly_style or {})
                self._plot_poly = Polygon(vertices, closed=True, **poly_style)
                ax.add_patch(self._plot_poly)
            else:
                self._plot_poly.set_xy(vertices)

        if self._plot_line is None:
            line_style: dict[str, Any] = {"color": "red", "linewidth": 1}
            line_style.update(self._line_style or {})
            (self._plot_line,) = ax.plot([], [], **line_style)

        if id is None:
            self._plot_line.set_visible(False)
        else:
            landmark = self.map[id]
            self._plot_line.set_data(
                [pose[0], landmark[0]], [pose[1], landmark[1]]
            )
            self._plot_line.set_visible(True)
        ax.figure.canvas.draw_idle()

    def visible(self) -> list[tuple[NDArray, int]]:
        """
        List of all visible landmarks

        :return: list of visible landmarks
        :rtype: list of int

        Return a list of the id of all landmarks that are visible, that is, it
        lies with the sensing range and field of view of the sensor at the
        robot's current configuration.

        .. seealso:: :meth:`isvisible` :meth:`h`
        """
        # get range/bearing to all landmarks
        z = self.h(self.robot.x)
        zk = [(z, k) for k, z in enumerate(z)]
        # a list of tuples, each tuple is ((range, bearing), k)

        if self._r_range is not None:
            r_range = self._r_range
            zk = [t for t in zk if r_range[0] <= t[0][0] <= r_range[1]]

        if self._theta_range is not None:
            # find all within angular range as well
            theta_range = self._theta_range
            zk = [t for t in zk if theta_range[0] <= t[0][1] <= theta_range[1]]

        return zk

    def isvisible(self, id: int) -> bool:
        """
        Test if landmark is visible

        :param id: landmark id
        :return: visibility
        :rtype: bool

        The landmark ``id`` is visible if it lies with the sensing range and
        field of view of the sensor at the robot's current configuration.

        .. seealso:: :meth:`visible` :meth:`h`
        """
        z = self.h(self.robot.x, id)

        return (
            (self._r_range is None) or self._r_range[0] <= z[0] <= self._r_range[1]
        ) and (
            (self._theta_range is None)
            or self._theta_range[0] <= z[1] <= self._theta_range[1]
        )

    def h(self, x: ArrayLike, landmark: int | ArrayLike | None = None) -> NDArray:
        r"""
        Landmark observation function

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3), array_like(N,3)
        :param landmark: landmark id or position, defaults to None
        :type landmark: int or array_like(2), optional
        :return: range and bearing angle to landmark :math:`(r,\beta)`
        :rtype: ndarray(2) or ndarray(N,2)

        Return the range and bearing to a landmark:

        - ``.h(x)`` is range and bearing to all landmarks, one row per landmark,
          including maps with just one landmark
        - ``.h(x, id)`` is range and bearing to landmark ``id``
        - ``.h(x, p)`` is range and bearing to landmark with coordinates ``p``

        For an explicit landmark, a single vehicle pose returns a vector of two
        elements. An ndarray of vehicle poses returns one row per pose, including
        a batch containing just one pose.

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

        .. seealso:: :meth:`reading` :meth:`Hx` :meth:`Hw` :meth:`Hp`
        """
        # get the landmarks, one per row

        vectorized = False
        if isinstance(x, np.ndarray) and x.ndim == 2:
            # x is Nx3 set of vehicle states, do vectorized form
            # used by particle filter
            vectorized = True
            xp, y, t = x.T
        else:
            xp, y, t = x

        if landmark is None:
            # self.h(XV)   all landmarks
            dx = self.map.landmarks[0, :] - xp
            dy = self.map.landmarks[1, :] - y
        elif base.isinteger(landmark):
            # landmark id
            # self.h(XV, JF)
            xlm = self.map[landmark]
            dx = xlm[0] - xp
            dy = xlm[1] - y
        else:
            # landmark position
            # self.h(XV, XF)
            xlm = base.getvector(landmark, 2)
            dx = xlm[0] - xp
            dy = xlm[1] - y

        # compute range and bearing (Vectorized code)

        z = np.c_[
            np.sqrt(dx**2 + dy**2), base.angdiff(np.arctan2(dy, dx), t)
        ]  # range & bearing as columns

        if z.shape[0] == 1 and landmark is not None and not vectorized:
            return z[0]
        else:
            return z

    def Hx(self, x: ArrayLike, landmark: int | ArrayLike) -> NDArray:
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

        .. seealso:: :meth:`h` :meth:`Hp` :meth:`Hw`
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

    def Hp(self, x: ArrayLike, landmark: int | ArrayLike) -> NDArray:
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

        .. seealso:: :meth:`h` :meth:`Hx` :meth:`Hw`
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

    def Hw(self, x: ArrayLike, landmark: int | ArrayLike | None) -> NDArray:
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

        .. seealso:: :meth:`h` :meth:`Hx` :meth:`Hp`
        """
        return np.eye(2)

    def g(self, x: ArrayLike, z: ArrayLike) -> NDArray:
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

        .. seealso:: :meth:`h` :meth:`Gx` :meth:`Gz`
        """

        range = z[0]
        bearing = z[1] + x[2]  # bearing angle in vehicle frame

        # fmt: off
        return np.r_[
            x[0] + range * cos(bearing),
            x[1] + range * sin(bearing)
                ]
        # fmt: on

    def Gx(self, x: ArrayLike, z: ArrayLike) -> NDArray:
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

        .. seealso:: :meth:`g`
        """
        theta = x[2]
        r, bearing = z

        # fmt: off
        return np.array([
            [1,   0,   -r*sin(theta + bearing)],
            [0,   1,    r*cos(theta + bearing)],
                ])
        # fmt: on

    def Gz(self, x: ArrayLike, z: ArrayLike) -> NDArray:
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

        .. seealso:: :meth:`g`
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
    from math import pi

    from roboticstoolbox import Bicycle, LandmarkMap, RangeBearingSensor

    robot = Bicycle()
    map = LandmarkMap(20)
    sensor = RangeBearingSensor(robot, map, range=(0.5, 20), angle=pi / 4)
    print(sensor.reading())
    print(sensor.visible())
    print(sensor.isvisible(3))
    print(sensor.isvisible(4))
