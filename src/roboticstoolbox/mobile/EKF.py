"""
Python EKF Planner
@Author: Peter Corke, original MATLAB code and Python version
@Author: Kristian Gibson, initial MATLAB port
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
from roboticstoolbox.tools.plot import apply_block
import numpy as np
from matplotlib import animation
from scipy.linalg import block_diag
from scipy.stats.distributions import chi2
from spatialmath import SE2, base
from spatialmath.base.types import ArrayLike3, R3x3

from roboticstoolbox.mobile import VehicleBase
from roboticstoolbox.mobile.landmarkmap import LandmarkMap
from roboticstoolbox.mobile.sensors import RangeBearingSensor
from roboticstoolbox.mobile.workspace import WorkspaceLike, resolve_workspace
from roboticstoolbox.tools.types import ArrayLike, NDArray


class EKFLog(NamedTuple):
    """One time step of EKF history: see :meth:`EKF.history`"""

    t: float
    xest: NDArray
    odo: NDArray
    P: NDArray
    innov: NDArray | None
    S: NDArray | None
    K: NDArray | None
    lm: int
    z: NDArray | None


class EKF:
    def __init__(
        self,
        robot: tuple[VehicleBase, NDArray | None],
        sensor: tuple[RangeBearingSensor, NDArray] | None = None,
        map: LandmarkMap | None = None,
        P0: R3x3 | None = None,
        x_est: NDArray | None = None,
        joseph: bool = True,
        animate: bool = True,
        x0: ArrayLike3 = (0, 0, 0),
        verbose: bool = False,
        history: bool = True,
        workspace: WorkspaceLike | None = None,
    ) -> None:
        r"""
        Extended Kalman filter

        :param robot: robot motion model
        :type robot: 2-tuple
        :param sensor: vehicle mounted sensor model, defaults to None
        :type sensor: 2-tuple, optional
        :param map: landmark map, defaults to None
        :type map: :class:`LandmarkMap`, optional
        :param P0: initial covariance matrix, defaults to None
        :type P0: ndarray(n,n), optional
        :param x_est: initial state estimate, defaults to None
        :type x_est: array_like(n), optional
        :param joseph: use Joseph update of covariance, defaults to True
        :param animate: show animation of vehicle motion, defaults to True
        :param x0: initial vehicle state :math:`(x, y, \\theta)`, defaults to (0, 0, 0)
        :type x0: array_like(n), optional
        :param verbose: display extra debug information, defaults to False
        :param history: retain step-by-step history, defaults to True
        :param workspace: dimension of workspace, see :func:`~spatialmath.base.graphics.expand_dims`
        :type workspace: scalar, array_like(2), array_like(4)

        This class solves several classical robotic estimation problems, which are
        selected according to the arguments:

        ======================    ======   ==========   ==========   =======   ======
        Problem                   len(x)   ``robot``    ``sensor``   ``map``   ``P0``
        ======================    ======   ==========   ==========   =======   ======
        Dead reckoning            3        (veh,V)      None         None      P0
        Map-based localization    3        (veh,V)      (smodel,W)   yes       P0
        Map creation              2N       (veh,None)   (smodel,W)   None      None
        SLAM                      3+2N     (veh,V)      (smodel,W)   None      P0
        ======================    ======   ==========   ==========   =======   ======

        where:

        - ``veh`` models the robotic vehicle kinematics and odometry and is a :class:`VehicleBase` subclass
        - ``V`` is the estimated odometry (process) noise covariance as an ndarray(2,2)
        - ``smodel`` models the robot mounted sensor and is a :class:`SensorBase` subclass
        - ``W`` is the estimated sensor (measurement) noise covariance as an ndarray(2,2)

        The state vector has different lengths depending on the particular
        estimation problem, see below.

        At each iteration of the EKF:

        - invoke the step method of the ``robot``

          - obtains the next control input from the driver agent, and apply it
            as the vehicle control input
          - the vehicle returns a noisy odometry estimate

        - the state prediction is computed
        - the true pose is used to determine a noisy sensor observation
        - the state is corrected, new landmarks are added to the map

        The working area of the robot is defined by ``workspace`` or inherited
        from the landmark map attached to the ``sensor`` (see
        :func:`~spatialmath.base.graphics.expand_dims`):

        ==============  =======  =======
        ``workspace``   x-range  y-range
        ==============  =======  =======
        A (scalar)      -A:A     -A:A
        [A, B]           A:B      A:B
        [A, B, C, D]     A:B      C:D
        ==============  =======  =======

        **Dead-reckoning localization**

        The state :math:`\vec{x} = (x, y, \theta)` is the estimated vehicle
        configuration.

        Create a vehicle with odometry covariance ``V``, add a driver to it,
        run the Kalman filter with estimated covariances ``V`` and initial
        vehicle state covariance ``P0``::

            V = np.diag([0.02, np.radians(0.5)]) ** 2;
            robot = Bicycle(covar=V)
            robot.control = RandomPath(workspace=10)

            x_sdev = [0.05, 0.05, np.radians(0.5)]
            P0 = np.diag(x_sdev) ** 2
            ekf = EKF(robot=(robot, V), P0=P0)

            ekf.run(T=20)  # run the simulation for 20 seconds

            robot.plot_xy(color="b")  # plot the true vehicle path
            ekf.plot_xy(color="r")    # overlay the estimated path
            ekf.plot_ellipse(filled=True, facecolor="g", alpha=0.3)  # overlay uncertainty ellipses

            # plot the covariance against time
            t = ekf.get_t();
            pn = ekf.get_Pnorm()
            plt.plot(t, pn);

        **Map-based vehicle localization**

        The state :math:`\vec{x} = (x, y, \theta)` is the estimated vehicle
        configuration.

        Create a vehicle with odometry covariance ``V``, add a driver to it,
        create a map with 20 point landmarks, create a sensor that uses the map
        and vehicle state to estimate landmark range and bearing with covariance
        ``W``, the Kalman filter with estimated covariances ``V`` and ``W`` and
        initial vehicle state covariance ``P0``::

            V = np.diag([0.02, np.radians(0.5)]) ** 2;
            robot = Bicycle(covar=V)
            robot.control = RandomPath(workspace=10)

            map = LandmarkMap(20, workspace=10)

            W = np.diag([0.1, np.radians(1)]) ** 2
            sensor = RangeBearingSensor(robot=robot, map=map, covar=W, angle=[-np.pi/2, np.pi/2], range=4, plot=True)

            x_sdev = [0.05, 0.05, np.radians(0.5)]
            P0 = np.diag(x_sdev) ** 2
            ekf = EKF(robot=(robot, V), P0=P0, map=map, sensor=(sensor, W))

            ekf.run(T=20)  # run the simulation for 20 seconds

            map.plot()  #  plot the map
            robot.plot_xy(color="b")  # plot the true vehicle path
            ekf.plot_xy(color="r")    # overlay the estimated path
            ekf.plot_ellipse()  # overlay uncertainty ellipses

            # plot the covariance against time
            t = ekf.get_t();
            pn = ekf.get_Pnorm()
            plt.plot(t, pn);

        **Vehicle-based map making**

        The state :math:`\vec{x} = (x_0, y_0, \dots, x_{N-1}, y_{N-1})` is the
        estimated landmark positions where :math:`N` is the number of landmarks.
        The state vector is initially empty, and is extended by 2 elements every
        time a new landmark is observed.

        Create a vehicle with perfect odometry (no covariance), add a driver to it,
        create a sensor that uses the map and vehicle state to estimate landmark range
        and bearing with covariance ``W``, the Kalman filter with estimated sensor
        covariance ``W``, then run the filter for N time steps::

            robot = Bicycle()
            robot.add_driver(RandomPath(workspace=10))

            map = LandmarkMap(20, workspace=10, seed=0)

            W = np.diag([0.1, np.radians(1)]) ** 2
            sensor = RangeBearingSensor(robot, map, covar=W)

            ekf = EKF(robot=(robot, None), sensor=(sensor, W))

            ekf.run(T=20)  # run the simulation for 20 seconds

            map.plot()  #  plot the map
            robot.plot_xy(color="b")  # plot the true vehicle path

        **Simultaneous localization and mapping (SLAM)**

        The state :math:`\vec{x} = (x, y, \theta, x_0, y_0, \dots, x_{N-1},
        y_{N-1})` is the estimated vehicle configuration followed by the
        estimated landmark positions where :math:`N` is the number of landmarks.
        The state vector is initially of length 3, and is extended by 2 elements
        every time a new landmark is observed.

        Create a vehicle with odometry covariance ``V``, add a driver to it,
        create a map with 20 point landmarks, create a sensor that uses the map
        and vehicle state to estimate landmark range and bearing with covariance
        ``W``, the Kalman filter with estimated covariances ``V`` and ``W`` and
        initial state covariance ``P0``, then run the filter to estimate the
        vehicle state at each time step and the map::

            V = np.diag([0.02, np.radians(0.5)]) ** 2;
            robot = Bicycle(covar=V)
            robot.control = RandomPath(workspace=10)

            map = LandmarkMap(20, workspace=10)

            W = np.diag([0.1, np.radians(1)]) ** 2
            sensor = RangeBearingSensor(robot=robot, map=map, covar=W, angle=[-np.pi/2, np.pi/2], range=4, plot=True)

            x_sdev = [0.05, 0.05, np.radians(0.5)]
            P0 = np.diag(x_sdev) ** 2
            ekf = EKF(robot=(robot, V), P0=P0, sensor=(sensor, W))

            ekf.run(T=20)  # run the simulation for 20 seconds

            map.plot(); # plot true map
            ekf.plot_map(); # plot estimated landmark position

            robot.plot_xy(); # plot true path
            ekf.plot_xy(); # plot estimated robot path
            ekf.plot_ellipse(); # plot estimated covariance

            # plot the covariance against time
            t = ekf.get_t();
            pn = ekf.get_Pnorm()
            plt.plot(t, pn);

        :seealso: :meth:`run`
        """

        if robot is not None:
            if (
                not isinstance(robot, tuple)
                or len(robot) != 2
                or not isinstance(robot[0], VehicleBase)
            ):
                raise TypeError("robot must be tuple (vehicle, V_est)")
            self._robot = robot[0]  # reference to the robot vehicle
            self._V_est = robot[1]  # estimate of vehicle state covariance V

        self._sensor: RangeBearingSensor | None
        self._W_est: NDArray | None
        if sensor is not None:
            if (
                not isinstance(sensor, tuple)
                or len(sensor) != 2
                or not isinstance(sensor[0], RangeBearingSensor)
            ):
                raise TypeError(
                    "sensor must be tuple (RangeBearingSensor, W_est), "
                    "the EKF needs the sensor's h, Hx, Hw, Hp, g, Gx, Gz and reading"
                )
            self._sensor = sensor[0]  # reference to the sensor
            self._W_est = sensor[1]  # estimate of sensor covariance W
        else:
            self._sensor = None
            self._W_est = None

        if map is not None and not isinstance(map, LandmarkMap):
            raise TypeError("map must be LandmarkMap instance")
        self._ekf_map = map  # prior map for localization

        # the workspace, in order of precedence: the explicit argument, the map,
        # the sensor's map, the vehicle
        world = map if map is not None else (sensor[0].map if sensor else None)
        self._workspace: NDArray | None = resolve_workspace(
            workspace, world, self._robot
        )
        if animate:
            if world is not None:
                # animate the vehicle in the workspace of the map
                self._robot.workspace = world.workspace
            elif self._workspace is None:
                raise ValueError("for animation robot must have a defined workspace")
        self.animate = animate

        self._P0 = P0  #  initial system covariance
        self._x0 = x0  # initial vehicle state

        # estimated state and its covariance, both set by init() below
        self._x_est: NDArray = np.empty((0,)) if x_est is None else x_est
        self._P_est: NDArray = np.empty((0, 0))
        self._landmarks: dict[int, list[int]] = {}  #  ekf_map state

        self._est_vehicle = False
        self._est_ekf_map = False
        if self._V_est is not None:
            # estimating vehicle pose by:
            #  - DR if sensor is None
            #  - localization if sensor is not None and map is not None
            self._est_vehicle = True

        # perfect vehicle case
        if map is None and sensor is not None:
            # estimating ekf_map
            self._est_ekf_map = True
        self._joseph = joseph  #  flag: use Joseph form to compute p

        self._verbose = verbose

        self._keep_history = history  #  keep history
        self._htuple = EKFLog

        self._dim: NDArray | None = self._workspace

        # self.robot.init()

        # #  clear the history
        # self._history = []

        if self.V_est is None:
            # perfect vehicle case

            self._est_vehicle = False
            self._x_est = np.empty((0,))
            self._P_est = np.empty((0, 0))
        else:
            # noisy odometry case
            if self.V_est.shape != (2, 2):
                raise ValueError("vehicle state covariance V_est must be 2x2")
            self._x_est = self.robot.x
            self._P_est = np.empty((0, 0)) if P0 is None else P0
            self._est_vehicle = True

        if self.W_est is not None:
            if self.W_est.shape != (2, 2):
                raise ValueError("sensor covariance W_est must be 2x2")

        # if np.any(self._sensor):
        #     self._landmarks = None*np.zeros(2, self._sensor.ekf_map.nlandmarks)

        # #  check types for passed objects
        # if np.any(self._map) and not isinstance(self._map, 'LandmarkMap'):
        #     raise ValueError('expecting LandmarkMap object')

        # if np.any(sensor) and not isinstance(sensor, 'Sensor'):
        #     raise ValueError('expecting Sensor object')

        self.init()

        self.xxdata: tuple[list[Any], list[Any]] = ([], [])

    def __str__(self) -> str:
        s = f"EKF object: {len(self._x_est)} states"

        def indent(s: str, n: int = 2) -> str:
            spaces = " " * n
            return s.replace("\n", "\n" + spaces)

        estimating = []
        if self._est_vehicle:
            estimating.append("vehicle pose")
        if self._est_ekf_map:
            estimating.append("map")
        if len(estimating) > 0:
            s += ", estimating: " + ", ".join(estimating)
        if self.robot is not None:
            s += indent("\nrobot: " + str(self.robot))
        if self.V_est is not None:
            s += indent("\nV_est:  " + base.array2str(self.V_est))

        if self.sensor is not None:
            s += indent("\nsensor: " + str(self.sensor))
        if self.W_est is not None:
            s += indent("\nW_est:  " + base.array2str(self.W_est))

        return s

    def __repr__(self) -> str:
        return str(self)

    @property
    def x_est(self) -> NDArray:
        """
        Get EKF state

        :return: state vector
        :rtype: ndarray(n)

        Returns the value of the estimated state vector at the end of
        simulation. The dimensions depend on the problem being solved.
        """
        return self._x_est

    @property
    def P_est(self) -> NDArray:
        """
        Get EKF covariance

        :return: covariance matrix
        :rtype: ndarray(n,n)

        Returns the value of the estimated covariance matrix at the end of
        simulation. The dimensions depend on the problem being solved.
        """
        return self._P_est

    @property
    def P0(self) -> NDArray | None:
        """
        Get initial EKF covariance

        :return: covariance matrix
        :rtype: ndarray(n,n)

        Returns the value of the covariance matrix passed to the constructor.
        """
        return self._P0

    @property
    def V_est(self) -> NDArray | None:
        """
        Get estimated odometry covariance

        :return: odometry covariance
        :rtype: ndarray(2,2)

        Returns the value of the estimated odometry covariance matrix passed to
        the constructor
        """
        return self._V_est

    @property
    def W_est(self) -> NDArray | None:
        """
        Get estimated sensor covariance

        :return: sensor covariance
        :rtype: ndarray(2,2)

        Returns the value of the estimated sensor covariance matrix passed to
        the constructor
        """
        return self._W_est

    @property
    def robot(self) -> VehicleBase:
        """
        Get robot object

        :return: robot used in simulation
        :rtype: :class:`VehicleBase` subclass
        """
        return self._robot

    @property
    def sensor(self) -> RangeBearingSensor | None:
        """
        Get sensor object

        :return: sensor used in simulation
        :rtype: :class:`SensorBase` subclass
        """
        return self._sensor

    @property
    def map(self) -> LandmarkMap | None:
        """
        Get map object

        :return: map used in simulation
        :rtype: :class:`LandmarkMap` subclass
        """
        return self._ekf_map

    @property
    def verbose(self) -> bool:
        """
        Get verbosity state

        :return: verbosity
        :rtype: bool
        """
        return self._verbose

    @property
    def history(self) -> list[EKFLog]:
        """
        Get EKF simulation history

        :return: simulation history
        :rtype: list of namedtuples

        At each simulation timestep a namedtuple of is appended to the history
        list.  It contains, for that time step, estimated state and covariance,
        and sensor observation.

        :seealso: :meth:`get_t` :meth:`get_xyt` :meth:`get_map` :meth:`get_P`
            :meth:`get_Pnorm`
        """
        return self._history

    @property
    def workspace(self) -> NDArray | None:
        """
        Size of robot workspace

        :return: workspace bounds [xmin, xmax, ymin, ymax]
        :rtype: ndarray(4)

        Returns the bounds of the workspace as specified by the constructor
        option ``workspace``
        """
        return self._workspace

    @property
    def landmarks(self) -> dict[int, list[int]]:
        """
        Get landmark information

        :return: landmark information
        :rtype: dict

        The dictionary is indexed by the landmark id and gives a 3-tuple:

        - order in which landmark was seen
        - number of times seen

        The order in which the landmark was first seen.  The first observed
        landmark has order 0 and so on.

        :seealso: :meth:`landmark`
        """
        return self._landmarks

    def landmark(self, id: int) -> tuple[int, int]:
        """
        Landmark information

        :param id: landmark index
        :return: order in which it was first seen, number of times seen
        :rtype: int, int

        The first observed landmark has order 0 and so on.

        :seealso: :meth:`landmarks` :meth:`landmark_index` :meth:`landmark_mindex`
        """
        try:
            l = self._landmarks[id]
            return l[0], l[1]
        except KeyError:
            raise ValueError(f"unknown landmark {id}") from None

    def landmark_index(self, id: int) -> int:
        """
        Landmark index in complete state vector

        :param id: landmark index
        :return: index in the state vector
        :rtype: int

        The return value ``j`` is the index of the x-coordinate of the landmark
        in the EKF state vector, and ``j+1`` is the index of the y-coordinate.

        :seealso: :meth:`landmark`
        """
        try:
            jx = self._landmarks[id][0] * 2
            if self._est_vehicle:
                jx += 3
            return jx
        except KeyError:
            raise ValueError(f"unknown landmark {id}") from None

    def landmark_mindex(self, id: int) -> int:
        """
        Landmark index in map state vector

        :param id: landmark index
        :return: index in the state vector
        :rtype: int

        The return value ``j`` is the index of the x-coordinate of the landmark
        in the map vector, and ``j+1`` is the index of the y-coordinate.

        :seealso: :meth:`landmark`
        """
        try:
            return self._landmarks[id][0] * 2
        except KeyError:
            raise ValueError(f"unknown landmark {id}") from None

    def landmark_x(self, id: int) -> NDArray:
        """
        Landmark position

        :param id: landmark index
        :return: landmark position :math:`(x,y)`
        :rtype: ndarray(2)

        Returns the landmark position from the current state vector.
        """
        jx = self.landmark_index(id)
        return self._x_est[jx : jx + 2]

    def init(self) -> None:
        """
        Reset the filter

        Resets the filter state and covariance to their initial values, and
        clears the landmarks and history.
        """
        self.robot.init()
        if self.sensor is not None:
            self.sensor.init()

        # clear the history
        self._history: list[EKFLog] = []

        if self._V_est is None:
            # perfect vehicle case
            self._x_est = np.empty((0,))
            self._P_est = np.empty((0, 0))
        else:
            # noisy odometry case, the state is a copy so that the filter
            # never changes the arrays the user passed in
            self._x_est = np.array(self._x0, dtype=float)
            self._P_est = (
                np.empty((0, 0))
                if self._P0 is None
                else np.array(self._P0, dtype=float)
            )

        if self.sensor is not None:
            # landmark dictionary maps lm_id to list[index, nseen]
            self._landmarks = {}

            # np.full((2, len(self.sensor.map)), -1, dtype=int)

    def run(self, T: float, animate: bool = False) -> None:
        """
        Run the EKF simulation

        :param T: maximum simulation time in seconds
        :param animate: animate motion of vehicle, defaults to False

        Simulates the motion of a vehicle (under the control of a driving agent)
        and the EKF estimator.  The steps are:

        - initialize the filter, vehicle and vehicle driver agent, sensor
        - for each time step:

            - step the vehicle and its driver agent, obtain odometry
            - take a sensor reading
            - execute the EKF
            - save information as a namedtuple to the history list for later display

        :seealso: :meth:`history` :meth:`landmark` :meth:`landmarks`
            :meth:`get_xyt` :meth:`get_t` :meth:`get_map` :meth:`get_P` :meth:`get_Pnorm`
            :meth:`plot_xy` :meth:`plot_ellipse` :meth:`plot_error` :meth:`plot_map`
            :meth:`run_animation`
        """
        self.init()
        if animate:
            if self.sensor is not None:
                self.sensor.map.plot()

            plt.xlabel("X")
            plt.ylabel("Y")

        for k in range(round(T / self.robot.dt)):
            if animate:
                # self.robot.plot()
                if self.robot._animation is not None:
                    self.robot._animation.update(self.robot.x)
            self.step()

    def run_animation(
        self,
        T: float = 10,
        x0: ArrayLike | None = None,
        control: Any = None,
        format: str | None = None,
        file: str | None = None,
    ) -> Any:
        r"""
        Run the EKF simulation

        :param T: maximum simulation time in seconds
        :param format: Output format
        :param file: File name
        :return: Matplotlib animation object
        :rtype: :meth:`matplotlib.animation.FuncAnimation`

        Simulates the motion of a vehicle (under the control of a driving agent)
        and the EKF estimator for ``T`` seconds and returns an animation
        in various formats::

            ``format``    ``file``   description
            ============  =========  ============================
            ``"html"``    str, None  return HTML5 video
            ``"jshtml"``  str, None  return JS+HTML video
            ``"gif"``     str        return animated GIF
            ``"mp4"``     str        return MP4/H264 video
            ``None``                 return a ``FuncAnimation`` object

        If ``file`` can be ``None`` then return the video as a string, otherwise it
        must be a filename.

        The simulation steps are:

        - initialize the filter, vehicle and vehicle driver agent, sensor
        - for each time step:

            - step the vehicle and its driver agent, obtain odometry
            - take a sensor reading
            - execute the EKF
            - save information as a namedtuple to the history list for later display

        :seealso: :meth:`history` :meth:`landmark` :meth:`landmarks`
            :meth:`get_xyt` :meth:`get_t` :meth:`get_map` :meth:`get_P` :meth:`get_Pnorm`
            :meth:`plot_xy` :meth:`plot_ellipse` :meth:`plot_error` :meth:`plot_map`
            :meth:`run_animation`
        """

        fig, ax = plt.subplots()

        def init() -> None:
            self.init()
            if self.sensor is not None:
                self.sensor.map.plot()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        def animate(i: int) -> None:
            if self.robot._animation is not None:
                self.robot._animation.update(self.robot.x)
            self.step(pause=False)

        nframes = round(T / self.robot._dt)
        anim = animation.FuncAnimation(
            fig=fig,
            func=animate,
            init_func=init,
            frames=nframes,
            interval=self.robot.dt * 1000,
            blit=False,
            repeat=False,
        )

        ret = None
        if format == "html":
            ret = anim.to_html5_video()  # convert to embeddable HTML5 animation
        elif format == "jshtml":
            ret = anim.to_jshtml()  # convert to embeddable Javascript/HTML animation
        elif format == "gif":
            if file is None:
                raise ValueError("file is required for gif format")
            anim.save(
                file,
                writer=animation.PillowWriter(fps=1 / self.robot.dt),  # type: ignore[arg-type]
            )  # convert to GIF
            ret = None
        elif format == "mp4":
            if file is None:
                raise ValueError("file is required for mp4 format")
            anim.save(
                file,
                writer=animation.FFMpegWriter(fps=1 / self.robot.dt),  # type: ignore[arg-type]
            )  # convert to mp4/H264
            ret = None
        elif format == None:
            # return the anim object
            return anim
        else:
            raise ValueError("unknown format")

        if ret is not None and file is not None:
            with open(file, "w") as f:
                f.write(ret)
            ret = None
        plt.close(fig)
        return ret

    def step(self, pause: bool = False) -> None:
        """
        Execute one timestep of the simulation

        :param pause: pause for the vehicle's time step after drawing the
            animation, defaults to False
        """

        # move the robot
        odo = self.robot.step(pause=pause)

        # =================================================================
        # P R E D I C T I O N
        # =================================================================
        if self._est_vehicle:
            # split the state vector and covariance into chunks for
            # vehicle and map
            xv_est = self._x_est[:3]
            xm_est = self._x_est[3:]
            Pvv_est = self._P_est[:3, :3]
            Pmm_est = self._P_est[3:, 3:]
            Pvm_est = self._P_est[:3, 3:]
        else:
            xm_est = self._x_est
            Pmm_est = self._P_est

        if self._est_vehicle:
            # evaluate the state update function and the Jacobians
            # if vehicle has uncertainty, predict its covariance
            xv_pred = self.robot.f(xv_est, odo)

            Fx = self.robot.Fx(xv_est, odo)
            Fv = self.robot.Fv(xv_est, odo)
            Pvv_pred = Fx @ Pvv_est @ Fx.T + Fv @ self.V_est @ Fv.T
        else:
            # otherwise we just take the true robot state
            xv_pred = self._robot.x

        if self._est_ekf_map:
            if self._est_vehicle:
                # SLAM case, compute the correlations
                Pvm_pred = Fx @ Pvm_est

            Pmm_pred = Pmm_est
            xm_pred = xm_est

        # put the chunks back together again
        if self._est_vehicle and not self._est_ekf_map:
            # vehicle only
            x_pred = xv_pred
            P_pred = Pvv_pred
        elif not self._est_vehicle and self._est_ekf_map:
            # map only
            x_pred = xm_pred
            P_pred = Pmm_pred
        elif self._est_vehicle and self._est_ekf_map:
            # vehicle and map
            x_pred = np.r_[xv_pred, xm_pred]
            # fmt: off
            P_pred = np.block([
                [Pvv_pred,   Pvm_pred], 
                [Pvm_pred.T, Pmm_pred]
            ])
            # fmt: on

        # at this point we have:
        #   xv_pred the state of the vehicle to use to
        #           predict observations
        #   xm_pred the state of the map
        #   x_pred  the full predicted state vector
        #   P_pred  the full predicted covariance matrix

        # initialize the variables that might be computed during
        # the update phase

        doUpdatePhase = False

        # disp('x_pred:') x_pred'

        # =================================================================
        # P R O C E S S    O B S E R V A T I O N S
        # =================================================================

        if self.sensor is not None:
            #  read the sensor
            z, lm_id = self.sensor.reading()
            sensorReading = z is not None
        else:
            lm_id = None  # keep history saving happy
            z = None
            sensorReading = False

        if sensorReading:
            assert self.sensor is not None and z is not None and lm_id is not None
            #  here for MBL, MM, SLAM

            # compute the innovation
            z_pred = self.sensor.h(xv_pred, lm_id)
            innov = np.array([z[0] - z_pred[0], base.wrap_mpi_pi(z[1] - z_pred[1])])

            if self._est_ekf_map:
                # the ekf_map is estimated MM or SLAM case
                if self._isseenbefore(lm_id):
                    # landmark is previously seen

                    # get previous estimate of its state
                    jx = self.landmark_mindex(lm_id)
                    xf = xm_pred[jx : jx + 2]

                    # compute Jacobian for this particular landmark
                    # xf = self.sensor.g(xv_pred, z) # HACK
                    Hx_k = self.sensor.Hp(xv_pred, xf)

                    z_pred = self.sensor.h(xv_pred, xf)
                    innov = np.array(
                        [z[0] - z_pred[0], base.wrap_mpi_pi(z[1] - z_pred[1])]
                    )

                    #  create the Jacobian for all landmarks
                    Hx = np.zeros((2, len(xm_pred)))
                    Hx[:, jx : jx + 2] = Hx_k

                    Hw = self.sensor.Hw(xv_pred, xf)

                    if self._est_vehicle:
                        # concatenate Hx for for vehicle and ekf_map
                        Hxv = self.sensor.Hx(xv_pred, xf)
                        Hx = np.block([Hxv, Hx])

                    self._landmark_increment(lm_id)  # update the count
                    if self._verbose:
                        print(
                            f"landmark {lm_id} seen"
                            f" {self._landmark_count(lm_id)} times,"
                            f" state_idx={self.landmark_index(lm_id)}"
                        )
                    doUpdatePhase = True

                else:
                    # new landmark, seen for the first time

                    # extend the state vector and covariance
                    x_pred, P_pred = self._extend_map(
                        P_pred, xv_pred, xm_pred, z, lm_id
                    )
                    # if lm_id == 17:
                    #     print(P_pred)
                    #     # print(x_pred[-2:], self._sensor._map.landmark(17), base.norm(x_pred[-2:] - self._sensor._map.landmark(17)))

                    self._landmark_add(lm_id)
                    if self._verbose:
                        print(
                            f"landmark {lm_id} seen for first time,"
                            f" state_idx={self.landmark_index(lm_id)}"
                        )
                    doUpdatePhase = False

            else:
                # LBL
                Hx = self.sensor.Hx(xv_pred, lm_id)
                Hw = self.sensor.Hw(xv_pred, lm_id)
                doUpdatePhase = True
        else:
            innov = None

        # doUpdatePhase flag indicates whether or not to do
        # the update phase of the filter
        #
        #  DR                        always false
        #  map-based localization    if sensor reading
        #  map creation              if sensor reading & not first
        #                              sighting
        #  SLAM                      if sighting of a previously
        #                              seen landmark

        if doUpdatePhase:
            # disp('do update\n')
            # #  we have innovation, update state and covariance
            #  compute x_est and P_est

            # compute innovation covariance
            S = Hx @ P_pred @ Hx.T + Hw @ self._W_est @ Hw.T

            # compute the Kalman gain
            K = P_pred @ Hx.T @ np.linalg.inv(S)

            # update the state vector
            x_est = x_pred + K @ innov

            if self._est_vehicle:
                #  wrap heading state for a vehicle
                x_est[2] = base.wrap_mpi_pi(x_est[2])

            # update the covariance
            if self._joseph:
                #  we use the Joseph form
                I = np.eye(P_pred.shape[0])
                P_est = (I - K @ Hx) @ P_pred @ (I - K @ Hx).T + K @ self._W_est @ K.T
            else:
                P_est = P_pred - K @ S @ K.T
                # enforce P to be symmetric
                P_est = 0.5 * (P_est + P_est.T)
        else:
            # no update phase, estimate is same as prediction
            x_est = x_pred
            P_est = P_pred
            S = None
            K = None

        self._x_est = x_est
        self._P_est = P_est

        if self._keep_history:
            hist = self._htuple(
                self.robot._t,
                x_est.copy(),
                odo.copy(),
                P_est.copy(),
                innov.copy() if innov is not None else None,
                S.copy() if S is not None else None,
                K.copy() if K is not None else None,
                lm_id if lm_id is not None else -1,
                z.copy() if z is not None else None,
            )
            self._history.append(hist)

    ## landmark management

    def _isseenbefore(self, lm_id: int) -> bool:

        # _landmarks[0, id] is the order in which seen
        # _landmarks[1, id] is the occurence count

        return lm_id in self._landmarks

    def _landmark_increment(self, lm_id: int) -> None:
        self._landmarks[lm_id][1] += 1  # update the count

    def _landmark_count(self, lm_id: int) -> int:
        return self._landmarks[lm_id][1]

    def _landmark_add(self, lm_id: int) -> None:
        self._landmarks[lm_id] = [len(self._landmarks), 1]

    def _extend_map(
        self, P: NDArray, xv: NDArray, xm: NDArray, z: NDArray, lm_id: int
    ) -> tuple[NDArray, NDArray]:
        # this is a new landmark, we haven't seen it before
        # estimate position of landmark in the world based on
        # noisy sensor reading and current vehicle pose

        # M = None

        # estimate its position based on observation and vehicle state
        assert self.sensor is not None
        xf = self.sensor.g(xv, z)

        # append this estimate to the state vector
        if self._est_vehicle:
            x_ext = np.r_[xv, xm, xf]
        else:
            x_ext = np.r_[xm, xf]

        # get the Jacobian for the new landmark
        Gz = self.sensor.Gz(xv, z)

        # extend the covariance matrix
        n = len(self._x_est)
        if self._est_vehicle:
            # estimating vehicle state
            Gx = self.sensor.Gx(xv, z)
            # fmt: off
            Yz = np.block([
                [np.eye(n), np.zeros((n, 2))    ],
                [Gx,        np.zeros((2, n-3)), Gz]
            ])
            # fmt: on
        else:
            # estimating landmarks only
            # P_ext = block_diag(P, Gz @ self._W_est @ Gz.T)
            # fmt: off
            Yz = np.block([
                [np.eye(n),        np.zeros((n, 2))    ],
                [np.zeros((2, n)), Gz]
            ])
            # fmt: on
        P_ext = Yz @ block_diag(P, self._W_est) @ Yz.T

        return x_ext, P_ext

    def get_t(self) -> NDArray:
        """
        Get time from simulation

        :return: simulation time vector
        :rtype: ndarray(n)

        Return simulation time vector, starts at zero.  The timestep is an
        attribute of the ``robot`` object.

        :seealso: :meth:`run` :meth:`history`
        """
        return np.array([h.t for h in self._history])

    def get_xyt(self) -> NDArray | None:
        r"""
        Get estimated vehicle trajectory

        :return: vehicle trajectory where each row is configuration :math:`(x, y, \theta)`
        :rtype: ndarray(n,3)

        :seealso: :meth:`plot_xy` :meth:`run` :meth:`history`
        """
        if self._est_vehicle:
            xyt = np.array([h.xest[:3] for h in self._history])
        else:
            xyt = None
        return xyt

    def plot_xy(self, *args: Any, block: bool | None = None, **kwargs: Any) -> None:
        """
        Plot estimated vehicle position

        :param args: position arguments passed to :meth:`~matplotlib.axes.Axes.plot`
        :param kwargs: keywords arguments passed to :meth:`~matplotlib.axes.Axes.plot`
        :param block: ``None`` leaves showing the figure to the environment (no
            ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
            it and waits until it is closed; ignored by notebook backends,
            defaults to None

        Plot the estimated vehicle path in the xy-plane.

        :seealso: :meth:`get_xyt` :meth:`plot_error` :meth:`plot_ellipse` :meth:`plot_P`
            :meth:`run` :meth:`history`
        """
        xyt = self.get_xyt()
        if xyt is None:
            raise ValueError("the vehicle pose is not part of the estimated state")
        plt.plot(xyt[:, 0], xyt[:, 1], *args, **kwargs)
        apply_block(block)

    def plot_ellipse(
        self,
        confidence: float = 0.95,
        N: int = 10,
        block: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot uncertainty ellipses

        :param confidence: ellipse confidence interval, defaults to 0.95
        :param N: number of ellipses to plot, defaults to 10
        :param block: ``None`` leaves showing the figure to the environment (no
            ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
            it and waits until it is closed; ignored by notebook backends,
            defaults to None
        :param kwargs: arguments passed to :meth:`spatialmath.base.graphics.plot_ellipse`

        Plot ``N`` uncertainty ellipses spaced evenly along the trajectory.

        :seealso: :meth:`get_P` :meth:`run` :meth:`history`
        """
        nhist = len(self._history)

        if "label" in kwargs:
            label = kwargs["label"]
            del kwargs["label"]
        else:
            label = f"{confidence * 100:.3g}% confidence"

        for kf in np.linspace(0, nhist - 1, N):
            k = round(kf)
            h = self._history[k]
            if k == 0:
                base.plot_ellipse(
                    h.P[:2, :2],
                    centre=h.xest[:2],
                    confidence=confidence,
                    label=label,
                    inverted=True,
                    **kwargs,
                )
            else:
                base.plot_ellipse(
                    h.P[:2, :2],
                    centre=h.xest[:2],
                    confidence=confidence,
                    inverted=True,
                    **kwargs,
                )
        apply_block(block)

    def plot_error(
        self,
        bgcolor: str = "r",
        confidence: float | None = 0.95,
        ax: Any = None,
        block: bool | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Plot error with uncertainty bounds

        :param bgcolor: color of the confidence bound polygons, defaults to 'r'
        :param confidence: confidence interval, defaults to 0.95, None for no bounds
        :param ax: axes to plot into, one for each of :math:`(x, y, \theta)`,
            defaults to a new figure with three stacked axes
        :type ax: array_like(3) of Axes, optional
        :param block: ``None`` leaves showing the figure to the environment (no
            ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
            it and waits until it is closed; ignored by notebook backends,
            defaults to None
        :raises ValueError: if the vehicle pose is not part of the estimated state

        Plot the error between actual and estimated vehicle
        path :math:`(x, y, \theta)` versus time as three stacked plots.
        Heading error is wrapped into the range :math:`[-\pi,\pi)`


        Behind each line draw a shaded polygon ``bgcolor`` showing the specified
        ``confidence`` bounds based on the covariance at each time step.
        Ideally the lines should be within the shaded polygon ``confidence``
        of the time.

        .. note:: Observations will decrease the uncertainty while periods of dead-reckoning increase it.

        .. versionchanged:: 1.4.5
            - The ``confidence`` bounds are the interval that contains that
              fraction of a one-dimensional Gaussian, :math:`\pm 1.96\sigma`
              for 0.95.  Earlier versions used the radius of a
              two-dimensional confidence region, :math:`\pm 2.45\sigma` for
              0.95, which made the shaded region about 25% wider than its
              label.  Plots produced with earlier versions show wider bounds.
            - The error is plotted against time.  Earlier versions plotted it
              against sample number on axes limited to the time range, so only
              the start of a run was visible.
            - ``bgcolor`` is used, it was previously ignored.
            - Works for SLAM, where the state also holds the landmarks, and
              raises ``ValueError`` if the vehicle pose is not estimated.

        :seealso: :meth:`get_P` :meth:`run` :meth:`history`
        """
        if not self._est_vehicle:
            raise ValueError("the vehicle pose is not part of the estimated state")

        errs: list[NDArray] = []
        bnds: list[NDArray] = []
        # the bound on a scalar error is the interval that contains
        # ``confidence`` of a 1-dimensional Gaussian, ie. chi-squared with one
        # degree of freedom
        ppf = chi2.ppf(confidence, df=1) if confidence is not None else None

        x_gt = self.robot.x_hist
        for k in range(len(self.history)):
            hk = self.history[k]
            # error is true - estimated, the vehicle pose is the first three
            # elements of the state which, for SLAM, is followed by the landmarks
            e = x_gt[k, :3] - hk.xest[:3]
            e[2] = base.wrap_mpi_pi(e[2])
            errs.append(e)

            if ppf is not None:
                P = np.diag(hk.P)
                bnds.append(np.sqrt(ppf * P[:3]))

        error = np.array(errs)
        bounds = np.array(bnds)
        t = self.get_t()

        if ax is None:
            fig, axes = plt.subplots(3)
        else:
            axes = ax[:3]
        labels = ["x", "y", r"$\theta$"]

        for k, ax in enumerate(axes):
            if confidence is not None:
                edge = np.array(
                    [
                        np.r_[t, t[::-1]],
                        np.r_[bounds[:, k], -bounds[::-1, k]],
                    ]
                )
                polygon = plt.Polygon(
                    edge.T, closed=True, facecolor=bgcolor, edgecolor="none", alpha=0.3
                )
                ax.add_patch(polygon)
            ax.plot(t, error[:, k], **kwargs)
            ax.grid(True)
            ax.set_ylabel(labels[k] + " error")
            ax.set_xlim(0, t[-1])

        apply_block(block)

    def get_map(self) -> NDArray:
        """
        Get estimated map

        :return: landmark coordinates :math:`(x, y)`
        :rtype: ndarray(n,2)

        Landmarks are returned in the order they were first observed.

        :seealso: :meth:`landmarks`  :meth:`run` :meth:`history`

        """
        xy = [self.landmark_x(lm_id) for lm_id in self._landmarks]
        return np.array(xy)

    def plot_map(
        self,
        marker_style: dict[str, Any] | None = None,
        ellipse_style: dict[str, Any] | None = None,
        confidence: float = 0.95,
        block: bool | None = None,
        *,
        marker: dict[str, Any] | None = None,
        ellipse: dict[str, Any] | None = None,
    ) -> None:
        """
        Plot estimated landmarks

        :param marker_style: Matplotlib options for the landmark marker, defaults
            to a red ``+``
        :param ellipse_style: options passed to
            :func:`~spatialmath.base.graphics.plot_ellipse`, ellipses are drawn
            only if this is given
        :param confidence: ellipse confidence interval, defaults to 0.95
        :param block: ``None`` leaves showing the figure to the environment (no
            ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
            it and waits until it is closed; ignored by notebook backends,
            defaults to None
        :param marker: deprecated, use ``marker_style``
        :param ellipse: deprecated, use ``ellipse_style``

        Plot a marker and, if ``ellipse_style`` is given, a covariance ellipse
        for each estimated landmark.

        .. deprecated:: 1.4.5
            ``marker`` and ``ellipse`` are deprecated, use ``marker_style`` and
            ``ellipse_style``.

        .. versionchanged:: 1.4.5
            Each ellipse is now drawn with the covariance of its own landmark.
            Previously the covariance was taken from the wrong place in the
            state covariance matrix, so that the ellipses after the first were
            wrong, and in SLAM were not landmark covariances at all.

        :seealso: :meth:`get_map` :meth:`run` :meth:`history`
        """
        for old, new, value in (
            ("marker", "marker_style", marker),
            ("ellipse", "ellipse_style", ellipse),
        ):
            if value is None:
                continue
            warnings.warn(
                f"{old} is deprecated, use {new} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if new == "marker_style" and marker_style is None:
                marker_style = value
            elif new == "ellipse_style" and ellipse_style is None:
                ellipse_style = value

        if marker_style is None:
            marker_style = {
                "marker": "+",
                "markersize": 10,
                "markerfacecolor": "red",
                "linewidth": 0,
            }

        xm = self._x_est
        P = self._P_est
        if self._est_vehicle:
            xm = xm[3:]
            P = P[3:, 3:]

        # mark the estimate as a point
        xm = xm.reshape((-1, 2))  # arrange as Nx2
        plt.plot(xm[:, 0], xm[:, 1], label="estimated landmark", **marker_style)

        # add an ellipse for each landmark, using its 2x2 block of the covariance
        if ellipse_style is not None:
            for i in range(len(xm)):
                Pi = P[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
                # put ellipse in the legend only once
                label = (
                    {"label": f"{confidence * 100:.3g}% confidence"} if i == 0 else {}
                )
                base.plot_ellipse(
                    Pi,
                    centre=xm[i, :],
                    confidence=confidence,
                    inverted=True,
                    **label,
                    **ellipse_style,
                )

        apply_block(block)

    def get_P(self, k: int | None = None) -> NDArray | list[NDArray]:
        """
        Get covariance matrices from simulation

        :param k: timestep, defaults to None
        :return: covariance matrix
        :rtype: ndarray(n,n) or list of ndarray(n,n)

        If ``k`` is given return covariance from simulation timestep ``k``, else
        return a list of all covariance matrices.

        :seealso: :meth:`get_Pnorm` :meth:`run` :meth:`history`
        """
        if k is not None:
            return self._history[k].P
        else:
            return [h.P for h in self._history]

    def get_Pnorm(self, k: int | None = None) -> float | NDArray:
        """
        Get covariance norm from simulation

        :param k: timestep, defaults to None
        :return: covariance matrix norm
        :rtype: float or ndarray(n)

        If ``k`` is given return covariance norm from simulation timestep ``k``, else
        return all covariance norms as a 1D NumPy array.

        :seealso: :meth:`get_P` :meth:`run` :meth:`history`
        """
        if k is not None:
            return np.sqrt(np.linalg.det(self._history[k].P))
        else:
            p = [np.sqrt(np.linalg.det(h.P)) for h in self._history]
            return np.array(p)

    def disp_P(self, P: NDArray, colorbar: bool | dict[str, Any] = False) -> None:
        """
        Display covariance matrix

        :param P: covariance matrix
        :type P: ndarray(n,n)
        :param colorbar: add a colorbar
        :type: bool or dict

        Plot the elements of the covariance matrix as an image. If ``colorbar``
        is True add a color bar, if `colorbar` is a dict add a color bar with
        these options passed to colorbar.

        .. note:: A log scale is used.

        :seealso: :meth:`~matplotlib.axes.Axes.imshow` :func:`matplotlib.pyplot.colorbar`
        """

        z = np.log10(abs(P))
        mn = min(z[~np.isinf(z)])
        z[np.isinf(z)] = mn
        plt.xlabel("State")
        plt.ylabel("State")

        plt.imshow(z, cmap="Reds")
        if colorbar is True:
            plt.colorbar(label="log covariance")
        elif isinstance(colorbar, dict):
            plt.colorbar(**colorbar)

    def get_transform(self, map: dict[int, ArrayLike] | NDArray) -> SE2:
        """
        Transformation from estimated map to true map frame

        :param map: known landmark positions
        :type map: :class:`LandmarkMap`
        :return: transform from ``map`` to estimated map frame
        :rtype: SE2 instance

        Uses a least squares technique to find the transform between the
        landmark is world frame and the estimated landmarks in the SLAM
        reference frame.

        :seealso: :func:`~spatialmath.base.transforms2d.points2tr2`
        """
        p = []
        q = []

        for lm_id in self._landmarks.keys():
            p.append(map[lm_id])
            q.append(self.landmark_x(lm_id))

        p = np.array(p).T
        q = np.array(q).T

        T = base.points2tr2(p, q)
        return SE2(T)


if __name__ == "__main__":
    from roboticstoolbox import *

    V = np.diag([0.02, np.deg2rad(0.5)]) ** 2
    robot = Bicycle(covar=V, animation="car")
    robot.control = RandomPath(workspace=10)
    P0 = np.diag([1, 1, 1])
    V = np.diag([1, 1])
    ekf = EKF(robot=(robot, V), P0=P0, animate=False)
    print(ekf)
    ekf.run_animation(T=20, format="mp4", file="ekf.mp4")

    # from roboticstoolbox import Bicycle

    # ### RVC2: Chapter 6

    # ##  6.1 Dead reckoning

    # ## 6.1.1 Modeling the vehicle
    # V = np.diag(np.r_[0.02, 0.5*pi/180] ** 2);

    # veh = Bicycle(covar=V)

    # odo = veh.step(1, 0.3)

    # print(veh.x)

    # veh.f([0, 0, 0], odo)

    # # veh.add_driver( RandomPath(10) )

    # # veh.run()

    # ###  6.1.2  Estimating pose
    # # veh.Fx( [0,0,0], [0.5, 0.1] )

    # P0 = np.diag(np.r_[0.005, 0.005, 0.001]**2);

    # ekf = EKF(veh, V, P0);

    # ekf.run(1000);

    # veh.plot_xy()

    # ekf.plot_xy('r')

    # P700 = ekf.history(700).P

    # sqrt(P700(1,1))

    # ekf.plot_ellipse('g')

    # #  6.2 Map-based localization
    # # randinit
    # # map = LandmarkMap(20, 10)

    # # map.plot()

    # # W = diag([0.1, 1*pi/180].^2);

    # # sensor = RangeBearingSensor(veh, map, 'covar', W)

    # # [z,i] = sensor.reading()

    # # map.landmark(17)

    # # randinit
    # # map = LandmarkMap(20);
    # # veh = Bicycle('covar', V);
    # # veh.add_driver( RandomPath(map.dim) );
    # # sensor = RangeBearingSensor(veh, map, 'covar', W, 'angle', ...
    # # [-pi/2 pi/2], 'range', 4, 'animate');
    # # ekf = EKF(veh, V, P0, sensor, W, map);

    # # ekf.run(1000);
    # # map.plot()
    # # veh.plot_xy();
    # # ekf.plot_xy('r');
    # # ekf.plot_ellipse('k')

    # #  6.3  Creating a map
    # # randinit
    # # map = LandmarkMap(20);
    # # veh = Bicycle(); error free vehicle
    # # veh.add_driver( RandomPath(map.dim) );
    # # W = diag([0.1, 1*pi/180].^2);
    # # sensor = RangeBearingSensor(veh, map, 'covar', W);
    # # ekf = EKF(veh, [], [], sensor, W, []);

    # # ekf.run(1000);

    # # map.plot();
    # # ekf.plot_map('g');
    # # veh.plot_xy('b');

    # # ekf.landmarks(:,6)

    # # ekf.x_est(19:20)'

    # # ekf.P_est(19:20,19:20)

    # #  6.4  EKF SLAM
    # # randinit
    # # P0 = diag([.01, .01, 0.005].^2);
    # # map = LandmarkMap(20);
    # # veh = Bicycle('covar', V);
    # # veh.add_driver( RandomPath(map.dim) );
    # # sensor = RangeBearingSensor(veh, map, 'covar', W);
    # # ekf = EKF(veh, V, P0, sensor, W, []);

    # # ekf.run(1000);

    # # map.plot();
    # # ekf.plot_map('g');
    # # ekf.plot_xy('r');
    # # veh.plot_xy('b');

    # #  6.6 Pose-graph SLAM
    # # syms x_i y_i theta_i x_j y_j theta_j x_m y_m theta_m assume real
    # # xi_e = inv( SE2(x_m, y_m, theta_m) ) * inv( SE2(x_i, y_i, theta_i) ) * SE2(x_j, y_j, theta_j);
    # # fk = simplify(xi_e.xyt);

    # # jacobian ( fk, [x_i y_i theta_i] );
    # # Ai = simplify (ans)

    # # pg = PoseGraph('pg1.g2o')

    # # clf
    # # pg.plot()

    # # pg.optimize('animate')

    # # pg = PoseGraph('killian-small.toro');

    # # pg.plot()

    # # pg.optimize()

    # # 6.7  Particle filter
    # # randinit
    # # map = LandmarkMap(20);
    # # W = diag([0.1, 1*pi/180].^2);
    # # veh = Bicycle('covar', V);
    # # veh.add_driver( RandomPath(10) );

    # # V = diag([0.005, 0.5*pi/180].^2);
    # # sensor = RangeBearingSensor(veh, map, 'covar', W);

    # # Q = diag([0.1, 0.1, 1*pi/180]).^2;

    # # L = diag([0.1 0.1]);

    # # pf = ParticleFilter(veh, sensor, Q, L, 1000);

    # # pf.run(1000);

    # # map.plot();
    # # veh.plot_xy('b');

    # # clf
    # # pf.plot_xy('r');

    # # clf
    # # plot(pf.std(1:100,:))

    # # clf
    # # pf.plot_pdf()

    # # 6.8  Application: Scanning laser rangefinder

    # # Laser odometry
    # # pg = PoseGraph('killian.g2o', 'laser');

    # # [r, theta] = pg.scan(2580);
    # # about r theta

    # # clf
    # # polar(theta, r)

    # # [x,y] = pol2cart (theta, r);
    # # plot (x, y, '.')

    # # p2580 = pg.scanxy(2580);
    # # p2581 = pg.scanxy(2581);
    # # about p2580

    # # T = icp( p2581, p2580, 'verbose' , 'T0', transl2(0.5, 0), 'distthresh', 3)

    # # pg.time(2581)-pg.time(2580)

    # # Laser-based map building
    # # map = pg.scanmap();
    # # pg.plot_occgrid(map)
