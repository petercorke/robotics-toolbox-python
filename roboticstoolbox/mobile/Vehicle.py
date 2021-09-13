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
from roboticstoolbox.mobile.drivers import VehicleDriver
from roboticstoolbox.mobile.Animations import VehiclePolygon


class VehicleBase(ABC):
    def __init__(self, covar=None, speed_max=np.inf, accel_max=np.inf, x0=[0, 0, 0], dt=0.1,
                 control=None, seed=0, animation=None, verbose=False, plot=False, workspace=None,
                 polygon=None):
        r"""
        Superclass for vehicle kinematic models

        :param covar: odometry covariance, defaults to zero
        :type covar: ndarray(2,2), optional
        :param speed_max: maximum speed, defaults to :math:`\infty`
        :type speed_max: float, optional
        :param accel_max: maximum acceleration, defaults to :math:`\infty`
        :type accel_max: float, optional
        :param x0: Initial state, defaults to (0,0,0)
        :type x0: array_like(3), optional
        :param dt: sample update interval, defaults to 0.1
        :type dt: float, optional
        :param control: vehicle control inputs, defaults to None
        :type control: array_like(2), interp1d, VehicleDriver
        :param animation: Graphical animation of vehicle, defaults to None
        :type animation: VehicleAnimation subclass, optional
        :param verbose: print lots of info, defaults to False
        :type verbose: bool, optional
        :param workspace: dimensions of 2D plot area, defaults to (-10:10) x (-10:10),
            see :func:`~spatialmath.base.animate.plotvol2`
        :type dims: float, array_like(2), , array_like(4)

        This is an abstract superclass that simulates the motion of a mobile
        robot under the action of a controller.  The controller provides
        control inputs to the vehicle, and the output odometry is returned.
        The true state, effectively unknowable in practice, is computed
        and accessible.

        :seealso: :func:`Bicycle`, :func:`Unicycle`
        """

        self._V = covar
        self._dt = dt
        if x0 is None:
            x0 = np.zeros((3,), dtype=float)
        else:
            x0 = base.getvector(x0)
            if len(x0) not in (2,3):
                raise ValueError('x0 must be length 2 or 3')
        self._x0 = x0
        self._x = x0.copy()

        self._random = np.random.default_rng(seed)
        self._seed = seed
        self._speed_max = speed_max
        self._accel_max = accel_max
        self._v_prev = 0
        self._polygon = polygon

        if isinstance(animation, str):
            animation = VehiclePolygon(animation)
        self._animation = animation
        self._ax = None

        if control is not None:
            self.add_driver(control)

        self._dt = dt
        self._t = 0
        self._stopsim = False

        self._verbose = verbose
        self._plot = False

        self._control = None
        self._x_hist = []

        if workspace:
            self._workspace = base.expand_dims(workspace)
        else:
            self._workspace = None
    def __str__(self):
        """
        String representation of vehicle (superclass method)

        :return: String representation of vehicle object
        :rtype: str
        """
        s = f"{self.__class__.__name__}: "
        s += f"x = {base.array2str(self._x)}"
        return s

    @property
    def workspace(self):
        """
        Size of robot workspace

        :return: workspace bounds [xmin, xmax, ymin, ymax]
        :rtype: ndarray(4)

        Returns the bounds of the workspace as specified by constructor
        option ``workspace``
        """

        # get workspace specified for Vehicle or from its driver
        if self._workspace is not None:
            return self._workspace
        if self._control is not None:
            return self._control._workspace

    @property
    def x(self):
        """
        Get vehicle state/configuration (superclass method)

        :return: Vehicle state :math:`(x, y, \theta)`
        :rtype: ndarray(3)
        """
        return self._x

    @property
    def q(self):
        """
        Get vehicle state/configuration (superclass method)

        :return: Vehicle state :math:`(x, y, \theta)`
        :rtype: ndarray(3)
        """
        return self._x

    @property
    def x0(self):
        """
        Get vehicle initial state/configuration (superclass method)

        :return: Vehicle state :math:`(x, y, \theta)`
        :rtype: ndarray(3)

        The state is set to this value at the beginning of each simulation
        run.

        Set by ``Vehicle`` subclass constructor.

        :seealso: :func:`run`
        """
        return self._x0

    @x0.setter
    def x0(self, x0):
        """
        Set vehicle initial state/configuration (superclass method)

        :param x0: Vehicle state :math:`(x, y, \theta)`
        :type x0: array_like(3)

        The state is set to this value at the beginning of each simulation
        run.

        Set by ``Vehicle`` subclass constructor.

        :seealso: :func:`run`
        """
        self._x0 = base.getvector(x0, 3)

    @property
    def random(self):
        """
        Get private random number generator

        :return: NumPy random number generator
        :rtype: Generator

        Has methods including:

            - ``integers(low, high, size, endpoint)``
            - ``random(size)``
            - ``uniform``
            - ``normal(mean, std, size)``
            - ``multivariate_normal(mean, covar, size)``

        The generator is initialized with the seed provided at constructor
        time every time ``init`` is called.

        :seealso: :meth:`init`
        """
        return self._random

    @property
    def x_hist(self):
        """
        Get vehicle state/configuration history (superclass method)

        :return: Vehicle state history
        :rtype: ndarray(n,3)

        The state at each time step resulting from a simulation
        run.

        :seealso: :func:`run`
        """
        return np.array(self._x_hist)

    @property
    def speed_max(self):
        """
        Get maximum speed of vehicle (superclass method)

        :return: maximum speed
        :rtype: float

        Set by ``Vehicle`` subclass constructor.
        """
        return self._speed_max

    @property
    def accel_max(self):
        """
        Get maximum acceleration of vehicle (superclass method)

        :return: maximum acceleration
        :rtype: float

        Set by ``Vehicle`` subclass constructor.
        """
        return self._accel_max

    @property
    def dt(self):
        """
        Get sample time (superclass method)

        :return: discrete time step for simulation
        :rtype: float

        Set by ``Vehicle`` subclass constructor.

        :seealso: :func:`run`
        """
        return self._dt

    @property
    def verbose(self):
        """
        Get verbosity (superclass method)

        :return: verbosity level
        :rtype: bool

        Set by ``Vehicle`` subclass constructor.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """
        Set verbosity (superclass method)

        :return: verbosity level
        :rtype: bool

        Set by ``Vehicle`` subclass constructor.
        """
        self._verbose = verbose

    @property
    def control(self):
        """
        Get vehicle control (superclass method)

        :return: current control
        :rtype: 2-tuple, callable, interp1d or VehicleDriver

        Control can be:

            * a constant tuple as the control inputs to the vehicle
            * a function called as f(vehicle, t, x) that returns a tuple
            * an interpolator called as f(t) that returns a tuple, see
              SciPy interp1d
            * a driver agent, subclass of :func:`VehicleDriver`

        :seealso: :func:`eval_control`
        """
        return self._control

    @control.setter
    def control(self, control):
        """
        Set vehicle control (superclass method)

        :param control: new control
        :type control: 2-tuple, callable, interp1d or VehicleDriver

        Control can be:

            * a constant tuple as the control inputs to the vehicle

            * a driver agent, subclass of :func:`VehicleDriver`

        Example:

        .. runblock:: pycon

            >>> bike = Bicycle()
            >>> bike.control = RandomPath(10)
            >>> print(bike)

        :seealso: :func:`eval_control`, :func:`RandomPath`, :func:`PurePursuit`
        """
        # * a function called as ``f(vehicle, t, x)`` that returns a tuple
        # * an interpolator called as f(t) that returns a tuple, see
        #   SciPy interp1d

        self._control = control
        if isinstance(control, VehicleDriver):
            # if this is a driver agent, connect it to the vehicle
            control.vehicle = self

    def polygon(self, q):
        return self._polygon.transformed(SE2(q))

    # This function is overridden by the child class
    @abstractmethod
    def deriv(self, x, u):
        pass

    def add_driver(self, driver):
        """
        Add a driver agent (superclass method)

        :param driver: a driver agent object
        :type driver: VehicleDriver subclass

        .. warning: Deprecated.  Use ``vehicle.control = driver`` instead.

        :seealso: :func:`RandomPath`
        """

        warnings.warn('add_driver is deprecated, use veh.control=driver instead')
        self._control = driver
        driver._veh = self


    def run(self, T=10, x0=None, control=None):
        """
        Simulate motion of vehicle (superclass method)

        :param N: Number of simulation steps, defaults to 1000
        :type N: int, optional
        :param x0: Initial state, defaults to value given to Vehicle constructor
        :type x0: array_like(3) or array_like(2)
        :param animation: vehicle animation object, defaults to None
        :type animation: VehicleAnimation subclass, optional
        :param plot: Enable plotting, defaults to False
        :type plot: bool, optional
        :return: State trajectory, each row is :math:`(x,y,\theta)`.
        :rtype: ndarray(n,3)

        Runs the vehicle simulation for ``N`` timesteps and optionally plots
        an animation.

        The control inputs are provied by ``control`` which can be:

            * a constant tuple as the control inputs to the vehicle
            * a function called as ``f(vehicle, t, x)`` that returns a tuple
            * an interpolator called as f(t) that returns a tuple, see
              SciPy interp1d
            * a driver agent, subclass of :func:`VehicleDriver`

        
        The simulation can be stopped prematurely by the control function
        calling :func:`stopif`.
        
        :seealso: :func:`init`, :func:`step`, :func:`control`
        """

        self.init(control=control, x0=x0)
        
        for i in range(round(T / self.dt)):
            self.step()

            # check for user requested stop
            if self._stopsim:
                print('USER REEQUESTED STOP AT time', self._t)
                break

        return self.x_hist

    def init(self, x0=None, control=None):
        """
        Initialize for simulation (superclass method)

        :param x0: Initial state, defaults to value given to Vehicle constructor
        :type x0: array_like(3) or array_like(2)


        Performs the following initializations:

            #. Clears the state history
            #. Sets state :math:`x = x_0`
            #. If a driver is attached, initialize it
            #. If plotting is enabled, initialize that

        If ``plot`` is set and no animation object is given, use a default
        ``VehiclePolygon('car')``

        :seealso: :func:`VehicleAnimation`
        """
        if x0 is not None:
            self._x = base.getvector(x0, 3)
        else:
            self._x = self._x0.copy()

        self._x_hist = []

        if self._seed is not None:
            self._random = np.random.default_rng(self._seed)

        if control is not None:
            # override control
            self._control = control
        
        if self._control is not None:
            self._control.init()

        self._t = 0

        # initialize the graphics
        if self._animation is not None:

            # setup the plot
            self._ax = base.plotvol2(self.workspace)
        
            self._ax.set_xlabel('x')
            self._ax.set_ylabel('y')
            self._ax.set_aspect('equal')
            self._ax.figure.canvas.manager.set_window_title(
                f"Robotics Toolbox for Python (Figure {self._ax.figure.number})")

            self._animation.add(ax=self._ax)  # add vehicle animation to axis
            self._timer = plt.figtext(0.85, 0.95, '')  # display time counter

        # initialize the driver
        if isinstance(self._control, VehicleDriver):
            self._control.init(ax=self._ax)

    def step(self, u=None, animate=False):
        """
        Step simulator by one time step (superclass method)

        :return: odometry :math:`(\delta_d, \delta_\theta)`
        :rtype: ndarray(2)

        - ``veh.step((vel, steer))`` for a Bicycle vehicle model
        - ``veh.step((vel, vel_diff))`` for a Unicycle vehicle model
        - ``veh.step()`` as above but control is taken from the ``control``
          attribute which might be a function or driver agent.

        #. Integrates the vehicle forward one timestep
        #. Updates the stored state and state history
        #. Returns the odometry

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import Bicycle
            >>> bike = Bicycle()  # default bicycle model
            >>> bike.step(1, 0.2)  # one step: v=1, γ=0.2
            >>> bike.x
            >>> bike.step(1, 0.2)  # another step: v=1, γ=0.2
            >>> bike.x

        .. note:: Vehicle control input limits are applied.

        :seealso: :func:`control`, :func:`update`, :func:`run`
        """
        # determine vehicle control
        if u is not None:
            u = self.eval_control(u, self._x)
        else:
            u = self.eval_control(self._control, self._x)

        # update state (used to be function control() in MATLAB version)
        xd = self._dt * self.deriv(self._x, u)  # delta state

        # update state vector
        self._x += xd
        self._x_hist.append(tuple(self._x))

        # print('VEH', u, self.x)

        # odometry comes from change in state vector
        odo = np.r_[np.linalg.norm(xd[0:2]), xd[2]]

        if self._V is not None:
            odo += self.random.multivariate_normal((0, 0), self._V)

        # do the graphics
        if animate and self._animation:
            self._animation.update(self._x)
            if self._timer is not None:
                self._timer.set_text(f"t = {self._t:.2f}")
            plt.pause(self._dt)

        self._t += self._dt

        # be verbose
        if self._verbose:
            print(f"{self._t:8.2f}: u=({u[0]:8.2f}, {u[1]:8.2f}), x=({self._x[0]:8.2f}, {self._x[1]:8.2f}, {self._x[2]:8.2f})")

        return odo


    def eval_control(self, control, x):
        """
        Evaluate vehicle control input (superclass method)

        :param control: vehicle control
        :type control: [type]
        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :raises ValueError: bad control
        :return: vehicle control inputs
        :rtype: ndarray(2)

        Evaluates the control for this time step and state. Control can be:

            * a constant 2-tuple as the control inputs to the vehicle
            * a function called as ``f(vehicle, t, x)`` that returns a tuple
            * an interpolator called as f(t) that returns a tuple, see
              SciPy interp1d
            * a ``VehicleDriver`` subclass object

        .. note:: Vehicle steering, speed and acceleration limits are applied.
        """
        # was called control() in the MATLAB version

        if base.isvector(control, 2):
            # control is a constant
            u = base.getvector(control, 2)
        
        elif isinstance(control, VehicleDriver):
            # vehicle has a driver object
            u = control.demand()

        elif isinstance(control, interpolate.interpolate.interp1d):
            # control is an interp1d object
            u = control(self._t)

        elif callable(control):
            # control is a user function of time and state
            u = control(self, self._t, x)

        else:
            raise ValueError('bad control specified')

        # apply limits
        ulim = self.u_limited(u)
        return ulim

    def stopif(self, stop):
        """
        Stop the simulation (superclass method)

        :param stop: stop condition
        :type stop: bool

        A control function can stop the simulation initated by the ``run``
        method if ``stop`` is True.

        :seealso: :func:`run`
        """
        if stop:
            self._stopsim = True

    def plot(self, x=None, shape='box', block=False, size=True, **kwargs):
        """
        [summary] (superclass method)

        :param path: [description], defaults to None
        :type path: [type], optional
        :param block: [description], defaults to True
        :type block: bool, optional
        """
        if shape == 'triangle':
            L = size
            W = 0.6 * size
            vertices = [(L, 0), (-L, -W), (-L, W)]
        elif shape == 'box':
            L1 = size
            L2 = size
            W = 0.6 * size
            vertices = [(-L1, W), (0.6*L2, W), (L2, 0.5*W), (L2, -0.5*W), (0.6*L2, -W), (-L1, -W)]
        elif isinstance(shape, np.ndarray):
            vertices = shape
        else:
            raise ValueError('bad vehicle shape specified')

        vertices = np.array(vertices).T
        base.plot_poly(SE2(x) * vertices, close=True, **kwargs)

    def plot_xy(self, *args, block=False, **kwargs):
        if args is None and 'color' not in kwargs:
            kwargs['color'] = 'b'
        xyt = self.x_hist
        plt.plot(xyt[:, 0], xyt[:, 1], *args, **kwargs)
        plt.show(block=block)

    def plot_xyt(self, block=False, **kwargs):
        xyt = self.x_hist
        t = np.arange(0, xyt.shape[0] * self._dt, self._dt)
        plt.plot(xyt[:,0], xyt[:, :], **kwargs)
        plt.legend(['x', 'y', '$\\theta$'])
        plt.show(block=block)

    def limits_va(self, v):
        """
        Apply velocity and acceleration limits (superclass method)

        :param v: commanded velocity
        :type v: float
        :return: allowed velocity
        :rtype: float

        .. note:: This function is stateful, requires previous velocity,
            ``_v_prev`` attribute, to enable acceleration limiting.  This
            is reset at the start of each simulation.
        """
        # acceleration limit
        if self._accel_max is not None:
            if (v - self._v_prev) / self._dt > self._accel_max:
                v = self._v_prev + self._accelmax * self._dt;
            elif (v - self._v_prev) / self._dt < -self._accel_max:
                v = self._v_prev - self._accel_max * self._dt;
        self._v_prev = v
        
        # speed limit
        if self._speed_max is not None:
            v = np.clip(v, -self._speed_max, self._speed_max)
        return v


    def path(self, t=10, u=None, x0=None):
        """
        Compute path by integration (superclass method)

        :param t: [description], defaults to None
        :type t: [type], optional
        :param u: [description], defaults to None
        :type u: [type], optional
        :param x0: initial state, defaults to (0,0,0)
        :type x0: array_like(3), optional
        :return: time vector and state history
        :rtype: (ndarray(1), ndarray(n,3))

                    % XF = V.path(TF, U) is the final state of the vehicle (3x1) from the initial
            % state (0,0,0) with the control inputs U (vehicle specific).  TF is  a scalar to 
            % specify the total integration time.
            %
            % XP = V.path(TV, U) is the trajectory of the vehicle (Nx3) from the initial
            % state (0,0,0) with the control inputs U (vehicle specific).  T is a vector (N) of 
            % times for which elements of the trajectory will be computed.
            %
            % XP = V.path(T, U, X0) as above but specify the initial state.
            %
            % Notes::
            % - Integration is performed using ODE45.
            % - The ODE being integrated is given by the deriv method of the vehicle object.

             # t, x = veh.path(5, u=control)
    # print(t)
        """
        if x0 is None:
            x0 = np.zeros(3)

        def dynamics(t, x, vehicle, u):
            # u = vehicle.control(demand, x)
            
            return vehicle.deriv(x, u)

        if base.isscalar(t):
            t_span = (0, t)
            t_eval = np.linspace(0, t, 100)
        elif isinstance(t, np.ndarray):
            t_span = (t[0], t[-1])
            t_eval = t
        else:
            raise ValueError('bad time argument')
        sol = integrate.solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method="RK45", args=(self, u))

        return (sol.t, sol.y.T)
# ========================================================================= #

class Bicycle(VehicleBase):

    def __init__(self,
                L=1,
                steer_max=0.45 * pi,
                **kwargs
                ):
        r"""
        Create new bicycle kinematic model

        :param L: wheel base, defaults to 1
        :type L: float, optional
        :param steer_max: [description], defaults to :math:`0.45\pi`
        :type steer_max: float, optional
        :param **kwargs: additional arguments passed to :class:`Vehicle`
            constructor

        :seealso: :class:`.Vehicle`
        """
        super().__init__(**kwargs)

        self._l = L
        self._steer_max = steer_max

    def __str__(self):

        s = super().__str__()
        s += f"\n  L={self._l}, steer_max={self._steer_max:g}, speed_max={self._speed_max:g}, accel_max={self._accel_max:g}"
        return s

    @property
    def l(self):
        """
        Vehicle wheelbase

        :return: vehicle wheelbase
        :rtype: float
        """
        return self._l

    @property
    def radius_min(self):
        """
        Vehicle turning radius

        :return: radius of minimum possible turning circle
        :rtype: float

        :seealso: :meth:`curvature_max`
        """
        return self.l / np.tan(self.steer_max)

    @property
    def curvature_max(self):
        """
        Vehicle maximum path curature

        :return: maximum curvature
        :rtype: float

        :seealso: :meth:`radius_min`
        """
        return 1.0 / self.radius_min

    @property
    def steer_max(self):
        """
        Vehicle maximum steered wheel angle

        :return: maximum angle
        :rtype: float
        """
        return self._steer_max


    def f(self, x, odo, v=None):
        r"""
        Predict next state based on odometry

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param odo: vehicle odometry :math:`(\delta_d, \delta_\theta)`
        :type odo: array_like(2)
        :param v: additive odometry noise, defaults to (0,0)
        :type v: array_like(2), optional
        :return: predicted vehicle state
        :rtype: ndarray(3)

        Returns the predicted next state based on current state and odometry 
        value.  ``v`` is a random variable that represents additive
        odometry noise for simulation purposes.

        .. math::

            f: \vec{x}_k, \vec{d}, \vec{v} \mapsto \vec{x}_{k+1}

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import Bicycle
            >>> bike = Bicycle()  # default bicycle model
            >>> bike.f([0,0,0], [0.2, 0.1])

        .. note:: This is the state update equation used for EKF localization.

        :seealso: :meth:`Fx` :meth:`Fv`
        """
        odo = base.getvector(odo, 2)

        if isinstance(x, np.ndarray) and x.ndim == 2:
            # x is Nx3 set of vehicle states, do vectorized form
            # used by particle filter
            dd, dth = odo
            theta = x[:, 2]
            return np.array(x) + np.c_[dd * np.cos(theta), dd * np.sin(theta), np.full(theta.shape, dth)]
            # TODO: v for this case?
        else:
            # x is a vector
            x = base.getvector(x, 3)
            dd, dth = odo
            theta = x[2]

            if v is not None:
                v = base.getvector(v, 2)
                dd += v[0]
                dth += v[1]

            return np.array(x) + np.r_[dd * np.cos(theta), dd * np.sin(theta), dth]

    def Fx(self, x, odo):
        r"""
        Jacobian df/dx

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param odo: vehicle odometry :math:`(\delta_d, \delta_\theta)`
        :type odo: array_like(2)
        :return: Jacobian matrix 
        :rtype: ndarray(3,3)
        
        Returns the Jacobian matrix :math:`\frac{\partial \vec{f}}{\partial \vec{x}}` for
        the given odometry.

        :seealso: :func:`Bicycle.f`, :func:`Bicycle.Fv`
        """
        dd, dth = odo
        theta = x[2]

        # fmt: off
        J = np.array([
                [1,   0,  -dd * sin(theta)],
                [0,   1,   dd * cos(theta)],
                [0,   0,   1],
            ])
        # fmt: on
        return J

    def Fv(self, x, odo):
        r"""
        Jacobian df/dv

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param odo: vehicle odometry :math:`(\delta_d, \delta_\theta)`
        :type odo: array_like(2)
        :return: Jacobian matrix 
        :rtype: ndarray(3,2)
        
        Returns the Jacobian matrix :math:`\frac{\partial \vec{f}}{\partial \vec{v}}` for
        the given odometry.

        :seealso: :func:`Bicycle.f`, :func:`Bicycle.Fx`
        """
        dd, dth = odo
        theta = x[2]

        # fmt: off
        J = np.array([
                [cos(theta),    0],
                [sin(theta),    0],
                [0,           1],
            ])
        # fmt: on
        return J

    def deriv(self, x, u, limits=True):
        r"""
        Time derivative of state

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param u: control input
        :type u: array_like(2)
        :return: state derivative :math:`(\dot{x}, \dot{y}, \dot{\theta})`
        :rtype: ndarray(3)

        Returns the time derivative of state (3x1) at the state ``x`` with 
        inputs ``u``.
        """
        
        # unpack some variables
        theta = x[2]

        if limits:
            u = self.u_limited(u)
        v = u[0]
        gamma = u[1]
            
        return v * np.r_[
                cos(theta), 
                sin(theta), 
                tan(gamma) / self.l
                    ]

    def u_limited(self, u):

        # limit speed and steer angle
        ulim = np.array(u)
        ulim[0] = self.limits_va(u[0])
        ulim[1] = np.clip(u[1], -self._steer_max, self._steer_max)

        return ulim

# ========================================================================= #

class Unicycle(VehicleBase):

    def __init__(self,
                W=1,
                **kwargs):
        r"""
        Create new unicycle kinematic model

        :param W: vehicle width, defaults to 1
        :type W: float, optional
        :param **kwargs: additional arguments passed to :class:`Vehicle`
            constructor

        :seealso: :class:`.Vehicle`
        """
        super().__init__(**kwargs)
        self._w = W

    def __str__(self):

        s = super().__str__()
        s += f"\n  W={self._w}, steer_max={self._steer_max}, vel_max={self._vel_max}, accel_max={self.accel_max}"
        return s

    def f(self, x=None, odo=None, v=None):
        r"""
        Predict next state based on odometry

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param odo: vehicle odometry :math:`(\delta_d, \delta_\theta)`
        :type odo: array_like(2)
        :param v: additive odometry noise, defaults to (0,0)
        :type v: array_like(2), optional
        :return: predicted vehicle state
        :rtype: ndarray(3)

        Returns the predicted next state based on current state and odometry 
        value.  ``v`` is a random variable that represents additive
        odometry noise for simulation purposes.

        .. math::

            f: \vec{x}_k, \vec{d}, \vec{v} \mapsto \vec{x}_{k+1}

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import Unicycle
            >>> unic = Unicycle()  # default bicycle model
            >>> unic.f([0,0,0], [0.2, 0.1])

        .. note:: This is the state update equation used for EKF localization.

        :seealso: :meth:`Fx` :meth:`Fv`
        """
        
        # x is a vector
        x = base.getvector(x, 3)
        dd, dth = odo
        theta = x[2]

        if v is not None:
            v = base.getvector(v, 2)
            dd += v[0]
            dth += v[1]
            
        x_next = x + [dd * np.cos(theta), dd * np.sin(theta), np.ones(np.size(x, 0)*dth)]

        return x_next

    def Fx(self, x, odo):
        r"""
        Jacobian df/dx

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param odo: vehicle odometry :math:`(\delta_d, \delta_\theta)`
        :type odo: array_like(2)
        :return: Jacobian matrix 
        :rtype: ndarray(3,3)
        
        Returns the Jacobian matrix :math:`\frac{\partial \vec{f}}{\partial \vec{x}}` for
        the given odometry.

        :seealso: :func:`Bicycle.f`, :func:`Bicycle.Fv`
        """
        dd = odo[0]
        dth = odo[1]
        theta = x[2] + dth

        J = np.array([
                [1,   0,  -dd * sin(theta)],
                [0,   1,   dd * cos(theta)],
                [0,   0,   1],
            ])
        return J

    def Fv(self, x, odo):
        r"""
        Jacobian df/dv

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param odo: vehicle odometry :math:`(\delta_d, \delta_\theta)`
        :type odo: array_like(2)
        :return: Jacobian matrix 
        :rtype: ndarray(3,2)
        
        Returns the Jacobian matrix :math:`\frac{\partial \vec{f}}{\partial \vec{v}}` for
        the given odometry.

        :seealso: :func:`Bicycle.f`, :func:`Bicycle.Fx`
        """

        dd = odo[0]
        dth = odo[1]
        theta = x[2]


        J = np.array([
                [cos(theta),    0],
                [sin(theta),    0],
                [0,           1],
            ])

    def deriv(self, t, x, u):
        r"""
        Time derivative of state

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param u: control input
        :type u: array_like(2)
        :return: state derivative :math:`(\dot{x}, \dot{y}, \dot{\theta})`
        :rtype: ndarray(3)

        Returns the time derivative of state (3x1) at the state ``x`` with 
        inputs ``u``.

        .. note:: Vehicle speed and steering limits are not applied here
        """
        
        # unpack some variables
        theta = x[2]
        v = u[0]
        vdiff = u[1]

        return np.r_[
                v * cos(theta), 
                v * sin(theta), 
                vdiff / self.w
                    ]

    def u_limited(self, u):

        # limit speed and steer angle
        ulim = np.array(u)
        ulim[0] = self.limits_va(u[0])
        ulim[1] = np.maximum(-self._steer_max, np.minimum(self._steer_max, u[1]))

        return ulim

class DiffSteer(Unicycle):
    pass

if __name__ == "__main__":

    from roboticstoolbox import RandomPath

    V = np.eye(2) * 0.001
    robot = Bicycle(covar=V, animation="car")
    odo = robot.step((1, 0.3), animate=False)

    robot.control = RandomPath(workspace=10)

    robot.run(T=10)

    # from math import pi

    # V = np.diag(np.r_[0.02, 0.5 * pi / 180] ** 2)

    # v = VehiclePolygon()
    # # v = VehicleIcon('greycar2', scale=2, rotation=90)

    # veh = Bicycle(covar=V, animation=v, control=RandomPath(10), verbose=False)
    # print(veh)

    # odo = veh.step(1, 0.3)
    # print(odo)

    # print(veh.x)

    # print(veh.f([0, 0, 0], odo))

    # def control(v, t, x):
    #     goal = (6,6)
    #     goal_heading = atan2(goal[1]-x[1], goal[0]-x[0])
    #     d_heading = base.angdiff(goal_heading, x[2])
    #     v.stopif(base.norm(x[0:2] - goal) < 0.1)

    #     return (1, d_heading)

    # veh.control=RandomPath(10)
    # p = veh.run(1000, plot=True)
    # # plt.show()
    # print(p)

    # veh.plot_xyt_t()
    # veh.plot(p)

    # t, x = veh.path(5, u=control)
    # print(t)

    # fig, ax = plt.subplots()

    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)


    # v = VehicleAnimation.Polygon(shape='triangle', maxdim=0.1, color='r')
    # v = VehicleAnimation.Icon('car3.png', maxdim=2, centre=[0.3, 0.5])
    # v = VehicleAnimation.Icon('/Users/corkep/Dropbox/code/robotics-toolbox-python/roboticstoolbox/data/car1.png', maxdim=2, centre=[0.3, 0.5])
    # v = VehicleAnimation.icon('car3.png', maxdim=2, centre=[0.3, 0.5])
    # v = VehicleAnimation.marker()
    # v.start()
    # plt.grid(True)
    # # plt.axis('equal')

    # for theta in np.linspace(0, 2 * np.pi, 100):
    #     v.update([0, 0, theta])
    #     plt.pause(0.1)