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
from roboticstoolbox import loaddata
from roboticstoolbox.mobile.drivers import VehicleDriver


class Vehicle(ABC):
    def __init__(self, covar=None, speed_max=np.inf, accel_max=np.inf, x0=None, dt=0.1,
                 control=None, animation=None, verbose=False, dim=10):
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
        :param dim: dimensions of 2D plot area, defaults to (-10:10) x (-10:10),
            see :func:`~spatialmath.base.animate.plotvol2`
        :type dims: float, array_like(2), , array_like(4)

        This is an abstract superclass that simulates the motion of a mobile
        robot under the action of a controller.  The controller provides
        control inputs to the vehicle, and the output odometry is returned.
        The true state, effectively unknowable in practice, is computed
        and accessible.

        :seealso: :func:`Bicycle`, :func:`Unicycle`
        """

        if covar is None:
            covar = np.zeros((2,2))
        self._V = covar
        self._dt = dt
        if x0 is None:
            x0 = np.zeros((3,), dtype=float)
        else:
            x0 = base.getvector(x0)
            if len(x0) not in (2,3):
                raise ValueError('x0 must be length 2 or 3')
        self._x0 = x0
        self._x = x0
        
        self._speed_max = speed_max
        self._accel_max = accel_max
        self._v_prev = 0

        self._vehicle_plot = None
        if control is not None:
            self.add_driver(control)
        self._animation = animation

        self._dt = dt
        self._t = 0
        self._stopsim = False

        self._verbose = verbose
        self._plot = False

        self._dim = dim

        self._x_hist = np.empty((0,len(x0)))

    def __str__(self):
        """
        String representation of vehicle (superclass method)

        :return: String representation of vehicle object
        :rtype: str
        """
        s = f"{self.__class__.__name__}: "
        s += f"x = {self._x}"
        return s

    @property
    def x(self):
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
        return self._x_hist

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
            * a function called as ``f(vehicle, t, x)`` that returns a tuple
            * an interpolator called as f(t) that returns a tuple, see
              SciPy interp1d
            * a driver agent, subclass of :func:`VehicleDriver`

        Example:

        .. runblock:: pycon

            >>> bike = Bicycle()
            >>> bike.control = RandomPath(10)
            >>> print(bike)

        :seealso: :func:`eval_control`, :func:`RandomPath`, :func:`PurePursuit`
        """
        self._control = control
        if isinstance(control, VehicleDriver):
            # if this is a driver agent, connect it to the vehicle
            control.vehicle = self


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


    def run(self, N=1000, x0=None, control=None, animation=None, plot=True):
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

        self.init(plot=plot, control=control, animation=animation, x0=x0)
            
        for i in range(N):
            self.step()

            # do the graphics
            if self._plot:
                if self._animation:
                    self._animation.update(self._x)
                if self._timer is not None:
                    self._timer.set_text(f"t = {self._t:.2f}")
                plt.pause(self._dt)

            # check for user requested stop
            if self._stopsim:
                print('USER REEQUESTED STOP AT time', self._t)
                break

        return self._x_hist

    def init(self, x0=None, animation=None, plot=False, control=None):
        """
        Initialize for simulation (superclass method)

        :param x0: Initial state, defaults to value given to Vehicle constructor
        :type x0: array_like(3) or array_like(2)
        :param animation: vehicle animation object, defaults to None
        :type animation: VehicleAnimation subclass, optional
        :param plot: Enable plotting, defaults to False
        :type plot: bool, optional

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
            self._x = self._x0

        self._x_hist = np.empty((0,3))

        if control is not None:
            self._control = control

        self._t = 0

        # initialize the graphics
        self._plot = plot
        if plot:

            if animation is None:
                animation = self._animation  # get default animation if set
            else:
                # use default animation
                animation = VehiclePolygon("car")
            self._animation = animation

            # setu[ the plot]
            plt.clf()

            self._ax = base.plotvol2(self._dim)
        
            plt.xlabel('x')
            plt.ylabel('y')
            self._ax.set_aspect('equal')
            self._ax.figure.canvas.set_window_title(f"Robotics Toolbox for Python (Figure {self._ax.figure.number})")

            animation.add()  # add vehicle animation to axis
            self._timer = plt.figtext(0.85, 0.95, '')  # display time counter

        # initialize the driver
        if isinstance(self._control, VehicleDriver):
            self._control.init(ax=self._ax)

    def step(self, u1=None, u2=None):
        """
        Step simulator by one time step (superclass method)

        :return: odometry :math:`(\delta_d, \delta_\theta)`
        :rtype: ndarray(2)

        - ``veh.step(vel, steer)`` for a Bicycle vehicle model
        - ``veh.step((vel, steer))`` as above but control is a tuple
        - ``veh.step(vel, vel_diff)`` for a Unicycle vehicle model
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
        if u1 is not None:
            if u2 is not None:
                u = self.eval_control((u1, u2), self._x)
            else:
                u = self.eval_control(u1, self._x)
        else:
            u = self.eval_control(self._control, self._x)

        # update state (used to be function control() in MATLAB version)
        xd = self._dt * self.deriv(self._x, u)  # delta state

        # update state vector
        self._x += xd
        self._x_hist = np.vstack((self._x_hist, self._x))

        # odometry comes from change in state vector
        odo = np.r_[np.linalg.norm(xd[0:2]), xd[2]]

        if self._V is not None:
            odo += linalg.sqrtm(self._V) @ np.random.randn(2)

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

    def plot(self, path=None, block=True):
        """
        [summary] (superclass method)

        :param path: [description], defaults to None
        :type path: [type], optional
        :param block: [description], defaults to True
        :type block: bool, optional
        """
 
        plt.plot(path[:,0], path[:,1])
        plt.show(block=block)

    def plot_x_y(self, block=True, **kwargs):
        xyt = self._x_hist
        plt.plot(xyt[:,0], xyt[:, 1], **kwargs)
        plt.show(block=block)

    def plot_xyt_t(self, block=True, **kwargs):
        xyt = self._x_hist
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
        if (v - self._v_prev) / self._dt > self._accel_max:
            v = self._v_prev + self._accelmax * self._dt;
        elif (v - self._v_prev) / self._dt < -self._accel_max:
            v = self._v_prev - self._accel_max * self._dt;
        self._v_prev = v
        
        # speed limit
        return min(self._speed_max, max(v, -self._speed_max));


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

        def dynamics(t, x, vehicle, demand):
            u = vehicle.control(demand, x)
            
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

        return (sol.t, sol.y)
# ========================================================================= #

class Bicycle(Vehicle):

    def __init__(self,
                l=1,
                steer_max=0.45 * pi,
                **kwargs
                ):
        r"""
        Create new bicycle kinematic model

        :param l: wheel base, defaults to 1
        :type l: float, optional
        :param steer_max: [description], defaults to :math:`0.45\pi`
        :type steer_max: float, optional
        :param **kwargs: additional arguments passed to :class:`Vehicle`
            constructor

        :seealso: :class:`.Vehicle`
        """
        super().__init__(**kwargs)

        self._l = l
        self._steer_max = steer_max

    def __str__(self):

        s = super().__str__()
        s += f"\n  L={self._l}, steer_max={self._steer_max}, speed_max={self._speed_max}, accel_max={self._accel_max}"
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
    def steer_max(self):
        """
        Vehicle maximum steering wheel angle

        :return: maximum angle
        :rtype: float
        """
        return self._steer_max


    def f(self, x, odo, w=None):
        r"""
        Predict next state based on odometry

        :param x: vehicle state :math:`(x, y, \theta)`
        :type x: array_like(3)
        :param odo: vehicle odometry :math:`(\delta_d, \delta_\theta)`
        :type odo: array_like(2)
        :param w: [description], defaults to (0,0)
        :type w: array_like(2), optional
        :return: predicted vehicle state
        :rtype: ndarray(3)

        Returns the predicted next state based on current state and odometry 
        value.  ``w`` is a random variable that represents additive
        odometry noise for simulation purposes.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import Bicycle
            >>> bike = Bicycle()  # default bicycle model
            >>> bike.f([0,0,0], [0.2, 0.1])

        .. note:: This is the state update equation used for EKF localization.
        """
        x = base.getvector(x, 3)
        odo = base.getvector(odo, 2)

        dd = odo[0]
        dth = odo[1]
        thp = x[2]

        if w is not None:
            w = base.getvector(w, 2)
            dd += w[0]
            dth += w[1]

        # TODO not sure when vectorized version is needed
        return x + np.r_[dd * np.cos(thp), dd * np.sin(thp), dth]

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
        thp = x[2] + dth

        J = np.array([
                [1,   0,  -dd * sin(thp)],
                [0,   1,   dd * cos(thp)],
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
        thp = x[2]


        J = np.array([
                [cos(thp),    0],
                [sin(thp),    0],
                [0,           1],
            ])

    def deriv(self, x, u):
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
        ulim[1] = np.maximum(-self._steer_max, np.minimum(self._steer_max, u[1]))

        return ulim

# ========================================================================= #

class Unicycle(Vehicle):

    def __init__(self,
                w=1,
                **kwargs):
        r"""
        Create new unicycle kinematic model

        :param w: vehicle width, defaults to 1
        :type w: float, optional
        :param **kwargs: additional arguments passed to :class:`Vehicle`
            constructor

        :seealso: :class:`.Vehicle`
        """
        super().__init__(**kwargs)
        self._w = w

    def __str__(self):

        s = super().__str__()
        s += f"\n  W={self._w}, steer_max={self._steer_max}, vel_max={self._vel_max}, accel_max={self.accel_max}"
        return s

    def f(self, x=None, odo=None, w=None):
        
        if w is None:
            w = np.array([0, 0])

        dd = odo[0] + w[0]
        dth = odo[1] + w[1]
        thp = x[:, 2]
        x_next = x + [dd * np.cos(thp), dd * np.sin(thp), np.ones(np.size(x, 0)*dth)]

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
        thp = x[2] + dth

        J = np.array([
                [1,   0,  -dd * sin(thp)],
                [0,   1,   dd * cos(thp)],
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
        thp = x[2]


        J = np.array([
                [cos(thp),    0],
                [sin(thp),    0],
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


if __name__ == "__main__":
    from math import pi

    V = np.diag(np.r_[0.02, 0.5 * pi / 180] ** 2)

    v = VehiclePolygon()
    # v = VehicleIcon('greycar2', scale=2, rotation=90)

    veh = Bicycle(covar=V, animation=v, control=RandomPath(10), verbose=False)
    print(veh)

    odo = veh.step(1, 0.3)
    print(odo)

    print(veh.x)

    print(veh.f([0, 0, 0], odo))

    def control(v, t, x):
        goal = (6,6)
        goal_heading = atan2(goal[1]-x[1], goal[0]-x[0])
        d_heading = base.angdiff(goal_heading, x[2])
        v.stopif(base.norm(x[0:2] - goal) < 0.1)

        return (1, d_heading)

    veh.control=RandomPath(10)
    p = veh.run(1000, plot=True)
    # plt.show()
    print(p)

    veh.plot_xyt_t()
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