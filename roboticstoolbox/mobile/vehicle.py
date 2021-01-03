"""
Python Vehicle
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing

Not ready for use yet.
"""
from abc import ABC, abstractmethod
from numpy import disp
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from roboticstoolbox.mobile import *
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *


class Vehicle(ABC):
    def __init__(self, covar=None, speed_max=None, l=None, x0=None, dt=None, r_dim=None,
                 steer_max=None, verbose=None):
        self._covar = np.array([]) # TODO, just set to None?
        self._r_dim = 0.2
        self._dt = 0.1
        self._x0 = np.zeros((3,), dtype=float)
        self._x = None
        self._speed_max = 1
        self._vehicle_plot = None
        self._driver = None
        self._odometry = None

        # TODO: Do we even need these if statements?
        if covar is not None:
            self._v = covar
        if speed_max is not None:
            self._speed_max = speed_max
        if l is not None:
            self._l = l
        if x0 is not None:
            self._x0 = base.getvector(x0, 3)
            # TODO: Add assert
        if dt is not None:
            self._dt = dt
        if r_dim is not None:
            self._r_dim = r_dim
        if verbose is not None:
            self._verbose = verbose
        if steer_max is not None:
            self._steer_max = steer_max

        self._x_hist = np.empty((0,3))

    def __str__(self):
        s = f"{self.__class__.__name__}: "
        s += f"x = {self._x}"
        return s

    @property
    def accel_max(self):
        return self._accel_max

    @property
    def speed_max(self):
        return self._speed_max

    @property
    def x(self):
        return self._v_prev

    @property
    def l(self):
        return self._l

    @property
    def x_hist(self):
        return self._x_hist

    @property
    def dim(self):
        return self._dim

    @property
    def r_dim(self):
        return self._r_dim

    @property
    def dt(self):
        return self._dt

    @property
    def v(self):
        return self._v

    @property
    def odometry(self):
        return self._odometry

    @property
    def verbose(self):
        return self._verbose

    @property
    def driver(self):
        return self._driver

    @property
    def x0(self):
        return self._x0

    @property
    def v_handle(self):
        return self._v_handle

    @property
    def v_trail(self):
        return self._v_trail

    @property
    def driver(self):
        return self._driver

    # Example
    def init(self, x0=None):
        if x0 is not None:
            self._x = base.getvector(x0, 3)
        else:
            self._x = self._x0

        self._x_hist = np.empty((0,3))
        if self._driver is not None:
            self._driver.init()  # TODO: make this work?

        self._v_handle = np.array([])

    def path(self, t=None, u=None, y0=None):  # TODO: Might be the source of some errors
        tt = None
        yy = None

        if len(t) == 1:
            tt = np.array([0, t[-1]])
        else:
            tt = t

        if y0 is None:
            y0 = np.array([0, 0, 0])

        ode_out = integrate.solve_ivp(self.deriv(t, None, u), [tt[0], tt[-1]], y0, t_eval=tt, method="RK45")
        y = np.transpose(ode_out.y)

        if t is None:
            plt.plot(y[:][0], y[:][2])
            plt.xlabel('X')
            plt.ylabel('Y')
        else:
            yy = y
            if len(t) == 1:
                yy = yy[-1][:]

        return yy

    # This function is overridden by the child class
    @abstractmethod
    def deriv(self, t, y=None, u=None):  # TODO: I have no idea where Y comes from, here!
        pass

    def add_driver(self, driver):
        self._driver = driver
        driver._veh = self

    def update(self, u):
        v = u[0]
        theta = self._x[2]
        xp = np.array(self._x)
        self._x[0] += v * self._dt * np.cos(theta)
        self._x[1] += v * self._dt * np.sin(theta)
        self._x[2] += v * self._dt / self._l * u[1]
        odo = np.r_[np.linalg.norm(self._x[0:2] - xp[0:2]), self._x[2] - xp[2]]
        self._odometry = odo

        self._x_hist = np.vstack((self._x_hist, self._x))
        return odo

    def step(self, speed=None, steer=None):
        u = self.control(speed, steer)
        odo = self.update(u)

        if self._v is not None:
            odo = self._odometry + linalg.sqrtm(self._v) @ np.random.randn(2)

        return odo

    def control(self, speed=None, steer=None):
        u = np.zeros(2)
        if speed is None and steer is None:
            if self._driver is not None:
                speed, steer = self._driver.demand()
            else:
                speed = 0
                steer = 0

        if self._speed_max is None:
            u[0] = speed
        else:
            u[0] = np.minimum(self._speed_max, np.maximum(-self._speed_max, speed))

        if self._steer_max is not None:
            u[1] = np.maximum(-self._steer_max, np.minimum(self._steer_max, steer))
        else:
            u[1] = steer

        return u

    def run(self, n_steps=None, plot=True):
        if n_steps is None:
            n_steps = 1000
        if self._driver is not None:
            self._driver.init()
            if plot:
                self._driver.plot()

        if plot:
            self.plot()
        for i in range(n_steps):
            self.step()
            if plot:
                self.plot()
                plt.pause(0.1)
            print(i, self._x)

        return self._x_hist

    def run_2(self, t, x0, speed, steer):
        self.init(x0)

        for i in range(0, (t/self._dt)):
            self.update(np.array([speed, steer]))

        p = self._x_hist
        return p

    def plot(self, path=None):
        if path is None:
            if self._vehicle_plot is None:
                self._vehicle_plot = VehicleAnimation(self._x)
            else:
                self._vehicle_plot.update(self._x)

    def plot_xy(self):
        # TODO: this also has some vargin
        xyt = self._x_hist
        plt.plot(xyt[0, :], xyt[1, :])

    def verbiosity(self, v):
        self._verbose = v

    def limits_va(self, v):
        # acceleration limit
        if (v - self._vprev) / self._dt > self._accelmax:
            v = self._vprev + self._accelmax * self._dt;
        elif (v - self._vprev) / self._dt < -self._accelmax:
            v = self._vprev - self._accelmax * self._dt;
        self._vprev = v
        
        # speed limit
        return min(self._speed_max, max(v, -self._speed_max));


class VehicleAnimation:

    # TODO update with triangle, image etc, handle orientation

    def __init__(self, x=None):

        self._ax = plt.gca()
        self._reference = plt.plot(x[0], x[1], marker='o', markersize=12)[0]

    def update(self, x):
        self._reference.set_xdata(x[0])
        self._reference.set_ydata(x[1])
        plt.draw()

    def __del__(self):

        if self._reference is not None:
            print('deleting vehicle graphics object')
            self._ax.remove(self._reference)



# ========================================================================= #

class Bicycle(Vehicle):

    def __init__(self,
                steer_max=None,
                accel_max=None,
                covar=0,
                speed_max=1,
                l=1, 
                x0=None,
                dt=0.1, 
                r_dim=0.2, 
                verbose=None
                ):
        super().__init__(covar, speed_max, l, x0, dt, r_dim, steer_max, verbose)

        self._l = 1
        self._steer_max = 0.5
        self._accel_max = np.inf

        if covar is not None:
            self._v = covar
        if speed_max is not None:
            self._speed_max = speed_max
        if l is not None:
            self._l = l
        if x0 is not None:
            self._x0 = x0
            # TODO: Add assert
        if dt is not None:
            self._dt = dt
        if r_dim is not None:
            self._r_dim = r_dim
        if verbose is not None:
            self._verbose = verbose
        if steer_max is not None:
            self._steer_max = steer_max
        if accel_max is not None:
            self._accel_max = accel_max

        self._v_prev = 0
        self._x = self._x0

    def __str__(self):

        s = super().__str__()
        s += f"\n  L={self._l}, steer_max={self._steer_max}, speed_max={self._speed_max}, accel_max={self._accel_max}"
        return s

    @property
    def steer_max(self):
        return self._steer_max


    def f(self, x=None, odo=None, w=None):
        """
        [summary]

        :param x: [description], defaults to None
        :type x: [type], optional
        :param odo: [description], defaults to None
        :type odo: [type], optional
        :param w: [description], defaults to None
        :type w: [type], optional
        :return: [description]
        :rtype: [type]

                    %Unicycle.f Predict next state based on odometry

        speed_maxXN = V.f(X, ODO) is the predicted next state XN (1x3) based on current
        speed_maxstate X (1x3) and odometry ODO (1x2) = [distance, heading_change].

        speed_maxXN = V.f(X, ODO, W) as above but with odometry noise W.

        speed_maxNotes::
        speed_max- Supports vectorized operation where X and XN (Nx3).
        """
        x = base.getvector(x, 3)
        odo = base.getvector(odo, 2)

        if w is None:
            w = [0, 0]

        dd = odo[0] + w[0]
        dth = odo[1] + w[1]
        thp = x[2]
        # TODO not sure when vectorized version is needed
        return x + np.r_[dd * np.cos(thp), dd * np.sin(thp), dth]

    def deriv(self, t, x, u):  # TODO: I have no idea where Y comes from, here!
        """
        [summary]

        :param t: [description]
        :type t: [type]
        :param x: [description]
        :type x: [type]
        :param u: [description]
        :type u: [type]
        :return: [description]
        :rtype: [type]

        Bicycle.deriv  Time derivative of state

        speed_maxDX = V.deriv(T, X, U) is the time derivative of state (3x1) at the state
        speed_maxX (3x1) with input U (2x1).

        speed_maxNotes::
        speed_max- The parameter T is ignored but  called from a continuous time integrator such as ode45 or
        speed_max  Simulink.
        """
        
        # unpack and implement speed and steer angle limits

        theta = x[2]

        v = self.limits_va(u[0])

        gamma = u[1]
        gamma = min(self._steermax, max(gamma, -self._steermax))
            
        return np.r_[v * cos(theta), v * sin(theta), v / self.l * tan(gamma)]


    def Fx(self, x, odo):
        """
        [summary]

        :param x: [description]
        :type x: [type]
        :param odo: [description]
        :type odo: [type]
        
        Bicycle.Fx  Jacobian df/dx

        speed_maxJ = V.Fx(X, ODO) is the Jacobian df/dx (3x3) at the state X, for
        speed_maxodometry input ODO (1x2) = [distance, heading_change].

        speed_maxSee also Bicycle.f, Vehicle.Fv.
        """
        dd = odo[0]
        dth = odo[1]
        thp = x[2] + dth

        J = np.array([
                [1,   0,  -dd*sin(thp)],
                [0,   1,   dd*cos(thp)],
                [0,   0,   1],
            ])
        return J

    def Fv(self, x, odo):
        """
        [summary]

        :param x: [description]
        :type x: [type]
        :param odo: [description]
        :type odo: [type]

        Bicycle.Fv  Jacobian df/dv

        speed_maxJ = V.Fv(X, ODO) is the Jacobian df/dv (3x2) at the state X, for
        speed_maxodometry input ODO (1x2) = [distance, heading_change].

        speed_maxSee also Bicycle.F, Vehicle.Fx.
        """

        dd = odo[0]
        dth = odo[1]
        thp = x[2]


        J = np.array([
                [cos(thp),    0],
                [sin(thp),    0],
                [0,           1],
            ])

# ========================================================================= #

class Unicycle(Vehicle):

    def __init__(self,
                steer_max=None,
                accel_max=None,
                covar=0,
                speed_max=1,
                w=1,
                x0=None,
                dt=0.1,
                r_dim=0.2, 
                verbose=None):
        super().__init__(covar, speed_max, l, x0, dt, r_dim, steer_max, verbose)

        self._w = 1
        self._steer_max = 0.5
        self._accel_max = np.inf

        if covar is not None:
            self._v = covar
        if speed_max is not None:
            self._speed_max = speed_max
        if l is not None:
            self._l = l
        if x0 is not None:
            self._x0 = x0
            # TODO: Add assert
        if dt is not None:
            self._dt = dt
        if r_dim is not None:
            self._r_dim = r_dim
        if verbose is not None:
            self._verbose = verbose
        if steer_max is not None:
            self._steer_max = steer_max
        if accel_max is not None:
            self._accel_max = accel_max

        self._v_prev = 0
        self._x = self._x0

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

    def deriv(self, t, x, u):  # TODO: I have no idea where Y comes from, here!
        theta = x[2]
        v = u[0]
        vd = u[1]

        return np.r_[v * cos(theta), v * sin(theta), v / self.w]

# ========================================================================= #

class RandomPath:
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




    TODO
    should be a subclass of VehicleDriver
    Vehicle should be an abstract superclass
    dim should be checked, can be a 4-vector like axis()
    """

    def __init__(self, dim, speed=1, dthresh=0.05, show=True, seed=None):
        """
        [summary]

        :param dim: [description]
        :type dim: [type]
        :param speed: [description], defaults to 1
        :type speed: int, optional
        :param dthresh: [description], defaults to 0.05
        :type dthresh: float, optional
        :param show: [description], defaults to True
        :type show: bool, optional
        :raises ValueError: [description]

        %RandomPath.RandomPath Create a driver object
        %
        % D = RandomPath(D, OPTIONS) returns a "driver" object capable of driving
        % a Vehicle subclass object through random waypoints.  The waypoints are positioned
        % inside a rectangular region of dimension D interpreted as:
        %      - D scalar; X: -D to +D, Y: -D to +D
        %      - D (1x2); X: -D(1) to +D(1), Y: -D(2) to +D(2)
        %      - D (1x4); X: D(1) to D(2), Y: D(3) to D(4)
        %
        % Options::
        % 'speed',S      Speed along path (default 1m/s).
        % 'dthresh',D    Distance from goal at which next goal is chosen.
        %
        % See also Vehicle.
                """
        
        # TODO options to specify region, maybe accept a Map object?
        
        dim = base.getvector(dim)

        if len(dim) == 1:
                self._xrange = np.r_[-dim, dim]
                self._yrange = np.r_[-dim, dim]
        elif len(dim) == 2:
                self._xrange = np.r_[-dim[0], dim[0]]
                self._yrange = np.r_[-dim[1], dim[1]]
        elif len(dim) == 4:
                self._xrange = np.r_[dim[0], dim[1]]
                self._yrange = np.r_[dim[2], dim[3]]
        else:
            raise ValueError('bad dimension specified')
        
        self._speed = speed
        self._dthresh = dthresh * np.diff(self._xrange)
        self._show = show
        self._h_goal = None
        
        self._d_prev = np.inf
        self._randstream = np.random.RandomState()
        self._seed = seed
        self.verbose = True

    def init(self):
        """
        [summary]
        
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
        self._randstream.seed(self._seed)
        # delete(driver.h_goal);   % delete the goal
        # driver.h_goal = [];
        

    # called by Vehicle superclass
    
    def plot(self):
        plt.clf()
        plt.axis(np.r_[self._xrange, self._yrange])
        
        plt.xlabel('x blah')
        plt.ylabel('y')


    ## private method, invoked from demand() to compute a new waypoint
    
    def choose_goal(self):
        
        # choose a uniform random goal within inner 80% of driving area
        while True:
            r = self._randstream.rand() * 0.8 + 0.1
            gx = self._xrange @ np.r_[r, 1-r]

            r = self._randstream.rand() * 0.8 + 0.1
            gy = self._yrange @ np.r_[r, 1-r]

            self._goal = np.r_[gx, gy]

            # check not too close to last goal
            if np.linalg.norm(self._goal - self._veh._x[0:2]) > 2 * self._dthresh:
                break

        if self.verbose:
            print(f"set goal: {self._goal}")

        # update the goal marker
        if self._show and self._h_goal is None:
            self._h_goal = plt.plot(self._goal[0], self._goal[1], 'rd', markersize=12, markerfacecolor='r')[0]
        else:
            self._h_goal.set_xdata(self._goal[0])
            self._h_goal.set_ydata(self._goal[1])


    def demand(self):
        """    %RandomPath.demand Compute speed and heading to waypoint
            %
            % [SPEED,STEER] = R.demand() is the speed and steer angle to
            % drive the vehicle toward the next waypoint.  When the vehicle is
            % within R.dtresh a new waypoint is chosen.
            %
            % See also Vehicle."""

        if self._goal is None:
            self.choose_goal()

        # if nearly at goal point, choose the next one
        d = np.linalg.norm(self._veh._x[0:2] - self._goal)
        if d < self._dthresh:
            self.choose_goal()
        # elif d > 2 * self._d_prev:
        #     self.choose_goal()
        # self._d_prev = d

        speed = self._speed

        goal_heading = math.atan2(self._goal[1]-self._veh._x[1], self._goal[0]-self._veh._x[0])
        d_heading = base.angdiff(goal_heading, self._veh._x[2])
        steer = d_heading

        print('  ', speed, steer)
        return speed, steer

    def __str__(self):
        """%RandomPath.char Convert to string
        %
        % s = R.char() is a string showing driver parameters and state in in 
        % a compact human readable format. """

        s = 'RandomPath driver object\n'
        s += f"  current goal={self._goal}, X {self._xrange[0]} : {self._xrange[1]}; Y {self._yrange[0]} : {self._yrange[1]}, dthresh={self.dthresh}"


if __name__ == "__main__":
    from math import pi

    V = np.diag(np.r_[0.02, 0.5 * pi / 180] ** 2)

    veh = Bicycle(covar=V)
    print(veh)

    odo = veh.step(1, 0.3)
    print(odo)

    print(veh.x)

    print(veh.f([0, 0, 0], odo))

    veh.add_driver(RandomPath(10))

    p = veh.run(1000, plot=True)
    # plt.show()
    print(p)
    veh.plot(p)