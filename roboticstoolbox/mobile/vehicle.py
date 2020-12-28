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
        self._x0 = np.r_[0, 0, 0]
        self._x = None
        self._speed_max = 1
        self._v_handle = np.array([])
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

        self._x_hist = np.array([])

    def __str__(self):
        s = f"{self.__class__.__name__}: "
        s += f"x = {self.x}"
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
            self._x = x0
        else:
            self._x = self._x0

        self._x_hist = np.array([])

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
        xp = self._x
        self._x[0] = self._x[0] + u[0] * self._dt * np.cos(self._x[2])
        self._x[1] = self._x[1] + u[0] * self._dt * np.sin(self._x[2])
        self._x[2] = self._x[2] + u[0] * self._dt / self._l * u[1]
        odo = np.array([col_norm(self._x[0:2] - xp[0:2], self._x[2] - xp[2])])  # TODO: Right indexing?
        self._odometry = odo

        self._x_hist = np.concatenate(self._x_hist, np.transpose(self._x))
        return odo

    def step(self, speed=None, steer=None):
        u = self.control(speed, steer)
        odo = self.update(u)

        if self._v is not None:
            odo = self._odometry + np.random.rand(1, 2) * linalg.sqrtm(self._v)  # TODO: linalg imported?

        return odo

    def control(self, speed=None, steer=None):
        u = np.zeros(2)
        if speed is None and steer is None:
            if self._driver is not None:
                speed, steep = self._driver.demand()
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

    def run(self, n_steps=None):
        if n_steps is None:
            n_steps = 1000
        if self._driver is not None:
            self._driver.init()
        if self._driver is not None:
            self._driver.plot()

        self._plot()
        for i in range(0, n_steps):
            self.step()
            # TODO: There's a nargout here... is this really needed or can it be done differently?

        p = self._x_hist
        return p

    def run_2(self, t, x0, speed, steer):
        self.init(x0)

        for i in range(0, (t/self._dt)):
            self.update(np.array([speed, steer]))

        p = self._x_hist
        return p

    def plot(self):
        # TODO: Add vargin arguments. There's more here.
        if self._v_handle is None:
            self._v_handle = plot_v(self._x)

        pos = self._x
        plot_v(self._v_handle, pos)

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


def plot_v(handle=None, pose_x=None):
    # TODO add vargin stuff
    if handle is not None:
        plot_vehicle(pose_x, handle)
    else:
        handle = None
        fillcolor = 'b'
        alpha = 0.5
        h = plot_vehicle(pose_x, handle, fillcolor, alpha)
        return h


# ========================================================================= #

class Bicycle(Vehicle):

    def __init__(self,
                steer_max=None,
                accel_max=None,
                covar=0,
                speed_max=1,
                l=1, 
                x0=np.array([0, 0, 0]),
                dt=0.1, 
                r_dim=0.2, 
                verbose=None
                ):
        super().__init__(covar, speed_max, l, x0, dt, r_dim, steer_max, verbose)

        self._x = np.r_[0, 0, 0]
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

        s = super.__str__(self)
        s += f"\n  L={self._l}, steer_max={self._steer_max}, vel_max={self._vel_max}, accel_max={self.accel_max}"
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
        
        if w is None:
            w = np.array([0, 0])

        dd = odo[0] + w[0]
        dth = odo[1] + w[1]
        thp = x[:, 2]
        # TODO not sure when vectorized version is needed
        x_next = x + [dd * np.cos(thp), dd * np.sin(thp), np.ones(np.size(x, 0)*dth)]

        return x_next

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
                x0=np.array([0, 0, 0]),
                dt=0.1,
                r_dim=0.2, 
                verbose=None):
        super().__init__(covar, speed_max, l, x0, dt, r_dim, steer_max, verbose)

        self._x = np.r_[0, 0, 0]
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

        s = super.__str__(self)
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