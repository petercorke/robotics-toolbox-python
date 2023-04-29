#!/usr/bin/env python3

"""
Python EKF Planner
@Author: Peter Corke, original MATLAB code and Python version
@Author: Kristian Gibson, initial MATLAB port

Based on code by Paul Newman, Oxford University, 
http://www.robots.ox.ac.uk/~pnewman
"""

from collections import namedtuple

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import animation

import spatialmath.base as smb

"""
Monte-carlo based localisation for estimating vehicle pose based on
odometry and observations of known landmarks.
"""

# TODO: refactor this and EKF, RNG, history, common plots, animation, movie


class ParticleFilter:
    def __init__(
        self,
        robot,
        sensor,
        R,
        L,
        nparticles=500,
        seed=0,
        x0=None,
        verbose=False,
        animate=False,
        history=True,
        workspace=None,
    ):
        """
        Particle filter

        :param robot: robot motion model
        :type robot: :class:`VehicleBase` subclass,
        :param sensor: vehicle mounted sensor model
        :type sensor: :class:`SensorBase` subclass
        :param R: covariance of the zero-mean Gaussian noise added to the particles at each step (diffusion)
        :type R: ndarray(3,3)
        :param L: covariance used in the sensor likelihood model
        :type L: ndarray(2,2)
        :param nparticles: number of particles, defaults to 500
        :type nparticles: int, optional
        :param seed: random number seed, defaults to 0
        :type seed: int, optional
        :param x0: initial state, defaults to [0, 0, 0]
        :type x0: array_like(3), optional
        :param verbose: display extra debug information, defaults to False
        :type verbose: bool, optional
        :param history: retain step-by-step history, defaults to True
        :type history: bool, optional
        :param workspace: dimension of workspace, see :func:`~spatialmath.base.graphics.expand_dims`
        :type workspace: scalar, array_like(2), array_like(4)

        This class implements a Monte-Carlo estimator or particle filter for
        vehicle state, based on odometry, a landmark map, and landmark
        observations.  The state of each particle is a possible vehicle
        configuration :math:`(x,y,\theta)`.  Bootstrap particle resampling is
        used.

        The working area is defined by ``workspace`` or inherited from the
        landmark map attached to the ``sensor`` (see
        :func:`~spatialmath.base.graphics.expand_dims`):

        ==============  =======  =======
        ``workspace``   x-range  y-range
        ==============  =======  =======
        A (scalar)      -A:A     -A:A
        [A, B]           A:B      A:B
        [A, B, C, D]     A:B      C:D
        ==============  =======  =======

        Particles are initially distributed uniform randomly over this area.

        Example::

            V = np.diag([0.02, np.radians(0.5)]) ** 2
            robot = Bicycle(covar=V, animation="car", workspace=10)
            robot.control = RandomPath(workspace=robot)

            map = LandmarkMap(nlandmarks=20, workspace=robot.workspace)

            W = np.diag([0.1, np.radians(1)]) ** 2
            sensor = RangeBearingSensor(robot, map, covar=W, plot=True)

            R = np.diag([0.1, 0.1, np.radians(1)]) ** 2
            L = np.diag([0.1, 0.1])
            pf = ParticleFilter(robot, sensor, R, L, nparticles=1000)

            pf.run(T=10)

            map.plot()
            robot.plot_xy()
            pf.plot_xy()

            plt.plot(pf.get_std()[:100,:])

        .. note:: Set ``seed=0`` to get different behaviour from run to run.

        :seealso: :meth:`run`
        """
        self._robot = robot
        self._sensor = sensor
        self.R = R
        self.L = L
        self.nparticles = nparticles
        self._animate = animate

        # self.dim = sensor.map.dim
        self._history = []
        self.x = ()
        self.weight = ()
        self.w0 = 0.05
        self._x0 = x0

        # create a private random number stream if required
        self._random = np.random.default_rng(seed)
        self._seed = seed

        self._keep_history = history  #  keep history
        self._htuple = namedtuple("PFlog", "t odo xest std weights")

        if workspace is not None:
            self._dim = smb.expand_dims(workspace)
        else:
            self._dim = sensor.map.workspace

        self._workspace = self.robot.workspace
        # self._init()

    def __str__(self):
        # ParticleFilter.char Convert to string
        #
        # PF.char() is a string representing the state of the ParticleFilter
        # object in human-readable form.
        #
        # See also ParticleFilter.display.

        def indent(s, n=2):
            spaces = " " * n
            return s.replace("\n", "\n" + spaces)

        s = f"ParticleFilter object: {self.nparticles} particles"
        s += "\nR:  " + smb.array2str(self.R)
        s += "\nL:  " + smb.array2str(self.L)
        if self.robot is not None:
            s += indent("\nrobot: " + str(self.robot))

        if self.sensor is not None:
            s += indent("\nsensor: " + str(self.sensor))
        return s

    @property
    def robot(self):
        """
        Get robot object

        :return: robot used in simulation
        :rtype: :class:`VehicleBase` subclass
        """
        return self._robot

    @property
    def sensor(self):
        """
        Get sensor object

        :return: sensor used in simulation
        :rtype: :class:`SensorBase` subclass
        """
        return self._sensor

    @property
    def map(self):
        """
        Get map object

        :return: map used in simulation
        :rtype: :class:`LandmarkMap` subclass
        """
        return self._map

    @property
    def verbose(self):
        """
        Get verbosity state

        :return: verbosity
        :rtype: bool
        """
        return self._verbose

    @property
    def history(self):
        """
        Get EKF simulation history

        :return: simulation history
        :rtype: list of namedtuples

        At each simulation timestep a namedtuple of is appended to the history
        list.  It contains, for that time step, estimated state and covariance,
        and sensor observation.

        :seealso: :meth:`get_t` :meth:`get_xy` :meth:`get_std`
            :meth:`get_Pnorm`
        """
        return self._history

    @property
    def workspace(self):
        """
        Size of robot workspace

        :return: workspace bounds [xmin, xmax, ymin, ymax]
        :rtype: ndarray(4)

        Returns the bounds of the workspace as specified by constructor
        option ``workspace``
        """
        return self._workspace

    @property
    def random(self):
        """
        Get private random number generator

        :return: NumPy random number generator
        :rtype: :class:`numpy.random.Generator`

        Has methods including:

        - ``integers(low, high, size, endpoint)``
        - ``random(size)``
        - ``uniform``
        - ``normal(mean, std, size)``
        - ``multivariate_normal(mean, covar, size)``

        The generator is initialized with the seed provided at constructor
        time every time ``init`` is called.

        """
        return self._random

    def _init(self, x0=None, animate=False, ax=None):
        # ParticleFilter.init Initialize the particle filter
        #
        # PF.init() initializes the particle distribution and clears the
        # history.
        #
        # Notes::
        # - If initial particle states were given to the constructor the states are
        #   set to this value, else a random distribution over the map is used.
        # - Invoked by the run() method.

        self.robot.init()
        self.sensor.init()

        # clear the history
        self._history = []

        # create a new private random number generator
        if self._seed is not None:
            self._random = np.random.default_rng(self._seed)

        self._t = 0

        # initialize particles
        if x0 is None:
            x0 = self._x0
        if x0 is None:
            # create initial particle distribution as uniformly randomly distributed
            # over the map workspace and heading angles
            x = self.random.uniform(
                self.workspace[0], self.workspace[1], size=(self.nparticles,)
            )
            y = self.random.uniform(
                self.workspace[2], self.workspace[3], size=(self.nparticles,)
            )
            t = self.random.uniform(-np.pi, np.pi, size=(self.nparticles,))
            self.x = np.c_[x, y, t]

        if animate:
            # display the initial particles
            (self.h,) = ax.plot(
                self.x[:, 0],
                self.x[:, 1],
                "go",
                zorder=0,
                markersize=3,
                markeredgecolor="none",
                alpha=0.3,
                label="particle",
            )

        self.weight = np.ones((self.nparticles,))

    def run(self, T=10, x0=None):
        """
        Run the particle filter simulation

        :param T: maximum simulation time in seconds
        :type T: float
        :param x0: Initial state, defaults to value given to Vehicle constructor
        :type x0: array_like(3) or array_like(2)

        Simulates the motion of a vehicle (under the control of a driving agent)
        and the particle-filter estimator.  The steps are:

        - initialize the filter, vehicle and vehicle driver agent, sensor
        - for each time step:

            - step the vehicle and its driver agent, obtain odometry
            - take a sensor reading
            - execute the EKF
            - save information as a namedtuple to the history list for later display

        :seealso: :meth:`history` :meth:`landmark` :meth:`landmarks`
            :meth:`get_xy` :meth:`get_t` :meth:`get_std`
            :meth:`plot_xy`
        """

        self._init(x0=x0)

        # anim = Animate(opt.movie)

        # display the initial particles
        ax = smb.axes_logic(None, 2)
        if self._animate:
            (self.h,) = ax.plot(
                self.x[:, 0],
                self.x[:, 1],
                "go",
                zorder=0,
                markersize=3,
                markeredgecolor="none",
                alpha=0.3,
                label="particle",
            )
        # set(self.h, 'Tag', 'particles')

        # self.robot.plot()

        # iterate over time
        import time

        for i in range(round(T / self.robot.dt)):
            self._step()
            # time.sleep(0.2)
            plt.pause(0.2)
            # plt.draw()
            # anim.add()
        # anim.close()

    def run_animation(self, T=10, x0=None, format=None, file=None):
        """
        Run the particle filter simulation

        :param T: maximum simulation time in seconds
        :type T: float
        :param x0: Initial state, defaults to value given to Vehicle constructor
        :type x0: array_like(3) or array_like(2)
        :param format: Output format
        :type format: str, optional
        :param file: File name
        :type file: str, optional
        :return: Matplotlib animation object
        :rtype: :meth:`matplotlib.animation.FuncAnimation`

        Simulates the motion of a vehicle (under the control of a driving agent)
        and the particle-filter estimator and returns an animation
        in various formats::

            ``format``    ``file``   description
            ============  =========  ============================
            ``"html"``    str, None  return HTML5 video
            ``"jshtml"``  str, None  return JS+HTML video
            ``"gif"``     str        return animated GIF
            ``"mp4"``     str        return MP4/H264 video
            ``None``                 return a ``FuncAnimation`` object

        The allowables types for ``file`` are given in the second column.  A str
        value is the file name.  If ``None`` is an option then return the video as a string.

        For the last case, a reference to the animation object must be held if used for
        animation in a Jupyter cell::

            anim = robot.run_animation(T=20)

        The steps are:

        - initialize the filter, vehicle and vehicle driver agent, sensor
        - for each time step:

            - step the vehicle and its driver agent, obtain odometry
            - take a sensor reading
            - execute the EKF
            - save information as a namedtuple to the history list for later display

        :seealso: :meth:`history` :meth:`landmark` :meth:`landmarks`
            :meth:`get_xy` :meth:`get_t` :meth:`get_std`
            :meth:`plot_xy`
        """

        fig, ax = plt.subplots()

        nframes = round(T / self.robot.dt)
        anim = animation.FuncAnimation(
            fig=fig,
            # func=lambda i: self._step(animate=True, pause=False),
            # init_func=lambda: self._init(animate=True),
            func=lambda i: self._step(),
            init_func=lambda: self._init(ax=ax, animate=True),
            frames=nframes,
            interval=self.robot.dt * 1000,
            blit=False,
            repeat=False,
        )
        # anim._interval = self.dt*1000/2
        # anim._repeat = True
        ret = None
        if format == "html":
            ret = anim.to_html5_video()  # convert to embeddable HTML5 animation
        elif format == "jshtml":
            ret = anim.to_jshtml()  # convert to embeddable Javascript/HTML animation
        elif format == "gif":
            anim.save(
                file, writer=animation.PillowWriter(fps=1 / self.dt)
            )  # convert to GIF
            ret = None
        elif format == "mp4":
            anim.save(
                file, writer=animation.FFMpegWriter(fps=1 / self.dt)
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
        # self._init(x0=x0)

        # # anim = Animate(opt.movie)

        # # display the initial particles
        # ax = smb.axes_logic(None, 2)
        # if self._animate:
        #     (self.h,) = ax.plot(
        #         self.x[:, 0],
        #         self.x[:, 1],
        #         "go",
        #         zorder=0,
        #         markersize=3,
        #         markeredgecolor="none",
        #         alpha=0.3,
        #         label="particle",
        #     )
        # # set(self.h, 'Tag', 'particles')

        # # self.robot.plot()

        # # iterate over time
        # import time

        # for i in range(round(T / self.robot.dt)):
        #     self._step()
        #     # time.sleep(0.2)
        #     plt.pause(0.2)
        #     # plt.draw()
        #     # anim.add()
        # # anim.close()

    def _step(self):

        # fprintf('---- step\n')
        odo = self.robot.step(animate=self._animate)  # move the robot

        # update the particles based on odometry
        self._predict(odo)

        # get a sensor reading
        z, lm_id = self.sensor.reading()

        if z is not None:
            self._observe(z, lm_id)
            # fprintf(' observe beacon #d\n', lm_id)

            self._select()

        # our estimate is simply the mean of the particles
        x_est = self.x.mean(axis=0)
        std_est = self.x.std(axis=0)

        # std is more complex for angles, need to account for 2pi wrap
        std_est[2] = np.sqrt(np.sum(smb.angdiff(self.x[:, 2], x_est[2]) ** 2)) / (
            self.nparticles - 1
        )

        # display the updated particles
        # set(self.h, 'Xdata', self.x(:,1), 'Ydata', self.x(:,2), 'Zdata', self.x(:,3))

        if self._animate:
            self.h.set_xdata(self.x[:, 0])
            self.h.set_ydata(self.x[:, 1])

        # if ~isempty(self.anim)
        #     self.anim.add()

        if self._keep_history:
            hist = self._htuple(
                self.robot._t, odo.copy(), x_est, std_est, self.weight.copy()
            )
            self._history.append(hist)

    def plot_pdf(self):
        """
        Plot particle PDF

        Displays a discrete PDF of vehicle position.  Creates a 3D plot where
        the x- and y-axes are the estimated vehicle position and the z-axis is
        the particle weight.  Each particle is represented by a a vertical line
        segment of height equal to particle weight.
        """

        ax = smb.plotvol3()
        for (x, y, t), weight in zip(self.x, self.weight):
            # ax.plot([x, x], [y, y], [0, weight], 'r')
            ax.plot([x, x], [y, y], [0, weight], "skyblue", linewidth=3)
            ax.plot(x, y, weight, "k.", markersize=6)

        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim()
        ax.set_zlabel("particle weight")
        ax.view_init(29, 59)

    def _predict(self, odo):
        # step 2
        # update the particle state based on odometry and a random perturbation

        # Straightforward code:
        #
        # for i=1:self.nparticles
        #    x = self.robot.f( self.x(i,:), odo)' + sqrt(self.R)*self.randn[2,0]
        #    x[2] = angdiff(x[2])
        #    self.x(i,:) = x
        #
        # Vectorized code:

        self.x = self.robot.f(self.x, odo) + self.random.multivariate_normal(
            (0, 0, 0), self.R, size=self.nparticles
        )
        self.x[:, 2] = smb.angdiff(self.x[:, 2])

    def _observe(self, z, lm_id):
        # step 3
        # predict observation and score the particles

        # Straightforward code:
        #
        # for p = 1:self.nparticles
        #    # what do we expect observation to be for this particle?
        #    # use the sensor model h(.)
        #    z_pred = self.sensor.h( self.x(p,:), lm_id)
        #
        #    # how different is it
        #    innov[0] = z[0] - z_pred[0]
        #    innov[1] = angdiff(z[1], z_pred[1])
        #
        #    # get likelihood (new importance). Assume Gaussian but any PDF works!
        #    # If predicted obs is very different from actual obs this score will be low
        #    #  ie. this particle is not very good at predicting the observation.
        #    # A lower score means it is less likely to be selected for the next generation...
        #    # The weight is never zero.
        #    self.weight(p) = exp(-0.5*innov'*inv(self.L)*innov) + 0.05
        # end
        #
        # Vectorized code:

        invL = np.linalg.inv(self.L)
        z_pred = self.sensor.h(self.x, lm_id)
        z_pred[:, 0] = z[0] - z_pred[:, 0]
        z_pred[:, 1] = smb.angdiff(z[1], z_pred[:, 1])

        LL = -0.5 * np.r_[invL[0, 0], invL[1, 1], 2 * invL[0, 1]]
        e = (
            np.c_[z_pred[:, 0] ** 2, z_pred[:, 1] ** 2, z_pred[:, 0] * z_pred[:, 1]]
            @ LL
        )
        self.weight = np.exp(e) + self.w0

    def _select(self):
        # step 4
        # select particles based on their weights
        #
        # particles with large weights will occupy a greater percentage of the
        # y axis in a cummulative plot
        cdf = np.cumsum(self.weight) / self.weight.sum()

        # so randomly (uniform) choosing y values is more likely to correspond to
        # better particles...
        iselect = self.random.uniform(0, 1, size=(self.nparticles,))

        # find the particle that corresponds to each y value (just a look up)
        interpfun = sp.interpolate.interp1d(
            cdf,
            np.arange(self.nparticles),
            assume_sorted=True,
            kind="nearest",
            fill_value="extrapolate",
        )
        inextgen = interpfun(iselect).astype(int)

        # copy selected particles for next generation..
        self.x = self.x[inextgen, :]

    def get_t(self):
        """
        Get time from simulation

        :return: simulation time vector
        :rtype: ndarray(n)

        Return simulation time vector, starts at zero.  The timestep is an
        attribute of the ``robot`` object.
        """
        return np.array([h.t for h in self._history])

    def get_xyt(self):
        r"""
        Get estimated vehicle trajectory

        :return: vehicle trajectory where each row is configuration :math:`(x, y, \theta)`
        :rtype: ndarray(n,3)

        :seealso: :meth:`plot_xy` :meth:`run` :meth:`history`
        """
        return np.array([h.xest[:2] for h in self._history])

    def get_std(self):
        r"""
        Get standard deviation of particles

        :return: standard deviation of vehicle position estimate
        :rtype: ndarray(n,2)

        Return the standard deviation :math:`(\sigma_x, \sigma_y)` of the
        particle cloud at each time step.

        :seealso: :meth:`get_xyt`
        """
        return np.array([h.std for h in self._history])

    def plot_xy(self, block=None, **kwargs):
        r"""
        Plot estimated vehicle position

        :param args: position arguments passed to :meth:`~matplotlib.axes.Axes.plot`
        :param kwargs: keywords arguments passed to :meth:`~matplotlib.axes.Axes.plot`
        :param block: hold plot until figure is closed, defaults to None
        :type block: bool, optional

        Plot the estimated vehicle path in the xy-plane.

        :seealso: :meth:`get_xy`
        """
        xyt = self.get_xyt()
        plt.plot(xyt[:, 0], xyt[:, 1], **kwargs)
        if block is not None:
            plt.show(block=block)


if __name__ == "__main__":
    from roboticstoolbox import *

    map = LandmarkMap(20, workspace=10)
    V = np.diag([0.02, np.deg2rad(0.5)]) ** 2
    robot = Bicycle(covar=V, animation="car", workspace=map)
    robot.control = RandomPath(workspace=map)
    W = np.diag([0.1, np.deg2rad(1)]) ** 2
    sensor = RangeBearingSensor(robot, map, covar=W, plot=True)
    R = np.diag([0.1, 0.1, np.deg2rad(1)]) ** 2
    L = np.diag([0.1, 0.1])
    pf = ParticleFilter(robot, sensor=sensor, R=R, L=L, nparticles=1000, animate=True)
    pf.run(T=10)
