#!/usr/bin/env python3

from collections import namedtuple

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from spatialmath import base


"""
#ParticleFilter Particle filter class

Monte-carlo based localisation for estimating vehicle pose based on
odometry and observations of known landmarks.

Methods::
run        run the particle filter
plot_xy    display estimated vehicle path
plot_pdf   display particle distribution

Properties::
 robot        reference to the robot object
 sensor       reference to the sensor object
 history      vector of structs that hold the detailed information from
              each time step
 nparticles   number of particles used
 x            particle states nparticles x 3
 weight       particle weights nparticles x 1
 x_est        mean of the particle population
 std          standard deviation of the particle population
 Q            covariance of noise added to state at each step
 L            covariance of likelihood model
 w0           offset in likelihood model
 dim          maximum xy dimension

Example::

Create a landmark map
   map = PointMap[19]
and a vehicle with odometry covariance and a driver
   W = diag([0.1, 1*pi/180].^2)
   veh = Vehicle(W)
   veh.add_driver( RandomPath[9] )
and create a range bearing sensor
   R = diag([0.005, 0.5*pi/180].^2)
   sensor = RangeBearingSensor(veh, map, R)

For the particle filter we need to define two covariance matrices.  The
first is is the covariance of the random noise added to the particle
states at each iteration to represent uncertainty in configuration.   
   Q = diag([0.1, 0.1, 1*pi/180]).^2
and the covariance of the likelihood function applied to innovation
   L = diag([0.1 0.1])
Now construct the particle filter
   self = ParticleFilter(veh, sensor, Q, L, 1000)
which is configured with 1000 particles.  The particles are initially
uniformly distributed over the 3-dimensional configuration space.
   
We run the simulation for 1000 time steps
   self.run[999]
then plot the map and the true vehicle path
   map.plot()
   veh.plot_xy('b')
and overlay the mean of the particle cloud
   self.plot_xy('r')
We can plot the standard deviation against time
   plot(self.std(1:100,:))
The particles are a sampled approximation to the PDF and we can display
this as
   self.plot_pdf()

Acknowledgement::

Based on code by Paul Newman, Oxford University, 
http://www.robots.ox.ac.uk/~pnewman

Reference::

  Robotics, Vision & Control,
  Peter Corke,
  Springer 2011

See also Vehicle, RandomPath, RangeBearingSensor, PointMap, EKF.
"""


#note this is not coded efficiently but rather to make the ideas clear
#all loops should be vectorized but that gets a little matlab-speak intensive
#and may obliterate the elegance of a particle filter....

#TODO
# x_est should be a weighted mean
# std should be a weighted std (x-mean)' W (x-mean)  ???
#     properties
#         robot
#         sensor
#         nparticles
#         x           # particle states nparticles x 3
#         weight      # particle weights nparticles x 1
#         x_est        # mean of the particle population
#         std         # standard deviation of the particle population
#         Q           # covariance of noise added to state at each step
#         L           # covariance of likelihood model
#         history
#         keephistory
#         dim         # maximum xy dimension

#         h           # graphics handle for particles
#         randstream
#         seed0
#         w0
#         x0          # initial particle distribution
#         anim
#     end # properties

class ParticleFilter:
    
    def __init__(self, robot, sensor, Q, L, nparticles=500, seed=0, x0=None,
    verbose=False, history=True, dim=None):
        #ParticleFilter.ParticleFilter Particle filter constructor
        #
        # PF = ParticleFilter(VEHICLE, SENSOR, Q, L, NP, OPTIONS) is a particle
        # filter that estimates the state of the VEHICLE with a landmark sensor
        # SENSOR.  Q is the covariance of the noise added to the particles
        # at each step (diffusion), L is the covariance used in the
        # sensor likelihood model, and NP is the number of particles.
        #
        # Options::
        # 'verbose'     Be verbose.
        # 'private'     Use private random number stream.
        # 'reset'       Reset random number stream.
        # 'seed',S      Set the initial state of the random number stream.  S must
        #               be a proper random number generator state such as saved in
        #               the seed0 property of an earlier run.
        # 'nohistory'   Don't save history.
        # 'x0'          Initial particle states (Nx3)
        #
        # Notes::
        # - ParticleFilter subclasses Handle, so it is a reference object. 
        # - If initial particle states not given they are set to a uniform
        #   distribution over the map, essentially the kidnapped robot problem
        #   which is quite unrealistic.
        # - Initial particle weights are always set to unity. 
        # - The 'private' option creates a private random number stream for the
        #   methods rand, randn and randi.  If not given the global stream is used.
        #
        #
        # See also Vehicle, Sensor, RangeBearingSensor, PointMap.

        self.robot = robot
        self.sensor = sensor
        self.Q = Q
        self.L = L
        self.nparticles = nparticles

        # self.dim = sensor.map.dim
        self.history = []
        self.x = ()
        self.weight = ()
        self.w0 = 0.05
        self._x0 = x0

        # create a private random number stream if required
        self._random = np.random.default_rng(seed)
        self._seed = seed

        self._keep_history = history     #  keep history
        self._htuple = namedtuple("PFlog", "t odo xest std weights")
        self.dim = 10  # TODO

        self._workspace = self.robot.workspace
        self.init()

    def __str__(self):
        #ParticleFilter.char Convert to string
        #
        # PF.char() is a string representing the state of the ParticleFilter
        # object in human-readable form.
        #
        # See also ParticleFilter.display.

        def indent(s, n=2):
            spaces = ' ' * n
            return s.replace('\n', '\n' + spaces)

        s = f"ParticleFilter object: {self.nparticles} particles"
        s += '\nQ:  ' + base.array2str(self.Q)
        s += '\nL:  ' + base.array2str(self.L)
        if self.robot is not None:
            s += indent("\nrobot: "  + str(self.robot))

        if self.sensor is not None:
            s += indent("\nsensor: " + str(self.sensor))
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
        return self._workspace

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

    def init(self, x0=None):
        #ParticleFilter.init Initialize the particle filter
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

        #clear the history
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
            x = self.random.uniform(self.workspace[0], self.workspace[1], size=(self.nparticles,))
            y = self.random.uniform(self.workspace[2], self.workspace[3], size=(self.nparticles,))
            t = self.random.uniform(-np.pi, np.pi, size=(self.nparticles,))
            self.x = np.c_[x, y, t] 

        self.weight = np.ones((self.nparticles,))


    def run(self, T=10, x0=None):
        #ParticleFilter.run Run the particle filter
        #
        # PF.run(N, OPTIONS) runs the filter for N time steps.
        #
        # Options::
        # 'noplot'    Do not show animation.
        # 'movie',M   Create an animation movie file M
        #
        # Notes::
        # - All previously estimated states and estimation history is
        #   cleared.

        self.init(x0=x0)

        # anim = Animate(opt.movie)

        # display the initial particles
        if self.robot._animation is not None:
            self.h, = plt.plot(self.x[:, 0], self.x[:, 1], 'go', zorder=0, markersize=3, markeredgecolor='none', alpha=0.3, label='particle')
        # set(self.h, 'Tag', 'particles')
        
        # self.robot.plot()

        # iterate over time
        for i in range(round(T / self.robot.dt)):
            self.step()
            # anim.add()
        # anim.close()

    def step(self):
             
        #fprintf('---- step\n')
        odo = self.robot.step()        # move the robot

        # update the particles based on odometry
        self._predict(odo)

        # get a sensor reading
        z, lm_id = self.sensor.reading()         

        if z is not None:
            self._observe(z, lm_id)
            #fprintf(' observe beacon #d\n', lm_id)

            self._select()

        # our estimate is simply the mean of the particles
        x_est = self.x.mean(axis=0)
        std_est = self.x.std(axis=0)

        # std is more complex for angles, need to account for 2pi wrap
        std_est[2] = np.sqrt(np.sum(base.angdiff(self.x[:,2], x_est[2]) ** 2)) / (self.nparticles-1)

        # display the updated particles
        # set(self.h, 'Xdata', self.x(:,1), 'Ydata', self.x(:,2), 'Zdata', self.x(:,3))

        if self.robot._animation is not None:
            self.h.set_xdata(self.x[:, 0])
            self.h.set_ydata(self.x[:, 1])
        
        # if ~isempty(self.anim)
        #     self.anim.add()

        # if self.keephistory:
        #     hist = ()
        #     hist.x_est = self.x
        #     hist.w = self.weight
        #     self.history = [self.history hist]

        if self._keep_history:
            hist = self._htuple(
                self.robot._t,
                odo.copy(),
                x_est,
                std_est,
                self.weight.copy()
            )
            self._history.append(hist)

    def plot_pdf(self):
        #ParticleFilter.plot_pdf Plot particles as a PDF
        #
        # PF.plot_pdf() plots a sparse PDF as a series of vertical line
        # segments of height equal to particle weight.
        ax = base.plotvol3()
        for (x, y, t), weight in zip(self.x, self.weight):
            # ax.plot([x, x], [y, y], [0, weight], 'r')
            ax.plot([x, x], [y, y], [0, weight], 'skyblue', linewidth=3)
            ax.plot(x, y, weight, 'k.', markersize=6)


        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim()
        ax.set_zlabel('particle weight')
        ax.view_init(29, 59)


    # def plot_xy(self, varargin):
    #     #ParticleFilter.plot_xy Plot vehicle position
    #     #
    #     # PF.plot_xy() plots the estimated vehicle path in the xy-plane.
    #     #
    #     # PF.plot_xy(LS) as above but the optional line style arguments
    #     # LS are passed to plot.
    #     plot(self.x_est(:,1), self.x_est(:,2), varargin{:})


        # step 2
        # update the particle state based on odometry and a random perturbation
    def _predict(self, odo):

        # Straightforward code:
        #
        # for i=1:self.nparticles
        #    x = self.robot.f( self.x(i,:), odo)' + sqrt(self.Q)*self.randn[2,0]   
        #    x[2] = angdiff(x[2])
        #    self.x(i,:) = x
        #
        # Vectorized code:

        self.x = self.robot.f(self.x, odo) + \
            self.random.multivariate_normal((0, 0, 0), self.Q, size=self.nparticles)   
        self.x[:, 2] = base.angdiff(self.x[:, 2])

        # step 3
        # predict observation and score the particles
    def _observe(self, z, lm_id):
        
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
        z_pred[:, 1] = base.angdiff(z[1], z_pred[:, 1])

        LL = -0.5 * np.r_[invL[0,0], invL[1,1], 2*invL[0,1]]
        e = np.c_[z_pred[:, 0]**2, z_pred[:, 1]**2, z_pred[:,0] * z_pred[:, 1]] @ LL
        self.weight = np.exp(e) + self.w0  


        # step 4
        # select particles based on their weights
    def _select(self):
            
        # particles with large weights will occupy a greater percentage of the
        # y axis in a cummulative plot
        cdf = np.cumsum(self.weight) / self.weight.sum()

        # so randomly (uniform) choosing y values is more likely to correspond to
        # better particles...
        iselect  = self.random.uniform(0, 1, size=(self.nparticles,))

        # find the particle that corresponds to each y value (just a look up)
        interpfun = sp.interpolate.interp1d(cdf, np.arange(self.nparticles), 
                assume_sorted=True, kind='nearest', fill_value='extrapolate')
        inextgen = interpfun(iselect).astype(np.int)

        # copy selected particles for next generation..
        self.x = self.x[inextgen, :]



    def get_t(self):
        return np.array([h.t for h in self._history])

    def get_xy(self):
        """[summary]

        :return: [description]
        :rtype: [type]

                %EKF.plot_xy Get vehicle position
        %
        % P = E.get_xy() is the estimated vehicle pose trajectory
        % as a matrix (Nx3) where each row is x, y, theta.
        %
        % See also EKF.plot_xy, EKF.plot_error, EKF.plot_ellipse, EKF.plot_P.
        """
        return np.array([h.xest[:2] for h in self._history])

    def get_std(self):

        return np.array([h.std for h in self._history])

    def plot_xy(self, block=False, **kwargs):
        """            %EKF.plot_xy Plot vehicle position
            %
            % E.plot_xy() overlay the current plot with the estimated vehicle path in
            % the xy-plane.
            %
            % E.plot_xy(LS) as above but the optional line style arguments
            % LS are passed to plot.
            %
            % See also EKF.get_xy, EKF.plot_error, EKF.plot_ellipse, EKF.plot_P.


        :param block: [description], defaults to False
        :type block: bool, optional
        """
        xyt = self.get_xy()
        plt.plot(xyt[:, 0], xyt[:, 1], **kwargs)
        # plt.show(block=block)