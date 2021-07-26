"""
Python EKF Planner
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing, change remaining matlab placeholders
TODO: Replace vargin with parameters

Not ready for use yet.
"""
from collections import namedtuple
import numpy as np
from math import pi
from scipy import integrate, randn
from scipy.linalg import sqrtm, block_diag
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt

from spatialmath.base.animate import Animate
from spatialmath import base
from roboticstoolbox.mobile import Vehicle
from roboticstoolbox.mobile.landmarkmap import LandmarkMap
from roboticstoolbox.mobile.sensors import Sensor


class EKF:
    def __init__(self, robot=None,  sensor=None, map=None, 
            P0=None, x_est=None, joseph=True,
            landmarks=None,
            verbose=False, history=True, dim=None):

        if robot is not None:
            if not isinstance(robot, tuple) or len(robot) != 2 \
                or not isinstance(robot[0], Vehicle):
                    raise TypeError('robot must be tuple (vehicle, V_est)')
            self._robot = robot[0]  # reference to the robot vehicle
            self._V_est = robot[1]  # estimate of vehicle state covariance V

        if sensor is not None:
            if not isinstance(sensor, tuple) or len(sensor) != 2 \
                or not isinstance(sensor[0], Sensor):
                    raise TypeError('sensor must be tuple (sensor, W_est)')
            self.sensor = sensor[0]  # reference to the sensor
            self._W_est = sensor[1]  # estimate of sensor covariance W
        else:
            self._sensor = None
            self._W_est = None

        if map is not None and not isinstance(map, LandmarkMap):
            raise TypeError('map must be LandmarkMap instance')
        self._ekf_map = map  # prior map for localization

        self._P0 = P0        #  initial system covariance

        self._x_est = x_est           #  estimated state
        self._landmarks = landmarks           #  ekf_map state
        
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
        self._joseph = joseph          #  flag: use Joseph form to compute p

        self._verbose = verbose

        self._keep_history = history     #  keep history
        self._htuple = namedtuple("EKFlog", "t xest odo P innov S K")

        if dim is None:
            # TODO get world size from landmark map, else
            # TODO unpack this value from len 1,2,4
            pass
        self._dim = dim  # robot workspace dimensions for animation

        # self.robot.init()

        # #  clear the history
        # self._history = []

        if self.V_est is None:
            # perfect vehicle case

            self._est_vehicle = False
            self._x_est = None
            self._P_est = None
        else:
            # noisy odometry case
            if self.V_est.shape != (2, 2):
                raise ValueError('vehicle state covariance V_est must be 2x2')
            self._x_est = self.robot.x
            self._P_est = P0
            self._est_vehicle = True

        if self.W_est is not None:
            if self.W_est.shape != (2, 2):
                raise ValueError('sensor covariance W_est must be 2x2')

        # if np.any(self._sensor):
        #     self._landmarks = None*np.zeros(2, self._sensor.ekf_map.nlandmarks)

        # #  check types for passed objects
        # if np.any(self._map) and not isinstance(self._map, 'LandmarkMap'):
        #     raise ValueError('expecting LandmarkMap object')

        # if np.any(sensor) and not isinstance(sensor, 'Sensor'):
        #     raise ValueError('expecting Sensor object')

        self.init()

    def __str__(self):
        s = f"EKF object: {len(self._x_est)} states"

        def indent(s, n=2):
            spaces = ' ' * n
            return s.replace('\n', '\n' + spaces)

        estimating = []
        if self._est_vehicle is not None:
            estimating.append('vehicle pose')
        if self._est_ekf_map is not None:
            estimating.append('map')
        if len(estimating) > 0:
            s += ', estimating: ' + ", ".join(estimating)
        if self.robot is not None:
            s += indent("\nrobot: "  + str(self.robot))
        if self.V_est is not None:
            s += indent('\nV_est:  ' + np.array2string(self.V_est.ravel(), precision=3))

        if self.sensor is not None:
            s += indent("\nsensor: " + str(self.sensor))
        if self.W_est is not None:
            s += indent('\nW_est:  ' + np.array2string(self.W_est.ravel(), precision=3))

        return s
    
    @property
    def x_est(self):
        return self._x_est
    
    @property
    def P_est(self):
        return self._P_est

    @property
    def P0(self):
        return self._P0
    
    @property
    def landmarks(self):
        return self._landmarks
    
    @property
    def V_est(self):
        return self._V_est

    @property
    def W_est(self):
        return self._W_est
    
    @property
    def robot(self):
        return self._robot
    
    @property
    def sensor(self):
        return self._sensor
    
    @property
    def est_vehicle(self):
        return self._est_vehicle
    
    @property
    def est_ekf_map(self):
        return self._est_ekf_map
    
    @property
    def joseph(self):
        return self._joseph
    
    @property
    def verbose(self):
        return self._verbose
    
    @property
    def p0(self):
        return self._p0
    
    @property
    def ekf_map(self):
        return self._ekf_map
    
    @property
    def history(self):
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
    

    def init(self):
        #EKF.init Reset the filter
        #
        # E.init() resets the filter state and clears landmarks and history.
        self.robot.init()
        if self.sensor is not None:
            self.sensor.init()

        #clear the history
        self._history = []
        
        if self._V_est is None:
            # perfect vehicle case
            self._estVehicle = False
            self._x_est = np.empty((0,))
            self._P_est = np.empty((0, 0))
        else:
            # noisy odometry case
            self._x_est = self.robot.x
            self._P_est = self.P0
            self._estVehicle = True
        
        if self.sensor is not None:
            self._landmarks = np.zeros((2, len(self.sensor.map)), dtype=int)

    def run(self, T, plot=True, movie=np.array([])):
        self.init()
        if plot:
            if self.sensor is not None:
                self.sensor.map.plot()
            elif self._dim is not None:
                if not Iterable(self._dim):
                    d = self._dim
                    plt.axis([-d, d, -d, d])
                elif len(self._dim) == 2:
                    w = self._dim[1]
                    h = self._dim(2)
                    plt.axis([-w, w, -h, h])
                elif len(self._dim) == 4:
                    plt.axis(self._dim)
            else:
                plot = False

            plt.xlabel('X')
            plt.ylabel('Y')

        # anim = Animate(movie)
        for k in range(round(T / self.robot.dt)):
            if plot:
                self.robot.plot()
            self.step()
        #     anim.add()

        # anim.close()


    # TODO: Make the following methods private.
    def step(self, z_pred=None):

        odo = self.robot.step()

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
            P_pred =  Pvv_pred
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

        sensorReading = False
        if self.sensor is not None:
            #  read the sensor
            z, lm_id = self.sensor.reading()

            # test if the sensor has returned a reading at this time interval
            sensorReading = z is not None

        if sensorReading:
            #  here for MBL, MM, SLAM

            # compute the innovation
            z_pred = self.sensor.h(xv_pred, lm_id)
            innov = np.array([
                z[0] - z_pred[0],
                base.angdiff(z[1], z_pred[1])
            ])

            if self._est_ekf_map:
                # the ekf_map is estimated MM or SLAM case
                if self.isseenbefore(lm_id):

                    # get previous estimate of its state
                    jx = self._landmarks[0, lm_id]
                    xf = xm_pred[jx: jx+2]

                    # compute Jacobian for this particular landmark
                    Hx_k = self.sensor.Hp(xv_pred, xf)
                    #  create the Jacobian for all landmarks
                    Hx = np.zeros((2, len(xm_pred)))
                    Hx[:, jx:jx+2] = Hx_k

                    Hw = self.sensor.Hw(xv_pred, xf)

                    if self._est_vehicle:
                        # concatenate Hx for for vehicle and ekf_map
                        Hxv = self.sensor.Hx(xv_pred, xf)
                        Hx = np.block([Hxv, Hx])

                    doUpdatePhase = True
                else:
                    # get the extended state
                    x_pred, P_pred = self.extend_map(P_pred, xv_pred, xm_pred, z, lm_id)
                    doUpdatePhase = False

            else:
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
            x_est = x_pred + K @ innov.T

            if self._est_vehicle:
                #  wrap heading state for a vehicle
                x_est[2] = base.angdiff(x_est[2])

            # update the covariance
            if self._joseph:
                #  we use the Joseph form
                I = np.eye(P_pred.shape[0])
                P_est = (I - K @ Hx) @ P_pred @ (I - K @ Hx).T \
                    + K @ self._W_est @ K.T
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
                K.copy() if K is not None else None
            )
            self._history.append(hist)

    def isseenbefore(self, lm_id):

        # _landmarks[0, id] is the index in state vector
        # _landmarks[1, id] is the occurence count

        if self._landmarks[1, lm_id] > 0:
            if self._verbose:
                print(f"landmark {lm_id} seen {self._landmarks[1, lm_id]} times before, state_idx={self._landmarks[0,lm_id]}")
            self._landmarks[1, lm_id] += 1 # update the count
            s = True
        else:
            s = False
        return s

    def extend_map(self, P, xv, xm, z, lm_id):
            # this is a new landmark, we haven't seen it before
            # estimate position of landmark in the world based on 
            # noisy sensor reading and current vehicle pose

        M = None
        if self._verbose:
            print(f'landmark {lm_id} first sighted')

        # estimate its position based on observation and vehicle state
        xf = self.sensor.g(xv, z)
        
        # append this estimate to the state vector
        if self._est_vehicle:
            x_ext = np.r_[xv, xm, xf]
        else:
            x_ext = np.r_[xm, xf]
        
        # get the Jacobian for the new landmark
        Gz = self.sensor.Gz(xv, z)

        # extend the covariance matrix
        if self._est_vehicle:
            Gx = self.sensor.Gx(xv, z)
            n = len(self._x_est)
            # fmt: off
            M = np.block([
                [np.eye(n), np.zeros((n, 2))    ],
                [Gx,        np.zeros((2, n-3)), Gz]
            ])
            # fmt: on
            P_ext = M @ block_diag(P, self._W_est) @ M.T
        else:
            P_ext = block_diag(P, Gz @ self._W_est @ Gz.T)

        # record the position in the state vector where this 
        # landmark's state starts
        self._landmarks[0, lm_id] = len(xm) # position in map state vector
        self._landmarks[1, lm_id] = 1  # seen once

        if self._verbose:
            print('extend state vector')
        return x_ext, P_ext

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
        if self._est_vehicle:
            xyt = np.array([h.xest[:2] for h in self._history])
        else:
            xyt = None
        return xyt

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

    def plot_ellipse(self, confidence=0.95, N=10, **kwargs):
        """
        %EKF.plot_ellipse Plot vehicle covariance as an ellipse
        %
        % E.plot_ellipse() overlay the current plot with the estimated
        % vehicle position covariance ellipses for 20 points along the
        % path.
        %
        % E.plot_ellipse(LS) as above but pass line style arguments
        % LS to plot_ellipse.
        %
        % Options::
        % 'interval',I        Plot an ellipse every I steps (default 20)
        % 'confidence',C      Confidence interval (default 0.95)
        %
        % See also plot_ellipse.
        """
        nhist = len(self._history)
        
        for k in np.linspace(0, nhist-1, N):
            k = round(k)
            h = self._history[k]
            base.plot_ellipse(h.P[:2, :2], centre=h.xest[:2], confidence=confidence, inverted=True, **kwargs)

    def plot_error(self, bgcolor='r', confidence=0.95, **kwargs):
        """
        %EKF.plot_error Plot vehicle position
        %
        % E.plot_error(OPTIONS) plot the error between actual and estimated vehicle 
        % path (x, y, theta) versus time.  Heading error is wrapped into the range [-pi,pi)
        %
        % Options::
        % 'bound',S    Display the confidence bounds (default 0.95).
        % 'color',C    Display the bounds using color C
        % LS           Use MATLAB linestyle LS for the plots
        %
        % Notes::
        % - The bounds show the instantaneous standard deviation associated
        %   with the state.  Observations tend to decrease the uncertainty
        %   while periods of dead-reckoning increase it.
        % - Set bound to zero to not draw confidence bounds.
        % - Ideally the error should lie "mostly" within the +/-3sigma
        %   bounds.
        %
        % See also EKF.plot_xy, EKF.plot_ellipse, EKF.plot_P.  
        """          

        error = []
        bounds = [] 
        ppf = chi2.ppf(confidence, df=2)

        x_gt = self.robot.x_hist
        for k in range(len(self.history)):
            hk = self.history[k]
            # error is true - estimated
            e = x_gt[k, :] - hk.xest
            e[2] = base.angdiff(e[2])
            error.append(e)

            P = np.diag(hk.P)
            bounds.append(np.sqrt(ppf * P[:3]))

        error = np.array(error)
        bounds = np.array(bounds)
        t = self.get_t()

        fig, axes = plt.subplots(3)
        labels = ["x", "y", r"$\theta$"]

        for k, ax in enumerate(axes):
            if confidence is not None:
                edge = np.array([
                    np.r_[t, t[::-1]],
                    np.r_[bounds[:, k], -bounds[::-1, k]],
                    ])
                polygon = plt.Polygon(edge.T, closed=True, facecolor='r', edgecolor='none', alpha=0.3)
                ax.add_patch(polygon)
            ax.plot(error[:, k], **kwargs);
            ax.grid(True)
            ax.set_ylabel(labels[k] + " error")
            ax.set_xlim(0, t[-1])
        
        # subplot(opt.nplots*100+12)
        # if opt.confidence
        #     edge = [pxy(:,2); -pxy(end:-1:1,2)];
        #     h = patch(t, edge, opt.color);
        #     set(h, 'EdgeColor', 'none', 'FaceAlpha', 0.3);
        # end
        # hold on
        # plot(err(:,2), args{:});
        # hold off
        # grid
        # ylabel('y error')
        
        # subplot(opt.nplots*100+13)
        # if opt.confidence
        #     edge = [pxy(:,3); -pxy(end:-1:1,3)];
        #     h = patch(t, edge, opt.color);
        #     set(h, 'EdgeColor', 'none', 'FaceAlpha', 0.3);
        # end
        # hold on
        # plot(err(:,3), args{:});
        # hold off
        # grid
        # xlabel('Time step')
        # ylabel('\theta error')

    def get_ekf_map(self):
        xy = []
        for ix, n in self._landmarks.T:
            if n == 0:
                #  this landmark never observed
                xy.append((np.nan, np.nan))
                continue
            else:
                #  n is an index into the *landmark* part of the state
                #  vector, we need to offset it to account for the vehicle
                #  state if we are estimating vehicle as well
                if self._est_vehicle:
                    n += 3
                xf = self._x_est[ix: ix+2]
                xy.append(xf)
        return np.array(xy)

    def plot_map(self, confidence = 0.95, **kwargs):
        """        %EKF.plot_map Plot landmarks
        %
        % E.plot_map(OPTIONS) overlay the current plot with the estimated landmark 
        % position (a +-marker) and a covariance ellipses.
        %
        % E.plot_map(LS, OPTIONS) as above but pass line style arguments
        % LS to plot_ellipse.
        %
        % Options::
        % 'confidence',C   Draw ellipse for confidence value C (default 0.95)
        %
        % See also EKF.get_map, EKF.plot_ellipse.

        % TODO:  some option to plot map evolution, layered ellipses
        """
        
        xf = self.get_ekf_map()

        for i in range(xf.shape[0]):
            Pi = self.P_est[i: i+2, i: i+2]
        # % TODO reinstate the interval landmark
            base.plot_ellipse(Pi, centre=xf[i, :], confidence=confidence, inverted=True, **kwargs)
        # plot_ellipse( P * chi2inv_rtb(opt.confidence, 2), xf, args{:});
        plt.plot(xf[:, 0], xf[:, 1], 'ro', markerfacecolor='none')

    def get_P(self, k=None):
        if k is not None:
            return np.sqrt(np.linalg.det(self._history[k].P))
        else:
            p = [np.sqrt(np.linalg.det(h.P)) for h in self._history]
            return np.array(p)

    def get_t(self):
        return np.array([h.t for h in self._history])

    def show_P(self, P):

        z = np.log10(abs(P))
        mn = min(z[~np.isinf(z)])
        z[np.isinf(z)] = mn

        plt.imshow(z, cmap='Reds')
        plt.colorbar(label='log covariance')
        plt.xlabel('State')
        plt.ylabel('State')



if __name__ == "__main__":

    from roboticstoolbox import Bicycle

    ### RVC2: Chapter 6


    ##  6.1 Dead reckoning

    ## 6.1.1 Modeling the vehicle
    V = np.diag(np.r_[0.02, 0.5*pi/180] ** 2);

    veh = Bicycle(covar=V)

    odo = veh.step(1, 0.3)

    print(veh.x)

    veh.f([0, 0, 0], odo)

    # veh.add_driver( RandomPath(10) )

    # veh.run()

    ###  6.1.2  Estimating pose
    # veh.Fx( [0,0,0], [0.5, 0.1] )

    P0 = np.diag(np.r_[0.005, 0.005, 0.001]**2);

    ekf = EKF(veh, V, P0);

    ekf.run(1000);

    veh.plot_xy()

    ekf.plot_xy('r')

    P700 = ekf.history(700).P

    sqrt(P700(1,1))

    ekf.plot_ellipse('g')

    # %%  6.2 Map-based localization
    # randinit
    # map = LandmarkMap(20, 10)

    # map.plot()

    # W = diag([0.1, 1*pi/180].^2);

    # sensor = RangeBearingSensor(veh, map, 'covar', W)


    # [z,i] = sensor.reading()

    # map.landmark(17)

    # randinit
    # map = LandmarkMap(20);
    # veh = Bicycle('covar', V);
    # veh.add_driver( RandomPath(map.dim) );
    # sensor = RangeBearingSensor(veh, map, 'covar', W, 'angle', ...
    # [-pi/2 pi/2], 'range', 4, 'animate');
    # ekf = EKF(veh, V, P0, sensor, W, map);

    # ekf.run(1000);
    # map.plot()
    # veh.plot_xy();
    # ekf.plot_xy('r');
    # ekf.plot_ellipse('k')

    # %%  6.3  Creating a map
    # randinit
    # map = LandmarkMap(20);
    # veh = Bicycle(); % error free vehicle
    # veh.add_driver( RandomPath(map.dim) );
    # W = diag([0.1, 1*pi/180].^2);
    # sensor = RangeBearingSensor(veh, map, 'covar', W);
    # ekf = EKF(veh, [], [], sensor, W, []);

    # ekf.run(1000);

    # map.plot();
    # ekf.plot_map('g');
    # veh.plot_xy('b');


    # ekf.landmarks(:,6)

    # ekf.x_est(19:20)'

    # ekf.P_est(19:20,19:20)

    # %%  6.4  EKF SLAM
    # randinit
    # P0 = diag([.01, .01, 0.005].^2);
    # map = LandmarkMap(20);
    # veh = Bicycle('covar', V);
    # veh.add_driver( RandomPath(map.dim) );
    # sensor = RangeBearingSensor(veh, map, 'covar', W);
    # ekf = EKF(veh, V, P0, sensor, W, []);

    # ekf.run(1000);

    # map.plot();
    # ekf.plot_map('g');
    # ekf.plot_xy('r');
    # veh.plot_xy('b');

    # %%  6.6 Pose-graph SLAM
    # syms x_i y_i theta_i x_j y_j theta_j x_m y_m theta_m assume real
    # xi_e = inv( SE2(x_m, y_m, theta_m) ) * inv( SE2(x_i, y_i, theta_i) ) * SE2(x_j, y_j, theta_j);
    # fk = simplify(xi_e.xyt);

    # jacobian ( fk, [x_i y_i theta_i] );
    # Ai = simplify (ans)

    # pg = PoseGraph('pg1.g2o')

    # clf
    # pg.plot()

    # pg.optimize('animate')

    # pg = PoseGraph('killian-small.toro');

    # pg.plot()

    # pg.optimize()

    # %% 6.7  Particle filter
    # randinit
    # map = LandmarkMap(20);
    # W = diag([0.1, 1*pi/180].^2);
    # veh = Bicycle('covar', V);
    # veh.add_driver( RandomPath(10) );

    # V = diag([0.005, 0.5*pi/180].^2);
    # sensor = RangeBearingSensor(veh, map, 'covar', W);

    # Q = diag([0.1, 0.1, 1*pi/180]).^2;

    # L = diag([0.1 0.1]);

    # pf = ParticleFilter(veh, sensor, Q, L, 1000);

    # pf.run(1000);

    # map.plot();
    # veh.plot_xy('b');

    # clf
    # pf.plot_xy('r');

    # clf
    # plot(pf.std(1:100,:))

    # clf
    # pf.plot_pdf()


    # %% 6.8  Application: Scanning laser rangefinder

    # %% Laser odometry
    # pg = PoseGraph('killian.g2o', 'laser');

    # [r, theta] = pg.scan(2580);
    # about r theta

    # clf
    # polar(theta, r)

    # [x,y] = pol2cart (theta, r);
    # plot (x, y, '.')

    # p2580 = pg.scanxy(2580);
    # p2581 = pg.scanxy(2581);
    # about p2580

    # T = icp( p2581, p2580, 'verbose' , 'T0', transl2(0.5, 0), 'distthresh', 3)

    # pg.time(2581)-pg.time(2580)


    # %% Laser-based map building
    # map = pg.scanmap();
    # pg.plot_occgrid(map)