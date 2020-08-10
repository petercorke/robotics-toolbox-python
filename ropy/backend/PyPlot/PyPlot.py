#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import numpy as np
from ropy.backend.Connector import Connector
import matplotlib
import matplotlib.pyplot as plt
import signal
from ropy.backend.PyPlot.RobotPlot import RobotPlot
from ropy.backend.PyPlot.EllipsePlot import EllipsePlot
from spatialmath.base.argcheck import getvector

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.size'] = 1.5
matplotlib.rcParams['ytick.major.size'] = 1.5
matplotlib.rcParams['axes.labelpad'] = 1
plt.rc('grid', linestyle="-", color='#dbdbdb')


class PyPlot(Connector):

    def __init__(self):

        super(PyPlot, self).__init__()
        self.robots = []
        self.ellipses = []

    def launch(self, name=None, limits=None):
        '''
        env = launch() launchs a blank 3D matplotlib figure

        '''

        super().launch()

        self.limits = limits
        if limits is not None:
            self.limits = getvector(limits, 6)

        projection = 'ortho'
        labels = ['X', 'Y', 'Z']

        if name is not None:
            self.fig = plt.figure()
        else:
            self.fig = plt.figure()

        self.fig.subplots_adjust(left=-0.09, bottom=0, top=1, right=0.99)

        # Create a 3D axes
        self.ax = self.fig.add_subplot(
            111, projection='3d', proj_type=projection)
        self.ax.set_facecolor('white')

        self.ax.set_xbound(-0.5, 0.5)
        self.ax.set_ybound(-0.5, 0.5)
        self.ax.set_zbound(0.0, 0.5)

        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        self.ax.set_zlabel(labels[2])

        if limits is not None:
            self.ax.set_xlim3d([limits[0], limits[1]])
            self.ax.set_ylim3d([limits[2], limits[3]])
            self.ax.set_zlim3d([limits[4], limits[5]])

        plt.ion()
        plt.show()

        # Set the signal handler and a 0.1 second plot updater
        signal.signal(signal.SIGALRM, self._plot_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)

    def step(self, dt=50):
        '''
        state = step(args) triggers the external program to make a time step
        of defined time updating the state of the environment as defined by
        the robot's actions.

        The will go through each robot in the list and make them act based on
        their control type (position, velocity, acceleration, or torque). Upon
        acting, the other three of the four control types will be updated in
        the internal state of the robot object. The control type is defined
        by the robot object, and not all robot objects support all control
        types.

        '''

        super().step()

        self._step_robots(dt)

        plt.ioff()
        self._draw_ellipses()
        self._draw_robots()
        self._set_axes_equal()
        plt.ion()

        self._update_robots()

    def reset(self):
        '''
        state = reset() triggers the external program to reset to the
        original state defined by launch

        '''

        super().reset()

    def restart(self):
        '''
        state = restart() triggers the external program to close and relaunch
        to thestate defined by launch

        '''

        super().restart()

    def close(self):
        '''
        close() closes the plot

        '''

        super().close()

        signal.setitimer(signal.ITIMER_REAL, 0)
        plt.close(self.fig)

    #
    #  Methods to interface with the robots created in other environemnts
    #

    def add(
            self, ob, readonly=False, display=True,
            jointaxes=True, eeframe=True, shadow=True, name=True):
        '''
        id = add(robot) adds the robot to the external environment. robot must
        be of an appropriate class. This adds a robot object to a list of
        robots which will act upon the step() method being called.

        '''

        super().add()

        if isinstance(ob, rp.SerialLink) or isinstance(ob, rp.ETS):
            self.robots.append(
                RobotPlot(
                    ob, self.ax, readonly, display,
                    jointaxes, eeframe, shadow, name))
            self.robots[len(self.robots) - 1].draw()

        elif isinstance(ob, EllipsePlot):
            ob.ax = self.ax
            self.ellipses.append(ob)
            self.ellipses[len(self.ellipses) - 1].draw()

        self._set_axes_equal()

    def remove(self):
        '''
        id = remove(robot) removes the robot to the external environment.

        '''

        super().remove()

    def hold(self):           # pragma: no cover
        signal.setitimer(signal.ITIMER_REAL, 0)
        plt.ioff()
        plt.show()

    #
    #  Private methods
    #

    def _step_robots(self, dt):

        for rpl in self.robots:
            robot = rpl.robot

            if rpl.readonly or robot.control_type == 'p':
                pass            # pragma: no cover

            elif robot.control_type == 'v':

                for i in range(robot.n):
                    robot.q[i] += robot.qd[i] * (dt / 1000)

            elif robot.control_type == 'a':
                pass

            else:            # pragma: no cover
                # Should be impossible to reach
                raise ValueError(
                    'Invalid robot.control_type. '
                    'Must be one of \'p\', \'v\', or \'a\'')

    def _update_robots(self):
        pass

    def _draw_robots(self):

        for i in range(len(self.robots)):
            self.robots[i].draw()

    def _draw_ellipses(self):

        for i in range(len(self.ellipses)):
            self.ellipses[i].draw()

    def _plot_handler(self, sig, frame):
        plt.pause(0.001)

    def _set_axes_equal(self):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as
        spheres, cubes as cubes, etc..  This is one possible solution to
        Matplotlib's ax.set_aspect('equal') and ax.axis('equal') not
        working for 3D.

        '''

        if self.limits is not None:
            return

        self.ax.autoscale(enable=True, axis='both', tight=False)

        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
