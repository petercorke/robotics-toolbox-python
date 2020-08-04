#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import numpy as np
import spatialmath as sm
from ropy.connect.Connector import Connector
import matplotlib
import matplotlib.pyplot as plt
import signal

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.size'] = 1.5
matplotlib.rcParams['ytick.major.size'] = 1.5
matplotlib.rcParams['axes.labelpad'] = 1
plt.rc('grid', linestyle="-", color='#dbdbdb')


class RobotPlot(object):

    def __init__(self, robot, ax):

        super(RobotPlot, self).__init__()

        self.robot = robot
        self.ax = ax

        # Line plot of robot links
        self.links = None

        # Z-axis Coordinate frame (quiver) of joints
        self.joints = []

        # Text of the robots name
        self.name = None

        # Shadow of the the line plot on the x-y axis
        self.sh_links = None

        # Coordinate frame of the ee (three quivers)
        self.ee_axes = []

        # Robot has been drawn
        self.drawn = False

    def draw(self):
        if not self.drawn:
            self.init()

        # plot.set_xdata(loc[0, :])
        # plot.set_ydata(loc[1, :])
        # plot.set_3d_properties(loc[2, :])

    def init(self):

        self.drawn = True

        # Joint and ee poses
        T = self.robot.allfkine()
        Te = self.robot.fkine()
        Tb = self.robot.base

        # Joint and ee position matrix
        loc = np.zeros([3, self.robot.n + 2])
        loc[:, 0] = Tb.t
        loc[:, self.robot.n + 1] = Te.t

        # Joint axes position matrix
        joints = np.zeros((3, self.robot.n))

        # Axes arrow transforms
        Tjx = sm.SE3.Tx(0.06)
        Tjy = sm.SE3.Ty(0.06)
        Tjz = sm.SE3.Tz(0.06)

        # ee axes arrows
        Tex = Te * Tjx
        Tey = Te * Tjy
        Tez = Te * Tjz

        # Joint axes arrow calcs
        for i in range(self.robot.n):
            loc[:, i + 1] = T[i].t
            Tji = T[i] * Tjz
            joints[:, i] = Tji.t

        # Plot robot name
        self.name = self.ax.text(
            0.05, 0, 0.05, self.robot.name, (Tb.t[0], Tb.t[1], 0))

        # Plot ee coordinate frame
        self.ee_axes.append(
            self._plot_quiver(loc[:, self.robot.n + 1], Tex.t, '#EE9494', 2))
        self.ee_axes.append(
            self._plot_quiver(loc[:, self.robot.n + 1], Tey.t, '#93E7B0', 2))
        self.ee_axes.append(
            self._plot_quiver(loc[:, self.robot.n + 1], Tez.t, '#54AEFF', 2))

        # Plot joint z coordinates
        for i in range(self.robot.n):
            self.joints.append(
                self._plot_quiver(loc[:, i+1], joints[:, i], '#8FC1E2', 2))

        # Plot the robot links
        self.links = self.ax.plot(
            loc[0, :], loc[1, :], loc[2, :], linewidth=5, color='#E16F6D')

        # Plot the shadow of the robot links
        self.sh_links = self.ax.plot(
            loc[0, :], loc[1, :], zs=0, zdir='z', linewidth=3, color='#464646')

    def _plot_quiver(self, p0, p1, col, width):
        qv = self.ax.quiver(
            p0[0], p0[1], p0[2],
            p1[0] - p0[0],
            p1[1] - p0[1],
            p1[2] - p0[2],
            linewidth=width,
            color=col
        )

        return qv


class PyPlot(Connector):

    def __init__(self):

        super(PyPlot, self).__init__()
        self.robots = []

    def launch(self):
        '''
        env = launch() launchs a blank 3D matplotlib figure

        '''

        super().launch()

        projection = 'ortho'
        labels = ['X', 'Y', 'Z']

        self.fig = plt.figure()
        self.fig.subplots_adjust(left=-0.09, bottom=0, top=1, right=0.99)
        # fig = plt.gcf()

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

        plt.ion()
        plt.show()

        # Set the signal handler and a 0.1 second plot updater
        signal.signal(signal.SIGALRM, self._plot_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)

    def step(self):
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

    def add(self, ob):
        '''
        id = add(robot) adds the robot to the external environment. robot must
        be of an appropriate class. This adds a robot object to a list of
        robots which will act upon the step() method being called.

        '''

        super().add()

        if isinstance(ob, rp.SerialLink):

            self.robots.append(RobotPlot(ob, self.ax))

            self.robots[0].draw()

            # self._draw_robot(self.robots[0])

        self._set_axes_equal()

    def remove(self):
        '''
        id = remove(robot) removes the robot to the external environment.

        '''

        super().remove()

    def hold(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        plt.ioff()
        plt.show()

    #
    #  Provate methods
    #

    def _draw_robot(self, robot_ob):
        pass

    def _plot_handler(self, sig, frame):
        plt.pause(0.001)

    def _set_axes_equal(self):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as
        spheres, cubes as cubes, etc..  This is one possible solution to
        Matplotlib's ax.set_aspect('equal') and ax.axis('equal') not
        working for 3D.

        '''

        self.ax.autoscale(enable=True, axis='both', tight=False)

        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        # z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([0.0, 2 * plot_radius])



