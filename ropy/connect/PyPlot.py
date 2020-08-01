#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from ropy.connect.Connector import Connector
import matplotlib
import matplotlib.pyplot as plt

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
        self.ax.autoscale(enable=True, axis='both')

        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        self.ax.set_zlabel(labels[2])

        plt.ion()
        plt.show()
        plt.pause(0.001)

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
        state = close() triggers the external program to gracefully close

        '''

        super().close()
        plt.close(self.fig)

    #
    #  Methods to interface with the robots created in other environemnts
    #

    def add(self):
        '''
        id = add(robot) adds the robot to the external environment. robot must
        be of an appropriate class. This adds a robot object to a list of
        robots which will act upon the step() method being called.

        '''

        super().add()

    def remove(self):
        '''
        id = remove(robot) removes the robot to the external environment.

        '''

        super().remove()
