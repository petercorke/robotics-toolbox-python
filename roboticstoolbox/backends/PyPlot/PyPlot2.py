#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import numpy as np
from roboticstoolbox.backends.Connector import Connector
from roboticstoolbox.backends.PyPlot.RobotPlot2 import RobotPlot2
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot

_mpl = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.style.use('ggplot')
    matplotlib.rcParams['font.size'] = 7
    matplotlib.rcParams['lines.linewidth'] = 0.5
    matplotlib.rcParams['xtick.major.size'] = 1.5
    matplotlib.rcParams['ytick.major.size'] = 1.5
    matplotlib.rcParams['axes.labelpad'] = 1
    plt.rc('grid', linestyle="-", color='#dbdbdb')
    _mpl = True
except ImportError:    # pragma nocover
    pass


class PyPlot2(Connector):

    def __init__(self):

        super(PyPlot2, self).__init__()
        self.robots = []
        self.ellipses = []

        if not _mpl:    # pragma nocover
            raise ImportError(
                '\n\nYou do not have matplotlib installed, do:\n'
                'pip install matplotlib\n\n')

    def launch(self, name=None, limits=None):
        '''
        env = launch() launchs a blank 2D matplotlib figure

        '''

        super().launch()

        labels = ['X', 'Y']

        if name is not None:
            self.fig = plt.figure(name)
        else:
            self.fig = plt.figure()

        # Create a 3D axes
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor('white')

        self.ax.set_xbound(-0.5, 0.5)
        self.ax.set_ybound(-0.5, 0.5)

        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])

        self.ax.autoscale(enable=True, axis='both', tight=False)
        self.ax.axis('equal')

        if limits is not None:
            self.ax.set_xlim([limits[0], limits[1]])
            self.ax.set_ylim([limits[2], limits[3]])

        plt.ion()
        plt.show()

        # Set the signal handler and a 0.1 second plot updater
        # signal.signal(signal.SIGALRM, self._plot_handler)
        # signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)

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

        # signal.setitimer(signal.ITIMER_REAL, 0)
        plt.close(self.fig)

    #
    #  Methods to interface with the robots created in other environemnts
    #

    def add(
            self, ob, readonly=False, display=True,
            eeframe=True, name=False):
        '''
        id = add(robot) adds the robot to the external environment. robot must
        be of an appropriate class. This adds a robot object to a list of
        robots which will act upon the step() method being called.

        '''

        super().add()

        if isinstance(ob, rp.DHRobot) or isinstance(ob, rp.ERobot):
            self.robots.append(
                RobotPlot2(
                    ob, self.ax, readonly, display,
                    eeframe, name))
            self.robots[len(self.robots) - 1].draw2()

        elif isinstance(ob, EllipsePlot):
            ob.ax = self.ax
            self.ellipses.append(ob)
            self.ellipses[len(self.ellipses) - 1].draw2()

    def remove(self):
        '''
        id = remove(robot) removes the robot to the external environment.

        '''

        super().remove()

    def hold(self):           # pragma: no cover
        # signal.setitimer(signal.ITIMER_REAL, 0)
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

            elif robot.control_type == 'a':  # pragma: no cover
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
            self.robots[i].draw2()

    def _draw_ellipses(self):

        for i in range(len(self.ellipses)):
            self.ellipses[i].draw2()

    # def _plot_handler(self, sig, frame):
    #     plt.pause(0.001)

    def _add_teach_panel(self, robot):
        fig = self.fig

        # Add text to the plots
        def text_trans(text):  # pragma: no cover
            T = robot.fkine()
            t = np.round(T.t, 3)
            r = np.round(T.rpy(), 3)
            text[0].set_text("x: {0}".format(t[0]))
            text[1].set_text("y: {0}".format(t[1]))
            text[2].set_text("yaw: {0}".format(r[2]))

        # Update the self state in mpl and the text
        def update(val, text, robot):  # pragma: no cover
            for i in range(robot.n):
                robot.q[i] = self.sjoint[i].val * np.pi/180

            text_trans(text)

            # Step the environment
            self.step(0)

        fig.subplots_adjust(left=0.38)
        text = []

        x1 = 0.04
        x2 = 0.22
        yh = 0.04
        ym = 0.5 - (robot.n * yh) / 2 + 0.17/2

        self.axjoint = []
        self.sjoint = []

        qlim = np.copy(robot.qlim) * 180/np.pi

        if np.all(qlim == 0):    # pragma nocover
            qlim[0, :] = -180
            qlim[1, :] = 180

        # Set the pose text
        T = robot.fkine()
        t = np.round(T.t, 3)
        r = np.round(T.rpy(), 3)

        fig.text(
            0.02,  1 - ym + 0.25, "End-effector Pose",
            fontsize=9, weight="bold", color="#4f4f4f")
        text.append(fig.text(
            0.03, 1 - ym + 0.20, "x: {0}".format(t[0]),
            fontsize=9, color="#2b2b2b"))
        text.append(fig.text(
            0.03, 1 - ym + 0.16, "y: {0}".format(t[1]),
            fontsize=9, color="#2b2b2b"))
        text.append(fig.text(
            0.15, 1 - ym + 0.20, "yaw: {0}".format(r[0]),
            fontsize=9, color="#2b2b2b"))
        fig.text(
            0.02,  1 - ym + 0.06, "Joint angles",
            fontsize=9, weight="bold", color="#4f4f4f")

        for i in range(robot.n):
            ymin = (1 - ym) - i * yh
            self.axjoint.append(
                fig.add_axes([x1, ymin, x2, 0.03], facecolor='#dbdbdb'))

            self.sjoint.append(
                Slider(
                    self.axjoint[i], 'q' + str(i),
                    qlim[0, i], qlim[1, i], robot.q[i] * 180/np.pi))

            self.sjoint[i].on_changed(lambda x: update(x, text, robot))
