#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import numpy as np
from roboticstoolbox.backends.Connector import Connector
from roboticstoolbox.backends.PyPlot.RobotPlot2 import RobotPlot2
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
import time

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
        self.sim_time = 0

        if not _mpl:    # pragma nocover
            raise ImportError(
                '\n\nYou do not have matplotlib installed, do:\n'
                'pip install matplotlib\n\n')

    def __repr__(self):
        s =  f"PyPlot2D backend, t = {self.sim_time}, scene:"
        for robot in self.robots:
            s += f"\n  {robot.name}"
        return s

    def launch(self, name=None, limits=None, **kwargs):
        '''
        env = launch() launchs a blank 2D matplotlib figure

        '''

        super().launch()

        labels = ['X', 'Y']

        if name is not None:
            self.fig = plt.figure(name)
        else:
            self.fig = plt.figure()

        # Create a 2D axes
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor('white')
        plt.get_current_fig_manager().set_window_title(f"Robotics Toolbox for Python (Figure {self.ax.figure.number})")

        self.ax.set_xbound(-0.5, 0.5)
        self.ax.set_ybound(-0.5, 0.5)

        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])

        self.ax.autoscale(enable=True, axis='both', tight=False)

        if limits is not None:
            self.ax.set_xlim([limits[0], limits[1]])
            self.ax.set_ylim([limits[2], limits[3]])

        self.ax.axis('equal')

        plt.ion()
        plt.show()

        # Set the signal handler and a 0.1 second plot updater
        # signal.signal(signal.SIGALRM, self._plot_handler)
        # signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)

    def step(self, dt=0.05):
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

        # plt.ioff()
        self._draw_ellipses()
        self._draw_robots()
        # plt.ion()

        if _isnotebook():
            plt.draw()
            self.fig.canvas.draw()
            time.sleep(dt)
        else:
            plt.draw()
            plt.pause(dt)

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
            eeframe=True, name=False, **kwargs):
        '''
        id = add(robot) adds the robot to the external environment. robot must
        be of an appropriate class. This adds a robot object to a list of
        robots which will act upon the step() method being called.

        '''

        super().add()

        if isinstance(ob, rp.ERobot2):
            self.robots.append(
                RobotPlot2(
                    ob, self, readonly, display,
                    eeframe, name))
            self.robots[len(self.robots) - 1].draw()


        elif isinstance(ob, EllipsePlot):
            ob.ax = self.ax
            self.ellipses.append(ob)
            self.ellipses[len(self.ellipses) - 1].draw2()

    def remove(self):
        '''
        id = remove(robot) removes the robot to the external environment.

        '''

        super().remove() # ???

    def hold(self):           # pragma: no cover
        '''
        hold() keeps the plot open i.e. stops the plot from closing once
        the main script has finished.

        '''

        # signal.setitimer(signal.ITIMER_REAL, 0)
        plt.ioff()

        # keep stepping the environment while figure is open
        while True:
            if not plt.fignum_exists(self.fig.number):
                break
            self.step()

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
            self.robots[i].draw()

    def _draw_ellipses(self):

        for i in range(len(self.ellipses)):
            self.ellipses[i].draw2()

    # def _plot_handler(self, sig, frame):
    #     plt.pause(0.001)

    def _add_teach_panel(self, robot, q):
        """
        Add a teach panel

        :param robot: Robot being taught
        :type robot: ERobot class
        :param q: inital joint angles in radians
        :type q: array_like(n)
        """
        fig = self.fig

        # Add text to the plots
        def text_trans(text, q):  # pragma: no cover
            # update displayed robot pose value
            T = robot.fkine(q, end=robot.ee_links[0])
            t = np.round(T.t, 3)
            r = np.round(T.theta(), 3)
            text[0].set_text("x: {0}".format(t[0]))
            text[1].set_text("y: {0}".format(t[1]))
            text[2].set_text("yaw: {0}".format(r))

        # Update the self state in mpl and the text
        def update(val, text, robot):  # pragma: no cover
            for j in range(robot.n):
                if robot.isrevolute(j):
                    robot.q[j] = np.radians(self.sjoint[j].val)
                else:
                    robot.q[j] = self.sjoint[j].val
            text_trans(text, robot.q)

        fig.subplots_adjust(left=0.38)
        text = []

        x1 = 0.04
        x2 = 0.22
        yh = 0.04
        ym = 0.5 - (robot.n * yh) / 2 + 0.17/2

        self.axjoint = []
        self.sjoint = []

        qlim = robot.todegrees(robot.qlim)

        # Set the pose text
        # if multiple EE, display only the first one
        T = robot.fkine(q, end=robot.ee_links[0])
        t = np.round(T.t, 3)
        r = np.round(T.theta(), 3)

        # TODO maybe put EE name in here, possible issue with DH robot
        # TODO maybe display pose of all EEs, layout hassles though

        if robot.nbranches == 0:
            header = "End-effector Pose"
        else:
            header = "End-effector #0 Pose"
        fig.text(
            0.02,  1 - ym + 0.25, header,
            fontsize=9, weight="bold", color="#4f4f4f")
        text.append(fig.text(
            0.03, 1 - ym + 0.20, "x: {0}".format(t[0]),
            fontsize=9, color="#2b2b2b"))
        text.append(fig.text(
            0.03, 1 - ym + 0.16, "y: {0}".format(t[1]),
            fontsize=9, color="#2b2b2b"))
        text.append(fig.text(
            0.15, 1 - ym + 0.20, "yaw: {0}".format(r),
            fontsize=9, color="#2b2b2b"))
        fig.text(
            0.02,  1 - ym + 0.06, "Joint angles",
            fontsize=9, weight="bold", color="#4f4f4f")

        for j in range(robot.n):
            # for each joint
            ymin = (1 - ym) - j * yh
            self.axjoint.append(
                fig.add_axes([x1, ymin, x2, 0.03], facecolor='#dbdbdb'))

            if robot.isrevolute(j):
                slider = Slider(
                    self.axjoint[j], 'q' + str(j),
                    qlim[0, j], qlim[1, j], np.degrees(q[j]), "% .1f°")
            else:
                slider = Slider(
                    self.axjoint[j], 'q' + str(j),
                    qlim[0, j], qlim[1, j], q[j], "% .1f")

            slider.on_changed(lambda x: update(x, text, robot))
            self.sjoint.append(slider)
        robot.q = q
        self.step()

def _isnotebook():
    """
    Determine if code is being run from a Jupyter notebook

    ``_isnotebook`` is True if running Jupyter notebook, else False

    :references:

        - https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-
        is-executed-in-the-ipython-notebook/39662359#39662359
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter