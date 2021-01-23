#!/usr/bin/env python
"""
@author Jesse Haviland
"""
import time
import roboticstoolbox as rp
import numpy as np
from roboticstoolbox.backends.Connector import Connector

from roboticstoolbox.backends.PyPlot.RobotPlot import RobotPlot
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
from spatialmath.base.argcheck import getvector
# from roboticstoolbox.tools import Ticker

_mpl = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
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


class PyPlot(Connector):
    """
    Graphical backend using matplotlib

    matplotlib is a common and highly portable graphics library for Python,
    but has relatively limited 3D capability.

    Example:

    .. code-block:: python
        :linenos:

        import roboticstoolbox as rtb

        robot = rtb.models.DH.Panda()  # create a robot

        pyplot = rtb.backends.PyPlot()  # create a PyPlot backend
        pyplot.add(robot)              # add the robot to the backend
        robot.q = robot.qz             # set the robot configuration
        pyplot.step()                  # update the backend and graphical view

    .. note::  PyPlot is the default backend, and ``robot.plot(q)`` effectively
        performs lines 7-8 above.

    """

    def __init__(self):

        super(PyPlot, self).__init__()
        self.robots = []
        self.ellipses = []

        if not _mpl:    # pragma nocover
            raise ImportError(
                '\n\nYou do not have matplotlib installed, do:\n'
                'pip install matplotlib\n\n')

    def launch(self, name=None, limits=None):
        """
        Launch a graphical interface

        ```env = launch()``` creates a blank 3D matplotlib figure and returns
        a reference to the backend.
        """

        super().launch()

        self.limits = limits
        if limits is not None:
            self.limits = getvector(limits, 6)

        projection = 'ortho'
        labels = ['X', 'Y', 'Z']

        if name is not None:
            self.fig = plt.figure(name)
        else:
            self.fig = plt.figure('Robotics Toolbox for Python')

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

        # disable the display of value under cursor
        self.ax.format_coord = lambda x, y: ''

        # add time display in top-right corner
        self.timer = plt.figtext(0.85, 0.95, '')

        if _isnotebook():
            plt.ion()
            self.fig.canvas.draw()
        else:
            plt.ion()
            plt.show()

        self.sim_time = 0

        # # Set the signal handler and a 0.1 second plot updater
        # signal.signal(signal.SIGALRM, self._plot_handler)
        # signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)
        # TODO still need to finish this, and get Jupyter animation working

    def step(self, dt=0.05):
        """
        Update the graphical scene

        :param dt: time step in seconds, defaults to 50 (0.05 s)
        :type dt: int, optional

        ``env.step(args)`` triggers an update of the 3D scene in the matplotlib
        window referenced by ``env``.

        .. note::

            - Each robot in the scene is updated based on
              their control type (position, velocity, acceleration, or torque).
            - Upon acting, the other three of the four control types will be
              updated in the internal state of the robot object.
            - The control type is defined by the robot object, and not all
              robot objects support all control types.
            - Execution is blocked for the specified interval

        """

        super().step()

        self._step_robots(dt)

        # plt.ioff()

        self._draw_ellipses()
        self._draw_robots()
        self._set_axes_equal()

        # update time and display it on plot
        if self.sim_time > 0:
            self.timer.set_text(f"t = {self.sim_time:.2f}")
        self.sim_time += dt

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
        """
        Reset the graphical scene

        ``env.reset()`` triggers a reset of the 3D scene in the matplotlib
        window referenced by ``env``. It is restored to the original state
        defined by ``launch()``.
        """
        # TODO what does this actually do for matplotlib??

        super().reset()

    def restart(self):
        """
        Restart the graphics display

        ``env.restart()`` triggers a restart of the matplotlib view referenced
        by ``env``. It is closed and relaunched to the original state defined
        by ``launch()``.

        """
        # TODO what does this actually do for matplotlib??

        super().restart()

    def close(self):
        """
        ``env.close()`` gracefully closes the matplotlib window
        referenced by ``env``.
        """
        # TODO what does this actually do for matplotlib??

        super().close()

        # signal.setitimer(signal.ITIMER_REAL, 0)
        plt.close(self.fig)

    #
    #  Methods to interface with the robots created in other environments
    #

    def add(
            self, ob, readonly=False, display=True,
            jointaxes=True, eeframe=True, shadow=True, name=True):
        """
        Add a robot to the graphical scene

        :param ob: The object to add to the plot (robot or ellipse)
        :type ob: DHRobot or EllipsePlot
        :param readonly: Do not update the state of the object
            (i.e. display not simulate), defaults to False
        :type readonly: bool, optional
        :param display: Display the object, defaults to True
        :type display: bool, optional
        :param jointaxes: Show the joint axes of the robot with arrows,
            defaults to True
        :type jointaxes: bool, optional
        :param eeframe: Show the end-effector frame of the robot,
            defaults to True
        :type eeframe: bool, optional
        :param shadow: Display a shadow of the robot on the x-y gound plane,
            defaults to True
        :type shadow: bool, optional
        :param name: Display the name of the robot, defaults to True
        :type name: bool, optional

        ``id = env.add(robot)`` adds the ``robot`` to the graphical
            environment.

        .. note::

            - ``robot`` must be of an appropriate class.
            - Adds the robot object to a list of robots which will be updated
              when the ``step()`` method is called.

        """

        super().add()

        if isinstance(ob, rp.DHRobot) or isinstance(ob, rp.ERobot):
            self.robots.append(
                RobotPlot(
                    ob, self.ax, readonly, display,
                    jointaxes, eeframe, shadow, name))
            self.robots[len(self.robots) - 1].draw()
            id = len(self.robots)

        elif isinstance(ob, EllipsePlot):
            ob.ax = self.ax
            self.ellipses.append(ob)
            self.ellipses[len(self.ellipses) - 1].draw()
            id = len(self.ellipses)

        plt.draw()
        plt.show(block=False)

        self._set_axes_equal()
        return id

    def remove(self):
        """
        Remove a robot to the graphical scene

        :param id: The id of the robot to remove. Can be either the DHLink or
            GraphicalRobot
        :type id: class:`~roboticstoolbox.robot.DHRobot.DHRobot`,
                  class:`roboticstoolbox.backends.VPython.graphics_robot.GraphicalRobot`
        :param fig_num: The canvas index to delete the robot from, defaults to
             the initial one
        :type fig_num: int, optional
        :raises ValueError: Figure number must be between 0 and total number
            of canvases
        :raises TypeError: Input must be a DHLink or GraphicalRobot

        ``env.remove(robot)`` removes the ``robot`` from the graphical
            environment.
        """
        # TODO should be an id to remove?

        super().remove()

    def hold(self):           # pragma: no cover
        '''
        hold() keeps the plot open i.e. stops the plot from closing once
        the main script has finished.

        '''

        # signal.setitimer(signal.ITIMER_REAL, 0)
        plt.ioff()

        try:
            plt.show()
        except AttributeError:
            pass

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
                    robot.q[i] += robot.qd[i] * (dt)

            elif robot.control_type == 'a':     # pragma: no cover
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

    # def _plot_handler(self, sig, frame):
    #     try:
    #         plt.pause(0.001)
    #     except(AttributeError):
    #         pass

    def _set_axes_equal(self):
        """
        Make axes of 3D plot have equal scale so that spheres appear as
        spheres, cubes as cubes, etc..  This is one possible solution to
        Matplotlib's ax.set_aspect('equal') and ax.axis('equal') not
        working for 3D.

        """

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

    def _add_teach_panel(self, robot):

        if _isnotebook():
            raise RuntimeError('cannot use teach panel under Jupyter')

        fig = self.fig

        # Add text to the plots
        def text_trans(text):  # pragma: no cover
            T = robot.fkine()
            t = np.round(T.t, 3)
            r = np.round(T.rpy('deg'), 3)
            text[0].set_text("x: {0}".format(t[0]))
            text[1].set_text("y: {0}".format(t[1]))
            text[2].set_text("z: {0}".format(t[2]))
            text[3].set_text("r: {0}".format(r[0]))
            text[4].set_text("p: {0}".format(r[1]))
            text[5].set_text("y: {0}".format(r[2]))

        # Update the self state in mpl and the text
        def update(val, text, robot):  # pragma: no cover
            for i in range(robot.n):
                robot.q[i] = self.sjoint[i].val * np.pi/180

            text_trans(text)

            # Step the environment
            self.step(0)

        fig.subplots_adjust(left=0.25)
        text = []

        x1 = 0.04
        x2 = 0.22
        yh = 0.04
        ym = 0.5 - (robot.n * yh) / 2 + 0.17/2

        self.axjoint = []
        self.sjoint = []

        qlim = np.copy(robot.qlim) * 180/np.pi

        if np.all(qlim == 0):     # pragma: no cover
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
            0.03, 1 - ym + 0.12, "z: {0}".format(t[2]),
            fontsize=9, color="#2b2b2b"))
        text.append(fig.text(
            0.15, 1 - ym + 0.20, "r: {0}".format(r[0]),
            fontsize=9, color="#2b2b2b"))
        text.append(fig.text(
            0.15, 1 - ym + 0.16, "p: {0}".format(r[1]),
            fontsize=9, color="#2b2b2b"))
        text.append(fig.text(
            0.15, 1 - ym + 0.12, "y: {0}".format(r[2]),
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
