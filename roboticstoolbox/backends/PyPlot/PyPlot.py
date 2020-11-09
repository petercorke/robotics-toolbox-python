#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import numpy as np
from roboticstoolbox.backends.Connector import Connector
import matplotlib
import matplotlib.pyplot as plt
import signal
from roboticstoolbox.backends.PyPlot.RobotPlot import RobotPlot
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
from spatialmath.base.argcheck import getvector
# from roboticstoolbox.tools import Ticker

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

        # # Set the signal handler and a 0.1 second plot updater
        # signal.signal(signal.SIGALRM, self._plot_handler)
        # signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)
        # TODO still need to finish this, and get Jupyter animation working

    def step(self, dt=50):
        """
        Update the graphical scene

        :param dt: time step in milliseconds, defaults to 50
        :type dt: int, optional
 
        ``env.step(args)`` triggers an update of the 3D scene in the matplotlib
        window referenced by ``env``.

        .. note:: 

            - Each robot in the scene is updated based on
              their control type (position, velocity, acceleration, or torque).
            - Upon acting, the other three of the four control types will be
              updated in the internal state of the robot object. 
            - The control type is defined by the robot object, and not all robot
              objects support all control types.
            - Execution is blocked for the specified interval

        """

        super().step()

        self._step_robots(dt)

        # plt.ioff()
        self._draw_ellipses()
        self._draw_robots()
        self._set_axes_equal()
        # plt.ion()
        plt.draw()
        plt.pause(dt / 1000)

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
        by ``env``. It is closed and relaunched to the original state defined by
        ``launch()``.

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
    #  Methods to interface with the robots created in other environemnts
    #

    def add(
            self, ob, readonly=False, display=True,
            jointaxes=True, eeframe=True, shadow=True, name=True):
        """
        Add a robot to the graphical scene

        :param ob: [description]
        :type ob: [type]
        :param readonly: [description], defaults to False
        :type readonly: bool, optional
        :param display: [description], defaults to True
        :type display: bool, optional
        :param jointaxes: [description], defaults to True
        :type jointaxes: bool, optional
        :param eeframe: [description], defaults to True
        :type eeframe: bool, optional
        :param shadow: [description], defaults to True
        :type shadow: bool, optional
        :param name: [description], defaults to True
        :type name: bool, optional

        ``id = env.add(robot)`` adds the ``robot`` to the graphical environment.

        .. note::

            - ``robot`` must be of an appropriate class. 
            - Adds the robot object to a list of robots which will be updated
              when the ``step()`` method is called.

        """
        # TODO please fill in the options
        # TODO it seems that add has different args for every backend, are
        # any common ones?  If yes, they should be in the superclass and we
        # pass kwargs to that

        super().add()

        if isinstance(ob, rp.DHRobot) or isinstance(ob, rp.ERobot):
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

        ``env.remove(robot)`` removes the ``robot`` from the graphical environment.
        """
        # TODO should be an id to remove?

        super().remove()

    def hold(self):           # pragma: no cover
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
        try:
            plt.pause(0.001)
        except(AttributeError):
            pass

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
