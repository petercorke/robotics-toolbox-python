#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import os
from subprocess import call, Popen
from roboticstoolbox.backend.Connector import Connector
import zerorpc
import roboticstoolbox as rp
import numpy as np
import spatialmath as sm


class Swift(Connector):  # pragma nocover
    """
    Graphical backend using Swift

    Swift is an Electron app built on three.js. It supports many 3D graphical
    primitives including meshes, boxes, ellipsoids and lines. It can render
    Collada objects in full color.

    Example:

    .. code-block:: python
        :linenos:

        import roboticstoolbox as rtb

        robot = rtb.models.DH.Panda()  # create a robot

        pyplot = rtb.backend.Swift()   # create a Swift backend
        pyplot.add(robot)              # add the robot to the backend
        robot.q = robot.qz             # set the robot configuration
        pyplot.step()                  # update the backend and graphical view

    :references:

        - https://github.com/jhavl/swift
        
    """
    def __init__(self):
        super(Swift, self).__init__()

        # Popen(['npm', 'start', '--prefix', os.environ['SIM_ROOT']])

    #
    #  Basic methods to do with the state of the external program
    #

    def launch(self):
        """
        Launch a graphical backend in Swift

        ``env = launch(args)`` create a 3D scene in a running Swift instance as
        defined by args, and returns a reference to the backend.

        """

        super().launch()

        self.robots = []
        self.shapes = []

        self.swift = zerorpc.Client()
        self.swift.connect("tcp://127.0.0.1:4242")

    def step(self, dt=50):
        """
        Update the graphical scene

        :param dt: time step in milliseconds, defaults to 50
        :type dt: int, optional
 
        ``env.step(args)`` triggers an update of the 3D scene in the Swift
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

        # TODO how is the pose of shapes updated prior to step?
        
        super().step

        self._step_robots(dt)
        self._step_shapes(dt)

        # self._draw_ellipses()
        self._draw_all()

    def reset(self):
        """
        Reset the graphical scene

        ``env.reset()`` triggers a reset of the 3D scene in the Swift window
        referenced by ``env``. It is restored to the original state defined by
        ``launch()``.

        """

        super().reset

    def restart(self):
        """
        Restart the graphics display

        ``env.restart()`` triggers a restart of the Swift view referenced by
        ``env``. It is closed and relaunched to the original state defined by
        ``launch()``.

        """

        super().restart

    def close(self):
        """
        Close the graphics display

        ``env.close()`` gracefully disconnectes from the Swift visualizer
        referenced by ``env``.
        """

        super().close()

    #
    #  Methods to interface with the robots created in other environemnts
    #

    def add(self, ob, show_robot=True, show_collision=False):
        """
        Add a robot to the graphical scene

        :param ob: the object to add
        :type ob: ???
        :param show_robot: ????, defaults to True
        :type show_robot: bool, optional
        :param show_collision: ???, defaults to False
        :type show_collision: bool, optional
        :return: object id within visualizer
        :rtype: int

        ``id = env.add(robot)`` adds the ``robot`` to the graphical environment.

        .. note::

            - Adds the robot object to a list of robots which will be updated
              when the ``step()`` method is called.

        """
        # id = add(robot) adds the robot to the external environment. robot must
        # be of an appropriate class. This adds a robot object to a list of
        # robots which will act upon the step() method being called.

        # TODO can add more than a robot right?

        super().add()

        if isinstance(ob, rp.ERobot):
            robot = ob.to_dict()
            robot['show_robot'] = show_robot
            robot['show_collision'] = show_collision
            id = self.swift.robot(robot)
            self.robots.append(ob)
            return id
        elif isinstance(ob, rp.Shape):
            shape = ob.to_dict()
            id = self.swift.shape(shape)
            self.shapes.append(ob)
            return id

    def remove(self):
        """
        Remove a robot to the graphical scene

        ``env.remove(robot)`` removes the ``robot`` from the graphical environment.
        """

        # TODO - shouldn't this have an id argument? which robot does it remove
        # TODO - it can remove any entity?

        super().remove()

    def _step_robots(self, dt):

        for robot in self.robots:

            # if rpl.readonly or robot.control_type == 'p':
            #     pass            # pragma: no cover

            if robot.control_type == 'v':

                for i in range(robot.n):
                    robot.q[i] += robot.qd[i] * (dt / 1000)

                    if np.any(robot.qlim[:, i] != 0) and \
                            not np.any(np.isnan(robot.qlim[:, i])):
                        robot.q[i] = np.min([robot.q[i], robot.qlim[1, i]])
                        robot.q[i] = np.max([robot.q[i], robot.qlim[0, i]])

            elif robot.control_type == 'a':
                pass

            else:            # pragma: no cover
                # Should be impossible to reach
                raise ValueError(
                    'Invalid robot.control_type. '
                    'Must be one of \'p\', \'v\', or \'a\'')

    def _step_shapes(self, dt):

        for shape in self.shapes:

            T = shape.base
            t = T.t
            r = T.rpy('rad')

            t += shape.v[:3] * (dt / 1000)
            r += shape.v[3:] * (dt / 1000)

            shape.base = sm.SE3(t) * sm.SE3.RPY(r)

    def _draw_all(self):

        for i in range(len(self.robots)):
            self.robots[i].fkine_all()
            self.swift.robot_poses([i, self.robots[i].fk_dict()])

        for i in range(len(self.shapes)):
            self.swift.shape_poses([i, self.shapes[i].fk_dict()])

    def record_start(self, file):
        self.swift.record_start(file)

    def record_stop(self):
        self.swift.record_stop(1)
