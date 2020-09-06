#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import os
from subprocess import call, Popen
from ropy.backend.Connector import Connector
import zerorpc
import ropy as rp
import numpy as np


class Sim(Connector):  # pragma nocover

    def __init__(self):
        super(Sim, self).__init__()

        # Popen(['npm', 'start', '--prefix', os.environ['SIM_ROOT']])

    #
    #  Basic methods to do with the state of the external program
    #

    def launch(self):
        '''
        env = launch(args) launch the external program with an empty or
        specific scene as defined by args

        '''

        super().launch()

        self.robots = []

        self.sim = zerorpc.Client()
        self.sim.connect("tcp://127.0.0.1:4242")

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

        super().step

        self._step_robots(dt)

        # self._draw_ellipses()
        self._draw_robots()

        # self._update_robots()

    def reset(self):
        '''
        state = reset() triggers the external program to reset to the
        original state defined by launch

        '''

        super().reset

    def restart(self):
        '''
        state = restart() triggers the external program to close and relaunch
        to thestate defined by launch

        '''

        super().restart

    def close(self):
        '''
        state = close() triggers the external program to gracefully close

        '''

        super().close()

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

        if isinstance(ob, rp.ETS):
            robot = ob.to_dict()
            id = self.sim.robot(robot)
            self.robots.append(ob)
            return id

    def remove(self):
        '''
        id = remove(robot) removes the robot to the external environment.

        '''

        super().remove()

    def _step_robots(self, dt):

        for robot in self.robots:

            # if rpl.readonly or robot.control_type == 'p':
            #     pass            # pragma: no cover

            if robot.control_type == 'v':

                for i in range(robot.n):
                    robot.q[i] += robot.qd[i] * (dt / 1000)

                    if np.any(robot.qlim[:, i] != 0):
                        robot.q[i] = np.min([robot.q[i], robot.qlim[1, i]])
                        robot.q[i] = np.max([robot.q[i], robot.qlim[0, i]])

            elif robot.control_type == 'a':
                pass

            else:            # pragma: no cover
                # Should be impossible to reach
                raise ValueError(
                    'Invalid robot.control_type. '
                    'Must be one of \'p\', \'v\', or \'a\'')

    def _draw_robots(self):

        for i in range(len(self.robots)):
            self.robots[i].allfkine()
            self.sim.poses([i, self.robots[i].fk_dict()])
