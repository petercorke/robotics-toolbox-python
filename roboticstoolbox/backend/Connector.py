#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from abc import ABC, abstractmethod


class Connector(ABC):

    def __init__(self):
        super().__init__()

    #
    #  Basic methods to do with the state of the external program
    #

    @abstractmethod
    def launch(self):
        '''
        env = launch(args) launch the external program with an empty or
        specific scene as defined by args

        '''

        pass

    @abstractmethod
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

        pass

    @abstractmethod
    def reset(self):
        '''
        state = reset() triggers the external program to reset to the
        original state defined by launch

        '''

        pass

    @abstractmethod
    def restart(self):
        '''
        state = restart() triggers the external program to close and relaunch
        to thestate defined by launch

        '''

        pass

    @abstractmethod
    def close(self):
        '''
        state = close() triggers the external program to gracefully close

        '''

        pass

    #
    #  Methods to interface with the robots created in other environemnts
    #

    @abstractmethod
    def add(self):
        '''
        id = add(robot) adds the robot to the external environment. robot must
        be of an appropriate class. This adds a robot object to a list of
        robots which will act upon the step() method being called.

        '''

        pass

    @abstractmethod
    def remove(self):
        '''
        id = remove(robot) removes the robot to the external environment.

        '''

        pass
