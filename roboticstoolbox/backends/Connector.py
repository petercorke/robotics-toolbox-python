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
        launch(args) launch the external program with an empty or
        specific scene as defined by args

        '''

        pass

    @abstractmethod
    def step(self):
        '''
        step(dt) triggers the external program to make a time step
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
        reset() triggers the external program to reset to the
        original state defined by launch

        '''

        pass

    @abstractmethod
    def restart(self):
        '''
        restart() triggers the external program to close and relaunch
        to thestate defined by launch

        '''

        pass

    @abstractmethod
    def close(self):
        '''
        close() triggers the external program to gracefully close

        '''

        pass

    #
    #  Methods to interface with the robots created in other environemnts
    #

    @abstractmethod
    def add(self):
        '''
        id = add(object) adds the object to the external environment. object must
        be of an appropriate class. This adds a object object to a list of
        objects which will act upon the step() method being called.

        '''

        pass

    @abstractmethod
    def remove(self):
        '''
        remove(id) removes the object from the external environment.

        '''

        pass

    @abstractmethod
    def hold(self):    # pragma nocover
        '''
        hold() keeps the backend open i.e. stops the program from closing once
        the main script has finished. This method may need keep an even loop
        running for the backend to keep it responsive.

        '''

        pass
