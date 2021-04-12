#!/usr/bin/env python3
"""
@author: Jesse Haviland
@author: Peter Corke
"""

import numpy as np


class KinematicState:

    def __init__(self, robot):
        self.robot = robot
        self._invalidate()

        self._fkine = np.empty((4, 4))
        self._jacobe = np.empty((6, robot.n))
        self._jacob0 = np.empty((6, robot.n))
        self._hessian0 = np.empty((6, robot.n, robot.n))

    def _invalidate(self):

        self.__fkine = False
        self.__jacobe = False
        self.__jacob0 = False
        self.__hessian0 = False

    def q(self):
        self.q = quit
        self._invalidate()

    def fkine(self):
        if self.__fkine:
            return self._fkine
        else:
            self._fkine = self.robot.fkine()
            self.__fkine = True
            return self._fkine
