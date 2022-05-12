#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
import fknm


class FastRobot(ERobot):
    def __new__(cls, robot):
        robot.__class__ = cls
        return robot

    def __init__(self, robot):
        pass

    def jacob0(self, q, end=None, start=None, tool=None):
        path, n, etool = self.get_path(end, start, _fknm=True)
        if tool is None:
            tool = self._eye_fknm
        J = np.empty((6, n))
        fknm.jacob0(len(path), n, path, q, etool, tool, J)
        return J

    def jacobe(self, q, end=None, start=None, tool=None):
        path, n, etool = self.get_path(end, start, _fknm=True)
        if tool is None:
            tool = self._eye_fknm
        J = np.empty((6, n))
        fknm.jacobe(len(path), n, path, q, etool, tool, J)
        return J

    def fkine(self, q, end=None, start=None, tool=None, include_base=True):
        if start is not None:
            include_base = False

        path, _, etool = self.get_path(end, start, _fknm=True)
        m = len(path)

        if tool is None:
            tool = self._eye_fknm

        T = np.empty((4, 4))
        fknm.fkine(m, path, q, etool, tool, T)

        if self._base is not None and start is None and include_base == True:
            return self.base.A @ T
        else:
            return T
