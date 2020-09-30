#!/usr/bin/env python
"""
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH
from math import pi
import numpy as np


class Ball(DHRobot):
    '''
    reate model of a ball manipulator

    Ball() creates the workspace variable ball which describes the
    kinematic characteristics of a serial link manipulator with 50 joints
    that folds into a ball shape.

    Ball(N) as above but creates a manipulator with N joints.

    Also define the workspace vectors:
    q  joint angle vector for default ball configuration
    Reference:
    - "A divide and conquer articulated-body algorithm for parallel O(log(n))
    calculation of rigid body dynamics, Part 2",
    Int. J. Robotics Research, 18(9), pp 876-892.

    Notes:
    - Unlike most other model scripts this one is actually a function that
    behaves like a script and writes to the global workspace.
    '''

    def __init__(self, N=None):

        links = []
        self._qz = []
        if not N:
            N = 10
        self.N = N

        for i in range(self.N):
            links.append(RevoluteDH(a=0.1, alpha=pi/2))
            self._qz.append(self.fract(i+1))

        # and build a serial link manipulator
        super(Ball, self).__init__(links, name='ball')

    @property
    def qz(self):
        return self._qz

    def fract(self, i):
        theta1 = 1
        theta2 = -2/3

        out = i % 3
        if out < 1:
            f = self.fract(i/3)
        elif out < 2:
            f = theta1
        else:
            f = theta2
        return f

if __name__ == '__main__':

    ball = Ball()
    print(ball)
