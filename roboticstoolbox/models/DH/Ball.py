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

    def __init__(self, N=10):

        links = []
        q1 = []

        for i in range(N):
            links.append(RevoluteDH(a=0.1, alpha=pi/2))
            q1.append(self._fract(i+1))

        # and build a serial link manipulator
        super(Ball, self).__init__(links, name='ball')

        # zero angles, L shaped pose
        self.addconfiguration("qz", np.zeros(N,))
        self.addconfiguration("q1", q1)

    def _fract(self, i):
        theta1 = 1
        theta2 = -2/3

        out = i % 3
        if out < 1:
            f = self._fract(i / 3)
        elif out < 2:
            f = theta1
        else:
            f = theta2
        return f

if __name__ == '__main__':

    ball = Ball()
    print(ball)
