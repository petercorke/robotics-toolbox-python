#!/usr/bin/env python
"""
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH
from math import pi
import numpy as np


class Ball(DHRobot):
    """
    Class that models a ball manipulator

    ``Ball()`` is a class which models a ball robot and
    describes its kinematic characteristics using standard DH
    conventions.

    The ball robot is an *abstract* robot with an arbitrary number of joints 
    that folds into a ball shape.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Ball()
        >>> print(robot)

    Defined joint configurations are:

        - qz, zero joint angles
        - q1, ball shaped configuration
        
    .. note::
        - SI units are used.
        - The model includes armature inertia and gear ratios.
        - The value of m1 is given as 0 here.  Armstrong found no value for it
          and it does not appear in the equation for tau1 after the
          substituion is made to inertia about link frame rather than COG
          frame.
        - Gravity load torque is the motor torque necessary to keep the joint
          static, and is thus -ve of the gravity caused torque.

    :references:
    
        - "A divide and conquer articulated-body algorithm for parallel O(log(n))
          calculation of rigid body dynamics, Part 2",
          Int. J. Robotics Research, 18(9), pp 876-892.

    .. codeauthor:: Peter Corke
    """

    def __init__(self, N=10):

        links = []
        q1 = []

        for i in range(N):
            links.append(RevoluteDH(a=0.1, alpha=pi/2))
            q1.append(self._fract(i+1))

        # and build a serial link manipulator
        super(Ball, self).__init__(links, name='ball')

        # zero angles, ball pose
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


if __name__ == '__main__':   # pragma nocover

    ball = Ball()
    print(ball)
