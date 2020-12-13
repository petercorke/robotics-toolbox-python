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

    :param N: number of links, defaults to 10
    :type N: int, optional
    :param symbolic: [description], defaults to False
    :type symbolic: bool, optional

    The ball robot is an *abstract* robot with an arbitrary number of joints.
    At zero joint angles it is straight along the x-axis, and as the joint
    angles are increased (equally) it wraps up into a 3D ball shape.

    - ``Ball()`` is an object which describes the kinematic characteristics of
      a ball robot using standard DH conventions.

    - ``Ball(N)`` as above, but models a robot with ``N`` joints.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Ball()
        >>> print(robot)

    Defined joint configurations are:

        - qz, zero joint angles
        - q1, ball shaped configuration

    :references:

        - "A divide and conquer articulated-body algorithm for parallel
          O(log(n)) calculation of rigid body dynamics, Part 2",
          Int. J. Robotics Research, 18(9), pp 876-892.

    :seealso: :func:`Hyper`, :func:`Ball`

    .. codeauthor:: Peter Corke
    """

    def __init__(self, N=10):

        links = []
        q1 = []

        for i in range(N):
            links.append(RevoluteDH(a=0.1, alpha=pi / 2))

        # and build a serial link manipulator
        super(Ball, self).__init__(links, name='ball')

        # zero angles, ball pose
        self.addconfiguration("qz", np.zeros(N,))
        self.addconfiguration("q1", [_fract(i) for i in range(N)])

def _fract(i):
    # i is "i-1" as per the paper
    theta1 = 1
    theta2 = -2/3

    if i == 0:
        return theta1
    elif i % 3 == 1:
        return theta1
    elif i % 3 == 2:
        return theta2
    elif i % 3 == 0:
        return _fract(i / 3)

if __name__ == '__main__':   # pragma nocover

    ball = Ball()
    print(ball)
