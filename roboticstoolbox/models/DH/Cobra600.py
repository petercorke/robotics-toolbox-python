#!/usr/bin/env python
"""
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from math import pi
import numpy as np


class Cobra600(DHRobot):
    """
    Class that models a Adept Cobra 600 SCARA manipulator

    ``Cobra600()`` is a class which models an Adept Cobra 600 SCARA robot and
    describes its kinematic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Cobra600()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration

    .. note::
        - SI units are used.
        - Robot has only 4 DoF.

    .. codeauthor:: Peter Corke
    """

    def __init__(self):
        deg = pi/180

        L = [RevoluteDH(d=0.387, a=0.325, qlim=[-50*deg, 50*deg]),
             RevoluteDH(a=0.275, alpha=pi, qlim=[-88*deg, 88*deg]),
             PrismaticDH(qlim=[0, 0.210]),
             RevoluteDH()]

        super().__init__(L, name='Cobra600', manufacturer='Adept')

        # zero angles, L shaped pose
        self.addconfiguration("qz", np.array([0, 0, 0, 0]))


if __name__ == '__main__':   # pragma nocover

    cobra = Cobra600()
    print(cobra)
