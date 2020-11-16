#!/usr/bin/env python
# Created by: Aditya Dua
# 5 October 2017

from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class Orion5(DHRobot):
    """
    Class that models a RAWR robotics Orion5 manipulator

    ``Orion5()`` is a class which models a RAWR Robotics Orion5 robot and
    describes its kinematic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Orion5()
        >>> print(robot)

    Defined joint configurations are:

        - qz, zero angles, all folded up
        - qv, stretched out vertically
        - qh, arm horizontal, hand down

    .. note::

      - SI units of metres are used.
      - Robot has only 4 DoF.

    :references:
        - https://rawrobotics.com.au/orion5
        - https://drive.google.com/file/d/0B6_9_-ZgiRdTNkVqMEkxN2RSc2c/view

    .. codeauthor:: Aditya Dua
    .. codeauthor:: Peter Corke
    """

    def __init__(self, base=None):

        mm = 1e-3
        deg = pi / 180

        # details from

        h = 53.0 * mm
        r = 30.309 * mm
        l2 = 170.384 * mm
        l3 = 136.307 * mm
        l4 = 86.00 * mm
        c = 40.0 * mm

        # tool_offset = l4 + c

        # Turret, Shoulder, Elbow, Wrist, Claw
        links = [
                 RevoluteDH(d=h, a=0, alpha=90 * deg),  # Turret
                 RevoluteDH(
                    d=0, a=l2, alpha=0,         # Shoulder
                    qlim=[10 * deg, 122.5 * deg]),
                 RevoluteDH(
                    d=0, a=-l3, alpha=0,         # Elbow
                    qlim=[20 * deg, 340 * deg]),
                 RevoluteDH(
                    d=0, a=l4+c, alpha=0,        # Wrist
                    qlim=[45 * deg, 315 * deg])
            ]

        super().__init__(
            links,
            name="Orion 5",
            manufacturer="RAWR Robotics",
            base=SE3(r, 0, 0),
            tool=SE3.Ry(90, 'deg'))

        # zero angles, all folded up
        self.addconfiguration("qz", np.r_[0, 90, 180, 180] * deg)

        # stretched out vertically
        self.addconfiguration("qv", np.r_[0, 90, 180, 180] * deg)

        # arm horizontal, hand down
        self.addconfiguration("qh", np.r_[0, 0, 180, 90] * deg)


if __name__ == '__main__':   # pragma nocover

    orion = Orion5()
    print(orion)
