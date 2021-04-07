#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from math import pi


class Puma560(ERobot):
    """
    Class that imports a Puma 560 URDF model

    ``Puma560()`` is a class which imports a Unimation Puma560 robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Puma560()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    .. warning:: This file has been modified so that the zero-angle pose is the
        same as the DH model in the toolbox. ``j3`` rotation is changed from
        -𝜋/2 to 𝜋/2.  Dimensions are also slightly different.  Both models
        include the pedestal height.

    .. note:: The original file is from https://github.com/nimasarli/puma560_description/blob/master/urdf/puma560_robot.urdf.xacro

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name = self.URDF_read(
            "puma560_description/urdf/puma560_robot.urdf.xacro")

        super().__init__(
            links,
            name=name)

        self.manufacturer = "Unimation"
        # self.ee_link = self.ets[9]

        # zero angles, L shaped pose
        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))

        # ready pose, arm up
        self.addconfiguration("qr", np.array([0, pi/2, -pi/2, 0, 0, 0]))

        # straight and horizontal
        self.addconfiguration("qs", np.array([0, 0, -pi/2, 0, 0, 0]))

        # nominal table top picking pose
        self.addconfiguration("qn", np.array([0, pi/4, pi, 0, pi/4, 0]))


if __name__ == '__main__':   # pragma nocover

    robot = Puma560()
    print(robot)
