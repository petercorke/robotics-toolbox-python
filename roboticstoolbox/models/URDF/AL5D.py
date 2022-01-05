#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from math import pi


class AL5D(ERobot):
    """
    Class that imports a Puma 560 URDF model

    ``Puma560()`` is a class which imports a Unimation Puma560 robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.AL5D()
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

    .. codeauthor:: Tassos Natsakis
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "al5d_description/urdf/al5d_robot.urdf"
        )

        super().__init__(
            links,
            name=name,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.manufacturer = "Lynxmotion"
        # self.ee_link = self.ets[9]

        # zero angles, upper arm horizontal, lower up straight ahead
        self.addconfiguration("qz", np.array([0, 0, 0, 0]))

        # reference pose, arm to the right, elbow up
        self.addconfiguration("ru", np.array([0.0000, 0.0000, 1.5707, 0.0000]))

if __name__ == "__main__":  # pragma nocover

    robot = AL5D()
    print(robot)
