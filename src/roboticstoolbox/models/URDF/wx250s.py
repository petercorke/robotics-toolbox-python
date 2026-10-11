#!/usr/bin/env python

import numpy as np
from spatialmath import SE3
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot


class wx250s(URDFRobot):
    """
    Class that imports a wx250s URDF model

    ``wx250s()`` is a class which imports an Interbotix wx250s robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.wx250s()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx250s.html

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        super().__init__(
            "trossen_descriptions/urdf/wx250s.urdf.xacro",
            manufacturer="Interbotix",
            gripper_link_name="/ee_arm_link",
        )

        self.grippers[0].tool = SE3(0.05057, 0, 0)

        self.qr = np.array([0, -0.3, 0.8, 0, -0.5, 0])
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = wx250s()
    print(robot)
