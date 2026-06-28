#!/usr/bin/env python

import numpy as np
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot


class Mico(URDFRobot):
    """
    Class that imports a Mico URDF model

    ``Panda()`` is a class which imports a Kinova Mico robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Mico()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        super().__init__(
            "kinova_description/urdf/j2n4s300_standalone.xacro",
            manufacturer="Kinova",
            gripper_link_index=8,
        )

        self.qr = np.array([0, 45, 60, 0]) * np.pi / 180
        self.qz = np.zeros(4)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = Mico()
    print(robot)
