#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot


class rx200(Robot):
    """
    Class that imports a RX200 URDF model

    ``rx200()`` is a class which imports an Interbotix rx200 robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.rx200()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/rx200.html

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    _urdf_path = "interbotix_descriptions/urdf/rx200.urdf.xacro"
    _manufacturer = "Interbotix"

    def __init__(self):

        super().__init__()

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = rx200()
    print(robot)
