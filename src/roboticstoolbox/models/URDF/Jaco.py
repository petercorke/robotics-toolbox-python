#!/usr/bin/env python

import numpy as np
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot


class Jaco(URDFRobot):
    """
    Class that imports a Kinova Jaco URDF model

    ``Jaco()`` is a class which imports a Kinova Jaco 2 (j2n6s200) robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Jaco()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, nominal 'ready' configuration

    .. codeauthor:: Peter Corke
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        super().__init__(
            "j2n6s200",
            manufacturer="Kinova",
            gripper_link_index=9,
        )

        self.qr = np.array([0, 45, 60, 0, 0, 0]) * np.pi / 180
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = Jaco()
    print(robot)
