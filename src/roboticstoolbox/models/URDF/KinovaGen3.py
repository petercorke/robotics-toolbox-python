#!/usr/bin/env python

import numpy as np
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot


class KinovaGen3(URDFRobot):
    """
    Class that imports a KinovaGen3 URDF model

    ``KinovaGen3()`` is a class which imports a KinovaGen3 robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.KinovaGen3()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):
        super().__init__(
            "gen3",
            manufacturer="Kinova",
            gripper_link_index=10,
        )

        # self.qdlim = np.array([
        # 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0])

        self.qr = np.deg2rad([0.0, 15.0, 180.0, 230.0, 0.0, 55.0, 90.0])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = KinovaGen3()
    robot.q = robot.qr
    print(robot)

    from swift import Swift

    env = Swift()
    env.launch()
    env.add(robot)
    env.hold()

    for link in robot.links:
        print("\n")
        print(link.isjoint)
