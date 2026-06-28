#!/usr/bin/env python

import numpy as np
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot

# from spatialmath import SE3


class Fetch(URDFRobot):
    """
    Class that imports a Fetch URDF model

    ``Fetch()`` is a class which imports a Fetch robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Fetch()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, arm is stretched out in the x-direction
    - qr, tucked arm configuration

    .. codeauthor:: Kerry He
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):
        super().__init__(
            "fetch",
            manufacturer="Fetch",
            gripper_link_index=11,
        )

        self.qdlim = np.array([4.0, 4.0, 0.1, 1.25, 1.45, 1.57, 1.52, 1.57, 2.26, 2.26])

        self.qz = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.qr = np.array([0, 0, 0.05, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = Fetch()
    print(robot)

    for link in robot.links:
        print(link.name)
        print(link.isjoint)
        print(len(link.collision))

    print()

    for link in robot.grippers[0].links:
        print(link.name)
        print(link.isjoint)
        print(len(link.collision))
