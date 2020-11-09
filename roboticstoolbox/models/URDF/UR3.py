#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot


class UR3(ERobot):
    """
    Class that imports a UR3 URDF model

    ``UR3()`` is a class which imports a Universal Robotics UR3 robot
    definition from a URDF file.  The model describes its kinematic and
    graphical characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.UR3()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """
    def __init__(self):

        args = self.urdf_to_ets_args(
            "ur_description/urdf/ur3_joint_limited_robot.urdf.xacro")

        super().__init__(
                args[0],
                name=args[1],
                manufacturer='Universal Robotics'
            )

        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))
        self.addconfiguration("qr", np.array([np.pi, 0, 0, 0, np.pi/2, 0]))


if __name__ == '__main__':   # pragma nocover

    robot = UR3()
    print(robot)
