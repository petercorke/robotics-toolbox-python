#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot


class Mico(ERobot):
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

        args = super().urdf_to_ets_args(
            "kinova_description/urdf/j2n4s300_standalone.xacro")

        super().__init__(
                args[0],
                name=args[1],
                manufacturer='Kinova'
            )

        self.addconfiguration(
            "qz", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.addconfiguration(
            "qr", np.array([0, 45, 60, 0, 0, 0, 0, 0, 0, 0]) * np.pi/180)


if __name__ == '__main__':   # pragma nocover

    robot = Mico()
    print(robot)
