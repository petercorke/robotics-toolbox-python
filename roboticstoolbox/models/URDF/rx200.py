#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot


class rx200(ERobot):
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
        - http://www.support.interbotix.com/html/specifications/rx200.html#

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """
    def __init__(self):

        args = super().urdf_to_ets_args(
            "interbotix_descriptions/urdf/rx200.urdf.xacro")

        super().__init__(
                args[0],
                name=args[1],
                manufacturer='Interbotix'
            )

        self.addconfiguration(
            "qz", np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        self.addconfiguration(
            "qr", np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4, 0]))


if __name__ == '__main__':   # pragma nocover

    robot = rx200()
    print(robot)
