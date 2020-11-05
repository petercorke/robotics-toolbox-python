#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot


class px100(ERobot):
    """
    Class that imports a PX100 URDF model

    ``px100()`` is a class which imports an Interbotix px100 robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.px100()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - http://www.support.interbotix.com/html/specifications/px100.html

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """
    def __init__(self):

        args = super().urdf_to_ets_args(
            "interbotix_descriptions/urdf/px100.urdf.xacro")

        super().__init__(
                args[0],
                name=args[1],
                manufacturer='Interbotix'
            )

        self.addconfiguration(
            "qz", np.array([0, 0, 0, 0, 0, 0, 0]))
        self.addconfiguration(
            "qr", np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]))


if __name__ == '__main__':   # pragma nocover

    robot = px100()
    print(robot.links[0].mesh)
