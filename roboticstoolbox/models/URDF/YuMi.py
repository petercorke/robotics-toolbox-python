#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot


class YuMi(ERobot):
    """
    Class that imports an ABB YuMi URDF model

    ``YuMi()`` is a class which imports an ABB YuMi (IRB14000) robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.YuMi()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - `https://github.com/OrebroUniversity/yumi <https://github.com/OrebroUniversity/yumi>`_

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name = self.URDF_read("yumi_description/urdf/yumi.urdf")

        super().__init__(
            links, name=name, manufacturer="ABB", gripper_links=[links[20]]
        )

        self.addconfiguration("qz", np.zeros((14,)))
        self.addconfiguration("qr", np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]))


if __name__ == "__main__":  # pragma nocover

    robot = YuMi()
    print(robot)
