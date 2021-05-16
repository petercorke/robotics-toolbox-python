#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot


class KinovaGen3(ERobot):
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

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "kortex_description/robots/gen3.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Kinova",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
            # gripper_links=elinks[9]
        )

        # self.qdlim = np.array([
        # 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0])

        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0, 0]))

        self.addconfiguration(
            "qr", np.array([np.pi, -0.3, 0, -1.6, 0, -1.0, np.pi / 2])
        )


if __name__ == "__main__":  # pragma nocover

    robot = KinovaGen3()
    print(robot)
