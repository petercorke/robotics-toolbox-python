#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot


class vx300s(Robot):
    """
    Class that imports a VX300s URDF model

    ``vx300s()`` is a class which imports an Interbotix vx300s robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.vx300s()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "interbotix_descriptions/urdf/vx300s.urdf.xacro"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Interbotix",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0, 0])
        self.qz = np.zeros(9)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    robot = vx300s()
    print(robot)
