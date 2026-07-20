#!/usr/bin/env python

import numpy as np
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot


class px150(URDFRobot):
    """
    Class that imports a PX150 URDF model

    ``px150()`` is a class which imports an Interbotix px150 robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.px150()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/px150.html

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        super().__init__(
            "trossen_descriptions/urdf/px150.urdf.xacro", manufacturer="Interbotix"
        )

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0])
        self.qz = np.zeros(8)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = px150()
    print(robot)
