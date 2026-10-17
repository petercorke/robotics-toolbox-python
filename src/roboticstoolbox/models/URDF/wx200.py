
from __future__ import annotations

#!/usr/bin/env python
import numpy as np
from spatialmath import SE3

from roboticstoolbox.models.URDF.URDFRobot import URDFRobot


class wx200(URDFRobot):
    """
    Class that imports a WX200 URDF model

    ``wx200()`` is a class which imports an Interbotix wx200 robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.wx200()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx200.html

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self) -> None:

        super().__init__(
            "trossen_descriptions/urdf/wx200.urdf.xacro",
            manufacturer="Interbotix",
            gripper_link_name="/ee_arm_link",
        )

        # ee_gripper_link is reached from ee_arm_link by two fixed x-translations
        # (0.023 m to fingers_link and 0.027575 m to ee_gripper_link).
        self.grippers[0].tool = SE3.Tx(0.050575)

        # Only the five arm joints belong to Robot.q; the three gripper joints
        # are managed by the Gripper object. Keep named arm poses within limits.
        self.qr = np.array([0, -0.3, 0, -2.0, 0])
        self.qz = np.zeros(5)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = wx200()
    print(robot)
