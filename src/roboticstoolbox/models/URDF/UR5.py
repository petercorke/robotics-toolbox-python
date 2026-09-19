#!/usr/bin/env python

import numpy as np
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot
from math import pi


class UR5(URDFRobot):
    """
    Class that imports a UR5 URDF model

    ``UR3()`` is a class which imports a Universal Robotics UR5 robot
    definition from a URDF file.  The model describes its kinematic and
    graphical characteristics.

    The model is loaded via the `robot_descriptions
    <https://github.com/robot-descriptions/robot_descriptions.py>`_ package.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.UR5()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        # Name-based lookup, not a raw positional index: robot_descriptions
        # 3.0.0 changed which upstream repo "ur5" resolves to, reordering
        # links so index 7 silently pointed at a real arm joint instead of
        # the tool attachment link (#578). tool0/ee_link/flange are stable
        # names across the versions checked.
        super().__init__(
            "ur5",
            manufacturer="Universal Robotics",
            gripper_link_name=["tool0", "ee_link", "flange"],
        )

        # for link in links:
        #     print(link)

        self.qr = np.array([np.pi, 0, 0, 0, np.pi / 2, 0])
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

        # sol=robot.ikine_LM(SE3(0.5, -0.2, 0.2)@SE3.OA([1,0,0],[0,0,-1]))
        self.addconfiguration_attr(
            "qn",
            np.array(
                [
                    -7.052413e-01,
                    3.604328e-01,
                    -1.494176e00,
                    1.133744e00,
                    -7.052413e-01,
                    0,
                ]
            ),
        )
        self.addconfiguration_attr("q1", [0, -pi / 2, pi / 2, 0, pi / 2, 0])


if __name__ == "__main__":  # pragma nocover
    robot = UR5()
    print(robot)
    print(robot.ets())
