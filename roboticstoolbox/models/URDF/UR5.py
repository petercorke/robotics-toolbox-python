#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot
from math import pi


class UR5(Robot):
    """
    Class that imports a UR5 URDF model

    ``UR3()`` is a class which imports a Universal Robotics UR5 robot
    definition from a URDF file.  The model describes its kinematic and
    graphical characteristics.

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

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "ur_description/urdf/ur5_joint_limited_robot.urdf.xacro"
        )
        # for link in links:
        #     print(link)

        super().__init__(
            links,
            name=name.upper(),
            manufacturer="Universal Robotics",
            gripper_links=links[7],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

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
