#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from math import pi


class Z1(ERobot):
    """
    Class that imports a Z1 URDF model

    ``Z1()`` is a class which imports a Unitree Z1 robot
    definition from a URDF file.  The model describes its kinematic and
    graphical characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Z1()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    .. codeauthor:: Keith Siilats
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
                      "z1_description/xacro/robot.xacro"
        )
        # for link in links:
        #     print(link)

        super().__init__(
            links,
            name=name.upper(),
            manufacturer="Unitree",
            gripper_links=links[9],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
        # forward, -0.000019, 1.542859, -1.037883, -0.531308, 0.002487, -0.012173, 0.999650, -0.002146, -0.026357, 0.389527, 0.002468, 0.999923, 0.012173, 0.000269, 0.026329, -0.012234, 0.999578, 0.402549, 0.000000, 0.000000, 0.000000, 1.000000,
        # start, -0.000336, 0.001634, 0.000000, 0.064640, 0.000248, 0.000230, 0.997805, 0.000104, 0.066225, 0.048696, -0.000087, 1.000000, -0.000252, 0.000011, -0.066225, 0.000246, 0.997805, 0.148729, 0.000000, 0.000000, 0.000000, 1.000000,

        self.qr = np.array([0.000922, 0.000680, -0.003654, -0.075006, -0.000130, 0.000035])
        self.qz = np.array([0.000922, 0.000680, -0.003654, -0.075006, -0.000130, 0.000035])
        self.grab3 = np.array([-0.247,1.271, -1.613, -0.267, -0.617,0.916])


        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)
        self.addconfiguration("grab3", self.grab3)

        # sol=robot.ikine_LM(SE3(0.5, -0.2, 0.2)@SE3.OA([1,0,0],[0,0,-1]))
        # 0, 0.000922, 0.000680, -0.003654, -0.075006, -0.000130, 0.000035, 0.000006, -0.002146, -0.002523, -0.003688, -0.002988, -0.000048, 0.001385, 0.016346,
        self.addconfiguration_attr(
            "qn",
            np.array(
                [0.000922, 0.000680, -0.003654, -0.075006, -0.000130, 0.000035]
            ),
        )
        self.addconfiguration_attr("q1", [0.000922, 0.000680, -0.003654, -0.075006, -0.000130, 0.000035])


if __name__ == "__main__":  # pragma nocover

    robot = Z1()
    print(robot)
    print(robot.ets())
