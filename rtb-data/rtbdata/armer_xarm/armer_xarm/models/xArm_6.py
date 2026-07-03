#!/usr/bin/env python

import numpy as np
from pathlib import PurePosixPath
from roboticstoolbox.robot.ERobot import ERobot
from roboticstoolbox.tools import xacro
from roboticstoolbox.tools import URDF
from roboticstoolbox.tools import data
from spatialmath import SE3
from rospkg import RosPack


class xArm_6(ERobot):
    def __init__(self):
        # Acquire current path of ABB armer driver/package xacro folder
        xacro_dir = RosPack().get_path("armer_xarm") + "/data/xacro/"

        # Extra arguments expected (rtb 0.11.0)
        # Pass in base path (armer_xarm package) and pointed xacro path
        links, name, _, _ = self.URDF_read("robots/xarm6/xarm6.xacro", tld=xacro_dir)

        # Inherit ERobot object
        super().__init__(
            links, name=name, manufacturer="Ufactory", gripper_links=links[7]
        )

        # Setup ready position joint angles
        self.addconfiguration("qr", np.array([0, 0, 0, 0, 0, 0]))


if __name__ == "__main__":  # pragma nocover
    robot = xArm_6()
    print(robot)
