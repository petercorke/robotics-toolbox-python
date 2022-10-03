"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as sm
from spatialmath import SE3
import unittest
from copy import copy, deepcopy

from roboticstoolbox.robot.Robot import BaseRobot


class TestBaseRobot(unittest.TestCase):
    def test_init(self):

        links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(
            "franka_description/robots/panda_arm_hand.urdf.xacro"
        )

        robot = rtb.Robot(
            links,
            name=name,
            manufacturer="Franka Emika",
            gripper_links=links[9],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        robot.grippers[0].tool = SE3(0, 0, 0.1034)

        print(robot.links)

        self.assertEqual(True, False)
