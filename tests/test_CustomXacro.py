#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import unittest
from roboticstoolbox import Robot
from spatialmath import SE3
from roboticstoolbox.tools.data import rtb_path_to_datafile
from distutils.dir_util import copy_tree
from os import mkdir, path
import tempfile as tf


class TestCustomXacro(unittest.TestCase):
    def test_custom(self):
        class CustomPanda(Robot):
            def __init__(self, xacro_path):

                links, name, urdf_string, urdf_filepath = self.URDF_read(
                    "franka_description/robots/panda_arm_hand.urdf.xacro",
                    tld=xacro_path,
                )

                super().__init__(
                    links,
                    name="Custom Robot",
                    manufacturer="N/A",
                    gripper_links=links[9],
                    urdf_string=urdf_string,
                    urdf_filepath=urdf_filepath,
                )

                self.grippers[0].tool = SE3(0, 0, 0.1034)
                self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        temp_dir = tf.mkdtemp()

        xacro_dir = path.join(temp_dir, "custom_xacro_folder")

        # Make xacro folder
        mkdir(xacro_dir)

        franka_dir = path.join(xacro_dir, "franka_description")

        # Make franka folder
        mkdir(franka_dir)

        xacro_path = rtb_path_to_datafile("xacro")
        panda_xacro = xacro_path / "franka_description"

        # Copy into temp franka directory
        copy_tree(panda_xacro, franka_dir)

        # Make our custom robot
        robot = CustomPanda(xacro_dir)

        nt.assert_almost_equal(
            robot.qr, np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        )


if __name__ == "__main__":

    unittest.main()
