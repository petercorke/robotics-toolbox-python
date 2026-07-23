#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import unittest
from roboticstoolbox import Robot
from roboticstoolbox.models.URDF.URDFRobot import URDF_read
from spatialmath import SE3

from tests import skip_on_pyodide


class TestCustomXacro(unittest.TestCase):
    @skip_on_pyodide
    def test_custom(self):
        class CustomPanda(Robot):
            def __init__(self):
                links, name, urdf_filepath = URDF_read("panda")
                super().__init__(
                    links,
                    name="Custom Robot",
                    manufacturer="N/A",
                    gripper_links=links[9],
                )
                self._urdf_filepath = str(urdf_filepath) if urdf_filepath else ""
                self.grippers[0].tool = SE3(0, 0, 0.1034)
                self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        robot = CustomPanda()

        nt.assert_almost_equal(
            robot.qr, np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        )


if __name__ == "__main__":
    unittest.main()
