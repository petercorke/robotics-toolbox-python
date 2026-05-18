#!/usr/bin/env python
"""
@author (Original) Matthew Matl, Github: mmatl
@author (Adapted by) Jesse Haviland
"""

import unittest
import roboticstoolbox as rp
from roboticstoolbox.tools import URDF, xacro
import numpy as np
import numpy.testing as nt

from roboticstoolbox.tools.data import rtb_path_to_datafile


class TestURDF(unittest.TestCase):

    def test_urdf_visuals(self):

        urdf_string = xacro.main(
            rtb_path_to_datafile("xacro/franka_description/robots/panda_arm_hand.urdf.xacro"))  # noqa
        urdf = URDF.loadstr(
            urdf_string,
            rtb_path_to_datafile("xacro/franka_description/robots/panda_arm_hand.urdf.xacro"))  # noqa

        urdf.links[0].visuals[0].name = "Lonk"
        self.assertTrue(urdf.links[0].visuals[0].name == "Lonk")

        self.assertTrue(
            isinstance(
                urdf.links[0].visuals[0].origin,
                np.ndarray))

        urdf.links[0].visuals[0].geometry.box = rp.tools.urdf.Box([1, 2, 3])
        self.assertTrue(
            isinstance(
                urdf.links[0].visuals[0].geometry.geometry,
                rp.tools.urdf.Box))

        urdf.links[0].visuals[0].geometry.cylinder = \
            rp.tools.urdf.Cylinder(1, 2)

        urdf.links[0].visuals[0].geometry.sphere = \
            rp.tools.urdf.Sphere(2)

        nt.assert_array_almost_equal(
            urdf.links[0].visuals[0].geometry.box.size,
            [1, 2, 3])

        self.assertEqual(
            urdf.links[0].visuals[0].geometry.cylinder.radius,
            1)

        self.assertEqual(
            urdf.links[0].visuals[0].geometry.sphere.radius,
            2)

        self.assertTrue(
            isinstance(
                urdf.links[0].visuals[0].geometry.mesh,
                rp.tools.urdf.Mesh))

        try:
            xacro.main("")
        except BaseException:    # lgtm [py/catch-base-exception]
            pass

    def test_urdf_load(self):
        rp.models.wx250s()
        rp.models.UR5()
        rp.models.Panda()

        try:
            xacro.main("")
        except BaseException:    # lgtm [py/catch-base-exception]
            pass

    def test_urdf_collisions(self):

        urdf_string = xacro.main(
            rtb_path_to_datafile("xacro/franka_description/robots/panda_arm_hand.urdf.xacro"))  # noqa
        urdf = URDF.loadstr(
            urdf_string,
            rtb_path_to_datafile("xacro/franka_description/robots/panda_arm_hand.urdf.xacro"))  # noqa

        urdf.links[0].collisions[0].name = "Lonk"
        self.assertTrue(urdf.links[0].collisions[0].name == "Lonk")

        self.assertTrue(
            isinstance(
                urdf.links[0].collisions[0].origin,
                np.ndarray))

        try:
            xacro.main("")
        except BaseException:    # lgtm [py/catch-base-exception]
            pass

    def test_urdf_dynamics(self):

        urdf_string = xacro.main(
            rtb_path_to_datafile("xacro/franka_description/robots/panda_arm_hand.urdf.xacro"))  # noqa
        urdf = URDF.loadstr(
            urdf_string,
            rtb_path_to_datafile("xacro/franka_description/robots/panda_arm_hand.urdf.xacro"))  # noqa

        self.assertEqual(urdf.joints[0].limit.effort, 87.0)
        self.assertEqual(urdf.joints[0].limit.velocity, 2.175)

        try:
            xacro.main("")
        except BaseException:    # lgtm [py/catch-base-exception]
            pass
