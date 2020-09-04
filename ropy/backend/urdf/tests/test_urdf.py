#!/usr/bin/env python
"""
@author (Original) Matthew Matl, Github: mmatl
@author (Adapted by) Jesse Haviland
"""

import unittest
import ropy as rp
from ropy.backend import URDF, Link, Joint, Transmission, xacro
import numpy as np
import numpy.testing as nt


class TestURDF(unittest.TestCase):

    def test_urdfpy(self):

        # Load
        u = URDF.load('ropy/backend/urdf/tests/data/ur5.urdf')

        self.assertIsInstance(u, URDF)
        for j in u.joints:
            self.assertIsInstance(j, Joint)
        for ln in u.links:
            self.assertIsInstance(ln, Link)
        for t in u.transmissions:
            self.assertIsInstance(t, Transmission)

    def test_urdf_visuals(self):

        urdf_string = xacro.main(
            "ropy/models/xacro/panda/robots/panda_arm_hand.urdf.xacro")
        urdf = URDF.loadstr(
            urdf_string,
            "ropy/models/xacro/panda/robots/panda_arm_hand.urdf.xacro")

        urdf.links[0].visuals[0].name = "Lonk"
        self.assertTrue(urdf.links[0].visuals[0].name == "Lonk")

        self.assertTrue(
            isinstance(
                urdf.links[0].visuals[0].origin,
                np.ndarray))

        urdf.links[0].visuals[0].geometry.box = rp.backend.urdf.Box([1, 2, 3])
        self.assertTrue(
            isinstance(
                urdf.links[0].visuals[0].geometry.geometry,
                rp.backend.urdf.Box))

        urdf.links[0].visuals[0].geometry.cylinder = \
            rp.backend.urdf.Cylinder(1, 2)

        urdf.links[0].visuals[0].geometry.sphere = \
            rp.backend.urdf.Sphere(2)

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
                rp.backend.urdf.Mesh))

        try:
            xacro.main("")
        except BaseException:
            pass

    def test_urdf_load(self):
        rp.wx250s()
        rp.UR5()
        rp.PandaURDF()

        try:
            xacro.main("")
        except BaseException:
            pass

    def test_urdf_collisions(self):

        urdf_string = xacro.main(
            "ropy/models/xacro/panda/robots/panda_arm_hand.urdf.xacro")
        urdf = URDF.loadstr(
            urdf_string,
            "ropy/models/xacro/panda/robots/panda_arm_hand.urdf.xacro")

        urdf.links[0].collisions[0].name = "Lonk"
        self.assertTrue(urdf.links[0].collisions[0].name == "Lonk")

        self.assertTrue(
            isinstance(
                urdf.links[0].collisions[0].origin,
                np.ndarray))

        try:
            xacro.main("")
        except BaseException:
            pass

    def test_urdf_dynamics(self):

        urdf_string = xacro.main(
            "ropy/models/xacro/panda/robots/panda_arm_hand.urdf.xacro")
        urdf = URDF.loadstr(
            urdf_string,
            "ropy/models/xacro/panda/robots/panda_arm_hand.urdf.xacro")

        self.assertEqual(urdf.joints[0].limit.effort, 87.0)
        self.assertEqual(urdf.joints[0].limit.velocity, 2.175)

        try:
            xacro.main("")
        except BaseException:
            pass
