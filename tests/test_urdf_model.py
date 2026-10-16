#!/usr/bin/env python
"""
Tests of the URDF model classes (``roboticstoolbox.tools.urdf``): the visuals,
collisions and joint limits read from a URDF, and loading some URDF robots.

@author (Original) Matthew Matl, Github: mmatl
@author (Adapted by) Jesse Haviland
"""

import unittest
from pathlib import Path

import numpy as np
import numpy.testing as nt
import rtbdata
from xacrodoc import XacroDoc, packages

import roboticstoolbox as rtb
from roboticstoolbox.tools.urdf import URDF, Box, Cylinder, Mesh, Sphere

XACRO_ROOT = Path(rtbdata.__file__).parent / "xacro"
PANDA_XACRO = XACRO_ROOT / "qut_frankie_description/robots/panda_arm_hand.urdf.xacro"


def panda_urdf() -> URDF:
    """The bundled Panda xacro, expanded and parsed into a ``URDF`` object"""
    # as ``URDF_file`` does, so that ``$(find <package>)`` resolves
    packages.update_package_cache(
        {d.name: str(d) for d in XACRO_ROOT.iterdir() if d.is_dir()}
    )
    return URDF.loadstr(XacroDoc.from_file(PANDA_XACRO).to_urdf_string(), None)


class TestURDF(unittest.TestCase):
    def test_urdf_visuals(self):
        urdf = panda_urdf()
        visual = urdf.links[0].visuals[0]

        visual.name = "Lonk"
        self.assertEqual(visual.name, "Lonk")

        self.assertIsInstance(visual.origin, np.ndarray)

        visual.geometry.box = Box([1, 2, 3])
        self.assertIsInstance(visual.geometry.geometry, Box)

        visual.geometry.cylinder = Cylinder(1, 2)
        visual.geometry.sphere = Sphere(2)

        nt.assert_array_almost_equal(visual.geometry.box.size, [1, 2, 3])
        self.assertEqual(visual.geometry.cylinder.radius, 1)
        self.assertEqual(visual.geometry.sphere.radius, 2)

        self.assertIsInstance(visual.geometry.mesh, Mesh)

    def test_urdf_load(self):
        # robots built from URDF/xacro files, by the current loader
        rtb.models.URDF.UR5()
        rtb.models.URDF.Panda()
        rtb.models.URDF.wx250s()

    def test_urdf_collisions(self):
        urdf = panda_urdf()
        collision = urdf.links[0].collisions[0]

        collision.name = "Lonk"
        self.assertEqual(collision.name, "Lonk")

        self.assertIsInstance(collision.origin, np.ndarray)

    def test_urdf_dynamics(self):
        urdf = panda_urdf()

        self.assertEqual(urdf.joints[0].limit.effort, 87.0)
        self.assertEqual(urdf.joints[0].limit.velocity, 2.175)


if __name__ == "__main__":  # pragma nocover
    unittest.main()
