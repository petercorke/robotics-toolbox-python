#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

# import numpy.testing as nt
# import numpy as np
import roboticstoolbox as rp
import unittest
import spatialmath as sm


class TestShape(unittest.TestCase):

    def test_init(self):
        rp.Box([1, 1, 1], sm.SE3(0, 0, 0))
        rp.Cylinder(1, 1, sm.SE3(2, 0, 0))
        rp.Sphere(1, sm.SE3(4, 0, 0))

    def test_closest(self):
        s0 = rp.Box([1, 1, 1], sm.SE3(0, 0, 0))
        s1 = rp.Cylinder(1, 1, sm.SE3(2, 0, 0))
        s2 = rp.Sphere(1, sm.SE3(4, 0, 0))

        d0, _, _ = s0.closest_point(s1, 10)
        d1, _, _ = s1.closest_point(s2, 10)
        d2, _, _ = s2.closest_point(s0, 10)
        d3, _, _ = s2.closest_point(s0)

        self.assertAlmostEqual(d0, 0.5)
        self.assertAlmostEqual(d1, 4.698463840213662e-13)
        self.assertAlmostEqual(d2, 2.5)
        self.assertAlmostEqual(d3, None)

    def test_to_dict(self):
        s1 = rp.Cylinder(1, 1)

        ans = {
            'stype': 'cylinder',
            'scale': [1.0, 1.0, 1.0],
            'filename': None,
            'radius': 1.0,
            'length': 1.0,
            't': [0.0, 0.0, 0.0],
            'q': [0.7071067811865476, 0.7071067811865475, 0.0, 0.0],
            'v': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

        self.assertEqual(s1.to_dict(), ans)

    def test_fk_dict(self):
        s1 = rp.Cylinder(1, 1)

        ans = {
            't': [0.0, 0.0, 0.0],
            'q': [0.7071067811865476, 0.7071067811865475, 0.0, 0.0]}

        self.assertEqual(s1.fk_dict(), ans)

    def test_fk_dict2(self):
        s1 = rp.Sphere(1)

        ans = {
            't': [0.0, 0.0, 0.0], 'q': [1, 0, 0, 0]}

        self.assertEqual(s1.fk_dict(), ans)

    def test_mesh(self):
        ur = rp.models.UR5()
        ur.links[1].collision[0].closest_point(ur.links[2].collision[0])

    def test_collision(self):
        s0 = rp.Box([1, 1, 1], sm.SE3(0, 0, 0))
        s1 = rp.Box([1, 1, 1], sm.SE3(0.5, 0, 0))
        s2 = rp.Box([1, 1, 1], sm.SE3(3, 0, 0))

        c0 = s0.collided(s1)
        c1 = s0.collided(s2)

        self.assertTrue(c0)
        self.assertFalse(c1)
