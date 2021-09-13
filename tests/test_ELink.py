#!/usr/bin/env python3
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rp
import spatialgeometry as gm
import unittest
import spatialmath as sm


class TestELink(unittest.TestCase):
    def test_str_ets(self):
        rx = rp.ETS.rx(1.543)
        ry = rp.ETS.ry(1.543)
        tz = rp.ETS.tz(1)

        l0 = rp.ELink(rx * ry * tz)

        ans = "ELink[Rx(88.41°) ⊕ Ry(88.41°) ⊕ tz(1)] "

        self.assertEqual(str(l0), ans)

    def test_init(self):
        rx = rp.ETS.rx(1.543)
        ry = rp.ETS.ry(1.543)
        tz = rp.ETS.tz()
        ty = rp.ETS.ty()

        with self.assertRaises(ValueError):
            rp.ELink(rx * ry * tz * ty)

    def test_init_fail(self):
        rx = rp.ETS.rx(1.543)
        ty = rp.ETS.ty()

        with self.assertRaises(TypeError):
            rp.ELink([rx, ty])

    def test_A(self):
        rx = rp.ETS.rx(1.543)
        ry = rp.ETS.ry(1.543)
        tz = rp.ETS.tz(1)

        l0 = rp.ELink(rx * ry * tz)

        ans = sm.SE3.Rx(1.543) * sm.SE3.Ry(1.543) * sm.SE3.Tz(1)

        nt.assert_array_almost_equal(l0.A().A, ans.A)

    def test_A2(self):
        rx = rp.ETS.rx(np.pi)
        ry = rp.ETS.ry(np.pi)
        tz = rp.ETS.tz()

        l0 = rp.ELink(rx * ry, tz)

        ans = sm.SE3.Rx(np.pi) * sm.SE3.Ry(np.pi) * sm.SE3.Tz(1.2)

        nt.assert_array_almost_equal(l0.A(1.2).A, ans.A)
        l0.A()

    def test_qlim(self):
        l0 = rp.ELink(qlim=[-1, 1])

        self.assertEqual(l0.islimit(-0.9), False)
        self.assertEqual(l0.islimit(-1.9), True)
        self.assertEqual(l0.islimit(2.9), True)

    def test_Tc(self):
        l0 = rp.ELink(Tc=1)
        l1 = rp.ELink(Tc=[1])
        l2 = rp.ELink(Tc=[1, 2])

        Tc0 = np.array([1, -1])
        Tc1 = np.array([1, -1])
        Tc2 = np.array([1, 2])

        nt.assert_array_almost_equal(l0.Tc, Tc0)
        nt.assert_array_almost_equal(l1.Tc, Tc1)
        nt.assert_array_almost_equal(l2.Tc, Tc2)

    def test_I(self):
        l0 = rp.ELink(I=[1, 2, 3])
        l1 = rp.ELink(I=[0, 1, 2, 3, 4, 5])
        l2 = rp.ELink(I=np.eye(3))

        I0 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        I1 = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]])

        I2 = np.eye(3)

        nt.assert_array_almost_equal(l0.I, I0)
        nt.assert_array_almost_equal(l1.I, I1)
        nt.assert_array_almost_equal(l2.I, I2)

    def test_friction(self):
        l0 = rp.ELink(Tc=[2, -1], B=3, G=2)

        tau = -124
        tau2 = 122

        nt.assert_almost_equal(l0.friction(10), tau)
        nt.assert_almost_equal(l0.friction(-10), tau2)

    def test_nofriction(self):
        l0 = rp.ELink(Tc=2, B=3)
        l1 = rp.ELink(Tc=2, B=3)
        l2 = rp.ELink(Tc=2, B=3)
        l3 = rp.ELink(Tc=2, B=3)

        n0 = l1.nofriction()
        n1 = l2.nofriction(viscous=True)
        n2 = l3.nofriction(coulomb=False)

        nt.assert_array_almost_equal(n0.B, l0.B)
        nt.assert_array_almost_equal(n0.Tc, [0, 0])

        nt.assert_array_almost_equal(n1.B, 0)
        nt.assert_array_almost_equal(n1.Tc, [0, 0])

        nt.assert_array_almost_equal(n2.B, l0.B)
        nt.assert_array_almost_equal(n2.Tc, l0.Tc)

    def test_dyn(self):
        l0 = rp.ELink(
            Tc=[0.4, -0.43], G=-62.61, qlim=[-2.79, 2.79], I=np.diag([0, 0.35, 0])
        )

        s0 = l0.dyn()

        self.assertEqual(
            s0,
            """m     =         0 
r     =         0        0        0 
        |        0        0        0 | 
I     = |        0     0.35        0 | 
        |        0        0        0 | 
Jm    =         0 
B     =         0 
Tc    =       0.4(+)    -0.43(-) 
G     =       -63 
qlim  =      -2.8 to      2.8""",
        )

    def test_properties(self):
        l0 = rp.ELink()

        self.assertEqual(l0.m, 0.0)
        nt.assert_array_almost_equal(l0.r, np.zeros(3))
        self.assertEqual(l0.Jm, 0.0)

    def test_fail_parent(self):

        with self.assertRaises(TypeError):
            rp.ELink(parent=1)

    def test_setB(self):
        l0 = rp.ELink()

        with self.assertRaises(TypeError):
            l0.B = [1, 2]

    def test_collision(self):
        p = rp.models.Panda()
        link = p.links[1]
        col = link.collision[0]

        self.assertIsInstance(col, gm.Shape)

        self.assertIsInstance(col.base, sm.SE3)

        col.radius = 2
        self.assertEqual(col.radius, 2)

        col.length = 2
        self.assertEqual(col.length, 2)

    def test_collision_fail(self):
        l0 = rp.ELink()
        col = gm.Cuboid([1, 1, 1])
        l0.collision = col

        with self.assertRaises(TypeError):
            l0.collision = [1, 1, 1]

        with self.assertRaises(TypeError):
            l0.collision = 1

    def test_geometry_fail(self):
        l0 = rp.ELink()
        col = gm.Cuboid([1, 1, 1])
        l0.geometry = col
        l0.geometry = [col, col]

        with self.assertRaises(TypeError):
            l0.geometry = [1, 1, 1]

        with self.assertRaises(TypeError):
            l0.geometry = 1

    def test_dist(self):
        s0 = gm.Cuboid([1, 1, 1], base=sm.SE3(0, 0, 0))
        s1 = gm.Cuboid([1, 1, 1], base=sm.SE3(3, 0, 0))
        p = rp.models.Panda()
        link = p.links[3]

        d0, _, _ = link.closest_point(s0)
        d1, _, _ = link.closest_point(s1, 5)
        d2, _, _ = link.closest_point(s1)

        self.assertAlmostEqual(d0, -0.49)
        self.assertAlmostEqual(d1, 2.44)
        self.assertAlmostEqual(d2, None)

    def test_collided(self):
        s0 = gm.Cuboid([1, 1, 1], base=sm.SE3(0, 0, 0))
        s1 = gm.Cuboid([1, 1, 1], base=sm.SE3(3, 0, 0))
        p = rp.models.Panda()
        link = p.links[3]
        c0 = link.collided(s0)
        c1 = link.collided(s1)

        self.assertTrue(c0)
        self.assertFalse(c1)


if __name__ == "__main__":

    unittest.main()
