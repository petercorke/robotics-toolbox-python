#!/usr/bin/env python3
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as gm
import unittest
import spatialmath as sm
from roboticstoolbox.robot.Link import BaseLink


class TestLink(unittest.TestCase):
    def test_str_et(self):
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)
        tz = rtb.ET.tz(1)

        l0 = rtb.Link(rx * ry * tz)

        ans = 'Link("", Rx(88.41°) ⊕ Ry(88.41°) ⊕ tz(1))'

        self.assertEqual(str(l0), ans)

    def test_init(self):
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)
        tz = rtb.ET.tz()
        ty = rtb.ET.ty()

        with self.assertRaises(ValueError):
            rtb.Link(rx * ry * tz * ty)

    def test_init_fail(self):
        rx = rtb.ET.Rx(1.543)
        ty = rtb.ET.ty()

        with self.assertRaises(TypeError):
            rtb.Link([rx, ty])  # type: ignore

    def test_A(self):
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)
        tz = rtb.ET.tz(1)

        l0 = rtb.Link(rx * ry * tz)

        ans = sm.SE3.Rx(1.543) * sm.SE3.Ry(1.543) * sm.SE3.Tz(1)

        nt.assert_array_almost_equal(l0.A().A, ans.A)

    def test_A2(self):
        rx = rtb.ET.Rx(np.pi)
        ry = rtb.ET.Ry(np.pi)
        tz = rtb.ET.tz()

        l0 = rtb.Link(rx * ry * tz)

        ans = sm.SE3.Rx(np.pi) * sm.SE3.Ry(np.pi) * sm.SE3.Tz(1.2)

        nt.assert_array_almost_equal(l0.A(1.2).A, ans.A)
        l0.A()

    def test_qlim(self):
        l0 = rtb.Link(rtb.ET.Rx())
        l0.qlim = [-1, 1]

        print(l0.qlim)

        self.assertEqual(l0.islimit(-0.9), False)
        self.assertEqual(l0.islimit(-1.9), True)
        self.assertEqual(l0.islimit(2.9), True)

    def test_Tc(self):
        l0 = rtb.Link(Tc=1)
        l1 = rtb.Link(Tc=[1])
        l2 = rtb.Link(Tc=[1, 2])

        Tc0 = np.array([1, -1])
        Tc1 = np.array([1, -1])
        Tc2 = np.array([1, 2])

        nt.assert_array_almost_equal(l0.Tc, Tc0)
        nt.assert_array_almost_equal(l1.Tc, Tc1)
        nt.assert_array_almost_equal(l2.Tc, Tc2)

    def test_B(self):
        l0 = rtb.Link(B=1.0)
        l1 = rtb.Link(B=None)

        nt.assert_array_almost_equal(l0.B, 1.0)
        nt.assert_array_almost_equal(l1.B, 0.0)

    def test_I(self):
        l0 = rtb.Link(I=[1, 2, 3])
        l1 = rtb.Link(I=[0, 1, 2, 3, 4, 5])
        l2 = rtb.Link(I=np.eye(3))

        I0 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        I1 = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]])

        I2 = np.eye(3)

        nt.assert_array_almost_equal(l0.I, I0)
        nt.assert_array_almost_equal(l1.I, I1)
        nt.assert_array_almost_equal(l2.I, I2)

    def test_friction(self):
        l0 = rtb.Link(Tc=[2, -1], B=3, G=2)

        tau = -124
        tau2 = 122

        nt.assert_almost_equal(l0.friction(10), tau)
        nt.assert_almost_equal(l0.friction(-10), tau2)

    def test_nofriction(self):
        l0 = rtb.Link(Tc=2, B=3)
        l1 = rtb.Link(Tc=2, B=3)
        l2 = rtb.Link(Tc=2, B=3)
        l3 = rtb.Link(Tc=2, B=3)

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
        l0 = rtb.Link(rtb.ET.Rx(), Tc=[0.4, -0.43], G=-62.61, I=np.diag([0, 0.35, 0]))
        l0.qlim = [-2.79, 2.79]

        s0 = l0.dyn()
        print(s0)

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
qlim  =      -2.8 to      2.8""",  # noqa
        )

    def test_properties(self):
        l0 = rtb.Link()

        self.assertEqual(l0.m, 0.0)
        nt.assert_array_almost_equal(l0.r, np.zeros(3))
        self.assertEqual(l0.Jm, 0.0)

    def test_fail_parent(self):
        with self.assertRaises(TypeError):
            rtb.Link(parent=1)

    def test_setB(self):
        l0 = rtb.Link()

        with self.assertRaises(TypeError):
            l0.B = [1, 2]  # type: ignore

    def test_collision(self):
        p = rtb.models.Panda()
        link = p.links[1]
        col = link.collision[0]

        self.assertIsInstance(col, gm.Shape)

        self.assertIsInstance(col._T, np.ndarray)

        col.radius = 2  # type: ignore
        self.assertEqual(col.radius, 2)  # type: ignore

        col.length = 2  # type: ignore
        self.assertEqual(col.length, 2)  # type: ignore

    # def test_collision_fail(self):
    #     l0 = rtb.Link()
    #     col = gm.Cuboid([1, 1, 1])
    #     l0.collision = col

    #     with self.assertRaises(TypeError):
    #         l0.collision = [1, 1, 1]  # type: ignore

    #     with self.assertRaises(TypeError):
    #         l0.collision = 1  # type: ignore

    # def test_geometry_fail(self):
    #     l0 = rtb.Link()
    #     col = gm.Cuboid([1, 1, 1])
    #     l0.geometry = col
    #     l0.geometry = [col, col]

    #     with self.assertRaises(TypeError):
    #         l0.geometry = [1, 1, 1]  # type: ignore

    #     with self.assertRaises(TypeError):
    #         l0.geometry = 1  # type: ignore

    def test_dist(self):
        s0 = gm.Cuboid([1, 1, 1], pose=sm.SE3(0, 0, 0))
        s1 = gm.Cuboid([1, 1, 1], pose=sm.SE3(3, 0, 0))
        p = rtb.models.Panda()
        link = p.links[3]

        d0, _, _ = link.closest_point(s0)
        d1, _, _ = link.closest_point(s1, 5)
        d2, _, _ = link.closest_point(s1)

        self.assertAlmostEqual(d0, -0.130999999)  # type: ignore
        self.assertAlmostEqual(d1, 2.44)  # type: ignore
        self.assertAlmostEqual(d2, None)  # type: ignore

    def test_collided(self):
        s0 = gm.Cuboid([1, 1, 1], pose=sm.SE3(0, 0, 0))
        s1 = gm.Cuboid([1, 1, 1], pose=sm.SE3(3, 0, 0))
        p = rtb.models.Panda()
        link = p.links[3]
        c0 = link.collided(s0)
        c1 = link.collided(s1)

        self.assertTrue(c0)
        self.assertFalse(c1)

    def test_init_ets2(self):
        e1 = rtb.ET2.R()
        link = BaseLink(e1)

        self.assertEqual(link.Ts, None)

    def test_get_ets(self):
        e1 = rtb.ETS(rtb.ET.Ry())
        link = rtb.Link(e1)

        self.assertEqual(link.ets, e1)

    def test_set_ets_fail(self):
        e1 = rtb.ETS(rtb.ET.Ry())
        e2 = rtb.ET.Ry() * rtb.ET.Rx(1.0)
        link = rtb.Link(e1)

        with self.assertRaises(ValueError):
            link.ets = e2

    def test_set_robot(self):
        e1 = rtb.ETS(rtb.ET.Ry())
        link = rtb.Link(e1)

        robot = rtb.models.Panda()

        link.robot = robot

        self.assertEqual(link.robot, robot)

    def test_set_qlim_fail(self):
        e1 = rtb.ETS(rtb.ET.Ry(1.0))
        link = rtb.Link(e1)

        with self.assertRaises(ValueError):
            link.qlim = [1.0, 2.0]

    def test_set_collision(self):
        e1 = rtb.ETS(rtb.ET.Ry())
        link = rtb.Link(e1)

        s1 = gm.Cuboid([1.0, 1.0, 1.0])

        link.collision = s1

        self.assertEqual(link.collision[0], s1)

    def test_set_collision2(self):
        e1 = rtb.ETS(rtb.ET.Ry())
        link = rtb.Link(e1)

        s1 = gm.Cuboid([1.0, 1.0, 1.0])

        sg = gm.SceneGroup()
        sg.append(s1)

        link.collision = sg

        self.assertEqual(link.collision[0], sg[0])

    def test_set_geometry(self):
        e1 = rtb.ETS(rtb.ET.Ry())
        link = rtb.Link(e1)

        s1 = gm.Cuboid([1.0, 1.0, 1.0])

        link.geometry = s1

        self.assertEqual(link.geometry[0], s1)

    def test_set_geometry2(self):
        e1 = rtb.ETS(rtb.ET.Ry())
        link = rtb.Link(e1)

        s1 = gm.Cuboid([1.0, 1.0, 1.0])

        sg = gm.SceneGroup()
        sg.append(s1)

        link.geometry = sg

        self.assertEqual(link.geometry[0], sg[0])

    def test_dyn2list(self):
        l1 = rtb.Link(I=[0, 1, 2, 3, 4, 5])

        s = l1._dyn2list()

        print(s)

        ans = [" 0", " 0,  0,  0", " 0,  1,  2,  3,  4,  5", " 0", " 0", " 0,  0", " 0"]

        self.assertEqual(s, ans)

    def test_init_fail4(self):
        with self.assertRaises(TypeError):
            rtb.Link(2.0)  # type: ignore

    def test_ets2_A(self):
        e1 = rtb.ETS2(rtb.ET2.R())
        link = rtb.Link2(e1)

        nt.assert_almost_equal(link.A(1.0).A, sm.SE2(0.0, 0.0, 1.0).A)

    def test_ets2_A2(self):
        e1 = rtb.ETS2(rtb.ET2.R(1.0))
        link = rtb.Link2(e1)

        nt.assert_almost_equal(link.A().A, sm.SE2(0.0, 0.0, 1.0).A)


if __name__ == "__main__":
    unittest.main()
