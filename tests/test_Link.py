#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rp
import spatialmath as sm
import unittest


class TestDHLink(unittest.TestCase):
    def test_link(self):
        rp.DHLink()

    def test_super_copy(self):
        l0 = rp.Link()

        with self.assertRaises(DeprecationWarning):
            l0._copy()

    def test_qlim(self):
        l0 = rp.DHLink(qlim=[-1, 1])

        self.assertEqual(l0.islimit(-0.9), False)
        self.assertEqual(l0.islimit(-1.9), True)
        self.assertEqual(l0.islimit(2.9), True)

    def test_Tc(self):
        l0 = rp.DHLink(Tc=1)
        l1 = rp.DHLink(Tc=[1])
        l2 = rp.DHLink(Tc=[1, 2])

        Tc0 = np.array([1, -1])
        Tc1 = np.array([1, -1])
        Tc2 = np.array([1, 2])

        nt.assert_array_almost_equal(l0.Tc, Tc0)
        nt.assert_array_almost_equal(l1.Tc, Tc1)
        nt.assert_array_almost_equal(l2.Tc, Tc2)

    def test_I(self):
        l0 = rp.DHLink(I=[1, 2, 3])
        l1 = rp.DHLink(I=[0, 1, 2, 3, 4, 5])
        l2 = rp.DHLink(I=np.eye(3))

        I0 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

        I1 = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]])

        I2 = np.eye(3)

        nt.assert_array_almost_equal(l0.I, I0)
        nt.assert_array_almost_equal(l1.I, I1)
        nt.assert_array_almost_equal(l2.I, I2)

    def test_A(self):
        l0 = rp.RevoluteMDH()
        l1 = rp.PrismaticMDH()
        l2 = rp.RevoluteMDH(flip=True)

        T0 = sm.SE3(
            np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        )

        T1 = sm.SE3(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, np.pi], [0, 0, 0, 1]])
        )

        nt.assert_array_almost_equal(l0.A(np.pi).A, T0.A)
        nt.assert_array_almost_equal(l1.A(np.pi).A, T1.A)
        nt.assert_array_almost_equal(l2.A(np.pi).A, T0.A)

    def test_friction(self):
        l0 = rp.RevoluteMDH(d=2, Tc=[2, -1], B=3, G=2)

        tau = -124
        tau2 = 122

        nt.assert_almost_equal(l0.friction(10), tau)
        nt.assert_almost_equal(l0.friction(-10), tau2)

    def test_nofriction(self):
        l0 = rp.DHLink(Tc=2, B=3)
        l1 = rp.DHLink(Tc=2, B=3)
        l2 = rp.DHLink(Tc=2, B=3)
        l3 = rp.DHLink(Tc=2, B=3)

        n0 = l1.nofriction()
        n1 = l2.nofriction(viscous=True)
        n2 = l3.nofriction(coulomb=False)

        nt.assert_array_almost_equal(n0.B, l0.B)
        nt.assert_array_almost_equal(n0.Tc, [0, 0])

        nt.assert_array_almost_equal(n1.B, 0)
        nt.assert_array_almost_equal(n1.Tc, [0, 0])

        nt.assert_array_almost_equal(n2.B, l0.B)
        nt.assert_array_almost_equal(n2.Tc, l0.Tc)

    def test_add(self):
        l0 = rp.DHLink()
        l1 = rp.DHLink()

        self.assertIsInstance(l0 + l1, rp.DHRobot)
        self.assertRaises(TypeError, l0.__add__, 1)

    def test_properties(self):
        l0 = rp.DHLink()

        self.assertEqual(l0.m, 0.0)
        nt.assert_array_almost_equal(l0.r, np.zeros(3))
        self.assertEqual(l0.Jm, 0.0)

    def test_str(self):
        l0 = rp.PrismaticMDH()
        l1 = rp.RevoluteMDH()

        s0 = l0.__str__()
        s1 = l1.__str__()

        self.assertEqual(s0, "PrismaticMDH:  θ=0.0,  d=q,  a=0.0,  ⍺=0.0")
        self.assertEqual(s1, "RevoluteMDH:   θ=q,  d=0.0,  a=0.0,  ⍺=0.0")

    def test_dyn(self):
        puma = rp.models.DH.Puma560()

        s0 = puma.links[0].dyn()

        self.assertEqual(
            s0,
            r"""m     =         0 
r     =         0        0        0 
        |        0        0        0 | 
I     = |        0     0.35        0 | 
        |        0        0        0 | 
Jm    =    0.0002 
B     =    0.0015 
Tc    =       0.4(+)    -0.43(-) 
G     =       -63 
qlim  =      -2.8 to      2.8""",  # noqa
        )

        puma.links[0].dyn(indent=2)

    def test_revolute(self):
        l0 = rp.RevoluteMDH()

        self.assertEqual(l0.sigma, 0)

    def test_prismatic(self):
        l0 = rp.PrismaticMDH()

        self.assertEqual(l0.sigma, 1)

    # def test_setB(self):
    #     l0 = rp.PrismaticDH()

    #     with self.assertRaises(TypeError):
    #         l0.B = [1, 2]

    def test_robot(self):
        l0 = rp.RevoluteMDH()
        r = rp.DHRobot([l0])

        self.assertIs(l0._robot, r)

    def test_copy(self):

        l0 = rp.RevoluteMDH()
        r = rp.DHRobot([l0])
        l1 = l0.copy()
        l0.m = 4
        l0.r[1] = 5

        self.assertEqual(l1.m, 0)
        self.assertEqual(l1.r[1], 0)
        self.assertIs(l0._robot, r)
        self.assertIs(l1._robot, r)

    def test_I_new(self):  # noqa
        r = rp.models.DH.Puma560()

        with self.assertRaises(ValueError):
            I = np.eye(3)  # noqa
            I[1, 0] = 4
            r.links[1].I = I  # noqa

        with self.assertRaises(ValueError):
            I = np.zeros(9)  # noqa
            I[1] = 4
            r.links[1].I = I  # noqa

        with self.assertRaises(ValueError):
            r.links[1].I = np.zeros(8)  # noqa

    def test_qlim_none(self):
        l0 = rp.RevoluteMDH()
        self.assertFalse(l0.islimit(0.1))


if __name__ == "__main__":

    unittest.main()
