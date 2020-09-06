#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import ropy as rp
import spatialmath as sm
import unittest


class TestLink(unittest.TestCase):

    def test_link(self):
        rp.Link()

    def test_qlim(self):
        l0 = rp.Link(qlim=[-1, 1])

        self.assertEqual(l0.islimit(-0.9), False)
        self.assertEqual(l0.islimit(-1.9), True)
        self.assertEqual(l0.islimit(2.9), True)

    def test_Tc(self):
        l0 = rp.Link(Tc=1)
        l1 = rp.Link(Tc=[1])
        l2 = rp.Link(Tc=[1, 2])

        Tc0 = np.array([1, -1])
        Tc1 = np.array([1, -1])
        Tc2 = np.array([1, 2])

        nt.assert_array_almost_equal(l0.Tc, Tc0)
        nt.assert_array_almost_equal(l1.Tc, Tc1)
        nt.assert_array_almost_equal(l2.Tc, Tc2)

    def test_I(self):
        l0 = rp.Link(I=[1, 2, 3])
        l1 = rp.Link(I=[0, 1, 2, 3, 4, 5])
        l2 = rp.Link(I=np.eye(3))

        I0 = np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])

        I1 = np.array([
            [0, 3, 5],
            [3, 1, 4],
            [5, 4, 2]
        ])

        I2 = np.eye(3)

        nt.assert_array_almost_equal(l0.I, I0)
        nt.assert_array_almost_equal(l1.I, I1)
        nt.assert_array_almost_equal(l2.I, I2)

    def test_A(self):
        l0 = rp.Link(sigma=0)
        l1 = rp.Link(sigma=1)
        l2 = rp.Link(sigma=0, mdh=0)
        l3 = rp.Link(sigma=1, mdh=1)
        l4 = rp.Link(flip=True)

        T0 = sm.SE3(np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]))

        T1 = sm.SE3(np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, np.pi],
            [0, 0, 0, 1]
        ]))

        nt.assert_array_almost_equal(l0.A(np.pi).A, T0.A)
        nt.assert_array_almost_equal(l1.A(np.pi).A, T1.A)
        nt.assert_array_almost_equal(l2.A(np.pi).A, T0.A)
        nt.assert_array_almost_equal(l3.A(np.pi).A, T1.A)
        nt.assert_array_almost_equal(l4.A(np.pi).A, T0.A)

    def test_friction(self):
        l0 = rp.Revolute(d=2, Tc=[2, -1], B=3, G=2)

        tau = -124
        tau2 = 122

        nt.assert_almost_equal(l0.friction(10), tau)
        nt.assert_almost_equal(l0.friction(-10), tau2)

    def test_nofriction(self):
        l0 = rp.Link(Tc=2, B=3)
        l1 = rp.Link(Tc=2, B=3)
        l2 = rp.Link(Tc=2, B=3)
        l3 = rp.Link(Tc=2, B=3)

        n0 = l1.nofriction()
        n1 = l2.nofriction(viscous=True)
        n2 = l3.nofriction(coulomb=False)

        nt.assert_array_almost_equal(n0.B, l0.B)
        nt.assert_array_almost_equal(n0.Tc, [0, 0])

        nt.assert_array_almost_equal(n1.B, 0)
        nt.assert_array_almost_equal(n1.Tc, [0, 0])

        nt.assert_array_almost_equal(n2.B, l0.B)
        nt.assert_array_almost_equal(n2.Tc, l0.Tc)

    def test_sigma(self):
        l0 = rp.Link(sigma=0)
        l1 = rp.Link(sigma=1)

        self.assertEqual(l0.isrevolute(), True)
        self.assertEqual(l0.isprismatic(), False)
        self.assertEqual(l1.isrevolute(), False)
        self.assertEqual(l1.isprismatic(), True)

    def test_add(self):
        l0 = rp.Link()
        l1 = rp.Link()

        self.assertIsInstance(l0 + l1, rp.SerialLink)
        self.assertRaises(TypeError, l0.__add__, 1)

    def test_properties(self):
        l0 = rp.Link()

        self.assertEqual(l0.m, 0.0)
        nt.assert_array_almost_equal(l0.r, np.zeros((3, 1)))
        self.assertEqual(l0.Jm, 0.0)

    def test_str(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()

        s0 = l0.__str__()
        s1 = l1.__repr__()

        self.assertEqual(
            s0, "Prismatic  theta= 0.00  d= 0.00  a= 0.00  "
                "alpha= 0.00  offset= 0.00")
        self.assertEqual(
            s1, "Revolute   theta= 0.00  d= 0.00  a= 0.00  "
                "alpha= 0.00  offset= 0.00")

    def test_dyn(self):
        puma = rp.Puma560()

        s0 = puma.links[0].dyn()

        self.assertEqual(
            s0,
            "m     =  0.00 \n"
            "r     =  0.00 0.00 0.00 \n"
            "        | 0.00 0.00 0.00 | \n"
            "I     = | 0.00 0.35 0.00 | \n"
            "        | 0.00 0.00 0.00 | \n"
            "Jm    =  0.00 \n"
            "B     =  0.00 \n"
            "Tc    =  0.40(+) -0.43(-) \n"
            "G     =  -62.61 \n"
            "qlim  =  -2.79 to 2.79")

    def test_revolute(self):
        l0 = rp.Revolute()

        self.assertEqual(l0.sigma, 0)
        with self.assertRaises(ValueError):
            l0.theta = 1

    def test_prismatic(self):
        l0 = rp.Prismatic()

        self.assertEqual(l0.sigma, 1)
        with self.assertRaises(ValueError):
            l0.d = 1

    def test_setB(self):
        l0 = rp.Prismatic()

        with self.assertRaises(TypeError):
            l0.B = [1, 2]
