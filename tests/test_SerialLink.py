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

    def test_seriallink(self):
        l0 = rp.Link()
        r0 = rp.SerialLink([l0])

    def test_isprismatic(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic()
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])

        ans = [True, False, True, False]

        self.assertEqual(r0.isprismatic(), ans)

    def test_isrevolute(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic()
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])

        ans = [False, True, False, True]

        self.assertEqual(r0.isrevolute(), ans)

    def test_todegrees(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic()
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        q = np.array([np.pi, np.pi, np.pi, np.pi/2.0])

        ans = np.array([np.pi, 180, np.pi, 90])

        nt.assert_array_almost_equal(r0.todegrees(q), ans)
        nt.assert_array_almost_equal(r0.todegrees(), np.zeros(4))

    def test_toradians(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic()
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        q = np.array([np.pi, 180, np.pi, 90])

        ans = np.array([np.pi, np.pi, np.pi, np.pi/2.0])

        nt.assert_array_almost_equal(r0.toradians(q), ans)

    def test_d(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute(d=2.0)
        l2 = rp.Prismatic()
        l3 = rp.Revolute(d=4.0)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        ans = [0.0, 2.0, 0.0, 4.0]

        self.assertEqual(r0.d, ans)

    def test_a(self):
        l0 = rp.Prismatic(a=1.0)
        l1 = rp.Revolute(a=2.0)
        l2 = rp.Prismatic(a=3.0)
        l3 = rp.Revolute(a=4.0)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        ans = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(r0.a, ans)

    def test_theta(self):
        l0 = rp.Prismatic(theta=1.0)
        l1 = rp.Revolute()
        l2 = rp.Prismatic(theta=3.0)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        ans = [1.0, 0.0, 3.0, 0.0]

        self.assertEqual(r0.theta, ans)

    def test_r(self):
        r = np.array([[1], [2], [3]])
        l0 = rp.Prismatic(r=r)
        l1 = rp.Revolute(r=r)
        l2 = rp.Prismatic(r=r)
        l3 = rp.Revolute(r=r)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        r1 = rp.SerialLink([l0])
        ans = np.c_[r, r, r, r]

        nt.assert_array_almost_equal(r0.r, ans)
        nt.assert_array_almost_equal(r1.r, r)

    def test_offset(self):
        l0 = rp.Prismatic(offset=1.0)
        l1 = rp.Revolute(offset=2.0)
        l2 = rp.Prismatic(offset=3.0)
        l3 = rp.Revolute(offset=4.0)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        ans = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(r0.offset, ans)

    def test_qlim(self):
        qlim = [-1, 1]
        l0 = rp.Prismatic(qlim=qlim)
        l1 = rp.Revolute(qlim=qlim)
        l2 = rp.Prismatic(qlim=qlim)
        l3 = rp.Revolute(qlim=qlim)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        r1 = rp.SerialLink([l0])
        ans = np.c_[qlim, qlim, qlim, qlim]

        nt.assert_array_almost_equal(r0.qlim, ans)
        nt.assert_array_almost_equal(r1.qlim, qlim)

    def test_fkine(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic(theta=2.0)
        l3 = rp.Revolute()

        q = np.array([1, 2, 3, 4])

        T1 = np.array([
            [-0.14550003, -0.98935825, 0, 0],
            [0.98935825, -0.14550003, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ])

        r0 = rp.SerialLink([l0, l1, l2, l3])

        nt.assert_array_almost_equal(r0.fkine(q).A, T1)



