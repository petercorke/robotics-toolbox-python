#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import ropy as rp
import spatialmath.base as sm
import unittest


class TestLink(unittest.TestCase):

    def test_link(self):
        l0 = rp.Link()

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
