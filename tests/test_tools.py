#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import ropy as rp
import spatialmath as sm
import unittest


class Testtools(unittest.TestCase):

    def test_null(self):

        a0 = np.array([1, 2, 3])

        a1 = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])

        a2 = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        ans0 = np.array([
            [-0.5345, -0.8018],
            [0.7745, -0.3382],
            [-0.3382,  0.4927]
        ])

        ans1 = np.array([
            [0.4082],
            [-0.8165],
            [0.4082]
        ])

        ans2 = np.array([
            [-0.4082],
            [0.8165],
            [-0.4082]
        ])

        r0 = rp.null(a0)
        r1 = rp.null(a1)
        r2 = rp.null(a2)

        nt.assert_array_almost_equal(r0, ans0, decimal=4)
        nt.assert_array_almost_equal(r1, ans1, decimal=4)
        nt.assert_array_almost_equal(r2, ans2, decimal=4)
