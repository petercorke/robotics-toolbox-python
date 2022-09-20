#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rp
import spatialmath as sm
import unittest


class Testtools(unittest.TestCase):
    def test_null(self):

        a0 = np.array([1, 2, 3])

        a1 = np.array([[1, 2, 3], [4, 5, 6]])

        a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        ans0 = np.array([[-0.5345, -0.8018], [0.7745, -0.3382], [-0.3382, 0.4927]])

        ans1 = np.array([[0.4082], [-0.8165], [0.4082]])

        ans2 = np.array([[-0.4082], [0.8165], [-0.4082]])

        r0 = rp.null(a0)
        r1 = rp.null(a1)
        r2 = rp.null(a2)

        nt.assert_array_almost_equal(np.abs(r0), np.abs(ans0), decimal=4)
        nt.assert_array_almost_equal(np.abs(r1), np.abs(ans1), decimal=4)
        nt.assert_array_almost_equal(np.abs(r2), np.abs(ans2), decimal=4)

    def test_p_servo_rpy(self):
        a = sm.SE3()
        b = sm.SE3.Rx(0.7) * sm.SE3.Tx(1)
        c = sm.SE3.Tz(0.59)

        v0, arrived0 = rp.p_servo(a, b, method="rpy")
        v1, _ = rp.p_servo(a.A, b.A, method="rpy")
        _, arrived1 = rp.p_servo(a, c, threshold=0.6, method="rpy")

        ans = np.array([1, 0, 0, 0.7, -0, 0])

        nt.assert_array_almost_equal(v0, ans, decimal=4)
        nt.assert_array_almost_equal(v1, ans, decimal=4)

        self.assertFalse(arrived0)
        self.assertTrue(arrived1)

    def test_p_servo_angle_axis(self):
        a = sm.SE3()
        b = sm.SE3.Rx(0.7) * sm.SE3.Tx(1)
        c = sm.SE3.Tz(0.59)

        v0, arrived0 = rp.p_servo(a, b, method="angle-axis")
        v1, _ = rp.p_servo(a.A, b.A)
        _, arrived1 = rp.p_servo(a, c, threshold=0.6)

        ans = np.array([1, 0, 0, 0.7, -0, 0])

        nt.assert_array_almost_equal(v0, ans, decimal=4)
        nt.assert_array_almost_equal(v1, ans, decimal=4)

        self.assertFalse(arrived0)
        self.assertTrue(arrived1)

    def test_jsingu(self):
        r = rp.models.Puma560()
        J = r.jacob0(r.qz)

        rp.jsingu(J)


if __name__ == "__main__":  # pragma nocover
    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])
