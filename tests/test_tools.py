#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import unittest

import numpy as np
import numpy.testing as nt
import spatialmath as sm
import sympy

import roboticstoolbox as rtb


class Testtools(unittest.TestCase):
    def test_trchain(self):
        T, tokens = rtb.trchain(
            "Tx(a) Rx(q1) Ry(45) Tz(2)",
            [90],
            "deg",
            variables={"a": 1},
            return_tokens=True,
        )
        expected = (
            sm.SE3.Tx(1)
            * sm.SE3.Rx(90, unit="deg")
            * sm.SE3.Ry(45, unit="deg")
            * sm.SE3.Tz(2)
        )
        nt.assert_allclose(T, expected.A)
        nt.assert_allclose(rtb.trchain(tokens, [90], unit="deg", variables={"a": 1}), T)

        q1, a = sympy.symbols("q1 a", real=True)
        symbolic = rtb.trchain("Rz(q1) Tx(a)", [q1], variables={"a": a})
        self.assertEqual(sympy.simplify(symbolic[0, 3] - a * sympy.cos(q1)), 0)

        with self.assertRaises(ValueError):
            rtb.trchain("Tx(__import__('os'))")

    def test_trchain2(self):
        T = rtb.trchain2(
            "R(theta1 - pi / 2) Tx(a) Rz(theta2) Ty(2)",
            [np.pi / 2, np.pi / 4],
            qvar="theta",
            variables={"a": 1},
        )
        expected = sm.SE2.Rot(0) * sm.SE2.Tx(1) * sm.SE2.Rot(np.pi / 4) * sm.SE2.Ty(2)
        nt.assert_allclose(T, expected.A)

    def test_null(self):

        a0 = np.array([1, 2, 3])

        a1 = np.array([[1, 2, 3], [4, 5, 6]])

        a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        ans0 = np.array([[-0.5345, -0.8018], [0.7745, -0.3382], [-0.3382, 0.4927]])

        ans1 = np.array([[0.4082], [-0.8165], [0.4082]])

        ans2 = np.array([[-0.4082], [0.8165], [-0.4082]])

        r0 = rtb.null(a0)
        r1 = rtb.null(a1)
        r2 = rtb.null(a2)

        nt.assert_array_almost_equal(np.abs(r0), np.abs(ans0), decimal=4)
        nt.assert_array_almost_equal(np.abs(r1), np.abs(ans1), decimal=4)
        nt.assert_array_almost_equal(np.abs(r2), np.abs(ans2), decimal=4)

    def test_p_servo_rpy(self):
        a = sm.SE3()
        b = sm.SE3.Rx(0.7) * sm.SE3.Tx(1)
        c = sm.SE3.Tz(0.59)

        v0, arrived0 = rtb.p_servo(a, b, method="rpy")
        v1, _ = rtb.p_servo(a.A, b.A, method="rpy")
        _, arrived1 = rtb.p_servo(a, c, threshold=0.6, method="rpy")

        ans = np.array([1, 0, 0, 0.7, -0, 0])

        nt.assert_array_almost_equal(v0, ans, decimal=4)
        nt.assert_array_almost_equal(v1, ans, decimal=4)

        self.assertFalse(arrived0)
        self.assertTrue(arrived1)

    def test_p_servo_angle_axis(self):
        a = sm.SE3()
        b = sm.SE3.Rx(0.7) * sm.SE3.Tx(1)
        c = sm.SE3.Tz(0.59)

        v0, arrived0 = rtb.p_servo(a, b, method="angle-axis")
        v1, _ = rtb.p_servo(a.A, b.A)
        _, arrived1 = rtb.p_servo(a, c, threshold=0.6)

        ans = np.array([1, 0, 0, 0.7, -0, 0])

        nt.assert_array_almost_equal(v0, ans, decimal=4)
        nt.assert_array_almost_equal(v1, ans, decimal=4)

        self.assertFalse(arrived0)
        self.assertTrue(arrived1)

    def test_jsingu(self):
        r = rtb.models.Puma560()
        J = r.jacob0(r.qz)

        rtb.jsingu(J)

    def test_c_angle_axis(self):
        n = 100

        coord = (np.random.random((n, 6)) - 1.0) * 3.0
        coord2 = (np.random.random((n, 6)) - 1.0) * 3.0

        for co, co2 in zip(coord, coord2):
            Te = (
                sm.SE3.Trans(co[:3])
                * sm.SE3.Rx(co[3])
                * sm.SE3.Ry(co[4])
                * sm.SE3.Rz(co[5])
            ).A
            Tep = (
                sm.SE3.Trans(co2[:3])
                * sm.SE3.Rx(co2[3])
                * sm.SE3.Ry(co2[4])
                * sm.SE3.Rz(co2[5])
            ).A

            e1 = rtb.angle_axis(Te, Tep)
            e2 = rtb.angle_axis_python(Te, Tep)

            nt.assert_allclose(e1, e2)


if __name__ == "__main__":  # pragma nocover
    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])
