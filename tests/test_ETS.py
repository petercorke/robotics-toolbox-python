#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import roboticstoolbox as rtb
import numpy as np

# import spatialmath.base as sm
from spatialmath import SE3
import unittest
import sympy


class TestETS(unittest.TestCase):
    def test_bad_arg(self):
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)

        with self.assertRaises(TypeError):
            rtb.ETS([rx, ry, 1.0])

    def test_str(self):
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)
        rz = rtb.ET.Rz(1.543)
        a = rtb.ET.Rx()
        b = rtb.ET.Ry()
        c = rtb.ET.Rz()
        d = rtb.ET.tx(1.0)
        e = rtb.ET.SE3(SE3.Rx(1.0))
        ets = rx * ry * rz * d
        ets2 = rx * a * b * c
        ets3 = d * e

        self.assertEqual(str(ets), "Rx(88.41°) ⊕ Ry(88.41°) ⊕ Rz(88.41°) ⊕ tx(1)")
        self.assertEqual(str(ets2), "Rx(88.41°) ⊕ Rx(q0) ⊕ Ry(q1) ⊕ Rz(q2)")
        self.assertEqual(str(ets3), "tx(1) ⊕ SE3(rpy: 57.30°, -0.00°, 0.00°)")
        self.assertEqual(
            ets2.__str__(q="θ{0}"), "Rx(88.41°) ⊕ Rx(θ0) ⊕ Ry(θ1) ⊕ Rz(θ2)"
        )

    def test_str_jindex(self):
        rx = rtb.ET.Rx(1.543)
        a = rtb.ET.Rx(jindex=2)
        b = rtb.ET.Ry(jindex=5)
        c = rtb.ET.Rz(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "Rx(88.41°) ⊕ Rx(q2) ⊕ Ry(q5) ⊕ Rz(q7)")

    def test_str_flip(self):
        rx = rtb.ET.Rx(1.543)
        a = rtb.ET.Rx(jindex=2, flip=True)
        b = rtb.ET.Ry(jindex=5)
        c = rtb.ET.Rz(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "Rx(88.41°) ⊕ Rx(-q2) ⊕ Ry(q5) ⊕ Rz(q7)")

    def test_str_sym(self):
        x = sympy.Symbol("x")
        rx = rtb.ET.Rx(x)
        a = rtb.ET.Rx(jindex=2)
        b = rtb.ET.Ry(jindex=5)
        c = rtb.ET.Rz(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "Rx(x) ⊕ Rx(q2) ⊕ Ry(q5) ⊕ Rz(q7)")

    def ets_mul(self):
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)
        rz = rtb.ET.Rz(1.543)
        a = rtb.ET.Rx()
        b = rtb.ET.Ry()

        ets1 = rx * ry
        ets2 = a * b

        self.assertIsInstance(ets1 * ets2, rtb.ETS)
        self.assertIsInstance(ets1 * rz, rtb.ETS)
        self.assertIsInstance(rz * ets1, rtb.ETS)
        self.assertIsInstance(ets1 + ets2, rtb.ETS)

        ets2 *= rz
        self.assertIsInstance(ets2, rtb.ETS)

    def test_n(self):
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)
        rz = rtb.ET.Rz(1.543)
        a = rtb.ET.Rx()
        b = rtb.ET.Ry()

        ets1 = rx * ry
        ets2 = a * b
        ets3 = a * b * ry

        self.assertEqual(ets1.n, 0)
        self.assertEqual(ets2.n, 2)
        self.assertEqual(ets3.n, 2)
        self.assertEqual(ets1.structure, "")
        self.assertEqual(ets2.structure, "RR")

    def test_fkine(self):
        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)
        rz = rtb.ET.Rz(1.543)
        tx = rtb.ET.tx(1.543)
        ty = rtb.ET.ty(1.543)
        tz = rtb.ET.tz(1.543)
        a = rtb.ET.Rx(jindex=0)
        b = rtb.ET.Ry(jindex=1)
        c = rtb.ET.Rz(jindex=2)
        d = rtb.ET.tx(jindex=3)
        e = rtb.ET.ty(jindex=4)
        f = rtb.ET.tz(jindex=5)

        r = rx * ry * rz * tx * ty * tz * a * b * c * d * e * f

        ans = (
            SE3.Rx(1.543)
            * SE3.Ry(1.543)
            * SE3.Rz(1.543)
            * SE3.Tx(1.543)
            * SE3.Ty(1.543)
            * SE3.Tz(1.543)
            * SE3.Rx(q[0])
            * SE3.Ry(q[1])
            * SE3.Rz(q[2])
            * SE3.Tx(q[3])
            * SE3.Ty(q[4])
            * SE3.Tz(q[5])
        )

        nt.assert_almost_equal(r.fkine(q), ans.A)

    def test_fkine_sym(self):
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        z = sympy.Symbol("z")

        q = np.array([y, z])
        qt = np.array([[1.0, y], [z, y], [x, 2.0]])
        a = rtb.ET.Rx(x)
        b = rtb.ET.Ry(jindex=0)
        c = rtb.ET.tz(jindex=1)

        r = a * b * c

        ans1 = SE3.Rx(x) * SE3.Ry(q[0]) * SE3.Tz(q[1])
        ans2 = SE3.Rx(x) * SE3.Ry(1.0) * SE3.Tz(y)
        ans3 = SE3.Rx(x) * SE3.Ry(z) * SE3.Tz(y)
        ans4 = SE3.Rx(x) * SE3.Ry(x) * SE3.Tz(2.0)

        fk_traj = r.fkine(qt)

        nt.assert_almost_equal(r.fkine(q), ans1.A)
        nt.assert_almost_equal(fk_traj[0], ans2.A)
        nt.assert_almost_equal(fk_traj[1], ans3.A)
        nt.assert_almost_equal(fk_traj[2], ans4.A)

        base = SE3.Rx(1.0)
        tool = SE3.Tz(0.5)
        ans5 = base * ans1

        r2 = rtb.ETS([rtb.ET.Rx(jindex=0)])

        nt.assert_almost_equal(r.fkine(q, base=base), ans5.A)
        nt.assert_almost_equal(r.fkine(q, base=base.A), ans5.A)  # type: ignore

        q2 = [y]
        ans6 = SE3.Rx(y) * tool
        nt.assert_almost_equal(r2.fkine(q2, tool=tool), ans6)  # type: ignore
        nt.assert_almost_equal(r2.fkine(q2, tool=tool.A), ans6)  # type: ignore


if __name__ == "__main__":

    unittest.main()
