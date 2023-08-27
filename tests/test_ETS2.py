
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
from spatialmath import SE2
import unittest
import sympy


class TestETS2(unittest.TestCase):
    def test_bad_arg(self):
        rx = rtb.ET2.R(1.543)
        ry = rtb.ET2.R(1.543)

        with self.assertRaises(TypeError):
            rtb.ETS2([rx, ry, 1.0])  # type: ignore

        with self.assertRaises(TypeError):
            rtb.ETS2(1.0)  # type: ignore

    def test_args(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET2.tx(0.333) * rtb.ET2.R(jindex=0)

        l1 = rtb.ET2.R(-90 * deg) * rtb.ET2.R(jindex=1)

        l2 = rtb.ET2.R(90 * deg) * rtb.ET2.tx(0.316) * rtb.ET2.R(jindex=2)

        l3 = rtb.ET2.tx(0.0825) * rtb.ET2.R(90 * deg) * rtb.ET2.R(jindex=3)

        l4 = (
            rtb.ET2.tx(-0.0825)
            * rtb.ET2.R(-90 * deg)
            * rtb.ET2.tx(0.384)
            * rtb.ET2.R(jindex=4)
        )

        l5 = rtb.ET2.R(90 * deg) * rtb.ET2.R(jindex=5)

        r1 = l0 + l1 + l2 + l3 + l3 + l4 + l5
        r2 = l0 * l1 * l2 * l3 * l3 * l4 * l5
        r3 = rtb.ETS2(l0 + l1 + l2 + l3 + l3 + l4 + l5)
        r4 = rtb.ETS2([l0, l1, l2, l3, l3, l4, l5])
        r5 = rtb.ETS2(
            [l0, l1, l2, l3, l3, l4, rtb.ET2.R(90 * deg), rtb.ET2.R(jindex=5)]
        )
        r6 = rtb.ETS2([r1])
        r7 = rtb.ETS2(rtb.ET2.R(1.0))

        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)
        self.assertEqual(r1, r4)
        self.assertEqual(r1, r5)
        self.assertEqual(r7[0], rtb.ET2.R(1.0))
        self.assertEqual(r1 + rtb.ETS2(), r2)

    def test_empty(self):
        r = rtb.ETS2()
        self.assertEqual(r.n, 0)
        self.assertEqual(r.m, 0)

    def test_str(self):
        rx = rtb.ET2.R(1.543)
        ry = rtb.ET2.R(1.543)
        rz = rtb.ET2.R(1.543)
        a = rtb.ET2.R()
        b = rtb.ET2.R()
        c = rtb.ET2.R()
        d = rtb.ET2.tx(1.0)
        e = rtb.ET2.SE2(SE2(1.0, unit="rad"))
        ets = rx * ry * rz * d
        ets2 = rx * a * b * c
        ets3 = d * e

        self.assertEqual(str(ets), "R(88.41°) ⊕ R(88.41°) ⊕ R(88.41°) ⊕ tx(1)")
        self.assertEqual(str(ets2), "R(88.41°) ⊕ R(q0) ⊕ R(q1) ⊕ R(q2)")
        self.assertEqual(str(ets3), "tx(1) ⊕ SE2(0, 0; 57.3°)")
        self.assertEqual(ets2.__str__(q="θ{0}"), "R(88.41°) ⊕ R(θ0) ⊕ R(θ1) ⊕ R(θ2)")

    def test_str_jindex(self):
        rx = rtb.ET2.R(1.543)
        a = rtb.ET2.R(jindex=2)
        b = rtb.ET2.R(jindex=5)
        c = rtb.ET2.R(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "R(88.41°) ⊕ R(q2) ⊕ R(q5) ⊕ R(q7)")

    def test_str_flip(self):
        rx = rtb.ET2.R(1.543)
        a = rtb.ET2.R(jindex=2, flip=True)
        b = rtb.ET2.R(jindex=5)
        c = rtb.ET2.R(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "R(88.41°) ⊕ R(-q2) ⊕ R(q5) ⊕ R(q7)")

    def test_str_sym(self):
        x = sympy.Symbol("x")
        rx = rtb.ET2.R(x)
        a = rtb.ET2.R(jindex=2)
        b = rtb.ET2.R(jindex=5)
        c = rtb.ET2.R(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "R(x) ⊕ R(q2) ⊕ R(q5) ⊕ R(q7)")

    def ets_mul(self):
        rx = rtb.ET2.R(1.543)
        ry = rtb.ET2.R(1.543)
        rz = rtb.ET2.R(1.543)
        a = rtb.ET2.R()
        b = rtb.ET2.R()

        ets1 = rx * ry
        ets2 = a * b

        self.assertIsInstance(ets1 * ets2, rtb.ETS)
        self.assertIsInstance(ets1 * rz, rtb.ETS)
        self.assertIsInstance(rz * ets1, rtb.ETS)
        self.assertIsInstance(ets1 + ets2, rtb.ETS)

        ets2 *= rz
        self.assertIsInstance(ets2, rtb.ETS)

    def test_n(self):
        rx = rtb.ET2.R(1.543)
        ry = rtb.ET2.R(1.543)
        rz = rtb.ET2.R(1.543)
        a = rtb.ET2.R()
        b = rtb.ET2.R()

        ets1 = rx * ry
        ets2 = a * b
        ets3 = a * b * ry

        self.assertEqual(ets1.n, 0)
        self.assertEqual(ets2.n, 2)
        self.assertEqual(ets3.n, 2)
        self.assertEqual(ets1.structure, "")
        self.assertEqual(ets2.structure, "RR")

    def test_fkine(self):
        q = np.array([0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        rx = rtb.ET2.R(1.543)
        ry = rtb.ET2.R(1.543)
        rz = rtb.ET2.R(1.543)
        tx = rtb.ET2.tx(1.543)
        ty = rtb.ET2.ty(1.543)
        tx = rtb.ET2.tx(1.543)
        a = rtb.ET2.R(jindex=0)
        b = rtb.ET2.R(jindex=1)
        c = rtb.ET2.R(jindex=2)
        d = rtb.ET2.tx(jindex=3)
        e = rtb.ET2.ty(jindex=4)
        f = rtb.ET2.tx(jindex=5)

        r = rx * ry * rz * tx * ty * tx * a * b * c * d * e * f

        ans = (
            SE2(1.543)
            * SE2(1.543)
            * SE2(1.543)
            * SE2.Tx(1.543)
            * SE2.Ty(1.543)
            * SE2.Tx(1.543)
            * SE2(q[0])
            * SE2(q[1])
            * SE2(q[2])
            * SE2.Tx(q[3])
            * SE2.Ty(q[4])
            * SE2.Tx(q[5])
        )

        nt.assert_almost_equal(r.fkine(q), ans.A)

    def test_fkine2(self):
        x = 1.0
        y = 2.0
        z = 3.0

        q = np.array([y, z])
        qt = np.array([[1.0, y], [z, y], [x, 2.0]])
        a = rtb.ET2.R(x)
        b = rtb.ET2.R(jindex=0)
        c = rtb.ET2.tx(jindex=1)

        r = a * b * c

        ans1 = SE2(x) * SE2(q[0]) * SE2.Tx(q[1])
        ans2 = SE2(x) * SE2(1.0) * SE2.Tx(y)
        ans3 = SE2(x) * SE2(z) * SE2.Tx(y)
        ans4 = SE2(x) * SE2(x) * SE2.Tx(2.0)

        fk_traj = r.fkine(qt)
        nt.assert_almost_equal(r.fkine(q), ans1.A)

        nt.assert_almost_equal(fk_traj[0], ans2.A)
        nt.assert_almost_equal(fk_traj[1], ans3.A)
        nt.assert_almost_equal(fk_traj[2], ans4.A)

        base = SE2(1.0)
        tool = SE2.Tx(0.5)
        ans5 = base * ans1

        r2 = rtb.ETS2([rtb.ET2.R(jindex=0)])

        nt.assert_almost_equal(r.fkine(q, base=base), ans5.A)
        # nt.assert_almost_equal(r.fkine(q, base=base.A), ans5.A)  # type: ignore

        q2 = [y]
        ans6 = SE2(y) * tool
        nt.assert_almost_equal(r2.fkine(q2, tool=tool), ans6.A)  # type: ignore
        # nt.assert_almost_equal(r2.fkine(q2, tool=tool.A), ans6)  # type: ignore

    def test_pop(self):
        q = [1.0, 2.0, 3.0]
        a = rtb.ET2.R(jindex=0)
        b = rtb.ET2.R(jindex=1)
        c = rtb.ET2.R(jindex=2)
        d = rtb.ET2.tx(1.543)

        ans1 = SE2(q[0]) * SE2(q[1]) * SE2(q[2]) * SE2.Tx(1.543)
        ans2 = SE2(q[0]) * SE2(q[2]) * SE2.Tx(1.543)
        ans3 = SE2(q[0]) * SE2(q[2])

        r = a + b + c + d

        nt.assert_almost_equal(r.fkine(q), ans1.A)

        et = r.pop(1)
        nt.assert_almost_equal(r.fkine(q), ans2.A)
        nt.assert_almost_equal(et.A(q[1]), SE2(q[1]).A)

        et = r.pop()
        nt.assert_almost_equal(r.fkine(q), ans3.A)
        nt.assert_almost_equal(et.A(), SE2.Tx(1.543).A)

    def test_inv(self):
        q = [1.0, 2.0, 3.0]
        a = rtb.ET2.R(jindex=0)
        b = rtb.ET2.R(jindex=1)
        c = rtb.ET2.R(jindex=2)
        d = rtb.ET2.tx(1.543)

        ans1 = SE2(q[0]) * SE2(q[1]) * SE2(q[2]) * SE2.Tx(1.543)

        r = a + b + c + d
        r_inv = r.inv()

        nt.assert_almost_equal(r_inv.fkine(q), ans1.inv().A)

    def test_jointset(self):
        q = [1.0, 2.0, 3.0]
        a = rtb.ET2.R(jindex=0)
        b = rtb.ET2.R(jindex=1)
        c = rtb.ET2.R(jindex=2)
        d = rtb.ET2.tx(1.543)

        ans = set((0, 1, 2))

        r = a + b + c + d
        nt.assert_equal(r.jindex_set(), ans)

    def test_split(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET2.tx(0.333) * rtb.ET2.R(jindex=0)

        l1 = rtb.ET2.R(-90 * deg) * rtb.ET2.R(jindex=1)

        l2 = rtb.ET2.R(90 * deg) * rtb.ET2.tx(0.316) * rtb.ET2.R(jindex=2)

        l3 = rtb.ET2.tx(0.0825) * rtb.ET2.R(90 * deg) * rtb.ET2.R(jindex=3)

        l4 = (
            rtb.ET2.tx(-0.0825)
            * rtb.ET2.R(-90 * deg)
            * rtb.ET2.tx(0.384)
            * rtb.ET2.R(jindex=4)
        )

        l5 = rtb.ET2.R(90 * deg) * rtb.ET2.R(jindex=5)

        l6 = (
            rtb.ET2.tx(0.088)
            * rtb.ET2.R(90 * deg)
            * rtb.ET2.tx(0.107)
            * rtb.ET2.R(jindex=6)
        )

        ee = rtb.ET2.tx(tool_offset) * rtb.ET2.R(-np.pi / 4)

        segs = [l0, l1, l2, l3, l4, l5, l6, ee]
        segs3 = [l0, l1, l2, l3, l4, l5, l6, rtb.ETS2([rtb.ET2.R(0.5)])]
        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee
        r2 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r3 = l0 * l1 * l2 * l3 * l4 * l5 * l6 * rtb.ET2.R(0.5)

        split = r.split()
        split2 = r2.split()
        split3 = r3.split()

        for i, link in enumerate(segs):
            self.assertEqual(link, split[i])

        for i, link in enumerate(split2):
            self.assertEqual(link, segs[i])

        for i, link in enumerate(split3):
            self.assertEqual(link, segs3[i])

    def test_compile(self):
        q = [0, 1.0, 2, 3, 4, 5, 6]
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET2.tx(0.333) * rtb.ET2.R(jindex=0)

        l1 = rtb.ET2.R(-90 * deg) * rtb.ET2.R(jindex=1)

        l2 = rtb.ET2.R(90 * deg) * rtb.ET2.tx(0.316) * rtb.ET2.R(jindex=2)

        l3 = rtb.ET2.tx(0.0825) * rtb.ET2.R(90 * deg) * rtb.ET2.R(jindex=3)

        l4 = (
            rtb.ET2.tx(-0.0825)
            * rtb.ET2.R(-90 * deg)
            * rtb.ET2.tx(0.384)
            * rtb.ET2.R(jindex=4)
        )

        l5 = rtb.ET2.R(90 * deg) * rtb.ET2.R(jindex=5)

        l6 = (
            rtb.ET2.tx(0.088)
            * rtb.ET2.R(90 * deg)
            * rtb.ET2.tx(0.107)
            * rtb.ET2.R(jindex=6)
        )

        ee = rtb.ET2.tx(tool_offset) * rtb.ET2.R(-np.pi / 4)

        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee
        r2 = r.compile()

        nt.assert_almost_equal(r.fkine(q).A, r2.fkine(q).A)
        self.assertTrue(len(r) > len(r2))

    def test_insert(self):
        q = [0, 1.0, 2, 3, 4, 5, 6]
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET2.tx(0.333) * rtb.ET2.R(jindex=0)

        l1 = rtb.ET2.R(-90 * deg) * rtb.ET2.R(jindex=1)

        l2 = rtb.ET2.R(90 * deg) * rtb.ET2.tx(0.316) * rtb.ET2.R(jindex=2)

        l3 = rtb.ET2.tx(0.0825) * rtb.ET2.R(90 * deg) * rtb.ET2.R(jindex=3)

        l4 = (
            rtb.ET2.tx(-0.0825)
            * rtb.ET2.R(-90 * deg)
            * rtb.ET2.tx(0.384)
            * rtb.ET2.R(jindex=4)
        )

        l5 = rtb.ET2.R(90 * deg) * rtb.ET2.R(jindex=5)

        l6 = (
            rtb.ET2.tx(0.088)
            * rtb.ET2.R(90 * deg)
            * rtb.ET2.tx(0.107)
            * rtb.ET2.R(jindex=6)
        )

        ee = rtb.ET2.tx(tool_offset) * rtb.ET2.R(-np.pi / 4)

        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee

        r2 = l0 * l1 * l2 * l3 * l4 * l6 * ee
        r2.insert(l5, 14)

        r3 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r3.insert(ee)

        r4 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r4.insert(rtb.ET2.tx(tool_offset))
        r4.insert(rtb.ET2.R(-np.pi / 4))

        r5 = l0 * l1 * l2 * l3 * l4 * l6 * ee
        r5.insert(rtb.ET2.R(90 * deg), 14)
        r5.insert(rtb.ET2.R(jindex=5), 15)

        nt.assert_almost_equal(r.fkine(q).A, r2.fkine(q).A)
        nt.assert_almost_equal(r.fkine(q).A, r3.fkine(q).A)
        nt.assert_almost_equal(r.fkine(q).A, r4.fkine(q).A)
        nt.assert_almost_equal(r.fkine(q).A, r5.fkine(q).A)

    def test_jacob0(self):
        q = [0, 0, 0]
        r = rtb.ETS2(rtb.ET2.R(jindex=0))
        tx = rtb.ETS2(rtb.ET2.tx(jindex=1))
        ty = rtb.ETS2(rtb.ET2.ty(jindex=2))

        r2 = tx + ty + r

        nt.assert_almost_equal(tx.jacob0(q), np.array([[1, 0, 0]]).T)
        nt.assert_almost_equal(ty.jacob0(q), np.array([[0, 1, 0]]).T)
        nt.assert_almost_equal(r.jacob0(q), np.array([[0, 0, 1]]).T)
        nt.assert_almost_equal(r2.jacob0(q), np.eye(3))

    def test_jacobe(self):
        q = [0, 0, 0]
        r = rtb.ETS2(rtb.ET2.R(jindex=0))
        tx = rtb.ETS2(rtb.ET2.tx(jindex=1))
        ty = rtb.ETS2(rtb.ET2.ty(jindex=2))

        r2 = tx + ty + r

        nt.assert_almost_equal(tx.jacobe(q), np.array([[1, 0, 0]]).T)
        nt.assert_almost_equal(ty.jacobe(q), np.array([[0, 1, 0]]).T)
        nt.assert_almost_equal(r.jacobe(q), np.array([[0, 0, 1]]).T)
        nt.assert_almost_equal(r2.jacobe(q), np.eye(3))

    def test_plot(self):
        q2 = np.array([0, 1, 2])
        rz = rtb.ETS2(rtb.ET2.R(jindex=0))
        tx = rtb.ETS2(rtb.ET2.tx(jindex=1, qlim=[-1, 1]))
        ty = rtb.ETS2(rtb.ET2.ty(jindex=2, qlim=[-1, 1]))
        a = rtb.ETS2(rtb.ET2.SE2(np.eye(3)))
        r = tx + ty + rz + a
        r.plot(q=q2, block=False)

    def test_teach(self):
        x = sympy.Symbol("x")
        q2 = np.array([0, 1, 2])
        rz = rtb.ETS2(rtb.ET2.R(jindex=0))
        tx = rtb.ETS2(rtb.ET2.tx(jindex=1, qlim=[-1, 1]))
        ty = rtb.ETS2(rtb.ET2.ty(jindex=2, qlim=[-1, 1]))
        a = rtb.ETS2(rtb.ET2.SE2(np.eye(3)))
        r = tx + ty + rz + a
        r.teach(q=q2, block=False)


if __name__ == "__main__":

    unittest.main()
