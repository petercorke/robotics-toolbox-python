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
from spatialmath.base import tr2jac
import unittest
import sympy


class TestETS(unittest.TestCase):
    def test_bad_arg(self):
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)

        with self.assertRaises(TypeError):
            rtb.ETS([rx, ry, 1.0])  # type: ignore

        with self.assertRaises(TypeError):
            rtb.ETS(1.0)  # type: ignore

    def test_args(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        r1 = l0 + l1 + l2 + l3 + l3 + l4 + l5
        r2 = l0 * l1 * l2 * l3 * l3 * l4 * l5
        r3 = rtb.ETS(l0 + l1 + l2 + l3 + l3 + l4 + l5)
        r4 = rtb.ETS([l0, l1, l2, l3, l3, l4, l5])
        r5 = rtb.ETS([l0, l1, l2, l3, l3, l4, rtb.ET.Rx(90 * deg), rtb.ET.Rz(jindex=5)])
        r6 = rtb.ETS([r1])
        r7 = rtb.ETS(rtb.ET.Rx(1.0))

        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)
        self.assertEqual(r1, r4)
        self.assertEqual(r1, r5)
        self.assertEqual(r7[0], rtb.ET.Rx(1.0))
        self.assertEqual(r1 + rtb.ETS(), r2)

    def test_empty(self):
        r = rtb.ETS()
        self.assertEqual(r.n, 0)
        self.assertEqual(r.m, 0)

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

        print(fk_traj[0])
        print(ans2)

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

    def test_jacob0_panda(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )

        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)

        r = l0 + l1 + l2 + l3 + l4 + l5 + l6 + ee

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array(
            [
                [
                    -1.61683957e-01,
                    1.07925929e-01,
                    -3.41453006e-02,
                    3.35029257e-01,
                    -1.07195463e-02,
                    1.03187865e-01,
                    0.00000000e00,
                ],
                [
                    4.46822947e-01,
                    6.25741987e-01,
                    4.16474664e-01,
                    -8.04745724e-02,
                    7.78257566e-02,
                    -1.17720983e-02,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    -2.35276631e-01,
                    -8.20187641e-02,
                    -5.14076923e-01,
                    -9.98040745e-03,
                    -2.02626953e-01,
                    0.00000000e00,
                ],
                [
                    1.29458954e-16,
                    -9.85449730e-01,
                    3.37672585e-02,
                    -6.16735653e-02,
                    6.68449878e-01,
                    -1.35361558e-01,
                    6.37462344e-01,
                ],
                [
                    9.07021273e-18,
                    1.69967143e-01,
                    1.95778638e-01,
                    9.79165111e-01,
                    1.84470262e-01,
                    9.82748279e-01,
                    1.83758244e-01,
                ],
                [
                    1.00000000e00,
                    -2.26036604e-17,
                    9.80066578e-01,
                    -1.93473657e-01,
                    7.20517510e-01,
                    -1.26028049e-01,
                    7.48247732e-01,
                ],
            ]
        )

        nt.assert_array_almost_equal(r.jacob0(q1), ans)
        nt.assert_array_almost_equal(r.jacob0(q2), ans)
        nt.assert_array_almost_equal(r.jacob0(q3), ans)
        nt.assert_array_almost_equal(r.jacob0(q4), ans)
        self.assertRaises(TypeError, r.jacob0, "Wfgsrth")

    def test_jacobe_panda(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)
        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)
        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)
        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)
        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )
        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)
        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )
        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)

        r = l0 + l1 + l2 + l3 + l4 + l5 + l6 + ee

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = tr2jac(r.fkine(q1).T) @ r.jacob0(q1)

        nt.assert_array_almost_equal(r.jacobe(q1), ans)

    def test_pop(self):
        q = [1.0, 2.0, 3.0]
        a = rtb.ET.Rx(jindex=0)
        b = rtb.ET.Ry(jindex=1)
        c = rtb.ET.Rz(jindex=2)
        d = rtb.ET.tz(1.543)

        ans1 = SE3.Rx(q[0]) * SE3.Ry(q[1]) * SE3.Rz(q[2]) * SE3.Tz(1.543)
        ans2 = SE3.Rx(q[0]) * SE3.Rz(q[2]) * SE3.Tz(1.543)
        ans3 = SE3.Rx(q[0]) * SE3.Rz(q[2])

        r = a + b + c + d

        nt.assert_almost_equal(r.fkine(q), ans1)

        et = r.pop(1)
        nt.assert_almost_equal(r.fkine(q), ans2)
        nt.assert_almost_equal(et.T(q[1]), SE3.Ry(q[1]))

        et = r.pop()
        nt.assert_almost_equal(r.fkine(q), ans3)
        nt.assert_almost_equal(et.T(), SE3.Tz(1.543))

    def test_inv(self):
        q = [1.0, 2.0, 3.0]
        a = rtb.ET.Rx(jindex=0)
        b = rtb.ET.Ry(jindex=1)
        c = rtb.ET.Rz(jindex=2)
        d = rtb.ET.tz(1.543)

        ans1 = SE3.Rx(q[0]) * SE3.Ry(q[1]) * SE3.Rz(q[2]) * SE3.Tz(1.543)

        r = a + b + c + d
        r_inv = r.inv()

        nt.assert_almost_equal(r_inv.fkine(q), ans1.inv())

    def test_jointset(self):
        q = [1.0, 2.0, 3.0]
        a = rtb.ET.Rx(jindex=0)
        b = rtb.ET.Ry(jindex=1)
        c = rtb.ET.Rz(jindex=2)
        d = rtb.ET.tz(1.543)

        ans = set((0, 1, 2))

        r = a + b + c + d
        nt.assert_equal(r.jointset(), ans)

    def test_split(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )

        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)

        segs = [l0, l1, l2, l3, l4, l5, l6, ee]
        segs3 = [l0, l1, l2, l3, l4, l5, l6, rtb.ETS([rtb.ET.Rx(0.5)])]
        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee
        r2 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r3 = l0 * l1 * l2 * l3 * l4 * l5 * l6 * rtb.ET.Rx(0.5)

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

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )

        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)

        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee
        r2 = r.compile()

        nt.assert_almost_equal(r.fkine(q), r2.fkine(q))
        self.assertTrue(len(r) > len(r2))

    def test_insert(self):
        q = [1.0, 2, 3, 4, 5, 6]
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )

        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)

        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee

        r2 = l0 * l1 * l2 * l3 * l4 * l6 * ee
        r2.insert(l5, 14)

        r3 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r3.insert(ee)

        r4 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r4.insert(rtb.ET.tz(tool_offset))
        r4.insert(rtb.ET.Rz(-np.pi / 4))

        r5 = l0 * l1 * l2 * l3 * l4 * l6 * ee
        r5.insert(rtb.ET.Rx(90 * deg), 14)
        r5.insert(rtb.ET.Rz(jindex=5), 15)

        nt.assert_almost_equal(r.fkine(q), r2.fkine(q))
        nt.assert_almost_equal(r.fkine(q), r3.fkine(q))
        nt.assert_almost_equal(r.fkine(q), r4.fkine(q))
        nt.assert_almost_equal(r.fkine(q), r5.fkine(q))

    def test_jacob0(self):
        q = [0]
        rx = rtb.ETS(rtb.ET.Rx())
        ry = rtb.ETS(rtb.ET.Ry())
        rz = rtb.ETS(rtb.ET.Rz())
        tx = rtb.ETS(rtb.ET.tx())
        ty = rtb.ETS(rtb.ET.ty())
        tz = rtb.ETS(rtb.ET.tz())

        r = tx + ty + tz + rx + ry + rz

        nt.assert_almost_equal(tx.jacob0(q), np.array([[1, 0, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(ty.jacob0(q), np.array([[0, 1, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(tz.jacob0(q), np.array([[0, 0, 1, 0, 0, 0]]).T)
        nt.assert_almost_equal(rx.jacob0(q), np.array([[0, 0, 0, 1, 0, 0]]).T)
        nt.assert_almost_equal(ry.jacob0(q), np.array([[0, 0, 0, 0, 1, 0]]).T)
        nt.assert_almost_equal(rz.jacob0(q), np.array([[0, 0, 0, 0, 0, 1]]).T)
        nt.assert_almost_equal(r.jacob0(q), np.eye(6))

    def test_jacobe(self):
        q = [0]
        rx = rtb.ETS(rtb.ET.Rx())
        ry = rtb.ETS(rtb.ET.Ry())
        rz = rtb.ETS(rtb.ET.Rz())
        tx = rtb.ETS(rtb.ET.tx())
        ty = rtb.ETS(rtb.ET.ty())
        tz = rtb.ETS(rtb.ET.tz())

        r = tx + ty + tz + rx + ry + rz

        nt.assert_almost_equal(tx.jacobe(q), np.array([[1, 0, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(ty.jacobe(q), np.array([[0, 1, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(tz.jacobe(q), np.array([[0, 0, 1, 0, 0, 0]]).T)
        nt.assert_almost_equal(rx.jacobe(q), np.array([[0, 0, 0, 1, 0, 0]]).T)
        nt.assert_almost_equal(ry.jacobe(q), np.array([[0, 0, 0, 0, 1, 0]]).T)
        nt.assert_almost_equal(rz.jacobe(q), np.array([[0, 0, 0, 0, 0, 1]]).T)
        nt.assert_almost_equal(r.jacobe(q), np.eye(6))

    def test_jacob0_sym(self):
        x = sympy.Symbol("x")
        q1 = np.array([x, x])
        q2 = np.array([0, x])
        rx = rtb.ETS(rtb.ET.Rx(jindex=0))
        ry = rtb.ETS(rtb.ET.Ry(jindex=0))
        rz = rtb.ETS(rtb.ET.Rz(jindex=0))
        tx = rtb.ETS(rtb.ET.tx(jindex=0))
        ty = rtb.ETS(rtb.ET.ty(jindex=0))
        tz = rtb.ETS(rtb.ET.tz(jindex=1))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))

        r = tx + ty + tz + rx + ry + rz + a

        print(r.jacob0(q2))

        nt.assert_almost_equal(tx.jacob0(q1), np.array([[1, 0, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(ty.jacob0(q1), np.array([[0, 1, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(tz.jacob0(q1), np.array([[0, 0, 1, 0, 0, 0]]).T)
        nt.assert_almost_equal(rx.jacob0(q1), np.array([[0, 0, 0, 1, 0, 0]]).T)
        nt.assert_almost_equal(ry.jacob0(q1), np.array([[0, 0, 0, 0, 1, 0]]).T)
        nt.assert_almost_equal(rz.jacob0(q1), np.array([[0, 0, 0, 0, 0, 1]]).T)
        nt.assert_almost_equal(r.jacob0(q2), np.eye(6))
        nt.assert_almost_equal(r.jacob0(q2, tool=SE3()), np.eye(6))
        nt.assert_almost_equal(r.jacob0(q2, tool=SE3().A), np.eye(6))

    def test_jacobe_sym(self):
        x = sympy.Symbol("x")
        q1 = np.array([x, x])
        q2 = np.array([0, x])
        rx = rtb.ETS(rtb.ET.Rx(jindex=0))
        ry = rtb.ETS(rtb.ET.Ry(jindex=0))
        rz = rtb.ETS(rtb.ET.Rz(jindex=0))
        tx = rtb.ETS(rtb.ET.tx(jindex=0))
        ty = rtb.ETS(rtb.ET.ty(jindex=0))
        tz = rtb.ETS(rtb.ET.tz(jindex=1))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))

        r = tx + ty + tz + rx + ry + rz + a

        print(r.jacobe(q2))

        nt.assert_almost_equal(tx.jacobe(q1), np.array([[1, 0, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(ty.jacobe(q1), np.array([[0, 1, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(tz.jacobe(q1), np.array([[0, 0, 1, 0, 0, 0]]).T)
        nt.assert_almost_equal(rx.jacobe(q1), np.array([[0, 0, 0, 1, 0, 0]]).T)
        nt.assert_almost_equal(ry.jacobe(q1), np.array([[0, 0, 0, 0, 1, 0]]).T)
        nt.assert_almost_equal(rz.jacobe(q1), np.array([[0, 0, 0, 0, 0, 1]]).T)
        nt.assert_almost_equal(r.jacobe(q2), np.eye(6))
        nt.assert_almost_equal(r.jacobe(q2, tool=SE3()), np.eye(6))
        nt.assert_almost_equal(r.jacobe(q2, tool=SE3().A), np.eye(6))

    def test_hessian0(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )

        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)

        r = l0 + l1 + l2 + l3 + l4 + l5 + l6 + ee

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array(
            [
                [
                    [
                        -4.46822947e-01,
                        -6.25741987e-01,
                        -4.16474664e-01,
                        8.04745724e-02,
                        -7.78257566e-02,
                        1.17720983e-02,
                        0.00000000e00,
                    ],
                    [
                        -6.25741987e-01,
                        -3.99892968e-02,
                        -1.39404950e-02,
                        -8.73761859e-02,
                        -1.69634134e-03,
                        -3.44399243e-02,
                        0.00000000e00,
                    ],
                    [
                        -4.16474664e-01,
                        -1.39404950e-02,
                        -4.24230421e-01,
                        -2.17748413e-02,
                        -7.82283735e-02,
                        -2.81325889e-02,
                        0.00000000e00,
                    ],
                    [
                        8.04745724e-02,
                        -8.73761859e-02,
                        -2.17748413e-02,
                        -5.18935898e-01,
                        5.28476698e-03,
                        -2.00682834e-01,
                        0.00000000e00,
                    ],
                    [
                        -7.78257566e-02,
                        -1.69634134e-03,
                        -7.82283735e-02,
                        5.28476698e-03,
                        -5.79159088e-02,
                        -2.88966443e-02,
                        0.00000000e00,
                    ],
                    [
                        1.17720983e-02,
                        -3.44399243e-02,
                        -2.81325889e-02,
                        -2.00682834e-01,
                        -2.88966443e-02,
                        -2.00614904e-01,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        -1.61683957e-01,
                        1.07925929e-01,
                        -3.41453006e-02,
                        3.35029257e-01,
                        -1.07195463e-02,
                        1.03187865e-01,
                        0.00000000e00,
                    ],
                    [
                        1.07925929e-01,
                        -2.31853293e-01,
                        -8.08253690e-02,
                        -5.06596965e-01,
                        -9.83518983e-03,
                        -1.99678676e-01,
                        0.00000000e00,
                    ],
                    [
                        -3.41453006e-02,
                        -8.08253690e-02,
                        -3.06951191e-02,
                        3.45709946e-01,
                        -1.01688580e-02,
                        1.07973135e-01,
                        0.00000000e00,
                    ],
                    [
                        3.35029257e-01,
                        -5.06596965e-01,
                        3.45709946e-01,
                        -9.65242924e-02,
                        1.45842251e-03,
                        -3.24608603e-02,
                        0.00000000e00,
                    ],
                    [
                        -1.07195463e-02,
                        -9.83518983e-03,
                        -1.01688580e-02,
                        1.45842251e-03,
                        -1.05221866e-03,
                        2.09794626e-01,
                        0.00000000e00,
                    ],
                    [
                        1.03187865e-01,
                        -1.99678676e-01,
                        1.07973135e-01,
                        -3.24608603e-02,
                        2.09794626e-01,
                        -4.04324654e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -6.34981134e-01,
                        -4.04611266e-01,
                        2.23596800e-02,
                        -7.48714002e-02,
                        -5.93773551e-03,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -4.04611266e-01,
                        2.07481281e-02,
                        -6.83089775e-02,
                        4.72662062e-03,
                        -2.05994912e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        2.23596800e-02,
                        -6.83089775e-02,
                        -3.23085806e-01,
                        5.69641385e-03,
                        -1.00311930e-01,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -7.48714002e-02,
                        4.72662062e-03,
                        5.69641385e-03,
                        5.40000550e-02,
                        -2.69041502e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -5.93773551e-03,
                        -2.05994912e-02,
                        -1.00311930e-01,
                        -2.69041502e-02,
                        -9.98142073e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        -9.07021273e-18,
                        -2.77555756e-17,
                        -2.77555756e-17,
                        -1.11022302e-16,
                        -2.77555756e-17,
                        0.00000000e00,
                        -2.77555756e-17,
                    ],
                    [
                        -1.69967143e-01,
                        -1.97756387e-17,
                        4.11786040e-17,
                        -1.48932398e-16,
                        -5.07612940e-17,
                        -8.38219650e-17,
                        -4.90138154e-17,
                    ],
                    [
                        -1.95778638e-01,
                        1.66579116e-01,
                        -1.38777878e-17,
                        1.04083409e-17,
                        -1.38777878e-17,
                        3.46944695e-18,
                        0.00000000e00,
                    ],
                    [
                        -9.79165111e-01,
                        -3.28841647e-02,
                        -9.97525009e-01,
                        -4.16333634e-17,
                        -1.14491749e-16,
                        1.38777878e-17,
                        -6.24500451e-17,
                    ],
                    [
                        -1.84470262e-01,
                        1.22464303e-01,
                        -3.97312016e-02,
                        7.41195745e-01,
                        -2.77555756e-17,
                        1.12757026e-16,
                        2.77555756e-17,
                    ],
                    [
                        -9.82748279e-01,
                        -2.14206274e-02,
                        -9.87832342e-01,
                        6.67336352e-02,
                        -7.31335770e-01,
                        2.08166817e-17,
                        -6.07153217e-17,
                    ],
                    [
                        -1.83758244e-01,
                        1.27177529e-01,
                        -3.36043908e-02,
                        7.68210453e-01,
                        5.62842325e-03,
                        7.58497864e-01,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        1.29458954e-16,
                        -1.11022302e-16,
                        8.67361738e-17,
                        -4.16333634e-17,
                        5.55111512e-17,
                        2.77555756e-17,
                        5.55111512e-17,
                    ],
                    [
                        -9.85449730e-01,
                        -6.36381327e-17,
                        -1.02735399e-16,
                        -1.83043043e-17,
                        -5.63484308e-17,
                        8.08886307e-18,
                        1.07112702e-18,
                    ],
                    [
                        3.37672585e-02,
                        9.65806345e-01,
                        8.32667268e-17,
                        -2.55871713e-17,
                        1.07552856e-16,
                        2.08166817e-17,
                        -5.20417043e-18,
                    ],
                    [
                        -6.16735653e-02,
                        -1.90658563e-01,
                        -5.39111251e-02,
                        -6.59194921e-17,
                        -2.77555756e-17,
                        2.38524478e-17,
                        -4.16333634e-17,
                    ],
                    [
                        6.68449878e-01,
                        7.10033786e-01,
                        6.30795483e-01,
                        -8.48905588e-02,
                        0.00000000e00,
                        3.46944695e-17,
                        2.77555756e-17,
                    ],
                    [
                        -1.35361558e-01,
                        -1.24194307e-01,
                        -1.28407717e-01,
                        1.84162966e-02,
                        -1.32869389e-02,
                        2.77555756e-17,
                        -2.08166817e-17,
                    ],
                    [
                        6.37462344e-01,
                        7.37360525e-01,
                        5.99489263e-01,
                        -7.71850655e-02,
                        -4.08633244e-02,
                        2.09458434e-02,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        -6.59521910e-17,
                        -1.31033786e-16,
                        -1.92457571e-16,
                        1.54134782e-17,
                        -7.69804929e-17,
                        1.11140361e-17,
                    ],
                    [
                        0.00000000e00,
                        -2.77555756e-17,
                        7.15573434e-17,
                        1.65666092e-16,
                        1.38777878e-17,
                        -8.67361738e-18,
                        3.46944695e-17,
                    ],
                    [
                        0.00000000e00,
                        -1.98669331e-01,
                        8.67361738e-18,
                        -1.46584134e-16,
                        6.02816408e-17,
                        -3.12250226e-17,
                        6.11490025e-17,
                    ],
                    [
                        0.00000000e00,
                        -9.54435515e-01,
                        4.51380881e-02,
                        1.38777878e-17,
                        1.08420217e-16,
                        3.46944695e-18,
                        6.24500451e-17,
                    ],
                    [
                        0.00000000e00,
                        -2.95400686e-01,
                        -1.24639152e-01,
                        -6.65899738e-01,
                        -4.85722573e-17,
                        -5.20417043e-18,
                        -5.55111512e-17,
                    ],
                    [
                        0.00000000e00,
                        -9.45442009e-01,
                        5.96856167e-02,
                        7.19317248e-02,
                        6.81888149e-01,
                        -2.77555756e-17,
                        1.04083409e-17,
                    ],
                    [
                        0.00000000e00,
                        -2.89432165e-01,
                        -1.18596498e-01,
                        -6.35513913e-01,
                        5.24032975e-03,
                        -6.51338823e-01,
                        0.00000000e00,
                    ],
                ],
            ]
        )

        ans_new = np.empty((7, 6, 7))

        for i in range(7):
            ans_new[i, :, :] = ans[:, :, i]

        J = r.jacob0(q1)

        # print(np.round(ans_new, 1))
        # print(np.round(r.hessian0(q1), 1))
        # print(r.jacob0(q1))

        nt.assert_array_almost_equal(r.hessian0(q1), ans_new)
        nt.assert_array_almost_equal(r.hessian0(q2), ans_new)
        nt.assert_array_almost_equal(r.hessian0(q3), ans_new)
        nt.assert_array_almost_equal(r.hessian0(q4), ans_new)

        nt.assert_array_almost_equal(r.hessian0(J0=J), ans_new)
        nt.assert_array_almost_equal(r.hessian0(J0=J), ans_new)
        nt.assert_array_almost_equal(r.hessian0(J0=J), ans_new)
        nt.assert_array_almost_equal(r.hessian0(J0=J), ans_new)

    def test_hessian0_tool(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )

        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)
        ee = ee.fkine([])

        r = l0 + l1 + l2 + l3 + l4 + l5 + l6

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array(
            [
                [
                    [
                        -4.46822947e-01,
                        -6.25741987e-01,
                        -4.16474664e-01,
                        8.04745724e-02,
                        -7.78257566e-02,
                        1.17720983e-02,
                        0.00000000e00,
                    ],
                    [
                        -6.25741987e-01,
                        -3.99892968e-02,
                        -1.39404950e-02,
                        -8.73761859e-02,
                        -1.69634134e-03,
                        -3.44399243e-02,
                        0.00000000e00,
                    ],
                    [
                        -4.16474664e-01,
                        -1.39404950e-02,
                        -4.24230421e-01,
                        -2.17748413e-02,
                        -7.82283735e-02,
                        -2.81325889e-02,
                        0.00000000e00,
                    ],
                    [
                        8.04745724e-02,
                        -8.73761859e-02,
                        -2.17748413e-02,
                        -5.18935898e-01,
                        5.28476698e-03,
                        -2.00682834e-01,
                        0.00000000e00,
                    ],
                    [
                        -7.78257566e-02,
                        -1.69634134e-03,
                        -7.82283735e-02,
                        5.28476698e-03,
                        -5.79159088e-02,
                        -2.88966443e-02,
                        0.00000000e00,
                    ],
                    [
                        1.17720983e-02,
                        -3.44399243e-02,
                        -2.81325889e-02,
                        -2.00682834e-01,
                        -2.88966443e-02,
                        -2.00614904e-01,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        -1.61683957e-01,
                        1.07925929e-01,
                        -3.41453006e-02,
                        3.35029257e-01,
                        -1.07195463e-02,
                        1.03187865e-01,
                        0.00000000e00,
                    ],
                    [
                        1.07925929e-01,
                        -2.31853293e-01,
                        -8.08253690e-02,
                        -5.06596965e-01,
                        -9.83518983e-03,
                        -1.99678676e-01,
                        0.00000000e00,
                    ],
                    [
                        -3.41453006e-02,
                        -8.08253690e-02,
                        -3.06951191e-02,
                        3.45709946e-01,
                        -1.01688580e-02,
                        1.07973135e-01,
                        0.00000000e00,
                    ],
                    [
                        3.35029257e-01,
                        -5.06596965e-01,
                        3.45709946e-01,
                        -9.65242924e-02,
                        1.45842251e-03,
                        -3.24608603e-02,
                        0.00000000e00,
                    ],
                    [
                        -1.07195463e-02,
                        -9.83518983e-03,
                        -1.01688580e-02,
                        1.45842251e-03,
                        -1.05221866e-03,
                        2.09794626e-01,
                        0.00000000e00,
                    ],
                    [
                        1.03187865e-01,
                        -1.99678676e-01,
                        1.07973135e-01,
                        -3.24608603e-02,
                        2.09794626e-01,
                        -4.04324654e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -6.34981134e-01,
                        -4.04611266e-01,
                        2.23596800e-02,
                        -7.48714002e-02,
                        -5.93773551e-03,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -4.04611266e-01,
                        2.07481281e-02,
                        -6.83089775e-02,
                        4.72662062e-03,
                        -2.05994912e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        2.23596800e-02,
                        -6.83089775e-02,
                        -3.23085806e-01,
                        5.69641385e-03,
                        -1.00311930e-01,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -7.48714002e-02,
                        4.72662062e-03,
                        5.69641385e-03,
                        5.40000550e-02,
                        -2.69041502e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -5.93773551e-03,
                        -2.05994912e-02,
                        -1.00311930e-01,
                        -2.69041502e-02,
                        -9.98142073e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        -9.07021273e-18,
                        -2.77555756e-17,
                        -2.77555756e-17,
                        -1.11022302e-16,
                        -2.77555756e-17,
                        0.00000000e00,
                        -2.77555756e-17,
                    ],
                    [
                        -1.69967143e-01,
                        -1.97756387e-17,
                        4.11786040e-17,
                        -1.48932398e-16,
                        -5.07612940e-17,
                        -8.38219650e-17,
                        -4.90138154e-17,
                    ],
                    [
                        -1.95778638e-01,
                        1.66579116e-01,
                        -1.38777878e-17,
                        1.04083409e-17,
                        -1.38777878e-17,
                        3.46944695e-18,
                        0.00000000e00,
                    ],
                    [
                        -9.79165111e-01,
                        -3.28841647e-02,
                        -9.97525009e-01,
                        -4.16333634e-17,
                        -1.14491749e-16,
                        1.38777878e-17,
                        -6.24500451e-17,
                    ],
                    [
                        -1.84470262e-01,
                        1.22464303e-01,
                        -3.97312016e-02,
                        7.41195745e-01,
                        -2.77555756e-17,
                        1.12757026e-16,
                        2.77555756e-17,
                    ],
                    [
                        -9.82748279e-01,
                        -2.14206274e-02,
                        -9.87832342e-01,
                        6.67336352e-02,
                        -7.31335770e-01,
                        2.08166817e-17,
                        -6.07153217e-17,
                    ],
                    [
                        -1.83758244e-01,
                        1.27177529e-01,
                        -3.36043908e-02,
                        7.68210453e-01,
                        5.62842325e-03,
                        7.58497864e-01,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        1.29458954e-16,
                        -1.11022302e-16,
                        8.67361738e-17,
                        -4.16333634e-17,
                        5.55111512e-17,
                        2.77555756e-17,
                        5.55111512e-17,
                    ],
                    [
                        -9.85449730e-01,
                        -6.36381327e-17,
                        -1.02735399e-16,
                        -1.83043043e-17,
                        -5.63484308e-17,
                        8.08886307e-18,
                        1.07112702e-18,
                    ],
                    [
                        3.37672585e-02,
                        9.65806345e-01,
                        8.32667268e-17,
                        -2.55871713e-17,
                        1.07552856e-16,
                        2.08166817e-17,
                        -5.20417043e-18,
                    ],
                    [
                        -6.16735653e-02,
                        -1.90658563e-01,
                        -5.39111251e-02,
                        -6.59194921e-17,
                        -2.77555756e-17,
                        2.38524478e-17,
                        -4.16333634e-17,
                    ],
                    [
                        6.68449878e-01,
                        7.10033786e-01,
                        6.30795483e-01,
                        -8.48905588e-02,
                        0.00000000e00,
                        3.46944695e-17,
                        2.77555756e-17,
                    ],
                    [
                        -1.35361558e-01,
                        -1.24194307e-01,
                        -1.28407717e-01,
                        1.84162966e-02,
                        -1.32869389e-02,
                        2.77555756e-17,
                        -2.08166817e-17,
                    ],
                    [
                        6.37462344e-01,
                        7.37360525e-01,
                        5.99489263e-01,
                        -7.71850655e-02,
                        -4.08633244e-02,
                        2.09458434e-02,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        -6.59521910e-17,
                        -1.31033786e-16,
                        -1.92457571e-16,
                        1.54134782e-17,
                        -7.69804929e-17,
                        1.11140361e-17,
                    ],
                    [
                        0.00000000e00,
                        -2.77555756e-17,
                        7.15573434e-17,
                        1.65666092e-16,
                        1.38777878e-17,
                        -8.67361738e-18,
                        3.46944695e-17,
                    ],
                    [
                        0.00000000e00,
                        -1.98669331e-01,
                        8.67361738e-18,
                        -1.46584134e-16,
                        6.02816408e-17,
                        -3.12250226e-17,
                        6.11490025e-17,
                    ],
                    [
                        0.00000000e00,
                        -9.54435515e-01,
                        4.51380881e-02,
                        1.38777878e-17,
                        1.08420217e-16,
                        3.46944695e-18,
                        6.24500451e-17,
                    ],
                    [
                        0.00000000e00,
                        -2.95400686e-01,
                        -1.24639152e-01,
                        -6.65899738e-01,
                        -4.85722573e-17,
                        -5.20417043e-18,
                        -5.55111512e-17,
                    ],
                    [
                        0.00000000e00,
                        -9.45442009e-01,
                        5.96856167e-02,
                        7.19317248e-02,
                        6.81888149e-01,
                        -2.77555756e-17,
                        1.04083409e-17,
                    ],
                    [
                        0.00000000e00,
                        -2.89432165e-01,
                        -1.18596498e-01,
                        -6.35513913e-01,
                        5.24032975e-03,
                        -6.51338823e-01,
                        0.00000000e00,
                    ],
                ],
            ]
        )

        ans_new = np.empty((7, 6, 7))

        for i in range(7):
            ans_new[i, :, :] = ans[:, :, i]

        J = r.jacob0(q1, tool=ee)

        nt.assert_array_almost_equal(r.hessian0(q1, tool=ee), ans_new)
        nt.assert_array_almost_equal(r.hessian0(q2, tool=SE3(ee)), ans_new)
        nt.assert_array_almost_equal(r.hessian0(q3, tool=ee), ans_new)
        nt.assert_array_almost_equal(r.hessian0(q4, tool=ee), ans_new)

        nt.assert_array_almost_equal(r.hessian0(J0=J), ans_new)
        nt.assert_array_almost_equal(r.hessian0(J0=J), ans_new)
        nt.assert_array_almost_equal(r.hessian0(J0=J), ans_new)
        nt.assert_array_almost_equal(r.hessian0(J0=J), ans_new)

    def test_hessiane(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )

        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)

        r = l0 + l1 + l2 + l3 + l4 + l5 + l6 + ee

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        H0 = r.hessian0(q1)
        He = np.empty((r.n, 6, r.n))
        T = r.fkine(q1, include_base=False)

        for i in range(r.n):
            He[i, :, :] = tr2jac(T.T) @ H0[i, :, :]

        J = r.jacobe(q1)

        nt.assert_array_almost_equal(r.hessiane(q1), He)
        nt.assert_array_almost_equal(r.hessiane(q2), He)
        nt.assert_array_almost_equal(r.hessiane(q3), He)
        nt.assert_array_almost_equal(r.hessiane(q4), He)

        nt.assert_array_almost_equal(r.hessiane(Je=J), He)
        nt.assert_array_almost_equal(r.hessiane(Je=J), He)
        nt.assert_array_almost_equal(r.hessiane(Je=J), He)
        nt.assert_array_almost_equal(r.hessiane(Je=J), He)

    def test_hessiane_tool(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )

        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)
        ee = ee.fkine([])

        r = l0 + l1 + l2 + l3 + l4 + l5 + l6

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        H0 = r.hessian0(q1, tool=ee)
        He = np.empty((r.n, 6, r.n))
        T = r.fkine(q1, tool=ee, include_base=False)

        for i in range(r.n):
            He[i, :, :] = tr2jac(T.T) @ H0[i, :, :]

        J = r.jacobe(q1, tool=ee)

        nt.assert_array_almost_equal(r.hessiane(q1, tool=ee), He)
        nt.assert_array_almost_equal(r.hessiane(q2, tool=SE3(ee)), He)
        nt.assert_array_almost_equal(r.hessiane(q3, tool=ee), He)
        nt.assert_array_almost_equal(r.hessiane(q4, tool=ee), He)

        nt.assert_array_almost_equal(r.hessiane(Je=J), He)
        nt.assert_array_almost_equal(r.hessiane(Je=J), He)
        nt.assert_array_almost_equal(r.hessiane(Je=J), He)
        nt.assert_array_almost_equal(r.hessiane(Je=J), He)

    def test_hessian_sym(self):
        x = sympy.Symbol("x")
        q2 = np.array([0, x])
        rx = rtb.ETS(rtb.ET.Rx(jindex=0))
        ry = rtb.ETS(rtb.ET.Ry(jindex=0))
        rz = rtb.ETS(rtb.ET.Rz(jindex=0))
        tx = rtb.ETS(rtb.ET.tx(jindex=0))
        ty = rtb.ETS(rtb.ET.ty(jindex=0))
        tz = rtb.ETS(rtb.ET.tz(jindex=1))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))

        r = tx + ty + tz + rx + ry + rz + a

        J0 = r.jacob0(q2)
        Je = r.jacobe(q2)

        ans = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )

        nt.assert_almost_equal(r.hessian0(q2), ans)
        nt.assert_almost_equal(r.hessiane(q2), ans)

        nt.assert_almost_equal(r.hessian0(J0=J0), ans)
        nt.assert_almost_equal(r.hessiane(Je=Je), ans)


if __name__ == "__main__":

    unittest.main()
