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
        # mm = 1e-3
        # tool_offset = (103) * mm

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
        # r6 = rtb.ETS([r1])
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
        self.assertEqual(str(ets3), "tx(1) ⊕ SE3(57.3°, -0°, 0°)")
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
        rx = rtb.ET.Rx(x)  # type: ignore
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
        # rz = rtb.ET.Rz(1.543)
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

        nt.assert_almost_equal(r.fkine(q).A, ans.A)

    def test_fkine_sym(self):
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        z = sympy.Symbol("z")

        q = np.array([y, z])
        qt = np.array([[1.0, y], [z, y], [x, 2.0]])
        a = rtb.ET.Rx(x)  # type: ignore
        b = rtb.ET.Ry(jindex=0)
        c = rtb.ET.tz(jindex=1)

        r = a * b * c

        ans1 = SE3.Rx(x) * SE3.Ry(q[0]) * SE3.Tz(q[1])
        ans2 = SE3.Rx(x) * SE3.Ry(1.0) * SE3.Tz(y)
        ans3 = SE3.Rx(x) * SE3.Ry(z) * SE3.Tz(y)
        ans4 = SE3.Rx(x) * SE3.Ry(x) * SE3.Tz(2.0)

        fk_traj = r.fkine(qt)

        # print(fk_traj[0])
        # print(fk_traj[1])
        # print(fk_traj[2])
        # print(ans4)

        nt.assert_almost_equal(r.fkine(q).A, ans1.A)
        nt.assert_almost_equal(fk_traj[0].A, ans2.A)
        nt.assert_almost_equal(fk_traj[1].A, ans3.A)
        nt.assert_almost_equal(fk_traj[2].A, sympy.simplify(ans4.A))

        base = SE3.Rx(1.0)
        tool = SE3.Tz(0.5)
        ans5 = base * ans1

        r2 = rtb.ETS([rtb.ET.Rx(jindex=0)])

        nt.assert_almost_equal(r.fkine(q, base=base).A, sympy.simplify(ans5.A))
        # nt.assert_almost_equal(r.fkine(q, base=base), ans5.A)  # type: ignore

        q2 = [y]
        ans6 = SE3.Rx(y) * tool
        nt.assert_almost_equal(
            r2.fkine(q2, tool=tool).A, sympy.simplify(ans6.A)  # type: ignore
        )
        # nt.assert_almost_equal(r2.fkine(q2, tool=tool), ans6)  # type: ignore

    def test_fkine_traj(self):
        robot = rtb.ERobot(
            [
                rtb.Link(rtb.ET.Rx()),
                rtb.Link(rtb.ET.Ry()),
                rtb.Link(rtb.ET.Rz()),
                rtb.Link(rtb.ET.tx()),
                rtb.Link(rtb.ET.ty()),
                rtb.Link(rtb.ET.tz()),
            ]
        )

        ets = robot.ets()

        qt = np.arange(10 * ets.n).reshape(10, ets.n)

        T_individual = []

        for q in qt:
            T_individual.append(ets.eval(q))

        T_traj = ets.eval(qt)

        for i in range(10):
            nt.assert_allclose(T_traj[i, :, :], T_individual[i])

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
        # q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        # q3 = np.expand_dims(q1, 0)
        # q4 = np.expand_dims(q1, 1)

        ans = tr2jac(r.eval(q1).T) @ r.jacob0(q1)

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

        nt.assert_almost_equal(r.fkine(q).A, ans1.A)

        et = r.pop(1)
        nt.assert_almost_equal(r.fkine(q).A, ans2.A)
        nt.assert_almost_equal(et.A(q[1]), SE3.Ry(q[1]).A)

        et = r.pop()
        nt.assert_almost_equal(r.fkine(q).A, ans3.A)
        nt.assert_almost_equal(et.A(), SE3.Tz(1.543).A)

    def test_inv(self):
        q = [1.0, 2.0, 3.0]
        a = rtb.ET.Rx(jindex=0)
        b = rtb.ET.Ry(jindex=1)
        c = rtb.ET.Rz(jindex=2)
        d = rtb.ET.tz(1.543)

        ans1 = SE3.Rx(q[0]) * SE3.Ry(q[1]) * SE3.Rz(q[2]) * SE3.Tz(1.543)

        r = a + b + c + d
        r_inv = r.inv()

        nt.assert_almost_equal(r_inv.fkine(q).A, ans1.inv().A)

    def test_jointset(self):
        # q = [1.0, 2.0, 3.0]
        a = rtb.ET.Rx(jindex=0)
        b = rtb.ET.Ry(jindex=1)
        c = rtb.ET.Rz(jindex=2)
        d = rtb.ET.tz(1.543)

        ans = set((0, 1, 2))

        r = a + b + c + d
        nt.assert_equal(r.jindex_set(), ans)

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

        nt.assert_almost_equal(r.eval(q), r2.eval(q))
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

        nt.assert_almost_equal(r.eval(q), r2.eval(q))
        nt.assert_almost_equal(r.eval(q), r3.eval(q))
        nt.assert_almost_equal(r.eval(q), r4.eval(q))
        nt.assert_almost_equal(r.fkine(q).A, r5.fkine(q).A)

    def test_jacob0(self):
        q = [0.0]
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
        q = [0.0]
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
        T = r.eval(q1, include_base=False)

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
        T = r.eval(q1, tool=ee, include_base=False)

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

    def test_plot(self):
        q2 = np.array([0, 1, 2, 3, 4, 5])
        rx = rtb.ETS(rtb.ET.Rx(jindex=0))
        ry = rtb.ETS(rtb.ET.Ry(jindex=1))
        rz = rtb.ETS(rtb.ET.Rz(jindex=2))
        tx = rtb.ETS(rtb.ET.tx(jindex=3, qlim=[-1, 1]))
        ty = rtb.ETS(rtb.ET.ty(jindex=4, qlim=[-1, 1]))
        tz = rtb.ETS(rtb.ET.tz(jindex=5, qlim=[-1, 1]))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))
        r = tx + ty + tz + rx + ry + rz + a
        r.plot(q=q2, block=False, backend="pyplot")

    def test_teach(self):
        # x = sympy.Symbol("x")
        q2 = np.array([0, 1, 2, 3, 4, 5])
        rx = rtb.ETS(rtb.ET.Rx(jindex=0))
        ry = rtb.ETS(rtb.ET.Ry(jindex=1))
        rz = rtb.ETS(rtb.ET.Rz(jindex=2))
        tx = rtb.ETS(rtb.ET.tx(jindex=3, qlim=[-1, 1]))
        ty = rtb.ETS(rtb.ET.ty(jindex=4, qlim=[-1, 1]))
        tz = rtb.ETS(rtb.ET.tz(jindex=5, qlim=[-1, 1]))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))
        r = tx + ty + tz + rx + ry + rz + a
        r.teach(q=q2, block=False, backend="pyplot")

    def test_partial_fkine(self):
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
        r2 = rtb.Robot(r)

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])

        ans = np.array(
            [
                [
                    [
                        [
                            1.61683957e-01,
                            -1.07925929e-01,
                            3.41453006e-02,
                            -3.35029257e-01,
                            1.07195463e-02,
                            -1.03187865e-01,
                            0.00000000e00,
                        ],
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
                            -3.17334834e-18,
                            1.10026604e-17,
                            2.42699171e-18,
                            1.47042021e-17,
                            2.53075280e-19,
                            4.65397353e-18,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            9.85449730e-01,
                            -3.37672585e-02,
                            6.16735653e-02,
                            -6.68449878e-01,
                            1.35361558e-01,
                            -6.37462344e-01,
                        ],
                        [
                            0.00000000e00,
                            -1.69967143e-01,
                            -1.95778638e-01,
                            -9.79165111e-01,
                            -1.84470262e-01,
                            -9.82748279e-01,
                            -1.83758244e-01,
                        ],
                        [
                            0.00000000e00,
                            -4.38924163e-17,
                            3.44245060e-18,
                            6.56556980e-18,
                            3.26553464e-17,
                            3.19571960e-18,
                            3.12169108e-17,
                        ],
                    ],
                    [
                        [
                            -7.73258609e-18,
                            2.31853293e-01,
                            8.08253690e-02,
                            5.06596965e-01,
                            9.83518983e-03,
                            1.99678676e-01,
                            0.00000000e00,
                        ],
                        [
                            5.05851038e-17,
                            -3.99892968e-02,
                            -1.39404950e-02,
                            -8.73761859e-02,
                            -1.69634134e-03,
                            -3.44399243e-02,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            -1.38777878e-17,
                            -1.38777878e-17,
                            5.55111512e-17,
                            0.00000000e00,
                            -1.38777878e-17,
                            0.00000000e00,
                        ],
                        [
                            -9.85449730e-01,
                            0.00000000e00,
                            -9.65806345e-01,
                            1.90658563e-01,
                            -7.10033786e-01,
                            1.24194307e-01,
                            -7.37360525e-01,
                        ],
                        [
                            1.69967143e-01,
                            0.00000000e00,
                            1.66579116e-01,
                            -3.28841647e-02,
                            1.22464303e-01,
                            -2.14206274e-02,
                            1.27177529e-01,
                        ],
                        [
                            4.38924163e-17,
                            0.00000000e00,
                            4.85722573e-17,
                            2.77555756e-17,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ],
                    [
                        [
                            1.58461042e-01,
                            -1.13719243e-01,
                            3.06951191e-02,
                            -3.45709946e-01,
                            1.01688580e-02,
                            -1.07973135e-01,
                            0.00000000e00,
                        ],
                        [
                            -4.37916236e-01,
                            -6.59330946e-01,
                            -4.24230421e-01,
                            -2.17748413e-02,
                            -7.82283735e-02,
                            -2.81325889e-02,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            1.38777878e-17,
                            1.73472348e-17,
                            0.00000000e00,
                            6.07153217e-18,
                            0.00000000e00,
                        ],
                        [
                            3.37672585e-02,
                            9.65806345e-01,
                            0.00000000e00,
                            5.39111251e-02,
                            -6.30795483e-01,
                            1.28407717e-01,
                            -5.99489263e-01,
                        ],
                        [
                            1.95778638e-01,
                            -1.66579116e-01,
                            0.00000000e00,
                            -9.97525009e-01,
                            -3.97312016e-02,
                            -9.87832342e-01,
                            -3.36043908e-02,
                        ],
                        [
                            -3.44245060e-18,
                            -4.85722573e-17,
                            0.00000000e00,
                            0.00000000e00,
                            4.16333634e-17,
                            0.00000000e00,
                            3.46944695e-17,
                        ],
                    ],
                    [
                        [
                            -3.12815864e-02,
                            3.53911729e-02,
                            -1.54782657e-03,
                            9.65242924e-02,
                            -1.45842251e-03,
                            3.24608603e-02,
                            0.00000000e00,
                        ],
                        [
                            8.64484696e-02,
                            -1.09310078e-01,
                            2.66964059e-04,
                            -5.18935898e-01,
                            5.28476698e-03,
                            -2.00682834e-01,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.38777878e-17,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            -6.16735653e-02,
                            -1.90658563e-01,
                            -5.39111251e-02,
                            0.00000000e00,
                            8.48905588e-02,
                            -1.84162966e-02,
                            7.71850655e-02,
                        ],
                        [
                            9.79165111e-01,
                            3.28841647e-02,
                            9.97525009e-01,
                            0.00000000e00,
                            7.41195745e-01,
                            6.67336352e-02,
                            7.68210453e-01,
                        ],
                        [
                            -6.56556980e-18,
                            -2.77555756e-17,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -2.77555756e-17,
                        ],
                    ],
                    [
                        [
                            1.16496122e-01,
                            -2.35033157e-01,
                            -3.02231459e-02,
                            -5.85029103e-01,
                            1.05221866e-03,
                            -2.09794626e-01,
                            0.00000000e00,
                        ],
                        [
                            -3.21943757e-01,
                            -4.94259600e-01,
                            -3.15207311e-01,
                            -3.68485664e-02,
                            -5.79159088e-02,
                            -2.88966443e-02,
                            0.00000000e00,
                        ],
                        [
                            1.38777878e-17,
                            0.00000000e00,
                            1.38777878e-17,
                            0.00000000e00,
                            0.00000000e00,
                            1.38777878e-17,
                            0.00000000e00,
                        ],
                        [
                            6.68449878e-01,
                            7.10033786e-01,
                            6.30795483e-01,
                            -8.48905588e-02,
                            0.00000000e00,
                            1.32869389e-02,
                            4.08633244e-02,
                        ],
                        [
                            1.84470262e-01,
                            -1.22464303e-01,
                            3.97312016e-02,
                            -7.41195745e-01,
                            0.00000000e00,
                            -7.31335770e-01,
                            5.62842325e-03,
                        ],
                        [
                            -3.26553464e-17,
                            0.00000000e00,
                            -4.16333634e-17,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ],
                    [
                        [
                            -2.03767136e-02,
                            4.54491056e-02,
                            6.79892209e-03,
                            1.11809337e-01,
                            0.00000000e00,
                            4.04324654e-02,
                            0.00000000e00,
                        ],
                        [
                            5.63122241e-02,
                            -1.52356663e-01,
                            -2.81163100e-02,
                            -5.15350265e-01,
                            -5.20417043e-18,
                            -2.00614904e-01,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            -5.55111512e-17,
                            1.38777878e-17,
                            -1.38777878e-17,
                            3.46944695e-18,
                            0.00000000e00,
                        ],
                        [
                            -1.35361558e-01,
                            -1.24194307e-01,
                            -1.28407717e-01,
                            1.84162966e-02,
                            -1.32869389e-02,
                            0.00000000e00,
                            -2.09458434e-02,
                        ],
                        [
                            9.82748279e-01,
                            2.14206274e-02,
                            9.87832342e-01,
                            -6.67336352e-02,
                            7.31335770e-01,
                            0.00000000e00,
                            7.58497864e-01,
                        ],
                        [
                            -3.19571960e-18,
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
                            1.20979654e-01,
                            -2.30735324e-01,
                            -2.67347299e-02,
                            -5.78389562e-01,
                            1.65874227e-03,
                            -2.06377139e-01,
                            0.00000000e00,
                        ],
                        [
                            -3.34334256e-01,
                            -5.11444043e-01,
                            -3.26697847e-01,
                            -3.42509562e-02,
                            -6.00669280e-02,
                            -2.84259272e-02,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            6.93889390e-18,
                            0.00000000e00,
                            0.00000000e00,
                            1.38777878e-17,
                            0.00000000e00,
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
                        [
                            1.83758244e-01,
                            -1.27177529e-01,
                            3.36043908e-02,
                            -7.68210453e-01,
                            -5.62842325e-03,
                            -7.58497864e-01,
                            0.00000000e00,
                        ],
                        [
                            -3.12169108e-17,
                            0.00000000e00,
                            -3.46944695e-17,
                            2.77555756e-17,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ],
                ],
                [
                    [
                        [
                            -1.07925929e-01,
                            2.31853293e-01,
                            8.08253690e-02,
                            5.06596965e-01,
                            9.83518983e-03,
                            1.99678676e-01,
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
                            1.10026604e-17,
                            -1.03268599e-17,
                            -3.60000174e-18,
                            -2.25640783e-17,
                            -4.38064199e-19,
                            -8.89378659e-18,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            -9.65806345e-01,
                            1.90658563e-01,
                            -7.10033786e-01,
                            1.24194307e-01,
                            -7.37360525e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            1.66579116e-01,
                            -3.28841647e-02,
                            1.22464303e-01,
                            -2.14206274e-02,
                            1.27177529e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            4.30174903e-17,
                            -8.49202631e-18,
                            3.16252545e-17,
                            -5.53167558e-18,
                            3.28424010e-17,
                        ],
                    ],
                    [
                        [
                            1.26880993e-17,
                            -1.07925929e-01,
                            -6.87706209e-02,
                            3.80041092e-03,
                            -1.27256780e-02,
                            -1.00921994e-03,
                            0.00000000e00,
                        ],
                        [
                            7.35641246e-17,
                            -6.25741987e-01,
                            -3.98724063e-01,
                            2.20343406e-02,
                            -7.37820011e-02,
                            -5.85133986e-03,
                            0.00000000e00,
                        ],
                        [
                            -1.24900090e-16,
                            2.35276631e-01,
                            8.20187641e-02,
                            5.14076923e-01,
                            9.98040745e-03,
                            2.02626953e-01,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            -3.37672585e-02,
                            -1.62222678e-01,
                            -5.02084106e-02,
                            -1.60694077e-01,
                            -4.91939581e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            -1.95778638e-01,
                            -9.40548220e-01,
                            -2.91102526e-01,
                            -9.31685572e-01,
                            -2.85220849e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            -9.80066578e-01,
                            1.93473657e-01,
                            -7.20517510e-01,
                            1.26028049e-01,
                            -7.48247732e-01,
                        ],
                    ],
                    [
                        [
                            -1.70045802e-02,
                            -4.16333634e-17,
                            3.52650006e-03,
                            -1.16102817e-02,
                            8.03370202e-04,
                            -3.50123667e-03,
                            0.00000000e00,
                        ],
                        [
                            -5.81147164e-01,
                            3.81639165e-17,
                            2.04462373e-02,
                            -6.73150634e-02,
                            4.65784701e-03,
                            -2.02997631e-02,
                            0.00000000e00,
                        ],
                        [
                            3.56738040e-01,
                            -9.54097912e-17,
                            1.02353729e-01,
                            -3.36978765e-01,
                            2.33171516e-02,
                            -1.01620481e-01,
                            0.00000000e00,
                        ],
                        [
                            9.65806345e-01,
                            3.37672585e-02,
                            0.00000000e00,
                            7.67199187e-03,
                            -2.11845605e-02,
                            1.01445937e-02,
                            -2.01575078e-02,
                        ],
                        [
                            -1.66579116e-01,
                            1.95778638e-01,
                            0.00000000e00,
                            4.44813167e-02,
                            -1.22825619e-01,
                            5.88171749e-02,
                            -1.16870886e-01,
                        ],
                        [
                            -4.30174903e-17,
                            9.80066578e-01,
                            0.00000000e00,
                            2.22673179e-01,
                            -6.14864240e-01,
                            2.94438391e-01,
                            -5.85054890e-01,
                        ],
                    ],
                    [
                        [
                            4.47344513e-01,
                            -2.45209972e-02,
                            1.31697505e-03,
                            -5.49139714e-02,
                            9.68203187e-04,
                            -1.70497322e-02,
                            0.00000000e00,
                        ],
                        [
                            2.75381501e-01,
                            -1.42169890e-01,
                            7.63566820e-03,
                            -3.18384821e-01,
                            5.61352949e-03,
                            -9.88523645e-02,
                            0.00000000e00,
                        ],
                        [
                            5.60528715e-01,
                            5.34553435e-02,
                            -1.57068039e-03,
                            1.83321890e-01,
                            -2.33543881e-03,
                            6.60980339e-02,
                            0.00000000e00,
                        ],
                        [
                            -1.90658563e-01,
                            1.62222678e-01,
                            -7.67199187e-03,
                            0.00000000e00,
                            -1.13181076e-01,
                            1.22260298e-02,
                            -1.08016484e-01,
                        ],
                        [
                            3.28841647e-02,
                            9.40548220e-01,
                            -4.44813167e-02,
                            0.00000000e00,
                            -6.56210717e-01,
                            7.08850988e-02,
                            -6.26267014e-01,
                        ],
                        [
                            8.49202631e-18,
                            -1.93473657e-01,
                            -2.22673179e-01,
                            0.00000000e00,
                            -4.23235448e-02,
                            -2.94908598e-02,
                            -5.45085339e-02,
                        ],
                    ],
                    [
                        [
                            5.42292833e-02,
                            6.77094759e-02,
                            4.83881552e-02,
                            -1.96475479e-02,
                            9.17823507e-03,
                            -4.57282155e-03,
                            0.00000000e00,
                        ],
                        [
                            -4.03096507e-01,
                            3.92571668e-01,
                            2.80548897e-01,
                            -1.13914198e-01,
                            5.32143397e-02,
                            -2.65126876e-02,
                            0.00000000e00,
                        ],
                        [
                            3.57094795e-01,
                            -1.47605469e-01,
                            2.37914951e-02,
                            -5.70253726e-01,
                            1.08807101e-02,
                            -2.01830577e-01,
                            0.00000000e00,
                        ],
                        [
                            7.10033786e-01,
                            5.02084106e-02,
                            2.11845605e-02,
                            1.13181076e-01,
                            0.00000000e00,
                            1.15898581e-01,
                            8.90683876e-04,
                        ],
                        [
                            -1.22464303e-01,
                            2.91102526e-01,
                            1.22825619e-01,
                            6.56210717e-01,
                            0.00000000e00,
                            6.71966493e-01,
                            5.16408154e-03,
                        ],
                        [
                            -3.16252545e-17,
                            7.20517510e-01,
                            6.14864240e-01,
                            4.23235448e-02,
                            0.00000000e00,
                            1.37396662e-01,
                            3.93121050e-02,
                        ],
                    ],
                    [
                        [
                            4.36046879e-01,
                            -3.24238553e-02,
                            -3.87838234e-03,
                            -5.41101044e-02,
                            0.00000000e00,
                            -1.69651356e-02,
                            0.00000000e00,
                        ],
                        [
                            2.31723847e-01,
                            -1.87989743e-01,
                            -2.24864098e-02,
                            -3.13724093e-01,
                            5.20417043e-18,
                            -9.83618836e-02,
                            0.00000000e00,
                        ],
                        [
                            5.70686384e-01,
                            7.06834356e-02,
                            1.14788448e-02,
                            1.97775093e-01,
                            8.67361738e-19,
                            7.39421042e-02,
                            0.00000000e00,
                        ],
                        [
                            -1.24194307e-01,
                            1.60694077e-01,
                            -1.01445937e-02,
                            -1.22260298e-02,
                            -1.15898581e-01,
                            0.00000000e00,
                            -1.10706199e-01,
                        ],
                        [
                            2.14206274e-02,
                            9.31685572e-01,
                            -5.88171749e-02,
                            -7.08850988e-02,
                            -6.71966493e-01,
                            0.00000000e00,
                            -6.41861667e-01,
                        ],
                        [
                            5.53167558e-18,
                            -1.26028049e-01,
                            -2.94438391e-01,
                            2.94908598e-02,
                            -1.37396662e-01,
                            0.00000000e00,
                            -1.49560791e-01,
                        ],
                    ],
                    [
                        [
                            4.85696011e-02,
                            6.44268400e-02,
                            4.61905080e-02,
                            -1.91831542e-02,
                            8.76704022e-03,
                            -4.49833175e-03,
                            0.00000000e00,
                        ],
                        [
                            -4.21413485e-01,
                            3.73539326e-01,
                            2.67807194e-01,
                            -1.11221697e-01,
                            5.08302797e-02,
                            -2.60808044e-02,
                            0.00000000e00,
                        ],
                        [
                            3.59829170e-01,
                            -1.40449380e-01,
                            2.91821673e-02,
                            -5.64152301e-01,
                            1.18440113e-02,
                            -1.98542822e-01,
                            0.00000000e00,
                        ],
                        [
                            7.37360525e-01,
                            4.91939581e-02,
                            2.01575078e-02,
                            1.08016484e-01,
                            -8.90683876e-04,
                            1.10706199e-01,
                            0.00000000e00,
                        ],
                        [
                            -1.27177529e-01,
                            2.85220849e-01,
                            1.16870886e-01,
                            6.26267014e-01,
                            -5.16408154e-03,
                            6.41861667e-01,
                            0.00000000e00,
                        ],
                        [
                            -3.28424010e-17,
                            7.48247732e-01,
                            5.85054890e-01,
                            5.45085339e-02,
                            -3.93121050e-02,
                            1.49560791e-01,
                            0.00000000e00,
                        ],
                    ],
                ],
                [
                    [
                        [
                            3.41453006e-02,
                            8.08253690e-02,
                            3.06951191e-02,
                            -3.45709946e-01,
                            1.01688580e-02,
                            -1.07973135e-01,
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
                            2.42699171e-18,
                            -3.60000174e-18,
                            2.66095901e-18,
                            1.61807814e-17,
                            2.82387683e-19,
                            5.25873718e-18,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            5.39111251e-02,
                            -6.30795483e-01,
                            1.28407717e-01,
                            -5.99489263e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -9.97525009e-01,
                            -3.97312016e-02,
                            -9.87832342e-01,
                            -3.36043908e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            7.10071903e-18,
                            2.95240677e-17,
                            3.56586331e-18,
                            2.80188451e-17,
                        ],
                    ],
                    [
                        [
                            8.10577678e-19,
                            -6.87706209e-02,
                            3.52650006e-03,
                            -1.16102817e-02,
                            8.03370202e-04,
                            -3.50123667e-03,
                            0.00000000e00,
                        ],
                        [
                            4.99661867e-17,
                            -3.98724063e-01,
                            2.04462373e-02,
                            -6.73150634e-02,
                            4.65784701e-03,
                            -2.02997631e-02,
                            0.00000000e00,
                        ],
                        [
                            1.04435486e-01,
                            8.20187641e-02,
                            1.02353729e-01,
                            -3.36978765e-01,
                            2.33171516e-02,
                            -1.01620481e-01,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            7.67199187e-03,
                            -2.11845605e-02,
                            1.01445937e-02,
                            -2.01575078e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            4.44813167e-02,
                            -1.22825619e-01,
                            5.88171749e-02,
                            -1.16870886e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            2.22673179e-01,
                            -6.14864240e-01,
                            2.94438391e-01,
                            -5.85054890e-01,
                        ],
                    ],
                    [
                        [
                            3.34646679e-02,
                            0.00000000e00,
                            3.41453006e-02,
                            -3.52192202e-01,
                            1.08915293e-02,
                            -1.09853801e-01,
                            0.00000000e00,
                        ],
                        [
                            -4.08172899e-01,
                            3.81639165e-17,
                            -4.16474664e-01,
                            -1.90341873e-02,
                            -7.68286193e-02,
                            -2.68762218e-02,
                            0.00000000e00,
                        ],
                        [
                            8.03838495e-02,
                            -7.37257477e-18,
                            8.20187641e-02,
                            1.59367259e-02,
                            1.49720700e-02,
                            9.15371670e-03,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            6.16735653e-02,
                            -6.42623254e-01,
                            1.37533281e-01,
                            -6.10758051e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -9.79165111e-01,
                            -3.47305003e-02,
                            -9.70156883e-01,
                            -2.89298617e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.93473657e-01,
                            2.90787547e-02,
                            1.89060494e-01,
                            2.68221308e-02,
                        ],
                    ],
                    [
                        [
                            -2.67749497e-02,
                            -4.27379584e-01,
                            0.00000000e00,
                            3.13469337e-02,
                            -3.14115013e-04,
                            1.21748711e-02,
                            0.00000000e00,
                        ],
                        [
                            7.32787717e-02,
                            -2.52079454e-01,
                            1.38777878e-17,
                            -4.97682007e-01,
                            4.98707121e-03,
                            -1.93295280e-01,
                            0.00000000e00,
                        ],
                        [
                            -4.45303046e-02,
                            -5.99740038e-01,
                            0.00000000e00,
                            9.83372027e-02,
                            -9.85397553e-04,
                            3.81932978e-02,
                            0.00000000e00,
                        ],
                        [
                            -5.39111251e-02,
                            -7.67199187e-03,
                            -6.16735653e-02,
                            0.00000000e00,
                            -4.71705446e-02,
                            -3.96650160e-03,
                            -4.87735455e-02,
                        ],
                        [
                            9.97525009e-01,
                            -4.44813167e-02,
                            9.79165111e-01,
                            0.00000000e00,
                            7.48906785e-01,
                            6.29744683e-02,
                            7.74356953e-01,
                        ],
                        [
                            -7.10071903e-18,
                            -2.22673179e-01,
                            -1.93473657e-01,
                            0.00000000e00,
                            -1.47976815e-01,
                            -1.24431524e-02,
                            -1.53005525e-01,
                        ],
                    ],
                    [
                        [
                            8.02939201e-02,
                            -8.68221386e-02,
                            2.61157664e-02,
                            -5.95998728e-01,
                            1.16033016e-02,
                            -2.10879959e-01,
                            0.00000000e00,
                        ],
                        [
                            -2.79925137e-01,
                            2.37618361e-01,
                            -3.18537393e-01,
                            -3.22106832e-02,
                            -5.85848803e-02,
                            -2.74121559e-02,
                            0.00000000e00,
                        ],
                        [
                            1.38239466e-01,
                            -1.44396771e-01,
                            6.27314110e-02,
                            2.69689911e-02,
                            1.13031672e-02,
                            1.27415350e-02,
                            0.00000000e00,
                        ],
                        [
                            6.30795483e-01,
                            2.11845605e-02,
                            6.42623254e-01,
                            4.71705446e-02,
                            0.00000000e00,
                            1.46521218e-01,
                            4.10747232e-02,
                        ],
                        [
                            3.97312016e-02,
                            1.22825619e-01,
                            3.47305003e-02,
                            -7.48906785e-01,
                            0.00000000e00,
                            -7.39783239e-01,
                            5.33927794e-03,
                        ],
                        [
                            -2.95240677e-17,
                            6.14864240e-01,
                            -2.90787547e-02,
                            1.47976815e-01,
                            0.00000000e00,
                            1.42731258e-01,
                            -2.48176748e-03,
                        ],
                    ],
                    [
                        [
                            -3.09721687e-02,
                            -4.14953750e-01,
                            2.19603632e-03,
                            4.72532385e-02,
                            1.73472348e-18,
                            2.00850184e-02,
                            0.00000000e00,
                        ],
                        [
                            4.28372826e-02,
                            -2.78984158e-01,
                            -2.67853401e-02,
                            -4.94327552e-01,
                            -5.20417043e-18,
                            -1.93245510e-01,
                            0.00000000e00,
                        ],
                        [
                            -4.82359054e-02,
                            -5.79629005e-01,
                            5.27499193e-03,
                            9.71190784e-02,
                            1.38777878e-17,
                            3.79108193e-02,
                            0.00000000e00,
                        ],
                        [
                            -1.28407717e-01,
                            -1.01445937e-02,
                            -1.37533281e-01,
                            3.96650160e-03,
                            -1.46521218e-01,
                            0.00000000e00,
                            -1.48046549e-01,
                        ],
                        [
                            9.87832342e-01,
                            -5.88171749e-02,
                            9.70156883e-01,
                            -6.29744683e-02,
                            7.39783239e-01,
                            0.00000000e00,
                            7.65372332e-01,
                        ],
                        [
                            -3.56586331e-18,
                            -2.94438391e-01,
                            -1.89060494e-01,
                            1.24431524e-02,
                            -1.42731258e-01,
                            0.00000000e00,
                            -1.47790395e-01,
                        ],
                    ],
                    [
                        [
                            7.85407802e-02,
                            -8.07082629e-02,
                            2.70032608e-02,
                            -5.88956619e-01,
                            1.17240955e-02,
                            -2.07444792e-01,
                            0.00000000e00,
                        ],
                        [
                            -2.92451072e-01,
                            2.26787537e-01,
                            -3.29362278e-01,
                            -2.97571131e-02,
                            -6.06113306e-02,
                            -2.69656206e-02,
                            0.00000000e00,
                        ],
                        [
                            1.36676893e-01,
                            -1.34689562e-01,
                            6.48632182e-02,
                            2.62362355e-02,
                            1.17038102e-02,
                            1.25339795e-02,
                            0.00000000e00,
                        ],
                        [
                            5.99489263e-01,
                            2.01575078e-02,
                            6.10758051e-01,
                            4.87735455e-02,
                            -4.10747232e-02,
                            1.48046549e-01,
                            0.00000000e00,
                        ],
                        [
                            3.36043908e-02,
                            1.16870886e-01,
                            2.89298617e-02,
                            -7.74356953e-01,
                            -5.33927794e-03,
                            -7.65372332e-01,
                            0.00000000e00,
                        ],
                        [
                            -2.80188451e-17,
                            5.85054890e-01,
                            -2.68221308e-02,
                            1.53005525e-01,
                            2.48176748e-03,
                            1.47790395e-01,
                            0.00000000e00,
                        ],
                    ],
                ],
                [
                    [
                        [
                            -3.35029257e-01,
                            5.06596965e-01,
                            -3.45709946e-01,
                            9.65242924e-02,
                            -1.45842251e-03,
                            3.24608603e-02,
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
                            1.47042021e-17,
                            -2.25640783e-17,
                            1.61807814e-17,
                            5.30332170e-19,
                            1.65636617e-20,
                            4.29940126e-19,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            8.48905588e-02,
                            -1.84162966e-02,
                            7.71850655e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            7.41195745e-01,
                            6.67336352e-02,
                            7.68210453e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.10485573e-17,
                            2.09158392e-19,
                            -1.09523226e-17,
                        ],
                    ],
                    [
                        [
                            2.35023767e-17,
                            3.80041092e-03,
                            -1.16102817e-02,
                            -5.49139714e-02,
                            9.68203187e-04,
                            -1.70497322e-02,
                            0.00000000e00,
                        ],
                        [
                            -1.27668360e-17,
                            2.20343406e-02,
                            -6.73150634e-02,
                            -3.18384821e-01,
                            5.61352949e-03,
                            -9.88523645e-02,
                            0.00000000e00,
                        ],
                        [
                            -3.43832524e-01,
                            5.14076923e-01,
                            -3.36978765e-01,
                            1.83321890e-01,
                            -2.33543881e-03,
                            6.60980339e-02,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.13181076e-01,
                            1.22260298e-02,
                            -1.08016484e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -6.56210717e-01,
                            7.08850988e-02,
                            -6.26267014e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -4.23235448e-02,
                            -2.94908598e-02,
                            -5.45085339e-02,
                        ],
                    ],
                    [
                        [
                            -3.28350978e-01,
                            5.00876302e-01,
                            -3.52192202e-01,
                            3.13469337e-02,
                            -3.14115013e-04,
                            1.21748711e-02,
                            0.00000000e00,
                        ],
                        [
                            7.88704387e-02,
                            -8.63895046e-02,
                            -1.90341873e-02,
                            -4.97682007e-01,
                            4.98707121e-03,
                            -1.93295280e-01,
                            0.00000000e00,
                        ],
                        [
                            -4.44218266e-03,
                            -6.93889390e-18,
                            1.59367259e-02,
                            9.83372027e-02,
                            -9.85397553e-04,
                            3.81932978e-02,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -4.71705446e-02,
                            -3.96650160e-03,
                            -4.87735455e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            7.48906785e-01,
                            6.29744683e-02,
                            7.74356953e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.47976815e-01,
                            -1.24431524e-02,
                            -1.53005525e-01,
                        ],
                    ],
                    [
                        [
                            6.48193356e-02,
                            -7.61193490e-02,
                            0.00000000e00,
                            -3.35029257e-01,
                            5.85989604e-03,
                            -1.04502264e-01,
                            0.00000000e00,
                        ],
                        [
                            -1.55697098e-02,
                            1.82839914e-02,
                            1.90819582e-17,
                            8.04745724e-02,
                            -6.71145042e-04,
                            3.26402475e-02,
                            0.00000000e00,
                        ],
                        [
                            -9.94603424e-02,
                            1.16799354e-01,
                            9.71445147e-17,
                            5.14076923e-01,
                            -5.26460556e-03,
                            1.98503607e-01,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -6.68449878e-01,
                            7.39961036e-02,
                            -6.37206328e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.84470262e-01,
                            -8.47491453e-03,
                            -1.87822895e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -7.20517510e-01,
                            -6.64790460e-02,
                            -7.47444596e-01,
                        ],
                    ],
                    [
                        [
                            5.61448369e-02,
                            8.05790870e-01,
                            2.26019439e-02,
                            2.77555756e-17,
                            5.26713933e-02,
                            1.42461283e-02,
                            0.00000000e00,
                        ],
                        [
                            1.65648643e-01,
                            2.46157927e-02,
                            1.13501279e-01,
                            8.32667268e-17,
                            1.45355786e-02,
                            3.93146458e-03,
                            0.00000000e00,
                        ],
                        [
                            5.26562926e-01,
                            1.50442818e-01,
                            5.40897217e-01,
                            1.52655666e-16,
                            5.67741313e-02,
                            1.53558033e-02,
                            0.00000000e00,
                        ],
                        [
                            -8.48905588e-02,
                            1.13181076e-01,
                            4.71705446e-02,
                            6.68449878e-01,
                            0.00000000e00,
                            6.65110413e-01,
                            -2.77482876e-03,
                        ],
                        [
                            -7.41195745e-01,
                            6.56210717e-01,
                            -7.48906785e-01,
                            1.84470262e-01,
                            0.00000000e00,
                            1.83548679e-01,
                            -7.65761810e-04,
                        ],
                        [
                            1.10485573e-17,
                            4.23235448e-02,
                            1.47976815e-01,
                            7.20517510e-01,
                            0.00000000e00,
                            7.16917924e-01,
                            -2.99096877e-03,
                        ],
                    ],
                    [
                        [
                            1.00823383e-02,
                            -9.12151146e-02,
                            -5.50296030e-02,
                            -3.33355506e-01,
                            6.93889390e-18,
                            -1.05557206e-01,
                            0.00000000e00,
                        ],
                        [
                            -2.17722592e-02,
                            3.75026544e-02,
                            -3.48488893e-03,
                            8.00725347e-02,
                            1.51788304e-18,
                            3.26578012e-02,
                            0.00000000e00,
                        ],
                        [
                            -9.16405905e-02,
                            1.94212992e-01,
                            3.02504923e-03,
                            5.11508680e-01,
                            5.20417043e-18,
                            1.98928729e-01,
                            0.00000000e00,
                        ],
                        [
                            1.84162966e-02,
                            -1.22260298e-02,
                            3.96650160e-03,
                            -7.39961036e-02,
                            -6.65110413e-01,
                            0.00000000e00,
                            -6.33715782e-01,
                        ],
                        [
                            -6.67336352e-02,
                            -7.08850988e-02,
                            -6.29744683e-02,
                            8.47491453e-03,
                            -1.83548679e-01,
                            0.00000000e00,
                            -1.86919743e-01,
                        ],
                        [
                            -2.09158392e-19,
                            2.94908598e-02,
                            1.24431524e-02,
                            6.64790460e-02,
                            -7.16917924e-01,
                            0.00000000e00,
                            -7.43986450e-01,
                        ],
                    ],
                    [
                        [
                            3.32773173e-02,
                            7.98996386e-01,
                            -2.22953651e-04,
                            1.39075322e-03,
                            5.01851912e-02,
                            1.40140634e-02,
                            0.00000000e00,
                        ],
                        [
                            1.62967320e-01,
                            3.25210512e-02,
                            1.11958911e-01,
                            -3.34061184e-04,
                            1.48025397e-02,
                            3.86742228e-03,
                            0.00000000e00,
                        ],
                        [
                            5.29555141e-01,
                            1.82151411e-01,
                            5.41683062e-01,
                            -2.13400507e-03,
                            5.89177408e-02,
                            1.51056622e-02,
                            0.00000000e00,
                        ],
                        [
                            -7.71850655e-02,
                            1.08016484e-01,
                            4.87735455e-02,
                            6.37206328e-01,
                            2.77482876e-03,
                            6.33715782e-01,
                            0.00000000e00,
                        ],
                        [
                            -7.68210453e-01,
                            6.26267014e-01,
                            -7.74356953e-01,
                            1.87822895e-01,
                            7.65761810e-04,
                            1.86919743e-01,
                            0.00000000e00,
                        ],
                        [
                            1.09523226e-17,
                            5.45085339e-02,
                            1.53005525e-01,
                            7.47444596e-01,
                            2.99096877e-03,
                            7.43986450e-01,
                            0.00000000e00,
                        ],
                    ],
                ],
                [
                    [
                        [
                            1.07195463e-02,
                            9.83518983e-03,
                            1.01688580e-02,
                            -1.45842251e-03,
                            1.05221866e-03,
                            -2.09794626e-01,
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
                            2.53075280e-19,
                            -4.38064199e-19,
                            2.82387683e-19,
                            1.65636617e-20,
                            5.08258833e-19,
                            9.97012277e-18,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.32869389e-02,
                            4.08633244e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -7.31335770e-01,
                            5.62842325e-03,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            6.41806152e-18,
                            -1.94195804e-18,
                        ],
                    ],
                    [
                        [
                            -1.61086476e-19,
                            -1.27256780e-02,
                            8.03370202e-04,
                            9.68203187e-04,
                            9.17823507e-03,
                            -4.57282155e-03,
                            0.00000000e00,
                        ],
                        [
                            9.37818968e-18,
                            -7.37820011e-02,
                            4.65784701e-03,
                            5.61352949e-03,
                            5.32143397e-02,
                            -2.65126876e-02,
                            0.00000000e00,
                        ],
                        [
                            2.37913955e-02,
                            9.98040745e-03,
                            2.33171516e-02,
                            -2.33543881e-03,
                            1.08807101e-02,
                            -2.01830577e-01,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.15898581e-01,
                            8.90683876e-04,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            6.71966493e-01,
                            5.16408154e-03,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.37396662e-01,
                            3.93121050e-02,
                        ],
                    ],
                    [
                        [
                            1.05058690e-02,
                            -5.01907994e-03,
                            1.08915293e-02,
                            -3.14115013e-04,
                            1.16033016e-02,
                            -2.10879959e-01,
                            0.00000000e00,
                        ],
                        [
                            -7.62744229e-02,
                            8.65674476e-04,
                            -7.68286193e-02,
                            4.98707121e-03,
                            -5.85848803e-02,
                            -2.74121559e-02,
                            0.00000000e00,
                        ],
                        [
                            1.48746510e-02,
                            -1.24683250e-18,
                            1.49720700e-02,
                            -9.85397553e-04,
                            1.13031672e-02,
                            1.27415350e-02,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.46521218e-01,
                            4.10747232e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -7.39783239e-01,
                            5.33927794e-03,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.42731258e-01,
                            -2.48176748e-03,
                        ],
                    ],
                    [
                        [
                            -2.07394982e-03,
                            -7.52143130e-02,
                            2.66073585e-03,
                            5.85989604e-03,
                            5.26713933e-02,
                            1.42461283e-02,
                            0.00000000e00,
                        ],
                        [
                            1.50572337e-02,
                            -4.28938883e-03,
                            1.54266371e-02,
                            -6.71145042e-04,
                            1.45355786e-02,
                            3.93146458e-03,
                            0.00000000e00,
                        ],
                        [
                            7.68653783e-02,
                            2.26756948e-03,
                            7.72256438e-02,
                            -5.26460556e-03,
                            5.67741313e-02,
                            1.53558033e-02,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            6.65110413e-01,
                            -2.77482876e-03,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.83548679e-01,
                            -7.65761810e-04,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            7.16917924e-01,
                            -2.99096877e-03,
                        ],
                    ],
                    [
                        [
                            7.72362080e-03,
                            -6.72512034e-03,
                            8.19876123e-03,
                            8.67361738e-19,
                            1.07195463e-02,
                            -1.56123717e-01,
                            0.00000000e00,
                        ],
                        [
                            -5.60748204e-02,
                            4.88255347e-02,
                            -5.95244219e-02,
                            -6.07153217e-18,
                            -7.78257566e-02,
                            -2.83646224e-03,
                            0.00000000e00,
                        ],
                        [
                            7.19105833e-03,
                            -6.26140691e-03,
                            7.63343666e-03,
                            7.58941521e-19,
                            9.98040745e-03,
                            1.45567764e-01,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.35361558e-01,
                            3.04094258e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -9.82748279e-01,
                            5.52479723e-04,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.26028049e-01,
                            -2.83533610e-02,
                        ],
                    ],
                    [
                        [
                            -3.06034236e-01,
                            -4.98379189e-01,
                            -2.79535843e-01,
                            6.74871090e-02,
                            -6.93889390e-18,
                            1.07195463e-02,
                            0.00000000e00,
                        ],
                        [
                            -1.00442146e-01,
                            -1.08393727e-01,
                            -7.27677599e-02,
                            -1.47405316e-01,
                            -8.67361738e-18,
                            -7.78257566e-02,
                            0.00000000e00,
                        ],
                        [
                            -2.50991746e-01,
                            -4.53195110e-01,
                            -2.26781234e-01,
                            5.79144367e-02,
                            -6.93889390e-18,
                            9.98040745e-03,
                            0.00000000e00,
                        ],
                        [
                            -1.32869389e-02,
                            -1.15898581e-01,
                            -1.46521218e-01,
                            -6.65110413e-01,
                            -1.35361558e-01,
                            0.00000000e00,
                            -1.35244491e-01,
                        ],
                        [
                            7.31335770e-01,
                            -6.71966493e-01,
                            7.39783239e-01,
                            -1.83548679e-01,
                            9.82748279e-01,
                            0.00000000e00,
                            9.81898349e-01,
                        ],
                        [
                            -6.41806152e-18,
                            -1.37396662e-01,
                            -1.42731258e-01,
                            -7.16917924e-01,
                            -1.26028049e-01,
                            0.00000000e00,
                            -1.25919053e-01,
                        ],
                    ],
                    [
                        [
                            5.67937662e-03,
                            -6.39875381e-05,
                            9.64647527e-03,
                            2.13841071e-02,
                            1.07102755e-02,
                            -1.53580512e-01,
                            0.00000000e00,
                        ],
                        [
                            -5.90802231e-02,
                            4.83482186e-02,
                            -6.12645420e-02,
                            4.97217187e-03,
                            -7.77584490e-02,
                            -2.79025718e-03,
                            0.00000000e00,
                        ],
                        [
                            3.37578191e-03,
                            1.97430655e-03,
                            8.84164962e-03,
                            1.31960342e-02,
                            9.97177590e-03,
                            1.43196511e-01,
                            0.00000000e00,
                        ],
                        [
                            -4.08633244e-02,
                            -8.90683876e-04,
                            -4.10747232e-02,
                            2.77482876e-03,
                            -3.04094258e-02,
                            1.35244491e-01,
                            0.00000000e00,
                        ],
                        [
                            -5.62842325e-03,
                            -5.16408154e-03,
                            -5.33927794e-03,
                            7.65761810e-04,
                            -5.52479723e-04,
                            -9.81898349e-01,
                            0.00000000e00,
                        ],
                        [
                            1.94195804e-18,
                            -3.93121050e-02,
                            2.48176748e-03,
                            2.99096877e-03,
                            2.83533610e-02,
                            1.25919053e-01,
                            0.00000000e00,
                        ],
                    ],
                ],
                [
                    [
                        [
                            -1.03187865e-01,
                            1.99678676e-01,
                            -1.07973135e-01,
                            3.24608603e-02,
                            -2.09794626e-01,
                            4.04324654e-02,
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
                            4.65397353e-18,
                            -8.89378659e-18,
                            5.25873718e-18,
                            4.29940126e-19,
                            9.97012277e-18,
                            6.10077242e-20,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -2.09458434e-02,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            7.58497864e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -6.32539630e-18,
                        ],
                    ],
                    [
                        [
                            7.34083974e-18,
                            -1.00921994e-03,
                            -3.50123667e-03,
                            -1.70497322e-02,
                            -4.57282155e-03,
                            -1.69651356e-02,
                            0.00000000e00,
                        ],
                        [
                            -2.38093275e-18,
                            -5.85133986e-03,
                            -2.02997631e-02,
                            -9.88523645e-02,
                            -2.65126876e-02,
                            -9.83618836e-02,
                            0.00000000e00,
                        ],
                        [
                            -1.03687324e-01,
                            2.02626953e-01,
                            -1.01620481e-01,
                            6.60980339e-02,
                            -2.01830577e-01,
                            7.39421042e-02,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.10706199e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -6.41861667e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.49560791e-01,
                        ],
                    ],
                    [
                        [
                            -1.01130978e-01,
                            1.94535915e-01,
                            -1.09853801e-01,
                            1.21748711e-02,
                            -2.10879959e-01,
                            2.00850184e-02,
                            0.00000000e00,
                        ],
                        [
                            1.15374401e-02,
                            -3.35529177e-02,
                            -2.68762218e-02,
                            -1.93295280e-01,
                            -2.74121559e-02,
                            -1.93245510e-01,
                            0.00000000e00,
                        ],
                        [
                            1.17964594e-03,
                            -3.46944695e-18,
                            9.15371670e-03,
                            3.81932978e-02,
                            1.27415350e-02,
                            3.79108193e-02,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.48046549e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            7.65372332e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.47790395e-01,
                        ],
                    ],
                    [
                        [
                            1.99641336e-02,
                            -4.44465872e-02,
                            7.19654101e-04,
                            -1.04502264e-01,
                            1.42461283e-02,
                            -1.05557206e-01,
                            0.00000000e00,
                        ],
                        [
                            -2.27759091e-03,
                            6.29701679e-03,
                            4.17247079e-03,
                            3.26402475e-02,
                            3.93146458e-03,
                            3.26578012e-02,
                            0.00000000e00,
                        ],
                        [
                            -1.78907915e-02,
                            4.60372682e-02,
                            2.08873614e-02,
                            1.98503607e-01,
                            1.53558033e-02,
                            1.98928729e-01,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -6.33715782e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.86919743e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -7.43986450e-01,
                        ],
                    ],
                    [
                        [
                            -7.43486637e-02,
                            1.42776647e-01,
                            -8.15965276e-02,
                            4.88405015e-03,
                            -1.56123717e-01,
                            1.07195463e-02,
                            0.00000000e00,
                        ],
                        [
                            8.48200297e-03,
                            -2.08454900e-02,
                            -6.50029553e-03,
                            -7.75419986e-02,
                            -2.83646224e-03,
                            -7.78257566e-02,
                            0.00000000e00,
                        ],
                        [
                            6.68043138e-02,
                            -1.27122045e-01,
                            7.73642547e-02,
                            1.53215570e-02,
                            1.45567764e-01,
                            9.98040745e-03,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.35244491e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            9.81898349e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.25919053e-01,
                        ],
                    ],
                    [
                        [
                            1.30045653e-02,
                            -3.10004133e-02,
                            -6.63647108e-03,
                            -1.02672356e-01,
                            3.46944695e-18,
                            -1.03187865e-01,
                            0.00000000e00,
                        ],
                        [
                            -1.48361458e-03,
                            3.53665533e-03,
                            7.57116061e-04,
                            1.17132869e-02,
                            7.80625564e-18,
                            1.17720983e-02,
                            0.00000000e00,
                        ],
                        [
                            -2.55366795e-02,
                            6.08745931e-02,
                            1.30318416e-02,
                            2.01614663e-01,
                            5.20417043e-17,
                            2.02626953e-01,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -6.37462344e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -1.83758244e-01,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            -7.48247732e-01,
                        ],
                    ],
                    [
                        [
                            2.13823046e-01,
                            5.50959991e-01,
                            1.84972186e-01,
                            -5.73283670e-02,
                            -1.11440324e-01,
                            5.20417043e-18,
                            0.00000000e00,
                        ],
                        [
                            1.14119484e-01,
                            8.61759621e-02,
                            7.65324714e-02,
                            8.54932885e-02,
                            1.00807085e-02,
                            1.38777878e-17,
                            0.00000000e00,
                        ],
                        [
                            4.05916016e-01,
                            3.51404343e-01,
                            3.90608748e-01,
                            -5.18727125e-02,
                            1.98301371e-01,
                            4.51028104e-17,
                            0.00000000e00,
                        ],
                        [
                            2.09458434e-02,
                            1.10706199e-01,
                            1.48046549e-01,
                            6.33715782e-01,
                            1.35244491e-01,
                            6.37462344e-01,
                            0.00000000e00,
                        ],
                        [
                            -7.58497864e-01,
                            6.41861667e-01,
                            -7.65372332e-01,
                            1.86919743e-01,
                            -9.81898349e-01,
                            1.83758244e-01,
                            0.00000000e00,
                        ],
                        [
                            6.32539630e-18,
                            1.49560791e-01,
                            1.47790395e-01,
                            7.43986450e-01,
                            1.25919053e-01,
                            7.48247732e-01,
                            0.00000000e00,
                        ],
                    ],
                ],
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ],
            ]
        )

        fk = r.partial_fkine0(q1, 3)
        fk2 = r2.partial_fkine0(q1, 3)

        nt.assert_almost_equal(fk, ans)
        nt.assert_almost_equal(fk2, ans)

    def test_qlim1(self):
        rx = rtb.ETS(rtb.ET.Rx())

        q = rx.qlim
        nt.assert_equal(q, np.array([[-np.pi], [np.pi]]))

    def test_qlim2(self):
        rx = rtb.ETS(rtb.ET.Rx(qlim=[-1, 1]))

        q = rx.qlim
        nt.assert_equal(q, np.array([[-1], [1]]))

    def test_qlim3(self):
        rx = rtb.ETS(rtb.ET.tx(qlim=[-1, 1]))

        q = rx.qlim
        nt.assert_equal(q, np.array([[-1], [1]]))

    def test_qlim4(self):
        rx = rtb.ETS(rtb.ET.tx())

        with self.assertRaises(ValueError):
            rx.qlim

    def test_random_q(self):
        rx = rtb.ETS(rtb.ET.Rx(qlim=[-1, 1]))

        q = rx.random_q()
        self.assertTrue(-1 <= q <= 1)

    def test_random_q2(self):
        rx = rtb.ETS([rtb.ET.Rx(qlim=[-1, 1]), rtb.ET.Rx(qlim=[1, 2])])

        q = rx.random_q(10)

        self.assertTrue(np.all(-1 <= q[:, 0]) and np.all(q[:, 0] <= 1))
        self.assertTrue(np.all(1 <= q[:, 1]) and np.all(q[:, 1] <= 2))

    def test_manip(self):
        r = rtb.models.Panda()
        ets = r.ets()
        q = r.qr

        m1 = ets.manipulability(q)
        m2 = ets.manipulability(q, axes="trans")
        m3 = ets.manipulability(q, axes="rot")

        nt.assert_almost_equal(m1, 0.0837, decimal=4)
        nt.assert_almost_equal(m2, 0.1438, decimal=4)
        nt.assert_almost_equal(m3, 2.7455, decimal=4)

    def test_yoshi(self):
        puma = rtb.models.Puma560()
        ets = puma.ets()
        q = puma.qn  # type: ignore

        m1 = ets.manipulability(q, axes=[True, True, True, True, True, True])
        m2 = ets.manipulability(np.c_[q, q].T)
        m3 = ets.manipulability(q, axes="trans")
        m4 = ets.manipulability(q, axes="rot")

        a0 = 0.0805
        a2 = 0.1354
        a3 = 2.44949

        nt.assert_almost_equal(m1, a0, decimal=4)
        nt.assert_almost_equal(m2[0], a0, decimal=4)  # type: ignore
        nt.assert_almost_equal(m2[1], a0, decimal=4)  # type: ignore
        nt.assert_almost_equal(m3, a2, decimal=4)
        nt.assert_almost_equal(m4, a3, decimal=4)

        with self.assertRaises(ValueError):
            puma.manipulability(axes="abcdef")  # type: ignore

    def test_cond(self):
        r = rtb.models.Panda()
        ets = r.ets()

        m = ets.manipulability(r.qr, method="invcondition")

        self.assertAlmostEqual(m, 0.11222, places=4)  # type: ignore

    def test_minsingular(self):
        r = rtb.models.Panda()
        ets = r.ets()

        m = ets.manipulability(r.qr, method="minsingular")

        self.assertAlmostEqual(m, 0.209013, places=4)  # type: ignore

    def test_manipulability_fail(self):
        puma = rtb.models.Puma560()
        ets = puma.ets()

        with self.assertRaises(ValueError):
            ets.manipulability(q=[1, 2, 3.0], method="notamethod")  # type: ignore

    def test_manip_fail2(self):
        r = rtb.models.Panda()
        ets = r.ets()
        q = r.qr

        with self.assertRaises(ValueError):
            ets.manipulability(q, axes="abcdef")  # type: ignore


if __name__ == "__main__":
    unittest.main()
