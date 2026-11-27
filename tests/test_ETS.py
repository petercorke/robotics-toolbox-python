#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import roboticstoolbox as rtb

# Rx/Ry/Rz/tx/ty/tz free functions for terser ET construction; ET also
# exports SE3 (a constant-transform ET constructor) via __all__, so this
# must precede the spatialmath SE3 import below to let that one win
from roboticstoolbox.ets.ET import *
import numpy as np

# import spatialmath.base as sm
from spatialmath import SE3
from spatialmath.base import tr2jac
import unittest
import sympy


class TestETS(unittest.TestCase):
    def test_bad_arg(self):
        rx = Rx(1.543)
        ry = Ry(1.543)

        with self.assertRaises(TypeError):
            rtb.ETS([rx, ry, 1.0])  # type: ignore

        with self.assertRaises(TypeError):
            rtb.ETS(1.0)  # type: ignore

    def test_args(self):
        deg = np.pi / 180
        # mm = 1e-3
        # tool_offset = (103) * mm

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        r1 = l0 + l1 + l2 + l3 + l3 + l4 + l5
        r2 = l0 * l1 * l2 * l3 * l3 * l4 * l5
        r3 = rtb.ETS(l0 + l1 + l2 + l3 + l3 + l4 + l5)
        r4 = rtb.ETS([l0, l1, l2, l3, l3, l4, l5])
        r5 = rtb.ETS([l0, l1, l2, l3, l3, l4, Rx(90 * deg), Rz(jindex=5)])
        # r6 = rtb.ETS([r1])
        r7 = rtb.ETS(Rx(1.0))

        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)
        self.assertEqual(r1, r4)
        self.assertEqual(r1, r5)
        self.assertEqual(r7[0], Rx(1.0))
        self.assertEqual(r1 + rtb.ETS(), r2)

    def test_empty(self):
        r = rtb.ETS()
        self.assertEqual(r.n, 0)
        self.assertEqual(r.m, 0)

    def test_str(self):
        rx = Rx(1.543)
        ry = Ry(1.543)
        rz = Rz(1.543)
        a = Rx()
        b = Ry()
        c = Rz()
        d = tx(1.0)
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
        rx = Rx(1.543)
        a = Rx(jindex=2)
        b = Ry(jindex=5)
        c = Rz(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "Rx(88.41°) ⊕ Rx(q2) ⊕ Ry(q5) ⊕ Rz(q7)")

    def test_str_flip(self):
        rx = Rx(1.543)
        a = Rx(jindex=2, flip=True)
        b = Ry(jindex=5)
        c = Rz(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "Rx(88.41°) ⊕ Rx(-q2) ⊕ Ry(q5) ⊕ Rz(q7)")

    def test_str_sym(self):
        x = sympy.Symbol("x")
        rx = Rx(x)  # type: ignore
        a = Rx(jindex=2)
        b = Ry(jindex=5)
        c = Rz(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "Rx(x) ⊕ Rx(q2) ⊕ Ry(q5) ⊕ Rz(q7)")

    def ets_mul(self):
        rx = Rx(1.543)
        ry = Ry(1.543)
        rz = Rz(1.543)
        a = Rx()
        b = Ry()

        ets1 = rx * ry
        ets2 = a * b

        self.assertIsInstance(ets1 * ets2, rtb.ETS)
        self.assertIsInstance(ets1 * rz, rtb.ETS)
        self.assertIsInstance(rz * ets1, rtb.ETS)
        self.assertIsInstance(ets1 + ets2, rtb.ETS)

        ets2 *= rz
        self.assertIsInstance(ets2, rtb.ETS)

    def test_n(self):
        rx = Rx(1.543)
        ry = Ry(1.543)
        # rz = Rz(1.543)
        a = Rx()
        b = Ry()

        ets1 = rx * ry
        ets2 = a * b
        ets3 = a * b * ry

        self.assertEqual(ets1.n, 0)
        self.assertEqual(ets2.n, 2)
        self.assertEqual(ets3.n, 2)
        self.assertEqual(ets1.structure, "")
        self.assertEqual(ets2.structure, "RR")

    def test_fixjoint_preserves_kinematics_and_limits(self):
        ets = (
            tx(0.2)
            * Rz(qlim=[-1, 1])
            * ty(0.3)
            * Rx(flip=True, qlim=[-2, 2])
            * tz(0.1)
            * tx(qlim=[-0.5, 0.5])
        )
        q = np.array([0.3, -0.25, 0.12])
        original = ets.eval(q)

        for j in range(ets.n):
            reduced = ets.fixjoint(j, q[j])
            self.assertEqual(reduced.n, ets.n - 1)
            self.assertEqual([e.jindex for e in reduced.joints()], [0, 1])
            nt.assert_allclose(reduced.eval(np.delete(q, j)), original)

            surviving = [e for k, e in enumerate(ets.joints()) if k != j]
            for actual, expected in zip(reduced.joints(), surviving):
                self.assertEqual(actual.kind, expected.kind)
                self.assertEqual(actual.isflip, expected.isflip)
                nt.assert_array_equal(actual.qlim, expected.qlim)

        nt.assert_allclose(ets.eval(q), original)
        self.assertEqual([e.jindex for e in ets.joints()], [0, 1, 2])

    def test_fixjoint_noncontiguous_jindices(self):
        ets = Rz(jindex=3, qlim=[-1, 1]) * tx(0.2) * Rx(
            jindex=7, flip=True, qlim=[-2, 2]
        )
        q = np.zeros(8)
        q[3] = 0.2
        q[7] = -0.4
        original = ets.eval(q)

        reduced = ets.fixjoint(0, q[3])
        self.assertEqual(reduced.n, 1)
        self.assertEqual([e.jindex for e in reduced.joints()], [7])
        self.assertTrue(reduced.joints()[0].isflip)
        nt.assert_allclose(reduced.eval([q[7]]), original)
        nt.assert_allclose(reduced.eval(q), original)

        reduced = ets.fixjoint(1, q[7])
        self.assertEqual([e.jindex for e in reduced.joints()], [3])
        nt.assert_allclose(reduced.eval([q[3]]), original)
        nt.assert_allclose(reduced.eval(q), original)
        self.assertEqual([e.jindex for e in ets.joints()], [3, 7])

    def test_fixjoint_rotation_in_degrees(self):
        ets = Rz() * tx(qlim=[-1, 1]) * Rx(flip=True)
        q = np.array([np.pi / 2, 0.25, -np.pi / 2])
        expected = ets.eval(q)

        for j in (0, 2):
            with self.subTest(j=j):
                reduced = ets.fixjoint(j, float(np.rad2deg(q[j])), unit="deg")
                self.assertIsInstance(reduced, rtb.ETS)
                nt.assert_allclose(reduced.eval(np.delete(q, j)), expected)
                nt.assert_allclose(
                    reduced.eval(np.delete(q, j)),
                    ets.fixjoint(j, q[j]).eval(np.delete(q, j)),
                )

        # A translation stays in metres and cannot accept angular degrees.
        nt.assert_allclose(
            ets.fixjoint(1, q[1]).eval(np.delete(q, 1)), expected
        )
        with self.assertRaisesRegex(ValueError, "only valid for rotational"):
            ets.fixjoint(1, 90, unit="deg")
        with self.assertRaisesRegex(ValueError, "unit must"):
            ets.fixjoint(0, 90, unit="grad")
        nt.assert_allclose(ets.eval(q), expected)

    def test_fixjoint_boolean_and_integer_indices(self):
        ets = Rz() * tx()
        q = np.array([0.2, 0.3])
        expected = ets.eval(q)

        # Both 0 and 1 are valid positions, so rejection must be based on type.
        for index in (False, True, np.bool_(False), np.bool_(True)):
            with self.subTest(index=index), self.assertRaises(ValueError):
                ets.fixjoint(index, 0.2)

        for index in (0, 1, np.int64(0), np.int64(1)):
            with self.subTest(index=index):
                fixed = ets.fixjoint(index, q[index])
                nt.assert_allclose(fixed.eval(np.delete(q, index)), expected)
        self.assertEqual(ets.n, 2)
        nt.assert_allclose(ets.eval(q), expected)

    def test_fixjoint_single_joint_and_bad_indices(self):
        ets = rtb.ETS(Rz())
        fixed = ets.fixjoint(0, 0.3)
        self.assertEqual(fixed.n, 0)
        nt.assert_allclose(fixed[0].A(), Rz(0.3).A())
        self.assertEqual(ets.n, 1)

        for index in (-1, 1, 0.5, "0"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                ets.fixjoint(index, 0.3)
        with self.assertRaises(ValueError):
            rtb.ETS().fixjoint(0, 0.3)

    def test_fkine(self):
        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        rx = Rx(1.543)
        ry = Ry(1.543)
        rz = Rz(1.543)
        t_x = tx(1.543)
        t_y = ty(1.543)
        t_z = tz(1.543)
        a = Rx(jindex=0)
        b = Ry(jindex=1)
        c = Rz(jindex=2)
        d = tx(jindex=3)
        e = ty(jindex=4)
        f = tz(jindex=5)

        r = rx * ry * rz * t_x * t_y * t_z * a * b * c * d * e * f

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
        a = Rx(x)  # type: ignore
        b = Ry(jindex=0)
        c = tz(jindex=1)

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

        r2 = rtb.ETS([Rx(jindex=0)])

        nt.assert_almost_equal(r.fkine(q, base=base).A, sympy.simplify(ans5.A))
        # nt.assert_almost_equal(r.fkine(q, base=base), ans5.A)  # type: ignore

        q2 = [y]
        ans6 = SE3.Rx(y) * tool
        nt.assert_almost_equal(
            r2.fkine(q2, tool=tool).A,
            sympy.simplify(ans6.A),  # type: ignore
        )
        # nt.assert_almost_equal(r2.fkine(q2, tool=tool), ans6)  # type: ignore

    def test_fkine_traj(self):
        robot = rtb.ERobot(
            [
                rtb.Link(Rx()),
                rtb.Link(Ry()),
                rtb.Link(Rz()),
                rtb.Link(tx()),
                rtb.Link(ty()),
                rtb.Link(tz()),
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

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )

        ee = tz(tool_offset) * Rz(-np.pi / 4)

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

        l0 = tz(0.333) * Rz(jindex=0)
        l1 = Rx(-90 * deg) * Rz(jindex=1)
        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)
        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)
        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )
        l5 = Rx(90 * deg) * Rz(jindex=5)
        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )
        ee = tz(tool_offset) * Rz(-np.pi / 4)

        r = l0 + l1 + l2 + l3 + l4 + l5 + l6 + ee

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        # q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        # q3 = np.expand_dims(q1, 0)
        # q4 = np.expand_dims(q1, 1)

        ans = tr2jac(r.eval(q1).T) @ r.jacob0(q1)

        nt.assert_array_almost_equal(r.jacobe(q1), ans)

    def test_pop(self):
        q = [1.0, 2.0, 3.0]
        a = Rx(jindex=0)
        b = Ry(jindex=1)
        c = Rz(jindex=2)
        d = tz(1.543)

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
        a = Rx(jindex=0)
        b = Ry(jindex=1)
        c = Rz(jindex=2)
        d = tz(1.543)

        ans1 = SE3.Rx(q[0]) * SE3.Ry(q[1]) * SE3.Rz(q[2]) * SE3.Tz(1.543)

        r = a + b + c + d
        r_inv = r.inv()

        nt.assert_almost_equal(r_inv.fkine(q).A, ans1.inv().A)

    def test_jointset(self):
        # q = [1.0, 2.0, 3.0]
        a = Rx(jindex=0)
        b = Ry(jindex=1)
        c = Rz(jindex=2)
        d = tz(1.543)

        ans = set((0, 1, 2))

        r = a + b + c + d
        nt.assert_equal(r.jindex_set(), ans)

    def test_split(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )

        ee = tz(tool_offset) * Rz(-np.pi / 4)

        segs = [l0, l1, l2, l3, l4, l5, l6]
        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee
        r2 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r3 = l0 * l1 * l2 * l3 * l4 * l5 * l6 * Rx(0.5)

        # split() returns [base, *segments, gripper]; base is always empty
        # for the default "last" method (leading content is folded into the
        # first segment)
        base, *split, gripper = r.split()
        self.assertEqual(len(base), 0)
        for i, link in enumerate(segs):
            self.assertEqual(link, split[i])
        self.assertEqual(gripper, ee)

        base2, *split2, gripper2 = r2.split()
        self.assertEqual(len(base2), 0)
        for i, link in enumerate(segs):
            self.assertEqual(link, split2[i])
        self.assertEqual(len(gripper2), 0)

        base3, *split3, gripper3 = r3.split()
        self.assertEqual(len(base3), 0)
        for i, link in enumerate(segs):
            self.assertEqual(link, split3[i])
        self.assertEqual(gripper3, rtb.ETS([Rx(0.5)]))

    def test_compile(self):
        q = [0, 1.0, 2, 3, 4, 5, 6]
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )

        ee = tz(tool_offset) * Rz(-np.pi / 4)

        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee
        r2 = r.compile()

        nt.assert_almost_equal(r.eval(q), r2.eval(q))
        self.assertTrue(len(r) > len(r2))

    def test_insert(self):
        # 7 joints (l0..l6, jindex 0-6) need 7 values -- q previously had
        # only 6, an off-by-one that silently "worked" only because the C
        # extension has no bounds checking (q[6] read past the end of the
        # array, undefined behaviour that happened not to crash). Confirmed
        # by comparing r.eval() with 6 vs 7 elements: they give genuinely
        # different results, proving index 6 really is read and used, not
        # just harmlessly ignored. Found via the pure-Python fallback
        # (_python_fkine), which correctly raises IndexError instead of
        # silently reading garbage.
        q = [1.0, 2, 3, 4, 5, 6, 7]
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )

        ee = tz(tool_offset) * Rz(-np.pi / 4)

        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee

        r2 = l0 * l1 * l2 * l3 * l4 * l6 * ee
        r2.insert(14, l5)

        r3 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r3.append(ee)

        r4 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r4.append(tz(tool_offset))
        r4.append(Rz(-np.pi / 4))

        r5 = l0 * l1 * l2 * l3 * l4 * l6 * ee
        r5.insert(14, Rx(90 * deg))
        r5.insert(15, Rz(jindex=5))

        nt.assert_almost_equal(r.eval(q), r2.eval(q))
        nt.assert_almost_equal(r.eval(q), r3.eval(q))
        nt.assert_almost_equal(r.eval(q), r4.eval(q))
        nt.assert_almost_equal(r.fkine(q).A, r5.fkine(q).A)

    def test_swap(self):
        q = [1.0]
        e = Rz(jindex=0) * Rz(1) * tx(2)

        ans = SE3.Rz(q[0]) * SE3.Rz(1) * SE3.Tx(2)

        self.assertTrue(e[0].isjoint)
        self.assertFalse(e[1].isjoint)
        nt.assert_almost_equal(e.fkine(q).A, ans.A)

        # swap two adjacent transforms that share the same axis - the
        # kinematics must be unaffected since same-axis rotations commute
        e.swap(0)
        self.assertFalse(e[0].isjoint)
        self.assertTrue(e[1].isjoint)
        self.assertAlmostEqual(e[0].param, 1.0)  # type: ignore
        nt.assert_almost_equal(e.fkine(q).A, ans.A)

        # swap back and check we're symmetric
        e.swap(0)
        self.assertTrue(e[0].isjoint)
        self.assertFalse(e[1].isjoint)
        nt.assert_almost_equal(e.fkine(q).A, ans.A)

    def test_swap_not_commutative(self):
        e = tx(1) * Rz(jindex=0) * Rz(1)

        with self.assertRaises(ValueError):
            e.swap(0)

    def test_swap_bad_index(self):
        e = tx(1) * tx(2)

        with self.assertRaises(IndexError):
            e.swap(-1)

        with self.assertRaises(IndexError):
            e.swap(1)

    def test_merge(self):
        q = [1.0]
        e = Rz(jindex=0) * tx(1) * tx(2) * Rz(1)

        ans = SE3.Rz(q[0]) * SE3.Tx(3) * SE3.Rz(1)

        n = len(e)
        e.merge(1)

        self.assertEqual(len(e), n - 1)
        self.assertEqual(e[1].kind, "tx")
        self.assertAlmostEqual(e[1].param, 3.0)  # type: ignore
        nt.assert_almost_equal(e.fkine(q).A, ans.A)

    def test_merge_not_same_type(self):
        e = tx(1) * Rz(1)

        with self.assertRaises(ValueError):
            e.merge(0)

    def test_merge_both_joints(self):
        e = Rz(jindex=0) * Rz(jindex=1)

        with self.assertRaises(ValueError):
            e.merge(0)

    def test_merge_bad_index(self):
        e = tx(1) * tx(2)

        with self.assertRaises(IndexError):
            e.merge(-1)

        with self.assertRaises(IndexError):
            e.merge(1)

    def test_joint_descriptor_ets_str(self):
        # ETS.__str__ has its own joint-formatting logic, independent of
        # each ET's own __str__ - a custom name from a string `param`
        # descriptor (e.g. "theta2") must show up there too, since printing
        # a full kinematic chain (not a lone ET) is the primary use case.
        ets = Rx("theta0") * tx("q1") * Rz("-q(2)") * Ry()

        self.assertEqual(str(ets), "Rx(theta0) ⊕ tx(q1) ⊕ Rz(-q(2)) ⊕ Ry(q3)")
        self.assertEqual([e.jindex for e in ets], [0, 1, 2, 3])
        self.assertEqual([e.isflip for e in ets], [False, False, True, False])

    def test_jacob0(self):
        q = [0.0]
        rx = rtb.ETS(Rx())
        ry = rtb.ETS(Ry())
        rz = rtb.ETS(Rz())
        t_x = rtb.ETS(tx())
        t_y = rtb.ETS(ty())
        t_z = rtb.ETS(tz())

        r = t_x + t_y + t_z + rx + ry + rz

        nt.assert_almost_equal(t_x.jacob0(q), np.array([[1, 0, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(t_y.jacob0(q), np.array([[0, 1, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(t_z.jacob0(q), np.array([[0, 0, 1, 0, 0, 0]]).T)
        nt.assert_almost_equal(rx.jacob0(q), np.array([[0, 0, 0, 1, 0, 0]]).T)
        nt.assert_almost_equal(ry.jacob0(q), np.array([[0, 0, 0, 0, 1, 0]]).T)
        nt.assert_almost_equal(rz.jacob0(q), np.array([[0, 0, 0, 0, 0, 1]]).T)
        nt.assert_almost_equal(r.jacob0(q), np.eye(6))

    def test_jacobe(self):
        q = [0.0]
        rx = rtb.ETS(Rx())
        ry = rtb.ETS(Ry())
        rz = rtb.ETS(Rz())
        t_x = rtb.ETS(tx())
        t_y = rtb.ETS(ty())
        t_z = rtb.ETS(tz())

        r = t_x + t_y + t_z + rx + ry + rz

        nt.assert_almost_equal(t_x.jacobe(q), np.array([[1, 0, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(t_y.jacobe(q), np.array([[0, 1, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(t_z.jacobe(q), np.array([[0, 0, 1, 0, 0, 0]]).T)
        nt.assert_almost_equal(rx.jacobe(q), np.array([[0, 0, 0, 1, 0, 0]]).T)
        nt.assert_almost_equal(ry.jacobe(q), np.array([[0, 0, 0, 0, 1, 0]]).T)
        nt.assert_almost_equal(rz.jacobe(q), np.array([[0, 0, 0, 0, 0, 1]]).T)
        nt.assert_almost_equal(r.jacobe(q), np.eye(6))

    def test_jacob0_sym(self):
        x = sympy.Symbol("x")
        q1 = np.array([x, x])
        q2 = np.array([0, x])
        rx = rtb.ETS(Rx(jindex=0))
        ry = rtb.ETS(Ry(jindex=0))
        rz = rtb.ETS(Rz(jindex=0))
        t_x = rtb.ETS(tx(jindex=0))
        t_y = rtb.ETS(ty(jindex=0))
        t_z = rtb.ETS(tz(jindex=1))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))

        r = t_x + t_y + t_z + rx + ry + rz + a

        print(r.jacob0(q2))

        nt.assert_almost_equal(t_x.jacob0(q1), np.array([[1, 0, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(t_y.jacob0(q1), np.array([[0, 1, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(t_z.jacob0(q1), np.array([[0, 0, 1, 0, 0, 0]]).T)
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
        rx = rtb.ETS(Rx(jindex=0))
        ry = rtb.ETS(Ry(jindex=0))
        rz = rtb.ETS(Rz(jindex=0))
        t_x = rtb.ETS(tx(jindex=0))
        t_y = rtb.ETS(ty(jindex=0))
        t_z = rtb.ETS(tz(jindex=1))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))

        r = t_x + t_y + t_z + rx + ry + rz + a

        print(r.jacobe(q2))

        nt.assert_almost_equal(t_x.jacobe(q1), np.array([[1, 0, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(t_y.jacobe(q1), np.array([[0, 1, 0, 0, 0, 0]]).T)
        nt.assert_almost_equal(t_z.jacobe(q1), np.array([[0, 0, 1, 0, 0, 0]]).T)
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

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )

        ee = tz(tool_offset) * Rz(-np.pi / 4)

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

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )

        ee = tz(tool_offset) * Rz(-np.pi / 4)
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

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )

        ee = tz(tool_offset) * Rz(-np.pi / 4)

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

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )

        ee = tz(tool_offset) * Rz(-np.pi / 4)
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
        rx = rtb.ETS(Rx(jindex=0))
        ry = rtb.ETS(Ry(jindex=0))
        rz = rtb.ETS(Rz(jindex=0))
        t_x = rtb.ETS(tx(jindex=0))
        t_y = rtb.ETS(ty(jindex=0))
        t_z = rtb.ETS(tz(jindex=1))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))

        r = t_x + t_y + t_z + rx + ry + rz + a

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
        rx = rtb.ETS(Rx(jindex=0))
        ry = rtb.ETS(Ry(jindex=1))
        rz = rtb.ETS(Rz(jindex=2))
        t_x = rtb.ETS(tx(jindex=3, qlim=[-1, 1]))
        t_y = rtb.ETS(ty(jindex=4, qlim=[-1, 1]))
        t_z = rtb.ETS(tz(jindex=5, qlim=[-1, 1]))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))
        r = t_x + t_y + t_z + rx + ry + rz + a
        r.plot(q=q2, block=False, backend="pyplot")

    def test_teach(self):
        # x = sympy.Symbol("x")
        q2 = np.array([0, 1, 2, 3, 4, 5])
        rx = rtb.ETS(Rx(jindex=0))
        ry = rtb.ETS(Ry(jindex=1))
        rz = rtb.ETS(Rz(jindex=2))
        t_x = rtb.ETS(tx(jindex=3, qlim=[-1, 1]))
        t_y = rtb.ETS(ty(jindex=4, qlim=[-1, 1]))
        t_z = rtb.ETS(tz(jindex=5, qlim=[-1, 1]))
        a = rtb.ETS(rtb.ET.SE3(np.eye(4)))
        r = t_x + t_y + t_z + rx + ry + rz + a
        r.teach(q=q2, block=False, backend="pyplot")

    def test_partial_fkine(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = tz(0.333) * Rz(jindex=0)

        l1 = Rx(-90 * deg) * Rz(jindex=1)

        l2 = Rx(90 * deg) * tz(0.316) * Rz(jindex=2)

        l3 = tx(0.0825) * Rx(90 * deg) * Rz(jindex=3)

        l4 = (
            tx(-0.0825)
            * Rx(-90 * deg)
            * tz(0.384)
            * Rz(jindex=4)
        )

        l5 = Rx(90 * deg) * Rz(jindex=5)

        l6 = (
            tx(0.088)
            * Rx(90 * deg)
            * tz(0.107)
            * Rz(jindex=6)
        )

        ee = tz(tool_offset) * Rz(-np.pi / 4)

        r = l0 + l1 + l2 + l3 + l4 + l5 + l6 + ee
        r2 = rtb.Robot(r)

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])

        # Differentiate the independently implemented Hessian. The previous
        # literal tensor reproduced the old recursion's incorrect downstream
        # angular dependencies and asymmetric position derivatives.
        step = 1e-5
        ans = np.stack(
            [
                (r.hessian0(q1 + delta) - r.hessian0(q1 - delta)) / (2 * step)
                for delta in step * np.eye(r.n)
            ]
        )

        fk = r.partial_fkine0(q1, 3)
        fk2 = r2.partial_fkine0(q1, 3)

        nt.assert_almost_equal(fk, ans)
        nt.assert_almost_equal(fk2, ans)

    def test_qlim1(self):
        rx = rtb.ETS(Rx())

        q = rx.qlim
        nt.assert_equal(q, np.array([[-np.pi], [np.pi]]))

    def test_qlim2(self):
        rx = rtb.ETS(Rx(qlim=[-1, 1]))

        q = rx.qlim
        nt.assert_equal(q, np.array([[-1], [1]]))

    def test_qlim3(self):
        rx = rtb.ETS(tx(qlim=[-1, 1]))

        q = rx.qlim
        nt.assert_equal(q, np.array([[-1], [1]]))

    def test_qlim4(self):
        rx = rtb.ETS(tx())

        with self.assertRaises(ValueError):
            rx.qlim

    def test_random_q(self):
        rx = rtb.ETS(Rx(qlim=[-1, 1]))

        q = rx.random_q()
        self.assertTrue(-1 <= q <= 1)

    def test_random_q2(self):
        rx = rtb.ETS([Rx(qlim=[-1, 1]), Rx(qlim=[1, 2])])

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

    def test_resolve_q_compact_and_global(self):
        # a sub-chain whose own jindex range doesn't start at 0 -- e.g. a
        # second arm on a branched robot, jindex 3-5 while some other
        # branch owns 0-2
        e = Rz(jindex=3) * tx(1) * Rz(jindex=4) * tx(1) * Rz(jindex=5)
        ets = rtb.ETS(e)

        self.assertEqual(ets.n, 3)
        nt.assert_array_equal(ets.jindices, [3, 4, 5])

        q_compact = [0.1, 0.2, 0.3]
        q_global = np.zeros(6)
        q_global[3:6] = q_compact

        nt.assert_almost_equal(ets.eval(q_compact), ets.eval(q_global))
        nt.assert_almost_equal(ets.jacob0(q_compact), ets.jacob0(q_global))
        nt.assert_almost_equal(ets.jacobe(q_compact), ets.jacobe(q_global))
        nt.assert_almost_equal(ets.hessian0(q_compact), ets.hessian0(q_global))
        nt.assert_almost_equal(ets.hessiane(q_compact), ets.hessiane(q_global))

    def test_resolve_q_too_short_raises(self):
        e = Rz(jindex=3) * tx(1) * Rz(jindex=4) * tx(1) * Rz(jindex=5)
        ets = rtb.ETS(e)

        # neither compact (3) nor global (>= 6)
        with self.assertRaises(ValueError):
            ets.eval([0.1, 0.2, 0.3, 0.4])

    def test_resolve_q_trajectory_still_works(self):
        panda = rtb.models.Panda().ets()
        qt = np.tile(panda.random_q(), (5, 1))

        T = panda.eval(qt)
        self.assertEqual(T.shape, (5, 4, 4))

    def test_swap_commuting_transforms(self):
        # translations along different axes, and a rotation with a translation
        # along its axis, commute: the kinematics are unchanged by the swap
        q = [0.3, -0.4]
        for e, i in (
            (ET.Rz() * ET.tx(1) * ET.ty(2) * ET.Rz(), 1),
            (ET.Rz() * ET.tz(1) * ET.Rz(0.2) * ET.Rx(), 0),
            (ET.tx(1) * ET.tx(2) * ET.Rz() * ET.Ry(), 0),
        ):
            before = e.fkine(q).A
            e.swap(i)
            nt.assert_array_almost_equal(e.fkine(q).A, before)

    def test_swap_not_commuting(self):
        e = ET.Rz() * ET.tx(1)
        with self.assertRaisesRegex(ValueError, "do not commute"):
            e.swap(0)

    def test_merge_joint_with_a_constant(self):
        # the constant becomes the offset of the joint
        e = ET.tx(1) * ET.tx()
        e.merge(0)
        self.assertEqual(len(e), 1)
        self.assertTrue(e[0].isjoint)
        nt.assert_almost_equal(e[0].offset, 1)
        # two joints are two variables
        with self.assertRaisesRegex(ValueError, "joint"):
            (ET.tx() * ET.tx()).merge(0)
        e = ET.tx(1) * ET.tx(2)
        e.merge(0)
        self.assertEqual(len(e), 1)
        nt.assert_almost_equal(e[0].param, 3)

    def test_fkine_no_argument(self):
        # a sequence with no joints needs no q
        e = ET.Rx(0.3) * ET.tz(0.75)
        nt.assert_array_almost_equal(e.fkine().A, e.fkine([]).A)
        nt.assert_array_almost_equal(e.eval(), e.eval([]))

    def test_fkine_missing_or_short_q(self):
        # a sequence with joints gives a ValueError that names the lengths,
        # not a low-level index error
        e = ET.Rz() * ET.tx(1) * ET.Ry()
        for call in (e.fkine, e.eval):
            with self.assertRaisesRegex(ValueError, "q has length 0, expected 2"):
                call()
            with self.assertRaisesRegex(ValueError, "q has length 1, expected 2"):
                call([0.1])


if __name__ == "__main__":
    unittest.main()
