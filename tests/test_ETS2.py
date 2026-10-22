#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import roboticstoolbox as rtb

# R/tx/ty free functions for terser ET2 construction; ET2 also exports SE2
# (a constant-transform ET2 constructor) via __all__, so this must precede
# the spatialmath SE2 import below to let that one win
from roboticstoolbox.ets.ET2 import *
import numpy as np

# import spatialmath.base as sm
from spatialmath import SE2
import unittest
import sympy


class TestETS2(unittest.TestCase):
    def test_bad_arg(self):
        rx = R(1.543)
        ry = R(1.543)

        with self.assertRaises(TypeError):
            rtb.ETS2([rx, ry, 1.0])  # type: ignore

        with self.assertRaises(TypeError):
            rtb.ETS2(1.0)  # type: ignore

    def test_args(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = tx(0.333) * R(jindex=0)

        l1 = R(-90 * deg) * R(jindex=1)

        l2 = R(90 * deg) * tx(0.316) * R(jindex=2)

        l3 = tx(0.0825) * R(90 * deg) * R(jindex=3)

        l4 = (
            tx(-0.0825)
            * R(-90 * deg)
            * tx(0.384)
            * R(jindex=4)
        )

        l5 = R(90 * deg) * R(jindex=5)

        r1 = l0 + l1 + l2 + l3 + l3 + l4 + l5
        r2 = l0 * l1 * l2 * l3 * l3 * l4 * l5
        r3 = rtb.ETS2(l0 + l1 + l2 + l3 + l3 + l4 + l5)
        r4 = rtb.ETS2([l0, l1, l2, l3, l3, l4, l5])
        r5 = rtb.ETS2(
            [l0, l1, l2, l3, l3, l4, R(90 * deg), R(jindex=5)]
        )
        r6 = rtb.ETS2([r1])
        r7 = rtb.ETS2(R(1.0))

        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)
        self.assertEqual(r1, r4)
        self.assertEqual(r1, r5)
        self.assertEqual(r7[0], R(1.0))
        self.assertEqual(r1 + rtb.ETS2(), r2)

    def test_empty(self):
        r = rtb.ETS2()
        self.assertEqual(r.n, 0)
        self.assertEqual(r.m, 0)

    def test_str(self):
        rx = R(1.543)
        ry = R(1.543)
        rz = R(1.543)
        a = R()
        b = R()
        c = R()
        d = tx(1.0)
        e = rtb.ET2.SE2(SE2(1.0, unit="rad"))
        ets = rx * ry * rz * d
        ets2 = rx * a * b * c
        ets3 = d * e

        self.assertEqual(str(ets), "R(88.41°) ⊕ R(88.41°) ⊕ R(88.41°) ⊕ tx(1)")
        self.assertEqual(str(ets2), "R(88.41°) ⊕ R(q0) ⊕ R(q1) ⊕ R(q2)")
        self.assertEqual(str(ets3), "tx(1) ⊕ SE2(0, 0; 57.3°)")
        self.assertEqual(ets2.__str__(q="θ{0}"), "R(88.41°) ⊕ R(θ0) ⊕ R(θ1) ⊕ R(θ2)")

    def test_str_jindex(self):
        rx = R(1.543)
        a = R(jindex=2)
        b = R(jindex=5)
        c = R(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "R(88.41°) ⊕ R(q2) ⊕ R(q5) ⊕ R(q7)")

    def test_str_flip(self):
        rx = R(1.543)
        a = R(jindex=2, flip=True)
        b = R(jindex=5)
        c = R(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "R(88.41°) ⊕ R(-q2) ⊕ R(q5) ⊕ R(q7)")

    def test_str_sym(self):
        x = sympy.Symbol("x")
        rx = R(x)
        a = R(jindex=2)
        b = R(jindex=5)
        c = R(jindex=7)
        ets = rx * a * b * c

        self.assertEqual(str(ets), "R(x) ⊕ R(q2) ⊕ R(q5) ⊕ R(q7)")

    def ets_mul(self):
        rx = R(1.543)
        ry = R(1.543)
        rz = R(1.543)
        a = R()
        b = R()

        ets1 = rx * ry
        ets2 = a * b

        self.assertIsInstance(ets1 * ets2, rtb.ETS)
        self.assertIsInstance(ets1 * rz, rtb.ETS)
        self.assertIsInstance(rz * ets1, rtb.ETS)
        self.assertIsInstance(ets1 + ets2, rtb.ETS)

        ets2 *= rz
        self.assertIsInstance(ets2, rtb.ETS)

    def test_n(self):
        rx = R(1.543)
        ry = R(1.543)
        rz = R(1.543)
        a = R()
        b = R()

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
            * R(qlim=[-1, 1])
            * ty(0.3)
            * tx(flip=True, qlim=[-2, 2])
            * R(flip=True, qlim=[-3, 3])
        )
        q = np.array([0.3, -0.25, 0.12])
        original = ets.eval(q)

        for j in range(ets.n):
            with self.subTest(j=j):
                reduced = ets.fixjoint(j, q[j])
                self.assertIsInstance(reduced, rtb.ETS2)
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
        ets = R(jindex=3, qlim=[-1, 1]) * tx(0.2) * ty(
            jindex=7, flip=True, qlim=[-2, 2]
        )
        q = np.zeros(8)
        q[3] = 0.2
        q[7] = -0.4
        original = ets.eval(q)

        reduced = ets.fixjoint(0, q[3])
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
        ets = R() * tx(qlim=[-1, 1]) * R(flip=True)
        q = np.array([np.pi / 2, 0.25, -np.pi / 2])
        expected = ets.eval(q)

        for j in (0, 2):
            with self.subTest(j=j):
                reduced = ets.fixjoint(j, float(np.rad2deg(q[j])), unit="deg")
                self.assertIsInstance(reduced, rtb.ETS2)
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
        ets = R() * tx()
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
        ets = rtb.ETS2(R())
        fixed = ets.fixjoint(0, 0.3)
        self.assertEqual(fixed.n, 0)
        nt.assert_allclose(fixed[0].A(), R(0.3).A())
        self.assertEqual(ets.n, 1)

        for index in (-1, 1, 0.5, "0"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                ets.fixjoint(index, 0.3)
        with self.assertRaises(ValueError):
            rtb.ETS2().fixjoint(0, 0.3)

    def test_fkine(self):
        q = np.array([0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        rx = R(1.543)
        ry = R(1.543)
        rz = R(1.543)
        t_x = tx(1.543)
        t_y = ty(1.543)
        t_x = tx(1.543)
        a = R(jindex=0)
        b = R(jindex=1)
        c = R(jindex=2)
        d = tx(jindex=3)
        e = ty(jindex=4)
        f = tx(jindex=5)

        r = rx * ry * rz * t_x * t_y * t_x * a * b * c * d * e * f

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
        a = R(x)
        b = R(jindex=0)
        c = tx(jindex=1)

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

        r2 = rtb.ETS2([R(jindex=0)])

        nt.assert_almost_equal(r.fkine(q, base=base), ans5.A)
        # nt.assert_almost_equal(r.fkine(q, base=base.A), ans5.A)  # type: ignore

        q2 = [y]
        ans6 = SE2(y) * tool
        nt.assert_almost_equal(r2.fkine(q2, tool=tool), ans6.A)  # type: ignore
        # nt.assert_almost_equal(r2.fkine(q2, tool=tool.A), ans6)  # type: ignore

    def test_pop(self):
        q = [1.0, 2.0, 3.0]
        a = R(jindex=0)
        b = R(jindex=1)
        c = R(jindex=2)
        d = tx(1.543)

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
        a = R(jindex=0)
        b = R(jindex=1)
        c = R(jindex=2)
        d = tx(1.543)

        ans1 = SE2(q[0]) * SE2(q[1]) * SE2(q[2]) * SE2.Tx(1.543)

        r = a + b + c + d
        r_inv = r.inv()

        nt.assert_almost_equal(r_inv.fkine(q), ans1.inv().A)

    def test_jointset(self):
        q = [1.0, 2.0, 3.0]
        a = R(jindex=0)
        b = R(jindex=1)
        c = R(jindex=2)
        d = tx(1.543)

        ans = set((0, 1, 2))

        r = a + b + c + d
        nt.assert_equal(r.jindex_set(), ans)

    def test_split(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = tx(0.333) * R(jindex=0)

        l1 = R(-90 * deg) * R(jindex=1)

        l2 = R(90 * deg) * tx(0.316) * R(jindex=2)

        l3 = tx(0.0825) * R(90 * deg) * R(jindex=3)

        l4 = (
            tx(-0.0825)
            * R(-90 * deg)
            * tx(0.384)
            * R(jindex=4)
        )

        l5 = R(90 * deg) * R(jindex=5)

        l6 = (
            tx(0.088)
            * R(90 * deg)
            * tx(0.107)
            * R(jindex=6)
        )

        ee = tx(tool_offset) * R(-np.pi / 4)

        segs = [l0, l1, l2, l3, l4, l5, l6]
        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee
        r2 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r3 = l0 * l1 * l2 * l3 * l4 * l5 * l6 * R(0.5)

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
        self.assertEqual(gripper3, rtb.ETS2([R(0.5)]))

    def test_compile(self):
        q = [0, 1.0, 2, 3, 4, 5, 6]
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = tx(0.333) * R(jindex=0)

        l1 = R(-90 * deg) * R(jindex=1)

        l2 = R(90 * deg) * tx(0.316) * R(jindex=2)

        l3 = tx(0.0825) * R(90 * deg) * R(jindex=3)

        l4 = (
            tx(-0.0825)
            * R(-90 * deg)
            * tx(0.384)
            * R(jindex=4)
        )

        l5 = R(90 * deg) * R(jindex=5)

        l6 = (
            tx(0.088)
            * R(90 * deg)
            * tx(0.107)
            * R(jindex=6)
        )

        ee = tx(tool_offset) * R(-np.pi / 4)

        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee
        r2 = r.compile()

        nt.assert_almost_equal(r.fkine(q).A, r2.fkine(q).A)
        self.assertTrue(len(r) > len(r2))

    def test_insert(self):
        q = [0, 1.0, 2, 3, 4, 5, 6]
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = tx(0.333) * R(jindex=0)

        l1 = R(-90 * deg) * R(jindex=1)

        l2 = R(90 * deg) * tx(0.316) * R(jindex=2)

        l3 = tx(0.0825) * R(90 * deg) * R(jindex=3)

        l4 = (
            tx(-0.0825)
            * R(-90 * deg)
            * tx(0.384)
            * R(jindex=4)
        )

        l5 = R(90 * deg) * R(jindex=5)

        l6 = (
            tx(0.088)
            * R(90 * deg)
            * tx(0.107)
            * R(jindex=6)
        )

        ee = tx(tool_offset) * R(-np.pi / 4)

        r = l0 * l1 * l2 * l3 * l4 * l5 * l6 * ee

        r2 = l0 * l1 * l2 * l3 * l4 * l6 * ee
        r2.insert(14, l5)

        r3 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r3.append(ee)

        r4 = l0 * l1 * l2 * l3 * l4 * l5 * l6
        r4.append(tx(tool_offset))
        r4.append(R(-np.pi / 4))

        r5 = l0 * l1 * l2 * l3 * l4 * l6 * ee
        r5.insert(14, R(90 * deg))
        r5.insert(15, R(jindex=5))

        nt.assert_almost_equal(r.fkine(q).A, r2.fkine(q).A)
        nt.assert_almost_equal(r.fkine(q).A, r3.fkine(q).A)
        nt.assert_almost_equal(r.fkine(q).A, r4.fkine(q).A)
        nt.assert_almost_equal(r.fkine(q).A, r5.fkine(q).A)

    def test_jacob0(self):
        q = [0, 0, 0]
        r = rtb.ETS2(R(jindex=0))
        t_x = rtb.ETS2(tx(jindex=1))
        t_y = rtb.ETS2(ty(jindex=2))

        r2 = t_x + t_y + r

        nt.assert_almost_equal(t_x.jacob0(q), np.array([[1, 0, 0]]).T)
        nt.assert_almost_equal(t_y.jacob0(q), np.array([[0, 1, 0]]).T)
        nt.assert_almost_equal(r.jacob0(q), np.array([[0, 0, 1]]).T)
        nt.assert_almost_equal(r2.jacob0(q), np.eye(3))

    def test_jacobe(self):
        q = [0, 0, 0]
        r = rtb.ETS2(R(jindex=0))
        t_x = rtb.ETS2(tx(jindex=1))
        t_y = rtb.ETS2(ty(jindex=2))

        r2 = t_x + t_y + r

        nt.assert_almost_equal(t_x.jacobe(q), np.array([[1, 0, 0]]).T)
        nt.assert_almost_equal(t_y.jacobe(q), np.array([[0, 1, 0]]).T)
        nt.assert_almost_equal(r.jacobe(q), np.array([[0, 0, 1]]).T)
        nt.assert_almost_equal(r2.jacobe(q), np.eye(3))

    def test_plot(self):
        q2 = np.array([0, 1, 2])
        rz = rtb.ETS2(R(jindex=0))
        t_x = rtb.ETS2(tx(jindex=1, qlim=[-1, 1]))
        t_y = rtb.ETS2(ty(jindex=2, qlim=[-1, 1]))
        a = rtb.ETS2(rtb.ET2.SE2(np.eye(3)))
        r = t_x + t_y + rz + a
        r.plot(q=q2, block=False)

    def test_teach(self):
        x = sympy.Symbol("x")
        q2 = np.array([0, 1, 2])
        rz = rtb.ETS2(R(jindex=0))
        t_x = rtb.ETS2(tx(jindex=1, qlim=[-1, 1]))
        t_y = rtb.ETS2(ty(jindex=2, qlim=[-1, 1]))
        a = rtb.ETS2(rtb.ET2.SE2(np.eye(3)))
        r = t_x + t_y + rz + a
        r.teach(q=q2, block=False)

    def test_swap_and_merge(self):
        e = ET2.tx(1) * ET2.ty(2)
        before = e.fkine([]).A
        e.swap(0)
        nt.assert_array_almost_equal(e.fkine([]).A, before)

        e = ET2.R() * ET2.tx(1)
        with self.assertRaisesRegex(ValueError, "do not commute"):
            e.swap(0)

        # a constant merged into a joint is the offset of the joint
        e = ET2.tx(1) * ET2.tx()
        e.merge(0)
        self.assertEqual(len(e), 1)
        nt.assert_almost_equal(e[0].offset, 1)
        with self.assertRaisesRegex(ValueError, "joint"):
            (ET2.tx() * ET2.tx()).merge(0)

    def test_motion_subspace(self):
        nt.assert_array_equal(ET2.R().s, [0, 0, 1])
        nt.assert_array_equal(ET2.tx().s, [1, 0, 0])
        nt.assert_array_equal(ET2.ty().s, [0, 1, 0])

    def test_fkine_no_argument(self):
        # a sequence with no joints needs no q
        e = ET2.R(0.3) * ET2.tx(0.75)
        nt.assert_array_almost_equal(e.fkine().A, e.fkine([]).A)
        nt.assert_array_almost_equal(e.eval(), e.eval([]))

    def test_fkine_missing_or_short_q(self):
        # a sequence with joints gives a ValueError that names the lengths,
        # not a low-level index error
        e = ET2.R() * ET2.tx(1) * ET2.R()
        for call in (e.fkine, e.eval):
            with self.assertRaisesRegex(ValueError, "q has length 0, expected 2"):
                call()
            with self.assertRaisesRegex(ValueError, "q has length 1, expected 2"):
                call([0.1])


if __name__ == "__main__":
    unittest.main()
