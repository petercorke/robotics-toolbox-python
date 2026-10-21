import math
import unittest

import numpy as np
import numpy.testing as nt
import pytest
import sympy as sp

from roboticstoolbox import ET, ET2, ETS

a = sp.Symbol("a")
q = sp.Symbol("q")
pi = math.pi
half_pi = sp.pi / 2

AXES = "xyz"


def kinds(e) -> list[str]:
    return [t.kind for t in e]


def padded(*mid) -> ETS:
    """Terms before and after, the middle ones at index 2 and 3."""
    e = ET.tz(1) * ET.Ry(0.2)
    for m in mid:
        e = e * m
    return e * ET.tx(2) * ET.Rx()


def sym_equal(T1, T2) -> bool:
    return sp.simplify(sp.Matrix(T1) - sp.Matrix(T2)) == sp.zeros(4, 4)


class TestKinematicsUnchanged(unittest.TestCase):
    def test_every_pair_and_angle(self):
        # every rotation, quarter and half turns either way, with every
        # transform on every other axis, a translation or a rotation
        for k in (1, 2, 3, -1, -2, 5):
            for r in AXES:
                for b in AXES:
                    if r == b:
                        continue
                    for kind in ("R", "t"):
                        with self.subTest(k=k, R=r, B=kind + b):
                            e = ET.Rx(0)  # placeholder to build below
                            e = getattr(ET, "R" + r)(k * pi / 2) * getattr(
                                ET, kind + b
                            )(0.7)
                            before = e.fkine().A
                            e.swap(0)
                            nt.assert_array_almost_equal(e.fkine().A, before)
                            # the quarter turn is untouched and is now second
                            self.assertEqual(e[1].kind, "R" + r)
                            self.assertAlmostEqual(e[1].param, k * pi / 2)

    def test_the_other_order(self):
        # B R = R (R^-1 B R)
        for k in (1, 2, 3):
            for r in AXES:
                for b in AXES:
                    if r == b:
                        continue
                    with self.subTest(k=k, R=r, B=b):
                        e = getattr(ET, "t" + b)(0.7) * getattr(ET, "R" + r)(k * pi / 2)
                        before = e.fkine().A
                        e.swap(0)
                        nt.assert_array_almost_equal(e.fkine().A, before)
                        self.assertEqual(e[0].kind, "R" + r)

    def test_known_results(self):
        # the rules of the paper
        for rx, b, expect in (
            (1, "ty", "tz(1)"),
            (1, "tz", "ty(-1)"),
            (-1, "ty", "tz(-1)"),
            (-1, "tz", "ty(1)"),
            (2, "ty", "ty(-1)"),
            (2, "tz", "tz(-1)"),
            (2, "tx", "tx(1)"),
        ):
            with self.subTest(k=rx, B=b):
                e = ET.Rx(rx * pi / 2) * getattr(ET, b)(1)
                e.swap(0)
                self.assertEqual(str(e[0]), expect)
        e = ET.Rx(pi / 2) * ET.Ry(0.3)
        e.swap(0)
        self.assertEqual(str(e), "Rz(17.19°) ⊕ Rx(90°)")
        e = ET.Rx(-pi / 2) * ET.Ry(0.3)
        e.swap(0)
        self.assertEqual(str(e), "Rz(-17.19°) ⊕ Rx(-90°)")

    def test_a_whole_turn_is_not_a_quarter_turn_problem(self):
        # zero turns, nothing to carry across, but it still swaps
        e = ET.Rx(0) * ET.ty(1)
        e.swap(0)
        self.assertEqual(kinds(e), ["ty", "Rx"])
        # five quarters is one
        e = ET.Rx(5 * pi / 2) * ET.ty(1)
        e.swap(0)
        self.assertEqual(str(e[0]), "tz(1)")

    def test_padding_is_untouched(self):
        e = padded(ET.Rx(pi / 2), ET.ty(1))
        e.swap(2)
        self.assertEqual(kinds(e), ["tz", "Ry", "tz", "Rx", "tx", "Rx"])
        self.assertEqual(e[0].param, 1)
        self.assertAlmostEqual(e[1].param, 0.2)
        self.assertEqual(e[4].param, 2)
        self.assertTrue(e[5].isjoint)


class TestSymbolic(unittest.TestCase):
    def test_exact_quarter_turn_numeric_other(self):
        e = padded(ET.Rx(half_pi), ET.ty(1.5))
        ref = padded(ET.Rx(half_pi), ET.ty(1.5))
        e.swap(2)
        self.assertEqual(kinds(e)[2:4], ["tz", "Rx"])
        self.assertAlmostEqual(e[2].param, 1.5)
        nt.assert_array_almost_equal(
            np.array(e.fkine([0.3]).A, dtype=float),
            np.array(ref.fkine([0.3]).A, dtype=float),
        )

    def test_numeric_quarter_turn_symbolic_other(self):
        e = padded(ET.Rx(pi / 2), ET.ty(a))
        e.swap(2)
        self.assertEqual(kinds(e)[2:4], ["tz", "Rx"])
        self.assertEqual(e[2].param, a)

    def test_symbolic_quarter_turn_symbolic_other(self):
        e = padded(ET.Rx(3 * half_pi), ET.tz(a))
        ref = padded(ET.Rx(3 * half_pi), ET.tz(a))
        e.swap(2)
        # a quarter turn of -90 degrees turns z onto y
        self.assertEqual(kinds(e)[2:4], ["ty", "Rx"])
        self.assertEqual(e[2].param, a)
        qs = np.array([0.3, -0.2], dtype=object)
        self.assertTrue(sym_equal(e.fkine(qs).A, ref.fkine(qs).A))

    def test_sign_change_of_a_symbol(self):
        e = ET.Rx(half_pi) * ET.tz(a)
        e.swap(0)
        self.assertEqual(e[0].kind, "ty")
        self.assertEqual(e[0].param, -a)

    def test_symbolic_expression_other(self):
        e = ET.Ry(-half_pi) * ET.tx(2 * a + 1)
        ref = ET.Ry(-half_pi) * ET.tx(2 * a + 1)
        e.swap(0)
        self.assertEqual(kinds(e), ["tz", "Ry"])
        self.assertTrue(sym_equal(e.fkine().A, ref.fkine().A))

    def test_symbolic_other_rotation(self):
        e = ET.Rx(half_pi) * ET.Ry(a)
        ref = ET.Rx(half_pi) * ET.Ry(a)
        e.swap(0)
        self.assertEqual(kinds(e), ["Rz", "Rx"])
        self.assertEqual(e[0].param, a)
        self.assertTrue(sym_equal(e.fkine().A, ref.fkine().A))

    def test_both_orders_symbolic(self):
        e = ET.ty(a) * ET.Rx(half_pi)
        ref = ET.ty(a) * ET.Rx(half_pi)
        e.swap(0)
        self.assertEqual(kinds(e), ["Rx", "tz"])
        self.assertTrue(sym_equal(e.fkine().A, ref.fkine().A))

    def test_a_symbolic_angle_that_is_not_a_quarter_turn(self):
        for angle in (a, a + half_pi, sp.pi / 3):
            with self.subTest(angle=angle):
                with self.assertRaisesRegex(ValueError, "do not commute"):
                    (ET.Rx(angle) * ET.ty(1)).swap(0)

    def test_float_close_to_a_quarter_turn(self):
        # as read from a file
        e = ET.Rx(1.5707963267949) * ET.ty(1)
        e.swap(0)
        self.assertEqual(kinds(e), ["tz", "Rx"])
        with self.assertRaisesRegex(ValueError, "do not commute"):
            (ET.Rx(1.5707963) * ET.ty(1)).swap(0)  # seven digits is not close enough
        with self.assertRaisesRegex(ValueError, "do not commute"):
            (ET.Rx(0.3) * ET.ty(1)).swap(0)


class TestJoints(unittest.TestCase):
    def test_joint_is_carried(self):
        e = ET.Rx(pi / 2) * ET.Ry(jindex=2, qlim=[-1, 2])
        e.swap(0)
        self.assertEqual(kinds(e), ["Rz", "Rx"])
        self.assertTrue(e[0].isjoint)
        self.assertFalse(e[0].isflip)
        self.assertEqual(e[0].jindex, 2)
        nt.assert_array_equal(e[0].qlim, [-1, 2])

    def test_joint_is_flipped_if_its_axis_reverses(self):
        e = ET.Rx(-pi / 2) * ET.Ry()
        ref = ET.Rx(-pi / 2) * ET.Ry()
        e.swap(0)
        self.assertEqual(kinds(e), ["Rz", "Rx"])
        self.assertTrue(e[0].isflip)
        nt.assert_array_almost_equal(e.fkine([0.4]).A, ref.fkine([0.4]).A)

    def test_flipped_joint_back_again(self):
        e = ET.Rx(-pi / 2) * ET.Ry(flip=True)
        ref = ET.Rx(-pi / 2) * ET.Ry(flip=True)
        e.swap(0)
        self.assertFalse(e[0].isflip)
        nt.assert_array_almost_equal(e.fkine([0.4]).A, ref.fkine([0.4]).A)

    def test_numeric_offset_changes_sign(self):
        ref = ET.Rx(-pi / 2) * ET.Ry() * ET.Ry(0.5)  # not merged
        e = ET.Rx(-pi / 2) * ET.Ry() * ET.Ry(0.5)
        e.merge(1)
        e.swap(0)
        self.assertAlmostEqual(e[0].offset, -0.5)
        self.assertTrue(e[0].isflip)
        for qv in ([0.4], [-1.0]):
            nt.assert_array_almost_equal(e.fkine(qv).A, ref.fkine(qv).A)
            nt.assert_array_almost_equal(e.jacob0(qv), ref.jacob0(qv))

    def test_symbolic_offset_changes_sign(self):
        e = ET.Rx(-half_pi) * ET.Ry() * ET.Ry(a)
        e.merge(1)
        self.assertEqual(e[1].offset, a)
        e.swap(0)
        self.assertEqual(e[0].offset, -a)
        self.assertTrue(str(e).startswith("Rz(-q - a)"))

    def test_mixed_offset_and_symbolic_quarter(self):
        ref = ET.Rx(-half_pi) * ET.Ry() * ET.Ry(a) * ET.Ry(0.5)  # not merged
        e = ET.Rx(-half_pi) * ET.Ry() * ET.Ry(a) * ET.Ry(0.5)
        e.merge(1)
        e.merge(1)
        self.assertEqual(sp.simplify(e[1].offset - (a + 0.5)), 0)
        e.swap(0)
        self.assertEqual(sp.simplify(e[0].offset + a + 0.5), 0)
        qs = np.array([q], dtype=object)
        self.assertTrue(sym_equal(e.fkine(qs).A, ref.fkine(qs).A))

    def test_translation_joint_is_carried(self):
        e = ET.Rx(pi / 2) * ET.tz(jindex=1)
        ref = ET.Rx(pi / 2) * ET.tz(jindex=1)
        e.swap(0)
        self.assertEqual(kinds(e), ["ty", "Rx"])
        self.assertTrue(e[0].isflip)  # z turns onto -y
        nt.assert_array_almost_equal(
            e.fkine([0, 0.6]).A, ref.fkine([0, 0.6]).A
        )

    def test_swapping_back(self):
        e = ET.Rx(pi / 2) * ET.Ry()
        before = e.fkine([0.5]).A
        e.swap(0)
        e.swap(0)
        self.assertEqual(kinds(e), ["Rx", "Ry"])
        nt.assert_array_almost_equal(e.fkine([0.5]).A, before)


class TestJindices(unittest.TestCase):
    def test_cached_joint_indices_follow_a_swap(self):
        e = ET.Rz(jindex=0) * ET.Rz(jindex=1)
        nt.assert_array_equal(e.jindices, [0, 1])  # cached
        e.swap(0)
        nt.assert_array_equal(e.jindices, [1, 0])

    def test_cached_joint_indices_follow_a_swap_across_a_quarter_turn(self):
        e = ET.Rx(pi / 2) * ET.Ry(jindex=0) * ET.Rz(jindex=1)
        nt.assert_array_equal(e.jindices, [0, 1])
        e.swap(0)
        nt.assert_array_equal(e.jindices, [0, 1])
        e.swap(1)  # Rx(90) and Rz(q1)
        nt.assert_array_equal(e.jindices, [0, 1])
        self.assertEqual([t.jindex for t in e.joints()], [0, 1])


class TestErrors(unittest.TestCase):
    def test_two_quarter_turns_about_different_axes(self):
        with self.assertRaisesRegex(ValueError, "both quarter turns"):
            (ET.Rx(pi / 2) * ET.Ry(pi / 2)).swap(0)
        with self.assertRaisesRegex(ValueError, "both quarter turns"):
            (ET.Rx(half_pi) * ET.Ry(3 * half_pi)).swap(0)

    def test_two_quarter_turns_about_the_same_axis_commute(self):
        e = ET.Rx(pi / 2) * ET.Rx(pi)
        e.swap(0)
        self.assertAlmostEqual(e[0].param, pi)

    def test_not_a_quarter_turn(self):
        with self.assertRaisesRegex(ValueError, "do not commute"):
            (ET.Rx(0.3) * ET.ty(1)).swap(0)

    def test_a_joint_is_not_a_quarter_turn(self):
        with self.assertRaisesRegex(ValueError, "do not commute"):
            (ET.Rx() * ET.ty(1)).swap(0)

    def test_arbitrary_transform_cannot_be_carried(self):
        from spatialmath import SE3

        with self.assertRaisesRegex(ValueError, "do not commute"):
            (ET.Rx(pi / 2) * ET.SE3(SE3(1, 2, 3))).swap(0)

    def test_planar_is_unchanged(self):
        with self.assertRaisesRegex(ValueError, "do not commute"):
            (ET2.R(pi / 2) * ET2.tx(1)).swap(0)


class TestSwapped(unittest.TestCase):
    def test_returns_a_copy(self):
        e = padded(ET.Rx(half_pi), ET.ty(a))
        s = e.swapped(2)
        self.assertEqual(kinds(e)[2:4], ["Rx", "ty"])
        self.assertEqual(kinds(s)[2:4], ["tz", "Rx"])
        self.assertEqual(e[3].param, a)
        self.assertEqual(s[2].param, a)

    def test_errors_leave_the_original(self):
        e = ET.Rx(0.3) * ET.ty(1)
        with self.assertRaises(ValueError):
            e.swapped(0)
        self.assertEqual(kinds(e), ["Rx", "ty"])

    def test_shared_transform_is_not_changed(self):
        b = ET.ty(1)
        e = ET.Rx(pi / 2) * b
        e.swap(0)
        self.assertEqual(b.kind, "ty")
        self.assertEqual(b.param, 1)


if __name__ == "__main__":  # pragma: no cover
    pytest.main(["-q", __file__])
