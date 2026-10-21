import math
import unittest

import numpy.testing as nt
import pytest
import sympy as sp

from roboticstoolbox import ET, ET2, ETS, ETS2

x = sp.Symbol("x")


def padded(a: ET, b: ET) -> ETS:
    """The two terms with other terms before and after, which are at index 2, 3."""
    return ET.tz(1) * ET.Ry(0.2) * a * b * ET.tx(2) * ET.Rz()


class TestMerge(unittest.TestCase):
    def test_symbol_plus_number(self):
        e = padded(ET.Rx(x), ET.Rx(0.3))
        e.merge(2)
        self.assertEqual(len(e), 5)
        self.assertEqual(e[2].kind, "Rx")
        self.assertEqual(sp.simplify(e[2].param - (x + 0.3)), 0)
        # neighbours are untouched
        self.assertEqual([t.kind for t in e], ["tz", "Ry", "Rx", "tx", "Rz"])

    def test_symbol_twice(self):
        e = padded(ET.Rx(x), ET.Rx(x))
        e.merge(2)
        self.assertEqual(len(e), 5)
        self.assertEqual(sp.simplify(e[2].param - 2 * x), 0)

    def test_symbol_cancels(self):
        e = padded(ET.Rx(x), ET.Rx(-x))
        e.merge(2)
        self.assertEqual([t.kind for t in e], ["tz", "Ry", "tx", "Rz"])

    def test_number_cancels(self):
        e = padded(ET.Rx(5), ET.Rx(-5))
        e.merge(2)
        self.assertEqual([t.kind for t in e], ["tz", "Ry", "tx", "Rz"])

    def test_translation_cancels(self):
        e = padded(ET.tx(x), ET.tx(-x))
        e.merge(2)
        self.assertEqual([t.kind for t in e], ["tz", "Ry", "tx", "Rz"])

    def test_whole_turns_of_symbolic_angle_cancel(self):
        e = padded(ET.Rx(sp.pi / 2), ET.Rx(3 * sp.pi / 2))
        e.merge(2)
        self.assertEqual([t.kind for t in e], ["tz", "Ry", "tx", "Rz"])

        e = padded(ET.Rx(x + sp.pi), ET.Rx(-x + sp.pi))
        e.merge(2)
        self.assertEqual([t.kind for t in e], ["tz", "Ry", "tx", "Rz"])

    def test_whole_turns_only_for_rotations(self):
        e = padded(ET.tx(sp.pi), ET.tx(sp.pi))
        e.merge(2)
        self.assertEqual(len(e), 5)

    def test_float_turns_are_not_removed(self):
        # not exactly zero
        e = padded(ET.Rx(math.pi), ET.Rx(math.pi))
        e.merge(2)
        self.assertEqual(len(e), 5)

    def test_a_half_turn_is_not_removed(self):
        e = padded(ET.Rx(sp.pi / 2), ET.Rx(sp.pi / 2))
        e.merge(2)
        self.assertEqual(len(e), 5)

    def test_kinematics_unchanged(self):
        q = [0.3]
        e = ET.tz(1) * ET.Rx(0.4) * ET.Rx(0.5) * ET.tx(2) * ET.Rz()
        before = e.fkine(q).A
        e.merge(1)
        nt.assert_array_almost_equal(e.fkine(q).A, before)

    def test_returns_none_and_changes_in_place(self):
        e = padded(ET.Rx(0.1), ET.Rx(0.2))
        ident = id(e)
        self.assertIsNone(e.merge(2))
        self.assertEqual(id(e), ident)
        self.assertEqual(len(e), 5)

    def test_does_not_change_a_shared_transform(self):
        a = ET.Rx(0.1)
        e = ET.tx(1) * a * ET.Rx(0.2)
        e.merge(1)
        self.assertAlmostEqual(a.param, 0.1)

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "not the same kind"):
            (ET.tx(1) * ET.Rz(1)).merge(0)
        with self.assertRaisesRegex(ValueError, "joint"):
            (ET.Rz() * ET.Rz()).merge(0)
        for i in (-1, 1):
            with self.assertRaises(IndexError):
                (ET.tx(1) * ET.tx(2)).merge(i)


class TestMerged(unittest.TestCase):
    def test_returns_a_copy(self):
        e = padded(ET.Rx(0.1), ET.Rx(0.2))
        m = e.merged(2)
        self.assertEqual(len(e), 6)
        self.assertEqual(len(m), 5)
        self.assertIsNot(m, e)
        self.assertIs(type(m), type(e))
        self.assertAlmostEqual(m[2].param, 0.3)
        self.assertAlmostEqual(e[2].param, 0.1)

    def test_same_result_as_merge(self):
        for a, b in ((ET.Rx(x), ET.Rx(0.3)), (ET.Rx(x), ET.Rx(-x))):
            e = padded(a, b)
            m = e.merged(2)
            e.merge(2)
            self.assertEqual(str(m), str(e))

    def test_errors_leave_original(self):
        e = ET.tx(1) * ET.Rz(1)
        with self.assertRaises(ValueError):
            e.merged(0)
        self.assertEqual(len(e), 2)


class TestSwap(unittest.TestCase):
    def test_swap(self):
        q = [0.3, -0.4]
        e = ET.Rz() * ET.tx(1) * ET.ty(2) * ET.Rz()
        before = e.fkine(q).A
        self.assertIsNone(e.swap(1))
        self.assertEqual([t.kind for t in e], ["Rz", "ty", "tx", "Rz"])
        nt.assert_array_almost_equal(e.fkine(q).A, before)

    def test_swapped_joints_keep_their_q(self):
        q = [0.3, -0.4]
        e = ET.Rz(jindex=0) * ET.Rz(jindex=1)
        s = e.swapped(0)
        self.assertEqual([t.jindex for t in s], [1, 0])
        nt.assert_array_almost_equal(s.fkine(q).A, e.fkine(q).A)

    def test_swapped_returns_a_copy(self):
        e = ET.Rz() * ET.tx(1) * ET.ty(2) * ET.Rz()
        s = e.swapped(1)
        self.assertEqual([t.kind for t in e], ["Rz", "tx", "ty", "Rz"])
        self.assertEqual([t.kind for t in s], ["Rz", "ty", "tx", "Rz"])
        self.assertIsNot(s, e)

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "do not commute"):
            (ET.Rz() * ET.tx(1)).swap(0)
        with self.assertRaisesRegex(ValueError, "do not commute"):
            (ET.Rz() * ET.tx(1)).swapped(0)
        for i in (-1, 1):
            with self.assertRaises(IndexError):
                (ET.tx(1) * ET.tx(2)).swap(i)


class TestPlanar(unittest.TestCase):
    def test_merge_and_swap_2d(self):
        e = ET2.tx(1) * ET2.R(0.2) * ET2.R(0.3) * ET2.ty(2)
        m = e.merged(1)
        self.assertIsInstance(m, ETS2)
        self.assertEqual(len(m), 3)
        self.assertAlmostEqual(m[1].param, 0.5)

        e = ET2.tx(1) * ET2.R(0.2) * ET2.R(-0.2) * ET2.ty(2)
        self.assertEqual(len(e.merged(1)), 2)

        s = (ET2.tx(1) * ET2.ty(2)).swapped(0)
        self.assertEqual([t.kind for t in s], ["ty", "tx"])


if __name__ == "__main__":  # pragma: no cover
    pytest.main(["-q", __file__])
