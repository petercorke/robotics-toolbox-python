import math
import unittest

import numpy as np
import numpy.testing as nt
import pytest
import sympy as sp

from roboticstoolbox import ET, ETS

a = sp.Symbol("a")
b = sp.Symbol("b")
q = sp.Symbol("q")
pi = math.pi

AXES = "xyz"


def kinds(e) -> list[str]:
    return [t.kind for t in e]


def sym_equal(T1, T2) -> bool:
    return sp.simplify(sp.Matrix(T1) - sp.Matrix(T2)) == sp.zeros(4, 4)


def padded(mid) -> ETS:
    """Terms before and after; the term in the middle is at index 2."""
    return ET.tz(1) * ET.Ry(0.2) * mid * ET.tx(2) * ET.Rx()


class TestKinematicsUnchanged(unittest.TestCase):
    def test_every_kind_and_axis(self):
        for kind in ("R", "t"):
            for old in AXES:
                for new in AXES:
                    if old == new:
                        continue
                    with self.subTest(kind=kind + old, to=new):
                        e = padded(getattr(ET, kind + old)(0.7))
                        before = e.fkine([0.3, -0.2]).A
                        e.replace(2, new)
                        self.assertEqual(len(e), 7)
                        nt.assert_array_almost_equal(
                            e.fkine([0.3, -0.2]).A, before
                        )
                        # the middle one of the three is on the new axis
                        self.assertEqual(e[3].kind, kind + new)
                        self.assertAlmostEqual(e[3].param, 0.7)

    def test_the_substitutions_of_the_paper(self):
        for kind, new, expect in (
            ("Rx", "z", "Ry(90°) ⊕ Rz(q) ⊕ Ry(-90°)"),
            ("Ry", "z", "Rx(-90°) ⊕ Rz(q) ⊕ Rx(90°)"),
            ("tx", "z", "Ry(90°) ⊕ tz(q) ⊕ Ry(-90°)"),
            ("ty", "z", "Rx(-90°) ⊕ tz(q) ⊕ Rx(90°)"),
        ):
            with self.subTest(kind=kind):
                e = getattr(ET, kind)()
                ets = ETS([e])
                ets.replace(0, new)
                self.assertEqual(str(ets), expect)

    def test_same_axis_does_nothing(self):
        e = padded(ET.tx(0.7))
        e.replace(2, "x")
        self.assertEqual(len(e), 5)
        self.assertEqual(kinds(e), ["tz", "Ry", "tx", "tx", "Rx"])

    def test_trajectory_of_joints(self):
        e = ET.Rz() * ET.tx(1) * ET.Ry() * ET.tx(0.5)
        qt = np.array([[0.1, 0.2], [0.3, -0.4]])
        before = e.fkine(qt)
        e.replace(2, "z")
        after = e.fkine(qt)
        for k in range(2):
            nt.assert_array_almost_equal(after[k].A, before[k].A)


class TestSymbolic(unittest.TestCase):
    def test_symbolic_translation(self):
        e = padded(ET.tx(a))
        ref = padded(ET.tx(a))
        e.replace(2, "z")
        self.assertEqual(kinds(e)[2:5], ["Ry", "tz", "Ry"])
        self.assertEqual(e[3].param, a)
        # exact quarter turns, since the transform is symbolic
        self.assertEqual(e[2].param, sp.pi / 2)
        self.assertEqual(e[4].param, -sp.pi / 2)
        qs = np.array([0.3, -0.2], dtype=object)
        self.assertTrue(sym_equal(e.fkine(qs).A, ref.fkine(qs).A))

    def test_symbolic_expression(self):
        e = padded(ET.ty(2 * a + b))
        ref = padded(ET.ty(2 * a + b))
        e.replace(2, "x")
        self.assertEqual(e[3].param, 2 * a + b)
        qs = np.array([0.3, -0.2], dtype=object)
        self.assertTrue(sym_equal(e.fkine(qs).A, ref.fkine(qs).A))

    def test_symbolic_rotation(self):
        e = padded(ET.Rx(a))
        ref = padded(ET.Rx(a))
        e.replace(2, "y")
        self.assertEqual(kinds(e)[2:5], ["Rz", "Ry", "Rz"])
        qs = np.array([0.3, -0.2], dtype=object)
        self.assertTrue(sym_equal(e.fkine(qs).A, ref.fkine(qs).A))

    def test_numeric_stays_numeric(self):
        e = padded(ET.tx(0.7))
        e.replace(2, "z")
        self.assertIsInstance(e[2].param, float)
        self.assertAlmostEqual(e[2].param, pi / 2)

    def test_joint_with_a_symbolic_offset(self):
        e = ET.Rz() * ET.Ry() * ET.Ry(a) * ET.tx(1)
        ref = ET.Rz() * ET.Ry() * ET.Ry(a) * ET.tx(1)
        e.merge(1)
        e.replace(1, "z")
        self.assertEqual(kinds(e), ["Rz", "Rx", "Rz", "Rx", "tx"])
        self.assertEqual(e[2].offset, a)
        self.assertEqual(e[1].param, -sp.pi / 2)
        qs = np.array([0.3, q], dtype=object)
        self.assertTrue(sym_equal(e.fkine(qs).A, ref.fkine(qs).A))

    def test_joint_with_a_numeric_offset(self):
        e = ET.Rz() * ET.Ry() * ET.Ry(0.5) * ET.tx(1)
        ref = ET.Rz() * ET.Ry() * ET.Ry(0.5) * ET.tx(1)
        e.merge(1)
        e.replace(1, "x")
        self.assertAlmostEqual(e[2].offset, 0.5)
        self.assertIsInstance(e[1].param, float)
        for qv in ([0.3, -0.2], [1.0, 2.0]):
            nt.assert_array_almost_equal(e.fkine(qv).A, ref.fkine(qv).A)
            nt.assert_array_almost_equal(e.jacob0(qv), ref.jacob0(qv))


class TestJoints(unittest.TestCase):
    def test_joint_keeps_everything(self):
        e = ET.tx(1) * ET.Ry(jindex=3, qlim=[-1, 2], flip=True)
        ref = ET.tx(1) * ET.Ry(jindex=3, qlim=[-1, 2], flip=True)
        e.replace(1, "z")
        j = e[2]
        self.assertTrue(j.isjoint)
        self.assertTrue(j.isflip)
        self.assertEqual(j.jindex, 3)
        nt.assert_array_equal(j.qlim, [-1, 2])
        self.assertEqual(j.kind, "Rz")
        qv = [0, 0, 0, 0.4]
        nt.assert_array_almost_equal(e.fkine(qv).A, ref.fkine(qv).A)

    def test_translation_joint(self):
        e = ET.Rz(jindex=0) * ET.tx(jindex=1)
        ref = ET.Rz(jindex=0) * ET.tx(jindex=1)
        e.replace(1, "y")
        self.assertEqual(kinds(e), ["Rz", "Rz", "ty", "Rz"])
        nt.assert_array_almost_equal(
            e.fkine([0.5, 0.7]).A, ref.fkine([0.5, 0.7]).A
        )

    def test_cached_joint_indices(self):
        e = ET.Rz(jindex=0) * ET.Ry(jindex=1)
        nt.assert_array_equal(e.jindices, [0, 1])
        e.replace(0, "x")
        nt.assert_array_equal(e.jindices, [0, 1])
        self.assertEqual([t.jindex for t in e.joints()], [0, 1])

    def test_with_swap_and_merge(self):
        # Ry(q) Tz(a) Ry(b): swap, replace and merge work together
        e = ET.tz(1) * ET.Rx(pi / 2) * ET.ty(a)
        e.swap(1)
        self.assertEqual(kinds(e), ["tz", "tz", "Rx"])
        e.merge(0)
        self.assertEqual(kinds(e), ["tz", "Rx"])
        self.assertEqual(sp.simplify(e[0].param - (1 + a)), 0)


class TestErrors(unittest.TestCase):
    def test_bad_axis(self):
        for axis in ("w", "X", "", "xy", None):
            with self.assertRaisesRegex(ValueError, "axis is"):
                (ET.tx(1) * ET.tx(2)).replace(0, axis)  # type: ignore[arg-type]

    def test_bad_index(self):
        e = ET.tx(1) * ET.tx(2)
        for i in (-1, 2):
            with self.assertRaises(IndexError):
                e.replace(i, "z")

    def test_arbitrary_transform_has_no_axis(self):
        from spatialmath import SE3

        e = ET.tx(1) * ET.SE3(SE3(1, 2, 3))
        with self.assertRaisesRegex(ValueError, "no axis"):
            e.replace(1, "z")


class TestReplaced(unittest.TestCase):
    def test_returns_a_copy(self):
        e = padded(ET.tx(a))
        r = e.replaced(2, "z")
        self.assertEqual(kinds(e), ["tz", "Ry", "tx", "tx", "Rx"])
        self.assertEqual(len(r), 7)
        self.assertIsNot(r, e)

    def test_errors_leave_the_original(self):
        e = ET.tx(1) * ET.tx(2)
        with self.assertRaises(ValueError):
            e.replaced(0, "w")
        self.assertEqual(len(e), 2)

    def test_shared_transform_is_not_changed(self):
        t = ET.tx(1)
        e = ET.tz(1) * t
        e.replace(1, "y")
        self.assertEqual(t.kind, "tx")
        self.assertEqual(t.param, 1)


if __name__ == "__main__":  # pragma: no cover
    pytest.main(["-q", __file__])
