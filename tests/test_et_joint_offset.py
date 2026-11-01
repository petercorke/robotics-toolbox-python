import math
import unittest

import numpy as np
import numpy.testing as nt
import pytest
import sympy as sp

from roboticstoolbox import ET, ET2, ETS2

a = sp.Symbol("a")
b = sp.Symbol("b")
q = sp.Symbol("q")


def kinds(e) -> list[str]:
    return [t.kind for t in e]


class TestOffsetProperty(unittest.TestCase):
    def test_default_is_zero(self):
        self.assertEqual(ET.Rz().offset, 0)
        self.assertEqual(ET2.R().offset, 0)

    def test_numeric(self):
        e = ET.Rz()
        e.offset = 0.5
        self.assertAlmostEqual(e.offset, 0.5)
        e.offset = 1  # an int becomes a float
        self.assertIsInstance(e.offset, float)

    def test_symbolic(self):
        e = ET.Rz()
        e.offset = a
        self.assertEqual(e.offset, a)

    def test_symbolic_nothing_is_zero(self):
        e = ET.Rz()
        e.offset = a - a
        self.assertEqual(e.offset, 0)
        # a whole number of turns of a rotation, but not of a translation
        e.offset = sp.pi + sp.pi
        self.assertEqual(e.offset, 0)
        e = ET.tz()
        e.offset = sp.pi + sp.pi
        self.assertNotEqual(e.offset, 0)

    def test_constant_has_no_offset(self):
        with self.assertRaisesRegex(TypeError, "only a joint"):
            ET.Rz(1).offset = 0.5

    def test_compiled_handle_follows_offset(self):
        e = ET.Rz()
        had_handle = e.fknm is not None
        e.offset = 0.5
        self.assertIsNone(e.fknm)
        e.offset = 0
        self.assertEqual(e.fknm is not None, had_handle)

    def test_copy_keeps_offset(self):
        from copy import deepcopy

        e = ET.Rz()
        e.offset = a
        self.assertEqual(deepcopy(e).offset, a)


class TestEvaluate(unittest.TestCase):
    def test_numeric(self):
        e = ET.Rz()
        e.offset = 0.5
        nt.assert_array_almost_equal(e.A(0.1), ET.Rz(0.6).A())

    def test_flipped(self):
        e = ET.Rz(flip=True)
        e.offset = 0.5
        nt.assert_array_almost_equal(e.A(0.1), ET.Rz(-0.1 + 0.5).A())

    def test_translation(self):
        e = ET.tx()
        e.offset = 2
        nt.assert_array_almost_equal(e.A(0.5), ET.tx(2.5).A())

    def test_symbolic_offset_numeric_q(self):
        e = ET.Rz()
        e.offset = a
        T = e.A(0.1)
        self.assertEqual(sp.simplify(T[0, 0] - sp.cos(a + 0.1)), 0)

    def test_numeric_offset_symbolic_q(self):
        e = ET.Rz()
        e.offset = 0.5
        T = e.A(q)
        self.assertEqual(sp.simplify(T[1, 0] - sp.sin(q + 0.5)), 0)

    def test_symbolic_offset_symbolic_q(self):
        e = ET.Rz()
        e.offset = a
        T = e.A(q)
        self.assertEqual(sp.simplify(T[1, 0] - sp.sin(q + a)), 0)

    def test_ets_fkine_and_jacobian_numeric(self):
        e = ET.tx(1) * ET.Rz() * ET.tx(2) * ET.Ry()
        e[1].offset = 0.4
        ref = ET.tx(1) * ET.Rz(0.4) * ET.Rz() * ET.tx(2) * ET.Ry()
        ref.merge(1)  # Rz(0.4) Rz(q): constant first, then the joint
        qv = [0.3, -0.2]
        nt.assert_array_almost_equal(e.fkine(qv).A, ref.fkine(qv).A)
        nt.assert_array_almost_equal(e.jacob0(qv), ref.jacob0(qv))

    def test_ets_fkine_trajectory(self):
        e = ET.Rz() * ET.tx(1)
        e[0].offset = 0.4
        qt = np.array([[0.1], [0.2], [0.3]])
        T = e.fkine(qt)
        for k in range(3):
            nt.assert_array_almost_equal(
                T[k].A, (ET.Rz(qt[k, 0] + 0.4) * ET.tx(1)).fkine().A
            )

    def test_ets_fkine_symbolic(self):
        e = ET.Rz() * ET.tx(1)
        e[0].offset = a
        T = e.fkine(np.array([q], dtype=object)).A
        self.assertEqual(sp.simplify(T[0, 3] - sp.cos(q + a)), 0)

    def test_compiled_ik_says_why_it_cannot(self):
        e = ET.Rz() * ET.tx(1)
        e[0].offset = 0.4
        with self.assertRaisesRegex(RuntimeError, "offset"):
            e.ik_LM(e.fkine([0.1]).A)
        # the Python solver does work
        sol = e.ikine_LM(e.fkine([0.1]), q0=[0.0])
        self.assertTrue(sol.success)
        self.assertAlmostEqual(sol.q[0], 0.1, delta=1e-2)


class TestCompiledIKLimits(unittest.TestCase):
    """The ik_XX solvers need the compiled handle, the ikine_XX solvers do not."""

    def test_sympy_parameter_with_a_numeric_value(self):
        e = ET.Rz() * ET.tx(1) * ET.Rx(sp.pi / 2) * ET.Rz() * ET.tx(0.5)
        T = e.fkine([0.3, 0.4])
        for name in ("ik_LM", "ik_NR", "ik_GN"):
            with self.assertRaisesRegex(RuntimeError, "SymPy parameter"):
                getattr(e, name)(T)
        sol = e.ikine_LM(T, q0=[0.2, 0.3])  # a start near the solution, not random
        self.assertTrue(sol.success)
        nt.assert_array_almost_equal(sol.q, [0.3, 0.4], decimal=4)

    def test_joint_offset(self):
        e = ET.Rz() * ET.tx(1)
        e[0].offset = 0.4
        for name in ("ik_LM", "ik_NR", "ik_GN"):
            with self.assertRaisesRegex(RuntimeError, "joint with an offset"):
                getattr(e, name)(e.fkine([0.1]).A)


class TestDisplay(unittest.TestCase):
    def test_numeric(self):
        e = ET.Rz()
        e.offset = math.pi / 2
        self.assertEqual(str(e), "Rz(q + 90°)")
        e.offset = -math.pi / 2
        self.assertEqual(str(e), "Rz(q - 90°)")

    def test_translation(self):
        e = ET.tx()
        e.offset = 2
        self.assertEqual(str(e), "tx(q + 2)")

    def test_symbolic(self):
        e = ET.Rz()
        e.offset = a
        self.assertEqual(str(e), "Rz(q + a)")
        e.offset = -a
        self.assertEqual(str(e), "Rz(q - a)")
        e.offset = a + b
        self.assertEqual(str(e), "Rz(q + a + b)")
        e.offset = a + 0.5
        self.assertEqual(str(e), "Rz(q + a + 0.5)")

    def test_ets(self):
        e = ET.Rz() * ET.tx(1) * ET.Ry(flip=True)
        e[0].offset = a
        e[2].offset = 0.5
        self.assertEqual(str(e), "Rz(q0 + a) ⊕ tx(1) ⊕ Ry(-q1 + 28.65°)")

    def test_repr_and_equality(self):
        e = ET.Rz()
        e.offset = 0.5
        self.assertIn("offset=0.5", repr(e))
        f = ET.Rz()
        self.assertNotEqual(e, f)
        f.offset = 0.5
        self.assertEqual(e, f)


class TestInverse(unittest.TestCase):
    def test_inverse_is_the_inverse(self):
        e = ET.Rz()
        e.offset = 0.5
        for qv in (0.0, 0.3, -1.1):
            nt.assert_array_almost_equal(e.A(qv) @ e.inv().A(qv), np.eye(4))

    def test_inverse_symbolic(self):
        e = ET.Rz()
        e.offset = a
        inv = e.inv()
        self.assertTrue(inv.isflip)
        self.assertEqual(inv.offset, -a)
        self.assertEqual(e.offset, a)  # the original is untouched

    def test_ets_inverse(self):
        e = ET.Rz() * ET.tx(1)
        e[0].offset = 0.5
        T = e.fkine([0.3]).A @ e.inv().fkine([0.3]).A
        nt.assert_array_almost_equal(T, np.eye(4))


class TestMergeJoint(unittest.TestCase):
    def padded(self, *mid):
        e = ET.tz(1) * ET.Ry(0.2)
        for m in mid:
            e = e * m
        return e * ET.tx(2) * ET.Rx()

    def test_joint_then_numeric_constant(self):
        e = self.padded(ET.Rz(), ET.Rz(0.5))
        e.merge(2)
        self.assertEqual(kinds(e), ["tz", "Ry", "Rz", "tx", "Rx"])
        self.assertTrue(e[2].isjoint)
        self.assertAlmostEqual(e[2].offset, 0.5)
        self.assertEqual(e[2].jindex, 0)
        self.assertEqual(e[4].jindex, 1)

    def test_constant_then_joint(self):
        e = self.padded(ET.Rz(0.5), ET.Rz())
        e.merge(2)
        self.assertEqual(kinds(e), ["tz", "Ry", "Rz", "tx", "Rx"])
        self.assertTrue(e[2].isjoint)
        self.assertAlmostEqual(e[2].offset, 0.5)

    def test_symbolic_constant(self):
        e = self.padded(ET.Rz(), ET.Rz(a))
        e.merge(2)
        self.assertEqual(e[2].offset, a)
        self.assertEqual(str(e[2]), "Rz(q0 + a)")

    def test_symbolic_then_numeric_constant(self):
        e = self.padded(ET.Rz(), ET.Rz(a), ET.Rz(0.5))
        e.merge(2)
        e.merge(2)
        self.assertEqual(kinds(e), ["tz", "Ry", "Rz", "tx", "Rx"])
        self.assertEqual(sp.simplify(e[2].offset - (a + 0.5)), 0)

    def test_numeric_then_symbolic_constant(self):
        e = self.padded(ET.Rz(), ET.Rz(0.5), ET.Rz(a))
        e.merge(2)
        e.merge(2)
        self.assertEqual(sp.simplify(e[2].offset - (a + 0.5)), 0)

    def test_constant_cancels_the_offset(self):
        # Rz(-a) Rz(q) Rz(a) is Rz(q), still a joint
        e = self.padded(ET.Rz(-a), ET.Rz(), ET.Rz(a))
        e.merge(3)
        self.assertEqual(e[3].offset, a)
        e.merge(2)  # Rz(-a) with Rz(q + a)
        self.assertEqual(kinds(e), ["tz", "Ry", "Rz", "tx", "Rx"])
        self.assertTrue(e[2].isjoint)
        self.assertEqual(e[2].offset, 0)
        self.assertEqual(str(e[2]), "Rz(q0)")

    def test_numeric_constant_cancels_the_offset(self):
        e = self.padded(ET.Rz(), ET.Rz(0.5), ET.Rz(-0.5))
        e.merge(2)
        e.merge(2)
        self.assertEqual(e[2].offset, 0)
        self.assertTrue(e[2].isjoint)

    def test_whole_turns_cancel_the_offset(self):
        e = self.padded(ET.Rz(sp.pi), ET.Rz(), ET.Rz(sp.pi))
        e.merge(3)
        e.merge(2)
        self.assertEqual(e[2].offset, 0)

    def test_translation_joint(self):
        e = self.padded(ET.tx(), ET.tx(a))
        e.merge(2)
        self.assertEqual(str(e[2]), "tx(q0 + a)")

    def test_flipped_joint(self):
        e = ET.Rz(flip=True) * ET.Rz(0.5)
        ref = ET.Rz(flip=True) * ET.Rz(0.5)
        e.merge(0)
        self.assertTrue(e[0].isflip)
        self.assertAlmostEqual(e[0].offset, 0.5)
        nt.assert_array_almost_equal(e.fkine([0.3]).A, ref.fkine([0.3]).A)

    def test_keeps_jindex_and_limits(self):
        e = ET.tx(1) * ET.Rz(jindex=3, qlim=[-1, 1]) * ET.Rz(0.5)
        e.merge(1)
        self.assertEqual(e[1].jindex, 3)
        nt.assert_array_equal(e[1].qlim, [-1, 1])

    def test_kinematics_unchanged_numeric(self):
        e = self.padded(ET.Rz(), ET.Rz(0.5))
        ref = self.padded(ET.Rz(), ET.Rz(0.5))
        qv = [0.3, -0.2]
        m = e.merged(2)
        nt.assert_array_almost_equal(m.fkine(qv).A, ref.fkine(qv).A)
        nt.assert_array_almost_equal(m.jacob0(qv), ref.jacob0(qv))

    def test_kinematics_unchanged_symbolic(self):
        e = ET.Rz() * ET.Rz(a) * ET.tx(1)
        ref = ET.Rz() * ET.Rz(a) * ET.tx(1)
        m = e.merged(0)
        qs = np.array([q], dtype=object)
        diff = sp.simplify(sp.Matrix(m.fkine(qs).A) - sp.Matrix(ref.fkine(qs).A))
        self.assertEqual(diff, sp.zeros(4, 4))

    def test_two_joints_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "both transforms are joints"):
            (ET.Rz() * ET.Rz()).merge(0)

    def test_different_kinds_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "not the same kind"):
            (ET.Rz() * ET.tz(1)).merge(0)

    def test_merged_returns_a_copy(self):
        e = self.padded(ET.Rz(), ET.Rz(0.5))
        m = e.merged(2)
        self.assertEqual(len(e), 6)
        self.assertEqual(len(m), 5)
        self.assertEqual(e[2].offset, 0)
        self.assertAlmostEqual(m[2].offset, 0.5)

    def test_shared_joint_is_not_changed(self):
        j = ET.Rz()
        e = ET.tx(1) * j * ET.Rz(0.5)
        e.merge(1)
        self.assertEqual(j.offset, 0)

    def test_planar(self):
        e = ET2.tx(1) * ET2.R() * ET2.R(a) * ET2.ty(2)
        m = e.merged(1)
        self.assertIsInstance(m, ETS2)
        self.assertEqual(str(m[1]), "R(q0 + a)")
        e = ET2.R() * ET2.R(0.5)
        e.merge(0)
        nt.assert_array_almost_equal(e.fkine([0.2]).A, ETS2([ET2.R(0.7)]).fkine().A)


class TestFixjoint(unittest.TestCase):
    def test_fixed_joint_has_its_offset(self):
        e = ET.Rz() * ET.tx(1) * ET.Rz()
        e[0].offset = 0.5
        f = e.fixjoint(0, 0.3)
        self.assertEqual(f.n, 1)
        ref = ET.Rz(0.8) * ET.tx(1) * ET.Rz()
        nt.assert_array_almost_equal(f.fkine([0.2]).A, ref.fkine([0.2]).A)

    def test_fixed_joint_in_degrees(self):
        e = ET.Rz() * ET.tx(1)
        e[0].offset = math.radians(10)
        f = e.fixjoint(0, 20, unit="deg")
        self.assertAlmostEqual(f[0].param, math.radians(30))

    def test_fixed_flipped_joint(self):
        e = ET.Rz(flip=True) * ET.tx(1)
        e[0].offset = 0.5
        f = e.fixjoint(0, 0.3)
        self.assertAlmostEqual(f[0].param, -0.3 + 0.5)

    def test_other_joint_keeps_its_offset(self):
        e = ET.Rz() * ET.tx(1) * ET.Rz()
        e[2].offset = a
        f = e.fixjoint(0, 0.3)
        self.assertEqual(f[2].offset, a)

    def test_symbolic_offset_of_fixed_joint(self):
        e = ET.Rz() * ET.tx(1)
        e[0].offset = a
        f = e.fixjoint(0, 0.3)
        self.assertEqual(sp.simplify(f[0].param - (a + 0.3)), 0)

    def test_planar(self):
        e = ET2.R() * ET2.tx(1)
        e[0].offset = 0.5
        f = e.fixjoint(0, 0.3)
        self.assertAlmostEqual(f[0].param, 0.8)


if __name__ == "__main__":  # pragma: no cover
    pytest.main(["-q", __file__])
