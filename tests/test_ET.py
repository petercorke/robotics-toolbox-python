#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as sm
from spatialmath import SE3
import unittest
from roboticstoolbox.ets._ET import BaseET
import sympy
from copy import copy, deepcopy


class TestET(unittest.TestCase):
    def test_TRx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.Rx(fl).A(), sm.trotx(fl))
        nt.assert_array_almost_equal(rtb.ET.Rx(-fl).A(), sm.trotx(-fl))
        nt.assert_array_almost_equal(rtb.ET.Rx(0).A(), sm.trotx(0))

        nt.assert_array_almost_equal(
            rtb.ET.Rx(90.0, unit="degr").A(), sm.trotx(np.pi / 2, unit="rad")
        )

        nt.assert_array_almost_equal(
            rtb.ET.Rx(90.0, unit="deg").A(), sm.trotx(np.pi / 2, unit="rad")
        )

        nt.assert_array_almost_equal(
            rtb.ET.Rx(90.0, unit="Deg").A(), sm.trotx(np.pi / 2, unit="rad")
        )

    def test_TRy(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.Ry(fl).A(), sm.troty(fl))
        nt.assert_array_almost_equal(rtb.ET.Ry(-fl).A(), sm.troty(-fl))
        nt.assert_array_almost_equal(rtb.ET.Ry(0).A(), sm.troty(0))

    def test_TRz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.Rz(fl).A(), sm.trotz(fl))
        nt.assert_array_almost_equal(rtb.ET.Rz(-fl).A(), sm.trotz(-fl))
        nt.assert_array_almost_equal(rtb.ET.Rz(0).A(), sm.trotz(0))

        nt.assert_array_almost_equal(rtb.ET.Rz().A(fl), sm.trotz(fl))

    def test_Ttx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.tx(fl).A(), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(rtb.ET.tx(-fl).A(), sm.transl(-fl, 0, 0))
        nt.assert_array_almost_equal(rtb.ET.tx(0.0).A(), sm.transl(0, 0, 0))

    def test_Tty(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.ty(fl).A(), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(rtb.ET.ty(-fl).A(), sm.transl(0, -fl, 0))
        nt.assert_array_almost_equal(rtb.ET.ty(0).A(), sm.transl(0, 0, 0))

    def test_Ttz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.tz(fl).A(), sm.transl(0, 0, fl))
        nt.assert_array_almost_equal(rtb.ET.tz(-fl).A(), sm.transl(0, 0, -fl))
        nt.assert_array_almost_equal(rtb.ET.tz(0).A(), sm.transl(0, 0, 0))

    def test_SE3(self):
        T = SE3.Rx(0.3) * SE3.Rz(0.3) * SE3.Ry(0.3)

        nt.assert_array_almost_equal(rtb.ET.SE3(T).A(), T.A)
        nt.assert_array_almost_equal(rtb.ET.SE3(T.A).A(), T.A)

    def test_str(self):
        x = sympy.Symbol("x")
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)
        rz = rtb.ET.Rz(1.543)
        tx = rtb.ET.tx(1.543)
        ty = rtb.ET.ty(1.543)
        tz = rtb.ET.tz(1.543)
        r2 = rtb.ET.tz(x)
        r3 = rtb.ET.tz(jindex=3)

        self.assertEqual(str(rx), "Rx(88.41°)")
        self.assertEqual(str(ry), "Ry(88.41°)")
        self.assertEqual(str(rz), "Rz(88.41°)")
        self.assertEqual(str(tx), "tx(1.543)")
        self.assertEqual(str(ty), "ty(1.543)")
        self.assertEqual(str(tz), "tz(1.543)")
        self.assertEqual(str(r2), "tz(x)")
        self.assertEqual(str(r3), "tz(q3)")

    def test_str_se3(self):
        a = rtb.ET.SE3(SE3(1.0, 0, 0))
        b = rtb.ET.SE3(SE3.RPY(1.0, 2.0, 3.00))
        c = rtb.ET.SE3(SE3(1.0, 0, 0) * SE3.RPY(1.0, 2.0, 3.00))

        self.assertEqual(str(a), "SE3(1, 0, 0)")
        self.assertEqual(str(b), "SE3(-122.7°, 65.41°, -8.113°)")
        self.assertEqual(str(c), "SE3(1, 0, 0; -122.7°, 65.41°, -8.113°)")

    def test_repr(self):
        rx = rtb.ET.Rx(1.543, jindex=5, flip=True, qlim=[-1, 1])
        tx = rtb.ET.tx(1.543, jindex=5, flip=True, qlim=[-1, 1])
        se = rtb.ET.SE3(SE3.Rx(0.3) * SE3.Ry(0.5), jindex=5, flip=True, qlim=[-1, 1])

        arx = "ET.Rx(param=1.543, jindex=5, flip=True, qlim=array([-1.,  1.]))"
        atx = "ET.tx(param=1.543, jindex=5, flip=True, qlim=array([-1.,  1.]))"
        ase = "ET.SE3(T=array([[ 0.87758256,  0.        ,  0.47942554,  0.        ],"

        print(repr(se))

        self.assertEqual(repr(rx), arx)
        self.assertEqual(repr(tx), atx)
        self.assertTrue(repr(se).startswith(ase))

    def test_str_q(self):
        rx = rtb.ET.Rx()
        ry = rtb.ET.Ry()
        rz = rtb.ET.Rz()
        tx = rtb.ET.tx()
        ty = rtb.ET.ty()
        tz = rtb.ET.tz()

        self.assertEqual(str(rx), "Rx(q)")
        self.assertEqual(str(ry), "Ry(q)")
        self.assertEqual(str(rz), "Rz(q)")
        self.assertEqual(str(tx), "tx(q)")
        self.assertEqual(str(ty), "ty(q)")
        self.assertEqual(str(tz), "tz(q)")

    def test_T_real(self):
        fl = 1.543
        rx = rtb.ET.Rx(fl)
        ry = rtb.ET.Ry(fl)
        rz = rtb.ET.Rz(fl)
        tx = rtb.ET.tx(fl)
        ty = rtb.ET.ty(fl)
        tz = rtb.ET.tz(fl)

        nt.assert_array_almost_equal(rx.A(), sm.trotx(fl))
        nt.assert_array_almost_equal(ry.A(), sm.troty(fl))
        nt.assert_array_almost_equal(rz.A(), sm.trotz(fl))
        nt.assert_array_almost_equal(tx.A(), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.A(), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.A(), sm.transl(0, 0, fl))

    def test_T_real_2(self):
        fl = 1.543
        rx = rtb.ET.Rx()
        ry = rtb.ET.Ry()
        rz = rtb.ET.Rz()
        tx = rtb.ET.tx()
        ty = rtb.ET.ty()
        tz = rtb.ET.tz()

        nt.assert_array_almost_equal(rx.A(fl), sm.trotx(fl))
        nt.assert_array_almost_equal(ry.A(fl), sm.troty(fl))
        nt.assert_array_almost_equal(rz.A(fl), sm.trotz(fl))
        nt.assert_array_almost_equal(tx.A(fl), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.A(fl), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.A(fl), sm.transl(0, 0, fl))

    def test_qlim(self):
        q1 = -1.0
        q2 = 1.0
        et1 = rtb.ET.Rx(1.5, qlim=[q1, q2])
        et2 = rtb.ET.Rx(1.5, qlim=np.array([q1, q2]))
        et3 = rtb.ET.Rx(1.5, qlim=np.array([[q1, q2]]))
        et4 = rtb.ET.Rx(1.5, qlim=np.array([[q1, q2]]).T)
        et5 = rtb.ET.Rx(1.5, qlim=(q1, q2))

        correct = np.array([q1, q2])
        nt.assert_array_almost_equal(et1.qlim, correct)
        nt.assert_array_almost_equal(et2.qlim, correct)
        nt.assert_array_almost_equal(et3.qlim, correct)
        nt.assert_array_almost_equal(et4.qlim, correct)
        nt.assert_array_almost_equal(et5.qlim, correct)

    def test_axis_error(self):
        with nt.assert_raises(TypeError):
            BaseET("Rx")

        with nt.assert_raises(TypeError):
            BaseET("Rx", param=0.5)

    def test_jindex(self):
        et1 = rtb.ET.Rx(1.5, jindex=2)
        self.assertEqual(et1.jindex, 2)

    def test_ets(self):
        ets = rtb.ET.Rx(1) * rtb.ET.tx(2)

        nt.assert_array_almost_equal(ets[0].A(), sm.trotx(1))
        nt.assert_array_almost_equal(ets[1].A(), sm.transl(2, 0, 0))

    def test_ets_add(self):
        ets = rtb.ET.Rx(1) + rtb.ET.tx(2)

        nt.assert_array_almost_equal(ets[0].A(), sm.trotx(1))
        nt.assert_array_almost_equal(ets[1].A(), sm.transl(2, 0, 0))

    def test_is_rot(self):
        e1 = rtb.ET.Rx()
        e2 = rtb.ET.tx()
        e3 = rtb.ET.SE3(SE3.Rx(0.5))

        self.assertTrue(e1.isrotation)
        self.assertFalse(e1.istranslation)

        self.assertTrue(e2.istranslation)
        self.assertFalse(e2.isrotation)

        self.assertFalse(e3.isrotation)
        self.assertFalse(e3.istranslation)

        self.assertTrue(e1.isjoint)
        self.assertTrue(e2.isjoint)
        self.assertFalse(e3.isjoint)

    def test_T(self):
        x = sympy.Symbol("x")
        fl = 1.543
        r1 = rtb.ET.Rx()
        r2 = rtb.ET.Rx(flip=True)
        r2.A(x)

        nt.assert_array_almost_equal(r1.A(fl), sm.trotx(fl))
        nt.assert_array_almost_equal(r2.A(fl), sm.trotx(-fl))
        nt.assert_array_almost_equal(r1.A(x), sm.trotx(x))

    def test_copy(self):
        r1 = rtb.ET.Rx(flip=True)
        r2 = copy(r1)
        r3 = deepcopy(r1)

        nt.assert_array_almost_equal(r1.A(1.0), sm.trotx(-1.0))
        nt.assert_array_almost_equal(r2.A(1.0), sm.trotx(-1.0))
        nt.assert_array_almost_equal(r3.A(1.0), sm.trotx(-1.0))

        self.assertEqual(r1.fknm, r2.fknm)
        self.assertNotEqual(r1.fknm, r3.fknm)

    def test_eq(self):
        r1 = rtb.ET.Rx(2.5)
        r2 = rtb.ET.Rx(2.5)

        self.assertEqual(r1, r2)

    def test_update_jindex(self):
        r1 = rtb.ET.Rx(2.5)
        r1.jindex = 3
        self.assertEqual(r1.jindex, 3)

    def test_isrotation(self):
        r1 = rtb.ET.Rx(2.5)
        r2 = rtb.ET.tx(1.0)
        r3 = rtb.ET.SE3(SE3.Rx(0.5) * SE3.Tx(1.0))
        r4 = rtb.ET.Rx()

        self.assertEqual(r1.isrotation, True)
        self.assertEqual(r2.isrotation, False)
        self.assertEqual(r3.isrotation, False)
        self.assertEqual(r4.isrotation, True)

    def test_istrasnslation(self):
        r1 = rtb.ET.Rx(2.5)
        r2 = rtb.ET.tx(1.0)
        r3 = rtb.ET.SE3(SE3.Rx(0.5) * SE3.Tx(1.0))
        r4 = rtb.ET.Rx()

        self.assertEqual(r1.istranslation, False)
        self.assertEqual(r2.istranslation, True)
        self.assertEqual(r3.istranslation, False)
        self.assertEqual(r4.istranslation, False)

    def test_iselementary(self):
        r1 = rtb.ET.Rx(2.5)
        r2 = rtb.ET.tx(1.0)
        r3 = rtb.ET.SE3(SE3.Rx(0.5) * SE3.Tx(1.0))
        r4 = rtb.ET.Rx()

        self.assertEqual(r1.iselementary, True)
        self.assertEqual(r2.iselementary, True)
        self.assertEqual(r3.iselementary, False)
        self.assertEqual(r4.iselementary, True)

    def test_inv(self):
        se3 = SE3.Rx(0.5) * SE3.Tx(1.0)
        r1 = rtb.ET.Rx(2.5)
        r2 = rtb.ET.tx(1.0)
        r3 = rtb.ET.SE3(se3)
        r4 = rtb.ET.Rx()

        r1i = r1.inv()
        r2i = r2.inv()
        r3i = r3.inv()
        r4i = r4.inv()

        nt.assert_almost_equal(r1i.A(), np.linalg.inv(SE3.Rx(2.5).A))
        nt.assert_almost_equal(r2i.A(), np.linalg.inv(SE3.Tx(1.0).A))
        nt.assert_almost_equal(r3i.A(), np.linalg.inv(se3.A))
        nt.assert_almost_equal(r4i.A(5.0), np.linalg.inv(SE3.Rx(5.0).A))

    def test_with_qlim(self):
        r1 = rtb.ET.Rx(2.5, qlim=(1, -1))
        r2 = rtb.ET.Rx(2.5, qlim=[1, -1])
        r3 = rtb.ET.Rx(2.5, qlim=np.array([1, -1]))

        nt.assert_almost_equal(r1.qlim, np.array([1, -1]))
        nt.assert_almost_equal(r2.qlim, np.array([1, -1]))
        nt.assert_almost_equal(r3.qlim, np.array([1, -1]))

    def test_update_qlim(self):
        r1 = rtb.ET.Rx(2.5, qlim=(1, -1))
        r2 = rtb.ET.Rx(2.5, qlim=[1, -1])
        r3 = rtb.ET.Rx(2.5, qlim=np.array([1, -1]))

        r1.qlim = (-2, 2)
        r2.qlim = [-2, 2]
        r3.qlim = np.array([-2, 2])

        nt.assert_almost_equal(r1.qlim, np.array([-2, 2]))
        nt.assert_almost_equal(r2.qlim, np.array([-2, 2]))
        nt.assert_almost_equal(r3.qlim, np.array([-2, 2]))

    def test_jindex_error(self):
        r1 = rtb.ET.Rx(2.5)

        with self.assertRaises(ValueError):
            r1.jindex = -2

    def test_param_setter_updates_transform(self):
        # ETS.merge() reassigns `.param` on an already-constructed ET (to
        # combine two adjacent static transforms). The compiled fast path
        # (`.A()` -> ET_T) and the qlim/jindex the C struct also carries
        # must reflect the new value, not the value at construction time.
        r1 = rtb.ET.tx(1.0)
        nt.assert_almost_equal(r1.A(), sm.transl(1.0, 0, 0))

        r1.param = 3.0
        self.assertEqual(r1.param, 3.0)
        nt.assert_almost_equal(r1.A(), sm.transl(3.0, 0, 0))

        # deepcopy must rebuild its own compiled struct from the updated
        # state, not the stale one from construction
        r2 = deepcopy(r1)
        nt.assert_almost_equal(r2.A(), sm.transl(3.0, 0, 0))

    def test_et2_no_compiled_accel(self):
        # ET2 is pure Python and must never build/hold a compiled
        # acceleration handle: calling the C fast path (ET_T) directly on
        # an ET2's data is undefined behaviour (it assumes a 4x4 SE(3)
        # buffer, but ET2 stores 3x3 SE(2) matrices). Asserting `.fknm`
        # doesn't exist keeps this structurally impossible rather than
        # relying on nothing ever calling the fast path by accident.
        e = rtb.ET2.tx(1.0)

        self.assertFalse(hasattr(e, "fknm"))
        self.assertFalse(hasattr(e, "_ET__fknm"))

        # param/qlim/jindex updates on ET2 must not attempt to touch a
        # compiled struct that doesn't exist
        e.param = 2.0
        nt.assert_almost_equal(e.A(), sm.transl2(2.0, 0))
        e.qlim = (-1, 1)
        e.jindex = 0

    def test_et_has_compiled_accel(self):
        # Counterpart to test_et2_no_compiled_accel: ET (3D) does build a
        # compiled struct, and it survives deepcopy as a distinct object
        # (see also test_copy).
        r1 = rtb.ET.Rx(1.0)
        self.assertIsNotNone(r1.fknm)

    def test_et2_T(self):
        fl = 1.543
        rx = rtb.ET2.R()
        tx = rtb.ET2.tx()
        ty = rtb.ET2.ty()
        se = rtb.ET2.SE2(sm.trot2(fl) @ sm.transl2(fl, 0))
        tyf = rtb.ET2.ty(flip=True)

        nt.assert_array_almost_equal(rx.A(fl), sm.trot2(fl))
        nt.assert_array_almost_equal(tx.A(fl), sm.transl2(fl, 0))
        nt.assert_array_almost_equal(ty.A(fl), sm.transl2(0, fl))
        nt.assert_array_almost_equal(se.A(), sm.trot2(fl) @ sm.transl2(fl, 0))
        nt.assert_array_almost_equal(tyf.A(fl), sm.transl2(0, -fl))

    def test_kind(self):
        self.assertEqual(rtb.ET.Rx(1.0).kind, "Rx")
        self.assertEqual(rtb.ET.Ry(1.0).kind, "Ry")
        self.assertEqual(rtb.ET.Rz(1.0).kind, "Rz")
        self.assertEqual(rtb.ET.tx(1.0).kind, "tx")
        self.assertEqual(rtb.ET.ty(1.0).kind, "ty")
        self.assertEqual(rtb.ET.tz(1.0).kind, "tz")
        self.assertEqual(rtb.ET.SE3(SE3.Rx(0.5)).kind, "SE3")

        self.assertEqual(rtb.ET2.R(1.0).kind, "R")
        self.assertEqual(rtb.ET2.tx(1.0).kind, "tx")
        self.assertEqual(rtb.ET2.ty(1.0).kind, "ty")
        self.assertEqual(rtb.ET2.SE2(sm.trot2(0.5)).kind, "SE2")

    def test_axis_deprecated(self):
        # .axis is a permanent deprecated alias for .kind (never repurposed
        # for the x/y/z meaning - see .ax below)
        e = rtb.ET.Rx(1.0)

        with self.assertWarns(DeprecationWarning):
            axis = e.axis

        self.assertEqual(axis, e.kind)
        self.assertEqual(axis, "Rx")

    def test_ax(self):
        self.assertEqual(rtb.ET.Rx(1.0).ax, "x")
        self.assertEqual(rtb.ET.Ry(1.0).ax, "y")
        self.assertEqual(rtb.ET.Rz(1.0).ax, "z")
        self.assertEqual(rtb.ET.tx(1.0).ax, "x")
        self.assertEqual(rtb.ET.ty(1.0).ax, "y")
        self.assertEqual(rtb.ET.tz(1.0).ax, "z")
        self.assertIsNone(rtb.ET.SE3(SE3.Rx(0.5)).ax)

        self.assertIsNone(rtb.ET2.R(1.0).ax)
        self.assertEqual(rtb.ET2.tx(1.0).ax, "x")
        self.assertEqual(rtb.ET2.ty(1.0).ax, "y")
        self.assertIsNone(rtb.ET2.SE2(sm.trot2(0.5)).ax)

    def test_eta_property_deprecated(self):
        # .eta is a permanent deprecated alias for .param - both getter and
        # setter must warn and behave identically to .param
        e = rtb.ET.tx(1.0)

        with self.assertWarns(DeprecationWarning):
            value = e.eta

        self.assertEqual(value, e.param)
        self.assertEqual(value, 1.0)

        with self.assertWarns(DeprecationWarning):
            e.eta = 2.0

        self.assertEqual(e.param, 2.0)
        nt.assert_almost_equal(e.A(), sm.transl(2.0, 0, 0))

    def test_eta_kwarg_deprecated(self):
        # eta= is a permanent deprecated alias for param= on every factory
        # classmethod and on BaseET.__init__ directly
        with self.assertWarns(DeprecationWarning):
            e = rtb.ET.tx(eta=1.5)

        self.assertEqual(e.param, 1.5)
        nt.assert_almost_equal(e.A(), sm.transl(1.5, 0, 0))

        with self.assertWarns(DeprecationWarning):
            e2 = BaseET("tx", eta=1.5, axis_func=lambda x: sm.transl(x, 0, 0))

        self.assertEqual(e2.param, 1.5)

    def test_joint_descriptor_string(self):
        cases = [
            ("theta2", 2, False),
            ("q2", 2, False),
            ("-q(3)", 3, True),
            ("θ_3", 3, False),
        ]
        for s, jindex, flip in cases:
            e = rtb.ET.Rx(s)
            self.assertTrue(e.isjoint)
            self.assertEqual(e.jindex, jindex, s)
            self.assertEqual(e.isflip, flip, s)
            self.assertEqual(str(e), f"Rx({s})")

        # ET2 gets the same treatment, no special-casing needed
        e2 = rtb.ET2.R("-q(4)")
        self.assertEqual(e2.jindex, 4)
        self.assertTrue(e2.isflip)

    def test_joint_descriptor_kinematics(self):
        # the parsed descriptor must behave exactly like a normal joint
        e = rtb.ET.Rx("-q(3)")
        nt.assert_almost_equal(e.A(0.5), sm.trotx(-0.5))

        e2 = rtb.ET.Rx("theta2")
        nt.assert_almost_equal(e2.A(0.5), sm.trotx(0.5))

    def test_joint_descriptor_no_digit_falls_back_to_auto_numbering(self):
        e = rtb.ET.Rx("theta")
        self.assertTrue(e.isjoint)
        self.assertIsNone(e.jindex)
        self.assertFalse(e.isflip)

    def test_joint_descriptor_numeric_string_is_static_value(self):
        # a string that parses as a plain number is a static value, not a
        # joint descriptor
        e = rtb.ET.tx("1.5")
        self.assertFalse(e.isjoint)
        self.assertEqual(e.param, 1.5)
        nt.assert_almost_equal(e.A(), sm.transl(1.5, 0, 0))

    def test_joint_descriptor_conflict_raises(self):
        with self.assertRaises(ValueError):
            rtb.ET.Rx("theta2", jindex=5)

        with self.assertRaises(ValueError):
            rtb.ET.Rx("theta2", flip=True)

    def test_free_functions(self):
        # roboticstoolbox.ets.ET/.ET2 expose Rx/Ry/Rz/tx/ty/tz/SE3 and
        # R/tx/ty/SE2 respectively as bare module-level functions (not just
        # ET.Rx/ET2.tx classmethods), so `from roboticstoolbox.ets.ET import *`
        # works. tx/ty deliberately mean different things (3D vs 2D) between
        # the two modules - that's the one thing wildcard-importing both at
        # once can't avoid.
        from roboticstoolbox.ets import ET as ET_module
        from roboticstoolbox.ets import ET2 as ET2_module

        # Rx/tx/etc are bound classmethods, so a fresh attribute access
        # (ET_module.Rx vs rtb.ET.Rx) produces a distinct-but-equal bound
        # method object each time - compare with == (same __func__/__self__),
        # not `is`.
        self.assertEqual(ET_module.Rx, rtb.ET.Rx)
        self.assertEqual(ET_module.tx, rtb.ET.tx)
        self.assertEqual(ET_module.SE3, rtb.ET.SE3)

        self.assertEqual(ET2_module.R, rtb.ET2.R)
        self.assertEqual(ET2_module.tx, rtb.ET2.tx)
        self.assertEqual(ET2_module.SE2, rtb.ET2.SE2)

        nt.assert_almost_equal(
            ET_module.tx(1.5).A(), rtb.ET.tx(1.5).A()
        )
        nt.assert_almost_equal(
            ET2_module.tx(1.5).A(), rtb.ET2.tx(1.5).A()
        )
        # confirm they really are different (3D vs 2D), not the same object
        self.assertNotEqual(ET_module.tx(1.5).A().shape, ET2_module.tx(1.5).A().shape)

    def test_sum(self):
        # __add__ is an alias for __mul__ (composition) on ET/ET2, and
        # __radd__ (treating a start value of 0 as identity) is what lets
        # sum() work without an explicit start
        e1 = rtb.ET.Rz(jindex=0)
        e2 = rtb.ET.tx(1)
        e3 = rtb.ET.Rz(jindex=1)
        expected = e1 * e2 * e3

        r_add = e1 + e2 + e3
        self.assertIsInstance(r_add, rtb.ETS)
        self.assertEqual(r_add, expected)

        r_sum = sum([e1, e2, e3])
        self.assertIsInstance(r_sum, rtb.ETS)
        self.assertEqual(r_sum, expected)

        f1 = rtb.ET2.R(jindex=0)
        f2 = rtb.ET2.tx(1)
        f3 = rtb.ET2.R(jindex=1)
        expected2 = f1 * f2 * f3

        s_add = f1 + f2 + f3
        self.assertIsInstance(s_add, rtb.ETS2)
        self.assertEqual(s_add, expected2)

        s_sum = sum([f1, f2, f3])
        self.assertIsInstance(s_sum, rtb.ETS2)
        self.assertEqual(s_sum, expected2)

        # BaseETS.__radd__ makes sum() work on a list of ETS/ETS2 too, not
        # just their individual elements
        ets_sum = sum([e1 * e2, e3])
        self.assertIsInstance(ets_sum, rtb.ETS)
        self.assertEqual(ets_sum, expected)

        ets2_sum = sum([f1 * f2, f3])
        self.assertIsInstance(ets2_sum, rtb.ETS2)
        self.assertEqual(ets2_sum, expected2)

        # a genuinely bad start value still fails loudly rather than being
        # silently swallowed
        with self.assertRaises(TypeError):
            sum([e1, e2, e3], 5)


if __name__ == "__main__":
    unittest.main()
