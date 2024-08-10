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
from roboticstoolbox.robot.ET import BaseET
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

        arx = "ET.Rx(eta=1.543, jindex=5, flip=True, qlim=array([-1.,  1.]))"
        atx = "ET.tx(eta=1.543, jindex=5, flip=True, qlim=array([-1.,  1.]))"
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
            BaseET("Rx", eta=0.5)

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


if __name__ == "__main__":

    unittest.main()
