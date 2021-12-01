#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import roboticstoolbox as rtb
import spatialmath.base as sm
import unittest


class TestET(unittest.TestCase):
    def test_TRx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.Rx(fl).T(), sm.trotx(fl))
        nt.assert_array_almost_equal(rtb.ET.Rx(-fl).T(), sm.trotx(-fl))
        nt.assert_array_almost_equal(rtb.ET.Rx(0).T(), sm.trotx(0))

    def test_TRy(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.Ry(fl).T(), sm.troty(fl))
        nt.assert_array_almost_equal(rtb.ET.Ry(-fl).T(), sm.troty(-fl))
        nt.assert_array_almost_equal(rtb.ET.Ry(0).T(), sm.troty(0))

    def test_TRz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.Rz(fl).T(), sm.trotz(fl))
        nt.assert_array_almost_equal(rtb.ET.Rz(-fl).T(), sm.trotz(-fl))
        nt.assert_array_almost_equal(rtb.ET.Rz(0).T(), sm.trotz(0))

    def test_Ttx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.tx(fl).T(), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(rtb.ET.tx(-fl).T(), sm.transl(-fl, 0, 0))
        nt.assert_array_almost_equal(rtb.ET.tx(0.0).T(), sm.transl(0, 0, 0))

    def test_Tty(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.ty(fl).T(), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(rtb.ET.ty(-fl).T(), sm.transl(0, -fl, 0))
        nt.assert_array_almost_equal(rtb.ET.ty(0).T(), sm.transl(0, 0, 0))

    def test_Ttz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rtb.ET.tz(fl).T(), sm.transl(0, 0, fl))
        nt.assert_array_almost_equal(rtb.ET.tz(-fl).T(), sm.transl(0, 0, -fl))
        nt.assert_array_almost_equal(rtb.ET.tz(0).T(), sm.transl(0, 0, 0))

    def test_str(self):
        rx = rtb.ET.Rx(1.543)
        ry = rtb.ET.Ry(1.543)
        rz = rtb.ET.Rz(1.543)
        tx = rtb.ET.tx(1.543)
        ty = rtb.ET.ty(1.543)
        tz = rtb.ET.tz(1.543)

        self.assertEqual(str(rx), "Rx(1.543)")
        self.assertEqual(str(ry), "Ry(1.543)")
        self.assertEqual(str(rz), "Rz(1.543)")
        self.assertEqual(str(tx), "tx(1.543)")
        self.assertEqual(str(ty), "ty(1.543)")
        self.assertEqual(str(tz), "tz(1.543)")

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

        nt.assert_array_almost_equal(rx.T(), sm.trotx(fl))
        nt.assert_array_almost_equal(ry.T(), sm.troty(fl))
        nt.assert_array_almost_equal(rz.T(), sm.trotz(fl))
        nt.assert_array_almost_equal(tx.T(), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.T(), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.T(), sm.transl(0, 0, fl))

    def test_T_real_2(self):
        fl = 1.543
        rx = rtb.ET.Rx()
        ry = rtb.ET.Ry()
        rz = rtb.ET.Rz()
        tx = rtb.ET.tx()
        ty = rtb.ET.ty()
        tz = rtb.ET.tz()

        nt.assert_array_almost_equal(rx.T(fl), sm.trotx(fl))
        nt.assert_array_almost_equal(ry.T(fl), sm.troty(fl))
        nt.assert_array_almost_equal(rz.T(fl), sm.trotz(fl))
        nt.assert_array_almost_equal(tx.T(fl), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.T(fl), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.T(fl), sm.transl(0, 0, fl))

    def test_ets(self):
        ets = rtb.ETS.rx(1) * rtb.ETS.tx(2)

        nt.assert_array_almost_equal(ets[0].T(), sm.trotx(1))
        nt.assert_array_almost_equal(ets[1].T(), sm.transl(2, 0, 0))

    def test_ets_var(self):
        ets = rtb.ETS.rx() * rtb.ETS.tx()

        nt.assert_array_almost_equal(ets[0].T(1), sm.trotx(1))
        nt.assert_array_almost_equal(ets[1].T(2), sm.transl(2, 0, 0))


if __name__ == "__main__":

    unittest.main()
