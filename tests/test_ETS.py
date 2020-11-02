#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import roboticstoolbox as rp
import spatialmath.base as sm
import unittest


class TestETS(unittest.TestCase):

    # def test_fail(self):
    #     self.assertRaises(ValueError, rp.ETS.rx)

    def test_TRx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ETS.rx(fl).T(), sm.trotx(fl))
        nt.assert_array_almost_equal(rp.ETS.rx(-fl).T(), sm.trotx(-fl))
        nt.assert_array_almost_equal(rp.ETS.rx(0).T(), sm.trotx(0))

    def test_TRy(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ETS.ry(fl).T(), sm.troty(fl))
        nt.assert_array_almost_equal(rp.ETS.ry(-fl).T(), sm.troty(-fl))
        nt.assert_array_almost_equal(rp.ETS.ry(0).T(), sm.troty(0))

    def test_TRz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ETS.rz(fl).T(), sm.trotz(fl))
        nt.assert_array_almost_equal(rp.ETS.rz(-fl).T(), sm.trotz(-fl))
        nt.assert_array_almost_equal(rp.ETS.rz(0).T(), sm.trotz(0))

    def test_Ttx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ETS.tx(fl).T(), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(
            rp.ETS.tx(-fl).T(), sm.transl(-fl, 0, 0))
        nt.assert_array_almost_equal(rp.ETS.tx(0).T(), sm.transl(0, 0, 0))

    def test_Tty(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ETS.ty(fl).T(), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(
            rp.ETS.ty(-fl).T(), sm.transl(0, -fl, 0))
        nt.assert_array_almost_equal(rp.ETS.ty(0).T(), sm.transl(0, 0, 0))

    def test_Ttz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ETS.tz(fl).T(), sm.transl(0, 0, fl))
        nt.assert_array_almost_equal(
            rp.ETS.tz(-fl).T(), sm.transl(0, 0, -fl))
        nt.assert_array_almost_equal(rp.ETS.tz(0).T(), sm.transl(0, 0, 0))

    def test_str(self):
        rx = rp.ETS.rx(1.543)
        ry = rp.ETS.ry(1.543)
        rz = rp.ETS.rz(1.543)
        tx = rp.ETS.tx(1.543)
        ty = rp.ETS.ty(1.543)
        tz = rp.ETS.tz(1.543)

        self.assertEqual(str(rx), 'Rx(88.41°)')
        self.assertEqual(str(ry), 'Ry(88.41°)')
        self.assertEqual(str(rz), 'Rz(88.41°)')
        self.assertEqual(str(tx), 'tx(1.543)')
        self.assertEqual(str(ty), 'ty(1.543)')
        self.assertEqual(str(tz), 'tz(1.543)')
        self.assertEqual(str(rx), repr(rx))
        self.assertEqual(str(ry), repr(ry))
        self.assertEqual(str(rz), repr(rz))
        self.assertEqual(str(tx), repr(tx))
        self.assertEqual(str(ty), repr(ty))
        self.assertEqual(str(tz), repr(tz))

    def test_str_q(self):
        rx = rp.ETS.rx()
        ry = rp.ETS.ry()
        rz = rp.ETS.rz()
        tx = rp.ETS.tx()
        ty = rp.ETS.ty()
        tz = rp.ETS.tz()

        self.assertEqual(str(rx), 'Rx(q)')
        self.assertEqual(str(ry), 'Ry(q)')
        self.assertEqual(str(rz), 'Rz(q)')
        self.assertEqual(str(tx), 'tx(q)')
        self.assertEqual(str(ty), 'ty(q)')
        self.assertEqual(str(tz), 'tz(q)')
        self.assertEqual(str(rx), repr(rx))
        self.assertEqual(str(ry), repr(ry))
        self.assertEqual(str(rz), repr(rz))
        self.assertEqual(str(tx), repr(tx))
        self.assertEqual(str(ty), repr(ty))
        self.assertEqual(str(tz), repr(tz))

    def test_T_real(self):
        fl = 1.543
        rx = rp.ETS.rx(fl)
        ry = rp.ETS.ry(fl)
        rz = rp.ETS.rz(fl)
        tx = rp.ETS.tx(fl)
        ty = rp.ETS.ty(fl)
        tz = rp.ETS.tz(fl)

        nt.assert_array_almost_equal(rx.T(), sm.trotx(fl))
        nt.assert_array_almost_equal(ry.T(), sm.troty(fl))
        nt.assert_array_almost_equal(rz.T(), sm.trotz(fl))
        nt.assert_array_almost_equal(tx.T(), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.T(), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.T(), sm.transl(0, 0, fl))

    def test_T_real_2(self):
        fl = 1.543
        rx = rp.ETS.rx()
        ry = rp.ETS.ry()
        rz = rp.ETS.rz()
        tx = rp.ETS.tx()
        ty = rp.ETS.ty()
        tz = rp.ETS.tz()

        nt.assert_array_almost_equal(rx.T(fl), sm.trotx(fl))
        nt.assert_array_almost_equal(ry.T(fl), sm.troty(fl))
        nt.assert_array_almost_equal(rz.T(fl), sm.trotz(fl))
        nt.assert_array_almost_equal(tx.T(fl), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.T(fl), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.T(fl), sm.transl(0, 0, fl))

    def test_ets(self):
        ets = rp.ETS.rx(1) * rp.ETS.tx(2)

        nt.assert_array_almost_equal(ets[0].T(), sm.trotx(1))
        nt.assert_array_almost_equal(ets[1].T(), sm.transl(2, 0, 0))

    def test_ets_var(self):
        ets = rp.ETS.rx() * rp.ETS.tx()

        nt.assert_array_almost_equal(ets[0].T(1), sm.trotx(1))
        nt.assert_array_almost_equal(ets[1].T(2), sm.transl(2, 0, 0))


if __name__ == '__main__':

    unittest.main()
