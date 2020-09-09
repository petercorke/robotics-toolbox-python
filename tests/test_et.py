#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rp
import spatialmath.base as sm
import unittest


class TestET(unittest.TestCase):

    # def test_fail(self):
    #     self.assertRaises(ValueError, rp.ET.rx)

    def test_TRx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.rx(fl).T().A, sm.trotx(fl))
        nt.assert_array_almost_equal(rp.ET.rx(-fl).T().A, sm.trotx(-fl))
        nt.assert_array_almost_equal(rp.ET.rx(0).T().A, sm.trotx(0))

    def test_TRy(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.ry(fl).T().A, sm.troty(fl))
        nt.assert_array_almost_equal(rp.ET.ry(-fl).T().A, sm.troty(-fl))
        nt.assert_array_almost_equal(rp.ET.ry(0).T().A, sm.troty(0))

    def test_TRz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.rz(fl).T().A, sm.trotz(fl))
        nt.assert_array_almost_equal(rp.ET.rz(-fl).T().A, sm.trotz(-fl))
        nt.assert_array_almost_equal(rp.ET.rz(0).T().A, sm.trotz(0))

    def test_Ttx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.tx(fl).T().A, sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(rp.ET.tx(-fl).T().A, sm.transl(-fl, 0, 0))
        nt.assert_array_almost_equal(rp.ET.tx(0).T().A, sm.transl(0, 0, 0))

    def test_Tty(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.ty(fl).T().A, sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(rp.ET.ty(-fl).T().A, sm.transl(0, -fl, 0))
        nt.assert_array_almost_equal(rp.ET.ty(0).T().A, sm.transl(0, 0, 0))

    def test_Ttz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.tz(fl).T().A, sm.transl(0, 0, fl))
        nt.assert_array_almost_equal(rp.ET.tz(-fl).T().A, sm.transl(0, 0, -fl))
        nt.assert_array_almost_equal(rp.ET.tz(0).T().A, sm.transl(0, 0, 0))

    def test_str(self):
        rx = rp.ET.rx(1.543)
        ry = rp.ET.ry(1.543)
        rz = rp.ET.rz(1.543)
        tx = rp.ET.tx(1.543)
        ty = rp.ET.ty(1.543)
        tz = rp.ET.tz(1.543)

        self.assertEqual(str(rx), 'Rx(88.4074)')
        self.assertEqual(str(ry), 'Ry(88.4074)')
        self.assertEqual(str(rz), 'Rz(88.4074)')
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
        rx = rp.ET.rx()
        ry = rp.ET.ry()
        rz = rp.ET.rz()
        tx = rp.ET.tx()
        ty = rp.ET.ty()
        tz = rp.ET.tz()

        self.assertEqual(str(rx), 'Rx(q0)')
        self.assertEqual(str(ry), 'Ry(q0)')
        self.assertEqual(str(rz), 'Rz(q0)')
        self.assertEqual(str(tx), 'tx(q0)')
        self.assertEqual(str(ty), 'ty(q0)')
        self.assertEqual(str(tz), 'tz(q0)')
        self.assertEqual(str(rx), repr(rx))
        self.assertEqual(str(ry), repr(ry))
        self.assertEqual(str(rz), repr(rz))
        self.assertEqual(str(tx), repr(tx))
        self.assertEqual(str(ty), repr(ty))
        self.assertEqual(str(tz), repr(tz))

    def test_T_real(self):
        fl = 1.543
        rx = rp.ET.rx(fl)
        ry = rp.ET.ry(fl)
        rz = rp.ET.rz(fl)
        tx = rp.ET.tx(fl)
        ty = rp.ET.ty(fl)
        tz = rp.ET.tz(fl)

        nt.assert_array_almost_equal(rx.T().A, sm.trotx(fl))
        nt.assert_array_almost_equal(ry.T().A, sm.troty(fl))
        nt.assert_array_almost_equal(rz.T().A, sm.trotz(fl))
        nt.assert_array_almost_equal(tx.T().A, sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.T().A, sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.T().A, sm.transl(0, 0, fl))

    def test_T_real_2(self):
        fl = 1.543
        rx = rp.ET.rx()
        ry = rp.ET.ry()
        rz = rp.ET.rz()
        tx = rp.ET.tx()
        ty = rp.ET.ty()
        tz = rp.ET.tz()

        nt.assert_array_almost_equal(rx.T(fl).A, sm.trotx(fl))
        nt.assert_array_almost_equal(ry.T(fl).A, sm.troty(fl))
        nt.assert_array_almost_equal(rz.T(fl).A, sm.trotz(fl))
        nt.assert_array_almost_equal(tx.T(fl).A, sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.T(fl).A, sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.T(fl).A, sm.transl(0, 0, fl))

    def test_ets(self):
        ets = rp.ET.rx(1) * rp.ET.tx(2)

        nt.assert_array_almost_equal(ets[0].T().A, sm.trotx(1))
        nt.assert_array_almost_equal(ets[1].T().A, sm.transl(2, 0, 0))

    def test_ets_var(self):
        ets = rp.ET.rx() * rp.ET.tx()

        nt.assert_array_almost_equal(ets[0].T(1).A, sm.trotx(1))
        nt.assert_array_almost_equal(ets[1].T(2).A, sm.transl(2, 0, 0))


if __name__ == '__main__':

    unittest.main()
