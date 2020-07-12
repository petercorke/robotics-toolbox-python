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

    def test_fail(self):
        self.assertRaises(ValueError, rp.ET.TRx)

    def test_TRx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.TRx(fl).T(), sm.trotx(fl))
        nt.assert_array_almost_equal(rp.ET.TRx(-fl).T(), sm.trotx(-fl))
        nt.assert_array_almost_equal(rp.ET.TRx(0).T(), sm.trotx(0))

    def test_TRy(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.TRy(fl).T(), sm.troty(fl))
        nt.assert_array_almost_equal(rp.ET.TRy(-fl).T(), sm.troty(-fl))
        nt.assert_array_almost_equal(rp.ET.TRy(0).T(), sm.troty(0))

    def test_TRz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.TRz(fl).T(), sm.trotz(fl))
        nt.assert_array_almost_equal(rp.ET.TRz(-fl).T(), sm.trotz(-fl))
        nt.assert_array_almost_equal(rp.ET.TRz(0).T(), sm.trotz(0))

    def test_Ttx(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.Ttx(fl).T(), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(rp.ET.Ttx(-fl).T(), sm.transl(-fl, 0, 0))
        nt.assert_array_almost_equal(rp.ET.Ttx(0).T(), sm.transl(0, 0, 0))

    def test_Tty(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.Tty(fl).T(), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(rp.ET.Tty(-fl).T(), sm.transl(0, -fl, 0))
        nt.assert_array_almost_equal(rp.ET.Tty(0).T(), sm.transl(0, 0, 0))

    def test_Ttz(self):
        fl = 1.543

        nt.assert_array_almost_equal(rp.ET.Ttz(fl).T(), sm.transl(0, 0, fl))
        nt.assert_array_almost_equal(rp.ET.Ttz(-fl).T(), sm.transl(0, 0, -fl))
        nt.assert_array_almost_equal(rp.ET.Ttz(0).T(), sm.transl(0, 0, 0))

    def test_str(self):
        rx = rp.ET.TRx(1.543)
        ry = rp.ET.TRy(1.543)
        rz = rp.ET.TRz(1.543)
        tx = rp.ET.Ttx(1.543)
        ty = rp.ET.Tty(1.543)
        tz = rp.ET.Ttz(1.543)

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
        rx = rp.ET.TRx(joint=86)
        ry = rp.ET.TRy(joint=86)
        rz = rp.ET.TRz(joint=86)
        tx = rp.ET.Ttx(joint=86)
        ty = rp.ET.Tty(joint=86)
        tz = rp.ET.Ttz(joint=86)

        self.assertEqual(str(rx), 'Rx(q86)')
        self.assertEqual(str(ry), 'Ry(q86)')
        self.assertEqual(str(rz), 'Rz(q86)')
        self.assertEqual(str(tx), 'tx(q86)')
        self.assertEqual(str(ty), 'ty(q86)')
        self.assertEqual(str(tz), 'tz(q86)')
        self.assertEqual(str(rx), repr(rx))
        self.assertEqual(str(ry), repr(ry))
        self.assertEqual(str(rz), repr(rz))
        self.assertEqual(str(tx), repr(tx))
        self.assertEqual(str(ty), repr(ty))
        self.assertEqual(str(tz), repr(tz))

    def test_T_real(self):
        fl = 1.543
        rx = rp.ET.TRx(fl)
        ry = rp.ET.TRy(fl)
        rz = rp.ET.TRz(fl)
        tx = rp.ET.Ttx(fl)
        ty = rp.ET.Tty(fl)
        tz = rp.ET.Ttz(fl)

        nt.assert_array_almost_equal(rx.T(), sm.trotx(fl))
        nt.assert_array_almost_equal(ry.T(), sm.troty(fl))
        nt.assert_array_almost_equal(rz.T(), sm.trotz(fl))
        nt.assert_array_almost_equal(tx.T(), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.T(), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.T(), sm.transl(0, 0, fl))

    def test_T_real(self):
        fl = 1.543
        rx = rp.ET.TRx(joint=86)
        ry = rp.ET.TRy(joint=86)
        rz = rp.ET.TRz(joint=86)
        tx = rp.ET.Ttx(joint=86)
        ty = rp.ET.Tty(joint=86)
        tz = rp.ET.Ttz(joint=86)

        nt.assert_array_almost_equal(rx.T(fl), sm.trotx(fl))
        nt.assert_array_almost_equal(ry.T(fl), sm.troty(fl))
        nt.assert_array_almost_equal(rz.T(fl), sm.trotz(fl))
        nt.assert_array_almost_equal(tx.T(fl), sm.transl(fl, 0, 0))
        nt.assert_array_almost_equal(ty.T(fl), sm.transl(0, fl, 0))
        nt.assert_array_almost_equal(tz.T(fl), sm.transl(0, 0, fl))


if __name__ == '__main__':

    unittest.main()
