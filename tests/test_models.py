#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import ropy as rp
import unittest


class TestModels(unittest.TestCase):

    def test_puma(self):
        puma = rp.Puma560()
        puma.qr
        puma.qz
        puma.qs
        puma.qn

    def test_frankie(self):
        frankie = rp.Frankie()
        frankie.qr
        frankie.qz

    def test_PandaURDF(self):
        panda = rp.PandaURDF()
        panda.qr
        panda.qz

    def test_UR5(self):
        ur = rp.UR5()
        ur.qr
        ur.qz

    def test_wx250s(self):
        wx = rp.wx250s()
        wx.qr
        wx.qz
