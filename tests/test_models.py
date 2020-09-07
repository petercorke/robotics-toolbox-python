#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import roboticstoolbox as rp
import unittest


class TestModels(unittest.TestCase):

    def test_puma(self):
        puma = rp.models.DH.Puma560()
        puma.qr
        puma.qz
        puma.qs
        puma.qn

    def test_pumaURDF(self):
        puma = rp.models.Puma560()
        puma.qr
        puma.qz
        puma.qs
        puma.qn

    def test_frankie(self):
        frankie = rp.models.ETS.Frankie()
        frankie.qr
        frankie.qz

    def test_PandaURDF(self):
        panda = rp.models.Panda()
        panda.qr
        panda.qz

    def test_UR3(self):
        ur = rp.models.UR3()
        ur.qr
        ur.qz

    def test_UR5(self):
        ur = rp.models.UR5()
        ur.qr
        ur.qz

    def test_UR10(self):
        ur = rp.models.UR10()
        ur.qr
        ur.qz

    def test_px100(self):
        r = rp.models.px100()
        r.qr
        r.qz

    def test_px150(self):
        r = rp.models.px150()
        r.qr
        r.qz

    def test_rx150(self):
        r = rp.models.rx150()
        r.qr
        r.qz

    def test_rx200(self):
        r = rp.models.rx200()
        r.qr
        r.qz

    def test_vx300(self):
        r = rp.models.vx300()
        r.qr
        r.qz

    def test_vx300s(self):
        r = rp.models.vx300s()
        r.qr
        r.qz

    def test_wx200(self):
        r = rp.models.wx200()
        r.qr
        r.qz

    def test_wx250(self):
        r = rp.models.wx250()
        r.qr
        r.qz

    def test_wx250s(self):
        r = rp.models.wx250s()
        r.qr
        r.qz

    def test_j2n4s300(self):
        r = rp.models.j2n4s300()
        r.qr
        r.qz
