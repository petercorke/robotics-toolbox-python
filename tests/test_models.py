#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import roboticstoolbox as rp
import unittest
import numpy.testing as nt


class TestModels(unittest.TestCase):
    def test_list(self):
        rp.models.list()
        rp.models.list("UR", 6)
        rp.models.list(type="DH")

    def test_puma(self):
        puma = rp.models.DH.Puma560()
        puma.qr
        puma.qz
        puma.qn
        puma = rp.models.DH.Puma560(symbolic=True)

    def test_pumaURDF(self):
        puma = rp.models.Puma560()
        puma.qr
        puma.qz

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

    def test_Mico(self):
        r = rp.models.Mico()
        r.qr
        r.qz

    def test_ball(self):
        r = rp.models.DH.Ball()
        r.qz

    def test_stanford(self):
        r = rp.models.DH.Stanford()
        r.qz

    def test_planar3(self):
        r = rp.models.DH.Planar3()
        r.qz

    def test_planar2(self):
        r = rp.models.DH.Planar2()
        r.qz

    def test_orion5(self):
        r = rp.models.DH.Orion5()
        r.qz

    def test_lwr4(self):
        r = rp.models.DH.LWR4()
        r.qz

    def test_kr5(self):
        r = rp.models.DH.KR5()
        r.qz

    def test_irb140(self):
        r = rp.models.DH.IRB140()
        r.qz

    def test_cobra600(self):
        r = rp.models.DH.Cobra600()
        r.qz

    def test_pr2(self):
        rp.models.PR2()


if __name__ == "__main__":  # pragma nocover
    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])
