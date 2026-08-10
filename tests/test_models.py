#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import roboticstoolbox as rp
import unittest
import numpy.testing as nt


# @unittest.skip("BUG in .models()")
class TestModels(unittest.TestCase):
    def test_catalog(self):
        rp.models.catalog()
        rp.models.catalog("UR", 6)
        rp.models.catalog(mtype="DH")

    def test_list_deprecated(self):
        with self.assertWarns(FutureWarning):
            rp.models.list(mtype="DH")

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

    def test_Jaco(self):
        r = rp.models.Jaco()
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


class TestModelSmoke(unittest.TestCase):
    """Generic smoke test: every model exported from DH/URDF/ETS must
    construct with no arguments.

    The tests above are hand-written per model, so a model can be added to
    __all__ and never get a dedicated test -- Valkyrie, Fetch, KinovaGen3,
    FetchCamera and LBR all sat with zero test coverage this way, three of
    them silently broken). Iterating
    __all__ directly means newly-added models are covered automatically.
    """

    # (category, class name) pairs known to currently fail to construct.
    # Remove an entry once its underlying issue is actually fixed -- if you
    # don't, this test starts failing for the *opposite* reason (a listed
    # failure unexpectedly started passing).
    EXPECTED_FAILURES = set()

    def test_all_models_construct(self):
        unexpected_failures = []
        unexpected_passes = []

        for category_name in ("DH", "URDF", "ETS"):
            category = getattr(rp.models, category_name)
            for name in category.__all__:
                cls = getattr(category, name)
                key = (category_name, name)
                try:
                    cls()
                except Exception as e:
                    if key not in self.EXPECTED_FAILURES:
                        unexpected_failures.append(
                            f"{category_name}.{name}: {type(e).__name__}: {e}"
                        )
                else:
                    if key in self.EXPECTED_FAILURES:
                        unexpected_passes.append(f"{category_name}.{name}")

        if unexpected_failures:
            self.fail(
                "Model(s) failed to construct:\n" + "\n".join(unexpected_failures)
            )
        if unexpected_passes:
            self.fail(
                "Model(s) in EXPECTED_FAILURES now construct successfully -- "
                "remove from EXPECTED_FAILURES (and close out the matching "
                "tracking issue):\n" + "\n".join(unexpected_passes)
            )


if __name__ == "__main__":  # pragma nocover
    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])
