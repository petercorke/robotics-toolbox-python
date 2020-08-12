#!/usr/bin/env python
"""
@author (Original) Matthew Matl, Github: mmatl
@author (Adapted by) Jesse Haviland
"""

import numpy as np
import unittest

from ropy.backend import URDF, Link, Joint, Transmission, Material


class TestURDF(unittest.TestCase):

    def test_urdfpy(self):

        # Load
        u = URDF.load('ropy/backend/urdf/tests/data/ur5.urdf')

        self.assertIsInstance(u, URDF)
        for j in u.joints:
            self.assertIsInstance(j, Joint)
        for ln in u.links:
            self.assertIsInstance(ln, Link)
        for t in u.transmissions:
            self.assertIsInstance(t, Transmission)
        for m in u.materials:
            self.assertIsInstance(m, Material)

        # # Test fk
        # fk = u.link_fk()
        # self.assertIsInstance(fk, dict)
        # for ln in fk:
        #     self.assertIsInstance(ln, Link)
        #     self.assertIsInstance(fk[ln], np.ndarray)
        #     assert fk[ln].shape == (4, 4)

        # fk = u.link_fk({'shoulder_pan_joint': 2.0})
        # self.assertIsInstance(fk, dict)
        # for ln in fk:
        #     self.assertIsInstance(ln, Link)
        #     self.assertIsInstance(fk[ln], np.ndarray)
        #     assert fk[ln].shape == (4, 4)

        # fk = u.link_fk(np.zeros(6))
        # self.assertIsInstance(fk, dict)
        # for ln in fk:
        #     self.assertIsInstance(ln, Link)
        #     self.assertIsInstance(fk[ln], np.ndarray)
        #     assert fk[ln].shape == (4, 4)

        # fk = u.link_fk(np.zeros(6), link='upper_arm_link')
        # self.assertIsInstance(fk, np.ndarray)
        # assert fk.shape == (4, 4)

        # fk = u.link_fk(links=['shoulder_link', 'upper_arm_link'])
        # self.assertIsInstance(fk, dict)
        # assert len(fk) == 2
        # for ln in fk:
        #     self.assertIsInstance(ln, Link)
        #     self.assertIsInstance(fk[ln], np.ndarray)
        #     assert fk[ln].shape == (4, 4)

        # fk = u.link_fk(links=list(u.links)[:2])
        # self.assertIsInstance(fk, dict)
        # assert len(fk) == 2
        # for ln in fk:
        #     self.assertIsInstance(ln, Link)
        #     self.assertIsInstance(fk[ln], np.ndarray)
        #     assert fk[ln].shape == (4, 4)

        # cfg = {j.name: np.random.uniform(size=1000) for j in u.actuated_joints}
        # fk = u.link_fk_batch(cfgs=cfg)
        # for key in fk:
        #     self.assertIsInstance(fk[key], np.ndarray)
        #     assert fk[key].shape == (1000, 4, 4)

        # # Test join
        # with self.assertRaises(ValueError):
        #     x = u.join(u, link=u.link_map['tool0'])
        # x = u.join(u, link=u.link_map['tool0'], name='copy', prefix='prefix')
        # self.assertIsInstance(x, URDF)
        # assert x.name == 'copy'
        # assert len(x.joints) == 2 * len(u.joints) + 1
        # assert len(x.links) == 2 * len(u.links)

        # Test scale
        x = u.copy(scale=3)
        self.assertIsInstance(x, URDF)
        x = x.copy(scale=[1, 1, 3])
        self.assertIsInstance(x, URDF)
