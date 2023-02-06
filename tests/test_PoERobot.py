"""
@author: Daniel Huczala
"""

import unittest
import numpy.testing as nt
from roboticstoolbox import Robot, PoERobot, PoERevolute, PoEPrismatic
from spatialmath import SE3


class TestPoERobot(unittest.TestCase):
    def test_ets(self):
        link1 = PoERevolute([0, 0, 1], [0, 0, 0])
        link2 = PoERevolute([0, 1, 0], [0, 0, 0.2])
        link3 = PoEPrismatic([0, 1, 0])
        link4 = PoERevolute([0, -1, 0], [0.2, 0, 0.5])
        TE0 = SE3.Rand()

        r = PoERobot([link1, link2, link3, link4], TE0)
        r_poe2ets = Robot(r.ets())

        q = [-1.3, 0, 2.5, -1.7]
        nt.assert_almost_equal(r.fkine(q).A, r_poe2ets.fkine(q).A)


if __name__ == '__main__':
    unittest.main()
