"""
@author: Daniel Huczala
"""

import unittest
import numpy.testing as nt
import numpy as np
from roboticstoolbox import Robot, PoERobot, PoERevolute, PoEPrismatic
from spatialmath import SE3
from spatialmath.base import trnorm


class TestPoERobot(unittest.TestCase):
    def test_poe2ets_conversion(self):
        # 2RPR standard structure
        link1 = PoERevolute([0, 0, 1], [0, 0, 0])
        link2 = PoERevolute([0, 1, 0], [0, 0, 0.2])
        link3 = PoEPrismatic([0, 1, 0])
        link4 = PoERevolute([0, -1, 0], [0.2, 0, 0.5])
        tool = SE3(
            np.array([[1, 0, 0, 0.3], [0, 0, -1, 0], [0, 1, 0, 0.5], [0, 0, 0, 1]])
        )

        r = PoERobot([link1, link2, link3, link4], tool)
        r_as_ets = Robot(r.ets())

        q = [-1.3, 0, 2.5, -1.7]

        # test fkine
        nt.assert_almost_equal(r.fkine(q).A, r_as_ets.fkine(q).A)

        # test jacobians
        nt.assert_almost_equal(r.jacob0(q), r_as_ets.jacob0(q))
        nt.assert_almost_equal(r.jacobe(q), r_as_ets.jacobe(q))

        #########################
        # 3RP arbitrary structure
        link1a = PoERevolute([0, 0, 1], [0, 0, 0])

        w = [-0.635, 0.495, 0.592]
        w = w / np.linalg.norm(w)
        p = [-0.152, -0.023, -0.144]
        link2a = PoERevolute(w, p)

        w = [-0.280, 0.790, 0.544]
        w = w / np.linalg.norm(w)
        p = [-0.300, -0.003, -0.150]
        link3a = PoERevolute(w, p)

        w = [-0.280, 0.790, 0.544]
        w = w / np.linalg.norm(w)
        link4a = PoEPrismatic(w)

        toola = np.array(
            [
                [0.2535, -0.5986, 0.7599, 0.2938],
                [-0.8063, 0.3032, 0.5078, -0.0005749],
                [-0.5344, -0.7414, -0.4058, 0.08402],
                [0, 0, 0, 1],
            ]
        )
        toola = SE3(trnorm(toola))

        ra = PoERobot([link1a, link2a, link3a, link4a], toola)
        ra_as_ets = Robot(ra.ets())

        qa = [-1.3, -0.4, 2.5, -1.7]

        # test fkine
        nt.assert_almost_equal(ra.fkine(qa).A, ra_as_ets.fkine(qa).A)

        # test jacobians
        nt.assert_almost_equal(ra.jacob0(qa), ra_as_ets.jacob0(qa))
        nt.assert_almost_equal(ra.jacobe(qa), ra_as_ets.jacobe(qa))


if __name__ == "__main__":
    unittest.main()
