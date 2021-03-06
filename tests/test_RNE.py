#!/usr/bin/env python3
import numpy.testing as nt
import numpy as np
import roboticstoolbox as rp
import unittest
import spatialmath as sm

class TestRNE(unittest.TestCase):

    def test_ur5(self):
        ur5 = rp.models.URDF.UR5()
        qZero = np.zeros((6,))
        tau = ur5.rne(qZero, qZero, qZero)

        # test against KDL::ChainIdSolver_RNE with the offical ur_description package 
        self.assertAlmostEqual(tau[0], -1.72777e-26, 3)
        self.assertAlmostEqual(tau[1], -57.9684, 3)
        self.assertAlmostEqual(tau[2], -14.4815, 3)
        self.assertAlmostEqual(tau[3], -1.27931e-11, 3)
        self.assertAlmostEqual(tau[4], 1.17427e-12, 3)
        self.assertAlmostEqual(tau[5], 0, 3)

if __name__ == '__main__':

    unittest.main()
