#!/usr/bin/env python3

import numpy as np
import scipy.interpolate
import math

from roboticstoolbox.blocks.arm import *
from spatialmath import SE3, SO3
import roboticstoolbox as rtb

import unittest
import numpy.testing as nt


class ArmTest(unittest.TestCase):
    def setup(self):
        # allow assertEqual for SE3
        self.addTypeEqualityFunc(SE3, lambda x, y: x == y)

    def test_fk(self):
        robot = rtb.models.DH.Puma560()

        block = Forward_Kinematics(robot)
        self.assertEqual(block._eval(robot.qn)[0], robot.fkine(robot.qn))

    def test_ik(self):
        robot = rtb.models.DH.Puma560()
        T = robot.fkine(robot.qn)

        block = Inverse_Kinematics(robot)
        q = block._eval(T)[0]
        self.assertEqual(robot.fkine(q), T)

    def test_rne(self):
        robot = rtb.models.DH.Puma560()
        block = Inverse_Dynamics(robot)
        tau = block._eval(robot.qn, robot.qz, robot.qz)[0]
        nt.assert_array_equal(tau, robot.rne(robot.qn, robot.qz, robot.qz))

    def test_jacobian(self):
        robot = rtb.models.DH.Puma560()
        q = robot.qn

        block = Jacobian(robot)
        nt.assert_array_equal(block._eval(q)[0], robot.jacob0(robot.qn))
        block = Jacobian(robot, inverse=True)
        nt.assert_array_equal(block._eval(q)[0], np.linalg.inv(robot.jacob0(robot.qn)))
        block = Jacobian(robot, transpose=True)
        nt.assert_array_equal(block._eval(q)[0], robot.jacob0(robot.qn).T)
        block = Jacobian(robot, inverse=True, transpose=True)
        nt.assert_array_almost_equal(
            block._eval(q)[0], np.linalg.inv(robot.jacob0(robot.qn).T)
        )

        block = Jacobian(robot, frame="e")
        nt.assert_array_equal(block._eval(q)[0], robot.jacobe(robot.qn))
        block = Jacobian(robot, frame="e", inverse=True)
        nt.assert_array_equal(block._eval(q)[0], np.linalg.inv(robot.jacobe(robot.qn)))
        block = Jacobian(robot, frame="e", transpose=True)
        nt.assert_array_equal(block._eval(q)[0], robot.jacobe(robot.qn).T)
        block = Jacobian(robot, frame="e", inverse=True, transpose=True)
        nt.assert_array_almost_equal(
            block._eval(q)[0], np.linalg.inv(robot.jacobe(robot.qn).T)
        )

    def test_tr2delta(self):

        T1 = SE3(1, 2, 3)
        T2 = SE3(1.1, 2.2, 3.3) * SE3.RPY(0.1, 0.2, 0.3)

        block = Tr2Delta()
        nt.assert_array_almost_equal(block._eval(T1, T2)[0], base.tr2delta(T1.A, T2.A))

    def test_delta2tr(self):

        delta = np.r_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] / 100

        block = Delta2Tr()
        out = block._eval(delta)[0]
        self.assertIsInstance(out, SE3)
        self.assertEqual(out, SE3.Delta(delta))

    def test_tpoly(self):

        block = Traj(0, 1, 10, traj="tpoly")
        block.start()
        # initial point
        nt.assert_array_almost_equal(block._eval(0)[0], 0)
        nt.assert_array_almost_equal(block._eval(0)[1], 0)
        nt.assert_array_almost_equal(block._eval(0)[2], 0)

        # final point
        nt.assert_array_almost_equal(block._eval(10)[0], 1)
        nt.assert_array_almost_equal(block._eval(10)[1], 0)
        nt.assert_array_almost_equal(block._eval(10)[2], 0)

        block = Traj([2, 3], [5, 6], 10, traj="tpoly")
        block.start()
        # initial point
        nt.assert_array_almost_equal(block._eval(0)[0], [2, 3])
        nt.assert_array_almost_equal(block._eval(0)[1], [0, 0])
        nt.assert_array_almost_equal(block._eval(0)[2], [0, 0])

        # final point
        nt.assert_array_almost_equal(block._eval(10)[0], [5, 6])
        nt.assert_array_almost_equal(block._eval(10)[1], [0, 0])
        nt.assert_array_almost_equal(block._eval(10)[2], [0, 0])

    def test_lspb(self):

        block = Traj(0, 1, 10, traj="lspb")
        block.start()
        # initial point
        nt.assert_array_almost_equal(block._eval(0)[0], 0)
        nt.assert_array_almost_equal(block._eval(0)[1], 0)

        # final point
        nt.assert_array_almost_equal(block._eval(10)[0], 1)
        nt.assert_array_almost_equal(block._eval(10)[1], 0)

        block = Traj([2, 3], [5, 6], 10, traj="lspb")
        block.start()
        # initial point
        nt.assert_array_almost_equal(block._eval(0)[0], [2, 3])
        nt.assert_array_almost_equal(block._eval(0)[1], [0, 0])

        # final point
        nt.assert_array_almost_equal(block._eval(10)[0], [5, 6])
        nt.assert_array_almost_equal(block._eval(10)[1], [0, 0])

    def test_circle(self):
        block = CirclePath(radius=1, centre=(0, 0, 1))
        nt.assert_array_almost_equal(block._eval(t=0)[0], [1, 0, 1])
        nt.assert_array_almost_equal(block._eval(t=0.25)[0], [0, 1, 1])
        nt.assert_array_almost_equal(block._eval(t=0.5)[0], [-1, 0, 1])

        block = CirclePath(radius=1, pose=SE3.Rx(pi / 2))
        nt.assert_array_almost_equal(block._eval(t=0)[0], [1, 0, 0])
        nt.assert_array_almost_equal(block._eval(t=0.25)[0], [0, 0, 1])
        nt.assert_array_almost_equal(block._eval(t=0.5)[0], [-1, 0, 0])

    def test_point2tr(self):
        T = SE3.Rx(pi / 2)
        block = Point2Tr(T)
        t = [1, 2, 3]

        out = block._eval(t)[0]
        self.assertIsInstance(out, SE3)
        nt.assert_array_almost_equal(out.R, T.R)
        nt.assert_array_almost_equal(out.t, t)


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
