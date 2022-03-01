#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 3 17:17:04 2021
@author: Peter Corke
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
from spatialmath.base import tr2x, numjac
from scipy.linalg import block_diag
import unittest


class Tests:
    def test_jacob0(self):
        q = self.q
        nt.assert_array_almost_equal(
            self.robot.jacob0(q), numjac(lambda q: self.robot.fkine(q).A, q, SE=3)
        )

    def test_jacobe(self):
        q = self.q
        J0 = numjac(lambda q: self.robot.fkine(q).A, q, SE=3)
        # velocity transform to EE frame
        TE = self.robot.fkine(q)
        Je = block_diag(TE.R.T, TE.R.T) @ J0
        nt.assert_array_almost_equal(self.robot.jacobe(q), Je)

    def test_jacob_analytical_eul(self):
        rep = "eul"
        q = self.q
        Ja = numjac(lambda q: tr2x(self.robot.fkine(q).A, representation=rep), q)
        nt.assert_array_almost_equal(self.robot.jacob0(q, analytical=rep), Ja)

    def test_jacob_analytical_rpy_xyz(self):
        rep = "rpy/xyz"
        q = self.q
        Ja = numjac(lambda q: tr2x(self.robot.fkine(q).A, representation=rep), q)
        nt.assert_array_almost_equal(self.robot.jacob0(q, analytical=rep), Ja)

    def test_jacob_analytical_rpy_zyx(self):
        rep = "rpy/zyx"
        q = self.q
        Ja = numjac(lambda q: tr2x(self.robot.fkine(q).A, representation=rep), q)
        nt.assert_array_almost_equal(self.robot.jacob0(q, analytical=rep), Ja)

    def test_jacob_analytical_exp(self):
        rep = "exp"
        q = self.q
        Ja = numjac(lambda q: tr2x(self.robot.fkine(q).A, representation=rep), q)
        nt.assert_array_almost_equal(self.robot.jacob0(q, analytical=rep), Ja)

    def test_jacob_dot(self):
        j0 = self.robot.jacob_dot(self.q, self.qd)

        H = numhess(lambda q: self.robot.jacob0(q), self.q)
        Jd = np.zeros((6, self.robot.n))
        for i in range(self.robot.n):
            Jd += H[:, :, i] * self.qd[i]
        nt.assert_array_almost_equal(j0, Jd, decimal=4)

    def test_jacob_dot_analytical_eul(self):
        rep = "eul"
        j0 = self.robot.jacob_dot(self.q, self.qd, analytical=rep)

        H = numhess(lambda q: self.robot.jacob0(q, analytical=rep), self.q)
        Jd = np.zeros((6, self.robot.n))
        for i in range(self.robot.n):
            Jd += H[:, :, i] * self.qd[i]
        print(j0)
        print()
        print(Jd)
        nt.assert_array_almost_equal(j0, Jd, decimal=4)

    # def test_jacob_dot_analytical_rpy_xyz(self):
    #     rep = 'rpy/xyz'
    #     j0 = self.robot.jacob_dot(self.q, self.qd, analytical=rep)

    #     H = numhess(lambda q: self.robot.jacob0(q, analytical=rep), self.q)
    #     Jd = np.zeros((6, self.robot.n))
    #     for i in range(self.robot.n):
    #         Jd += H[:, :, i] * self.qd[i]
    #     nt.assert_array_almost_equal(j0, Jd, decimal=4)

    # def test_jacob_dot_analytical_rpy_zyx(self):
    #     rep = 'rpy/zyx'
    #     j0 = self.robot.jacob_dot(self.q, self.qd, analytical=rep)

    #     H = numhess(lambda q: self.robot.jacob0(q, analytical=rep), self.q)
    #     Jd = np.zeros((6, self.robot.n))
    #     for i in range(self.robot.n):
    #         Jd += H[:, :, i] * self.qd[i]
    #     nt.assert_array_almost_equal(j0, Jd, decimal=4)

    # def test_jacob_dot_analytical_exp(self):
    #     rep = 'exp'
    #     j0 = self.robot.jacob_dot(self.q, self.qd, analytical=rep)

    #     H = numhess(lambda q: self.robot.jacob0(q, analytical=rep), self.q)
    #     Jd = np.zeros((6, self.robot.n))
    #     for i in range(self.robot.n):
    #         Jd += H[:, :, i] * self.qd[i]
    #     nt.assert_array_almost_equal(j0, Jd, decimal=4)


# class TestJacobians_DH(unittest.TestCase, Tests):

#     def setUp(self):
#         self.robot = rtb.models.DH.Puma560()
#         self.q = np.r_[0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
#         self.qd = np.r_[0.1, -0.2, 0.3, -0.1, 0.2, -0.3]


class TestJacobians_ERobot(unittest.TestCase, Tests):
    def setUp(self):
        self.robot = rtb.models.ETS.Puma560()
        self.q = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
        self.qd = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])


if __name__ == "__main__":

    unittest.main()
