#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 3 17:17:04 2021
@author: Peter Corke
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
from spatialmath.base import tr2x, numjac, numhess
from scipy.linalg import block_diag
import unittest
import pytest


class Tests:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.robot = rtb.models.ETS.Puma560()
        self.q = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
        self.qd = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])
        # self.robot = rtb.models.ETS.Panda()
        # self.q = self.robot.qr
        # self.qd = self.robot.qr

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
        nt.assert_array_almost_equal(
            self.robot.jacob0_analytical(q, representation=rep), Ja
        )

    def test_jacob_analytical_rpy_xyz(self):
        rep = "rpy/xyz"
        q = self.q
        Ja = numjac(lambda q: tr2x(self.robot.fkine(q).A, representation=rep), q)
        nt.assert_array_almost_equal(
            self.robot.jacob0_analytical(q, representation=rep), Ja
        )

    def test_jacob_analytical_rpy_zyx(self):
        rep = "rpy/zyx"
        q = self.q
        Ja = numjac(lambda q: tr2x(self.robot.fkine(q).A, representation=rep), q)
        nt.assert_array_almost_equal(
            self.robot.jacob0_analytical(q, representation=rep), Ja
        )

    def test_jacob_analytical_exp(self):
        rep = "exp"
        q = self.q
        Ja = numjac(lambda q: tr2x(self.robot.fkine(q).A, representation=rep), q)
        nt.assert_array_almost_equal(
            self.robot.jacob0_analytical(q, representation=rep), Ja
        )

    def test_jacob_dot(self):
        j0 = self.robot.jacob0_dot(self.q, self.qd)

        H = numhess(lambda q: self.robot.jacob0(q), self.q)
        Jd = np.zeros((6, self.robot.n))
        for i in range(self.robot.n):
            Jd += H[i, :, :] * self.qd[i]

        nt.assert_array_almost_equal(j0, Jd, decimal=4)

    def test_jacob_dot_analytical_eul(self):
        rep = "eul"
        j0 = self.robot.jacob0_dot(self.q, self.qd, representation=rep)

        H = numhess(
            lambda q: self.robot.jacob0_analytical(q, representation=rep), self.q
        )
        Jd = np.zeros((6, self.robot.n))
        for i in range(self.robot.n):
            Jd += H[i, :, :] * self.qd[i]

        print(np.round(j0, 2))
        print()
        print(np.round(Jd, 2))
        nt.assert_array_almost_equal(j0, Jd, decimal=4)

    # ------ This section tests various ets' with flipped joints ------ #

    def test_jacob0_flipped0(self):
        robot = rtb.ETS(
            [
                rtb.ET.Rz(jindex=0, qlim=np.array([-2.9668, 2.9668])),
                rtb.ET.SE3(
                    T=np.array(
                        [
                            [1.0000e00, 0.0000e00, 0.0000e00, -4.3624e-04],
                            [0.0000e00, 1.0000e00, 0.0000e00, 0.0000e00],
                            [0.0000e00, 0.0000e00, 1.0000e00, 3.6000e-01],
                            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                        ]
                    )
                ),
                rtb.ET.Ry(jindex=1, qlim=np.array([-2.0942, 2.0942])),
                rtb.ET.Rz(jindex=2, qlim=np.array([-2.9668, 2.9668])),
                rtb.ET.SE3(
                    T=np.array(
                        [
                            [1.0000e00, 0.0000e00, 0.0000e00, 4.3624e-04],
                            [0.0000e00, 1.0000e00, 0.0000e00, 0.0000e00],
                            [0.0000e00, 0.0000e00, 1.0000e00, 4.2000e-01],
                            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                        ]
                    )
                ),
                rtb.ET.Ry(jindex=3, flip=True, qlim=np.array([-2.0942, 2.0942])),
                rtb.ET.Rz(jindex=4, qlim=np.array([-2.9668, 2.9668])),
                rtb.ET.tz(0.4),
                rtb.ET.Ry(jindex=5, qlim=np.array([-2.0942, 2.0942])),
                rtb.ET.Rz(jindex=6, qlim=np.array([-3.0541, 3.0541])),
                rtb.ET.tz(0.126),
            ]
        )

        q = np.array([0, -0.3, 0, -2.2, 0, 2, 0.79])

        nt.assert_array_almost_equal(
            robot.jacob0(q), numjac(lambda q: robot.fkine(q).A, q, SE=3)
        )

    def test_jacob0_flipped1(self):
        ets_list = [
            rtb.ET.Rx(flip=True)
            * rtb.ET.Ry()
            * rtb.ET.Rz()
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry(flip=True)
            * rtb.ET.Rz()
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz(flip=True)
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz()
            * rtb.ET.tx(flip=True)
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz()
            * rtb.ET.tx()
            * rtb.ET.ty(flip=True)
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz()
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(flip=True),
            rtb.ET.Rx(flip=True)
            * rtb.ET.Ry()
            * rtb.ET.Rz(flip=True)
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz(flip=True)
            * rtb.ET.tx()
            * rtb.ET.ty(flip=True)
            * rtb.ET.tz(),
            rtb.ET.Rx(flip=True)
            * rtb.ET.Ry(flip=True)
            * rtb.ET.Rz(flip=True)
            * rtb.ET.tx(flip=True)
            * rtb.ET.ty(flip=True)
            * rtb.ET.tz(flip=True),
        ]

        for ets in ets_list:
            q = np.array([-0.3, 0, -2.2, 0, 2, 0.79])

            nt.assert_array_almost_equal(
                ets.jacob0(q), numjac(lambda q: ets.fkine(q).A, q, SE=3)
            )

    def test_jacobe_flipped0(self):
        robot = rtb.ETS(
            [
                rtb.ET.Rz(jindex=0, qlim=np.array([-2.9668, 2.9668])),
                rtb.ET.SE3(
                    T=np.array(
                        [
                            [1.0000e00, 0.0000e00, 0.0000e00, -4.3624e-04],
                            [0.0000e00, 1.0000e00, 0.0000e00, 0.0000e00],
                            [0.0000e00, 0.0000e00, 1.0000e00, 3.6000e-01],
                            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                        ]
                    )
                ),
                rtb.ET.Ry(jindex=1, qlim=np.array([-2.0942, 2.0942])),
                rtb.ET.Rz(jindex=2, qlim=np.array([-2.9668, 2.9668])),
                rtb.ET.SE3(
                    T=np.array(
                        [
                            [1.0000e00, 0.0000e00, 0.0000e00, 4.3624e-04],
                            [0.0000e00, 1.0000e00, 0.0000e00, 0.0000e00],
                            [0.0000e00, 0.0000e00, 1.0000e00, 4.2000e-01],
                            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                        ]
                    )
                ),
                rtb.ET.Ry(jindex=3, flip=True, qlim=np.array([-2.0942, 2.0942])),
                rtb.ET.Rz(jindex=4, qlim=np.array([-2.9668, 2.9668])),
                rtb.ET.tz(0.4),
                rtb.ET.Ry(jindex=5, qlim=np.array([-2.0942, 2.0942])),
                rtb.ET.Rz(jindex=6, qlim=np.array([-3.0541, 3.0541])),
                rtb.ET.tz(0.126),
            ]
        )

        q = np.array([0, -0.3, 0, -2.2, 0, 2, 0.79])

        J0 = numjac(lambda q: robot.fkine(q).A, q, SE=3)
        TE = robot.fkine(q)
        Je = block_diag(TE.R.T, TE.R.T) @ J0

        nt.assert_array_almost_equal(robot.jacobe(q), Je)

    def test_jacobe_flipped1(self):
        ets_list = [
            rtb.ET.Rx(flip=True)
            * rtb.ET.Ry()
            * rtb.ET.Rz()
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry(flip=True)
            * rtb.ET.Rz()
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz(flip=True)
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz()
            * rtb.ET.tx(flip=True)
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz()
            * rtb.ET.tx()
            * rtb.ET.ty(flip=True)
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz()
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(flip=True),
            rtb.ET.Rx(flip=True)
            * rtb.ET.Ry()
            * rtb.ET.Rz(flip=True)
            * rtb.ET.tx()
            * rtb.ET.ty()
            * rtb.ET.tz(),
            rtb.ET.Rx()
            * rtb.ET.Ry()
            * rtb.ET.Rz(flip=True)
            * rtb.ET.tx()
            * rtb.ET.ty(flip=True)
            * rtb.ET.tz(),
            rtb.ET.Rx(flip=True)
            * rtb.ET.Ry(flip=True)
            * rtb.ET.Rz(flip=True)
            * rtb.ET.tx(flip=True)
            * rtb.ET.ty(flip=True)
            * rtb.ET.tz(flip=True),
        ]

        for ets in ets_list:
            q = np.array([-0.3, 0, -2.2, 0, 2, 0.79])

            J0 = numjac(lambda q: ets.fkine(q).A, q, SE=3)
            TE = ets.fkine(q)
            Je = block_diag(TE.R.T, TE.R.T) @ J0

            nt.assert_array_almost_equal(ets.jacobe(q), Je)

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


# class TestJacobians_ERobot(unittest.TestCase, Tests):
#     def setUp(self):
#         self.robot = rtb.models.ETS.Puma560()
#         self.q = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
#         self.qd = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])


if __name__ == "__main__":
    unittest.main()
