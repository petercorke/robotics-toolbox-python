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

    def test_jacob_dot_precomputed_j0(self):
        J0 = self.robot.jacob0(self.q)
        j0_direct = self.robot.jacob0_dot(self.q, self.qd)
        j0_precomputed = self.robot.jacob0_dot(None, self.qd, J0=J0)
        nt.assert_array_almost_equal(j0_direct, j0_precomputed)

    def test_jacob0_dot_planar2_handworked_and_sympy(self):
        """
        Verify jacob0, hessian0, and jacob0_dot against both handworked
        analytical ground truth and independently derived SymPy symbolic equations
        for a 2-link planar RR manipulator.
        """
        import sympy as sp

        robot = rtb.models.DH.Planar2()

        # Symbolic derivation from end-effector forward kinematics in {0}
        q1_s, q2_s, qd1_s, qd2_s = sp.symbols("q1 q2 qd1 qd2")
        a1_s, a2_s = 1.0, 1.0
        px_s = a1_s * sp.cos(q1_s) + a2_s * sp.cos(q1_s + q2_s)
        py_s = a1_s * sp.sin(q1_s) + a2_s * sp.sin(q1_s + q2_s)

        # Geometric Jacobian matrix (6x2): linear velocity (px, py, pz=0) and angular velocity (wz)
        J_sym = sp.Matrix(
            [
                [sp.diff(px_s, q1_s), sp.diff(px_s, q2_s)],
                [sp.diff(py_s, q1_s), sp.diff(py_s, q2_s)],
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ]
        )

        # Symbolic Hessian tensor slices dJ/dq1 and dJ/dq2
        H_sym_0 = sp.Matrix(
            [[sp.diff(J_sym[r, c], q1_s) for c in range(2)] for r in range(6)]
        )
        H_sym_1 = sp.Matrix(
            [[sp.diff(J_sym[r, c], q2_s) for c in range(2)] for r in range(6)]
        )

        # Symbolic time derivative of Jacobian: dJ/dt = (dJ/dq1)*qd1 + (dJ/dq2)*qd2
        Jdot_sym = H_sym_0 * qd1_s + H_sym_1 * qd2_s

        test_cases = [
            (np.array([0.0, 0.0]), np.array([1.0, 0.5])),
            (np.array([np.pi / 4, -np.pi / 3]), np.array([-0.5, 1.2])),
            (np.array([0.3, 0.7]), np.array([1.5, -2.0])),
            (np.array([-1.2, 2.1]), np.array([0.8, -0.4])),
            (np.array([np.pi / 2, np.pi / 2]), np.array([0.0, 2.5])),
        ]

        a1 = 1.0
        a2 = 1.0

        for q, qd in test_cases:
            q1, q2 = q
            qd1, qd2 = qd
            s1, c1 = np.sin(q1), np.cos(q1)
            s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)

            # --- Handworked Analytical Equations ---
            # Geometric Jacobian J0 (6x2)
            J_hand = np.zeros((6, 2))
            J_hand[0, 0] = -a1 * s1 - a2 * s12
            J_hand[0, 1] = -a2 * s12
            J_hand[1, 0] = a1 * c1 + a2 * c12
            J_hand[1, 1] = a2 * c12
            J_hand[5, 0] = 1.0
            J_hand[5, 1] = 1.0

            # Geometric Hessian H0 (2x6x2)
            H_hand = np.zeros((2, 6, 2))
            # dJ/dq1
            H_hand[0, 0, 0] = -a1 * c1 - a2 * c12
            H_hand[0, 0, 1] = -a2 * c12
            H_hand[0, 1, 0] = -a1 * s1 - a2 * s12
            H_hand[0, 1, 1] = -a2 * s12
            # dJ/dq2
            H_hand[1, 0, 0] = -a2 * c12
            H_hand[1, 0, 1] = -a2 * c12
            H_hand[1, 1, 0] = -a2 * s12
            H_hand[1, 1, 1] = -a2 * s12

            # Time derivative of Jacobian: j0_dot (6x2)
            j0_dot_hand = np.zeros((6, 2))
            j0_dot_hand[0, 0] = -a1 * c1 * qd1 - a2 * c12 * (qd1 + qd2)
            j0_dot_hand[0, 1] = -a2 * c12 * (qd1 + qd2)
            j0_dot_hand[1, 0] = -a1 * s1 * qd1 - a2 * s12 * (qd1 + qd2)
            j0_dot_hand[1, 1] = -a2 * s12 * (qd1 + qd2)

            # --- SymPy Numerical Evaluation ---
            subs = {q1_s: q1, q2_s: q2, qd1_s: qd1, qd2_s: qd2}
            J_sp = np.array(J_sym.evalf(subs=subs), dtype=float)
            H_sp = np.stack(
                [
                    np.array(H_sym_0.evalf(subs=subs), dtype=float),
                    np.array(H_sym_1.evalf(subs=subs), dtype=float),
                ],
                axis=0,
            )
            j0_dot_sp = np.array(Jdot_sym.evalf(subs=subs), dtype=float)

            # --- RTB Computed Quantities ---
            J_rtb = robot.jacob0(q)
            H_rtb = robot.hessian0(q)
            j0_dot_rtb = robot.jacob0_dot(q, qd)
            j0_dot_rtb_precomputed = robot.jacob0_dot(None, qd, J0=J_rtb)

            # --- Ground-Truth Assertions ---
            # 1. Jacobian matches handworked & sympy
            nt.assert_allclose(J_rtb, J_hand, atol=1e-12)
            nt.assert_allclose(J_rtb, J_sp, atol=1e-12)

            # 2. Hessian matches handworked & sympy
            nt.assert_allclose(H_rtb, H_hand, atol=1e-12)
            nt.assert_allclose(H_rtb, H_sp, atol=1e-12)

            # 3. Time derivative of Jacobian matches handworked & sympy
            nt.assert_allclose(j0_dot_rtb, j0_dot_hand, atol=1e-12)
            nt.assert_allclose(j0_dot_rtb, j0_dot_sp, atol=1e-12)

            # 4. Precomputed J0 path gives identical result
            nt.assert_allclose(j0_dot_rtb_precomputed, j0_dot_hand, atol=1e-12)

    def test_jacob_dot_analytical_all_representations(self):
        for rep in ["rpy/xyz", "rpy/zyx", "eul", "exp"]:
            j0 = self.robot.jacob0_dot(self.q, self.qd, representation=rep)

            H = numhess(
                lambda q, rep=rep: self.robot.jacob0_analytical(q, representation=rep),
                self.q,
            )
            Jd = np.zeros((6, self.robot.n))
            for i in range(self.robot.n):
                Jd += H[i, :, :] * self.qd[i]

            nt.assert_array_almost_equal(j0, Jd, decimal=4)

    def test_jacob_dot_representation_q_none_raises(self):
        J0 = self.robot.jacob0(self.q)
        with pytest.raises(
            ValueError, match="q must be provided when representation is specified"
        ):
            self.robot.jacob0_dot(None, self.qd, J0=J0, representation="eul")

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
