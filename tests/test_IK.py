"""
@author: Jesse Haviland
"""

# import numpy.testing as nt
import contextlib
import importlib.util
import io
import roboticstoolbox as rtb
import numpy as np
import unittest
import numpy.testing as nt

# import sympy
import pytest
from spatialmath import SE3
from tests import skip_no_qp

test_tol = 1e-5


class TestIK(unittest.TestCase):
    def test_IK_NR1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_IK_NR2(self):

        q0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol)

        sol = solver.solve(panda, Tep, q0=q0)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_IK_NR3(self):

        q0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol)

        sol = solver.solve(panda, Tep, q0=q0)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_IK_NR4(self):

        q0 = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, np.pi / 4],
            ]
        )

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol, slimit=5)

        sol = solver.solve(panda, Tep, q0=q0)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_IK_NR5(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(test_tol, E)

    def test_IK_NR6(self):

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=False, slimit=1)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, False)

    def test_IK_NR7(self):

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -1.3, 0, 1.2, 0, 2.0, 0.1])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, ilimit=2, slimit=1)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, False)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_IK_NR8(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(
            joint_limits=True,
            seed=0,
            pinv=True,
            tol=tol,
            kq=0.01,
            km=1.0,
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_IK_LM1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM(
            method="chan", joint_limits=True, seed=0, tol=tol, kq=0.1, km=0.1
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_exact_q0_converges_without_solver_step(self):
        panda = rtb.models.Panda().ets()
        q0 = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, np.pi / 4])
        Tep = panda.eval(q0)

        solvers = (
            rtb.IK_LM(method="chan", slimit=1),
            rtb.IK_NR(pinv=False, slimit=1),
            rtb.IK_GN(pinv=False, slimit=1),
        )

        for solver in solvers:
            with self.subTest(solver=solver.name):
                sol = solver.solve(panda, Tep, q0=q0)

                self.assertEqual(sol.success, True)
                self.assertEqual(sol.iterations, 0)
                self.assertLess(sol.residual, solver.tol)
                nt.assert_allclose(sol.q, q0, atol=1e-12)

    def test_IK_LM2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM(
            method="sugihara", k=0.0001, joint_limits=True, seed=0, tol=tol
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_IK_LM3(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM(
            method="wampler", k=0.001, joint_limits=True, seed=0, tol=tol
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_IK_GN1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_GN(
            pinv=True, joint_limits=True, seed=0, tol=tol, kq=1.0, km=1.0
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_IK_GN2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_GN(pinv=True, joint_limits=True, seed=0, tol=tol)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_IK_GN3(self):

        tol = 1e-6

        ur5 = rtb.models.UR5().ets()

        Tep = ur5.eval([0, -0.3, 0, -2.2, 0, 2.0])

        solver = rtb.IK_GN(pinv=False, joint_limits=True, seed=0, tol=tol)

        sol = solver.solve(ur5, Tep)

        self.assertEqual(sol.success, True)

        Tq = ur5.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    @skip_no_qp
    def test_IK_QP1(self):

        q0 = np.array(
            [
                -1.66441371,
                -1.20998727,
                1.04248366,
                -2.10222463,
                1.05097407,
                1.41173279,
                0.0053529,
            ]
        )

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP(joint_limits=True, seed=0, tol=tol, kq=2.0, km=100.0)

        sol = solver.solve(panda, Tep, q0=q0)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    @skip_no_qp
    def test_IK_QP2(self):

        q0 = np.array(
            [
                -1.66441371,
                -1.20998727,
                1.04248366,
                -2.10222463,
                1.05097407,
                1.41173279,
                0.0053529,
            ]
        )

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP(joint_limits=True, seed=0, tol=tol)

        sol = solver.solve(panda, Tep, q0=q0)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    @skip_no_qp
    def test_IK_QP3(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP(
            joint_limits=True,
            seed=0,
            tol=tol,
            kq=1000.0,
            pi=4.0,
            ps=2.0,
            kj=10000.0,
            slimit=1,
        )

        try:
            solver.solve(panda, Tep)
        except BaseException:
            pass

    #     self.assertEqual(sol.success, False)

    def test_ets_ikine_NR1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()
        panda2 = rtb.models.Panda()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR()

        sol = panda.ikine_NR(Tep, pinv=True, tol=tol)
        sol2 = panda2.ikine_NR(Tep, pinv=True, tol=tol)

        self.assertEqual(sol.success, True)
        self.assertEqual(sol2.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_ets_ikine_NR2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR()

        sol = panda.ikine_NR(Tep, pinv=True, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(test_tol, E)

    def test_ets_ikine_LM1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()
        panda2 = rtb.models.Panda()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM()

        sol = panda.ikine_LM(Tep, tol=tol)
        sol2 = panda2.ikine_LM(Tep, tol=tol)

        self.assertEqual(sol.success, True)
        self.assertEqual(sol2.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_ets_ikine_LM2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM()

        sol = panda.ikine_LM(Tep, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(test_tol, E)

    def test_ets_ikine_GN1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()
        panda2 = rtb.models.Panda()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_GN()

        sol = panda.ikine_GN(Tep, pinv=True, tol=tol)
        sol2 = panda2.ikine_GN(Tep, pinv=True, tol=tol)

        self.assertEqual(sol.success, True)
        self.assertEqual(sol2.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    def test_ets_ikine_GN2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_GN()

        sol = panda.ikine_GN(Tep, pinv=True, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(test_tol, E)

    @skip_no_qp
    def test_ets_ikine_QP1(self):

        q0 = np.array(
            [
                -1.66441371,
                -1.20998727,
                1.04248366,
                -2.10222463,
                1.05097407,
                1.41173279,
                0.0053529,
            ]
        )

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP()

        sol = panda.ikine_QP(Tep, tol=tol, q0=q0)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(test_tol, E)

    @skip_no_qp
    def test_ets_ikine_QP2(self):

        q0 = np.array(
            [
                -1.66441371,
                -1.20998727,
                1.04248366,
                -2.10222463,
                1.05097407,
                1.41173279,
                0.0053529,
            ]
        )

        tol = 1e-6

        panda = rtb.models.Panda().ets()
        panda2 = rtb.models.Panda()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP()

        sol = panda.ikine_QP(Tep, tol=tol, q0=q0)
        sol2 = panda2.ikine_QP(Tep, tol=tol, q0=q0)

        self.assertEqual(sol.success, True)
        self.assertEqual(sol2.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(test_tol, E)

    def test_robot_ikine_accepts_pose_arrays(self):
        # Robot.ikine_* must treat an (N, 4, 4) array like the ETS methods do
        panda = rtb.models.Panda()
        Ts = panda.fkine(np.array([panda.qr, panda.qr + 0.1, panda.qr - 0.1]))
        self.assertEqual(len(Ts), 3)
        Tarr = np.array(Ts.A)  # Ts.A is a list of matrices, make a real 3-D array
        self.assertEqual(Tarr.shape, (3, 4, 4))

        for method in ("ikine_LM", "ikine_GN", "ikine_NR"):
            sol = getattr(panda, method)(Tarr, q0=panda.qr, seed=0, joint_limits=False)
            self.assertEqual(sol.q.shape, (3, panda.n), method)
            sol_se3 = getattr(panda, method)(
                Ts, q0=panda.qr, seed=0, joint_limits=False
            )
            nt.assert_allclose(sol.q, sol_se3.q, err_msg=method)

        # a single pose as a 4x4 array still works
        sol = panda.ikine_LM(Ts[0].A, q0=panda.qr, seed=0)
        self.assertEqual(sol.q.shape, (panda.n,))

    def test_robot_ikine_rejects_badly_shaped_arrays(self):
        panda = rtb.models.Panda()
        for bad in (np.eye(3), np.zeros((2, 3, 4)), np.zeros(16)):
            with self.assertRaises(ValueError):
                panda.ikine_LM(bad, q0=panda.qr)

    def test_ik_nr(self):

        tol = 1e-6

        solver = rtb.IK_LM()

        r = rtb.models.Panda().ets()
        r2 = rtb.models.Panda()

        Tep = r.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        sol = r.ik_NR(Tep, tol=tol)
        sol2 = r2.ik_NR(Tep, tol=tol)

        self.assertTrue(sol.success)
        self.assertTrue(sol2.success)

        Tq = r.eval(sol.q)
        Tq2 = r.eval(sol2.q)

        _, E = solver.error(Tep, Tq)
        _, E2 = solver.error(Tep, Tq2)

        self.assertGreater(test_tol, E)
        self.assertGreater(test_tol, E2)

    def test_ik_lm_chan(self):

        tol = 1e-6

        solver = rtb.IK_LM()

        r = rtb.models.Panda().ets()
        r2 = rtb.models.Panda()

        Tep = r.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        sol = r.ik_LM(Tep, tol=tol, method="chan")
        sol2 = r2.ik_LM(Tep, tol=tol, method="chan")

        self.assertTrue(sol.success)
        self.assertTrue(sol2.success)

        Tq = r.eval(sol.q)
        Tq2 = r.eval(sol2.q)

        _, E = solver.error(Tep, Tq)
        _, E2 = solver.error(Tep, Tq2)

        self.assertGreater(test_tol, E)
        self.assertGreater(test_tol, E2)

    def test_ik_lm_wampler(self):

        tol = 1e-6

        solver = rtb.IK_LM()

        r = rtb.models.Panda().ets()
        r2 = rtb.models.Panda()

        Tep = r.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        sol = r.ik_LM(Tep, tol=tol, method="wampler", k=0.01)
        sol2 = r2.ik_LM(Tep, tol=tol, method="wampler", k=0.01)

        self.assertTrue(sol.success)
        self.assertTrue(sol2.success)

        Tq = r.eval(sol.q)
        Tq2 = r.eval(sol2.q)

        _, E = solver.error(Tep, Tq)
        _, E2 = solver.error(Tep, Tq2)

        self.assertGreater(test_tol, E)
        self.assertGreater(test_tol, E2)

    def test_ik_lm_sugihara(self):

        tol = 1e-6

        solver = rtb.IK_LM()

        r = rtb.models.Panda().ets()
        r2 = rtb.models.Panda()

        Tep = r.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        sol = r.ik_LM(Tep, tol=tol, k=0.01, method="sugihara")
        sol2 = r2.ik_LM(Tep, tol=tol, k=0.01, method="sugihara")

        self.assertTrue(sol.success)
        self.assertTrue(sol2.success)

        Tq = r.eval(sol.q)
        Tq2 = r.eval(sol2.q)

        _, E = solver.error(Tep, Tq)
        _, E2 = solver.error(Tep, Tq2)

        self.assertGreater(test_tol, E)
        self.assertGreater(test_tol, E2)

    def test_ik_gn(self):

        tol = 1e-6

        solver = rtb.IK_LM()

        r = rtb.models.Panda().ets()
        r2 = rtb.models.Panda()

        Tep = r.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        sol = r.ik_GN(Tep, tol=tol)
        sol2 = r2.ik_GN(Tep, tol=tol)

        self.assertTrue(sol.success)
        self.assertTrue(sol2.success)

        Tq = r.eval(sol.q)
        Tq2 = r.eval(sol2.q)

        print(sol.residual)
        print(Tep)
        print(Tq)

        _, E = solver.error(Tep, Tq)
        _, E2 = solver.error(Tep, Tq2)

        self.assertGreater(test_tol, E)
        self.assertGreater(test_tol, E2)

    def test_trajectory_residual_is_maximum(self):
        # for a trajectory, residual is the worst case over the poses, not the best
        residuals = iter([0.5, 3.0, 0.25])

        class Stub(rtb.IKSolver):
            def step(self, ets, Tep, q):
                raise NotImplementedError

            def _solve(self, ets, Tep, q0):
                return rtb.IKSolution(
                    q=np.zeros(ets.n),
                    success=True,
                    iterations=1,
                    searches=1,
                    residual=next(residuals),
                )

        panda = rtb.models.Panda().ets()
        Teps = np.array([np.eye(4)] * 3)

        sol = Stub().solve(panda, Teps)

        self.assertEqual(sol.q.shape, (3, panda.n))
        self.assertEqual(sol.residual, 3.0)
        self.assertEqual(sol.iterations, 3)
        self.assertEqual(sol.searches, 3)

    def test_trajectory_failure_is_reported(self):
        # one reachable and one unreachable pose: the failure must not be hidden
        # by the small residual of the pose that was solved
        panda = rtb.models.Panda().ets()
        good = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        bad = SE3(5, 5, 5).A

        solver = rtb.IK_LM(seed=0, slimit=5)
        sol = solver.solve(panda, SE3([SE3(good), SE3(bad)]))

        self.assertEqual(sol.q.shape, (2, panda.n))
        self.assertFalse(sol.success)
        self.assertGreater(sol.residual, solver.tol)

    def test_q0_matrix_seeds_the_first_searches(self):
        # row k of an (m, n) q0 is the starting point of search k
        panda = rtb.models.Panda().ets()
        q_true = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        Tep = panda.eval(q_true)
        q0 = np.vstack([np.zeros(panda.n), q_true])

        # one iteration per search: the first seed cannot converge, the second is exact
        solver = rtb.IK_LM(seed=0, ilimit=1, joint_limits=False)
        sol = solver.solve(panda, Tep, q0)

        self.assertTrue(sol.success)
        self.assertEqual(sol.searches, 2)
        nt.assert_allclose(sol.q, q_true, atol=1e-6)

    def test_sol_print1(self):

        sol = rtb.IKSolution(
            q=np.zeros(3),
            success=True,
            iterations=1,
            searches=2,
            residual=3.0,
            reason="no",
        )

        s = sol.__str__()

        ans = (
            "IKSolution: q=[0, 0, 0], success=True, iterations=1, searches=2,"
            " residual=3"
        )

        self.assertEqual(s, ans)

    def test_sol_print2(self):

        sol = rtb.IKSolution(
            q=None,  # type: ignore
            success=True,
            iterations=1,
            searches=2,
            residual=3.0,
            reason="no",
        )

        s = sol.__str__()

        ans = "IKSolution: q=None, success=True, iterations=1, searches=2, residual=3"

        self.assertEqual(s, ans)

    def test_sol_print3(self):

        sol = rtb.IKSolution(
            q=np.zeros(3),
            success=False,
            iterations=0,
            searches=0,
            residual=3.0,
            reason="no",
        )

        s = sol.__str__()

        ans = "IKSolution: q=[0, 0, 0], success=False, reason=no, residual=3"

        self.assertEqual(s, ans)

    def test_sol_print4(self):

        sol = rtb.IKSolution(
            q=np.zeros(3),
            success=True,
            iterations=0,
            searches=0,
            residual=3.0,
            reason="no",
        )

        s = sol.__str__()

        ans = "IKSolution: q=[0, 0, 0], success=True, residual=3"

        self.assertEqual(s, ans)

    def test_sol_print5(self):

        sol = rtb.IKSolution(
            q=np.zeros(3),
            success=False,
            iterations=1,
            searches=2,
            residual=3.0,
            reason="no",
        )

        s = sol.__str__()

        ans = (
            "IKSolution: q=[0, 0, 0], success=False, reason=no, iterations=1,"
            " searches=2, residual=3"
        )

        self.assertEqual(s, ans)

    def test_iksol_single_pose_is_one_row(self):
        q = np.array([1.0, 2.0, 3.0])
        sol = rtb.IKSolution(q, success=True)

        self.assertEqual(len(sol), 1)
        nt.assert_array_equal(sol[0], q)
        nt.assert_array_equal(sol[-1], q)
        rows = list(sol)
        self.assertEqual(len(rows), 1)
        nt.assert_array_equal(rows[0], q)
        with self.assertRaises(IndexError):
            sol[1]

    def test_iksol_trajectory_is_a_sequence_of_rows(self):
        q = np.arange(12.0).reshape(4, 3)
        sol = rtb.IKSolution(q, success=True)

        self.assertEqual(len(sol), 4)
        nt.assert_array_equal(sol[0], q[0])
        nt.assert_array_equal(sol[2], q[2])
        nt.assert_array_equal(sol[-1], q[-1])
        nt.assert_array_equal(sol[1:3], q[1:3])
        nt.assert_array_equal(sol[[0, 3]], q[[0, 3]])
        for k, row in enumerate(sol):
            nt.assert_array_equal(row, q[k])
        with self.assertRaises(IndexError):
            sol[4]

        # the sequence protocol means NumPy sees the rows
        nt.assert_array_equal(np.array(sol), q)
        nt.assert_array_equal(np.array(rtb.IKSolution(q[0], success=True)), q[:1])

    def test_iksol_without_q_is_empty(self):
        sol = rtb.IKSolution(None, success=False)  # type: ignore

        self.assertEqual(len(sol), 0)
        self.assertEqual(list(sol), [])
        with self.assertRaises(IndexError):
            sol[0]

    def test_iksol_bool_is_success(self):
        q = np.zeros((2, 3))

        self.assertTrue(rtb.IKSolution(q, success=True))
        # not the same as having rows
        self.assertFalse(rtb.IKSolution(q, success=False))
        self.assertFalse(rtb.IKSolution(None, success=False))  # type: ignore
        self.assertFalse(rtb.IKSolution(np.zeros(3), success=False))

    def test_iksol_astuple(self):
        sol = rtb.IKSolution(
            np.array([1.0, 2.0, 3.0]),
            success=True,
            iterations=10,
            searches=100,
            residual=0.1,
            reason="ok",
        )

        q, success, iterations, searches, residual, reason = sol.astuple()

        nt.assert_almost_equal(q, np.array([1.0, 2.0, 3.0]))  # type: ignore
        self.assertEqual(success, True)
        self.assertEqual(iterations, 10)
        self.assertEqual(searches, 100)
        self.assertEqual(residual, 0.1)
        self.assertEqual(reason, "ok")

    def test_iksol_from_a_solver_trajectory(self):
        panda = rtb.models.Panda().ets()
        Tep = panda.eval(np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]))
        Teps = SE3([SE3(Tep), SE3(Tep)])

        sol = rtb.IK_LM(seed=0).solve(panda, Teps)

        self.assertEqual(len(sol), 2)
        self.assertEqual(sol[0].shape, (panda.n,))
        self.assertEqual(bool(sol), sol.success)

    def test_sol_print_trajectory(self):
        q = np.arange(9.0).reshape(3, 3)
        sol = rtb.IKSolution(q, success=True, iterations=4, searches=3, residual=1e-8)

        lines = str(sol).split("\n")

        self.assertEqual(
            lines[0],
            "IKSolution: 3 poses, success=True, iterations=4, searches=3,"
            " residual=1e-08",
        )
        self.assertEqual(lines[1], "q=[[0, 1, 2],")
        self.assertEqual(lines[-1], " [6, 7, 8]]")
        self.assertEqual(repr(sol), str(sol))

    def test_sol_print_long_trajectory_is_abbreviated(self):
        q = np.arange(300.0).reshape(100, 3)
        sol = rtb.IKSolution(q, success=False, iterations=9, searches=2, reason="no")

        lines = str(sol).split("\n")

        self.assertTrue(lines[0].startswith("IKSolution: 100 poses, success=False,"))
        self.assertIn("reason=no", lines[0])
        self.assertLessEqual(len(lines), 8)
        self.assertIn("[0, 1, 2]", lines[1])  # first pose
        self.assertIn("...", str(sol))
        self.assertIn("[297, 298, 299]", lines[-1])  # last pose

    def test_sol_print_failed_residual_is_not_rounded_to_zero(self):
        sol = rtb.IKSolution(
            np.zeros(3),
            success=False,
            iterations=3,
            searches=2,
            residual=1.5e-5,
            reason="no",
        )

        self.assertIn("residual=1.5e-05", str(sol))

    def test_sol_print_analytic_without_residual(self):
        # an analytic solution that did not compute a residual has none to show
        sol = rtb.IKSolution(np.zeros(3), success=True)
        self.assertEqual(str(sol), "IKSolution: q=[0, 0, 0], success=True")

        sol = rtb.IKSolution(np.zeros(3), success=False, reason="Out of reach")
        self.assertEqual(
            str(sol), "IKSolution: q=[0, 0, 0], success=False, reason=Out of reach"
        )

    def test_repr_iksol(self):
        sol = rtb.IKSolution(np.array([1.0, 2.0, 3.0]), success=True)
        self.assertEqual(repr(sol), str(sol))

    def test_ik_LM_returns_iksolution(self):
        panda = rtb.models.Panda().ets()
        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        sol = panda.ik_LM(Tep)
        self.assertIsInstance(sol, rtb.IKSolution)
        self.assertIsInstance(sol.success, bool)

    def test_ik_NR_returns_iksolution(self):
        panda = rtb.models.Panda().ets()
        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        sol = panda.ik_NR(Tep)
        self.assertIsInstance(sol, rtb.IKSolution)
        self.assertIsInstance(sol.success, bool)

    def test_ik_GN_returns_iksolution(self):
        panda = rtb.models.Panda().ets()
        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        sol = panda.ik_GN(Tep)
        self.assertIsInstance(sol, rtb.IKSolution)
        self.assertIsInstance(sol.success, bool)

    def test_ik_lm_failure_returns_compact_q(self):
        # regression: the failure-return branch of IK.py's _solve() must
        # compact q via ets.jindices, just like the success branch does --
        # otherwise a solver on a sub-chain whose jindex doesn't start at 0
        # (e.g. YuMi's l_gripper, jindex 7-13) returns a zero-padded,
        # wrong-length q on failure instead of length ets.n.
        from spatialmath import SE3

        yumi = rtb.models.YuMi()
        ets = yumi.ets(end="l_gripper")

        Tep = SE3(0.6, -0.2, 0.3) * SE3.Rx(0.2)

        solver = rtb.IK_LM(ilimit=1, slimit=1)
        sol = solver.solve(ets, Tep)

        self.assertFalse(sol.success)
        self.assertEqual(sol.q.shape[0], ets.n)

    def test_random_q_rejects_non_finite_qlim(self):
        # _random_q() used to sample straight from ets.qlim with no check --
        # a joint with a bad (non-finite) limit baked into its model data
        # would silently produce garbage (NaN, or an opaque numpy internal
        # error) instead of a clear diagnostic. A finite joint's random_q
        # should be unaffected.
        et = rtb.ET.Rz(qlim=[-np.inf, np.inf])
        ets = rtb.ETS([et])
        solver = rtb.IK_LM()

        with self.assertRaises(ValueError):
            solver._random_q(ets, 1)

        good_et = rtb.ET.Rz(qlim=[-np.pi, np.pi])
        good_ets = rtb.ETS([good_et])
        q = solver._random_q(good_ets, 5)
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertEqual(q.shape, (5, 1))

    def test_ik_lm_c_rejects_non_finite_qlim(self):
        # Same guard, mirrored in the compiled fast-path solver (ets.ik_LM(),
        # backed by ik.cpp's own _rand_q()) -- this is a genuinely separate
        # implementation from IK_LM/_random_q() above, and used to silently
        # return a NaN "solution" (success=0) after burning through every
        # random restart, rather than raising.
        et = rtb.ET.Rz(qlim=[-np.inf, np.inf])
        ets = rtb.ETS([et])

        with self.assertRaises(ValueError):
            ets.ik_LM(np.eye(4))


class TestIKInitialCoordinates(unittest.TestCase):
    """The length of q0 is checked, by the C++ and by the Python solvers."""

    def setUp(self):
        self.robot = rtb.models.Panda()
        self.ets = self.robot.ets()
        self.Tep = self.ets.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])

    def test_cpp_wrong_length(self):
        for method in (self.ets.ik_LM, self.ets.ik_NR, self.ets.ik_GN):
            for q0 in (np.zeros(3), np.zeros(10)):
                with self.assertRaisesRegex(ValueError, "q0 has length"):
                    method(self.Tep, q0=q0)
            with self.assertRaisesRegex(ValueError, "q0 must be a vector"):
                method(self.Tep, q0=np.zeros((2, 7)))

    def test_cpp_robot_wrapper(self):
        with self.assertRaisesRegex(ValueError, "q0 has length 3"):
            self.robot.ik_LM(self.Tep, q0=np.zeros(3))

    def test_python_wrong_length(self):
        for method in (self.ets.ikine_LM, self.ets.ikine_NR, self.ets.ikine_GN):
            with self.assertRaisesRegex(ValueError, "q0 has length 3"):
                method(self.Tep, q0=np.zeros(3))
            with self.assertRaisesRegex(ValueError, "q0 has 3 columns"):
                method(self.Tep, q0=np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "rows, but slimit is"):
            self.ets.ikine_LM(self.Tep, q0=np.zeros((5, 7)), slimit=2)

    def test_mask_wrong_length(self):
        for method in (
            self.ets.ik_LM,
            self.ets.ik_NR,
            self.ets.ik_GN,
            self.ets.ikine_LM,
            self.ets.ikine_NR,
            self.ets.ikine_GN,
        ):
            with self.assertRaisesRegex(ValueError, "mask must be a vector of 6"):
                method(self.Tep, mask=np.ones(3))

    def test_right_length_still_works(self):
        q = np.array([0, -0.3, 0, -2.2, 0, 2, 0.7854])
        for sol in (
            self.ets.ik_LM(self.Tep, q0=q),
            self.ets.ikine_LM(self.Tep, q0=q),
            self.ets.ikine_LM(self.Tep, q0=np.array([q, q])),
            self.ets.ikine_LM(self.Tep),
            self.ets.ik_LM(self.Tep),
        ):
            self.assertTrue(sol.success)


class TestIKVerbose(unittest.TestCase):
    """verbose=True prints the progress of each search, from the Python solvers."""

    def setUp(self):
        self.puma = rtb.models.DH.Puma560()
        self.panda = rtb.models.Panda()

    def test_silent_by_default(self):
        Tep = self.panda.fkine(self.panda.qr)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.panda.ikine_LM(Tep, q0=self.panda.qr)
        self.assertEqual(out.getvalue(), "")

    def test_each_solver(self):
        # an exact q0 converges at iteration 0, whichever solver
        Tep = self.panda.fkine(self.panda.qr)
        for method in ("ikine_LM", "ikine_NR", "ikine_GN"):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                sol = getattr(self.panda, method)(Tep, q0=self.panda.qr, verbose=True)
            self.assertTrue(sol.success, msg=method)
            text = out.getvalue()
            self.assertIn("Search 1", text, msg=method)
            self.assertRegex(text, r"Iteration +0 ", msg=method)
            self.assertIn("Converged", text, msg=method)
            self.assertIn("accepted", text, msg=method)

    @skip_no_qp
    def test_qp(self):
        Tep = self.panda.fkine(self.panda.qr)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.panda.ikine_QP(Tep, q0=self.panda.qr, verbose=True)
        self.assertIn("Search 1", out.getvalue())

    def test_rejected_by_joint_limits(self):
        # this solution of the Puma560 has q[4] outside its limits for every
        # equivalent angle (q[2] = 3.106 is -3.177, inside [-225, 45] degrees), so
        # the search is rejected and another is made.  None of the 22 distinct
        # solutions of this pose is inside the limits of the Puma560 (each is
        # outside at least one), so the solver fails after its searches
        q = np.array([0.2528, 0.7041, 3.106, 0.0, 2.473, -0.2528])
        Tep = self.puma.fkine(q)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            sol = self.puma.ikine_LM(
                Tep, q0=self.puma.qn, verbose=True, seed=0, slimit=3
            )
        text = out.getvalue()
        self.assertFalse(sol.success)
        self.assertIn("rejected", text)
        self.assertIn("q[4]=-3.81 outside [-1.745, 1.745]", text)
        self.assertNotIn("q[2]=", text)
        self.assertIn("Search 2", text)

    def test_trajectory(self):
        Ts = self.panda.fkine(np.array([self.panda.qr, self.panda.qr]))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.panda.ikine_LM(Ts, q0=self.panda.qr, verbose=True)
        self.assertIn("Pose 1 of 2", out.getvalue())
        self.assertIn("Pose 2 of 2", out.getvalue())

    def test_not_for_the_cpp_solvers(self):
        Tep = self.panda.fkine(self.panda.qr)
        with self.assertRaises(TypeError):
            self.panda.ik_LM(Tep, verbose=True)


def _solver_classes():
    """The four Python solvers, the QP solver only if qpsolvers is installed"""
    classes = [rtb.IK_LM, rtb.IK_NR, rtb.IK_GN]
    if importlib.util.find_spec("qpsolvers") is not None:
        classes.append(rtb.IK_QP)
    return classes


class TestIKVerboseAllSolvers(unittest.TestCase):
    """
    verbose=True, for each of the solver classes, and the methods that wrap them

    The cases are deterministic: q0 is either an exact solution, which converges
    at iteration 0, or a solution that violates the joint limits, or the number
    of iterations allowed is zero.
    """

    # a solution of the Puma560 with q[4] outside its limits, for every equivalent
    # angle
    Q_BAD = np.array([0.2528, 0.7041, 3.106, 0.0, 2.473, -0.2528])

    # a configuration inside the limits
    Q_OK = np.array([0.2, 0.9, 0.1, 0.3, -0.5, 0.4])

    def setUp(self):
        self.puma = rtb.models.DH.Puma560()
        self.ets = self.puma.ets()
        self.T_good = self.puma.fkine(self.Q_OK)
        self.T_bad = self.puma.fkine(self.Q_BAD)

    @staticmethod
    def capture(call):
        """Call ``call()``, return what it printed and its result"""
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = call()
        return out.getvalue(), result

    def test_attribute(self):
        for cls in _solver_classes():
            self.assertFalse(cls().verbose, msg=cls.__name__)
            self.assertTrue(cls(verbose=True).verbose, msg=cls.__name__)

    def test_silent_by_default(self):
        for cls in _solver_classes():
            text, sol = self.capture(
                lambda: cls().solve(self.ets, self.T_good, self.Q_OK)
            )
            self.assertEqual(text, "", msg=cls.__name__)
            self.assertTrue(sol.success, msg=cls.__name__)

    def test_accepted(self):
        for cls in _solver_classes():
            text, sol = self.capture(
                lambda: cls(verbose=True).solve(self.ets, self.T_good, self.Q_OK)
            )
            self.assertTrue(sol.success, msg=cls.__name__)
            lines = text.splitlines()
            self.assertEqual(lines[0], "Search 1", msg=cls.__name__)
            self.assertRegex(lines[1], r"^  Iteration +0  E=", msg=cls.__name__)
            self.assertRegex(lines[2], r"^  Converged +E=.*accepted$", msg=cls.__name__)
            self.assertEqual(len(lines), 3, msg=cls.__name__)

    def test_rejected_names_the_joints(self):
        for cls in _solver_classes():
            text, sol = self.capture(
                lambda: cls(verbose=True, slimit=1).solve(
                    self.ets, self.T_bad, self.Q_BAD
                )
            )
            self.assertFalse(sol.success, msg=cls.__name__)
            self.assertIn("rejected", text, msg=cls.__name__)
            # only q[4] is outside its limits: q[2] = 3.106 is -3.177 when wrapped,
            # which is inside [-225, 45] degrees
            self.assertIn("q[4]=-3.81 outside [-1.745, 1.745]", text, msg=cls.__name__)
            self.assertNotIn("q[2]", text, msg=cls.__name__)
            self.assertNotIn("q[0]", text, msg=cls.__name__)

    def test_joint_limits_off_accepts(self):
        for cls in _solver_classes():
            text, sol = self.capture(
                lambda: cls(verbose=True, slimit=1, joint_limits=False).solve(
                    self.ets, self.T_bad, self.Q_BAD
                )
            )
            self.assertTrue(sol.success, msg=cls.__name__)
            self.assertIn("accepted", text, msg=cls.__name__)
            self.assertNotIn("rejected", text, msg=cls.__name__)

    def test_iteration_limit(self):
        for cls in _solver_classes():
            text, sol = self.capture(
                lambda: cls(verbose=True, ilimit=0, slimit=2).solve(
                    self.ets, self.T_good, self.Q_BAD
                )
            )
            self.assertFalse(sol.success, msg=cls.__name__)
            self.assertEqual(text.count("Abandoned"), 2, msg=cls.__name__)
            self.assertIn("iteration limit reached", text, msg=cls.__name__)
            self.assertIn("Search 2", text, msg=cls.__name__)

    def test_columns_line_up(self):
        # the error starts in the same column on every line that has one
        for cls in _solver_classes():
            text, _ = self.capture(
                lambda: cls(verbose=True, slimit=1).solve(
                    self.ets, self.T_bad, self.Q_BAD
                )
            )
            columns = {line.index("E=") for line in text.splitlines() if "E=" in line}
            self.assertEqual(len(columns), 1, msg=f"{cls.__name__}: {columns}")

    def test_trajectory(self):
        Ts = self.puma.fkine(np.array([self.Q_OK, self.Q_OK, self.Q_OK]))
        for cls in _solver_classes():
            text, sol = self.capture(
                lambda: cls(verbose=True).solve(self.ets, Ts, self.Q_OK)
            )
            self.assertTrue(sol.success, msg=cls.__name__)
            for k in (1, 2, 3):
                self.assertIn(f"Pose {k} of 3", text, msg=cls.__name__)
            self.assertEqual(text.count("Search 1"), 3, msg=cls.__name__)

    def test_methods_pass_verbose_through(self):
        # ETS, DHRobot and URDF robot methods give the keyword to the solver
        panda = rtb.models.Panda()
        cases = [
            ("ETS", self.ets, self.T_good, self.Q_OK),
            ("DHRobot", self.puma, self.T_good, self.Q_OK),
            ("Robot", panda, panda.fkine(panda.qr), panda.qr),
        ]
        names = ["ikine_LM", "ikine_NR", "ikine_GN"]
        if importlib.util.find_spec("qpsolvers") is not None:
            names.append("ikine_QP")
        for label, obj, T, q0 in cases:
            for name in names:
                if name == "ikine_NR" and label == "DHRobot":
                    continue  # DHRobot has only ikine_LM of its own
                text, sol = self.capture(
                    lambda: getattr(obj, name)(T, q0=q0, verbose=True)
                )
                self.assertIn("Search 1", text, msg=f"{label}.{name}")
                self.assertIn("accepted", text, msg=f"{label}.{name}")
                text, _ = self.capture(lambda: getattr(obj, name)(T, q0=q0))
                self.assertEqual(text, "", msg=f"{label}.{name}")

    def test_cpp_solvers_do_not_take_it(self):
        panda = rtb.models.Panda()
        Tep = panda.fkine(panda.qr)
        for name in ("ik_LM", "ik_NR", "ik_GN"):
            for obj in (panda, panda.ets()):
                with self.assertRaises(TypeError, msg=name):
                    getattr(obj, name)(Tep, verbose=True)


if __name__ == "__main__":
    unittest.main()
