"""
@author: Jesse Haviland
"""

# import numpy.testing as nt
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


if __name__ == "__main__":
    unittest.main()
