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

        self.assertEqual(sol[1], True)
        self.assertEqual(sol2[1], True)

        Tq = r.eval(sol[0])
        Tq2 = r.eval(sol2[0])

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

        self.assertEqual(sol[1], True)
        self.assertEqual(sol2[1], True)

        Tq = r.eval(sol[0])
        Tq2 = r.eval(sol2[0])

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

        self.assertEqual(sol[1], True)
        self.assertEqual(sol2[1], True)

        Tq = r.eval(sol[0])
        Tq2 = r.eval(sol2[0])

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

        self.assertEqual(sol[1], True)
        self.assertEqual(sol2[1], True)

        Tq = r.eval(sol[0])
        Tq2 = r.eval(sol2[0])

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

        self.assertEqual(sol[1], True)
        self.assertEqual(sol2[1], True)

        Tq = r.eval(sol[0])
        Tq2 = r.eval(sol2[0])

        print(sol[4])
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

        ans = "IKSolution: q=[0, 0, 0], success=False, reason=no"

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

        ans = "IKSolution: q=[0, 0, 0], success=True"

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

    def test_iter_iksol(self):
        sol = rtb.IKSolution(
            np.array([1.0, 2.0, 3.0]),
            success=True,
            iterations=10,
            searches=100,
            residual=0.1,
        )

        a, b, c, d, e, f = sol

        nt.assert_almost_equal(a, np.array([1.0, 2.0, 3.0]))  # type: ignore
        self.assertEqual(b, True)
        self.assertEqual(c, 10)
        self.assertEqual(d, 100)
        self.assertEqual(e, 0.1)
        self.assertEqual(f, "")


if __name__ == "__main__":

    unittest.main()
